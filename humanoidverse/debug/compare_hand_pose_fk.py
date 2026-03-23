import json
import os
import sys
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from isaac_utils.rotations import my_quat_rotate, quat_to_angle_axis
from loguru import logger
from omegaconf import DictConfig

from humanoidverse.utils.logging import HydraLoggerBridge
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
from humanoidverse.utils.torch_utils import quat_conjugate, quat_mul, quat_rotate_inverse

import logging


def _safe_normalize_quat_xyzw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    quat_norm = torch.linalg.norm(quat_xyzw, dim=-1, keepdim=True)
    identity = torch.zeros_like(quat_xyzw)
    identity[..., 3] = 1.0
    valid_mask = torch.isfinite(quat_xyzw).all(dim=-1, keepdim=True) & (quat_norm > 1e-8)
    normalized_quat = quat_xyzw / quat_norm.clamp_min(1e-8)
    return torch.where(valid_mask, normalized_quat, identity)


def _compose_palm_world_pose(parent_pos, parent_rot_xyzw, palm_pos_in_parent, palm_rot_in_parent_xyzw):
    parent_rot_xyzw = _safe_normalize_quat_xyzw(parent_rot_xyzw)
    palm_rot_in_parent_xyzw = _safe_normalize_quat_xyzw(palm_rot_in_parent_xyzw)
    world_pos = my_quat_rotate(parent_rot_xyzw, palm_pos_in_parent) + parent_pos
    world_rot = _safe_normalize_quat_xyzw(quat_mul(parent_rot_xyzw, palm_rot_in_parent_xyzw))
    return world_pos, world_rot


def _world_pose_to_local_frame(frame_pos, frame_rot_xyzw, world_pos, world_rot_xyzw):
    frame_rot_xyzw = _safe_normalize_quat_xyzw(frame_rot_xyzw)
    frame_rot_inv = quat_conjugate(frame_rot_xyzw)
    world_rot_xyzw = _safe_normalize_quat_xyzw(world_rot_xyzw)
    pos_local = quat_rotate_inverse(frame_rot_xyzw, world_pos - frame_pos)
    rot_local = _safe_normalize_quat_xyzw(quat_mul(frame_rot_inv, world_rot_xyzw))
    return pos_local, rot_local


def _quat_angle_error(quat_a_xyzw, quat_b_xyzw):
    quat_a_xyzw = _safe_normalize_quat_xyzw(quat_a_xyzw)
    quat_b_xyzw = _safe_normalize_quat_xyzw(quat_b_xyzw)
    quat_rel = _safe_normalize_quat_xyzw(quat_mul(quat_conjugate(quat_a_xyzw), quat_b_xyzw))
    quat_rel_w = quat_rel[..., 3].abs().clamp(max=1.0)
    return 2.0 * torch.acos(quat_rel_w)


def _quat_xyzw_to_axis_angle(quat_xyzw):
    quat_xyzw = _safe_normalize_quat_xyzw(quat_xyzw)
    angle, axis = quat_to_angle_axis(quat_xyzw)
    return angle.unsqueeze(-1) * axis


def _scalar_stats(values: torch.Tensor):
    values_cpu = values.detach().cpu()
    return {
        "mean": float(values_cpu.mean().item()),
        "max": float(values_cpu.max().item()),
        "p95": float(torch.quantile(values_cpu, 0.95).item()),
    }


def _compare_pose_pair(pos_a, rot_a, pos_b, rot_b):
    pos_err = torch.linalg.norm(pos_a - pos_b, dim=-1)
    rot_err = _quat_angle_error(rot_a, rot_b)
    return {
        "pos": _scalar_stats(pos_err),
        "rot_rad": _scalar_stats(rot_err),
        "pos_err_tensor": pos_err,
        "rot_err_tensor": rot_err,
    }


def _extract_motion_dof_scalars(config: DictConfig, mesh_parser, dof_pos: torch.Tensor):
    motion_dof_names = list(config.robot.motion.dof_names)
    motion_body_list = list(mesh_parser.body_names_augment)
    expected_dofs = len(motion_dof_names)

    if dof_pos.shape[1] == expected_dofs:
        return dof_pos

    if dof_pos.shape[1] == mesh_parser.num_bodies - 1:
        body_slot_indices = [motion_body_list.index(name) - 1 for name in motion_dof_names]
        logger.warning(
            "dof_pos appears to be stored in body-order slots rather than compact actuated-dof order; "
            "gathering actuated entries by motion dof_names."
        )
        return dof_pos[:, body_slot_indices]

    raise ValueError(
        f"Unsupported dof_pos shape {tuple(dof_pos.shape)} for {expected_dofs} motion dofs and "
        f"{mesh_parser.num_bodies - 1} body-order slots."
    )


def _build_pose_from_dof_pos(config: DictConfig, mesh_parser, root_rot_xyzw, dof_pos):
    device = dof_pos.device
    dtype = dof_pos.dtype
    dof_pos = _extract_motion_dof_scalars(config, mesh_parser, dof_pos)
    B = dof_pos.shape[0]

    dof_axis = mesh_parser.dof_axis.to(device=device, dtype=dtype)
    if dof_pos.shape[1] < dof_axis.shape[0]:
        raise ValueError(
            f"dof_pos has shape {tuple(dof_pos.shape)}, but parser expects at least {dof_axis.shape[0]} dofs"
        )
    if dof_pos.shape[1] > dof_axis.shape[0]:
        logger.warning(
            f"dof_pos has {dof_pos.shape[1]} entries while parser has {dof_axis.shape[0]} actuated axes; "
            "truncating trailing non-actuated entries for FK reconstruction."
        )
        dof_pos = dof_pos[:, : dof_axis.shape[0]]

    if not torch.all(torch.sum(dof_axis.abs(), dim=-1) == 1):
        raise ValueError("This debug script assumes each joint axis has exactly one active component.")

    # The stored scalar dof_pos already carries the sign of the active axis component
    # because it was originally read back via pose.sum(dim=-1). Reconstruct the
    # axis-angle vector by placing the scalar on the active coordinate only.
    axis_component_selector = dof_axis.abs()

    pose_aa = torch.zeros(
        B,
        1,
        mesh_parser.num_bodies_augment,
        3,
        dtype=dtype,
        device=device,
    )
    pose_aa[:, 0, 0, :] = _quat_xyzw_to_axis_angle(root_rot_xyzw)

    motion_body_list = list(mesh_parser.body_names_augment)
    dof_body_indices = torch.as_tensor(
        [motion_body_list.index(name) for name in config.robot.motion.dof_names],
        dtype=torch.long,
        device=device,
    )
    pose_aa[:, 0, dof_body_indices, :] = dof_pos.unsqueeze(-1) * axis_component_selector.unsqueeze(0)
    return pose_aa


def _frame_aligned_times(motion_lib: MotionLibRobot, motion_ids: torch.Tensor):
    num_frames = motion_lib._motion_num_frames[motion_ids]
    frame_idx = torch.floor(torch.rand_like(num_frames.float()) * num_frames.float()).long()
    return frame_idx * motion_lib._motion_dt[motion_ids]


def _continuous_times(motion_lib: MotionLibRobot, motion_ids: torch.Tensor):
    return motion_lib.sample_time(motion_ids)


def _extract_extend_cfg(config: DictConfig):
    extend_cfg_by_name = {}
    for entry in config.robot.motion.extend_config:
        extend_cfg_by_name[entry.joint_name] = entry
    return extend_cfg_by_name


def _compute_current_logic_pose(motion_res, torso_index, parent_index, palm_pos_in_parent, palm_rot_in_parent_xyzw):
    world_pos, world_rot = _compose_palm_world_pose(
        motion_res["rg_pos_t"][:, parent_index, :],
        motion_res["rg_rot_t"][:, parent_index, :],
        palm_pos_in_parent,
        palm_rot_in_parent_xyzw,
    )
    local_pos, local_rot = _world_pose_to_local_frame(
        motion_res["rg_pos_t"][:, torso_index, :],
        motion_res["rg_rot_t"][:, torso_index, :],
        world_pos,
        world_rot,
    )
    return world_pos, world_rot, local_pos, local_rot


def _compute_fk_pose(config: DictConfig, mesh_parser, motion_res, torso_index, palm_index):
    pose_aa = _build_pose_from_dof_pos(config, mesh_parser, motion_res["root_rot"], motion_res["dof_pos"])
    fk_res = mesh_parser.fk_batch(
        pose_aa,
        motion_res["root_pos"].unsqueeze(1),
        convert_to_mat=True,
        return_full=False,
    )
    world_pos = fk_res.global_translation_extend[:, 0, palm_index, :]
    world_rot = fk_res.global_rotation_extend[:, 0, palm_index, :]
    torso_world_pos = fk_res.global_translation_extend[:, 0, torso_index, :]
    torso_world_rot = fk_res.global_rotation_extend[:, 0, torso_index, :]
    local_pos, local_rot = _world_pose_to_local_frame(
        torso_world_pos,
        torso_world_rot,
        world_pos,
        world_rot,
    )
    return world_pos, world_rot, local_pos, local_rot


def _run_comparison(config: DictConfig, motion_lib: MotionLibRobot, sample_mode: str, worst_k: int):
    mesh_parser = motion_lib.mesh_parsers
    motion_body_list = list(mesh_parser.body_names_augment)
    torso_name = config.robot.torso_name
    extend_cfg = _extract_extend_cfg(config)

    left_palm_cfg = extend_cfg["left_palm_link"]
    right_palm_cfg = extend_cfg["right_palm_link"]

    torso_index = motion_body_list.index(torso_name)
    left_parent_index = motion_body_list.index(left_palm_cfg.parent_name)
    right_parent_index = motion_body_list.index(right_palm_cfg.parent_name)
    left_palm_index = motion_body_list.index("left_palm_link")
    right_palm_index = motion_body_list.index("right_palm_link")

    motion_ids = torch.arange(motion_lib.num_envs, device=motion_lib._device)
    if sample_mode == "frame":
        motion_times = _frame_aligned_times(motion_lib, motion_ids)
    elif sample_mode == "continuous":
        motion_times = _continuous_times(motion_lib, motion_ids)
    else:
        raise ValueError(f"Unknown sample mode: {sample_mode}")

    motion_res = motion_lib.get_motion_state(motion_ids, motion_times, offset=None)

    left_palm_pos_in_parent = torch.tensor(
        left_palm_cfg.pos, dtype=torch.float32, device=motion_lib._device
    ).unsqueeze(0).repeat(motion_lib.num_envs, 1)
    right_palm_pos_in_parent = torch.tensor(
        right_palm_cfg.pos, dtype=torch.float32, device=motion_lib._device
    ).unsqueeze(0).repeat(motion_lib.num_envs, 1)
    left_palm_rot_in_parent_xyzw = torch.tensor(
        left_palm_cfg.rot, dtype=torch.float32, device=motion_lib._device
    )[[1, 2, 3, 0]].unsqueeze(0).repeat(motion_lib.num_envs, 1)
    right_palm_rot_in_parent_xyzw = torch.tensor(
        right_palm_cfg.rot, dtype=torch.float32, device=motion_lib._device
    )[[1, 2, 3, 0]].unsqueeze(0).repeat(motion_lib.num_envs, 1)

    left_current = _compute_current_logic_pose(
        motion_res, torso_index, left_parent_index, left_palm_pos_in_parent, left_palm_rot_in_parent_xyzw
    )
    right_current = _compute_current_logic_pose(
        motion_res, torso_index, right_parent_index, right_palm_pos_in_parent, right_palm_rot_in_parent_xyzw
    )

    left_fk = _compute_fk_pose(config, mesh_parser, motion_res, torso_index, left_palm_index)
    right_fk = _compute_fk_pose(config, mesh_parser, motion_res, torso_index, right_palm_index)

    comparisons = {
        "left_current_vs_fk_world": _compare_pose_pair(left_current[0], left_current[1], left_fk[0], left_fk[1]),
        "right_current_vs_fk_world": _compare_pose_pair(right_current[0], right_current[1], right_fk[0], right_fk[1]),
        "left_current_vs_fk_local": _compare_pose_pair(left_current[2], left_current[3], left_fk[2], left_fk[3]),
        "right_current_vs_fk_local": _compare_pose_pair(right_current[2], right_current[3], right_fk[2], right_fk[3]),
    }

    logger.info(f"Sample mode: {sample_mode}")
    for key, value in comparisons.items():
        logger.info(
            f"{key}: "
            f"pos(mean={value['pos']['mean']:.6e}, p95={value['pos']['p95']:.6e}, max={value['pos']['max']:.6e}) "
            f"rot(mean={value['rot_rad']['mean']:.6e}, p95={value['rot_rad']['p95']:.6e}, max={value['rot_rad']['max']:.6e})"
        )

    ranking_score = (
        comparisons["left_current_vs_fk_local"]["pos_err_tensor"]
        + comparisons["right_current_vs_fk_local"]["pos_err_tensor"]
        + comparisons["left_current_vs_fk_local"]["rot_err_tensor"]
        + comparisons["right_current_vs_fk_local"]["rot_err_tensor"]
    )
    top_k = min(worst_k, ranking_score.shape[0])
    top_indices = torch.topk(ranking_score, k=top_k).indices.detach().cpu().tolist()
    logger.info("Worst samples by combined current-vs-fk local mismatch:")
    for rank, sample_idx in enumerate(top_indices, start=1):
        logger.info(
            f"{rank}. motion_id={int(motion_ids[sample_idx].item())} "
            f"time={float(motion_times[sample_idx].item()):.6f} "
            f"left_pos_err={float(comparisons['left_current_vs_fk_local']['pos_err_tensor'][sample_idx].item()):.6e} "
            f"right_pos_err={float(comparisons['right_current_vs_fk_local']['pos_err_tensor'][sample_idx].item()):.6e} "
            f"left_rot_err={float(comparisons['left_current_vs_fk_local']['rot_err_tensor'][sample_idx].item()):.6e} "
            f"right_rot_err={float(comparisons['right_current_vs_fk_local']['rot_err_tensor'][sample_idx].item()):.6e}"
        )

    serializable = {
        "sample_mode": sample_mode,
        "comparisons": {
            key: {"pos": value["pos"], "rot_rad": value["rot_rad"]}
            for key, value in comparisons.items()
        },
        "worst_samples": [
            {
                "rank": rank,
                "motion_id": int(motion_ids[sample_idx].item()),
                "time": float(motion_times[sample_idx].item()),
                "left_current_vs_fk_local_pos": float(
                    comparisons["left_current_vs_fk_local"]["pos_err_tensor"][sample_idx].item()
                ),
                "right_current_vs_fk_local_pos": float(
                    comparisons["right_current_vs_fk_local"]["pos_err_tensor"][sample_idx].item()
                ),
                "left_current_vs_fk_local_rot": float(
                    comparisons["left_current_vs_fk_local"]["rot_err_tensor"][sample_idx].item()
                ),
                "right_current_vs_fk_local_rot": float(
                    comparisons["right_current_vs_fk_local"]["rot_err_tensor"][sample_idx].item()
                ),
            }
            for rank, sample_idx in enumerate(top_indices, start=1)
        ],
    }
    return serializable


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "compare_hand_pose_fk.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    logger.add(sys.stdout, level=os.environ.get("LOGURU_LEVEL", "INFO").upper(), colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())

    if "left_palm_link" not in {entry.joint_name for entry in config.robot.motion.extend_config}:
        raise ValueError(
            "This debug script expects the hand-pose robot config with left_palm_link/right_palm_link. "
            "Run it with the hand-pose experiment, for example: "
            "`python humanoidverse/debug/compare_hand_pose_fk.py +exp=decoupled_locomotion_stand_height_waist_wbc_hand_pose_ma_ppo_ma_env`"
        )

    device_str = config.get("debug_device", "cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    num_samples = int(config.get("debug_num_samples", 64))
    seed = int(config.get("debug_seed", 0))
    worst_k = int(config.get("debug_worst_k", 10))
    random_sample = bool(config.get("debug_random_sample_motions", False))
    compare_continuous = bool(config.get("debug_compare_continuous_times", True))
    output_json = config.get("debug_output_json", None)

    torch.manual_seed(seed)
    logger.info(f"Device: {device}")
    logger.info(f"Num samples: {num_samples}")
    logger.info(f"Seed: {seed}")

    motion_lib = MotionLibRobot(config.robot.motion, num_envs=num_samples, device=device)
    motion_lib.load_motions(random_sample=random_sample)

    all_results = {
        "frame_aligned": _run_comparison(config, motion_lib, sample_mode="frame", worst_k=worst_k)
    }
    if compare_continuous:
        all_results["continuous"] = _run_comparison(
            config, motion_lib, sample_mode="continuous", worst_k=worst_k
        )

    if output_json:
        output_path = Path(str(output_json))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_results, indent=2))
        logger.info(f"Saved comparison summary to {output_path}")


if __name__ == "__main__":
    main()
