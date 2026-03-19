from isaacgym.torch_utils import *

import torch

from humanoidverse.envs.decoupled_locomotion.decoupled_locomotion_stand_height_waist_wbc_ma_diff_force import (
    LeggedRobotDecoupledLocomotionStanceHeightWBCForce,
)
from humanoidverse.utils.torch_utils import quat_conjugate, quat_mul, quat_unit

from isaac_utils.rotations import my_quat_rotate, quaternion_to_matrix


class LeggedRobotDecoupledLocomotionStanceHeightWBCForceHandPose(
    LeggedRobotDecoupledLocomotionStanceHeightWBCForce
):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)

        self.left_palm_link = "left_palm_link"
        self.right_palm_link = "right_palm_link"
        self.left_palm_link_index = self.simulator._body_list.index(self.left_palm_link)
        self.right_palm_link_index = self.simulator._body_list.index(self.right_palm_link)
        self._init_palm_link_settings()
        self._init_hand_pose_buffers()

    def _init_palm_link_settings(self):
        palm_cfg = {}
        for extend_cfg in self.config.robot.motion.extend_config:
            palm_cfg[extend_cfg["joint_name"]] = extend_cfg

        left_palm_cfg = palm_cfg[self.left_palm_link]
        right_palm_cfg = palm_cfg[self.right_palm_link]

        self.left_palm_parent_index = self.simulator._body_list.index(left_palm_cfg["parent_name"])
        self.right_palm_parent_index = self.simulator._body_list.index(right_palm_cfg["parent_name"])

        self.left_palm_pos_in_parent = torch.tensor(
            left_palm_cfg["pos"], dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self.right_palm_pos_in_parent = torch.tensor(
            right_palm_cfg["pos"], dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

        self.left_palm_rot_in_parent_xyzw = torch.tensor(
            left_palm_cfg["rot"], dtype=torch.float32, device=self.device
        )[[1, 2, 3, 0]].unsqueeze(0).repeat(self.num_envs, 1)
        self.right_palm_rot_in_parent_xyzw = torch.tensor(
            right_palm_cfg["rot"], dtype=torch.float32, device=self.device
        )[[1, 2, 3, 0]].unsqueeze(0).repeat(self.num_envs, 1)

    def _init_hand_pose_buffers(self):
        self.ref_left_palm_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.ref_right_palm_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.curr_left_palm_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.curr_right_palm_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)

        self.ref_left_palm_rot_6d = torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device)
        self.ref_right_palm_rot_6d = torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device)
        self.curr_left_palm_rot_6d = torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device)
        self.curr_right_palm_rot_6d = torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device)

    def _safe_normalize_quat_xyzw(self, quat_xyzw):
        quat_xyzw = quat_xyzw.clone()
        quat_norm = torch.linalg.norm(quat_xyzw, dim=-1, keepdim=True)
        identity = torch.zeros_like(quat_xyzw)
        identity[..., 3] = 1.0
        valid_mask = torch.isfinite(quat_xyzw).all(dim=-1, keepdim=True) & (quat_norm > 1e-8)
        normalized_quat = quat_xyzw / quat_norm.clamp_min(1e-8)
        return torch.where(valid_mask, normalized_quat, identity)

    def _quat_xyzw_to_rot6d(self, quat_xyzw):
        quat_xyzw = self._safe_normalize_quat_xyzw(quat_xyzw)
        quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]]
        rot_mat = quaternion_to_matrix(quat_wxyz)
        # Use the first two columns of the rotation matrix, flattened column-wise.
        first_two_columns = rot_mat[..., :, :2]
        return first_two_columns.transpose(-1, -2).reshape(rot_mat.shape[:-2] + (6,))

    def _world_pose_to_base_frame(self, world_pos, world_rot_xyzw):
        base_pos = self.simulator.robot_root_states[:, 0:3]
        base_rot_inv = quat_conjugate(self._safe_normalize_quat_xyzw(self.base_quat))
        world_rot_xyzw = self._safe_normalize_quat_xyzw(world_rot_xyzw)

        pos_base = quat_rotate_inverse(self.base_quat, world_pos - base_pos)
        rot_base = self._safe_normalize_quat_xyzw(quat_mul(base_rot_inv, world_rot_xyzw))
        return pos_base, rot_base

    def _compose_palm_world_pose(self, parent_pos, parent_rot_xyzw, palm_pos_in_parent, palm_rot_in_parent_xyzw):
        parent_rot_xyzw = self._safe_normalize_quat_xyzw(parent_rot_xyzw)
        palm_rot_in_parent_xyzw = self._safe_normalize_quat_xyzw(palm_rot_in_parent_xyzw)

        world_pos = my_quat_rotate(parent_rot_xyzw, palm_pos_in_parent) + parent_pos
        world_rot = self._safe_normalize_quat_xyzw(quat_mul(parent_rot_xyzw, palm_rot_in_parent_xyzw))
        return world_pos, world_rot

    def _update_palm_pose_buffers(self):
        curr_left_parent_pos = self.simulator._rigid_body_pos[:, self.left_palm_parent_index, :]
        curr_right_parent_pos = self.simulator._rigid_body_pos[:, self.right_palm_parent_index, :]
        curr_left_parent_rot = self.simulator._rigid_body_rot[:, self.left_palm_parent_index, :]
        curr_right_parent_rot = self.simulator._rigid_body_rot[:, self.right_palm_parent_index, :]

        if self.config.rewards.fix_upper_body:
            self.ref_left_palm_pos.zero_()
            self.ref_right_palm_pos.zero_()
            self.ref_left_palm_rot_6d.zero_()
            self.ref_right_palm_rot_6d.zero_()
        else:
            motion_res = self._latest_motion_res
            if motion_res is None:
                motion_res = self._motion_lib.get_motion_state(self.motion_ids, self.motion_times, offset=self.env_origins)
            ref_left_parent_pos = motion_res["rg_pos_t"][:, self.left_palm_parent_index, :]
            ref_right_parent_pos = motion_res["rg_pos_t"][:, self.right_palm_parent_index, :]
            ref_left_parent_rot = motion_res["rg_rot_t"][:, self.left_palm_parent_index, :]
            ref_right_parent_rot = motion_res["rg_rot_t"][:, self.right_palm_parent_index, :]

            ref_left_world_pos, ref_left_world_rot = self._compose_palm_world_pose(
                ref_left_parent_pos,
                ref_left_parent_rot,
                self.left_palm_pos_in_parent,
                self.left_palm_rot_in_parent_xyzw,
            )
            ref_right_world_pos, ref_right_world_rot = self._compose_palm_world_pose(
                ref_right_parent_pos,
                ref_right_parent_rot,
                self.right_palm_pos_in_parent,
                self.right_palm_rot_in_parent_xyzw,
            )

            self.ref_left_palm_pos[:], ref_left_rot_base = self._world_pose_to_base_frame(
                ref_left_world_pos, ref_left_world_rot
            )
            self.ref_right_palm_pos[:], ref_right_rot_base = self._world_pose_to_base_frame(
                ref_right_world_pos, ref_right_world_rot
            )
            self.ref_left_palm_rot_6d[:] = self._quat_xyzw_to_rot6d(ref_left_rot_base)
            self.ref_right_palm_rot_6d[:] = self._quat_xyzw_to_rot6d(ref_right_rot_base)

        curr_left_world_pos, curr_left_world_rot = self._compose_palm_world_pose(
            curr_left_parent_pos,
            curr_left_parent_rot,
            self.left_palm_pos_in_parent,
            self.left_palm_rot_in_parent_xyzw,
        )
        curr_right_world_pos, curr_right_world_rot = self._compose_palm_world_pose(
            curr_right_parent_pos,
            curr_right_parent_rot,
            self.right_palm_pos_in_parent,
            self.right_palm_rot_in_parent_xyzw,
        )

        self.curr_left_palm_pos[:], curr_left_rot_base = self._world_pose_to_base_frame(
            curr_left_world_pos, curr_left_world_rot
        )
        self.curr_right_palm_pos[:], curr_right_rot_base = self._world_pose_to_base_frame(
            curr_right_world_pos, curr_right_world_rot
        )
        self.curr_left_palm_rot_6d[:] = self._quat_xyzw_to_rot6d(curr_left_rot_base)
        self.curr_right_palm_rot_6d[:] = self._quat_xyzw_to_rot6d(curr_right_rot_base)

    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()
        self._update_palm_pose_buffers()

    def _reward_tracking_palm_pos(self):
        if self.config.rewards.fix_upper_body:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        palm_pos_error = torch.sum(torch.square(self.curr_left_palm_pos - self.ref_left_palm_pos), dim=1)
        palm_pos_error += torch.sum(torch.square(self.curr_right_palm_pos - self.ref_right_palm_pos), dim=1)
        return torch.exp(-palm_pos_error / self.config.rewards.reward_tracking_sigma.palm_pos)

    def _reward_tracking_palm_rot(self):
        if self.config.rewards.fix_upper_body:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        palm_rot_error = torch.sum(torch.square(self.curr_left_palm_rot_6d - self.ref_left_palm_rot_6d), dim=1)
        palm_rot_error += torch.sum(torch.square(self.curr_right_palm_rot_6d - self.ref_right_palm_rot_6d), dim=1)
        return torch.exp(-palm_rot_error / self.config.rewards.reward_tracking_sigma.palm_rot)

    def _get_obs_ref_left_palm_pos(self):
        return self.ref_left_palm_pos

    def _get_obs_ref_right_palm_pos(self):
        return self.ref_right_palm_pos

    def _get_obs_ref_left_palm_rot_6d(self):
        return self.ref_left_palm_rot_6d

    def _get_obs_ref_right_palm_rot_6d(self):
        return self.ref_right_palm_rot_6d
