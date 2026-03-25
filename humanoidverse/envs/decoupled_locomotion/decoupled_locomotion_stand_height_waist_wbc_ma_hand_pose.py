from isaacgym.torch_utils import *

import torch

from humanoidverse.envs.decoupled_locomotion.decoupled_locomotion_stand_height_waist_wbc_ma_diff_force import (
    LeggedRobotDecoupledLocomotionStanceHeightWBCForce,
)
from humanoidverse.envs.env_utils.visualization import Point
from humanoidverse.utils.torch_utils import quat_conjugate, quat_mul

from isaac_utils.rotations import get_euler_xyz_in_tensor, my_quat_rotate, quaternion_to_matrix


class LeggedRobotDecoupledLocomotionStanceHeightWBCForceHandPose(
    LeggedRobotDecoupledLocomotionStanceHeightWBCForce
):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)

        # Use synthetic palm links as the canonical hand-pose frame for this branch.
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

        motion_body_list = None
        if self._motion_lib is not None:
            motion_body_list = list(self._motion_lib.mesh_parsers.body_names_augment)

        self.left_palm_parent_index = self.simulator._body_list.index(left_palm_cfg["parent_name"])
        self.right_palm_parent_index = self.simulator._body_list.index(right_palm_cfg["parent_name"])
        self.motion_torso_index = self.torso_index
        self.motion_left_palm_parent_index = self.left_palm_parent_index
        self.motion_right_palm_parent_index = self.right_palm_parent_index
        if motion_body_list is not None:
            self.motion_torso_index = motion_body_list.index(self.torso_name)
            self.motion_left_palm_parent_index = motion_body_list.index(left_palm_cfg["parent_name"])
            self.motion_right_palm_parent_index = motion_body_list.index(right_palm_cfg["parent_name"])

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
        self.ref_left_palm_pos_world = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.ref_right_palm_pos_world = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.curr_left_palm_pos_world = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.curr_right_palm_pos_world = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)

        self.ref_left_palm_rot_xyzw = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self.ref_right_palm_rot_xyzw = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self.curr_left_palm_rot_xyzw = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self.curr_right_palm_rot_xyzw = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self.ref_left_palm_rot_world_xyzw = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.device
        )
        self.ref_right_palm_rot_world_xyzw = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.device
        )
        self.curr_left_palm_rot_world_xyzw = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.device
        )
        self.curr_right_palm_rot_world_xyzw = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.device
        )

        self.ref_left_palm_rot_6d = torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device)
        self.ref_right_palm_rot_6d = torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device)
        self.curr_left_palm_rot_6d = torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device)
        self.curr_right_palm_rot_6d = torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device)

        self.palm_pos_sq_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.palm_rot_sq_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.palm_pos_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.palm_rot_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.upper_body_dofs_error = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.far_hand_pose_tracking_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

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

    def _quat_xyzw_to_rotmat(self, quat_xyzw):
        quat_xyzw = self._safe_normalize_quat_xyzw(quat_xyzw)
        quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]]
        return quaternion_to_matrix(quat_wxyz)

    def _world_pose_to_local_frame(self, frame_pos, frame_rot_xyzw, world_pos, world_rot_xyzw):
        frame_rot_xyzw = self._safe_normalize_quat_xyzw(frame_rot_xyzw)
        frame_rot_inv = quat_conjugate(frame_rot_xyzw)
        world_rot_xyzw = self._safe_normalize_quat_xyzw(world_rot_xyzw)

        pos_local = quat_rotate_inverse(frame_rot_xyzw, world_pos - frame_pos)
        rot_local = self._safe_normalize_quat_xyzw(quat_mul(frame_rot_inv, world_rot_xyzw))
        return pos_local, rot_local

    def _local_pose_to_world_frame(self, frame_pos, frame_rot_xyzw, local_pos, local_rot_xyzw):
        frame_rot_xyzw = self._safe_normalize_quat_xyzw(frame_rot_xyzw)
        local_rot_xyzw = self._safe_normalize_quat_xyzw(local_rot_xyzw)

        world_pos = my_quat_rotate(frame_rot_xyzw, local_pos) + frame_pos
        world_rot = self._safe_normalize_quat_xyzw(quat_mul(frame_rot_xyzw, local_rot_xyzw))
        return world_pos, world_rot

    def _compose_palm_world_pose(self, parent_pos, parent_rot_xyzw, palm_pos_in_parent, palm_rot_in_parent_xyzw):
        parent_rot_xyzw = self._safe_normalize_quat_xyzw(parent_rot_xyzw)
        palm_rot_in_parent_xyzw = self._safe_normalize_quat_xyzw(palm_rot_in_parent_xyzw)

        world_pos = my_quat_rotate(parent_rot_xyzw, palm_pos_in_parent) + parent_pos
        world_rot = self._safe_normalize_quat_xyzw(quat_mul(parent_rot_xyzw, palm_rot_in_parent_xyzw))
        return world_pos, world_rot

    def _quat_angle_error(self, quat_a_xyzw, quat_b_xyzw):
        quat_a_xyzw = self._safe_normalize_quat_xyzw(quat_a_xyzw)
        quat_b_xyzw = self._safe_normalize_quat_xyzw(quat_b_xyzw)
        quat_rel = self._safe_normalize_quat_xyzw(quat_mul(quat_conjugate(quat_a_xyzw), quat_b_xyzw))
        quat_rel_w = quat_rel[..., 3].abs().clamp(max=1.0)
        return 2.0 * torch.acos(quat_rel_w)

    def _update_hand_pose_tracking_metrics(self):
        if self.config.rewards.fix_upper_body:
            self.palm_pos_sq_error.zero_()
            self.palm_rot_sq_error.zero_()
            self.palm_pos_error.zero_()
            self.palm_rot_error.zero_()
            self.log_dict["palm_pos_error"] = torch.tensor(0.0, device=self.device)
            self.log_dict["palm_rot_error"] = torch.tensor(0.0, device=self.device)
            return

        left_pos_delta = self.curr_left_palm_pos - self.ref_left_palm_pos
        right_pos_delta = self.curr_right_palm_pos - self.ref_right_palm_pos
        left_rot_error = self._quat_angle_error(self.curr_left_palm_rot_xyzw, self.ref_left_palm_rot_xyzw)
        right_rot_error = self._quat_angle_error(self.curr_right_palm_rot_xyzw, self.ref_right_palm_rot_xyzw)

        self.palm_pos_sq_error[:] = torch.sum(torch.square(left_pos_delta), dim=1)
        self.palm_pos_sq_error += torch.sum(torch.square(right_pos_delta), dim=1)
        self.palm_rot_sq_error[:] = torch.square(left_rot_error) + torch.square(right_rot_error)
        self.palm_pos_error[:] = 0.5 * (
            torch.linalg.norm(left_pos_delta, dim=1) + torch.linalg.norm(right_pos_delta, dim=1)
        )
        self.palm_rot_error[:] = 0.5 * (left_rot_error + right_rot_error)

        self.log_dict["palm_pos_error"] = self.palm_pos_error.mean()
        self.log_dict["palm_rot_error"] = self.palm_rot_error.mean()

    def _update_palm_pose_buffers(self):
        curr_torso_pos = self.simulator._rigid_body_pos[:, self.torso_index, :]
        curr_torso_rot = self.simulator._rigid_body_rot[:, self.torso_index, :]
        curr_left_parent_pos = self.simulator._rigid_body_pos[:, self.left_palm_parent_index, :]
        curr_right_parent_pos = self.simulator._rigid_body_pos[:, self.right_palm_parent_index, :]
        curr_left_parent_rot = self.simulator._rigid_body_rot[:, self.left_palm_parent_index, :]
        curr_right_parent_rot = self.simulator._rigid_body_rot[:, self.right_palm_parent_index, :]

        if self.config.rewards.fix_upper_body:
            self.ref_left_palm_pos.zero_()
            self.ref_right_palm_pos.zero_()
            self.ref_left_palm_pos_world.zero_()
            self.ref_right_palm_pos_world.zero_()
            self.ref_left_palm_rot_xyzw.zero_()
            self.ref_right_palm_rot_xyzw.zero_()
            self.ref_left_palm_rot_world_xyzw.zero_()
            self.ref_right_palm_rot_world_xyzw.zero_()
            self.ref_left_palm_rot_6d.zero_()
            self.ref_right_palm_rot_6d.zero_()
        else:
            motion_res = self._latest_motion_res
            if motion_res is None:
                motion_res = self._motion_lib.get_motion_state(self.motion_ids, self.motion_times, offset=self.env_origins)
            ref_torso_pos = motion_res["rg_pos_t"][:, self.motion_torso_index, :]
            ref_torso_rot = motion_res["rg_rot_t"][:, self.motion_torso_index, :]
            ref_left_parent_pos = motion_res["rg_pos_t"][:, self.motion_left_palm_parent_index, :]
            ref_right_parent_pos = motion_res["rg_pos_t"][:, self.motion_right_palm_parent_index, :]
            ref_left_parent_rot = motion_res["rg_rot_t"][:, self.motion_left_palm_parent_index, :]
            ref_right_parent_rot = motion_res["rg_rot_t"][:, self.motion_right_palm_parent_index, :]

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
            self.ref_left_palm_pos[:], ref_left_rot_torso = self._world_pose_to_local_frame(
                ref_torso_pos,
                ref_torso_rot,
                ref_left_world_pos,
                ref_left_world_rot,
            )
            self.ref_right_palm_pos[:], ref_right_rot_torso = self._world_pose_to_local_frame(
                ref_torso_pos,
                ref_torso_rot,
                ref_right_world_pos,
                ref_right_world_rot,
            )
            self.ref_left_palm_rot_xyzw[:] = ref_left_rot_torso
            self.ref_right_palm_rot_xyzw[:] = ref_right_rot_torso
            self.ref_left_palm_rot_6d[:] = self._quat_xyzw_to_rot6d(ref_left_rot_torso)
            self.ref_right_palm_rot_6d[:] = self._quat_xyzw_to_rot6d(ref_right_rot_torso)

            # Visualize the desired palm pose in the current robot torso frame, so
            # the world-space debug view directly shows the torso-relative mismatch.
            self.ref_left_palm_pos_world[:], self.ref_left_palm_rot_world_xyzw[:] = self._local_pose_to_world_frame(
                curr_torso_pos,
                curr_torso_rot,
                self.ref_left_palm_pos,
                self.ref_left_palm_rot_xyzw,
            )
            self.ref_right_palm_pos_world[:], self.ref_right_palm_rot_world_xyzw[:] = self._local_pose_to_world_frame(
                curr_torso_pos,
                curr_torso_rot,
                self.ref_right_palm_pos,
                self.ref_right_palm_rot_xyzw,
            )

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
        self.curr_left_palm_pos_world[:] = curr_left_world_pos
        self.curr_right_palm_pos_world[:] = curr_right_world_pos
        self.curr_left_palm_rot_world_xyzw[:] = curr_left_world_rot
        self.curr_right_palm_rot_world_xyzw[:] = curr_right_world_rot

        self.curr_left_palm_pos[:], curr_left_rot_torso = self._world_pose_to_local_frame(
            curr_torso_pos,
            curr_torso_rot,
            curr_left_world_pos,
            curr_left_world_rot,
        )
        self.curr_right_palm_pos[:], curr_right_rot_torso = self._world_pose_to_local_frame(
            curr_torso_pos,
            curr_torso_rot,
            curr_right_world_pos,
            curr_right_world_rot,
        )
        self.curr_left_palm_rot_xyzw[:] = curr_left_rot_torso
        self.curr_right_palm_rot_xyzw[:] = curr_right_rot_torso
        self.curr_left_palm_rot_6d[:] = self._quat_xyzw_to_rot6d(curr_left_rot_torso)
        self.curr_right_palm_rot_6d[:] = self._quat_xyzw_to_rot6d(curr_right_rot_torso)
        self._update_hand_pose_tracking_metrics()

    def _pre_compute_observations_callback(self):
        # Keep the hand-pose branch temporally synchronous with the current
        # control step, while leaving the other WBC envs on the inherited
        # one-step lookahead reference.
        self.base_quat[:] = self.simulator.base_quat[:]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        if self.config.rewards.fix_upper_body:
            self._latest_motion_res = None
            self.ref_upper_dof_pos *= 0.0
        else:
            offset = self.env_origins
            self.motion_times = (self.episode_length_buf * self.dt + self.motion_start_times) * (
                1 - self.fix_upper_body
            ) + self.fix_upper_body_motion_times * self.fix_upper_body
            motion_res = self._motion_lib.get_motion_state(
                self.motion_ids, self.motion_times, offset=offset
            )
            self._latest_motion_res = motion_res

            ref_joint_pos = motion_res["dof_pos"]
            self.ref_body_pos_extend[:, : motion_res["rg_pos_t"].shape[1], :] = motion_res["rg_pos_t"]
            self.ref_upper_dof_pos = ref_joint_pos[:, self.upper_dof_indices]
            self.ref_upper_dof_pos *= self.action_scale_upper_body

        B = self.motion_ids.shape[0]
        rotated_pos_in_parent = my_quat_rotate(
            self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
            self.extend_body_pos_in_parent.reshape(-1, 3),
        )
        self.extend_curr_pos = my_quat_rotate(
            self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
            rotated_pos_in_parent,
        ).view(self.num_envs, -1, 3) + self.simulator._rigid_body_pos[:, self.extend_body_parent_ids]
        self._rigid_body_pos_extend = torch.cat([self.simulator._rigid_body_pos, self.extend_curr_pos], dim=1)
        self.marker_coords[:] = self._rigid_body_pos_extend.reshape(B, -1, 3)

        left_ee_apply_force_pos = (
            self.extend_curr_pos[:, 0, :] - self.simulator._rigid_body_pos[:, self.left_hand_link_index, :]
        ) * self.left_ee_apply_force_pos_ratio + self.simulator._rigid_body_pos[:, self.left_hand_link_index, :]
        right_ee_apply_force_pos = (
            self.extend_curr_pos[:, 1, :] - self.simulator._rigid_body_pos[:, self.right_hand_link_index, :]
        ) * self.right_ee_apply_force_pos_ratio + self.simulator._rigid_body_pos[:, self.right_hand_link_index, :]
        self.apply_force_pos_tensor[:, self.left_hand_link_index, :] = left_ee_apply_force_pos
        self.apply_force_pos_tensor[:, self.right_hand_link_index, :] = right_ee_apply_force_pos

        self._update_palm_pose_buffers()

    def _reward_tracking_palm_pos(self):
        if self.config.rewards.fix_upper_body:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        return torch.exp(-self.palm_pos_sq_error / self.config.rewards.reward_tracking_sigma.palm_pos)

    def _reward_tracking_palm_rot(self):
        if self.config.rewards.fix_upper_body:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        return torch.exp(-self.palm_rot_sq_error / self.config.rewards.reward_tracking_sigma.palm_rot)

    def _reward_tracking_upper_body_dofs(self):
        if self.config.rewards.fix_upper_body:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        upper_body_pos = self.simulator.dof_pos[:, self.upper_dof_indices]
        self.upper_body_dofs_error[:] = torch.sum(
            torch.square(upper_body_pos - self.ref_upper_dof_pos), dim=1
        )
        upper_body_dofs_reward = torch.exp(
            -self.upper_body_dofs_error
            / self.config.rewards.reward_tracking_sigma.upper_body_dofs
        )
        self.upper_body_dofs_tracking_reward += upper_body_dofs_reward
        self.log_dict["upper_body_dofs_error"] = self.upper_body_dofs_error.mean()
        return upper_body_dofs_reward

    def _update_far_upper_dof_pos_buf(self):
        if self.config.rewards.fix_upper_body:
            self.far_upper_dof_pos_buf[:] = False
            self.far_hand_pose_tracking_buf[:] = False
            return

        if self.config.termination.terminate_when_low_upper_dof_tracking:
            dof_dev = torch.exp(
                -0.5
                * torch.norm(
                    self.simulator.dof_pos[:, self.upper_dof_indices] - self.ref_upper_dof_pos,
                    dim=1,
                )
            )
            self.far_upper_dof_pos_buf[:] = (
                dof_dev
                < self.config.termination_scales.terminate_when_low_upper_dof_tracking_threshold
            )
            self.reset_buf |= self.far_upper_dof_pos_buf
        else:
            self.far_upper_dof_pos_buf[:] = False

        if self.config.termination.get("terminate_when_low_hand_pose_tracking", False):
            palm_tracking_score = 0.5 * (
                self._reward_tracking_palm_pos() + self._reward_tracking_palm_rot()
            )
            self.far_hand_pose_tracking_buf[:] = (
                palm_tracking_score
                < self.config.termination_scales.get(
                    "terminate_when_low_hand_pose_tracking_threshold",
                    self.config.termination_scales.terminate_when_low_upper_dof_tracking_threshold,
                )
            )
            self.reset_buf |= self.far_hand_pose_tracking_buf
        else:
            self.far_hand_pose_tracking_buf[:] = False

    def _get_obs_ref_left_palm_pos(self):
        return self.ref_left_palm_pos

    def _get_obs_ref_right_palm_pos(self):
        return self.ref_right_palm_pos

    def _get_obs_ref_left_palm_rot_6d(self):
        return self.ref_left_palm_rot_6d

    def _get_obs_ref_right_palm_rot_6d(self):
        return self.ref_right_palm_rot_6d

    def _hand_pose_debug_cfg_get(self, key, default):
        visualization_cfg = self.config.robot.motion.get("visualization", None)
        if visualization_cfg is None:
            return default
        hand_pose_debug_cfg = visualization_cfg.get("hand_pose_tracking", None)
        if hand_pose_debug_cfg is None:
            return default
        return hand_pose_debug_cfg.get(key, default)

    def _draw_thick_line(self, env_id, start_point, end_point, color, line_width):
        color_point = Point(torch.tensor(color, dtype=torch.float32, device=self.device))
        direction = end_point - start_point
        direction_norm = torch.linalg.norm(direction)
        if direction_norm < 1e-8 or line_width <= 0.0:
            self.simulator.draw_line(Point(start_point), Point(end_point), color_point, env_id)
            return

        direction = direction / direction_norm
        if torch.abs(direction[2]) < 0.9:
            reference_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)
        else:
            reference_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device)

        ortho_axis_a = torch.cross(direction, reference_axis, dim=0)
        ortho_axis_a_norm = torch.linalg.norm(ortho_axis_a)
        if ortho_axis_a_norm < 1e-8:
            reference_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
            ortho_axis_a = torch.cross(direction, reference_axis, dim=0)
            ortho_axis_a_norm = torch.linalg.norm(ortho_axis_a)
        ortho_axis_a = ortho_axis_a / ortho_axis_a_norm.clamp_min(1e-8)
        ortho_axis_b = torch.cross(direction, ortho_axis_a, dim=0)
        ortho_axis_b = ortho_axis_b / torch.linalg.norm(ortho_axis_b).clamp_min(1e-8)

        diagonal_scale = 0.7 * line_width
        offsets = (
            torch.zeros(3, dtype=torch.float32, device=self.device),
            ortho_axis_a * line_width,
            -ortho_axis_a * line_width,
            ortho_axis_b * line_width,
            -ortho_axis_b * line_width,
            (ortho_axis_a + ortho_axis_b) * diagonal_scale,
            (ortho_axis_a - ortho_axis_b) * diagonal_scale,
            (-ortho_axis_a + ortho_axis_b) * diagonal_scale,
            (-ortho_axis_a - ortho_axis_b) * diagonal_scale,
        )
        for offset in offsets:
            self.simulator.draw_line(
                Point(start_point + offset),
                Point(end_point + offset),
                color_point,
                env_id,
            )

    def _draw_pose_axes(
        self,
        env_id,
        origin,
        rot_xyzw,
        axis_length,
        sphere_radius,
        axis_line_width,
        origin_color,
        axis_colors,
    ):
        self.simulator.draw_sphere(origin, sphere_radius, origin_color, env_id)
        rot_mat = self._quat_xyzw_to_rotmat(rot_xyzw.unsqueeze(0))[0]
        for axis_id, axis_color in enumerate(axis_colors):
            end_point = origin + rot_mat[:, axis_id] * axis_length
            self._draw_thick_line(env_id, origin, end_point, axis_color, axis_line_width)

    def _draw_hand_pose_debug_vis(self):
        num_draw_envs = int(self._hand_pose_debug_cfg_get("num_draw_envs", 1))
        num_draw_envs = max(1, min(self.num_envs, num_draw_envs))
        desired_axis_length = float(self._hand_pose_debug_cfg_get("desired_axis_length", 0.14))
        current_axis_length = float(self._hand_pose_debug_cfg_get("current_axis_length", 0.10))
        desired_axis_line_width = float(self._hand_pose_debug_cfg_get("desired_axis_line_width", 0.008))
        current_axis_line_width = float(self._hand_pose_debug_cfg_get("current_axis_line_width", 0.006))
        desired_sphere_radius = float(self._hand_pose_debug_cfg_get("desired_sphere_radius", 0.02))
        current_sphere_radius = float(self._hand_pose_debug_cfg_get("current_sphere_radius", 0.015))
        draw_error_line = bool(self._hand_pose_debug_cfg_get("draw_error_line", False))

        desired_origin_color = (1.0, 0.82, 0.12)
        current_origin_color = (0.93, 0.93, 0.93)
        desired_axis_colors = (
            (0.95, 0.12, 0.12),
            (0.12, 0.9, 0.18),
            (0.15, 0.45, 0.98),
        )
        current_axis_colors = (
            (0.55, 0.30, 0.30),
            (0.28, 0.55, 0.30),
            (0.28, 0.42, 0.65),
        )
        error_line_colors = (
            (1.0, 0.65, 0.15),
            (0.2, 0.85, 1.0),
        )

        for env_id in range(num_draw_envs):
            self._draw_pose_axes(
                env_id,
                self.curr_left_palm_pos_world[env_id],
                self.curr_left_palm_rot_world_xyzw[env_id],
                current_axis_length,
                current_sphere_radius,
                current_axis_line_width,
                current_origin_color,
                current_axis_colors,
            )
            self._draw_pose_axes(
                env_id,
                self.curr_right_palm_pos_world[env_id],
                self.curr_right_palm_rot_world_xyzw[env_id],
                current_axis_length,
                current_sphere_radius,
                current_axis_line_width,
                current_origin_color,
                current_axis_colors,
            )

            if not self.config.rewards.fix_upper_body:
                self._draw_pose_axes(
                    env_id,
                    self.ref_left_palm_pos_world[env_id],
                    self.ref_left_palm_rot_world_xyzw[env_id],
                    desired_axis_length,
                    desired_sphere_radius,
                    desired_axis_line_width,
                    desired_origin_color,
                    desired_axis_colors,
                )
                self._draw_pose_axes(
                    env_id,
                    self.ref_right_palm_pos_world[env_id],
                    self.ref_right_palm_rot_world_xyzw[env_id],
                    desired_axis_length,
                    desired_sphere_radius,
                    desired_axis_line_width,
                    desired_origin_color,
                    desired_axis_colors,
                )

                if draw_error_line:
                    self.simulator.draw_line(
                        Point(self.ref_left_palm_pos_world[env_id]),
                        Point(self.curr_left_palm_pos_world[env_id]),
                        Point(torch.tensor(error_line_colors[0], dtype=torch.float32, device=self.device)),
                        env_id,
                    )
                    self.simulator.draw_line(
                        Point(self.ref_right_palm_pos_world[env_id]),
                        Point(self.curr_right_palm_pos_world[env_id]),
                        Point(torch.tensor(error_line_colors[1], dtype=torch.float32, device=self.device)),
                        env_id,
                    )

    def _draw_debug_vis(self):
        hand_pose_debug_enabled = bool(self._hand_pose_debug_cfg_get("enabled", False)) or bool(
            getattr(self.simulator, "vis_hand_pose_tracking", False)
        )
        force_debug_enabled = bool(getattr(self.simulator, "vis_force_range", False))

        if force_debug_enabled:
            super()._draw_debug_vis()
        else:
            self.simulator.clear_lines()

        if hand_pose_debug_enabled:
            self._draw_hand_pose_debug_vis()
