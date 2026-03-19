from isaacgym.torch_utils import *

import torch

from humanoidverse.envs.decoupled_locomotion.decoupled_locomotion_stand_height_waist_wbc_ma_diff_force import (
    LeggedRobotDecoupledLocomotionStanceHeightWBCForce,
)
from humanoidverse.utils.motion_lib.motion_utils.rotation_conversions import (
    matrix_to_rotation_6d,
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
        self._init_hand_pose_buffers()

    def _init_hand_pose_buffers(self):
        total_bodies = self.num_bodies + self.num_extend_bodies
        self.extend_curr_rot = torch.zeros(
            self.num_envs, self.num_extend_bodies, 4, dtype=torch.float32, device=self.device
        )
        self._rigid_body_rot_extend = torch.zeros(
            self.num_envs, total_bodies, 4, dtype=torch.float32, device=self.device
        )
        self.ref_body_rot_extend = torch.zeros(
            self.num_envs, total_bodies, 4, dtype=torch.float32, device=self.device
        )

        self.ref_left_palm_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.ref_right_palm_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.curr_left_palm_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.curr_right_palm_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)

        self.ref_left_palm_rot_6d = torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device)
        self.ref_right_palm_rot_6d = torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device)
        self.curr_left_palm_rot_6d = torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device)
        self.curr_right_palm_rot_6d = torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device)

    def _quat_xyzw_to_rot6d(self, quat_xyzw):
        quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]]
        rot_mat = quaternion_to_matrix(quat_wxyz)
        return matrix_to_rotation_6d(rot_mat)

    def _world_pose_to_base_frame(self, world_pos, world_rot_xyzw):
        base_pos = self.simulator.robot_root_states[:, 0:3]
        base_rot_inv = quat_conjugate(self.base_quat)

        pos_base = quat_rotate_inverse(self.base_quat, world_pos - base_pos)
        rot_base = quat_unit(quat_mul(base_rot_inv, world_rot_xyzw))
        return pos_base, rot_base

    def _update_palm_pose_buffers(self):
        parent_rot = self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4)
        self.extend_curr_rot = quat_unit(
            quat_mul(parent_rot, self.extend_body_rot_in_parent_xyzw.reshape(-1, 4))
        ).view(self.num_envs, -1, 4)
        self._rigid_body_rot_extend = torch.cat([self.simulator._rigid_body_rot, self.extend_curr_rot], dim=1)

        if self.config.rewards.fix_upper_body:
            self.ref_body_rot_extend.zero_()
            self.ref_left_palm_pos.zero_()
            self.ref_right_palm_pos.zero_()
            self.ref_left_palm_rot_6d.zero_()
            self.ref_right_palm_rot_6d.zero_()
        else:
            motion_res = self._latest_motion_res
            if motion_res is None:
                motion_res = self._motion_lib.get_motion_state(self.motion_ids, self.motion_times, offset=self.env_origins)
            self.ref_body_rot_extend.zero_()
            self.ref_body_rot_extend[:, : motion_res["rg_rot_t"].shape[1], :] = motion_res["rg_rot_t"]

            ref_left_world_pos = self.ref_body_pos_extend[:, self.left_palm_link_index, :]
            ref_right_world_pos = self.ref_body_pos_extend[:, self.right_palm_link_index, :]
            ref_left_world_rot = self.ref_body_rot_extend[:, self.left_palm_link_index, :]
            ref_right_world_rot = self.ref_body_rot_extend[:, self.right_palm_link_index, :]

            self.ref_left_palm_pos[:], ref_left_rot_base = self._world_pose_to_base_frame(
                ref_left_world_pos, ref_left_world_rot
            )
            self.ref_right_palm_pos[:], ref_right_rot_base = self._world_pose_to_base_frame(
                ref_right_world_pos, ref_right_world_rot
            )
            self.ref_left_palm_rot_6d[:] = self._quat_xyzw_to_rot6d(ref_left_rot_base)
            self.ref_right_palm_rot_6d[:] = self._quat_xyzw_to_rot6d(ref_right_rot_base)

        curr_left_world_pos = self._rigid_body_pos_extend[:, self.left_palm_link_index, :]
        curr_right_world_pos = self._rigid_body_pos_extend[:, self.right_palm_link_index, :]
        curr_left_world_rot = self._rigid_body_rot_extend[:, self.left_palm_link_index, :]
        curr_right_world_rot = self._rigid_body_rot_extend[:, self.right_palm_link_index, :]

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
