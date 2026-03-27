from isaacgym.torch_utils import *

import torch

from humanoidverse.envs.decoupled_locomotion.decoupled_locomotion_stand_height_waist_wbc_ma_hand_pose import (
    LeggedRobotDecoupledLocomotionStanceHeightWBCForceHandPose,
)
from humanoidverse.envs.env_utils.visualization import Point


class LeggedRobotDecoupledLocomotionStanceHeightWBCHybridForceTracking(
    LeggedRobotDecoupledLocomotionStanceHeightWBCForceHandPose
):
    def __init__(self, config, device):
        super().__init__(config, device)
        self._init_hybrid_force_tracking_settings()
        self._init_hybrid_force_tracking_buffers()

    def _hybrid_force_cfg_get(self, key, default):
        force_cfg = self.config.get('hybrid_force_tracking', None)
        if force_cfg is None:
            return default
        return force_cfg.get(key, default)

    def _init_hybrid_force_tracking_settings(self):
        self.force_tracking_mode_name = self._hybrid_force_cfg_get('mode', 'binary')
        self.force_zero_probability = float(self._hybrid_force_cfg_get('zero_force_probability', 0.2))
        self.force_hold_duration_s = float(self._hybrid_force_cfg_get('hold_duration_s', 1.5))
        self.force_mode_upper_body_dofs_scale = float(
            self._hybrid_force_cfg_get('force_mode_upper_body_dofs_scale', 0.05)
        )
        self.force_mode_ee_lin_acc_scale = float(
            self._hybrid_force_cfg_get('force_mode_ee_lin_acc_scale', 0.5)
        )
        self.force_mode_ee_ang_acc_scale = float(
            self._hybrid_force_cfg_get('force_mode_ee_ang_acc_scale', 0.5)
        )

        self.force_cmd_low = torch.tensor(
            [
                self._hybrid_force_cfg_get('force_x_range', [-70.0, 70.0])[0],
                self._hybrid_force_cfg_get('force_y_range', [-70.0, 70.0])[0],
                self._hybrid_force_cfg_get('force_z_range', [-70.0, 70.0])[0],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.force_cmd_high = torch.tensor(
            [
                self._hybrid_force_cfg_get('force_x_range', [-70.0, 70.0])[1],
                self._hybrid_force_cfg_get('force_y_range', [-70.0, 70.0])[1],
                self._hybrid_force_cfg_get('force_z_range', [-70.0, 70.0])[1],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.force_clip_low = self.force_cmd_low.clone()
        self.force_clip_high = self.force_cmd_high.clone()

        self.force_ramp_duration_range_s = self._hybrid_force_cfg_get('ramp_duration_s', [2.0, 4.0])
        self.force_kp_range = self._hybrid_force_cfg_get('kp_range', [25.0, 400.0])
        desired_base_height = self.config.rewards.get('desired_base_height', None)
        if desired_base_height is None:
            desired_base_height = float(self.simulator.robot_root_states[0, 2].item())
        self.force_frame_base_height = float(desired_base_height)

        self.identity_rot6d = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

    def _init_hybrid_force_tracking_buffers(self):
        self.force_tracking_mode = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        self.force_mode_needs_anchor_reset = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.force_tracking_zero_force = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self.force_phase_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.force_ramp_steps = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.force_hold_steps = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.force_mode_motion_times = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        self.force_kp = torch.ones(self.num_envs, 1, dtype=torch.float32, device=self.device) * self.force_kp_range[0]
        self.force_kd = 2.0 * torch.sqrt(self.force_kp.clamp_min(0.0))

        self.left_force_cmd_peak = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.right_force_cmd_peak = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.left_force_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.right_force_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)

        self.left_force_anchor = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.right_force_anchor = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.left_force_anchor_world = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.right_force_anchor_world = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)

        self.curr_left_palm_pos_yaw = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.curr_right_palm_pos_yaw = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.ref_left_palm_pos_yaw = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.ref_right_palm_pos_yaw = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.prev_left_palm_pos_yaw = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.prev_right_palm_pos_yaw = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.curr_left_palm_vel_yaw = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.curr_right_palm_vel_yaw = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.palm_yaw_state_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.left_applied_force_yaw = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.right_applied_force_yaw = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.force_error_x = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.force_error_y = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.force_error_z = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.force_error_norm = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.phase_episode_sums = {
            "rew/upper_body/reward_tracking_upper_body_dofs/pose_mode": torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            ),
            "rew/upper_body/reward_tracking_palm_pos/pose_mode": torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            ),
            "rew/upper_body/reward_tracking_palm_rot/pose_mode": torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            ),
            "rew/upper_body/reward_tracking_upper_body_dofs/force_mode": torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            ),
            "rew/upper_body/reward_tracking_palm_rot/force_mode": torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            ),
            "rew/upper_body/reward_tracking_force/force_mode": torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            ),
        }

    def _log_phase_reward(self, log_key, reward, mask):
        return

    def _log_phase_metric(self, log_key, values, mask):
        if torch.any(mask):
            self.log_dict[log_key] = values[mask].mean()
        else:
            self.log_dict[log_key] = torch.tensor(0.0, device=self.device)

    def _accumulate_phase_episode_reward(self, episode_key, reward, mask, reward_name):
        if not torch.any(mask):
            return
        self.phase_episode_sums[episode_key][mask] += reward[mask] * self.reward_scales[reward_name]

    def _sample_force_tracking_mode(self, env_ids):
        if self.force_tracking_mode_name == 'mixed':
            self.force_tracking_mode[env_ids, 0] = torch.rand(len(env_ids), device=self.device)
        elif self.force_tracking_mode_name == 'binary':
            self.force_tracking_mode[env_ids, 0] = torch.randint(
                0, 2, (len(env_ids),), device=self.device
            ).float()
        elif self.force_tracking_mode_name == 'force':
            self.force_tracking_mode[env_ids, 0] = 1.0
        elif self.force_tracking_mode_name == 'position':
            self.force_tracking_mode[env_ids, 0] = 0.0
        else:
            raise ValueError(f'Unsupported paper force tracking mode: {self.force_tracking_mode_name}')

    def _resample_hybrid_force_tracking(self, env_ids):
        if len(env_ids) == 0:
            return

        self._sample_force_tracking_mode(env_ids)
        self.force_phase_step[env_ids] = 0
        self.force_mode_needs_anchor_reset[env_ids] = self.force_tracking_mode[env_ids, 0] > 0.5
        self.palm_yaw_state_valid[env_ids] = False
        current_motion_times = (
            (self.episode_length_buf[env_ids] * self.dt + self.motion_start_times[env_ids])
            * (1.0 - self.fix_upper_body[env_ids])
            + self.fix_upper_body_motion_times[env_ids] * self.fix_upper_body[env_ids]
        )
        self.force_mode_motion_times[env_ids] = current_motion_times

        ramp_duration_s = torch_rand_float(
            self.force_ramp_duration_range_s[0],
            self.force_ramp_duration_range_s[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(-1)
        self.force_ramp_steps[env_ids] = torch.clamp(
            torch.round(ramp_duration_s / self.dt).long(), min=1
        )
        self.force_hold_steps[env_ids] = max(1, int(round(self.force_hold_duration_s / self.dt)))

        self.force_kp[env_ids, 0] = torch_rand_float(
            self.force_kp_range[0], self.force_kp_range[1], (len(env_ids), 1), device=self.device
        ).squeeze(-1)
        self.force_kd[env_ids, 0] = 2.0 * torch.sqrt(self.force_kp[env_ids, 0].clamp_min(0.0))

        force_span = self.force_cmd_high - self.force_cmd_low
        self.left_force_cmd_peak[env_ids] = (
            torch.rand((len(env_ids), 3), device=self.device) * force_span + self.force_cmd_low
        )
        self.right_force_cmd_peak[env_ids] = (
            torch.rand((len(env_ids), 3), device=self.device) * force_span + self.force_cmd_low
        )

        zero_force_mask = torch.rand(len(env_ids), device=self.device) < self.force_zero_probability
        force_mode_mask = self.force_tracking_mode[env_ids, 0] > 0.5
        zero_force_mask = zero_force_mask & force_mode_mask
        zero_force_env_ids = env_ids[zero_force_mask]
        self.force_tracking_zero_force[env_ids] = False
        self.force_tracking_zero_force[zero_force_env_ids] = True
        if len(zero_force_env_ids) > 0:
            self.left_force_cmd_peak[zero_force_env_ids] = 0.0
            self.right_force_cmd_peak[zero_force_env_ids] = 0.0

        position_env_ids = env_ids[self.force_tracking_mode[env_ids, 0] <= 0.5]
        if len(position_env_ids) > 0:
            self.left_force_cmd_peak[position_env_ids] = 0.0
            self.right_force_cmd_peak[position_env_ids] = 0.0

        self.left_force_cmd[env_ids] = 0.0
        self.right_force_cmd[env_ids] = 0.0

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        self._resample_hybrid_force_tracking(env_ids)

    def _get_yaw_frame_pose(self):
        yaw = self.rpy[:, 2]
        yaw_frame_rot = quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
        )
        yaw_frame_pos = self.simulator._rigid_body_pos[:, self.torso_index, :].clone()
        yaw_frame_pos[:, 2] = self.force_frame_base_height
        return yaw_frame_pos, yaw_frame_rot

    def _update_force_command_profile(self):
        self.left_force_cmd.zero_()
        self.right_force_cmd.zero_()

        force_mask = self.force_tracking_mode[:, 0] > 0.5
        if not torch.any(force_mask):
            return

        force_env_ids = torch.where(force_mask)[0]
        phase_step = self.force_phase_step[force_env_ids].float()
        ramp_steps = self.force_ramp_steps[force_env_ids].float().clamp_min(1.0)
        hold_steps = self.force_hold_steps[force_env_ids].float()

        command_scale = torch.zeros_like(phase_step)
        ramp_up_mask = phase_step < ramp_steps
        hold_mask = (phase_step >= ramp_steps) & (phase_step < ramp_steps + hold_steps)
        ramp_down_mask = (phase_step >= ramp_steps + hold_steps) & (
            phase_step < 2.0 * ramp_steps + hold_steps
        )

        command_scale[ramp_up_mask] = phase_step[ramp_up_mask] / ramp_steps[ramp_up_mask]
        command_scale[hold_mask] = 1.0
        command_scale[ramp_down_mask] = 1.0 - (
            (phase_step[ramp_down_mask] - ramp_steps[ramp_down_mask] - hold_steps[ramp_down_mask])
            / ramp_steps[ramp_down_mask]
        )

        command_scale = command_scale.unsqueeze(-1)
        self.left_force_cmd[force_env_ids] = self.left_force_cmd_peak[force_env_ids] * command_scale
        self.right_force_cmd[force_env_ids] = self.right_force_cmd_peak[force_env_ids] * command_scale

        self.force_phase_step[force_env_ids] += 1

    def _update_hybrid_force_tracking_frame_buffers(self):
        yaw_frame_pos, yaw_frame_rot = self._get_yaw_frame_pose()

        self.curr_left_palm_pos_yaw[:], _ = self._world_pose_to_local_frame(
            yaw_frame_pos,
            yaw_frame_rot,
            self.curr_left_palm_pos_world,
            self.curr_left_palm_rot_world_xyzw,
        )
        self.curr_right_palm_pos_yaw[:], _ = self._world_pose_to_local_frame(
            yaw_frame_pos,
            yaw_frame_rot,
            self.curr_right_palm_pos_world,
            self.curr_right_palm_rot_world_xyzw,
        )
        self.ref_left_palm_pos_yaw[:], _ = self._world_pose_to_local_frame(
            yaw_frame_pos,
            yaw_frame_rot,
            self.ref_left_palm_pos_world,
            self.ref_left_palm_rot_world_xyzw,
        )
        self.ref_right_palm_pos_yaw[:], _ = self._world_pose_to_local_frame(
            yaw_frame_pos,
            yaw_frame_rot,
            self.ref_right_palm_pos_world,
            self.ref_right_palm_rot_world_xyzw,
        )

        valid_env_ids = torch.where(self.palm_yaw_state_valid)[0]
        if len(valid_env_ids) > 0:
            self.curr_left_palm_vel_yaw[valid_env_ids] = (
                self.curr_left_palm_pos_yaw[valid_env_ids] - self.prev_left_palm_pos_yaw[valid_env_ids]
            ) / self.dt
            self.curr_right_palm_vel_yaw[valid_env_ids] = (
                self.curr_right_palm_pos_yaw[valid_env_ids] - self.prev_right_palm_pos_yaw[valid_env_ids]
            ) / self.dt

        invalid_env_ids = torch.where(~self.palm_yaw_state_valid)[0]
        if len(invalid_env_ids) > 0:
            self.curr_left_palm_vel_yaw[invalid_env_ids] = 0.0
            self.curr_right_palm_vel_yaw[invalid_env_ids] = 0.0

        self.prev_left_palm_pos_yaw[:] = self.curr_left_palm_pos_yaw
        self.prev_right_palm_pos_yaw[:] = self.curr_right_palm_pos_yaw
        self.palm_yaw_state_valid[:] = True

        # Mirror the released compliance setup: the spring-damper field is anchored to
        # an internal palm reference while that reference is hidden from the actor in force mode.
        self.left_force_anchor[:] = self.ref_left_palm_pos_yaw
        self.right_force_anchor[:] = self.ref_right_palm_pos_yaw
        self.left_force_anchor_world[:] = self.ref_left_palm_pos_world
        self.right_force_anchor_world[:] = self.ref_right_palm_pos_world
        self.force_mode_needs_anchor_reset[:] = False

        self._update_force_command_profile()

        force_mode_mask = self.force_tracking_mode[:, 0] > 0.5
        position_mode_mask = ~force_mode_mask
        self.log_dict.pop("palm_pos_error", None)
        self.log_dict.pop("palm_rot_error", None)
        self._log_phase_metric("palm_pos_error/pose_mode", self.palm_pos_error, position_mode_mask)
        self._log_phase_metric("palm_rot_error/pose_mode", self.palm_rot_error, position_mode_mask)
        self._log_phase_metric("palm_rot_error/force_mode", self.palm_rot_error, force_mode_mask)

        self.log_dict['mode_fraction/pose_mode'] = position_mode_mask.float().mean()
        self.log_dict['mode_fraction/force_mode'] = force_mode_mask.float().mean()
        self.log_dict['zero_force_fraction/force_mode'] = self.force_tracking_zero_force.float().mean()

    def _pre_compute_observations_callback(self):
        force_mode_mask = self.force_tracking_mode[:, 0] > 0.5
        original_fix_upper_body = self.fix_upper_body
        original_fix_upper_body_motion_times = self.fix_upper_body_motion_times

        if torch.any(force_mode_mask):
            self.fix_upper_body = torch.maximum(original_fix_upper_body, force_mode_mask.float())
            self.fix_upper_body_motion_times = torch.where(
                force_mode_mask,
                self.force_mode_motion_times,
                original_fix_upper_body_motion_times,
            )

        try:
            super()._pre_compute_observations_callback()
        finally:
            self.fix_upper_body = original_fix_upper_body
            self.fix_upper_body_motion_times = original_fix_upper_body_motion_times

        self._update_hybrid_force_tracking_frame_buffers()
        self.apply_force_pos_tensor[:, self.left_hand_link_index, :] = self.curr_left_palm_pos_world
        self.apply_force_pos_tensor[:, self.right_hand_link_index, :] = self.curr_right_palm_pos_world

    def _calculate_ee_forces(self):
        self.apply_force_tensor.zero_()
        self.left_applied_force_yaw.zero_()
        self.right_applied_force_yaw.zero_()
        self.left_ee_apply_force.zero_()
        self.right_ee_apply_force.zero_()

        force_env_ids = torch.where(self.force_tracking_mode[:, 0] > 0.5)[0]
        if len(force_env_ids) == 0:
            return

        left_force_yaw = (
            self.force_kp[force_env_ids] * (self.left_force_anchor[force_env_ids] - self.curr_left_palm_pos_yaw[force_env_ids])
            - self.force_kd[force_env_ids] * self.curr_left_palm_vel_yaw[force_env_ids]
        )
        right_force_yaw = (
            self.force_kp[force_env_ids] * (self.right_force_anchor[force_env_ids] - self.curr_right_palm_pos_yaw[force_env_ids])
            - self.force_kd[force_env_ids] * self.curr_right_palm_vel_yaw[force_env_ids]
        )

        left_force_yaw = torch.clip(left_force_yaw, self.force_clip_low, self.force_clip_high)
        right_force_yaw = torch.clip(right_force_yaw, self.force_clip_low, self.force_clip_high)

        _, yaw_frame_rot = self._get_yaw_frame_pose()
        left_force_world = quat_rotate(yaw_frame_rot[force_env_ids], left_force_yaw)
        right_force_world = quat_rotate(yaw_frame_rot[force_env_ids], right_force_yaw)

        self.left_applied_force_yaw[force_env_ids] = left_force_yaw
        self.right_applied_force_yaw[force_env_ids] = right_force_yaw
        self.left_ee_apply_force[force_env_ids] = left_force_yaw
        self.right_ee_apply_force[force_env_ids] = right_force_yaw

        self.apply_force_tensor[force_env_ids, self.left_hand_link_index, :] = left_force_world
        self.apply_force_tensor[force_env_ids, self.right_hand_link_index, :] = right_force_world

    def _update_force_tracking_metrics(self):
        self.force_error_x.zero_()
        self.force_error_y.zero_()
        self.force_error_z.zero_()
        self.force_error_norm.zero_()

        force_mode_mask = self.force_tracking_mode[:, 0] > 0.5
        if not torch.any(force_mode_mask):
            self.log_dict['force_error_x/force_mode'] = torch.tensor(0.0, device=self.device)
            self.log_dict['force_error_y/force_mode'] = torch.tensor(0.0, device=self.device)
            self.log_dict['force_error_z/force_mode'] = torch.tensor(0.0, device=self.device)
            self.log_dict['force_error_norm/force_mode'] = torch.tensor(0.0, device=self.device)
            return

        left_force_error = self.left_applied_force_yaw - self.left_force_cmd
        right_force_error = self.right_applied_force_yaw - self.right_force_cmd

        self.force_error_x[:] = 0.5 * (
            torch.abs(left_force_error[:, 0]) + torch.abs(right_force_error[:, 0])
        )
        self.force_error_y[:] = 0.5 * (
            torch.abs(left_force_error[:, 1]) + torch.abs(right_force_error[:, 1])
        )
        self.force_error_z[:] = 0.5 * (
            torch.abs(left_force_error[:, 2]) + torch.abs(right_force_error[:, 2])
        )
        self.force_error_norm[:] = 0.5 * (
            torch.linalg.norm(left_force_error, dim=1) + torch.linalg.norm(right_force_error, dim=1)
        )

        self.log_dict['force_error_x/force_mode'] = self.force_error_x[force_mode_mask].mean()
        self.log_dict['force_error_y/force_mode'] = self.force_error_y[force_mode_mask].mean()
        self.log_dict['force_error_z/force_mode'] = self.force_error_z[force_mode_mask].mean()
        self.log_dict['force_error_norm/force_mode'] = self.force_error_norm[force_mode_mask].mean()

    def _reward_tracking_force(self):
        self._update_force_tracking_metrics()

        force_mode_mask = self.force_tracking_mode[:, 0] > 0.5
        reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        if not torch.any(force_mode_mask):
            return reward

        left_force_error = self.left_applied_force_yaw - self.left_force_cmd
        right_force_error = self.right_applied_force_yaw - self.right_force_cmd
        combined_force_error = torch.linalg.norm(left_force_error, dim=1) + torch.linalg.norm(
            right_force_error, dim=1
        )
        reward[force_mode_mask] = torch.exp(
            -combined_force_error[force_mode_mask]
            / self.config.rewards.reward_tracking_sigma.force
        )
        self._log_phase_reward("force/reward_tracking_force", reward, force_mode_mask)
        self._accumulate_phase_episode_reward(
            "rew/upper_body/reward_tracking_force/force_mode", reward, force_mode_mask, "tracking_force"
        )
        return reward

    def _reward_tracking_palm_pos(self):
        reward = super()._reward_tracking_palm_pos()
        position_mode_mask = self.force_tracking_mode[:, 0] <= 0.5
        reward = reward * position_mode_mask.float()
        self._log_phase_reward("pose/reward_tracking_palm_pos", reward, position_mode_mask)
        self._accumulate_phase_episode_reward(
            "rew/upper_body/reward_tracking_palm_pos/pose_mode", reward, position_mode_mask, "tracking_palm_pos"
        )
        return reward

    def _reward_tracking_palm_rot(self):
        reward = super()._reward_tracking_palm_rot()
        force_mode_mask = self.force_tracking_mode[:, 0] > 0.5
        position_mode_mask = ~force_mode_mask
        self._log_phase_reward("pose/reward_tracking_palm_rot", reward, position_mode_mask)
        self._log_phase_reward("force/reward_tracking_palm_rot", reward, force_mode_mask)
        self._accumulate_phase_episode_reward(
            "rew/upper_body/reward_tracking_palm_rot/pose_mode", reward, position_mode_mask, "tracking_palm_rot"
        )
        self._accumulate_phase_episode_reward(
            "rew/upper_body/reward_tracking_palm_rot/force_mode", reward, force_mode_mask, "tracking_palm_rot"
        )
        return reward

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

        force_mode_mask = self.force_tracking_mode[:, 0] > 0.5
        position_mode_mask = ~force_mode_mask
        reward_scale = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        reward_scale[force_mode_mask] = self.force_mode_upper_body_dofs_scale
        masked_reward = upper_body_dofs_reward * reward_scale
        self.upper_body_dofs_tracking_reward += masked_reward

        self._log_phase_reward("pose/reward_tracking_upper_body_dofs", masked_reward, position_mode_mask)
        self._log_phase_reward("force/reward_tracking_upper_body_dofs", masked_reward, force_mode_mask)
        self._accumulate_phase_episode_reward(
            "rew/upper_body/reward_tracking_upper_body_dofs/pose_mode",
            masked_reward,
            position_mode_mask,
            "tracking_upper_body_dofs",
        )
        self._accumulate_phase_episode_reward(
            "rew/upper_body/reward_tracking_upper_body_dofs/force_mode",
            masked_reward,
            force_mode_mask,
            "tracking_upper_body_dofs",
        )

        self._log_phase_metric("upper_body_dofs_error/pose_mode", self.upper_body_dofs_error, position_mode_mask)
        self._log_phase_metric("upper_body_dofs_error/force_mode", self.upper_body_dofs_error, force_mode_mask)

        return masked_reward

    def _reward_penalty_ee_lin_acc(self):
        reward = super()._reward_penalty_ee_lin_acc()
        force_mode_mask = self.force_tracking_mode[:, 0] > 0.5
        reward_scale = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        reward_scale[force_mode_mask] = self.force_mode_ee_lin_acc_scale
        return reward * reward_scale

    def _reward_penalty_ee_ang_acc(self):
        reward = super()._reward_penalty_ee_ang_acc()
        force_mode_mask = self.force_tracking_mode[:, 0] > 0.5
        reward_scale = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        reward_scale[force_mode_mask] = self.force_mode_ee_ang_acc_scale
        return reward * reward_scale

    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        if len(env_ids) == 0:
            return

        super().reset_envs_idx(env_ids, target_states=target_states, target_buf=target_buf)

        for key in [
            "rew_tracking_upper_body_dofs",
            "rew_tracking_palm_pos",
            "rew_tracking_palm_rot",
            "rew_tracking_force",
        ]:
            self.extras["episode"].pop(key, None)

        for key, value in self.phase_episode_sums.items():
            self.extras["episode"][key] = torch.mean(value[env_ids]) / self.max_episode_length_s
            value[env_ids] = 0.0

    def _get_obs_ref_left_palm_pos(self):
        ref_left_palm_pos = super()._get_obs_ref_left_palm_pos().clone()
        force_mode_mask = self.force_tracking_mode[:, 0] > 0.5
        ref_left_palm_pos[force_mode_mask] = 0.0
        return ref_left_palm_pos

    def _get_obs_ref_right_palm_pos(self):
        ref_right_palm_pos = super()._get_obs_ref_right_palm_pos().clone()
        force_mode_mask = self.force_tracking_mode[:, 0] > 0.5
        ref_right_palm_pos[force_mode_mask] = 0.0
        return ref_right_palm_pos

    def _get_obs_ref_left_palm_rot_6d(self):
        ref_left_palm_rot_6d = super()._get_obs_ref_left_palm_rot_6d().clone()
        force_mode_mask = self.force_tracking_mode[:, 0] > 0.5
        ref_left_palm_rot_6d[force_mode_mask] = self.identity_rot6d[force_mode_mask]
        return ref_left_palm_rot_6d

    def _get_obs_ref_right_palm_rot_6d(self):
        ref_right_palm_rot_6d = super()._get_obs_ref_right_palm_rot_6d().clone()
        force_mode_mask = self.force_tracking_mode[:, 0] > 0.5
        ref_right_palm_rot_6d[force_mode_mask] = self.identity_rot6d[force_mode_mask]
        return ref_right_palm_rot_6d

    def _get_obs_force_tracking_mode(self):
        return self.force_tracking_mode

    def _get_obs_left_force_cmd(self):
        return self.left_force_cmd

    def _get_obs_right_force_cmd(self):
        return self.right_force_cmd

    def _draw_force_vector(self, env_id, origin, force_world, color, force_scale=0.015, line_width=0.01):
        color_point = Point(torch.tensor(color, dtype=torch.float32, device=self.device))
        end_point = origin + force_world * force_scale
        for _ in range(10):
            start_jitter = torch.rand(3, device=self.device) * line_width
            end_jitter = torch.rand(3, device=self.device) * line_width
            self.simulator.draw_line(
                Point(origin + start_jitter),
                Point(end_point + end_jitter),
                color_point,
                env_id,
            )

    def _draw_force_tracking_debug_vis(self):
        force_mode_env_ids = torch.where(self.force_tracking_mode[:, 0] > 0.5)[0]
        if len(force_mode_env_ids) == 0:
            return

        _, yaw_frame_rot = self._get_yaw_frame_pose()
        left_desired_force_world = quat_rotate(yaw_frame_rot, self.left_force_cmd)
        right_desired_force_world = quat_rotate(yaw_frame_rot, self.right_force_cmd)

        desired_origin_offset = torch.tensor([0.0, 0.0, 0.04], dtype=torch.float32, device=self.device)
        applied_color = (0.851, 0.144, 0.07)
        desired_color = (0.05, 0.65, 0.95)

        for env_id in force_mode_env_ids.tolist():
            left_origin = self.curr_left_palm_pos_world[env_id]
            right_origin = self.curr_right_palm_pos_world[env_id]

            self._draw_force_vector(
                env_id,
                left_origin,
                self.apply_force_tensor[env_id, self.left_hand_link_index, :],
                applied_color,
            )
            self._draw_force_vector(
                env_id,
                right_origin,
                self.apply_force_tensor[env_id, self.right_hand_link_index, :],
                applied_color,
            )
            self._draw_force_vector(
                env_id,
                left_origin + desired_origin_offset,
                left_desired_force_world[env_id],
                desired_color,
            )
            self._draw_force_vector(
                env_id,
                right_origin + desired_origin_offset,
                right_desired_force_world[env_id],
                desired_color,
            )

    def _draw_debug_vis(self):
        hand_pose_debug_enabled = bool(self._hand_pose_debug_cfg_get("enabled", False)) or bool(
            getattr(self.simulator, "vis_hand_pose_tracking", False)
        )
        force_debug_enabled = bool(getattr(self.simulator, "vis_force_range", False))

        self.simulator.clear_lines()

        if force_debug_enabled:
            self._draw_force_tracking_debug_vis()

        if hand_pose_debug_enabled:
            self._draw_hand_pose_debug_vis()
