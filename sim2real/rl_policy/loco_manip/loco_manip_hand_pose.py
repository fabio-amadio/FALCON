import sys
import argparse
import yaml

import numpy as np

sys.path.append("../")
sys.path.append("./rl_policy")

from termcolor import colored

from sim2real.rl_policy.base_policy import BasePolicy
from sim2real.rl_policy.dec_loco.dec_loco import DecLocomotionPolicy
from sim2real.utils.math import quaternion_to_rotation_matrix, rpy_to_quat


class LocoManipHandPosePolicy(DecLocomotionPolicy):
    def __init__(self, config, model_path, rl_rate=50, policy_action_scale=0.25):
        super().__init__(config, model_path, rl_rate, policy_action_scale)

        self.policy_mode = self.config.get("policy_mode", "hand_pose")
        self.upper_body_action_mode = self.config.get("upper_body_action_mode", "direct_policy")
        self.hand_pose_target_source = self.config.get("hand_pose_target_source", "fixed")
        if self.policy_mode != "hand_pose":
            raise ValueError(f"Unsupported policy_mode: {self.policy_mode}")
        if self.upper_body_action_mode != "direct_policy":
            raise ValueError(f"Unsupported upper_body_action_mode: {self.upper_body_action_mode}")
        if self.hand_pose_target_source != "fixed":
            raise NotImplementedError(
                f"Unsupported hand_pose_target_source: {self.hand_pose_target_source}"
            )

        self.ref_left_palm_pos = np.zeros((1, 3), dtype=np.float32)
        self.ref_right_palm_pos = np.zeros((1, 3), dtype=np.float32)
        self.ref_left_palm_rot_6d = np.zeros((1, 6), dtype=np.float32)
        self.ref_right_palm_rot_6d = np.zeros((1, 6), dtype=np.float32)

        self.fixed_hand_pose_targets = self.config.get("fixed_hand_pose_targets", {})
        if not self.fixed_hand_pose_targets:
            raise ValueError("fixed_hand_pose_targets must be provided for hand-pose deployment")

        self.fixed_hand_pose_target_names = list(self.fixed_hand_pose_targets.keys())
        target_name = self.config.get(
            "fixed_hand_pose_target_name", self.fixed_hand_pose_target_names[0]
        )
        if target_name not in self.fixed_hand_pose_targets:
            raise ValueError(
                f"Unknown fixed hand pose target '{target_name}'. "
                f"Available targets: {self.fixed_hand_pose_target_names}"
            )
        self.active_target_index = self.fixed_hand_pose_target_names.index(target_name)
        self._set_fixed_hand_pose_target_by_index(self.active_target_index, announce=False)

    def _rotation_matrix_to_rot6d(self, rotation_matrix):
        first_two_columns = rotation_matrix[:, :2]
        return first_two_columns.T.reshape(1, 6).astype(np.float32)

    def _target_rotation_to_rot6d(self, target_cfg, side):
        rot6d_key = f"{side}_rot_6d"
        quat_key = f"{side}_quat_wxyz"
        rpy_key = f"{side}_rpy"

        if rot6d_key in target_cfg:
            rot6d = np.asarray(target_cfg[rot6d_key], dtype=np.float32).reshape(1, 6)
            return rot6d

        if quat_key in target_cfg:
            quat_wxyz = np.asarray(target_cfg[quat_key], dtype=np.float32)
        else:
            rpy = np.asarray(target_cfg.get(rpy_key, [0.0, 0.0, 0.0]), dtype=np.float32)
            quat_wxyz = rpy_to_quat(rpy)

        rotation_matrix = quaternion_to_rotation_matrix(quat_wxyz, w_first=True)
        return self._rotation_matrix_to_rot6d(rotation_matrix)

    def _set_fixed_hand_pose_target_by_index(self, target_index, announce=True):
        self.active_target_index = target_index % len(self.fixed_hand_pose_target_names)
        self.active_hand_pose_target_name = self.fixed_hand_pose_target_names[self.active_target_index]
        target_cfg = self.fixed_hand_pose_targets[self.active_hand_pose_target_name]

        self.ref_left_palm_pos = np.asarray(target_cfg["left_pos"], dtype=np.float32).reshape(1, 3)
        self.ref_right_palm_pos = np.asarray(target_cfg["right_pos"], dtype=np.float32).reshape(1, 3)
        self.ref_left_palm_rot_6d = self._target_rotation_to_rot6d(target_cfg, "left")
        self.ref_right_palm_rot_6d = self._target_rotation_to_rot6d(target_cfg, "right")

        if announce:
            self.logger.info(
                colored(
                    f"Switched hand pose target to '{self.active_hand_pose_target_name}'",
                    "green",
                )
            )

    def _cycle_fixed_hand_pose_target(self, step):
        self._set_fixed_hand_pose_target_by_index(self.active_target_index + step)

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_buffer_dict = BasePolicy.get_current_obs_buffer_dict(self, robot_state_data)
        current_obs_buffer_dict["actions"] = self.last_policy_action
        current_obs_buffer_dict["command_lin_vel"] = self.lin_vel_command
        current_obs_buffer_dict["command_ang_vel"] = self.ang_vel_command
        current_obs_buffer_dict["command_stand"] = self.stand_command
        current_obs_buffer_dict["command_waist_dofs"] = self.waist_dofs_command
        current_obs_buffer_dict["command_base_height"] = self.base_height_command
        current_obs_buffer_dict["ref_left_palm_pos"] = self.ref_left_palm_pos
        current_obs_buffer_dict["ref_right_palm_pos"] = self.ref_right_palm_pos
        current_obs_buffer_dict["ref_left_palm_rot_6d"] = self.ref_left_palm_rot_6d
        current_obs_buffer_dict["ref_right_palm_rot_6d"] = self.ref_right_palm_rot_6d
        return current_obs_buffer_dict

    def rl_inference(self, robot_state_data):
        obs = self.prepare_obs_for_rl(robot_state_data)
        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)

        self.last_policy_action = policy_action.copy()
        scaled_policy_action = policy_action * self.policy_action_scale
        return scaled_policy_action

    def _handle_base_height_control(self, keycode):
        if keycode == "1":
            self.base_height_command[0, 0] += 0.1
        elif keycode == "2":
            self.base_height_command[0, 0] -= 0.1

    def _handle_joystick_base_height_control(self, cur_key):
        if cur_key == "B+up":
            self.base_height_command[0, 0] += 0.1
        elif cur_key == "B+down":
            self.base_height_command[0, 0] -= 0.1

    def handle_keyboard_button(self, keycode):
        BasePolicy.handle_keyboard_button(self, keycode)

        if keycode in ["w", "s", "a", "d"]:
            self._handle_velocity_control(keycode)
        elif keycode in ["q", "e"]:
            self._handle_angular_velocity_control(keycode)
        elif keycode == "=":
            self._handle_stand_command()
        elif keycode == "z":
            self._handle_zero_velocity()
        elif keycode == ",":
            self.waist_dofs_command[:, 0] -= 0.2
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif keycode == ".":
            self.waist_dofs_command[:, 0] += 0.2
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif keycode in ["1", "2"]:
            self._handle_base_height_control(keycode)
        elif keycode == "p":
            self._cycle_fixed_hand_pose_target(1)

        self._print_control_status()

    def handle_joystick_button(self, cur_key):
        BasePolicy.handle_joystick_button(self, cur_key)

        if cur_key == "R2":
            self._handle_stand_command()
        elif cur_key == "L2":
            self._handle_zero_velocity()
        elif cur_key in ["B+up", "B+down"]:
            self._handle_joystick_base_height_control(cur_key)
        elif cur_key == "select+left":
            self.waist_dofs_command[:, 0] -= 0.1
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif cur_key == "select+right":
            self.waist_dofs_command[:, 0] += 0.1
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif cur_key == "select+up":
            self.waist_dofs_command[:, 2] -= 0.05
            self.logger.info(colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "select+down":
            self.waist_dofs_command[:, 2] += 0.05
            self.logger.info(colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "R1+left":
            self._cycle_fixed_hand_pose_target(-1)
        elif cur_key == "R1+right":
            self._cycle_fixed_hand_pose_target(1)

    def _print_control_status(self):
        super()._print_control_status()
        print(f"Base height command: {self.base_height_command}")
        print(f"Waist dofs command: {self.waist_dofs_command}")
        print(f"Hand pose target: {self.active_hand_pose_target_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof_falcon_hand_pose.yaml", help="config file")
    parser.add_argument("--model_path", type=str, help="path to the ONNX model file")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided either via --model_path argument or in config file")

    policy = LocoManipHandPosePolicy(
        config=config, model_path=model_path, rl_rate=50, policy_action_scale=0.25
    )
    policy.run()
