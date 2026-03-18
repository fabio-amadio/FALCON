# FALCON Training Notes: Observations, Commands, Actions

This note documents the current setup used with:

- `+obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma`
- `+env=decoupled_locomotion_stand_height_waist_wbc_ma_diff_force`
- `+robot=g1/g1_29dof_waist_fakehand`
- `+algo=ppo_decoupled_wbc_ma`

## 1) Observation Pipeline

### Where observations are defined

- Observation fields and dimensions are declared in:
  - `humanoidverse/config/obs/dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma.yaml`
- Field extraction is done through `_get_obs_<name>()` methods in env classes.
- Per-field scaling/noise is applied in:
  - `humanoidverse/utils/helpers.py::parse_observation`
- Final concatenation is done in sorted key order in:
  - `humanoidverse/envs/legged_base_task/legged_robot_base_ma.py::_post_config_observation_callback`

### Actor observation (`actor_obs`)

Configured fields:

- `base_ang_vel` (3)
- `projected_gravity` (3)
- `command_lin_vel` (2)
- `command_ang_vel` (1)
- `command_stand` (1)
- `command_waist_dofs` (3)
- `command_base_height` (1)
- `ref_upper_dof_pos` (14)
- `dof_pos` (29)
- `dof_vel` (29)
- `actions` (29)

Per-frame actor dim:

- `3+3+2+1+1+3+1+14+29+29+29 = 115`

History:

- `history_length.actor_obs = 5`

Final actor input dim:

- `115 * 5 = 575`

### Critic observation (`critic_obs`)

Configured fields:

- `base_orientation` (4)
- `base_lin_vel` (3)
- `base_ang_vel` (3)
- `projected_gravity` (3)
- `command_lin_vel` (2)
- `command_ang_vel` (1)
- `command_stand` (1)
- `command_waist_dofs` (3)
- `command_base_height` (1)
- `ref_upper_dof_pos` (14)
- `dof_pos` (29)
- `dof_vel` (29)
- `actions` (29)
- `left_ee_apply_force` (3)
- `right_ee_apply_force` (3)

Per-frame critic dim:

- `128`

History:

- `history_length.critic_obs = 1`

Final critic input dim:

- `128`

## 2) Commands (`self.commands`)

For this env variant, command tensor size is 9:

- Defined in:
  - `humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma.py`

Index meaning:

- `commands[:, 0]`: target linear velocity x
- `commands[:, 1]`: target linear velocity y
- `commands[:, 2]`: yaw rate command (computed from heading error each step)
- `commands[:, 3]`: heading target
- `commands[:, 4]`: stand/tapping mode flag (`0=stand`, `1=tapping/walking`)
- `commands[:, 5]`: waist yaw command
- `commands[:, 6]`: waist roll command
- `commands[:, 7]`: waist pitch command
- `commands[:, 8]`: base height command

Ranges/probabilities come from:

- `humanoidverse/config/env/decoupled_locomotion_stand_height_waist_wbc_ma_diff_force.yaml`

Resampling:

- Commands are resampled every `locomotion_command_resampling_time` seconds.
- With current sim settings (`fps=200`, `control_decimation=4`):
  - env step `dt = 0.02`
  - `10.0 s` resample time means every `500` env steps.

## 3) Actions

### Action dimensions and split

From robot config:

- Total action dim: `29`
- Lower-body action dim: `15`
- Upper-body action dim: `14`
- Body keys: `["lower_body", "upper_body"]`

Defined in:

- `humanoidverse/config/robot/g1/g1_29dof_waist_fakehand.yaml`

### Multi-actor policy behavior

- Two actor heads are used in `PPOMultiActorCritic`:
  - lower body actor outputs 15 actions
  - upper body actor outputs 14 actions
- Final action passed to env is concatenation:
  - `[lower_body_actions, upper_body_actions]`

Implemented in:

- `humanoidverse/agents/decouple/ppo_decoupled_wbc_ma.py`

### Action preprocessing in env

- Actions are clipped first by `action_clip_value` in:
  - `humanoidverse/envs/legged_base_task/legged_robot_base_ma.py::_pre_physics_step`
- In torque computation, actions are scaled by `robot.control.action_scale` (here `0.25`).

### Residual upper-body action mode

In this setup:

- `residual_upper_body_action=True` (env config)
- Upper-body policy output is applied as residual wrt motion reference:
  - upper action is added to `(ref_upper_dof_pos - default_upper_dof_pos)`

Implemented in:

- `humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma.py::_compute_torques`

## 4) Useful Code Anchors

- Train entry:
  - `humanoidverse/train_agent.py`
- PPO (single):
  - `humanoidverse/agents/ppo/ppo.py`
- PPO decoupled multi-actor/multi-critic:
  - `humanoidverse/agents/decouple/ppo_decoupled_wbc_ma.py`
- Base env loop:
  - `humanoidverse/envs/legged_base_task/legged_robot_base_ma.py`
- Decoupled env with waist/base-height logic:
  - `humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma.py`
- Force-augmented env:
  - `humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma_diff_force.py`

