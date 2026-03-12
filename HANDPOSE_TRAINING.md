# Hand-Pose Instructions

## Training command

```bash
python humanoidverse/train_agent.py \
+exp=decoupled_locomotion_stand_height_waist_wbc_diff_force_handpose_ma_ppo_ma_env \
+simulator=isaacgym \
+domain_rand=domain_rand_rl_gym \
+robot=g1/g1_29dof_waist_fakehand \
+terrain=terrain_locomotion_plane \
+obs=dec_loco/g1_29dof_obs_diff_force_history_handpose_ma \
+opt=wandb \
num_envs=4096 \
project_name=g1_29dof_falcon \
experiment_name=g1_29dof_falcon_handpose \
obs.add_noise=True \
env.config.fix_upper_body_prob=0.3 \
robot.dof_effort_limit_scale=0.9
```

## What this exp already selects

The selected exp (`+exp=...handpose...`) already points to:
- custom hand-pose env config
- custom hand-pose rewards file

So you do **not** need to pass `+rewards=...` separately unless you want to override it.

## Common overrides

- Reduce torque/effort budget:
  - `robot.dof_effort_limit_scale=0.9`
- Change upper-body freeze probability:
  - `env.config.fix_upper_body_prob=0.3`
- Disable observation noise:
  - `obs.add_noise=False`

## Play command

```bash
python humanoidverse/eval_agent.py \
+checkpoint=/path/to/your/model.pt \
+env.config.debug_draw_hand_frames=True \
+env.config.debug_draw_hand_frames_num_envs=1 \
+env.config.debug_draw_hand_frame_axis_scale=0.12 \
+env.config.debug_draw_hand_frame_axis_scale_command=0.09 \
+headless=False
```
