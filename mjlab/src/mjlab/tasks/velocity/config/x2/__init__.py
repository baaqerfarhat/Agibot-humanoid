from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  x2_flat_deploy_env_cfg,
  x2_flat_env_cfg,
  x2_rough_env_cfg,
)
from .rl_cfg import x2_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-X2",
  env_cfg=x2_flat_env_cfg(),
  play_env_cfg=x2_flat_env_cfg(play=True),
  rl_cfg=x2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-X2-Deploy",
  env_cfg=x2_flat_deploy_env_cfg(),
  play_env_cfg=x2_flat_deploy_env_cfg(play=True),
  rl_cfg=x2_ppo_runner_cfg(experiment_name="x2_velocity_deploy"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-X2",
  env_cfg=x2_rough_env_cfg(),
  play_env_cfg=x2_rough_env_cfg(play=True),
  rl_cfg=x2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
