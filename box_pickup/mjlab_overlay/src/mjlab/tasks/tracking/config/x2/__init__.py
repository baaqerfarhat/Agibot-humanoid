from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .box_env_cfgs import x2_box_tracking_env_cfg
from .env_cfgs import x2_flat_tracking_env_cfg
from .rl_cfg import x2_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-X2",
  env_cfg=x2_flat_tracking_env_cfg(),
  play_env_cfg=x2_flat_tracking_env_cfg(play=True),
  rl_cfg=x2_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-X2-No-State-Estimation",
  env_cfg=x2_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=x2_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=x2_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-X2-Box",
  env_cfg=x2_box_tracking_env_cfg(),
  play_env_cfg=x2_box_tracking_env_cfg(play=True),
  rl_cfg=x2_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
