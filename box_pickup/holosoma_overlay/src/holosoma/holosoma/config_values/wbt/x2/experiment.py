"""Whole Body Tracking experiment presets for the AgiBot X2 (31-DoF) robot.

Mirrors the Unitree G1 WBT presets but wires in the X2 RobotConfig, the X2
tracked-body selection, and the X2 reference-motion .npz files. The
``*_w_object`` variants spawn the physical box (largebox) in IsaacSim and add
the object position/orientation tracking rewards + termination, reproducing the
OmniRetarget / InterMimic-style object-interaction whole-body tracking task.
"""

from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import (
    algo,
    robot,
    simulator,
    terrain,
)
from holosoma.config_values.loco.g1.action import g1_29dof_joint_pos
from holosoma.config_values.wbt.x2 import command, curriculum, observation, randomization, reward, termination

_X2_WBT_INIT_HEIGHT = 0.9

x2_31dof_wbt = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="x2_31dof_wbt_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=30000,
            num_learning_epochs=5,
            # Purely the checkpoint-write frequency (NOT a training-dynamics
            # parameter): identical to G1's recipe otherwise, just saved more
            # often so progress is always captured. Set back to 500 from G1's
            # 4000 for practical checkpointing on a shared machine.
            save_interval=500,
            entropy_coef=0.005,
            init_noise_std=1.0,
            # Match G1's canonical WBT recipe exactly.
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            init_at_random_ep_len=True,
            empirical_normalization=True,
            use_symmetry=False,
            actor_optimizer=replace(algo.ppo.config.actor_optimizer, weight_decay=0.000),
            critic_optimizer=replace(algo.ppo.config.critic_optimizer, weight_decay=0.000),
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    ),
    robot=replace(
        robot.x2_31dof,
        control=replace(
            robot.x2_31dof.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=replace(robot.x2_31dof.asset, enable_self_collisions=True),
        init_state=replace(robot.x2_31dof.init_state, pos=[0.0, 0.0, _X2_WBT_INIT_HEIGHT]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.x2_31dof_wbt_observation,
    action=g1_29dof_joint_pos,
    termination=termination.x2_31dof_wbt_termination,
    randomization=randomization.x2_31dof_wbt_randomization,
    command=command.x2_31dof_wbt_command,
    curriculum=curriculum.x2_31dof_wbt_curriculum,
    reward=reward.x2_31dof_wbt_reward,
    nightly=NightlyConfig(
        iterations=8000,
        metrics={},
    ),
)

x2_31dof_wbt_fast_sac = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="x2_31dof_wbt_fast_sac_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.fast_sac,
        config=replace(
            algo.fast_sac.config,
            num_learning_iterations=400000,
            v_max=20.0,
            v_min=-20.0,
            gamma=0.99,
            num_steps=1,
            num_updates=4,
            num_atoms=501,
            policy_frequency=2,
            target_entropy_ratio=0.5,
            tau=0.05,
            use_symmetry=False,
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    ),
    robot=replace(
        robot.x2_31dof,
        control=replace(
            robot.x2_31dof.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=replace(robot.x2_31dof.asset, enable_self_collisions=True),
        init_state=replace(robot.x2_31dof.init_state, pos=[0.0, 0.0, _X2_WBT_INIT_HEIGHT]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.x2_31dof_wbt_observation,
    action=g1_29dof_joint_pos,
    termination=termination.x2_31dof_wbt_termination,
    randomization=randomization.x2_31dof_wbt_randomization,
    command=command.x2_31dof_wbt_command,
    curriculum=curriculum.x2_31dof_wbt_curriculum,
    reward=reward.x2_31dof_wbt_fast_sac_reward,
    nightly=NightlyConfig(
        iterations=200000,
        metrics={},
    ),
)

x2_31dof_wbt_w_object = replace(
    x2_31dof_wbt,
    command=command.x2_31dof_wbt_command_w_object,
    robot=replace(
        robot.x2_31dof_w_object,
        asset=replace(robot.x2_31dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.x2_31dof_w_object.object,
            object_urdf_path="holosoma/data/motions/x2_31dof/whole_body_tracking/objects_largebox.urdf",
        ),
        init_state=replace(robot.x2_31dof_w_object.init_state, pos=[0.0, 0.0, _X2_WBT_INIT_HEIGHT]),
    ),
    randomization=randomization.x2_31dof_wbt_randomization_w_object,
    observation=observation.x2_31dof_wbt_observation_w_object,
    reward=reward.x2_31dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

x2_31dof_wbt_fast_sac_w_object = replace(
    x2_31dof_wbt_fast_sac,
    command=command.x2_31dof_wbt_command_w_object,
    robot=replace(
        robot.x2_31dof_w_object,
        asset=replace(robot.x2_31dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.x2_31dof_w_object.object,
            object_urdf_path="holosoma/data/motions/x2_31dof/whole_body_tracking/objects_largebox.urdf",
        ),
        init_state=replace(robot.x2_31dof_w_object.init_state, pos=[0.0, 0.0, _X2_WBT_INIT_HEIGHT]),
    ),
    randomization=randomization.x2_31dof_wbt_randomization_w_object,
    observation=observation.x2_31dof_wbt_observation_w_object,
    reward=reward.x2_31dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

# Prone slope-crawl whole-body tracking. Uses crawl-specific rewards (no
# planted-feet suite), chest projected-gravity in the actor, tighter
# termination, and a 20 s episode so the full ~19 s reference can finish.
_X2_CRAWL_INIT_HEIGHT = 0.45

x2_31dof_wbt_crawl = replace(
    x2_31dof_wbt,
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="x2_31dof_wbt_crawl",
        num_envs=4096,
    ),
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            # PPO counts these as ADDITIONAL iters from the loaded checkpoint
            # iter (warm-start from model_49999 + 30000 => ~80k).
            num_learning_iterations=30000,
            num_learning_epochs=5,
            save_interval=500,
            entropy_coef=0.005,
            init_noise_std=0.8,
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            init_at_random_ep_len=True,
            empirical_normalization=True,
            use_symmetry=False,
            actor_optimizer=replace(algo.ppo.config.actor_optimizer, weight_decay=0.000),
            critic_optimizer=replace(algo.ppo.config.critic_optimizer, weight_decay=0.000),
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=20.0,
            ),
            scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0),
        ),
    ),
    robot=replace(
        robot.x2_31dof,
        control=replace(
            robot.x2_31dof.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=replace(robot.x2_31dof.asset, enable_self_collisions=True),
        init_state=replace(robot.x2_31dof.init_state, pos=[0.0, 0.0, _X2_CRAWL_INIT_HEIGHT]),
    ),
    observation=observation.x2_31dof_wbt_crawl_observation,
    termination=termination.x2_31dof_wbt_crawl_termination,
    command=command.x2_31dof_wbt_crawl_command,
    reward=reward.x2_31dof_wbt_crawl_reward,
    nightly=NightlyConfig(
        iterations=10000,
        metrics={},
    ),
)

__all__ = [
    "x2_31dof_wbt",
    "x2_31dof_wbt_crawl",
    "x2_31dof_wbt_fast_sac",
    "x2_31dof_wbt_fast_sac_w_object",
    "x2_31dof_wbt_w_object",
]
