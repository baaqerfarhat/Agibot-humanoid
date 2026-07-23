"""Sequenced hybrid rollout: WBT pickup -> arm-locked walking policy -> WBT set-down.

Runs the full hybrid controller in one continuous Isaac Sim episode and records
it to an NPZ (same format as eval_record_driver, renderable by
render_box_rollout.py):

  Phase A (pickup)   WBT policy tracks the in-place clip from t=0 until the
                     motion clock reaches the middle of the HOLD segment
                     (robot standing still, box squeezed at chest).
  Phase B (carry)    Motion clock frozen. The mjlab walking policy drives the
                     robot (its own PD gains + action scales are swapped in),
                     with the 14 arm joints overridden to the exact position
                     targets the WBT policy commanded at the switch (keeps the
                     squeeze). Velocity command ramps 0 -> vx -> 0, then holds
                     zero to settle.
  Phase C (set-down) WBT gains restored, clock resumes from mid-HOLD: the
                     policy plays the remaining hold + set-down + stand-up.

Usage:
    python hybrid_eval_driver.py <wbt_checkpoint.pt> <walking_policy.npz> \
        <output.npz> [walk_seconds] [vx]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from holosoma.config_types.eval_callback import (
    EvalCallbacksConfig,
    RecordingCallbackConfig,
    RecordingConfig,
)
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)

ARM_IDX = list(range(15, 29))    # both arms incl. wrists (dof order: legs, waist, arms, head)
UPPER_IDX = list(range(12, 31))  # waist + arms + head


class WalkingPolicy:
    """Numpy walking policy (mjlab export) evaluated with torch tensors."""

    def __init__(self, npz_path: str, device):
        d = np.load(npz_path, allow_pickle=True)
        self.meta = json.loads(str(d["meta_json"]))
        t = lambda x: torch.tensor(np.asarray(x, np.float32), device=device)
        self.mean, self.std = t(d["mean"]), t(d["std"])
        n = int(d["n_layers"])
        self.W = [t(d[f"W{i}"]) for i in range(n)]
        self.b = [t(d[f"b{i}"]) for i in range(n)]
        self.default = t(self.meta["default_joint_pos"])
        self.action_scale = t(self.meta["action_scale"])
        self.stiffness = t(self.meta["joint_stiffness"])
        self.damping = t(self.meta["joint_damping"])

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.mean) / self.std
        for i in range(len(self.W) - 1):
            x = torch.nn.functional.elu(x @ self.W[i].T + self.b[i])
        return x @ self.W[-1].T + self.b[-1]


def quat_rotate_inverse(q_xyzw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_w = q_xyzw[:, 3:4]
    q_vec = q_xyzw[:, :3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0
    return a - b + c


def main() -> None:
    init_eval_logging()

    wbt_ckpt = sys.argv[1]
    walk_npz = sys.argv[2]
    output_npz = sys.argv[3]
    walk_seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
    vx_cmd = float(sys.argv[5]) if len(sys.argv) > 5 else 0.35
    grav_bias = float(os.environ.get("HYBRID_GRAV_BIAS", "0.08"))

    checkpoint_cfg = CheckpointConfig(checkpoint=wbt_ckpt)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)

    # ---- demo-mode config edits (same as eval_record_driver) ----
    motion_term = saved_cfg.command.setup_terms["motion_command"]
    motion_config = motion_term.params["motion_config"]
    if isinstance(motion_config, dict):
        motion_config = dict(motion_config)
        motion_config["use_adaptive_timesteps_sampler"] = False
        motion_config["start_at_timestep_zero_prob"] = 1.0
        noise = dict(motion_config.get("noise_to_initial_pose") or {})
        noise["overall_noise_scale"] = 0.0
        motion_config["noise_to_initial_pose"] = noise
        motion_term.params["motion_config"] = motion_config
    else:
        import dataclasses

        motion_term.params["motion_config"] = dataclasses.replace(
            motion_config,
            use_adaptive_timesteps_sampler=False,
            start_at_timestep_zero_prob=1.0,
            noise_to_initial_pose=dataclasses.replace(
                motion_config.noise_to_initial_pose, overall_noise_scale=0.0
            ),
        )
    saved_cfg.termination.terms.pop("bad_tracking", None)

    # Neutralize domain randomization: the hybrid demo should run with nominal
    # physics (exact PD gains, nominal friction/COM/masses), like the real robot.
    # Keep the term structure (some setup terms initialize required env state)
    # but disable every term via its params.
    rand = saved_cfg.randomization
    if rand is not None:
        neutral = {
            "enabled": False,
            "kp_range": [1.0, 1.0],
            "kd_range": [1.0, 1.0],
            "rfi_lim_range": [1.0, 1.0],
            "enable_pd_gain": False,
            "enable_rfi_lim": False,
        }
        for group in (rand.setup_terms, rand.reset_terms, rand.step_terms):
            for term in group.values():
                params = dict(term.params or {})
                for k, v in neutral.items():
                    if k in params or k == "enabled":
                        params[k] = v
                object.__setattr__(term, "params", params)

    eval_cfg = saved_cfg.get_eval_config()
    object.__setattr__(eval_cfg.training, "headless", True)
    object.__setattr__(eval_cfg.training, "num_envs", 1)

    # ---- environment + algo (mirrors run_eval_with_tyro) ----
    from holosoma.agents.base_algo.base_algo import BaseAlgo
    from holosoma.utils.config_utils import CONFIG_NAME
    from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
    from holosoma.utils.helpers import get_class
    from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment

    env, device, simulation_app = setup_simulation_environment(eval_cfg)

    eval_log_dir = get_experiment_dir(eval_cfg.logger, eval_cfg.training, get_timestamp(), task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg.save_config(str(eval_log_dir / CONFIG_NAME))

    eval_cbs_cfg = EvalCallbacksConfig(
        recording=RecordingCallbackConfig(
            config=RecordingConfig(enabled=True, output_path=output_npz, env_id=0),
        ),
    )
    cb_configs = eval_cbs_cfg.collect_active_callbacks()
    object.__setattr__(eval_cfg.algo.config, "eval_callbacks", cb_configs)

    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    algo_class = get_class(eval_cfg.algo._target_)
    algo: BaseAlgo = algo_class(device=device, env=env, config=eval_cfg.algo.config,
                                log_dir=str(eval_log_dir), multi_gpu_cfg=None)
    algo.setup()
    algo.attach_checkpoint_metadata(saved_cfg, saved_wandb_path)
    algo.load(str(checkpoint))

    walk = WalkingPolicy(walk_npz, device)

    # ---- sequenced evaluation loop ----
    algo._create_eval_callbacks()
    algo._pre_evaluate_policy()
    eval_policy = algo.get_inference_policy()

    task = algo._unwrap_env()
    cmd = task.command_manager.get_state("motion_command")
    # These tensors are the live references used by the PD torque computation
    # (JointPositionActionTerm exposes them on the env); in-place writes stick.
    p_gains, d_gains = task.p_gains, task.d_gains
    env_scales = task.action_scales
    env_default = task.default_dof_pos.reshape(-1)
    wbt_gains = (p_gains.clone(), d_gains.clone())

    # mjlab (walking-policy training sim) joint properties: armature 0.03,
    # frictionloss 0.3 on every joint. The WBT env uses ~0.01 / 0.0, which makes
    # the mjlab policy tremble; swap these in during the walk phase.
    robot_art = task.simulator._robot
    joint_ids_all = list(range(31))

    def set_joint_props(armature: torch.Tensor, friction: torch.Tensor):
        robot_art.write_joint_armature_to_sim(armature.unsqueeze(0))
        if hasattr(robot_art, "write_joint_friction_coefficient_to_sim"):
            robot_art.write_joint_friction_coefficient_to_sim(friction.unsqueeze(0))
        else:
            robot_art.write_joint_friction_to_sim(friction.unsqueeze(0))

    wbt_armature = robot_art.data.joint_armature[0].clone()
    try:
        wbt_friction = robot_art.data.joint_friction_coeff[0].clone()
    except AttributeError:
        wbt_friction = robot_art.data.joint_friction[0].clone()
    mj_armature = torch.full((31,), 0.03, device=device)
    mj_friction = torch.full((31,), 0.3, device=device)
    mjprops_mode = os.environ.get("HYBRID_MJPROPS", "1")
    if mjprops_mode == "0":
        mj_armature, mj_friction = wbt_armature, wbt_friction
    elif mjprops_mode == "2":  # armature only
        mj_friction = wbt_friction

    n_frames = int(cmd.motion.motion_end_idx[0].item())
    hold_mid = int(sys.argv[6]) if len(sys.argv) > 6 else 211  # mid-HOLD frame (50 Hz)
    walk_steps = int(walk_seconds * 50)

    obs_dict = task.reset_all()
    actor_state = {"done_indices": [], "stop": False, "obs": obs_dict}

    def wbt_action():
        actor_obs = torch.cat([actor_state["obs"][k] for k in algo.actor_obs_keys], dim=1)
        return eval_policy({"actor_obs": actor_obs})

    walk_prev_action = torch.zeros(1, 31, device=device)
    walk_obs_log: list[np.ndarray] = []
    quiet = 0
    arm_hold_target = None
    phase, walk_i = "pickup", 0
    log_every = 50

    step = 0
    while True:
        if phase == "pickup":
            actions = wbt_action()
            frame = int(cmd.time_steps[0].item())
            root_speed = float(task.simulator.robot_root_states[0, 7:10].norm())
            quiet = quiet + 1 if root_speed < 0.15 else 0
            # switch once inside the hold AND quiet for 0.2 s (or at hold end)
            if frame >= hold_mid - 30 and (quiet >= 10 or frame >= hold_mid + 40):
                # walking gains on LEGS only; the whole upper body (waist, arms,
                # head) keeps the WBT gains and stays driven by the live WBT
                # policy so the squeeze + chest-wedge posture is maintained
                mask = torch.ones(31, dtype=torch.bool, device=device)
                mask[UPPER_IDX] = False
                p_gains[mask] = walk.stiffness[mask]
                d_gains[mask] = walk.damping[mask]
                set_joint_props(mj_armature, mj_friction)
                # heading lock: remember the yaw at hand-off; the walk phase
                # closes a wz feedback loop on it to walk a straight line
                qx, qy, qz, qw = task.simulator.robot_root_states[0, 3:7].tolist()
                yaw0 = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2)))
                phase = "walk"
                print(f"[hybrid] step {step}: HOLD reached -> WALK ({walk_seconds:.1f}s @ vx={vx_cmd})")
        elif phase == "walk":
            root = task.simulator.robot_root_states
            quat = root[:, 3:7]
            lin_vel_b = quat_rotate_inverse(quat, root[:, 7:10])
            ang_vel_b = quat_rotate_inverse(quat, root[:, 10:13])
            grav = quat_rotate_inverse(quat, torch.tensor([[0.0, 0.0, -1.0]], device=device))
            # lean-back bias: exaggerate forward pitch in the gravity obs so the
            # policy leans back, cancelling the arms-forward CoM offset of the
            # carry posture (otherwise it pitches forward and runs away)
            grav[0, 0] += grav_bias
            grav = grav / grav.norm(dim=1, keepdim=True)
            qpos = task.simulator.dof_pos - walk.default
            qvel = task.simulator.dof_vel.clone()
            # mask the upper body: the walking policy never saw box-hold poses
            # in training (OOD obs -> falls). Report "upper body at default,
            # still" like an unloaded walk.
            qpos = qpos.clone()
            qpos[:, UPPER_IDX] = 0.0
            qvel[:, UPPER_IDX] = 0.0
            # 1.5 s balance at zero command, 1 s ramp up, walk, 1 s ramp down,
            # then zero command (with light braking) until the robot is still
            tcmd = walk_i / 50.0
            if tcmd < 1.5:
                ramp = 0.0
            elif tcmd < 2.5:
                ramp = tcmd - 1.5
            elif tcmd > walk_seconds - 5.0:
                # very gentle 4 s ramp-down, then hold zero command and let
                # the policy bleed off speed on its own (no reverse commands)
                ramp = max(0.0, (walk_seconds - 1.0 - tcmd) / 4.0)
            else:
                ramp = 1.0
            # heading-hold: steer back to the hand-off yaw (IMU-available on hw)
            qx, qy, qz, qw = root[0, 3:7].tolist()
            yaw = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2)))
            yaw_err = (yaw0 - yaw + np.pi) % (2 * np.pi) - np.pi
            wz = float(np.clip(1.2 * yaw_err, -0.4, 0.4))
            command = torch.tensor([[vx_cmd * ramp, 0.0, wz]], device=device)
            parts = {
                "base_lin_vel": lin_vel_b,  # true speed (policy self-regulates)
                "base_ang_vel": ang_vel_b,
                "projected_gravity": grav,
                "joint_pos": qpos,
                "joint_vel": qvel,
                "actions": walk_prev_action,
                "command": command,
            }
            obs = torch.cat([parts[n] for n in walk.meta["observation_names"]], dim=1)
            walk_obs_log.append(obs[0].detach().cpu().numpy().copy())
            a_walk = walk(obs)
            walk_prev_action = a_walk.clone()
            target = a_walk[0] * walk.action_scale + walk.default
            actions = ((target - env_default) / env_scales).unsqueeze(0)
            # upper body (waist+arms+head): live WBT policy output with the
            # motion clock frozen at the hold frame — actively maintains the
            # squeeze and the torso posture that wedges the box against the chest
            wbt_a = wbt_action()
            actions[0, UPPER_IDX] = wbt_a[0, UPPER_IDX]
            walk_i += 1
            speed_now = float(root[0, 7:10].norm())
            if walk_i >= walk_steps and (speed_now < 0.2 or walk_i >= walk_steps + 100):
                p_gains[:] = wbt_gains[0]
                d_gains[:] = wbt_gains[1]
                set_joint_props(wbt_armature, wbt_friction)
                cmd.time_steps[:] = hold_mid
                phase = "setdown"
                print(f"[hybrid] step {step}: walk done -> SET-DOWN")
        else:
            actions = wbt_action()
            if int(cmd.time_steps[0].item()) >= n_frames - 3:
                break

        actor_state["actions"] = actions
        for c in algo.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        obs_dict, _, _, _ = task.step(actor_state)
        actor_state["obs"] = obs_dict

        # freeze the motion clock + episode timer during the walk
        if phase == "walk":
            cmd.time_steps[:] = hold_mid
            task.episode_length_buf[:] = hold_mid
        task.episode_length_buf.clamp_(max=n_frames - 5)  # never trip the timeout

        for c in algo.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)

        if step % log_every == 0:
            r = task.simulator.robot_root_states[0]
            print(f"[hybrid] step {step:4d} phase={phase:8s} frame={int(cmd.time_steps[0])}/{n_frames} "
                  f"root=({r[0]:.2f},{r[1]:.2f},{r[2]:.2f})")
        step += 1
        if step > 3000:
            print("[hybrid] safety stop")
            break

    algo._post_evaluate_policy()
    if walk_obs_log:
        np.save("/tmp/isaac_walk_obs.npy", np.stack(walk_obs_log))
    print(f"[hybrid] done after {step} steps -> {output_npz}")

    if simulation_app:
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
