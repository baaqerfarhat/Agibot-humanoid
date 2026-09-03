#!/usr/bin/env python3
"""Roll the mjlab squat npz policy on the X2 in the existing Isaac Sim stack.

Boots the Holosoma WBT Isaac env (same setup as box_eval_isaac / hybrid_eval_driver)
only to get a physical X2 + PD plant, then drives the squat npz — not the WBT policy.

    OMNI_KIT_ACCEPT_EULA=1 CUDA_VISIBLE_DEVICES=1 \
      ~/.holosoma_deps/miniconda3/envs/hssim/bin/python \
      ~/baaqer_ws/Agibot-humanoid/squat/run_squat_isaac.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

SQUAT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SQUAT_DIR.parent
sys.path.insert(0, str(SQUAT_DIR))
sys.path.insert(0, str(REPO_ROOT / "adaptation"))

from squat_policy_common import (  # noqa: E402
    CONTROL_DT,
    DEFAULT_POLICY,
    NUM_STEPS,
    NumpyPolicy,
    SquatCommand,
    build_obs,
    name_index,
    quat_rotate_inverse_xyzw,
    rpy_from_xyzw,
    save_rollout,
)

OUT_DIR = SQUAT_DIR / "compare"
OUT_NPZ = OUT_DIR / "isaac_rollout.npz"
OUT_MP4 = OUT_DIR / "isaac.mp4"
VIDEO_RAW_DIR = OUT_DIR / "isaac_video_raw"


def _neutralize_dr(saved_cfg) -> None:
    rand = saved_cfg.randomization
    if rand is None:
        return
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


def _pin_demo_motion(saved_cfg) -> None:
    motion_term = saved_cfg.command.setup_terms["motion_command"]
    motion_config = motion_term.params["motion_config"]
    pins = dict(
        use_adaptive_timesteps_sampler=False,
        start_at_timestep_zero_prob=1.0,
        freeze_at_timestep_zero_prob=1.0,
    )
    if isinstance(motion_config, dict):
        motion_config = dict(motion_config)
        motion_config.update(pins)
        noise = dict(motion_config.get("noise_to_initial_pose") or {})
        noise["overall_noise_scale"] = 0.0
        motion_config["noise_to_initial_pose"] = noise
        motion_term.params["motion_config"] = motion_config
    else:
        motion_term.params["motion_config"] = dataclasses.replace(
            motion_config,
            **pins,
            noise_to_initial_pose=dataclasses.replace(
                motion_config.noise_to_initial_pose, overall_noise_scale=0.0
            ),
        )


def _set_mjlab_joint_props(robot_art, device) -> None:
    mj_armature = torch.full((31,), 0.03, device=device)
    mj_friction = torch.full((31,), 0.3, device=device)
    robot_art.write_joint_armature_to_sim(mj_armature.unsqueeze(0))
    if hasattr(robot_art, "write_joint_friction_coefficient_to_sim"):
        robot_art.write_joint_friction_coefficient_to_sim(mj_friction.unsqueeze(0))
    else:
        robot_art.write_joint_friction_to_sim(mj_friction.unsqueeze(0))
    print("[isaac] joint armature=0.03  friction=0.3 (mjlab x2.xml)")


def _match_mjlab_collisions(sim) -> None:
    """mjlab X2 squat trains with feet-only collisions. Holosoma's Isaac X2
    collides thighs/knees/box with the ground, which the squat policy never saw.

    Collision meshes live on USD instance proxies (URDF payloads). Authoring
    on the proxy or uninstancing after PhysX is cooked breaks the sim. Write
    collisionEnabled=False onto the prototype through the session layer.
    """
    try:
        import omni.usd
        from pxr import Usd, UsdPhysics
    except Exception as exc:
        print(f"[isaac] collision patch skipped: {exc}")
        return
    stage = omni.usd.get_context().get_stage()
    session = stage.GetSessionLayer()
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)

    def _keep_feet(pl: str) -> bool:
        return ("ankle_roll" in pl) or ("foot" in pl)

    def _disable_under(root_path: str, keep) -> int:
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            print(f"[isaac] missing prim {root_path}")
            return 0
        n_ok = n_fail = 0
        samples: list[str] = []
        edit = Usd.EditContext(stage, session) if session is not None else None
        try:
            if edit is not None:
                edit.__enter__()
            for prim in Usd.PrimRange(root, predicate):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                path = str(prim.GetPath())
                pl = path.lower()
                if keep(pl):
                    continue
                author = prim.GetPrimInPrototype() if prim.IsInstanceProxy() else prim
                if not author.IsValid():
                    author = prim
                if len(samples) < 6:
                    samples.append(f"{path} -> {author.GetPath()} proxy={prim.IsInstanceProxy()}")
                try:
                    UsdPhysics.CollisionAPI(author).GetCollisionEnabledAttr().Set(False)
                    n_ok += 1
                except Exception:
                    n_fail += 1
        finally:
            if edit is not None:
                try:
                    edit.__exit__(None, None, None)
                except Exception:
                    pass
        for s in samples:
            print(f"[isaac]   collider {s}")
        print(f"[isaac] {root_path}: disabled={n_ok} failed={n_fail}")
        return n_ok

    try:
        n_robot = _disable_under("/World/envs/env_0/Robot", _keep_feet)
        n_obj = _disable_under("/World/envs/env_0/Object", lambda _pl: False)
        print(f"[isaac] feet-only session-layer patch robot={n_robot} object={n_obj}")
    except Exception as exc:
        print(f"[isaac] collision patch aborted (continuing): {exc}")


def _print_contacts(sim, step: int) -> None:
    cs = getattr(sim, "contact_sensor", None)
    if cs is None or getattr(cs, "data", None) is None:
        return
    try:
        forces = cs.data.net_forces_w[0]
        names = list(cs.body_names)
        mag = torch.linalg.norm(forces, dim=-1).detach().cpu().numpy()
        hits = [(names[i], float(mag[i])) for i in range(len(names)) if mag[i] > 5.0]
        hits.sort(key=lambda kv: -kv[1])
        top = ", ".join(f"{n}={f:.0f}N" for n, f in hits[:8]) or "none>5N"
        print(f"[isaac] contacts@{step}: {top}")
    except Exception as exc:
        print(f"[isaac] contact dump failed: {exc}")


def _hide_object(sim, device) -> None:
    env_ids = torch.tensor([0], device=device)
    st = torch.zeros(1, 13, device=device)
    st[0, 0], st[0, 1], st[0, 2], st[0, 6] = 20.0, 20.0, 1.0, 1.0
    names = []
    try:
        names = [n for n in sim.scene.rigid_objects.keys() if n != "usd_scene_objects"]
    except Exception:
        pass
    if "object" not in names and names:
        pass
    try:
        sim.set_actor_states(["object"], env_ids, st)
        return
    except Exception as exc:
        print(f"[isaac] hide via set_actor_states(object) failed: {exc}")
    if hasattr(sim, "_object") and sim._object is not None:
        try:
            pose = torch.tensor([[20.0, 20.0, 1.0, 1.0, 0.0, 0.0, 0.0]], device=device)
            vel = torch.zeros(1, 6, device=device)
            sim._object.write_root_pose_to_sim(pose, env_ids)
            sim._object.write_root_velocity_to_sim(vel, env_ids)
            if hasattr(sim, "scene"):
                sim.scene.write_data_to_sim()
            return
        except Exception as exc:
            print(f"[isaac] hide _object failed: {exc}")
    for name in names:
        try:
            sim.set_actor_states([name], env_ids, st)
            return
        except Exception as exc:
            print(f"[isaac] hide '{name}' failed: {exc}")


def _write_standing_pose(sim, q_env: torch.Tensor, standing_z: float, device) -> None:
    env_ids = torch.tensor([0], device=device)
    origin = sim.scene.env_origins[0]
    pose = torch.zeros(1, 7, device=device)
    pose[0, 0:3] = origin
    pose[0, 2] = origin[2] + standing_z
    pose[0, 3] = 1.0  # wxyz identity
    vel = torch.zeros(1, 6, device=device)
    sim._robot.write_root_pose_to_sim(pose, env_ids)
    sim._robot.write_root_velocity_to_sim(vel, env_ids)
    zeros = torch.zeros_like(q_env)
    sim._robot.write_joint_state_to_sim(q_env.unsqueeze(0), zeros.unsqueeze(0), sim.dof_ids, env_ids)
    sim.scene.write_data_to_sim()
    if hasattr(sim, "sim"):
        try:
            sim.sim.forward()
        except Exception:
            pass
    sim.refresh_sim_tensors()


def main() -> None:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "1")
    import paths

    paths.enter_holosoma()

    from holosoma.config_types.video import CartesianCameraConfig, VideoConfig
    from holosoma.utils.eval_utils import (
        CheckpointConfig,
        init_eval_logging,
        load_checkpoint,
        load_saved_experiment_config,
    )
    from holosoma.utils.config_utils import CONFIG_NAME
    from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
    from holosoma.utils.helpers import get_class
    from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
    from holosoma.agents.base_algo.base_algo import BaseAlgo

    init_eval_logging()
    policy_path = Path(os.environ.get("SQUAT_POLICY", str(DEFAULT_POLICY)))
    policy = NumpyPolicy(policy_path)
    meta = policy.meta
    squat_cmd = SquatCommand(meta)
    npz_names = list(meta["joint_names"])
    default_npz = np.asarray(meta["default_joint_pos"], np.float32)
    scale_npz = np.asarray(meta["action_scale"], np.float32)
    stiff_npz = np.asarray(meta["joint_stiffness"], np.float32)
    damp_npz = np.asarray(meta["joint_damping"], np.float32)
    standing_z = float(meta["standing_height"])
    print(f"[isaac] policy={policy_path}")

    ckpt = paths.resolve_ckpt()
    checkpoint_cfg = CheckpointConfig(checkpoint=str(ckpt))
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    print(f"[isaac] boot env from {ckpt}")
    print(f"[isaac] simulator -> {saved_cfg.simulator._target_}")

    _pin_demo_motion(saved_cfg)
    saved_cfg.termination.terms.clear()
    _neutralize_dr(saved_cfg)
    for group_name, group in saved_cfg.observation.groups.items():
        if getattr(group, "enable_noise", False):
            object.__setattr__(group, "enable_noise", False)

    eval_cfg = saved_cfg.get_eval_config()
    object.__setattr__(eval_cfg.training, "headless", True)
    object.__setattr__(eval_cfg.training, "num_envs", 1)
    object.__setattr__(eval_cfg.training, "max_eval_steps", NUM_STEPS + 20)
    object.__setattr__(eval_cfg.training, "seed", 42)
    # WBT checkpoints often enable the torso elastic-band gantry. That is not
    # part of squat training; it yanks the robot and must stay off.
    try:
        sim_init = eval_cfg.simulator.config
        object.__setattr__(
            sim_init,
            "virtual_gantry",
            dataclasses.replace(sim_init.virtual_gantry, enabled=False),
        )
        print("[isaac] virtual_gantry.enabled=False")
    except Exception as exc:
        print(f"[isaac] WARN: could not disable gantry in config: {exc}")

    want_video = os.environ.get("SQUAT_ISAAC_VIDEO", "0") == "1"
    VIDEO_RAW_DIR.mkdir(parents=True, exist_ok=True)
    video_cfg = VideoConfig(
        enabled=want_video,
        interval=10_000,
        width=640,
        height=480,
        playback_rate=1.0,
        output_format="h264",
        save_dir=str(VIDEO_RAW_DIR),
        upload_to_wandb=False,
        show_command_overlay=False,
        record_env_id=0,
        camera=CartesianCameraConfig(
            type="cartesian",
            offset=[2.4, 2.0, 1.1],
            target_offset=[0.0, 0.0, 0.35],
            smoothing=0.3,
            tracking_body_name="pelvis",
        ),
        use_recording_thread=False,
    )
    object.__setattr__(
        eval_cfg,
        "logger",
        dataclasses.replace(eval_cfg.logger, headless_recording=want_video, video=video_cfg),
    )
    print(f"[isaac] cameras/video={'on' if want_video else 'off'}")

    env, device, simulation_app = setup_simulation_environment(eval_cfg)
    eval_log_dir = get_experiment_dir(eval_cfg.logger, eval_cfg.training, get_timestamp(), task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg.save_config(str(eval_log_dir / CONFIG_NAME))

    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    algo_class = get_class(eval_cfg.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device, env=env, config=eval_cfg.algo.config, log_dir=str(eval_log_dir), multi_gpu_cfg=None
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_cfg, saved_wandb_path)
    algo.load(str(checkpoint))

    task = algo._unwrap_env()
    sim = task.simulator
    env_names = list(sim.dof_names)
    print(f"[isaac] dof_names ({len(env_names)}): {env_names}")
    env_of_npz = name_index(env_names, npz_names)  # npz[i] = env[env_of_npz[i]]
    npz_of_env = name_index(npz_names, env_names)

    p_gains, d_gains = task.p_gains, task.d_gains
    p_gains[:] = torch.tensor(stiff_npz[npz_of_env], device=device)
    d_gains[:] = torch.tensor(damp_npz[npz_of_env], device=device)
    _set_mjlab_joint_props(sim._robot, device)
    if hasattr(task, "action_manager"):
        for term in getattr(task.action_manager, "_terms", {}).values():
            if hasattr(term, "_kp_scale"):
                term._kp_scale.fill_(1.0)
                term._kd_scale.fill_(1.0)
    task._randomize_ctrl_delay = False
    q_stand_env = torch.tensor(default_npz[npz_of_env], device=device, dtype=torch.float32)

    obs_dict = task.reset_all()
    # WBT reset overwrites default_dof_pos to the motion clip. PD is
    # q_des = action * scale + default_dof_pos, so pin it to the squat pose.
    task.default_dof_pos[:] = q_stand_env
    env_scales = task.action_scales.detach().reshape(-1).cpu().numpy().astype(np.float32)
    env_default = task.default_dof_pos.reshape(-1).detach().cpu().numpy().astype(np.float32)
    print("[isaac] squat vs env default max|d|", float(np.max(np.abs(default_npz[npz_of_env] - env_default))))
    print("[isaac] action_scales", np.round(env_scales, 4).tolist())
    print("[isaac] p_gains", np.round(p_gains.detach().cpu().numpy(), 1).tolist())
    _write_standing_pose(sim, q_stand_env, standing_z, device)
    _hide_object(sim, device)
    _match_mjlab_collisions(sim)
    if hasattr(sim, "_object") and sim._object is not None:
        try:
            view = sim._object.root_physx_view
            n = int(view.count)
            view.set_disable_gravities(
                torch.ones(n, dtype=torch.bool, device="cpu"),
                torch.arange(n, dtype=torch.int32),
            )
            print("[isaac] object gravity disabled")
        except Exception as exc:
            print(f"[isaac] WARN: could not disable object gravity: {exc}")
    if getattr(sim, "virtual_gantry", None) is not None:
        try:
            object.__setattr__(sim.virtual_gantry, "enabled", False)
        except Exception:
            sim.virtual_gantry = None
        print("[isaac] gantry force disabled at runtime")
    task.episode_length_buf[:] = 0
    task.reset_buf[:] = 0
    try:
        cmd = task.command_manager.get_state("motion_command")
        cmd.time_steps[:] = 0
    except Exception:
        cmd = None

    rec = sim.video_recorder
    if rec is not None:
        rec.start_recording(0)
        print(f"[isaac] video recording started -> {VIDEO_RAW_DIR}")
    else:
        print("[isaac] WARN: video_recorder is None")

    logs = {k: [] for k in (
        "t", "obs", "action", "target", "command", "joint_pos", "joint_vel",
        "root_pos", "root_quat_xyzw", "base_ang_vel", "projected_gravity",
        "pelvis_height", "roll", "pitch", "yaw",
    )}
    last_action = np.zeros(len(npz_names), np.float32)
    grav_w = np.array([0.0, 0.0, -1.0], np.float32)

    for step in range(NUM_STEPS):
        _hide_object(sim, device)
        if cmd is not None:
            cmd.time_steps[:] = 0
        task.default_dof_pos[:] = q_stand_env
        root = sim.robot_root_states[0].detach().cpu().numpy().astype(np.float32)
        quat = root[3:7]
        ang_w = root[10:13]
        ang_b = quat_rotate_inverse_xyzw(quat, ang_w)
        grav_b = quat_rotate_inverse_xyzw(quat, grav_w)
        q_env = sim.dof_pos[0].detach().cpu().numpy().astype(np.float32)
        dq_env = sim.dof_vel[0].detach().cpu().numpy().astype(np.float32)
        q_npz = q_env[env_of_npz]
        dq_npz = dq_env[env_of_npz]
        t = step * CONTROL_DT
        command = squat_cmd.command(t)
        obs_np = build_obs(
            meta, ang_b, grav_b, q_npz - default_npz, dq_npz, last_action, command
        )
        action = policy(obs_np)
        target_npz = action * scale_npz + default_npz
        target_env = target_npz[npz_of_env]
        env_action = (target_env - env_default) / np.maximum(env_scales, 1e-8)
        last_action = action.astype(np.float32)

        roll, pitch, yaw = rpy_from_xyzw(quat)
        logs["t"].append(t)
        logs["obs"].append(obs_np)
        logs["action"].append(last_action)
        logs["target"].append(target_npz.astype(np.float32))
        logs["command"].append(command)
        logs["joint_pos"].append(q_npz)
        logs["joint_vel"].append(dq_npz)
        logs["root_pos"].append(root[:3].copy())
        logs["root_quat_xyzw"].append(quat.copy())
        logs["base_ang_vel"].append(ang_b)
        logs["projected_gravity"].append(grav_b)
        logs["pelvis_height"].append(float(root[2]))
        logs["roll"].append(roll)
        logs["pitch"].append(pitch)
        logs["yaw"].append(yaw)

        actor_state = {
            "actions": torch.tensor(env_action, device=device).unsqueeze(0),
            "obs": obs_dict,
            "done_indices": [],
            "stop": False,
        }
        obs_dict, _, reset_buf, _ = task.step(actor_state)
        task.episode_length_buf[:] = min(step + 1, NUM_STEPS + 50)
        task.reset_buf[:] = 0
        _hide_object(sim, device)
        if cmd is not None:
            cmd.time_steps[:] = 0
        if step % 50 == 0 or step in (70, 80, 90, 100):
            print(
                f"[isaac] step {step:3d}/{NUM_STEPS}  z={root[2]:.3f}  "
                f"cmd_h={command[2]:.3f}  reset={int(reset_buf[0]) if reset_buf is not None else 0}"
            )
            _print_contacts(sim, step)
            if hasattr(sim, "_object") and sim._object is not None:
                try:
                    op = sim._object.data.root_pos_w[0].detach().cpu().numpy()
                    print(f"[isaac] object_pos@{step}={op.round(3).tolist()}")
                except Exception:
                    pass

    stacked = {k: np.asarray(v) for k, v in logs.items()}
    stacked["joint_names"] = np.array(npz_names)
    stacked["simulator"] = np.array("isaacsim")
    stacked["policy"] = np.array(str(policy_path))
    stacked["meta_json"] = np.array(json.dumps(meta))
    stacked["dt"] = np.array(CONTROL_DT)
    stacked["wbt_boot_checkpoint"] = np.array(str(ckpt))
    save_rollout(OUT_NPZ, **stacked)

    if rec is not None:
        try:
            rec.stop_recording()
        except Exception as exc:
            print(f"[isaac] stop_recording failed: {exc}")
        mp4s = sorted(VIDEO_RAW_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if mp4s:
            shutil.copy2(mp4s[-1], OUT_MP4)
            print(f"[isaac] wrote {OUT_MP4} from {mp4s[-1]}")
        else:
            print(f"[isaac] WARN: no mp4 in {VIDEO_RAW_DIR}")

    z = stacked["pelvis_height"]
    print(
        f"[isaac] height min={z.min():.3f} final={z[-1]:.3f}  "
        f"max|roll|={np.abs(stacked['roll']).max():.3f}  "
        f"max|pitch|={np.abs(stacked['pitch']).max():.3f}  "
        f"xy_drift={np.linalg.norm(stacked['root_pos'][:, :2] - stacked['root_pos'][0, :2], axis=1).max():.3f}"
    )

    if simulation_app:
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
