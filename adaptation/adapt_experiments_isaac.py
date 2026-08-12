#!/usr/bin/env python3
"""Multi-seed ACE adaptation experiments in our Isaac / holosoma box-pickup env.

Tests the two candidate fixes for the input map:
  * gx_level=2 with the TRUE actuated inertia from PhysX (Schur complement of the
    floating-base generalized mass matrix), instead of the diagonal surrogate.
  * error masks that leave the balance-critical stance joints alone.

All variants run in one Isaac session (startup dominates runtime).

  OMNI_KIT_ACCEPT_EULA=1 CUDA_VISIBLE_DEVICES=0 \\
    /home/baaqer/.holosoma_deps/miniconda3/envs/hssim/bin/python \\
    adaptation/adapt_experiments_isaac.py --seeds 5 --steps 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE / "ACC_ADAPTATION_PACKAGE"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(HERE))

import paths  # noqa: E402
from ace_adapt import AdaptConfig, ExportedPolicy, LayerAdapter  # noqa: E402

import eval_adapt_isaac as base  # noqa: E402

N_DOF = 31


class ActionClipHook:
    """Bounds |action| on selected joints, mirroring the deployment's --action-clip.

    action_scale = 0.25*effort_limit/kp, so |a| = 4 is exactly the effort limit and
    anything beyond it asks for torque the actuator cannot make.
    """

    def __init__(self, task, clip: float, subs: tuple[str, ...]):
        import torch

        self.torch = torch
        names = list(task.simulator.dof_names)
        self.clip = float(clip)
        self.mask = np.array([(not subs) or any(s in n for s in subs) for n in names])

    def __call__(self, actions):
        a = actions[0].detach().cpu().numpy().astype(float)
        a[self.mask] = np.clip(a[self.mask], -self.clip, self.clip)
        return self.torch.as_tensor(a, device=actions.device,
                                    dtype=actions.dtype).unsqueeze(0)


class W0LeakAdapter(LayerAdapter):
    """LayerAdapter with sigma-modification around the FROZEN weights.

    The shipped law uses -gamma*W, which decays layer 2 toward zero and accounts
    for ~30% of the weight change. -gamma*(W - W0) keeps it bounded near the
    pretrained policy, which is what adaptation around a nominal controller wants.
    """

    def update(self, joint_error, dt):
        from ace_adapt import _elu_jacobian

        self.step += 1
        if self.diverged or self.step <= self.cfg.engage_step or self._cache is None:
            return
        a, z = self._cache
        L, layer = self.pol.n_layers, self.cfg.layer

        d = self.delta_L(joint_error)
        for l in range(L - 1, layer, -1):
            d = _elu_jacobian(a[l - 1]) * (self.W[l].T @ d)

        Wdot = (self.cfg.gain * np.outer(d, z[layer])
                - self.cfg.leak * (self.W[layer] - self.pol.W0[layer]))
        self.W[layer] = self.W[layer] + dt * Wdot

        if not np.isfinite(self.W[layer]).all() or self.weight_drift > self.cfg.max_weight_drift:
            self.diverged = True
            self.W[layer] = self.pol.W0[layer].copy()


def make_mass_matrix_fn(task, mode: str = "schur"):
    """Return a callable giving the 31x31 actuated inertia in POLICY joint order.

    PhysX reports the generalized mass matrix in its own joint ordering and, for a
    floating base, with the 6 root DoFs first. holosoma already warns that the
    config joint order differs from the IsaacSim order, so both the permutation and
    the root offset have to be undone or the input map is silently scrambled.

    mode="schur" returns M_jj - M_jb M_bb^-1 M_bj, the reduced inertia seen by the
    joints with the base free. mode="block" returns the raw M_jj block.
    """
    robot = task.simulator._robot
    view = robot.root_physx_view
    dof_ids = np.asarray(task.simulator.dof_ids, dtype=int)  # policy idx -> physx idx

    getter = None
    for name in ("get_generalized_mass_matrices", "get_mass_matrices", "get_generalized_mass_matrix"):
        if hasattr(view, name):
            getter = getattr(view, name)
            print(f"[mass] using root_physx_view.{name}()")
            break
    if getter is None:
        print("[mass] no mass-matrix accessor on root_physx_view; gx_level=2 unavailable")
        return None

    try:
        probe = getter()
        M0 = probe[0].detach().cpu().numpy() if hasattr(probe, "detach") else np.asarray(probe[0])
    except Exception as exc:  # PhysX may refuse before the first sim step
        print(f"[mass] accessor raised {exc!r}; gx_level=2 unavailable")
        return None
    n_root = M0.shape[0] - N_DOF
    print(f"[mass] matrix shape {M0.shape} -> root DoFs {n_root}, mode={mode}")
    if n_root < 0:
        print("[mass] unexpected shape; gx_level=2 unavailable")
        return None

    rows = n_root + dof_ids  # physx rows for the policy-ordered joints

    def fn():
        M = getter()
        M = M[0].detach().cpu().numpy().astype(float) if hasattr(M, "detach") else np.asarray(M[0], dtype=float)
        Mjj = M[np.ix_(rows, rows)]
        if mode == "block" or n_root == 0:
            return Mjj
        base_idx = np.arange(n_root)
        Mbb = M[np.ix_(base_idx, base_idx)]
        Mjb = M[np.ix_(rows, base_idx)]
        Mbj = M[np.ix_(base_idx, rows)]
        try:
            return Mjj - Mjb @ np.linalg.solve(Mbb, Mbj)
        except np.linalg.LinAlgError:
            return Mjj

    return fn


# Box-held thresholds. The box rests at z=0.184 and sits ~0.40 m in front of the
# root while carried, so "clear of the ground" and "still in the arms" separate
# cleanly from "dropped and left behind".
BOX_LIFT_Z = 0.45
BOX_HELD_Z = 0.35
BOX_HELD_DIST = 0.75


def box_metrics(records, ctrl_dt):
    """Task success, as opposed to merely staying upright.

    Root height alone scores a run that lifts the box and then drops it as a
    complete success, which is exactly what the first pass did.
    """
    if not records.get("object_pos"):
        return {"lifted": False, "carry_s": 0.0, "max_box_z": float("nan"),
                "final_dist": float("nan"), "placed": False, "success": False}
    box = np.asarray(records["object_pos"], dtype=float)
    root = np.asarray(records["root_pos"], dtype=float)
    dist = np.linalg.norm(box[:, :2] - root[:, :2], axis=1)

    above = np.flatnonzero(box[:, 2] > BOX_LIFT_Z)
    if above.size == 0:
        return {"lifted": False, "carry_s": 0.0, "max_box_z": float(box[:, 2].max()),
                "final_dist": float(dist[-1]), "placed": False, "success": False}

    lift = int(above[0])
    held = (box[:, 2] > BOX_HELD_Z) & (dist < BOX_HELD_DIST)
    carry = 0
    for i in range(lift, len(held)):
        if not held[i]:
            break
        carry += 1

    # The reference ends by SETTING THE BOX DOWN (~step 420 of 734), so "still holding
    # at the last step" scores a correct placement as a failure. A set-down leaves the
    # box low and still next to the robot; a drop leaves it far away (>1.2 m observed).
    placed = bool(box[-1, 2] < BOX_HELD_Z and dist[-1] < BOX_HELD_DIST)
    return {
        "lifted": True,
        "lift_step": lift,
        "carry_s": carry * ctrl_dt,
        "max_box_z": float(box[:, 2].max()),
        "final_dist": float(dist[-1]),
        "placed": placed,
        "success": bool(held[-1] or placed),
    }


def make_fault_fn(task, spec: str):
    """Emulate degraded actuators by scaling their PD stiffness.

    holosoma computes torque in python as `kp_scale * p_gains * (target - q) - ...`, so
    scaling `kp_scale` for a joint is a weak actuator: it still tracks, just with less
    authority, and it sags under load. The policy and the adapter both keep using the
    NOMINAL Kp -- neither is told the fault happened, which is the point.

    Spec is `substring:scale[,substring:scale...]`, e.g. "knee:0.3".
    """
    if not spec:
        return None, {}

    from holosoma.managers.randomization.terms.locomotion import _get_joint_action_term

    term = _get_joint_action_term(task)
    if term is None:
        raise SystemExit("no joint action term found; cannot inject an actuator fault")

    names = list(task.simulator.dof_names)
    hits: dict[str, float] = {}
    for part in spec.split(","):
        key, _, val = part.partition(":")
        key, scale = key.strip(), float(val)
        matched = [i for i, n in enumerate(names) if key in n]
        if not matched:
            raise SystemExit(f"fault '{key}' matched no joint in {names}")
        for i in matched:
            hits[names[i]] = scale

    idx = [names.index(n) for n in hits]
    scales = [hits[n] for n in hits]
    print(f"[fault] scaling PD stiffness: " + ", ".join(f"{n}x{s}" for n, s in hits.items()))

    def fn():
        for i, s in zip(idx, scales):
            term._kp_scale[:, i] = s

    return fn, hits


def check_export_match(algo, task, pol, steps: int) -> None:
    """Is the numpy export the same function as the torch checkpoint?

    Adapted runs use the numpy `ExportedPolicy`; the `frozen` baseline uses torch. If
    the two are not the same function, "frozen vs adapted" partly measures the export.
    The torch policy drives the sim here, so both see bit-identical observations and
    the difference reported is the forward pass alone, with no closed-loop amplification.
    """
    import torch

    eval_policy = algo.get_inference_policy()
    actor_state = {"done_indices": [], "stop": False, "obs": task.reset_all()}

    diffs = []
    for step in range(steps):
        actor_obs = torch.cat([actor_state["obs"][k] for k in algo.actor_obs_keys], dim=1)
        a_torch = eval_policy({"actor_obs": actor_obs})
        a_np = pol.forward(actor_obs[0].detach().cpu().numpy().astype(np.float64))[0]
        diffs.append(np.abs(a_torch[0].detach().cpu().numpy().astype(np.float64) - a_np).max())

        actor_state["actions"] = a_torch
        actor_state["step"] = step
        actor_state = algo.env_step(actor_state)

    diffs = np.asarray(diffs)
    tol = 1e-4
    print("\n=== numpy export vs torch checkpoint, identical observations ===")
    print(f"  steps compared     : {len(diffs)}")
    print(f"  step 0 diff        : {diffs[0]:.3e}")
    print(f"  max |action diff|  : {diffs.max():.3e}")
    print(f"  mean |action diff| : {diffs.mean():.3e}")
    print(f"  -> {'MATCH' if diffs.max() < tol else 'MISMATCH'} "
          f"(tol {tol:g}; float32 inference alone should sit near 1e-6)")


LEGS_WAIST = ("hip", "knee", "ankle", "waist")

# `cls` selects the adaptation law:
#   None            -> no adapter at all, holosoma's torch policy (the deployed baseline)
#   LayerAdapter    -> the shipped law, leak = -gamma*W  (decays toward zero)
#   W0LeakAdapter   -> leak = -gamma*(W - W0)            (decays toward the trained weights)
#
# `frozen_npz` is the control that makes the comparison honest: same numpy forward pass
# as every adapted run, gain 0, so the only thing separating it from them is adaptation
# rather than torch-vs-numpy inference.
VARIANTS = [
    # name,                  gain,   gx,  error_joints, cls
    ("frozen",               None,   None, None,        None),
    ("frozen_npz",           0.0,    1,    LEGS_WAIST,  W0LeakAdapter),
    ("his_g3e-4_gx1",        3e-4,   1,    LEGS_WAIST,  LayerAdapter),
    ("his_g1e-5_gx1",        1e-5,   1,    LEGS_WAIST,  LayerAdapter),
    ("w0_g3e-4_gx1",         3e-4,   1,    LEGS_WAIST,  W0LeakAdapter),
    ("w0_g1e-5_gx1",         1e-5,   1,    LEGS_WAIST,  W0LeakAdapter),
    ("w0_g3e-4_gx2_schur",   3e-4,   2,    LEGS_WAIST,  W0LeakAdapter),
    ("w0_g1e-5_gx2_schur",   1e-5,   2,    LEGS_WAIST,  W0LeakAdapter),
    ("w0_g3e-4_waistonly",   3e-4,   1,    ("waist",),  W0LeakAdapter),
    ("w0_g3e-4_noankle",     3e-4,   1,    ("hip", "knee", "waist"), W0LeakAdapter),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=base.DEFAULT_CKPT)
    ap.add_argument("--policy-npz", default=base.DEFAULT_POLICY_NPZ)
    ap.add_argument("--motion", default=base.DEFAULT_MOTION)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=600)
    ap.add_argument("--out-dir", default=str(HERE / "isaac_runs"))
    ap.add_argument("--only", default="", help="comma-separated variant names to run")
    ap.add_argument("--fault", default="", metavar="SUBSTR:SCALE",
                    help="degrade actuators by scaling their PD stiffness, e.g. "
                         "'knee:0.3' or 'right_knee:0.2,right_hip_pitch:0.5'")
    ap.add_argument("--check-export", type=int, default=0, metavar="N",
                    help="compare the numpy export against the torch policy for N steps "
                         "on identical observations, then exit")
    ap.add_argument("--record-seed", type=int, default=None,
                    help="save a renderable NPZ for this seed of every variant")
    ap.add_argument("--record-dir", default=str(HERE / "isaac_runs"))
    ap.add_argument("--action-clip", type=float, default=0.0,
                    help="Bound |action| on --action-clip-joints to this (4 = the effort "
                         "limit). Matches deploy_x2_box_pickup.py --action-clip. 0 = off.")
    ap.add_argument("--action-clip-joints", default="ankle_roll,wrist")
    ap.add_argument("--obs-noise", default="off", choices=["off", "on"],
                    help="training observation noise. Leaving it ON was why the frozen "
                         "baseline dropped the box on 3/5 seeds in the first pass.")
    ap.add_argument("--dr", default="no-push", choices=["no-push", "none", "all"],
                    help="no-push: drop the push randomizer only (default); "
                         "none: also drop PD-gain/latency/CoM randomization; "
                         "all: leave training randomization untouched")
    args = ap.parse_args()

    # Resolve before the chdir below, or relative output paths land inside holosoma.
    args.out_dir = str(Path(args.out_dir).resolve())
    args.record_dir = str(Path(args.record_dir).resolve())

    paths.enter_holosoma()

    import dataclasses  # noqa: E402

    from holosoma.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from holosoma.utils.common import seeding  # noqa: E402
    from holosoma.utils.config_utils import CONFIG_NAME  # noqa: E402
    from holosoma.utils.eval_utils import (  # noqa: E402
        CheckpointConfig,
        init_eval_logging,
        load_checkpoint,
        load_saved_experiment_config,
    )
    from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp  # noqa: E402
    from holosoma.utils.helpers import get_class  # noqa: E402
    from holosoma.utils.sim_utils import (  # noqa: E402
        close_simulation_app,
        setup_simulation_environment,
    )

    init_eval_logging()

    checkpoint_cfg = CheckpointConfig(checkpoint=args.ckpt)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)

    motion_term = saved_cfg.command.setup_terms["motion_command"]
    mc = motion_term.params["motion_config"]
    if isinstance(mc, dict):
        mc = dict(mc)
        mc["use_adaptive_timesteps_sampler"] = False
        mc["start_at_timestep_zero_prob"] = 1.0
        mc["freeze_at_timestep_zero_prob"] = 0.0
        noise = dict(mc.get("noise_to_initial_pose") or {})
        noise["overall_noise_scale"] = 0.0
        mc["noise_to_initial_pose"] = noise
        mc["motion_file"] = args.motion
        mc["motion_dir"] = ""
        motion_term.params["motion_config"] = mc
    else:
        motion_term.params["motion_config"] = dataclasses.replace(
            mc,
            use_adaptive_timesteps_sampler=False,
            start_at_timestep_zero_prob=1.0,
            freeze_at_timestep_zero_prob=0.0,
            noise_to_initial_pose=dataclasses.replace(mc.noise_to_initial_pose, overall_noise_scale=0.0),
            motion_file=args.motion,
            motion_dir="",
        )
    saved_cfg.termination.terms.pop("bad_tracking", None)

    if args.obs_noise == "off":
        for group_name, group in saved_cfg.observation.groups.items():
            if getattr(group, "enable_noise", False):
                object.__setattr__(group, "enable_noise", False)
                print(f"[obs] disabled observation noise on '{group_name}'")

    # get_eval_config() does NOT disable randomization, so the push randomizer keeps
    # shoving the robot (up to 0.7 m/s, every 1-2.5 s) through every rollout. That is a
    # training-robustness term and it makes the frozen baseline drop the box.
    drop = []
    if args.dr != "all":
        drop.append("push")
    if args.dr == "none":
        # "randomize_action_delay" and not "delay": dropping the delay SETUP term leaves
        # env._randomize_ctrl_delay undefined and joint_control.reset() then throws.
        drop += ["actuator", "randomize_action_delay", "com", "bias", "friction", "mass"]
    if drop:
        for bucket in ("setup_terms", "reset_terms", "step_terms"):
            terms = getattr(saved_cfg.randomization, bucket, None)
            if not terms:
                continue
            for key in [k for k in terms if any(d in k.lower() for d in drop)]:
                terms.pop(key)
                print(f"[dr] disabled {bucket}.{key}")

    eval_cfg = saved_cfg.get_eval_config()
    object.__setattr__(eval_cfg.training, "headless", True)
    object.__setattr__(eval_cfg.training, "num_envs", 1)
    object.__setattr__(eval_cfg.training, "max_eval_steps", args.steps)

    env, device, simulation_app = setup_simulation_environment(eval_cfg)
    eval_log_dir = get_experiment_dir(eval_cfg.logger, eval_cfg.training, get_timestamp(), task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg.save_config(str(eval_log_dir / CONFIG_NAME))
    object.__setattr__(eval_cfg.algo.config, "eval_callbacks", {})

    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    algo_class = get_class(eval_cfg.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device, env=env, config=eval_cfg.algo.config, log_dir=str(eval_log_dir), multi_gpu_cfg=None
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_cfg, saved_wandb_path)
    algo.load(str(checkpoint))
    algo._create_eval_callbacks()
    algo._pre_evaluate_policy()

    task = algo._unwrap_env()
    ctrl_dt = float(task.dt)
    pol = ExportedPolicy(args.policy_npz)
    ref_pos = pol.ref_pos

    task.reset_all()  # PhysX buffers must be valid before probing the mass matrix
    mass_fn = make_mass_matrix_fn(task, mode="schur")
    fault_fn, fault_hits = make_fault_fn(task, args.fault)

    if args.check_export:
        check_export_match(algo, task, pol, args.check_export)
        if simulation_app:
            close_simulation_app(simulation_app)
        return

    wanted = [v.strip() for v in args.only.split(",") if v.strip()]
    variants = [v for v in VARIANTS if not wanted or v[0] in wanted]
    seeds = list(range(args.seed0, args.seed0 + args.seeds))

    results: dict[str, list[dict]] = {}
    for name, gain, gx, mask, cls in variants:
        if gx == 2 and mass_fn is None:
            print(f"\n[skip] {name}: no mass matrix available")
            continue
        print(f"\n=== {name}  ({args.seeds} seeds) ===")
        rows = []
        for s in seeds:
            seeding(s, torch_deterministic=False)
            if cls is None:
                adapter = None
            else:
                cfg = AdaptConfig(layer=2, gain=gain, leak=1e-2, gx_level=gx,
                                  error_joints=mask, engage_step=0)
                adapter = cls(
                    pol, cfg, joint_names=pol.meta["joint_names"],
                    mass_matrix_fn=(mass_fn if gx == 2 else None),
                )
            # The adapter has no authority over a joint whose action is already past
            # the effort limit: moving that action from 25 to 20 changes the delivered
            # torque by zero, so the error it is regulating cannot respond and the
            # update just spends the norm budget. Clipping first is what gives the
            # adaptation a working control channel on the ankles and wrists.
            clip_hook = (ActionClipHook(task, args.action_clip,
                                        tuple(s for s in args.action_clip_joints.split(",") if s))
                         if args.action_clip > 0 else None)
            r = base._rollout(algo, task, adapter, args.steps, ref_pos, ctrl_dt,
                              on_reset=fault_fn, action_hook=clip_hook)
            r.update(box_metrics(r["records"], ctrl_dt))
            rows.append(r)
            if args.record_seed is not None and s == args.record_seed:
                base._save_npz(
                    Path(args.record_dir) / f"isaac_{name}_seed{s}.npz", r,
                    {"mode": name, "seed": s, "ckpt": args.ckpt, "gain": gain,
                     "gx_level": gx, "error_joints": list(mask) if mask else None,
                     "law": cls.__name__ if cls else "torch"},
                )
            print(f"  seed {s}: survival {r['survival']:4d} ({r['survival']*ctrl_dt:5.2f}s)  "
                  f"legErr {r['leg_err']:6.2f} deg  carry {r['carry_s']:5.2f}s  "
                  f"boxZmax {r['max_box_z']:.2f}  endDist {r['final_dist']:.2f}  "
                  f"{'PLACED' if r['placed'] else ('HELD' if r['success'] else 'DROPPED')}"
                  f"{'  DIVERGED' if r['diverged'] else ''}")
        results[name] = [
            {k: r[k] for k in ("survival", "tracked", "leg_err", "drift", "diverged",
                               "lifted", "carry_s", "max_box_z", "final_dist",
                               "placed", "success")} for r in rows
        ]
        surv = np.array([r["survival"] for r in rows], float) * ctrl_dt
        err = np.array([r["leg_err"] for r in rows], float)
        carry = np.array([r["carry_s"] for r in rows], float)
        full = int(sum(1 for r in rows if r["survival"] >= args.steps))
        ok = int(sum(1 for r in rows if r["success"]))
        print(f"  -> survival {surv.mean():.2f} +/- {surv.std():.2f} s   "
              f"legErr {err.mean():.2f} +/- {err.std():.2f} deg   "
              f"carry {carry.mean():.2f} +/- {carry.std():.2f} s   "
              f"never fell {full}/{len(rows)}   box held {ok}/{len(rows)}")

    print(f"\n=== SUMMARY (mean +/- std over seeds, dr={args.dr}) ===")
    print(f"  {'variant':22s} {'survival(s)':>16s} {'legErr(deg)':>16s} "
          f"{'carry(s)':>16s} {'nofall':>8s} {'boxok':>8s}")
    for name, rows in results.items():
        surv = np.array([r["survival"] for r in rows], float) * ctrl_dt
        err = np.array([r["leg_err"] for r in rows], float)
        carry = np.array([r["carry_s"] for r in rows], float)
        full = int(sum(1 for r in rows if r["survival"] >= args.steps))
        ok = int(sum(1 for r in rows if r["success"]))
        print(f"  {name:22s} {surv.mean():7.2f} +/-{surv.std():5.2f} "
              f"{err.mean():9.2f} +/-{err.std():5.2f} "
              f"{carry.mean():9.2f} +/-{carry.std():5.2f} "
              f"{full:5d}/{len(rows)} {ok:5d}/{len(rows)}")

    out = Path(args.out_dir) / "adapt_experiments_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"seeds": seeds, "steps": args.steps, "ctrl_dt": ctrl_dt, "dr": args.dr,
         "obs_noise": args.obs_noise, "ckpt": args.ckpt, "motion": args.motion,
         "fault": fault_hits or None,
         "config": {n: {"gain": g, "gx_level": gx, "error_joints": list(m) if m else None,
                        "law": c.__name__ if c else "torch"}
                    for n, g, gx, m, c in variants},
         "results": results}, indent=2))
    print(f"\nwrote {out}")

    if simulation_app:
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
