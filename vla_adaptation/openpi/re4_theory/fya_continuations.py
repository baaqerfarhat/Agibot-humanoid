#!/usr/bin/env python3
"""FrozenYet Adaptive recovery campaign: fixed-command continuations with imposed delay, correction cap,
sign reversal and a cap-respecting known-fault reference (papers/frozen_yet_adaptive/EXPERIMENT_PLAN.md
sections 3-4; prereg_records/PREREG_FYA_RECOVERY_DEADLINE_V1.md).

Descends from e1_continuations.py v2 (same snapshot/restore, same deployed estimator, same physical and
observation logging). Differences, all registered before the fresh sources were collected:

  * one checkpoint per source (default step 30 of the healthy nominal command stream), horizon 50;
  * a frozen configuration bundle (serialised W, M, constants; --bundle) instead of refitting inside the driver;
  * per scenario {fault vector, correction cap C, imposed adaptation delay tau} and four branches per scenario:
      off          no correction, no estimate-driven action
      nt           the deployed normalised-target observer (adaptive_law law="legacy"), cap C, delay tau
      innovation   the deployed innovation observer (law="innov"), cap C, delay tau
      reference    c_k = 0 for k < tau, afterwards c_k = clip(f, -C, C) on the correction support (privileged, capped)
    plus healthy off / nt / innovation (cap = the healthy cap, no delay) and a duplicate healthy off replay;
  * delay semantics: for transitions k = 0..tau-1 the adaptive estimate is held at zero and its update is
    suppressed; intended and sent command histories advance throughout; the first enabled update happens at
    transition k = tau and supplies the estimate used at k = tau + 1 (one-step observer timing, recorded as such);
  * logging per step: fhat_before (the estimate that generated the command), fhat_after, residual, the law's
    norm/attenuation/deadzone diagnostics, intended (nominal) command, requested correction, sent command
    u_k = a_k - c_k, injected fault, simulator input u_k + f_k, a saturation flag (|input| > 1 on any arm channel:
    the controller clips there, which is physical-model error, not an applied correction), measured response,
    correction mask, remaining disturbance f + c, physical state, wall time;
  * fidelity: duplicate-restore gap AND a fresh-prefix replay (reset, replay the prefix, continue healthy without
    any snapshot) compared to the restored healthy branch, plus the physics fingerprint at the checkpoint from
    both routes; a declared tolerance (--fidelity-tol, rad) marks the source valid/invalid before any scoring.

The branch loop is a pure function of a step callback so that fya_synthetic_check.py can drive it with a
linear fake plant and assert the delay/cap/reference timing without a simulator.

Simulator only; no policy server. Run in the LIBERO client venv with MUJOCO_GL=egl and PYTHONPATH including
openpi/ and the openpi examples/libero + third_party/libero dirs.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import adaptive_law as AL
from so3 import rot_delta

DEFAULT_SCENARIOS = [
    dict(id="reference", fault=[0, 0, 0, 0, 0.05, 0], cap=0.05, delay=0),
    dict(id="delay10", fault=[0, 0, 0, 0, 0.05, 0], cap=0.05, delay=10),
    dict(id="cap_half", fault=[0, 0, 0, 0, 0.05, 0], cap=0.025, delay=0),
    dict(id="sign_reverse", fault=[0, 0, 0, 0, -0.05, 0], cap=0.05, delay=0),
]
ARMS = ("off", "nt", "innovation", "reference")
LAW = {"nt": "legacy", "innovation": "innov"}


def sha256_file(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def branch_specs(scenarios, healthy_cap):
    """Ordered list of (branch name, scenario or None, arm). Healthy branches first, then each scenario."""
    out = [("healthy_off", None, "off"), ("healthy_duplicate", None, "off"),
           ("healthy_nt", None, "nt"), ("healthy_innovation", None, "innovation")]
    for sc in scenarios:
        for arm in ARMS:
            out.append((f"{sc['id']}__{arm}", sc, arm))
    return out


def run_branch(name, scenario, arm, seg, hist0, *, step_fn, phys_fn, W, M, M_inv, mask, K, consts, healthy_cap):
    """seg: the H nominal 7-dim commands. step_fn(cmd7) -> (y6 measured normalised increment, done flag);
    phys_fn() -> dict of physical quantities. Returns per-step records. Pure with respect to the simulator."""
    f = np.asarray(scenario["fault"], float) if scenario is not None else np.zeros(6)
    cap = float(scenario["cap"]) if scenario is not None else float(healthy_cap)
    delay = int(scenario.get("delay", 0)) if scenario is not None else 0
    adapt = arm in ("nt", "innovation"); law = LAW.get(arm)
    hist = list(hist0); f_hat = np.zeros(6); est_state = None; recs = []
    ref_corr = -np.clip(f, -cap, cap) * mask
    for k, c7 in enumerate(seg):
        nominal = np.asarray(c7, float).copy()
        if arm == "reference":
            corr = ref_corr.copy() if k >= delay else np.zeros(6)
        elif adapt:
            corr = -f_hat * mask                       # generated from the pre-update estimate; zero during the delay
        else:
            corr = np.zeros(6)
        sent = nominal.copy(); sent[:6] += corr        # u_k = a_k - c_k, what the adapter sends
        sim_in = sent.copy(); sim_in[:6] += f          # u_k + f_k, what the world executes
        saturated = bool(np.any(np.abs(sim_in[:6]) > 1.0))
        wall = time.time()
        y, done = step_fn(sim_in)
        hist.insert(0, sent[:6].copy()); hist = hist[:K + 1]; Hh = np.array(hist)
        pred = np.array([W[i, :K + 1] @ Hh[:, i] + W[i, -1] for i in range(6)]); r = y - pred
        fb = f_hat.copy(); diag = dict(nr=None, attenuation=None, deadzone_fired=None, update_applied=False)
        enabled = adapt and k >= delay
        if enabled:
            f_hat, diag = AL.estimator_step(f_hat, r, M_inv, gamma=consts["gamma"], dead=consts["dead"], norm_r=consts["norm_r"],
                                            clip=cap, mask=mask, norm_channels=consts["norm_channels"], law=law, M=M, state=est_state)
            est_state = diag.get("estimator_state")
        recs.append(dict(k=k, wall=wall, delay_active=bool(adapt and k < delay), update_enabled=bool(enabled),
                         fhat_before=fb.tolist(), fhat_after=f_hat.tolist(), residual=r.tolist(),
                         nr=(None if diag.get("nr") is None else float(diag["nr"])), attenuation=diag.get("attenuation"),
                         deadzone_fired=diag.get("deadzone_fired"), update_applied=bool(diag.get("update_applied", False)),
                         intended=nominal[:6].tolist(), requested_correction=corr.tolist(), sent=sent[:6].tolist(),
                         injected=f.tolist(), simulator_input=sim_in[:6].tolist(), saturated=saturated,
                         measured=y.tolist(), mask=mask.tolist(), remaining_disturbance=(f + corr).tolist(), done=bool(done),
                         **phys_fn()))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="error_signal.py healthy log (records[0] with raw_cmd, ep_len, episode_keys)")
    ap.add_argument("--bundle", type=pathlib.Path, required=True, help="frozen configuration bundle (fya_bundle.py)")
    ap.add_argument("--bundle-sha256", default=None, help="expected sha256 of the bundle; the run refuses to start on a mismatch")
    ap.add_argument("--suite", default="libero_spatial"); ap.add_argument("--episodes", default="all")
    ap.add_argument("--checkpoint", type=int, default=30); ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--scenarios", type=pathlib.Path, default=None, help="JSON list of {id, fault, cap, delay}; default: the four registered scenarios")
    ap.add_argument("--healthy-cap", type=float, default=0.05)
    ap.add_argument("--fidelity-tol", type=float, default=1e-8, help="max joint gap (rad) allowed for duplicate and fresh-prefix checks")
    ap.add_argument("--no-fresh-prefix", action="store_true", help="skip the fresh-prefix replay check (smoke tests only)")
    ap.add_argument("--out", type=pathlib.Path, required=True); ap.add_argument("--split-label", default="unlabelled")
    a = ap.parse_args()
    from libero.libero import benchmark
    import main as libero_main
    from libero_reset import reset_libero, physics_fingerprint
    from paired_rollout import contact_flag
    bundle_sha = sha256_file(a.bundle)
    if a.bundle_sha256 and a.bundle_sha256 != bundle_sha:
        raise SystemExit(f"bundle sha256 mismatch: {bundle_sha} != {a.bundle_sha256}")
    B = json.loads(a.bundle.read_text())
    W = np.array(B["W"], float); M = np.array(B["M"], float); M_inv = np.linalg.pinv(M); K = int(B["K"])
    cd = [int(x) for x in B["corr_dims"]]; mask = AL.correction_mask(cd)
    consts = dict(gamma=float(B["gamma"]), dead=float(B["dead"]), norm_r=float(B["norm_r"]), norm_channels=B["norm_channels"])
    assert W.shape == (6, K + 2) and M.shape == (6, 6) and np.allclose(np.array(B["OUT"]), AL.OUT) and K == AL.K_FIR
    scenarios = json.loads(a.scenarios.read_text()) if a.scenarios else DEFAULT_SCENARIOS
    specs = branch_specs(scenarios, a.healthy_cap)
    d = json.loads(pathlib.Path(a.log).read_text()); rec = d["records"][0] if isinstance(d, dict) else d[0]
    cmds_all = np.asarray(rec["raw_cmd"], float); lens = rec["ep_len"]; keys = rec["episode_keys"]
    per_ep = rec.get("per_ep") or [{} for _ in lens]
    starts = np.cumsum([0] + list(lens[:-1]))
    ks, H = a.checkpoint, a.horizon
    suite = benchmark.get_benchmark_dict()[a.suite]()
    ep_list = list(range(len(lens))) if a.episodes == "all" else [int(x) for x in a.episodes.split(",")]
    envs, out_eps, t0 = {}, [], time.time()
    for ei in ep_list:
        tid, init = int(keys[ei][0]), int(keys[ei][1]); n = int(lens[ei]); cmds = cmds_all[starts[ei]:starts[ei] + n]
        src = dict(episode=ei, task=tid, init=init, n_commands=n, source_success=per_ep[ei].get("ok"),
                   sampler_seed=per_ep[ei].get("sampler_seed"), checkpoint=ks, horizon=H)
        if ks + H > n:
            src.update(status="excluded: fewer than checkpoint + horizon nominal commands", branches=None)
            out_eps.append(src); print(f"episode {ei} (task {tid}, init {init}): EXCLUDED, {n} < {ks + H} commands", flush=True); continue
        if tid not in envs:
            task = suite.get_task(tid); env, _ = libero_main._get_libero_env(task, libero_main.LIBERO_ENV_RESOLUTION, 7)
            envs[tid] = (env, suite.get_task_init_states(tid))
        env, inits = envs[tid]
        state = {"obs": None}; S = {}

        def bind():
            # a scenario reset may construct a new MjSim: re-acquire every simulator handle after each reset
            sim = env.sim; robot = env.robots[0]; core = env.env if hasattr(env, "env") else env
            S.update(sim=sim, robot=robot, core=core, jidx=np.array(robot._ref_joint_pos_indexes, int), vidx=np.array(robot._ref_joint_vel_indexes, int),
                     site=sim.model.site_name2id("gripper0_grip_site"), objs=dict(getattr(core, "obj_body_id", {}) or {}))

        def fresh_reset():
            obs, _ = reset_libero(env, inits[init], suite=a.suite, task=tid, init=init)
            bind()
            for _ in range(AL.WARMUP_STEPS):
                obs, _, _, _ = env.step(libero_main.LIBERO_DUMMY_ACTION)
            state["obs"] = obs

        def phys():
            sim, robot, site, jidx, vidx, objs = S["sim"], S["robot"], S["site"], S["jidx"], S["vidx"], S["objs"]
            R = np.array(sim.data.site_xmat[site], float).reshape(3, 3)
            return dict(q=sim.data.qpos[jidx].tolist(), v=sim.data.qvel[vidx].tolist(), ee_pos=sim.data.site_xpos[site].tolist(),
                        ee_mat=R.reshape(-1).tolist(), goal_pos=np.asarray(robot.controller.goal_pos, float).tolist(),
                        goal_mat=np.asarray(robot.controller.goal_ori, float).reshape(-1).tolist(),
                        objects={k: dict(pos=sim.data.body_xpos[v].tolist(), quat_wxyz=sim.data.body_xquat[v].tolist()) for k, v in objs.items()},
                        contact=bool(contact_flag(sim)), sim_time=float(sim.data.time))

        def step_measure(cmd7):
            """Step the simulator with cmd7; return the normalised measured increment and the wrapper's done flag."""
            o0 = state["obs"]; x0 = np.array(o0["robot0_eef_pos"], float); q0 = np.array(o0["robot0_eef_quat"], float)
            S["core"].done = False
            o1, _, done, _ = env.step(list(cmd7)); state["obs"] = o1
            x1 = np.array(o1["robot0_eef_pos"], float); q1 = np.array(o1["robot0_eef_quat"], float)
            return np.concatenate([x1 - x0, rot_delta(q0, q1)]) / AL.OUT, bool(done)

        # ---- prefix to the checkpoint, snapshot
        fresh_reset()
        for c7 in cmds[:ks]:
            step_measure(c7)
        sim, robot, core = S["sim"], S["robot"], S["core"]          # the handles the snapshot belongs to
        snap = sim.get_state()
        extra = dict(warm=sim.data.qacc_warmstart.copy(), act=sim.data.act.copy(), ctrl=sim.data.ctrl.copy(),
                     grip=np.array(robot.gripper.current_action, float).copy(), obs=state["obs"], timestep=int(getattr(core, "timestep", 0)))
        fp_snapshot = physics_fingerprint(env)
        hist0 = [np.asarray(c, float)[:6] for c in cmds[max(0, ks - K - 1):ks][::-1]]
        hist0 = (hist0 + [np.zeros(6)] * (K + 1))[:K + 1]
        seg = cmds[ks:ks + H]

        def restore():
            assert S["sim"] is sim, "restore called on a different simulator object than the snapshot's"
            sim.set_state(snap); sim.data.qacc_warmstart[:] = extra["warm"]; sim.data.act[:] = extra["act"]; sim.data.ctrl[:] = extra["ctrl"]
            robot.gripper.current_action = extra["grip"].copy()
            if hasattr(core, "timestep"):
                core.timestep = extra["timestep"]
            sim.forward(); state["obs"] = extra["obs"]

        branches = {}; timing = {}
        for name, sc, arm in specs:
            restore(); tb = time.time()
            branches[name] = run_branch(name, sc, arm, seg, hist0, step_fn=step_measure, phys_fn=phys, W=W, M=M, M_inv=M_inv,
                                        mask=mask, K=K, consts=consts, healthy_cap=a.healthy_cap)
            timing[name] = time.time() - tb
        gap = lambda A, Bb: max(float(np.linalg.norm(np.array(x["q"]) - np.array(y["q"]))) for x, y in zip(A, Bb))
        dup_gap = gap(branches["healthy_off"], branches["healthy_duplicate"])
        fidelity = dict(duplicate_max_joint_gap=dup_gap, tolerance=a.fidelity_tol)
        if not a.no_fresh_prefix:
            # fresh route: reset, replay the same prefix, continue healthy with NO snapshot/restore in between
            fresh_reset()
            for c7 in cmds[:ks]:
                step_measure(c7)
            fp_fresh = physics_fingerprint(env)
            fresh_recs = run_branch("healthy_fresh_prefix", None, "off", seg, hist0, step_fn=step_measure, phys_fn=phys, W=W, M=M,
                                    M_inv=M_inv, mask=mask, K=K, consts=consts, healthy_cap=a.healthy_cap)
            fidelity.update(fresh_prefix_max_joint_gap=gap(branches["healthy_off"], fresh_recs),
                            fresh_prefix_max_ee_gap_m=max(float(np.linalg.norm(np.array(x["ee_pos"]) - np.array(y["ee_pos"]))) for x, y in zip(branches["healthy_off"], fresh_recs)),
                            fingerprint_match=(fp_fresh["fields"] == fp_snapshot["fields"]),
                            fingerprint_mismatched_fields=sorted(k for k in fp_snapshot["fields"] if fp_fresh["fields"].get(k) != fp_snapshot["fields"][k]))
        fidelity["valid"] = bool(dup_gap <= a.fidelity_tol and fidelity.get("fresh_prefix_max_joint_gap", 0.0) <= a.fidelity_tol)
        src.update(status="ok" if fidelity["valid"] else "invalid: fidelity tolerance exceeded", fingerprint=fp_snapshot, fidelity=fidelity,
                   prefix_sha256=hashlib.sha256(np.ascontiguousarray(cmds[:ks]).tobytes()).hexdigest(),
                   continuation_sha256=hashlib.sha256(np.ascontiguousarray(seg).tobytes()).hexdigest(),
                   hist0=[h.tolist() for h in hist0], branch_seconds=timing, branches=branches)
        out_eps.append(src)
        print(f"episode {ei} (task {tid}, init {init}) checkpoint {ks}: {len(branches)} branches, duplicate gap {dup_gap:.2e}, "
              f"fresh-prefix gap {fidelity.get('fresh_prefix_max_joint_gap', float('nan')):.2e}, valid={fidelity['valid']}, "
              f"healthy contact steps {sum(s['contact'] for s in branches['healthy_off'])}/{H}", flush=True)
    meta = dict(driver="fya_continuations.py", driver_sha256=sha256_file(__file__), adaptive_law_sha256=sha256_file(HERE / "adaptive_law.py"),
                log=a.log, log_sha256=sha256_file(a.log), suite=a.suite, split_label=a.split_label,
                bundle=str(a.bundle), bundle_sha256=bundle_sha, checkpoint=ks, horizon=H, scenarios=scenarios, healthy_cap=a.healthy_cap,
                branches=[s[0] for s in specs], arms=list(ARMS), law_map=LAW, constants=consts, corr_dims=cd, fidelity_tol=a.fidelity_tol,
                delay_semantics="updates suppressed for transitions k < tau; estimate held at zero; histories advance; first update at k = tau "
                                "supplies the estimate for k = tau + 1; the reference corrects from k = tau with clip(f, -C, C) on the mask",
                units=dict(q="rad", v="rad/s", ee_pos="m", ee_mat="row-major 3x3", goal="controller OSC goal (m, 3x3)", objects="m, wxyz",
                           commands_residual_estimates="normalised action units (motion divided by OUT)", wall="unix seconds"),
                seconds=time.time() - t0, W=W.tolist(), M=M.tolist())
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(dict(meta=meta, episodes=out_eps)))
    print("wrote", a.out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
