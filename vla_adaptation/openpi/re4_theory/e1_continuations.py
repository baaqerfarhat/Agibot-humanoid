#!/usr/bin/env python3
"""E1 (iclr2027/EXPERIMENT_PLAN.md), v2 (2026-09-15: overlapping checkpoints replayed correctly, replay index asserted,
prefix/continuation hashes and the applied estimate recorded): physical continuations from full-state snapshots, Panda/LIBERO.

For each healthy source episode (recorded nominal commands from an error_signal.py log): reset the
scenario, replay the nominal commands to each checkpoint (fixed step rule, declared before any
outcome is seen), snapshot the full simulator state (MjSimState + solver warm start + actuator
state + ctrl + accumulated gripper target), then run the SAME next H nominal commands in seven
branches: healthy, duplicate healthy (snapshot check), faulted/off, legacy from zero, innovation
from zero, exact same-interface cancellation, and hold from one common supplied estimate. The
fault is a constant +0.05 on the normalised rotation-y command (or --fault-vec). Adaptive
branches run the deployed estimator (adaptive_law.estimator_step) on the deployed per-axis FIR
residual with the predictor history conditioned on the healthy prefix, exactly as the runner does.

Per step and branch it logs physical quantities (joint position/velocity, end-effector position
and orientation matrix, controller goal position/orientation, object body poses, contact flag)
next to observation quantities (predictor residual, estimate before/after, applied correction,
remaining injected disturbance) and the executed command. Physical and observation error are
never mixed: the scorer computes physical error against the healthy branch of the same checkpoint.

Simulator only; no policy server. Run in the LIBERO client venv with MUJOCO_GL=egl and PYTHONPATH
including openpi/ and the openpi examples/libero + third_party/libero dirs.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import adaptive_law as AL
from so3 import rot_delta
from paired_rollout import contact_flag

BRANCHES = ("healthy", "healthy_duplicate", "faulted_off", "legacy_from_zero", "innovation_from_zero", "exact_cancellation", "hold_supplied")


def quat_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], float)


def mat_to_quat_xyzw(R):
    # robust conversion (w, x, y, z) then reorder; matches robosuite's mat2quat convention (xyzw)
    from robosuite.utils.transform_utils import mat2quat
    return np.asarray(mat2quat(np.asarray(R, float)), float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="error_signal.py healthy log (records[0] with raw_cmd, ep_len, episode_keys)")
    ap.add_argument("--suite", default="libero_spatial"); ap.add_argument("--episodes", default="all")
    ap.add_argument("--fit-episodes", default="", help="indices whose FIR fit is used (comma list); default: --plant-log")
    ap.add_argument("--plant-log", default=None, help="log to fit the FIR on (default: the deployed shipped log)")
    ap.add_argument("--openloop", default=str(HERE.parent / "results/phase05/openloop_so3.json"))
    ap.add_argument("--checkpoints", default="30,70", help="fixed step rule: checkpoints at these steps of each source")
    ap.add_argument("--horizon", type=int, default=100); ap.add_argument("--fault-vec", default="0,0,0,0,0.05,0")
    ap.add_argument("--corr-dims", default="3,4,5"); ap.add_argument("--gamma", type=float, default=0.08)
    ap.add_argument("--dead", type=float, default=0.008); ap.add_argument("--norm-r", type=float, default=0.15)
    ap.add_argument("--clip", type=float, default=0.15); ap.add_argument("--norm-channels", default="all")
    ap.add_argument("--hold-estimate", default=None, help="the common supplied estimate for the hold branch (6 comma values); "
                    "if absent the branch is skipped and the run is a fit/qualification pass")
    ap.add_argument("--dc-constrain", choices=["none", "corrected"], default="none")
    ap.add_argument("--out", type=pathlib.Path, required=True); ap.add_argument("--split-label", default="unlabelled")
    a = ap.parse_args()
    from libero.libero import benchmark
    import main as libero_main
    from libero_reset import reset_libero, physics_fingerprint
    d = json.loads(pathlib.Path(a.log).read_text()); rec = d["records"][0] if isinstance(d, dict) else d[0]
    cmds_all = np.asarray(rec["raw_cmd"], float); lens = rec["ep_len"]; keys = rec["episode_keys"]
    starts = np.cumsum([0] + list(lens[:-1]))
    plant_log = a.plant_log or str(HERE.parent / "results/phase05/error_signal_so3.json")
    fit_eps = [int(x) for x in a.fit_episodes.split(",")] if a.fit_episodes else None
    M = np.array(json.loads(pathlib.Path(a.openloop).read_text())["M"]); cd = [int(x) for x in a.corr_dims.split(",")]
    W = AL.fit_plant(plant_log, episodes=fit_eps, dc=({i: float(M[i, i]) for i in cd} if a.dc_constrain == "corrected" else None))
    M_inv = np.linalg.pinv(M); mask = AL.correction_mask(cd); K = AL.K_FIR
    fvec = np.array([float(x) for x in a.fault_vec.split(",")]); hold = np.array([float(x) for x in a.hold_estimate.split(",")]) if a.hold_estimate else None
    cps = [int(x) for x in a.checkpoints.split(",")]; H = a.horizon
    suite = benchmark.get_benchmark_dict()[a.suite]()
    ep_list = list(range(len(lens))) if a.episodes == "all" else [int(x) for x in a.episodes.split(",")]
    envs, out_eps, t0 = {}, [], time.time()
    for ei in ep_list:
        tid, init = int(keys[ei][0]), int(keys[ei][1]); n = int(lens[ei]); cmds = cmds_all[starts[ei]:starts[ei] + n]
        if tid not in envs:
            task = suite.get_task(tid); env, _ = libero_main._get_libero_env(task, libero_main.LIBERO_ENV_RESOLUTION, 7)
            envs[tid] = (env, suite.get_task_init_states(tid))
        env, inits = envs[tid]
        obs, _ = reset_libero(env, inits[init], suite=a.suite, task=tid, init=init)
        for _ in range(AL.WARMUP_STEPS):
            obs, _, _, _ = env.step(libero_main.LIBERO_DUMMY_ACTION)
        sim = env.sim; robot = env.robots[0]; core = env.env if hasattr(env, "env") else env
        jidx = np.array(robot._ref_joint_pos_indexes, int); vidx = np.array(robot._ref_joint_vel_indexes, int)
        site = sim.model.site_name2id("gripper0_grip_site"); objs = dict(getattr(core, "obj_body_id", {}) or {})

        def phys():
            R = np.array(sim.data.site_xmat[site], float).reshape(3, 3)
            return dict(q=sim.data.qpos[jidx].tolist(), v=sim.data.qvel[vidx].tolist(), ee_pos=sim.data.site_xpos[site].tolist(),
                        ee_mat=R.reshape(-1).tolist(), goal_pos=np.asarray(robot.controller.goal_pos, float).tolist(),
                        goal_mat=np.asarray(robot.controller.goal_ori, float).reshape(-1).tolist(),
                        objects={k: dict(pos=sim.data.body_xpos[v].tolist(), quat_wxyz=sim.data.body_xquat[v].tolist()) for k, v in objs.items()},
                        contact=bool(contact_flag(sim)))

        def step(cmd7):
            core.done = False
            o, _, done, _ = env.step(list(cmd7)); return o, bool(done)

        def run_branch(name, seg, hist0, obs0):
            """seg: the H nominal 7-dim commands. Returns per-step records."""
            hist = list(hist0); f_hat = np.zeros(6); est_state = None; recs = []
            law = "innov" if name == "innovation_from_zero" else "legacy"
            adapt = name in ("legacy_from_zero", "innovation_from_zero")
            obs_prev = obs0
            for t, c7 in enumerate(seg):
                nominal = np.asarray(c7, float).copy(); f = fvec if name != "healthy" and name != "healthy_duplicate" else np.zeros(6)
                if name == "exact_cancellation":
                    corr = -fvec * mask
                elif name == "hold_supplied":
                    corr = -hold * mask
                elif adapt:
                    corr = -f_hat * mask
                else:
                    corr = np.zeros(6)
                a_corr = nominal.copy(); a_corr[:6] += corr            # what the adapter believes it sent
                a_exec = a_corr.copy(); a_exec[:6] += f                 # what the world executes
                x0 = np.array(obs_prev["robot0_eef_pos"], float); q0 = np.array(obs_prev["robot0_eef_quat"], float)
                obs_new, done = step(a_exec)
                x1 = np.array(obs_new["robot0_eef_pos"], float); q1 = np.array(obs_new["robot0_eef_quat"], float)
                y = np.concatenate([x1 - x0, rot_delta(q0, q1)]) / AL.OUT
                hist.insert(0, a_corr[:6].copy()); hist = hist[:K + 1]; Hh = np.array(hist)
                pred = np.array([W[i, :K + 1] @ Hh[:, i] + W[i, -1] for i in range(6)]); r = y - pred
                fb = f_hat.copy(); diag = {}
                if adapt:
                    f_hat, diag = AL.estimator_step(f_hat, r, M_inv, gamma=a.gamma, dead=a.dead, norm_r=a.norm_r, clip=a.clip,
                                                    mask=mask, norm_channels=a.norm_channels, law=law, M=M, state=est_state)
                    est_state = diag.get("estimator_state")
                est_used = (hold if name == "hold_supplied" else (fvec if name == "exact_cancellation" else f_hat))
                recs.append(dict(t=t, estimate_applied=np.asarray(est_used, float).tolist(),
                                 nominal=nominal[:6].tolist(), correction=corr.tolist(), believed=a_corr[:6].tolist(), executed=a_exec[:6].tolist(),
                                 injected=f.tolist(), remaining_disturbance=(f + corr).tolist(), measured=y.tolist(), residual=r.tolist(),
                                 f_hat_before=fb.tolist(), f_hat=f_hat.tolist(), attenuation=diag.get("attenuation"), done=done, **phys()))
                obs_prev = obs_new
            return recs

        k = 0; obs_cur = obs; out_cps = []; step_count = 0
        for ci, ks in enumerate(sorted(cps)):
            if ks + H > n:                                            # commands must exist for the whole branch
                out_cps.append(dict(checkpoint=ks, status="missing: fewer than H nominal commands remain")); continue
            assert k <= ks, f"checkpoint {ks} lies before the replay index {k}"
            for c7 in cmds[k:ks]:                                     # healthy prefix from the previous checkpoint to this one
                obs_cur, _ = step(c7); step_count += 1
            k = ks
            assert step_count == ks, f"replay index {step_count} != requested checkpoint {ks}"
            snap = sim.get_state()
            extra = dict(warm=sim.data.qacc_warmstart.copy(), act=sim.data.act.copy(), ctrl=sim.data.ctrl.copy(),
                         grip=np.array(robot.gripper.current_action, float).copy(), obs=obs_cur)
            fp = physics_fingerprint(env)
            hist0 = [np.asarray(c, float)[:6] for c in cmds[max(0, ks - K - 1):ks][::-1]]
            hist0 = (hist0 + [np.zeros(6)] * (K + 1))[:K + 1]
            seg = cmds[ks:ks + H]

            def restore():
                sim.set_state(snap); sim.data.qacc_warmstart[:] = extra["warm"]; sim.data.act[:] = extra["act"]; sim.data.ctrl[:] = extra["ctrl"]
                robot.gripper.current_action = extra["grip"].copy(); sim.forward()

            branches = {}
            for name in BRANCHES:
                if name == "hold_supplied" and hold is None:
                    continue
                restore(); branches[name] = run_branch(name, seg, hist0, extra["obs"])
            dup = max(float(np.linalg.norm(np.array(x["q"]) - np.array(y["q"]))) for x, y in zip(branches["healthy"], branches["healthy_duplicate"]))
            out_cps.append(dict(checkpoint=ks, status="ok", horizon=H, replay_index=step_count, fingerprint=fp, duplicate_max_joint_gap=dup,
                                prefix_sha256=hashlib.sha256(np.ascontiguousarray(cmds[:ks]).tobytes()).hexdigest(),
                                continuation_sha256=hashlib.sha256(np.ascontiguousarray(seg).tobytes()).hexdigest(),
                                hist0=[h.tolist() for h in hist0], branches=branches))
            print(f"episode {ei} (task {tid}, init {init}) checkpoint {ks}: {len(branches)} branches, duplicate gap {dup:.2e}, "
                  f"healthy contact steps {sum(s['contact'] for s in branches['healthy'])}/{H}", flush=True)
            restore(); obs_cur = extra["obs"]                        # back at the checkpoint state; the next checkpoint's prefix
            k = ks                                                    # is replayed from here (overlapping checkpoints are correct)
        out_eps.append(dict(episode=ei, task=tid, init=init, n_commands=n, checkpoints=out_cps))
    meta = dict(log=a.log, log_sha256=hashlib.sha256(pathlib.Path(a.log).read_bytes()).hexdigest(), suite=a.suite, split_label=a.split_label,
                plant_log=plant_log, fit_episodes=fit_eps, dc_constrain=a.dc_constrain, openloop=a.openloop, checkpoint_rule=a.checkpoints, horizon=H,
                fault_vec=fvec.tolist(), corr_dims=cd, law_constants=dict(gamma=a.gamma, dead=a.dead, norm_r=a.norm_r, clip=a.clip, norm_channels=a.norm_channels),
                hold_estimate=(hold.tolist() if hold is not None else None), branches=list(BRANCHES), units=dict(
                    q="rad", v="rad/s", ee_pos="m", ee_mat="row-major 3x3", goal="controller OSC goal (m, 3x3)", objects="m, wxyz",
                    measured_residual_estimates="normalised action units (divided by OUT)"),
                quantity_labels=dict(physical=["q", "v", "ee_pos", "ee_mat", "goal_pos", "goal_mat", "objects", "contact"],
                                     observation=["measured", "residual", "f_hat_before", "f_hat", "attenuation"],
                                     command=["nominal", "correction", "believed", "executed", "injected", "remaining_disturbance"]),
                seconds=time.time() - t0, W=W.tolist(), M=M.tolist())
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(dict(meta=meta, episodes=out_eps)))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
