#!/usr/bin/env python3
"""Crossed command-stream replay (papers/frozen_yet_adaptive_closed_loop/EXPERIMENT_PLAN.md §§2-4;
prereg_records/PREREG_FYA_CROSSED_REPLAY_V1.md). Simulator only; no policy inference.

For each extracted key: scenario reset (libero-reset-v1), ten warm-up steps with the archived warm-up command, the
thirty-step common prefix of raw policy actions (no fault, no correction; identical across the three arms by the
eligibility rule), snapshot of the full simulator state at env step 40, then seven fifty-step continuations from that
snapshot (order of the four matrix cells rotated by key index, recorded):

  ref   healthy_off raw stream, fault off, correction off      -> must reproduce the archived healthy_off path
  J00   M0 (fault_off raw stream), fault on, correction off    -> must reproduce the archived fault_off path
  J11   M1 (fault_nt raw stream), fault on, NT rerun causally  -> must reproduce the archived fault_nt path
  J10   M0, fault on, NT rerun causally
  J01   M1, fault on, correction off
  plus J00_fresh and J11_fresh: reset + warm-up + prefix + continuation without any snapshot/restore.

The NT observer is the deployed one (adaptive_law.estimator_step, law "legacy") with the deployed header W, M, mask
and constants; the FIR history is initialised from the prefix's sent commands exactly as the live runner's deque; the
estimate starts at zero at the first window step (as archived); each transition's correction uses the pre-update
estimate. The logged archived `nominal_command` is never replayed (it already contains the correction): the replayed
input is always the seven-dimensional `raw_action`, including the gripper coordinate.

Per step and branch the log keeps raw_action[7], correction[7] (signed adapter addition, gripper entry 0),
nominal_command[7] = raw + correction, command[7] = nominal + injected fault, clipped_command (|.|<=1 on the arm
coordinates, the inner controller's input range; reported as a flag, no separately measurable effective action),
measured response, predictor residual, norm/attenuation/deadzone/update flags, f_hat_before, f_hat_after, physical
state (joints, velocities, end-effector position and rotation, controller goals, objects, contact), simulator time and
the wrapper's done flag. Continuation after a wrapper done: the terminated-episode guard is cleared and physics
continues; done times are recorded (registered continuation semantics).

Fidelity per key: (i) prefix positions/joints against the archived prefix at every step, (ii) ref/J00/J11 against the
archived healthy_off/fault_off/fault_nt continuation (positions, joints, quaternions; for J11 also corrections and
estimates), (iii) J00_fresh/J11_fresh against J00/J11 (restored-snapshot route versus fresh route). Tolerances are
arguments (registered: 1e-8 rad joints, 1e-8 m positions, 1e-8 rad SO(3) angle, exact raw actions).
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import adaptive_law as AL
from so3 import rot_delta

CELLS = ("J00", "J10", "J01", "J11")


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def quat_angle(q_a, q_b):
    """angle (rad) between two xyzw quaternions, robust near zero (4 asin(|q_a -/+ q_b| / 2); the arccos(dot) form
    cannot resolve below ~1e-5 rad and marked exact replays as failing the 1e-8 tolerance in the first collection)."""
    qa, qb = np.asarray(q_a, float), np.asarray(q_b, float); qa = qa / np.linalg.norm(qa); qb = qb / np.linalg.norm(qb)
    d = min(float(np.linalg.norm(qa - qb)), float(np.linalg.norm(qa + qb)))
    return float(4.0 * np.arcsin(min(1.0, d / 2.0)))


def cell_spec(cell):
    return dict(J00=("M0", False), J10=("M0", True), J01=("M1", False), J11=("M1", True))[cell]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("results/fya_crossed_replay_v1"))
    ap.add_argument("--keys", default="eligible", help="'eligible' (from source_keys.csv), 'all', or comma list task_init_seed")
    ap.add_argument("--suite", default="libero_spatial"); ap.add_argument("--out", type=pathlib.Path, required=True); ap.add_argument("--label", default="unlabelled")
    ap.add_argument("--tol-joint", type=float, default=1e-8); ap.add_argument("--tol-pos", type=float, default=1e-8); ap.add_argument("--tol-angle", type=float, default=1e-8)
    ap.add_argument("--cells", default="all", help="'all' (matrix + ref + fresh repeats) or 'diagonal' (ref, J00, J11, fresh repeats only; pilot)")
    ap.add_argument("--matrix", choices=["nt", "delay", "innovation", "healthy"], default="nt",
                    help="nt: M1 = fault_nt, NT causal, fault on (primary); delay: M1 = delay_nt, NT causal with updates and corrections "
                         "suppressed for the first 10 window steps, fault on; innovation: M1 = fault_innovation, innovation law causal, fault on; "
                         "healthy: M0 = healthy_off, M1 = healthy_nt, NT causal, fault OFF (ref = healthy_off path)")
    a = ap.parse_args()
    MATRIX = dict(nt=dict(m0="fault_off", m1="fault_nt", law="legacy", delay=0, fault=True), delay=dict(m0="fault_off", m1="delay_nt", law="legacy", delay=10, fault=True),
                  innovation=dict(m0="fault_off", m1="fault_innovation", law="innov", delay=0, fault=True), healthy=dict(m0="healthy_off", m1="healthy_nt", law="legacy", delay=0, fault=False))[a.matrix]
    from libero.libero import benchmark
    import main as libero_main
    from libero_reset import reset_libero, physics_fingerprint
    from paired_rollout import contact_flag
    cfg = json.loads((a.root / "configuration.json").read_text())
    W = np.array(cfg["W"], float); M = np.array(cfg["M"], float); M_inv = np.linalg.pinv(M); mask = np.array(cfg["mask"], float); K = AL.K_FIR
    args = cfg["args"]; gamma, dead, norm_r, clip, nch = float(args["gamma"]), float(args["dead"]), float(args["norm_r"]), float(args["clip"]), args["norm_channels"]
    fvec = np.array([float(x) for x in args["fault_vec"].split(",")]); assert args["law"] == "legacy" and args["deadzone_mode"] == "zero" and W.shape == (6, K + 2)
    LAW, DELAY, FAULT_ON = MATRIX["law"], int(MATRIX["delay"]), bool(MATRIX["fault"])
    import csv
    rows = list(csv.DictReader(open(a.root / "source_keys.csv")))
    if a.keys == "eligible":
        sel = [r for r in rows if r["eligible"] == "True" and (a.matrix == "nt" or r.get(f"eligible_{MATRIX['m1']}", r["eligible"]) == "True")]
    elif a.keys == "all":
        sel = rows
    else:
        want = set(a.keys.split(",")); sel = [r for r in rows if f"{r['task']}_{r['init']}_{r['sampler_seed']}" in want]
    suite = benchmark.get_benchmark_dict()[a.suite](); envs = {}; out_keys = []; t0 = time.time()
    for ki, r in enumerate(sel):
        t, i, seed = int(r["task"]), int(r["init"]), int(r["sampler_seed"]); key_id = f"{t}_{i}_{seed}"
        ex = json.loads((a.root / "extracted_streams" / f"{key_id}.json").read_text()); arms = ex["arms"]; P, H = ex["prefix"], ex["window"]
        streams = dict(ref=np.array(arms["healthy_off"]["raw_action"], float), M0=np.array(arms[MATRIX["m0"]]["raw_action"], float), M1=np.array(arms[MATRIX["m1"]]["raw_action"], float))
        n_win = min(H, *(len(s) - P for s in streams.values()))          # complete window only if every stream is long enough
        prefix = streams["M0"][:P]; warm_cmd = arms[MATRIX["m0"]]["warmup_commands"][0]
        if t not in envs:
            task = suite.get_task(t); env, _ = libero_main._get_libero_env(task, libero_main.LIBERO_ENV_RESOLUTION, 7); envs[t] = (env, suite.get_task_init_states(t))
        env, inits = envs[t]; state = {"obs": None}; S = {}

        def bind():
            sim = env.sim; robot = env.robots[0]; core = env.env if hasattr(env, "env") else env
            S.update(sim=sim, robot=robot, core=core, jidx=np.array(robot._ref_joint_pos_indexes, int), vidx=np.array(robot._ref_joint_vel_indexes, int),
                     site=sim.model.site_name2id("gripper0_grip_site"), objs=dict(getattr(core, "obj_body_id", {}) or {}))

        def phys():
            sim, robot = S["sim"], S["robot"]; Rm = np.array(sim.data.site_xmat[S["site"]], float).reshape(3, 3)
            o = state["obs"]
            return dict(q=sim.data.qpos[S["jidx"]].tolist(), v=sim.data.qvel[S["vidx"]].tolist(), ee_pos=sim.data.site_xpos[S["site"]].tolist(), ee_mat=Rm.reshape(-1).tolist(),
                        position=np.array(o["robot0_eef_pos"], float).tolist(), quaternion=np.array(o["robot0_eef_quat"], float).tolist(),
                        joint_position=np.asarray(robot._joint_positions, float).tolist(),
                        goal_pos=np.asarray(robot.controller.goal_pos, float).tolist(), goal_mat=np.asarray(robot.controller.goal_ori, float).reshape(-1).tolist(),
                        objects={k: dict(pos=sim.data.body_xpos[v].tolist(), quat_wxyz=sim.data.body_xquat[v].tolist()) for k, v in S["objs"].items()},
                        contact=bool(contact_flag(sim)), sim_time=float(sim.data.time))

        def step_measure(cmd7):
            o0 = state["obs"]; x0 = np.array(o0["robot0_eef_pos"], float); q0 = np.array(o0["robot0_eef_quat"], float)
            S["core"].done = False
            o1, _, done, _ = env.step(list(cmd7)); state["obs"] = o1
            x1 = np.array(o1["robot0_eef_pos"], float); q1 = np.array(o1["robot0_eef_quat"], float)
            return np.concatenate([x1 - x0, rot_delta(q0, q1)]) / AL.OUT, bool(done)

        def fresh_prefix():
            obs, _ = reset_libero(env, inits[i], suite=a.suite, task=t, init=i); bind()
            for _ in range(AL.WARMUP_STEPS):
                obs, _, _, _ = env.step(list(warm_cmd))
            state["obs"] = obs; recs = []
            for c7 in prefix:
                step_measure(c7); recs.append(phys())
            return recs

        def run_cont(stream, adapt, faulted, hist0):
            hist = list(hist0); f_hat = np.zeros(6); est_state = None; recs = []
            for k in range(n_win):
                raw = np.asarray(stream[P + k], float).copy(); corr = np.zeros(7); enabled = adapt and k >= DELAY
                if enabled:
                    corr[:6] = AL.applied_correction(f_hat, adapt=True, mask=mask)
                nominal = raw + corr; f = fvec if faulted else np.zeros(6)
                cmd = nominal.copy(); cmd[:6] += f; clipped = bool(np.any(np.abs(cmd[:6]) > 1.0))
                wall = time.time(); y, done = step_measure(cmd)
                hist.insert(0, nominal[:6].copy()); hist = hist[:K + 1]; Hh = np.array(hist)
                pred = np.array([W[j, :K + 1] @ Hh[:, j] + W[j, -1] for j in range(6)]); res = y - pred
                fb = f_hat.copy(); diag = dict(nr=None, attenuation=None, deadzone_fired=None, update_applied=False)
                if enabled:
                    f_hat, diag = AL.estimator_step(f_hat, res, M_inv, gamma=gamma, dead=dead, norm_r=norm_r, clip=clip, mask=mask, norm_channels=nch, law=LAW, M=M, state=est_state)
                    est_state = diag.get("estimator_state")
                recs.append(dict(k=k, env_t=AL.WARMUP_STEPS + P + k, wall=wall, raw_action=raw.tolist(), correction=corr.tolist(), nominal_command=nominal.tolist(), command=cmd.tolist(),
                                 clipped_input=clipped, injected=f.tolist(), measured=y.tolist(), residual=res.tolist(), nr=(None if diag.get("nr") is None else float(diag["nr"])),
                                 attenuation=diag.get("attenuation"), deadzone_fired=diag.get("deadzone_fired"), update_applied=bool(diag.get("update_applied", False)),
                                 f_hat_before=fb.tolist(), f_hat_after=f_hat.tolist(), done=bool(done), **phys()))
            return recs

        # ---- prefix, snapshot
        pre_recs = fresh_prefix(); sim, robot, core = S["sim"], S["robot"], S["core"]
        snap = sim.get_state(); extra = dict(warm=sim.data.qacc_warmstart.copy(), act=sim.data.act.copy(), ctrl=sim.data.ctrl.copy(), grip=np.array(robot.gripper.current_action, float).copy(),
                                            obs=state["obs"], timestep=int(getattr(core, "timestep", 0)))
        fp = physics_fingerprint(env)
        hist0 = [np.asarray(c, float)[:6] for c in prefix[max(0, P - K - 1):P][::-1]]; hist0 = (hist0 + [np.zeros(6)] * (K + 1))[:K + 1]

        def restore():
            assert S["sim"] is sim
            sim.set_state(snap); sim.data.qacc_warmstart[:] = extra["warm"]; sim.data.act[:] = extra["act"]; sim.data.ctrl[:] = extra["ctrl"]
            robot.gripper.current_action = extra["grip"].copy()
            if hasattr(core, "timestep"):
                core.timestep = extra["timestep"]
            sim.forward(); state["obs"] = extra["obs"]

        order = list(CELLS[ki % 4:] + CELLS[:ki % 4]) if a.cells == "all" else ["J00", "J11"]
        branches = {}; timing = {}
        restore(); tb = time.time(); branches["ref"] = run_cont(streams["ref"], False, False, hist0); timing["ref"] = time.time() - tb
        for cell in order:
            src, adapt = cell_spec(cell); restore(); tb = time.time(); branches[cell] = run_cont(streams[src], adapt, FAULT_ON, hist0); timing[cell] = time.time() - tb
        for cell in ("J00", "J11"):
            src, adapt = cell_spec(cell); fresh_prefix(); tb = time.time(); branches[cell + "_fresh"] = run_cont(streams[src], adapt, FAULT_ON, hist0); timing[cell + "_fresh"] = time.time() - tb
        # ---- fidelity
        def gaps(recs, arm, offset):
            arch = arms[arm]; n = len(recs)
            gj = max(float(np.max(np.abs(np.array(x["joint_position"]) - np.array(arch["joint_position"][offset + k])))) for k, x in enumerate(recs))
            gp = max(float(np.linalg.norm(np.array(x["position"]) - np.array(arch["position"][offset + k]))) for k, x in enumerate(recs))
            ga = max(quat_angle(x["quaternion"], arch["quaternion"][offset + k]) for k, x in enumerate(recs))
            return dict(joint=gj, position=gp, angle=ga, n=n)
        fid = dict(prefix_vs_archive=gaps(pre_recs, MATRIX["m0"], 0), ref_vs_healthy_off=gaps(branches["ref"], "healthy_off", P), J00_vs_fault_off=gaps(branches["J00"], MATRIX["m0"], P), J11_vs_fault_nt=gaps(branches["J11"], MATRIX["m1"], P))
        fid["J11_corrections_vs_archive"] = max(float(np.max(np.abs(np.array(x["correction"]) - np.array(arms[MATRIX["m1"]]["correction"][P + k])))) for k, x in enumerate(branches["J11"]))
        fid["J11_estimates_vs_archive"] = max(float(np.max(np.abs(np.array(x["f_hat_after"]) - np.array(arms[MATRIX["m1"]]["f_hat_after"][P + k])))) for k, x in enumerate(branches["J11"]))
        fid["diagonal_archives"] = dict(J00=MATRIX["m0"], J11=MATRIX["m1"])
        for cell in ("J00", "J11"):
            fr = branches[cell + "_fresh"]; bb = branches[cell]
            fid[f"{cell}_fresh_vs_restored"] = dict(joint=max(float(np.max(np.abs(np.array(x["joint_position"]) - np.array(y["joint_position"])))) for x, y in zip(fr, bb)),
                                                     position=max(float(np.linalg.norm(np.array(x["position"]) - np.array(y["position"]))) for x, y in zip(fr, bb)))
        checks = dict(prefix=fid["prefix_vs_archive"]["joint"] <= a.tol_joint and fid["prefix_vs_archive"]["position"] <= a.tol_pos,
                      ref=fid["ref_vs_healthy_off"]["joint"] <= a.tol_joint and fid["ref_vs_healthy_off"]["position"] <= a.tol_pos and fid["ref_vs_healthy_off"]["angle"] <= a.tol_angle,
                      J00=fid["J00_vs_fault_off"]["joint"] <= a.tol_joint and fid["J00_vs_fault_off"]["position"] <= a.tol_pos and fid["J00_vs_fault_off"]["angle"] <= a.tol_angle,
                      J11=fid["J11_vs_fault_nt"]["joint"] <= a.tol_joint and fid["J11_vs_fault_nt"]["position"] <= a.tol_pos and fid["J11_vs_fault_nt"]["angle"] <= a.tol_angle and fid["J11_corrections_vs_archive"] <= 1e-8,
                      fresh=all(fid[f"{c}_fresh_vs_restored"]["joint"] <= a.tol_joint and fid[f"{c}_fresh_vs_restored"]["position"] <= a.tol_pos for c in ("J00", "J11")),
                      complete_window=(n_win == H))
        fid["checks"] = checks; fid["valid"] = all(checks.values())
        out_keys.append(dict(key=ex["key"], key_id=key_id, eligible_at_extraction=ex["eligible"], prefix=P, window=H, n_window_steps=n_win, cell_order=order, fingerprint_at_snapshot=fp,
                             stream_sha256={k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in streams.items()}, fidelity=fid, branch_seconds=timing,
                             done_steps={name: next((x["k"] for x in b if x["done"]), None) for name, b in branches.items()}, branches=branches))
        print(f"key {key_id} [{ki + 1}/{len(sel)}] window {n_win}/{H}: prefix gap {fid['prefix_vs_archive']['joint']:.1e} | ref {fid['ref_vs_healthy_off']['joint']:.1e} | J00 {fid['J00_vs_fault_off']['joint']:.1e} | "
              f"J11 {fid['J11_vs_fault_nt']['joint']:.1e} (corr {fid['J11_corrections_vs_archive']:.1e}) | fresh {max(fid['J00_fresh_vs_restored']['joint'], fid['J11_fresh_vs_restored']['joint']):.1e} | valid={fid['valid']}", flush=True)
    meta = dict(driver="fya_crossed_replay.py", driver_sha256=sha(__file__), adaptive_law_sha256=sha(HERE / "adaptive_law.py"), configuration_sha256=sha(a.root / "configuration.json"),
                label=a.label, suite=a.suite, cells=a.cells, matrix=dict(name=a.matrix, **MATRIX), tolerances=dict(joint=a.tol_joint, position=a.tol_pos, angle=a.tol_angle), n_keys=len(out_keys), seconds=time.time() - t0,
                continuation_semantics="after a wrapper done the terminated-episode guard is cleared and physics continues; done steps recorded per branch",
                law=dict(law=LAW, delay=DELAY, fault_on=FAULT_ON, gamma=gamma, dead=dead, norm_r=norm_r, clip=clip, norm_channels=nch, mask=mask.tolist()), fault_vec=fvec.tolist(), W=W.tolist(), M=M.tolist())
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(dict(meta=meta, keys=out_keys))); print("wrote", a.out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
