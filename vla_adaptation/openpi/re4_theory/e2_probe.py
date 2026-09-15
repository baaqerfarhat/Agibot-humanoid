#!/usr/bin/env python3
"""E2 preparation (iclr2027/EXPERIMENT_PLAN.md): local command-response probes with settling and
linearity qualification, Panda/LIBERO, simulator only.

At declared checkpoints of healthy source episodes (fit and qualification partitions kept apart):
snapshot the full state, run the baseline continuation of the next H nominal commands, then for
each of six action axes x two signs x two amplitudes (0.01, 0.02 normalised units) add a CONSTANT
command offset over the H steps and record the per-step end-effector increment difference in the
normalised increment units the FIR predicts (divided by OUT). 25 branches per checkpoint.

Qualification, per axis (declared here): the response is called a DC response only if
 (i) settling: the mean increment difference over the last window (steps H-10..H) differs from the
     mean over the previous window (H-20..H-10) by less than 20 % of the latter;
 (ii) local linearity: the last-window response at 0.02 is within 25 % of twice the response at 0.01,
      and the +/- signs agree within 25 % in magnitude;
otherwise it is reported as a finite-horizon response at H steps and named so. Cross-axis effects,
contact and domain exits (a branch whose joint gap to baseline exceeds 0.5 rad or that touches a
non-arena geom) are reported per branch. The output gives per-axis gains in the two conventions
and is the ONLY source a constrained predictor may take its rotation-y gain from.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(HERE))
import adaptive_law as AL
from so3 import rot_delta
from paired_rollout import contact_flag
AMPS = (0.01, 0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True); ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--episodes", default="all"); ap.add_argument("--checkpoints", default="30,70")
    ap.add_argument("--horizon", type=int, default=40); ap.add_argument("--split-label", default="unlabelled")
    ap.add_argument("--out", type=pathlib.Path, required=True); a = ap.parse_args()
    from libero.libero import benchmark
    import main as libero_main
    from libero_reset import reset_libero, physics_fingerprint
    d = json.loads(pathlib.Path(a.log).read_text()); rec = d["records"][0] if isinstance(d, dict) else d[0]
    cmds_all = np.asarray(rec["raw_cmd"], float); lens = rec["ep_len"]; keys = rec["episode_keys"]; starts = np.cumsum([0] + list(lens[:-1]))
    suite = benchmark.get_benchmark_dict()[a.suite](); H = a.horizon; cps = [int(x) for x in a.checkpoints.split(",")]
    ep_list = list(range(len(lens))) if a.episodes == "all" else [int(x) for x in a.episodes.split(",")]
    envs, out, t0 = {}, [], time.time()
    for ei in ep_list:
        tid, init = int(keys[ei][0]), int(keys[ei][1]); n = int(lens[ei]); cmds = cmds_all[starts[ei]:starts[ei] + n]
        if tid not in envs:
            env, _ = libero_main._get_libero_env(suite.get_task(tid), libero_main.LIBERO_ENV_RESOLUTION, 7); envs[tid] = (env, suite.get_task_init_states(tid))
        env, inits = envs[tid]
        obs, _ = reset_libero(env, inits[init], suite=a.suite, task=tid, init=init)
        for _ in range(AL.WARMUP_STEPS):
            obs, _, _, _ = env.step(libero_main.LIBERO_DUMMY_ACTION)
        sim = env.sim; robot = env.robots[0]; core = env.env if hasattr(env, "env") else env
        jidx = np.array(robot._ref_joint_pos_indexes, int)

        def step(c7):
            core.done = False; o, _, done, _ = env.step(list(c7)); return o

        def run(seg, obs0, dc):
            recs = []; o = obs0
            for c7 in seg:
                c = np.asarray(c7, float).copy(); c[:6] += dc
                x0 = np.array(o["robot0_eef_pos"], float); q0 = np.array(o["robot0_eef_quat"], float)
                o = step(c); x1 = np.array(o["robot0_eef_pos"], float); q1 = np.array(o["robot0_eef_quat"], float)
                recs.append(dict(y=(np.concatenate([x1 - x0, rot_delta(q0, q1)]) / AL.OUT).tolist(), q=sim.data.qpos[jidx].tolist(), contact=bool(contact_flag(sim))))
            return recs

        k = 0; obs_cur = obs
        for ks in cps:
            if ks + H > n:
                out.append(dict(episode=ei, task=tid, init=init, checkpoint=ks, status="missing")); continue
            for c7 in cmds[k:ks]:
                obs_cur = step(c7)
            k = ks; snap = sim.get_state()
            extra = dict(warm=sim.data.qacc_warmstart.copy(), act=sim.data.act.copy(), ctrl=sim.data.ctrl.copy(), grip=np.array(robot.gripper.current_action, float).copy(), obs=obs_cur)
            def restore():
                sim.set_state(snap); sim.data.qacc_warmstart[:] = extra["warm"]; sim.data.act[:] = extra["act"]; sim.data.ctrl[:] = extra["ctrl"]
                robot.gripper.current_action = extra["grip"].copy(); sim.forward()
            seg = cmds[ks:ks + H]; restore(); base = run(seg, extra["obs"], np.zeros(6)); Yb = np.array([s["y"] for s in base]); Qb = np.array([s["q"] for s in base])
            probes = []
            for axis in range(6):
                for sign in (1.0, -1.0):
                    for amp in AMPS:
                        dc = np.zeros(6); dc[axis] = sign * amp; restore(); tr = run(seg, extra["obs"], dc)
                        Y = np.array([s["y"] for s in tr]) - Yb; Q = np.array([s["q"] for s in tr])
                        jgap = np.linalg.norm(Q - Qb, axis=1)
                        w_last = Y[H - 10:H].mean(axis=0); w_prev = Y[H - 20:H - 10].mean(axis=0)
                        probes.append(dict(axis=axis, sign=sign, amp=amp, resp_per_unit_last=(w_last / (sign * amp)).tolist(), resp_per_unit_prev=(w_prev / (sign * amp)).tolist(),
                                           resp_per_unit_by_step=(Y / (sign * amp)).tolist(), max_joint_gap=float(jgap.max()),
                                           domain_exit=bool(jgap.max() > 0.5), contact_steps=int(sum(s["contact"] for s in tr))))
            # per-axis qualification on the own-axis entry
            qual = {}
            for axis in range(6):
                P = [p for p in probes if p["axis"] == axis]
                own = {(p["sign"], p["amp"]): p["resp_per_unit_last"][axis] for p in P}; prev = {(p["sign"], p["amp"]): p["resp_per_unit_prev"][axis] for p in P}
                g = float(np.mean(list(own.values()))); gp = float(np.mean(list(prev.values())))
                settled = abs(g - gp) <= 0.2 * max(abs(gp), 1e-9)
                lin = all(abs(own[(s, 0.02)] - own[(s, 0.01)]) <= 0.25 * max(abs(own[(s, 0.01)]), 1e-9) for s in (1.0, -1.0))
                sym = abs(abs(own[(1.0, 0.02)]) - abs(own[(-1.0, 0.02)])) <= 0.25 * max(abs(own[(1.0, 0.02)]), 1e-9)
                qual[str(axis)] = dict(gain_last_window_per_unit=g, gain_prev_window_per_unit=gp, settled=bool(settled), linear=bool(lin), sign_symmetric=bool(sym),
                                       label=("DC response" if settled and lin and sym else f"finite-horizon response at {H} steps"),
                                       cross_axis_last_window=np.mean([p["resp_per_unit_last"] for p in P], axis=0).tolist(),
                                       domain_exits=int(sum(p["domain_exit"] for p in P)))
            out.append(dict(episode=ei, task=tid, init=init, checkpoint=ks, status="ok", fingerprint=physics_fingerprint(env), baseline_contact_steps=int(sum(s["contact"] for s in base)),
                            probes=probes, qualification=qual))
            print(f"episode {ei} (task {tid}, init {init}) cp {ks}: own-axis gains last window {[round(qual[str(i)]['gain_last_window_per_unit'],3) for i in range(6)]} "
                  f"labels {[qual[str(i)]['label'][:2] for i in range(6)]}", flush=True)
            restore()
            for c7 in seg:
                obs_cur = step(c7)
            k = ks + H
    meta = dict(log=a.log, log_sha256=hashlib.sha256(pathlib.Path(a.log).read_bytes()).hexdigest(), suite=a.suite, split_label=a.split_label, checkpoints=cps, horizon=H, amplitudes=AMPS,
                qualification_rule="settled: |last-prev| <= 20% of prev window mean; linear: 0.02 within 25% of 0.01 per sign; symmetric: |+|,|-| within 25% at 0.02; else finite-horizon response",
                units="normalised increment per unit normalised command (divided by OUT), i.e. the FIR/M units", seconds=time.time() - t0)
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(dict(meta=meta, checkpoints=out))); print("wrote", a.out)


if __name__ == "__main__":
    main()
