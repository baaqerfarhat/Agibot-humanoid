#!/usr/bin/env python3
"""E1 on ALOHA (iclr2027/EXPERIMENT_PLAN.md, the second interface): physical continuations from
full simulator snapshots under the joint-position-target interface. Sources are recorded healthy
command streams (results/aloha/healthy_log.json: per-episode 14-dim absolute joint targets `u` and
measured positions `q`); the scene is reset with a declared seed and the nominal targets are replayed
to each checkpoint (fixed step rule), the state is snapshotted (physics state, time, ctrl/applied
forces/warm start/mocap, task RNG, wrapper counters; aloha_local_probe.Snapshot), and the same next
H targets run in seven branches: healthy, duplicate healthy, faulted/off (+0.02 rad on the six left-arm
joints), legacy from zero, innovation from zero, exact same-mask cancellation, hold from a supplied
estimate. Adaptive branches run aloha_adapt.estimator_step on the deployed position-FIR residual with
the history initialised at the current joint position (the ALOHA convention). Physical quantities: joint
positions/velocities (rad, rad/s) and the task's env_state (cube pose); observation quantities: residual
and estimate; command quantities: nominal, correction, executed, remaining disturbance. Simulator only.
Run in the ALOHA venv with MUJOCO_GL=egl.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, pathlib, sys, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(HERE))
import aloha_adapt as AA
from aloha_local_probe import Snapshot
BRANCHES = ("healthy", "healthy_duplicate", "faulted_off", "legacy_from_zero", "innovation_from_zero", "exact_cancellation", "hold_supplied")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(HERE.parent / "results/aloha/healthy_log.json")); ap.add_argument("--episodes", default="all")
    ap.add_argument("--seed-base", type=int, default=300, help="scene seed for source episode i is seed_base + i (declared; the log records none)")
    ap.add_argument("--openloop", default=str(HERE.parent / "results/aloha/openloop.json")); ap.add_argument("--checkpoints", default="60,120")
    ap.add_argument("--horizon", type=int, default=50); ap.add_argument("--fault", type=float, default=0.02); ap.add_argument("--corr-joints", default="0,1,2,3,4,5")
    ap.add_argument("--gamma", type=float, default=0.08); ap.add_argument("--dead", type=float, default=0.002); ap.add_argument("--norm-r", type=float, default=0.4); ap.add_argument("--clip", type=float, default=0.08)
    ap.add_argument("--hold-estimate", default=None, help="14 comma values; absent = fit/qualification pass without the hold branch")
    ap.add_argument("--split-label", default="unlabelled"); ap.add_argument("--out", type=pathlib.Path, required=True); a = ap.parse_args()
    import gymnasium as gym, gym_aloha  # noqa: F401
    d = json.loads(pathlib.Path(a.log).read_text()); W, r2 = AA.fit_plant(a.log); M = np.array(json.loads(pathlib.Path(a.openloop).read_text())["M"]); M_inv = np.linalg.pinv(M)
    cj = [int(x) for x in a.corr_joints.split(",")]; mask = np.isin(np.arange(AA.NJ), cj).astype(float); f = np.zeros(AA.NJ); f[cj] = a.fault
    hold = np.array([float(x) for x in a.hold_estimate.split(",")]) if a.hold_estimate else None
    cps = sorted(int(x) for x in a.checkpoints.split(",")); H = a.horizon
    env = gym.make(AA.TASK, obs_type="pixels_agent_pos", render_mode="rgb_array"); dm = env.unwrapped._env; physics = dm.physics
    ep_list = list(range(len(d))) if a.episodes == "all" else [int(x) for x in a.episodes.split(",")]
    out_eps, t0 = [], time.time()

    def phys():
        return dict(q=np.asarray(dm.task.get_qpos(physics), float).tolist(), v=np.asarray(dm.task.get_qvel(physics), float).tolist(),
                    env_state=np.asarray(dm.task.get_env_state(physics), float).tolist(), time=float(physics.data.time))

    def step(target):
        obs, r, term, trunc, info = env.step(np.asarray(target, float)); return obs, bool(term or trunc), float(r)

    for ei in ep_list:
        U = np.asarray(d[ei]["u"], float); n = len(U); seed = int(d[ei]["init"]) if "init" in d[ei] else a.seed_base + ei   # the log records its scene seed when it has one
        obs, _ = env.reset(seed=seed); q = np.asarray(obs["agent_pos"], float); k = 0; out_cps = []; step_count = 0
        for ks in cps:
            if ks + H > n:
                out_cps.append(dict(checkpoint=ks, status="missing: fewer than H nominal commands remain")); continue
            assert k <= ks
            for tgt in U[k:ks]:
                obs, _, _ = step(tgt); step_count += 1
            k = ks; assert step_count == ks, f"replay index {step_count} != {ks}"
            q_cp = np.asarray(obs["agent_pos"], float); snap = Snapshot(dm); seg = U[ks:ks + H]

            def run_branch(name):
                snap.restore(); qq = q_cp.copy(); hist = [qq.copy()] * (AA.K_FIR + 1); f_hat = np.zeros(AA.NJ); recs = []
                law = "innov" if name == "innovation_from_zero" else "legacy"; adapt = name in ("legacy_from_zero", "innovation_from_zero")
                for t, tgt in enumerate(seg):
                    nominal = np.asarray(tgt, float).copy()
                    inj = f if name not in ("healthy", "healthy_duplicate") else np.zeros(AA.NJ)
                    if name == "exact_cancellation":
                        c = -f * mask
                    elif name == "hold_supplied":
                        c = -hold * mask
                    elif adapt:
                        c = -f_hat * mask
                    else:
                        c = np.zeros(AA.NJ)
                    a_corr = nominal + c; a_exec = a_corr + inj
                    o, done, rew = step(a_exec); q1 = np.asarray(o["agent_pos"], float)
                    hist.insert(0, a_corr.copy()); hist = hist[:AA.K_FIR + 1]; Hh = np.array(hist)
                    pred = np.array([W[j, :AA.K_FIR + 1] @ Hh[:, j] + W[j, -1] for j in range(AA.NJ)]); res = q1 - pred
                    fb = f_hat.copy(); diag = {}
                    if adapt:
                        f_hat, diag = AA.estimator_step(f_hat, res, M_inv, gamma=a.gamma, dead=a.dead, norm_r=a.norm_r, clip=a.clip, mask=mask, law=law, M=M)
                    est_used = hold if name == "hold_supplied" else (f if name == "exact_cancellation" else f_hat)
                    recs.append(dict(t=t, estimate_applied=np.asarray(est_used, float).tolist(), nominal=nominal.tolist(), correction=c.tolist(), believed=a_corr.tolist(),
                                     executed=a_exec.tolist(), injected=inj.tolist(), remaining_disturbance=(inj + c).tolist(), residual=res.tolist(),
                                     f_hat_before=fb.tolist(), f_hat=f_hat.tolist(), attenuation=diag.get("attenuation"), done=done, reward=rew, **phys()))
                    qq = q1
                return recs

            branches = {}
            for name in BRANCHES:
                if name == "hold_supplied" and hold is None:
                    continue
                branches[name] = run_branch(name)
            dup = max(float(np.linalg.norm(np.array(x["q"]) - np.array(y["q"]))) for x, y in zip(branches["healthy"], branches["healthy_duplicate"]))
            out_cps.append(dict(checkpoint=ks, status="ok", horizon=H, replay_index=step_count, snapshot_identity=snap.identity, duplicate_max_joint_gap=dup,
                                prefix_sha256=hashlib.sha256(np.ascontiguousarray(U[:ks]).tobytes()).hexdigest(), continuation_sha256=hashlib.sha256(np.ascontiguousarray(seg).tobytes()).hexdigest(), branches=branches))
            print(f"episode {ei} (seed {seed}) checkpoint {ks}: {len(branches)} branches, duplicate gap {dup:.2e}", flush=True)
            snap.restore(); obs = dict(agent_pos=q_cp)     # back at the checkpoint; the next checkpoint's prefix replays from here
            k = ks
        out_eps.append(dict(episode=ei, seed=seed, n_commands=n, checkpoints=out_cps))
    meta = dict(log=a.log, log_sha256=hashlib.sha256(pathlib.Path(a.log).read_bytes()).hexdigest(), interface="ALOHA TransferCube, absolute joint targets at 50 Hz", split_label=a.split_label,
                seed_base=a.seed_base, openloop=a.openloop, checkpoint_rule=a.checkpoints, horizon=H, fault=dict(joints=cj, rad=a.fault), law_constants=dict(gamma=a.gamma, dead=a.dead, norm_r=a.norm_r, clip=a.clip),
                hold_estimate=(hold.tolist() if hold is not None else None), plant_r2=r2.tolist(), branches=list(BRANCHES),
                units=dict(q="rad", v="rad/s", env_state="task env_state (cube pose)", commands="rad (absolute joint targets)"),
                quantity_labels=dict(physical=["q", "v", "env_state"], observation=["residual", "f_hat_before", "f_hat", "attenuation"], command=["nominal", "correction", "believed", "executed", "injected", "remaining_disturbance"]),
                seconds=time.time() - t0)
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(dict(meta=meta, episodes=out_eps))); print("wrote", a.out)


if __name__ == "__main__":
    main()
