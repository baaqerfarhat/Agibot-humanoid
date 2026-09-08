"""The same law on a humanoid: Fourier GR1 (arms + waist + dexterous hands) in the RoboCasa
GR1 tabletop simulator, driven by GR00T N1.5 zero-shot over groot15_server.py.

Joint space, exactly as aloha_adapt.py -- that is what this robot's action interface is:
  action   29 absolute joint targets: left arm 7, right arm 7, left hand 6, right hand 6,
           waist 3 (control_delta = False in the benchmark's own wrapper), 16-step chunks
  state    the same 29 joint positions, measured
  plant    per-joint FIR on POSITION, identified on healthy rollouts (mode log)
  fault    additive offset on chosen joints' commanded targets (encoder / calibration
           offset), or a gain (loss of effectiveness)
  M        d(position)/d(fault) per joint by open-loop replay (mode openloop)
  law      f_hat <- f_hat + gamma (M^-1 r - f_hat), deadzone / normaliser / clip -- unchanged

Runs in the robocasa-gr1 venv (python 3.10; robosuite 1.5.1, mujoco 3.2.6). Needs
MUJOCO_GL=egl. The policy server runs in the Isaac-GR00T-n15 venv on another port.
"""
from __future__ import annotations

import argparse, atexit, collections, json, pathlib, sys, os
import numpy as np
import gymnasium as gym
import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401  (registers gr1_unified/* envs)

OPENPI_CLIENT = pathlib.Path(os.environ.get(
    "OPENPI_CLIENT", "/home/mtaheri/ws_AgibotX2/openpi/packages/openpi-client/src"))
sys.path.insert(0, str(OPENPI_CLIENT))
from openpi_client import websocket_client_policy as _wc  # noqa: E402

PARTS = [("left_arm", 7), ("right_arm", 7), ("left_hand", 6), ("right_hand", 6), ("waist", 3)]
NJ = 29
K_FIR = 6
VIDEO_KEY = "video.ego_view_bg_crop_pad_res256_freq20"
LANG_KEY = "annotation.human.coarse_action"
DEFAULT_TASK = "gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
ARM = {"left": list(range(0, 7)), "right": list(range(7, 14))}


def clip_report(vals, clip, name):
    v = np.abs(np.asarray(vals, float))
    if v.size == 0:
        return
    rail = (np.abs(v - clip) < 1e-6).mean(axis=0)
    if np.any(rail > 0.2):
        hit = ", ".join(f"j{i} {100*rail[i]:.0f}%" for i in range(len(rail)) if rail[i] > 0.2)
        print(f"  !! {name}: at the clip (+-{clip}) in >20% of episodes on [{hit}] -- SATURATED, not estimated.")


class GR1:
    def __init__(self, task, host, port, seed=0, render=True):
        self.env = gym.make(task, enable_render=render)
        self.client = _wc.WebsocketClientPolicy(host, port)
        self.seed = seed
        atexit.register(self.env.close)

    def reset(self, ep):
        obs, _ = self.env.reset(seed=self.seed + ep)
        return obs

    @staticmethod
    def q(obs):
        return np.concatenate([np.asarray(obs[f"state.{n}"], float).reshape(-1) for n, _ in PARTS])

    @staticmethod
    def policy_obs(obs):
        return {"gr1/video": np.asarray(obs[VIDEO_KEY], np.uint8),
                "gr1/state": GR1.q(obs).astype(np.float32), "prompt": str(obs[LANG_KEY])}

    def step(self, u):
        act, i = {}, 0
        for n, d in PARTS:
            act[f"action.{n}"] = np.asarray(u[i:i + d], float); i += d
        obs, r, term, trunc, info = self.env.step(act)
        return obs, r, term, trunc, info


def fit_plant(log_path):
    d = json.loads(pathlib.Path(log_path).read_text())
    W = np.zeros((NJ, K_FIR + 2)); r2 = np.zeros(NJ)
    for j in range(NJ):
        X, Y = [], []
        for ep in d:
            u = np.array(ep["u"])[:, j]; qq = np.array(ep["q"])[:, j]
            for t in range(K_FIR, len(u)):
                X.append(np.r_[u[t - K_FIR:t + 1][::-1], 1.0]); Y.append(qq[t])
        X, Y = np.array(X), np.array(Y)
        w, *_ = np.linalg.lstsq(X, Y, rcond=None); W[j] = w
        ss = ((Y - X @ w) ** 2).sum(); st = ((Y - Y.mean()) ** 2).sum()
        r2[j] = 1 - ss / max(st, 1e-12)
    return W, r2


def episode(A, ep, W=None, M_inv=None, fvec=None, gain=None, adapt=False, gamma=0.08, dead=0.002,
            norm_r=0.05, clip=0.3, corr=None, profile="step", prof_p=60.0, onset=0, log=None,
            static_corr=None, f_init=None, freeze_after=None, horizon=16, max_steps=720,
            stop_on_success=True, law="legacy", M=None):
    obs = A.reset(ep); q = A.q(obs)
    hist = collections.deque([q.copy()] * (K_FIR + 1), maxlen=K_FIR + 1)   # seeded at q0 (Sec 27.7)
    f_hat = np.zeros(NJ) if f_init is None else np.asarray(f_init, float).copy()
    plan = collections.deque(); traj = []; success = False
    fvec = np.zeros(NJ) if fvec is None else np.asarray(fvec, float)
    m = np.zeros(NJ) if corr is None else np.isin(np.arange(NJ), corr).astype(float)
    for t in range(max_steps):
        if not plan:
            chunk = np.asarray(A.client.infer(A.policy_obs(obs))["actions"], float)
            plan.extend(chunk[:horizon])
        a_cmd = np.asarray(plan.popleft(), float)
        c = (-np.asarray(static_corr, float)) if static_corr is not None else ((-f_hat * m) if adapt else np.zeros(NJ))
        a_corr = a_cmd + c
        live = t >= onset; u = t - onset
        scale = {"step": 1.0, "ramp": min(1.0, u / max(prof_p, 1e-9)),
                 "sine_bias": 0.5 * (1 + np.sin(2 * np.pi * u / max(prof_p, 1e-9))),
                 "intermittent": 1.0 if int(u // max(prof_p, 1)) % 2 == 0 else 0.0}[profile] if live else 0.0
        f_now = fvec * scale
        a_exec = a_corr + f_now
        if gain is not None and live:
            a_exec = q + gain * (a_exec - q)
        obs, r, term, trunc, info = A.step(a_exec)
        q1 = A.q(obs); q = q1
        hist.appendleft(a_corr.copy())
        if log is not None:
            log["u"].append(a_corr.tolist()); log["q"].append(q1.tolist())
        if W is not None:
            H = np.array(hist)
            pred = np.array([W[j, :K_FIR + 1] @ H[:, j] + W[j, -1] for j in range(NJ)])
            res = q1 - pred
            if adapt and (freeze_after is None or t < freeze_after):
                # The normaliser and deadzone act on the residual of the joints being
                # corrected. On the GR1 the 12 hand joints have R^2 ~ 0 (they are commanded
                # open/closed and modelled as nothing) and their residual, median 0.17 and 90th
                # pct 0.61, is three times the arm's; a norm over all 29 joints attenuated every
                # arm update by 4-10x and the estimate settled at 13% of the fault (run 1).
                sel = (m > 0) if corr is not None else np.ones(NJ, bool)
                if law == "innov":
                    # innovation form: e = r - M f_hat; unbiased fixed point (adaptive_law --law innov)
                    Mm = M if M is not None else np.linalg.pinv(M_inv)
                    e = res - Mm @ (f_hat * m if corr is not None else f_hat)
                    ne = float(np.linalg.norm(e[sel]))
                    step = np.zeros(NJ) if ne < dead else (M_inv @ e) / (1.0 + (ne / norm_r) ** 2)
                    f_hat = np.clip(f_hat + gamma * step, -clip, clip)
                else:
                    est = M_inv @ res
                    nr = float(np.linalg.norm(res[sel]))
                    if nr < dead: est = np.zeros(NJ)
                    est = est / (1.0 + (nr / norm_r) ** 2)
                    f_hat = np.clip(f_hat + gamma * (est - f_hat), -clip, clip)
            traj.append(dict(t=t, f_hat=f_hat.tolist(), f_true=f_now.tolist()))
        success = success or bool(info.get("success", False))
        if term or trunc or (success and stop_on_success):
            break
    return success, f_hat, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["log", "openloop", "run"])
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8004)
    ap.add_argument("--episodes", type=int, default=10); ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=16, help="steps executed per 16-step chunk")
    ap.add_argument("--max-steps", type=int, default=720)
    ap.add_argument("--log", type=pathlib.Path); ap.add_argument("--openloop", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--fault-vec", default=None, help="29 comma-separated joint offsets (rad), or 'arm:right:0.05'")
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--corr-joints", default=None, help="joints to correct, e.g. 7,8,9,10,11,12,13 or 'arm:right'")
    ap.add_argument("--profile", default="step", choices=["step", "ramp", "sine_bias", "intermittent"])
    ap.add_argument("--prof-p", type=float, default=60.0); ap.add_argument("--onset", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.08); ap.add_argument("--dead", type=float, default=0.002)
    ap.add_argument("--norm-r", type=float, default=0.05); ap.add_argument("--clip", type=float, default=0.3)
    ap.add_argument("--probe", type=float, default=0.02); ap.add_argument("--probe-joints", default=None)
    ap.add_argument("--static-corr", default=None)
    ap.add_argument("--identify-episodes", type=int, default=None)
    ap.add_argument("--freeze-after", type=int, default=None)
    ap.add_argument("--warm-start", action="store_true")
    ap.add_argument("--law", choices=["legacy", "innov"], default="legacy")
    ap.add_argument("--with-healthy", action="store_true",
                    help="also run a HEALTHY arm (no fault, no correction) on the same seeds in the same process. "
                         "The simulator randomises objects and their placement at every reset (Sec 32.10), so "
                         "the arms are UNPAIRED samples of the scene distribution; test them with Fisher exact, "
                         "and measure the ceiling with the same number of episodes.")
    ap.add_argument("--hold-stat", choices=["last", "mean50"], default="last",
                    help="what identify-then-hold carries: the final estimate, or the mean of the last 50 steps "
                         "(the record's statistic; the final value on a contact-rich episode is one contact spike away)")
    a = ap.parse_args()

    def joints(spec):
        if spec is None: return None
        if spec.startswith("arm:"): return ARM[spec.split(":")[1]]
        return [int(x) for x in spec.split(",")]

    def fault(spec):
        if spec is None: return None
        if spec.startswith("arm:"):
            _, side, mag = spec.split(":"); f = np.zeros(NJ); f[ARM[side]] = float(mag); return f.tolist()
        return [float(x) for x in spec.split(",")]

    A = GR1(a.task, a.host, a.port, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == "log":
        eps = []
        for ep in range(a.episodes):
            L = dict(u=[], q=[]); s, _, _ = episode(A, ep, fvec=fault(a.fault_vec), log=L, horizon=a.horizon,
                                                   max_steps=a.max_steps)
            L["success"] = s; eps.append(L)
            print(f"  ep {ep}: success={s} steps={len(L['u'])}", flush=True)
            a.out.write_text(json.dumps(eps))
        if not a.fault_vec:
            W, r2 = fit_plant(a.out); print("FIR R2 per joint:", np.round(r2, 3))
        print(f"successes {sum(e['success'] for e in eps)}/{len(eps)}")
        return
    if a.mode == "openloop":
        d = json.loads(a.log.read_text()); cmds = np.array(d[0]["u"])[:120]
        pj = joints(a.probe_joints) or list(range(NJ))
        def replay(f):
            obs = A.reset(0); D = []
            for u_t in cmds:
                obs, *_ = A.step(np.asarray(u_t) + f); D.append(A.q(obs))
            return np.array(D)[len(cmds) // 2:]
        base = replay(np.zeros(NJ)); M = np.eye(NJ)
        for j in pj:
            acc = []
            for sgn in (1.0, -1.0):
                f = np.zeros(NJ); f[j] = sgn * a.probe
                acc.append((replay(f) - base).mean(0) / (sgn * a.probe))
            M[:, j] = np.mean(acc, axis=0)
            print(f"  probe j{j}: diag {M[j, j]:.3f}", flush=True)
        print("M diagonal:", np.round(np.diag(M), 3)); print("cond(M):", round(float(np.linalg.cond(M)), 1))
        a.out.write_text(json.dumps({"M": M.tolist(), "probe": a.probe, "probe_joints": pj})); return
    W, r2 = fit_plant(a.log); M = np.array(json.loads(a.openloop.read_text())["M"]); M_inv = np.linalg.pinv(M)
    fvec = fault(a.fault_vec); corr = joints(a.corr_joints)
    sc = [float(x) for x in a.static_corr.split(",")] if a.static_corr else None
    res = dict(args={k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(a).items()}, arms={})
    arms = [("frozen_faulted", False), ("adaptive", True)]
    if a.with_healthy:
        arms = [("healthy", None)] + arms
    for tag, adapt in arms:
        ok, fh, per_ep, trajs = 0, [], [], []
        f_carry = None
        for ep in range(a.episodes):
            if adapt is None:                      # healthy control: no fault, no correction, same seeds
                s, f_hat, traj = episode(A, ep, W, M_inv, None, None, False, horizon=a.horizon, max_steps=a.max_steps)
                ok += int(s); fh.append(f_hat.tolist()); per_ep.append(dict(task=0, init=ep, ok=bool(s))); trajs.append(traj)
                print(f"  [{tag}] ep {ep}: success={s}", flush=True)
                res["arms"][tag] = dict(successes=ok, n=ep + 1, f_hat=fh, per_ep=per_ep, traj=[], f_true=[])
                a.out.write_text(json.dumps(res)); continue
            s, f_hat, traj = episode(A, ep, W, M_inv, fvec, a.gain, adapt, a.gamma, a.dead, a.norm_r, a.clip,
                                     corr, a.profile, a.prof_p, a.onset, static_corr=(sc if adapt else None),
                                     f_init=(f_carry if (adapt and (a.warm_start or a.identify_episodes is not None)) else None),
                                     freeze_after=(0 if (a.identify_episodes is not None and ep >= a.identify_episodes) else a.freeze_after),
                                     horizon=a.horizon, max_steps=a.max_steps, law=a.law, M=M)
            f_carry = f_hat if a.hold_stat == "last" or not traj else np.mean([st["f_hat"] for st in traj[-50:]], axis=0)
            ok += int(s); fh.append(f_hat.tolist()); per_ep.append(dict(task=0, init=ep, ok=bool(s))); trajs.append(traj)
            shown = corr if corr else list(range(7))
            print(f"  [{tag}] ep {ep}: success={s}  f_hat[{shown[0]}..]={np.round(f_hat[shown], 3)}", flush=True)
            res["arms"][tag] = dict(successes=ok, n=ep + 1, f_hat=fh, per_ep=per_ep,
                                    traj=[[st["f_hat"] for st in tr] for tr in trajs],
                                    f_true=[[st["f_true"] for st in tr] for tr in trajs])
            a.out.write_text(json.dumps(res))
        clip_report(fh, a.clip, tag)
        print(f"{tag}: {ok}/{a.episodes} = {100*ok/a.episodes:.0f}%\n", flush=True)


if __name__ == "__main__":
    main()
