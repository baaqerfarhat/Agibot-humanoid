"""The same law on a fourth robot in a second simulator: WidowX in SimplerEnv (SAPIEN),
driven by GR00T N1.7's Bridge finetune over groot_widowx_server.py.

The action interface is Cartesian, like LIBERO's: 6 end-effector increments (x, y, z, roll,
pitch, yaw) plus a gripper, executed by the benchmark's own controller. So the plant is the
LIBERO one -- a per-channel FIR on the measured POSE INCREMENT driven by the commanded
increments -- re-identified on this robot's healthy rollouts; M by open-loop replay of a
recorded command sequence with a per-axis probe; the law of adaptive_law.py, unchanged.
The fault is additive on the commanded increments (a tool-frame / calibration offset).
Runs in the SimplerEnv venv; the policy server runs in the Isaac-GR00T venv.
"""
from __future__ import annotations

import argparse, atexit, collections, json, pathlib, sys, os
import numpy as np
import gymnasium as gym
from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs  # noqa: E402

OPENPI_CLIENT = pathlib.Path(os.environ.get(
    "OPENPI_CLIENT", "/home/mtaheri/ws_AgibotX2/openpi/packages/openpi-client/src"))
sys.path.insert(0, str(OPENPI_CLIENT))
from openpi_client import websocket_client_policy as _wc  # noqa: E402

register_simpler_envs()
POSE = ["x", "y", "z", "roll", "pitch", "yaw"]
K_FIR = 6
DEFAULT_TASK = "simpler_env_widowx/widowx_spoon_on_towel"
LANG_KEY = "annotation.human.action.task_description"


def clip_report(vals, clip, name):
    v = np.abs(np.asarray(vals, float))
    if v.size == 0:
        return
    rail = (np.abs(v - clip) < 1e-6).mean(axis=0)
    if np.any(rail > 0.2):
        hit = ", ".join(f"{POSE[i]} {100*rail[i]:.0f}%" for i in range(6) if rail[i] > 0.2)
        print(f"  !! {name}: at the clip (+-{clip}) in >20% of episodes on [{hit}] -- SATURATED, not estimated.")


class WidowX:
    def __init__(self, task, host, port, seed=0):
        self.env = gym.make(task)
        self.client = _wc.WebsocketClientPolicy(host, port)
        self.seed = seed
        atexit.register(self.env.close)

    def reset(self, ep):
        obs, _ = self.env.reset(seed=self.seed + ep)
        return obs

    @staticmethod
    def pose(obs):
        return np.array([float(np.asarray(obs[f"state.{k}"]).reshape(-1)[0]) for k in POSE])

    @staticmethod
    def policy_obs(obs):
        st = np.array([float(np.asarray(obs[f"state.{k}"]).reshape(-1)[0]) for k in POSE + ["pad", "gripper"]], np.float32)
        return {"wx/image": np.asarray(obs["video.image_0"], np.uint8), "wx/state": st, "prompt": str(obs[LANG_KEY])}

    def step(self, u7):
        act = {f"action.{k}": np.asarray([u7[i]], float) for i, k in enumerate(POSE)}
        act["action.gripper"] = np.asarray([u7[6]], float)
        obs, r, term, trunc, info = self.env.step(act)
        return obs, r, term, trunc, info


def wrap_angle(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def increment(p1, p0):
    d = p1 - p0; d[3:] = wrap_angle(d[3:]); return d


def fit_plant(log_path, scale):
    """Per-channel FIR: increment_t / scale = sum_k h_k u_{t-k} + c, on healthy (u, y)."""
    d = json.loads(pathlib.Path(log_path).read_text())
    W = np.zeros((6, K_FIR + 2)); r2 = np.zeros(6)
    for j in range(6):
        X, Y = [], []
        for ep in d:
            u = np.array(ep["u"])[:, j]; y = np.array(ep["y"])[:, j] / scale[j]
            for t in range(K_FIR, len(u)):
                X.append(np.r_[u[t - K_FIR:t + 1][::-1], 1.0]); Y.append(y[t])
        X, Y = np.array(X), np.array(Y)
        w, *_ = np.linalg.lstsq(X, Y, rcond=None); W[j] = w
        ss = ((Y - X @ w) ** 2).sum(); st = ((Y - Y.mean()) ** 2).sum(); r2[j] = 1 - ss / max(st, 1e-12)
    return W, r2


def episode(A, ep, W=None, M_inv=None, M=None, scale=None, fvec=None, adapt=False, gamma=0.08, dead=0.01,
            norm_r=0.15, clip=0.3, corr=None, log=None, f_init=None, freeze_after=None, law="innov",
            horizon=8, max_steps=300, profile="step", prof_p=60.0, onset=0):
    obs = A.reset(ep); p = A.pose(obs)
    hist = collections.deque([np.zeros(6)] * (K_FIR + 1), maxlen=K_FIR + 1)
    f_hat = np.zeros(6) if f_init is None else np.asarray(f_init, float).copy()
    plan = collections.deque(); traj = []; success = False
    fvec = np.zeros(6) if fvec is None else np.asarray(fvec, float)
    m = np.ones(6) if corr is None else np.isin(np.arange(6), corr).astype(float); sel = m > 0
    for t in range(max_steps):
        if not plan:
            plan.extend(np.asarray(A.client.infer(A.policy_obs(obs))["actions"], float)[:horizon])
        a_cmd = np.asarray(plan.popleft(), float)                     # (7,)
        c = (-f_hat * m) if adapt else np.zeros(6)
        a_corr = a_cmd.copy(); a_corr[:6] += c
        live = t >= onset; u = t - onset
        sc = {"step": 1.0, "ramp": min(1.0, u / max(prof_p, 1e-9)),
              "sine_bias": 0.5 * (1 + np.sin(2 * np.pi * u / max(prof_p, 1e-9))),
              "intermittent": 1.0 if int(u // max(prof_p, 1)) % 2 == 0 else 0.0}[profile] if live else 0.0
        f_now = fvec * sc
        a_exec = a_corr.copy(); a_exec[:6] += f_now
        obs, r, term, trunc, info = A.step(a_exec)
        p1 = A.pose(obs); y = increment(p1, p); p = p1
        hist.appendleft(a_corr[:6].copy())
        if log is not None:
            log["u"].append(a_corr[:6].tolist()); log["y"].append(y.tolist())
        if W is not None:
            H = np.array(hist); yn = y / scale
            pred = np.array([W[j, :K_FIR + 1] @ H[:, j] + W[j, -1] for j in range(6)])
            res = yn - pred
            if adapt and (freeze_after is None or t < freeze_after):
                if law == "innov":
                    e = res - M @ (f_hat * m); ne = float(np.linalg.norm(e[sel]))
                    step = np.zeros(6) if ne < dead else (M_inv @ e) / (1.0 + (ne / norm_r) ** 2)
                    f_hat = np.clip(f_hat + gamma * step * m, -clip, clip)
                else:
                    est = M_inv @ res; nr = float(np.linalg.norm(res[sel]))
                    if nr < dead: est = np.zeros(6)
                    est = est / (1.0 + (nr / norm_r) ** 2)
                    f_hat = np.clip(f_hat + gamma * (est - f_hat) * m, -clip, clip)
            traj.append(dict(t=t, f_hat=f_hat.tolist(), f_true=f_now.tolist()))
        success = success or bool(info.get("success", False))
        if term or trunc or success:
            break
    return success, f_hat, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["log", "openloop", "run"])
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8005)
    ap.add_argument("--episodes", type=int, default=10); ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=8); ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--log", type=pathlib.Path); ap.add_argument("--openloop", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--fault-vec", default=None, help="6 comma-separated offsets in action units")
    ap.add_argument("--corr-dims", default=None, help="e.g. 0,1,2 or 3,4,5")
    ap.add_argument("--profile", default="step", choices=["step", "ramp", "sine_bias", "intermittent"])
    ap.add_argument("--prof-p", type=float, default=60.0); ap.add_argument("--onset", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.08); ap.add_argument("--dead", type=float, default=0.01)
    ap.add_argument("--norm-r", type=float, default=0.15); ap.add_argument("--clip", type=float, default=0.3)
    ap.add_argument("--law", choices=["legacy", "innov"], default="innov")
    ap.add_argument("--probe", type=float, default=0.02); ap.add_argument("--probe-steps", type=int, default=120)
    ap.add_argument("--with-healthy", action="store_true")
    ap.add_argument("--identify-episodes", type=int, default=None); ap.add_argument("--hold-stat", choices=["last", "window"], default="last")
    a = ap.parse_args()

    A = WidowX(a.task, a.host, a.port, a.seed); a.out.parent.mkdir(parents=True, exist_ok=True)
    fvec = np.array([float(x) for x in a.fault_vec.split(",")]) if a.fault_vec else None
    corr = [int(x) for x in a.corr_dims.split(",")] if a.corr_dims else None
    if a.mode == "log":
        eps = []
        for ep in range(a.episodes):
            L = dict(u=[], y=[]); s, _, _ = episode(A, ep, fvec=fvec, log=L, horizon=a.horizon, max_steps=a.max_steps)
            L["success"] = s; eps.append(L); print(f"  ep {ep}: success={s} steps={len(L['u'])}", flush=True)
            a.out.write_text(json.dumps(eps))
        if not a.fault_vec:
            U = np.concatenate([np.array(e["u"]) for e in eps]); Y = np.concatenate([np.array(e["y"]) for e in eps])
            scale = np.maximum(np.abs(Y).mean(0) / np.maximum(np.abs(U).mean(0), 1e-9), 1e-9)
            print("per-channel |increment| / |command| (motion per unit action):", np.round(scale, 4))
            print("command |mean| per channel:", np.round(np.abs(U).mean(0), 4), "| increment |mean|:", np.round(np.abs(Y).mean(0), 5))
            W, r2 = fit_plant(a.out, scale); print("FIR R2 per channel:", np.round(r2, 3))
            json.dump({"scale": scale.tolist()}, open(str(a.out).replace(".json", "_scale.json"), "w"))
        print(f"successes {sum(e['success'] for e in eps)}/{len(eps)}")
        return
    scale = np.array(json.load(open(str(a.log).replace(".json", "_scale.json")))["scale"])
    if a.mode == "openloop":
        d = json.loads(a.log.read_text()); cmds = np.array(d[0]["u"])[:a.probe_steps]
        def replay(f):
            obs = A.reset(0); p = A.pose(obs); D = []
            for u_t in cmds:
                u7 = np.r_[np.asarray(u_t) + f, 1.0]
                obs, *_ = A.step(u7); p1 = A.pose(obs); D.append(increment(p1, p) / scale); p = p1
            return np.array(D)[len(cmds) // 2:]
        base = replay(np.zeros(6)); M = np.zeros((6, 6))
        for j in range(6):
            acc = []
            for sgn in (1.0, -1.0):
                f = np.zeros(6); f[j] = sgn * a.probe
                acc.append((replay(f) - base).mean(0) / (sgn * a.probe))
            M[:, j] = np.mean(acc, axis=0); print(f"  probe {POSE[j]}: column {np.round(M[:, j], 3)}", flush=True)
        print("M diagonal:", np.round(np.diag(M), 3)); print("cond(M):", round(float(np.linalg.cond(M)), 1))
        a.out.write_text(json.dumps({"M": M.tolist(), "probe": a.probe})); return
    W, r2 = fit_plant(a.log, scale); M = np.array(json.loads(a.openloop.read_text())["M"]); M_inv = np.linalg.pinv(M)
    res = dict(args={k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(a).items()}, arms={})
    arms = ([("healthy", None)] if a.with_healthy else []) + [("frozen_faulted", False), ("adaptive", True)]
    for tag, adapt in arms:
        ok, fh, per_ep, trajs = 0, [], [], []; f_carry = None; windows = []
        for ep in range(a.episodes):
            if adapt is None:
                s, f_hat, traj = episode(A, ep, W, M_inv, M, scale, None, False, horizon=a.horizon, max_steps=a.max_steps)
            else:
                s, f_hat, traj = episode(A, ep, W, M_inv, M, scale, fvec, adapt, a.gamma, a.dead, a.norm_r, a.clip, corr,
                                         f_init=(f_carry if (adapt and a.identify_episodes is not None) else None),
                                         freeze_after=(0 if (a.identify_episodes is not None and ep >= a.identify_episodes) else None),
                                         law=a.law, horizon=a.horizon, max_steps=a.max_steps, profile=a.profile, prof_p=a.prof_p, onset=a.onset)
                if adapt and a.identify_episodes is not None and ep < a.identify_episodes and traj:
                    win = np.median([st["f_hat"] for st in traj[30:150]] or [st["f_hat"] for st in traj], axis=0) if a.hold_stat == "window" else f_hat
                    windows.append(win); f_carry = np.median(windows, axis=0)
            ok += int(s); fh.append(f_hat.tolist()); per_ep.append(dict(task=0, init=ep, ok=bool(s))); trajs.append(traj)
            print(f"  [{tag}] ep {ep}: success={s}  f_hat={np.round(f_hat, 3)}", flush=True)
            res["arms"][tag] = dict(successes=ok, n=ep + 1, f_hat=fh, per_ep=per_ep,
                                    traj=[[st["f_hat"] for st in tr] for tr in trajs], f_true=[[st["f_true"] for st in tr] for tr in trajs])
            a.out.write_text(json.dumps(res))
        clip_report(fh, a.clip, tag); print(f"{tag}: {ok}/{a.episodes} = {100*ok/a.episodes:.0f}%\n", flush=True)


if __name__ == "__main__":
    main()
