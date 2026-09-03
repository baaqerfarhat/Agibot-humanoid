"""The same law on a different manipulator: bimanual ALOHA (ViperX x2) in gym_aloha, driven
by pi0_aloha_sim over openpi's websocket server.

Everything that made LIBERO work is re-derived here in JOINT space, because that is what
this robot's action interface is:
  action   14 absolute joint targets (rad), 6 arm + 1 gripper per side, at 50 Hz
  state    the same 14 joint positions, measured
  plant    per-joint FIR on POSITION: q_t = sum_k h_k u_{t-k} + c, identified on HEALTHY
           rollouts (mode log). Not on the increment dq as in LIBERO: there the command IS an
           increment, here it is an absolute target that a position servo tracks, so an
           offset in the target lives in the steady-state POSITION error and leaves almost
           nothing in dq -- a dq plant gave M ~ 0.008 and a law that amplified noise 125x.
  fault    additive offset on the commanded targets (an encoder / calibration offset), or a
           gain (loss of effectiveness), on chosen joints
  M        d(motion)/d(fault) per joint by open-loop replay (mode openloop)
  law      f_hat <- f_hat + gamma (M^-1 r - f_hat), deadzone / normaliser / clip, identical
           to adaptive_law.py -- the point is that nothing about the law changes.
No SO(3) anywhere: joint space is a vector space, which removes the whole class of bug that
Sec 2.2b of the LIBERO record was about.
"""
from __future__ import annotations
import argparse, atexit, collections, json, pathlib, time
import numpy as np
import gymnasium as gym
import gym_aloha  # noqa: F401  (registers the envs)
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wc

NJ, HORIZON, K_FIR, DT = 14, 10, 6, 0.02
TASK = "gym_aloha/AlohaTransferCube-v0"
PROMPT = "Transfer cube"


def clip_report(vals, clip, name):
    v = np.abs(np.asarray(vals, float))
    if v.size == 0: return
    rail = (np.abs(v - clip) < 1e-6).mean(axis=0)
    if np.any(rail > 0.2):
        hit = ", ".join(f"j{i} {100*rail[i]:.0f}%" for i in range(len(rail)) if rail[i] > 0.2)
        print(f"  !! {name}: at the clip (+-{clip}) in >20% of episodes on [{hit}] -- SATURATED, not estimated.")


class Aloha:
    def __init__(self, host, port, seed=0):
        self.env = gym.make(TASK, obs_type="pixels_agent_pos", render_mode="rgb_array")
        self.client = _wc.WebsocketClientPolicy(host, port)
        self.seed = seed
        atexit.register(self.env.close)

    def reset(self, ep):
        obs, _ = self.env.reset(seed=self.seed + ep)
        return obs

    def policy_obs(self, obs):
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(obs["pixels"]["top"], 224, 224))
        return {"state": np.asarray(obs["agent_pos"], np.float64),
                "images": {"cam_high": np.transpose(img, (2, 0, 1))}, "prompt": PROMPT}


def fit_plant(log_path):
    """Per-joint FIR on healthy (u, dq) pairs. Returns W (NJ, K_FIR+2): taps + intercept."""
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
            norm_r=0.05, clip=0.3, corr=None, profile="step", prof_p=60.0, onset=0, log=None):
    obs = A.reset(ep); q = np.asarray(obs["agent_pos"], float)
    hist = collections.deque([np.zeros(NJ)] * (K_FIR + 1), maxlen=K_FIR + 1)
    f_hat = np.zeros(NJ); plan = collections.deque(); traj = []; success = False
    fvec = np.zeros(NJ) if fvec is None else np.asarray(fvec, float)
    m = np.zeros(NJ) if corr is None else np.isin(np.arange(NJ), corr).astype(float)
    for t in range(300):
        if not plan:
            chunk = np.asarray(A.client.infer(A.policy_obs(obs))["actions"], float)
            plan.extend(chunk[:HORIZON])
        a_cmd = np.asarray(plan.popleft(), float)
        c = (-f_hat * m) if adapt else np.zeros(NJ)
        a_corr = a_cmd + c
        live = t >= onset; u = t - onset
        scale = {"step": 1.0, "ramp": min(1.0, u / max(prof_p, 1e-9)),
                 "sine_bias": 0.5 * (1 + np.sin(2 * np.pi * u / max(prof_p, 1e-9))),
                 "intermittent": 1.0 if int(u // max(prof_p, 1)) % 2 == 0 else 0.0}[profile] if live else 0.0
        f_now = fvec * scale
        a_exec = a_corr + f_now
        if gain is not None and live:
            # loss of effectiveness on the commanded MOTION, not the absolute target: the
            # controller receives q + g (target - q), i.e. it only gets a fraction of the way
            a_exec = q + gain * (a_exec - q)
        obs, r, term, trunc, info = A.env.step(a_exec)
        q1 = np.asarray(obs["agent_pos"], float); dq = q1 - q; q = q1
        hist.appendleft(a_corr[:NJ].copy())
        if log is not None:
            log["u"].append(a_corr.tolist()); log["q"].append(q1.tolist())
        if W is not None:
            H = np.array(hist)
            pred = np.array([W[j, :K_FIR + 1] @ H[:, j] + W[j, -1] for j in range(NJ)])
            res = q1 - pred                       # position residual
            if adapt:
                est = M_inv @ res
                nr = float(np.linalg.norm(res))
                if nr < dead: est = np.zeros(NJ)
                est = est / (1.0 + (nr / norm_r) ** 2)
                f_hat = np.clip(f_hat + gamma * (est - f_hat), -clip, clip)
            traj.append(dict(t=t, f_hat=f_hat.tolist(), f_true=f_now.tolist()))
        success = success or bool(info.get("is_success", False)) or (r >= 4)
        if term or trunc: break
    return success, f_hat, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["log", "openloop", "run"])
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--episodes", type=int, default=10); ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--log", type=pathlib.Path); ap.add_argument("--openloop", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--fault-vec", default=None, help="14 comma-separated joint offsets (rad)")
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--corr-joints", default=None, help="joints to correct, e.g. 0,1,2,3,4,5")
    ap.add_argument("--profile", default="step", choices=["step", "ramp", "sine_bias", "intermittent"])
    ap.add_argument("--prof-p", type=float, default=60.0); ap.add_argument("--onset", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.08); ap.add_argument("--dead", type=float, default=0.002)
    ap.add_argument("--norm-r", type=float, default=0.05); ap.add_argument("--clip", type=float, default=0.3)
    ap.add_argument("--probe", type=float, default=0.02, help="openloop: per-joint probe (rad)")
    a = ap.parse_args()
    A = Aloha(a.host, a.port, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)

    if a.mode == "log":
        eps = []
        for ep in range(a.episodes):
            L = dict(u=[], q=[]); s, _, _ = episode(A, ep, log=L); L["success"] = s; eps.append(L)
            print(f"  healthy ep {ep}: success={s} steps={len(L['u'])}")
        a.out.write_text(json.dumps(eps)); W, r2 = fit_plant(a.out)
        print("FIR R2 per joint:", np.round(r2, 3)); print(f"successes {sum(e['success'] for e in eps)}/{len(eps)}")
        return

    if a.mode == "openloop":
        # replay a healthy command sequence open loop with and without a per-joint probe
        d = json.loads(a.log.read_text()); cmds = np.array(d[0]["u"])[:120]
        def replay(f):
            obs = A.reset(0); q = np.asarray(obs["agent_pos"], float); D = []
            for u_t in cmds:
                obs, *_ = A.env.step(np.asarray(u_t) + f); D.append(np.asarray(obs["agent_pos"], float))
            return np.array(D)[len(cmds) // 2:]          # steady state only
        base = replay(np.zeros(NJ)); M = np.zeros((NJ, NJ))
        for j in range(NJ):
            acc = []
            for sgn in (1.0, -1.0):
                f = np.zeros(NJ); f[j] = sgn * a.probe
                acc.append((replay(f) - base).mean(0) / (sgn * a.probe))
            M[:, j] = np.mean(acc, axis=0)
        print("M diagonal:", np.round(np.diag(M), 3)); print("cond(M):", round(float(np.linalg.cond(M)), 1))
        a.out.write_text(json.dumps({"M": M.tolist(), "probe": a.probe})); return

    W, r2 = fit_plant(a.log); M = np.array(json.loads(a.openloop.read_text())["M"]); M_inv = np.linalg.pinv(M)
    fvec = [float(x) for x in a.fault_vec.split(",")] if a.fault_vec else None
    corr = [int(x) for x in a.corr_joints.split(",")] if a.corr_joints else None
    res = dict(args=vars(a) | {"out": str(a.out), "log": str(a.log), "openloop": str(a.openloop)}, arms={})
    for tag, adapt in (("frozen_faulted", False), ("adaptive", True)):
        ok, fh, per_ep, trajs = 0, [], [], []
        for ep in range(a.episodes):
            s, f_hat, traj = episode(A, ep, W, M_inv, fvec, a.gain, adapt, a.gamma, a.dead, a.norm_r, a.clip,
                                     corr, a.profile, a.prof_p, a.onset)
            ok += int(s); fh.append(f_hat.tolist()); per_ep.append(dict(task=0, init=ep, ok=bool(s))); trajs.append(traj)
            print(f"  [{tag}] ep {ep}: success={s}  f_hat[:6]={np.round(f_hat[:6], 3)}")
        res["arms"][tag] = dict(successes=ok, n=a.episodes, f_hat=fh, per_ep=per_ep,
                                traj=[[st["f_hat"] for st in tr] for tr in trajs],
                                f_true=[[st["f_true"] for st in tr] for tr in trajs])
        a.out.write_text(json.dumps(res)); clip_report(fh, a.clip, tag)
        print(f"{tag}: {ok}/{a.episodes} = {100*ok/a.episodes:.0f}%\n")


if __name__ == "__main__":
    main()
