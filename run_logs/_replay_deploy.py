#!/usr/bin/env python3
"""Replay the box policy offline on a deploy CSV log.

Every input the policy consumed at run time is in the log (measured joint
pos/vel, the base_ang_vel that was actually fed, the torso quat) except
prev_action, which is just this replay's own previous output. The recursion is
deterministic and starts from zeros at engage, so the reconstructed action
sequence is the one the robot really produced.

That gives the RAW target the policy asked for, which the log does not contain:
the logged `__tgt` is post-EMA, post-rate-limit. Comparing the two isolates how
much the deploy-side smoothing distorted the commanded trajectory.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], np.float32)


def quat_inv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], np.float32)


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]],
                    np.float32)


def yaw_quat(q):
    x, y, z, w = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], np.float32)


class Policy:
    def __init__(self, path):
        d = np.load(path, allow_pickle=True)
        self.mean = d["mean"].astype(np.float32)
        self.std = d["std"].astype(np.float32)
        n = int(d["n_layers"])
        self.W = [d[f"W{i}"].astype(np.float32) for i in range(n)]
        self.b = [d[f"b{i}"].astype(np.float32) for i in range(n)]
        self.meta = json.loads(str(d["meta_json"]))
        self.ref_joint_pos = d["ref_joint_pos"].astype(np.float32)
        self.ref_joint_vel = d["ref_joint_vel"].astype(np.float32)
        self.ref_quat_xyzw = d["ref_quat_xyzw"].astype(np.float32)

    def __call__(self, obs):
        x = (obs.astype(np.float32) - self.mean) / self.std
        for i in range(len(self.W) - 1):
            x = x @ self.W[i].T + self.b[i]
            x = np.where(x > 0.0, x, np.exp(np.clip(x, -30.0, 0.0)) - 1.0)
        return x @ self.W[-1].T + self.b[-1]


def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def replay(csv_path, policy):
    meta_path = csv_path.replace(".csv", ".meta.json")
    meta = json.load(open(meta_path))
    jn = policy.meta["joint_names"]
    default = np.array(policy.meta["default_joint_pos"], np.float32)
    ascale = np.array(policy.meta["action_scale"], np.float32)
    rows = [r for r in load_csv(csv_path) if r["phase"] == "policy"]
    if len(rows) < 5:
        return None

    q_align = np.array([float(rows[0][f"base_quat_{c}"]) for c in "xyzw"], np.float32)
    yaw_off = quat_mul(yaw_quat(q_align), quat_inv(yaw_quat(policy.ref_quat_xyzw[0])))

    T = policy.ref_joint_pos.shape[0]
    last_action = np.zeros(31, np.float32)
    out = {"t": [], "frame": [], "raw": [], "tgt": [], "meas": [], "act": [],
           "vmeas": [], "w": [], "ori6": []}
    for r in rows:
        frame = min(int(r["frame"]), T - 1)
        q = np.array([float(r[f"{n}__pos_meas"]) for n in jn], np.float32)
        dq = np.array([float(r[f"{n}__vel_meas"]) for n in jn], np.float32)
        tgt = np.array([float(r[f"{n}__tgt"]) for n in jn], np.float32)
        quat = np.array([float(r[f"base_quat_{c}"]) for c in "xyzw"], np.float32)
        w = np.array([float(r[f"obs_ang_vel_{c}"]) for c in "xyz"], np.float32)

        q_ref = quat_mul(yaw_off, policy.ref_quat_xyzw[frame])
        ori6 = quat_to_mat(quat_mul(quat_inv(quat), q_ref))[:, :2].reshape(-1)
        obs = np.concatenate([last_action, w, q - default, dq,
                              policy.ref_joint_pos[frame],
                              policy.ref_joint_vel[frame], ori6]).astype(np.float32)
        a = policy(obs).reshape(-1)
        last_action = a.astype(np.float32)

        out["t"].append(float(r["t_s"]))
        out["frame"].append(frame)
        out["act"].append(a)
        out["raw"].append(a * ascale + default)
        out["tgt"].append(tgt)
        out["meas"].append(q)
        out["vmeas"].append(dq)
        out["w"].append(w)
        out["ori6"].append(ori6)

    for k in ("raw", "tgt", "meas", "act", "vmeas", "w", "ori6"):
        out[k] = np.array(out[k])
    out["t"] = np.array(out["t"])
    out["frame"] = np.array(out["frame"])
    out["meta"] = meta
    out["jn"] = jn
    return out


def report(name, R, max_step=0.15):
    jn = R["jn"]
    m = R["meta"]
    raw, tgt, meas = R["raw"], R["tgt"], R["meas"]
    print("=" * 100)
    print(f"{name}   leg_filter={m['leg_filter']}  gain={m['gain_scale']}  "
          f"ticks={len(R['t'])}  frames {R['frame'][0]}->{R['frame'][-1]}")
    print("=" * 100)

    # --- how far the executed command drifted from what the policy asked for ---
    lag = tgt - raw
    print(f"  |tgt - raw_policy_target|:  mean {np.abs(lag).mean():.4f}  "
          f"p95 {np.percentile(np.abs(lag),95):.4f}  max {np.abs(lag).max():.4f} rad")

    # --- rate-limit saturation: |d tgt| pinned at max_step ---
    dtgt = np.diff(tgt, axis=0)
    sat = np.abs(dtgt) > max_step - 1e-6
    print(f"  rate-limiter ({max_step} rad/tick) engaged on "
          f"{100.0*sat.mean():.2f}% of joint-ticks; "
          f"{100.0*sat.any(axis=1).mean():.1f}% of ticks clipped >=1 joint")
    order = np.argsort(-sat.mean(axis=0))
    for i in order[:8]:
        if sat[:, i].mean() <= 0:
            break
        print(f"      {jn[i]:30s} clipped {100.0*sat[:,i].mean():5.1f}% of ticks   "
              f"max|d raw| {np.abs(np.diff(raw[:,i])).max():.3f} rad/tick   "
              f"max lag {np.abs(lag[:,i]).max():+.3f}")

    # --- what the policy WANTED to move faster than the limiter allows ---
    draw = np.abs(np.diff(raw, axis=0))
    want = draw > max_step
    print(f"  policy asked for >{max_step} rad/tick on {100.0*want.mean():.2f}% of joint-ticks")

    # --- servo tracking error (actuator/gain limited, not deploy-code limited) ---
    err = meas - tgt
    print("  worst position tracking error (meas - tgt), policy phase:")
    for i in np.argsort(-np.abs(err).max(axis=0))[:8]:
        k = int(np.argmax(np.abs(err[:, i])))
        print(f"      {jn[i]:30s} max {err[k,i]:+.3f} rad @ frame {R['frame'][k]:3d}   "
              f"rms {np.sqrt((err[:,i]**2).mean()):.3f}")

    # --- action saturation: policy output near its trained range ---
    act = R["act"]
    print(f"  |action| max {np.abs(act).max():.2f}   "
          f"ticks with |action|>3: {100.0*(np.abs(act)>3).any(axis=1).mean():.1f}%   "
          f">5: {100.0*(np.abs(act)>5).any(axis=1).mean():.1f}%")
    for i in np.argsort(-np.abs(act).max(axis=0))[:5]:
        print(f"      {jn[i]:30s} |a|max {np.abs(act[:,i]).max():6.2f}")
    print()


def main():
    policy = Policy(os.path.join(REPO, "box_pickup/policy/x2_box_policy_v33_iter253000.npz"))
    files = sys.argv[1:] or sorted(
        f for f in os.listdir(HERE)
        if f.startswith("20260812_") and f.endswith(".csv"))
    for f in files:
        p = f if os.path.isabs(f) else os.path.join(HERE, f)
        R = replay(p, policy)
        if R is None:
            print(f"(skip {os.path.basename(f)}: no policy-phase rows)")
            continue
        report(os.path.basename(f), R)


if __name__ == "__main__":
    main()
