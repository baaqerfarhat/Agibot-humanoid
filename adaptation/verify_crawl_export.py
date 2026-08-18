#!/usr/bin/env python3
"""Check the exported crawl policy .npz against a recorded Isaac rollout.

numpy only -- no Isaac, no torch. Answers the three questions that decide
whether `deploy_x2_crawl.py` is feeding the policy what training fed it:

  1. Is the exported network + observation layout right? Rebuild the actor
     observation from the rollout's own state and compare the forward pass
     against the actions Isaac recorded. A wrong term order or a missing scale
     shows up as an error the size of an action (1-30); the noise the actor
     observations carry in training only moves actions by ~0.05.

  2. Which body frame does the policy's attitude come from? holosoma builds
     both `base_ang_vel` and `projected_gravity` from `env.base_quat`, the
     articulation root -- the PELVIS. Feeding the torso IMU instead is the
     substitution that broke the v33 box-pickup policy on hardware. This
     reconstructs the observation both ways and reports which one matches, plus
     how far apart the two gravity signals actually are.

  3. Does the deploy path actually recover that pelvis frame? The robot has no
     pelvis IMU, so `base_frame.PelvisEstimator` composes the torso IMU with the
     measured waist joints. The rollout records the true pelvis, so the
     reconstruction can be scored directly against it.

  4. Which joints ask for torque they cannot get? |action| >= 4 is exactly the
     effort limit (action_scale = 0.25*effort_limit/kp). Those requests are
     absorbed by contact in training; the ones that are not are what
     --action-clip bounds on hardware.

    python verify_crawl_export.py                       # committed trimmed rollout
    python verify_crawl_export.py --rollout <full>.npz  # any render_crawl_rollout dump
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = REPO_ROOT / "box_pickup/policy/x2_crawl_policy_v5_iter86000.npz"
DEFAULT_ROLLOUT = REPO_ROOT / "adaptation/crawl_v5_iter86000_rollout_trim.npz"

# Actor observation slices, in holosoma's ALPHABETICAL term order (167 dims).
SLICES = {
    "actions": slice(0, 31),
    "base_ang_vel": slice(31, 34),
    "dof_pos": slice(34, 65),
    "dof_vel": slice(65, 96),
    "motion_command": slice(96, 158),
    "motion_ref_ori_b": slice(158, 164),
    "projected_gravity": slice(164, 167),
}
# `noise` from the run's observation config (holosoma draws uniform(-n, +n)).
OBS_NOISE = {
    "base_ang_vel": 0.2,
    "dof_pos": 0.01,
    "dof_vel": 0.5,
    "motion_ref_ori_b": 0.05,
    "projected_gravity": 0.05,
}


# ------------------------------- math helpers -------------------------------
def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def quat_inv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def projected_gravity(q):
    """World gravity [0,0,-1] in the body frame of q (xyzw)."""
    q = np.asarray(q, float)
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    v = np.array([0.0, 0.0, -1.0])
    qv = np.array([qx, qy, qz])
    return v * (2 * qw * qw - 1) - np.cross(qv, v) * qw * 2 + qv * np.dot(qv, v) * 2


def _import_base_frame():
    """The deploy-side pelvis reconstruction, so this scores the shipped code."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "agibot_control_functions"))
    try:
        import base_frame

        return base_frame
    except Exception:
        return None


class Actor:
    """The exported MLP, normalization baked in (rsl_rl actor, ELU)."""

    def __init__(self, npz):
        n = int(npz["n_layers"])
        self.W = [npz[f"W{i}"].astype(np.float32) for i in range(n)]
        self.b = [npz[f"b{i}"].astype(np.float32) for i in range(n)]
        self.mean = npz["mean"].astype(np.float32)
        self.std = npz["std"].astype(np.float32)

    def __call__(self, obs):
        x = (np.atleast_2d(obs).astype(np.float32) - self.mean) / self.std
        for i in range(len(self.W) - 1):
            x = x @ self.W[i].T + self.b[i]
            x = np.where(x > 0.0, x, np.exp(np.clip(x, -30.0, 0.0)) - 1.0)
        return x @ self.W[-1].T + self.b[-1]


def build_obs(roll, policy, meta, base: str, rotate_ang_vel: bool) -> np.ndarray:
    """Rebuild the actor observation from recorded state, noise-free.

    base: which body supplies base_ang_vel + projected_gravity ("pelvis"/"torso").
    rotate_ang_vel: rotate the recorded world-frame root rate into the base frame
    (holosoma does `quat_rotate_inverse(base_quat, root_states[:, 10:13])`).
    """
    dof, dvel, act = roll["dof_pos"], roll["dof_vel"], roll["actions"]
    q_pelvis, w_root, q_torso = roll["root_quat_xyzw"], roll["root_ang_vel"], roll["torso_quat_xyzw"]
    ref_p, ref_v, ref_q = policy["ref_joint_pos"], policy["ref_joint_vel"], policy["ref_quat_xyzw"]
    dflt = np.array(meta["default_joint_pos"])

    T = len(act)
    obs = np.zeros((T, int(meta["obs_dim"])))
    for t in range(T):
        f = min(t, len(ref_p) - 1)
        q_base = q_pelvis[t] if base == "pelvis" else q_torso[t]
        w = quat_to_mat(q_base).T @ w_root[t] if rotate_ang_vel else w_root[t]
        # motion_ref_ori_b tracks torso_link regardless of the base frame.
        q_rel = quat_mul(quat_inv(q_torso[t]), ref_q[f])
        ori6 = quat_to_mat(q_rel)[:, :2].reshape(-1)
        obs[t] = np.concatenate([
            act[t - 1] if t > 0 else np.zeros(31),
            w, dof[t] - dflt, dvel[t], ref_p[f], ref_v[f], ori6,
            projected_gravity(q_base),
        ])
    return obs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default=str(DEFAULT_POLICY))
    ap.add_argument("--rollout", default=str(DEFAULT_ROLLOUT),
                    help="render_crawl_rollout.py dump, full or trimmed")
    ap.add_argument("--seeds", type=int, default=5, help="noise draws for the calibration")
    args = ap.parse_args()

    p = np.load(args.policy, allow_pickle=True)
    meta = json.loads(str(p["meta_json"]))
    actor = Actor(p)
    joint_names = meta["joint_names"]

    r = dict(np.load(args.rollout, allow_pickle=True))
    if "torso_quat_xyzw" not in r:  # full dump: pull torso out of the body array
        body_names = json.loads(str(r["_metadata_json"]))["body_names"]
        r["torso_quat_xyzw"] = r["body_quat_xyzw"][:, body_names.index(meta["ref_body"])]
    act = r["actions"]

    print("=" * 78)
    print(f"  policy:   {args.policy}")
    print(f"  task:     {meta.get('task')}   run: {meta.get('run_path', '?')}")
    print(f"  rollout:  {args.rollout}  ({len(act)} frames)")
    print("=" * 78)

    # ---- 1 + 2: which frame reproduces the recorded actions? ----------------
    print("\n[1/4] rebuild the observation and compare against Isaac's actions")
    print("      (mean |action error| over all 31 joints x all frames)\n")
    print(f"      {'base frame':12s} {'root rate':16s} {'mean':>8s} {'median':>8s}")
    results = {}
    for base in ("pelvis", "torso"):
        for rot in (True, False):
            obs = build_obs(r, p, meta, base, rot)
            err = np.abs(actor(obs) - act)
            results[(base, rot)] = (obs, err)
            label = "world->base" if rot else "as-stored"
            print(f"      {base:12s} {label:16s} {err.mean():8.4f} {np.median(err):8.4f}")

    best = min(results, key=lambda k: results[k][1].mean())
    obs_best, err_best = results[best]
    print(f"\n      best: base={best[0]}, root rate "
          f"{'rotated into the base frame' if best[1] else 'as stored'}")
    print(f"      -> holosoma's env.base_quat is the articulation root, so 'pelvis' is")
    print(f"         the expected winner; deploy_x2_crawl.py reconstructs it from the")
    print(f"         torso IMU + waist joints (--base-frame pelvis, the default).")

    # calibrate that residual against the noise the actor obs carry in training
    a_clean = actor(obs_best)
    rng = np.random.default_rng(0)
    noise_vec = np.zeros(obs_best.shape[1])
    for term, nz in OBS_NOISE.items():
        noise_vec[SLICES[term]] = nz
    spread = np.mean([
        np.abs(actor(obs_best + rng.uniform(-1, 1, obs_best.shape) * noise_vec) - a_clean).mean()
        for _ in range(args.seeds)
    ])
    print(f"\n      residual                        {err_best.mean():.4f}")
    print(f"      action change from training obs noise alone   {spread:.4f}")
    verdict = ("consistent with an exact export"
               if err_best.mean() < 10 * spread else "TOO LARGE -- check the term order")
    print(f"      ratio {err_best.mean() / spread:.2f}x  -> {verdict}")
    print("      (the rollout itself was recorded with that noise applied, so the")
    print("       reconstruction cannot and should not match to zero)")

    # how much does the wrong frame actually change the attitude signal?
    g_pelvis = np.array([projected_gravity(q) for q in r["root_quat_xyzw"]])
    g_torso = np.array([projected_gravity(q) for q in r["torso_quat_xyzw"]])
    ang = np.degrees(np.arccos(np.clip((g_pelvis * g_torso).sum(1), -1, 1)))
    print(f"\n[2/4] pelvis vs torso projected gravity: mean {ang.mean():.1f} deg, "
          f"max {ang.max():.1f} deg")
    print(f"      per-axis max |difference| = {np.abs(g_pelvis - g_torso).max(0).round(3)}")
    print("      Gravity is this policy's only attitude signal, so that gap is the")
    print("      cost of reading it off the torso IMU instead of the pelvis.")

    # ---- 3: does PelvisEstimator close that gap on real geometry? -----------
    print("\n[3/4] pelvis reconstruction (torso IMU + waist joints) vs the TRUE pelvis")
    bf = _import_base_frame()
    if bf is None:
        print("      SKIPPED: agibot_control_functions/base_frame.py not importable")
    else:
        iw = [joint_names.index(n) for n in bf.WAIST_JOINTS]
        dof, q_torso = r["dof_pos"], r["torso_quat_xyzw"]
        est_err = np.empty(len(dof))
        for t in range(len(dof)):
            yaw, pitch, roll = dof[t][iw]
            R_pt = bf.pelvis_from_torso_rot(float(yaw), float(pitch), float(roll))
            q_est = bf._mat_to_quat_xyzw(bf._quat_xyzw_to_mat(q_torso[t]) @ R_pt.T)
            est_err[t] = np.degrees(np.arccos(np.clip(
                np.dot(g_pelvis[t], projected_gravity(q_est)), -1, 1)))
        print(f"      reconstructed from torso IMU + waist:  mean {est_err.mean():6.2f} deg"
              f"   max {est_err.max():6.2f} deg")
        print(f"      raw torso IMU (the naive choice):      mean {ang.mean():6.2f} deg"
              f"   max {ang.max():6.2f} deg")
        print(f"      -> the reconstruction removes "
              f"{100 * (1 - est_err.mean() / max(ang.mean(), 1e-9)):.1f}% of the error")
        w = dof[:, iw]
        print(f"      waist travel over the crawl: yaw {w[:, 0].min():+.2f}..{w[:, 0].max():+.2f}"
              f", pitch {w[:, 1].min():+.2f}..{w[:, 1].max():+.2f}"
              f", roll {w[:, 2].min():+.2f}..{w[:, 2].max():+.2f} rad")
        print("      (that waist travel is exactly what the raw IMU misreads as body tilt)")

    # ---- 4: saturated torque requests --------------------------------------
    print("\n[4/4] |action| >= 4 -- asking for more torque than the actuator can give")
    rows = [(n, 100.0 * float((np.abs(act[:, i]) >= 4.0).mean()), float(np.abs(act[:, i]).max()))
            for i, n in enumerate(joint_names)]
    for n, frac, mx in sorted(rows, key=lambda x: -x[1]):
        if frac >= 5.0:
            print(f"      {n:30s} {frac:5.1f}% of frames   max|a| = {mx:5.1f}")
    print(f"      whole body: {100.0 * float((np.abs(act) >= 4.0).mean()):.1f}% of "
          "(frame, joint) pairs")
    print("\n      grouped, against deploy_x2_crawl.py's --action-clip-joints default:")
    for key in ("ankle_roll", "wrist", "head", "knee", "hip", "shoulder", "elbow", "waist"):
        idx = [i for i, n in enumerate(joint_names) if key in n]
        if idx:
            f = 100.0 * float((np.abs(act[:, idx]) >= 4.0).mean())
            tag = "  <- clipped by default" if key in ("ankle_roll", "wrist") else ""
            print(f"        {key:11s} ({len(idx)} joints): {f:5.1f}%{tag}")
    print("\n      ankle rolls and wrists are clipped because training only gets away")
    print("      with those requests thanks to contact (palms on the slope, feet on")
    print("      the ramp) holding the joint at the reference. Nothing pushes back on")
    print("      the head, so sim and hardware already agree there and it is left alone.")


if __name__ == "__main__":
    main()
