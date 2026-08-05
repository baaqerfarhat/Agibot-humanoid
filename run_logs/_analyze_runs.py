#!/usr/bin/env python3
"""Preprocess deploy run_logs CSVs into a compact JSON for the analysis canvas.

Desired trajectory  = policy PD target  (`<joint>__tgt`)
Actual trajectory    = measured encoder  (`<joint>__pos_meas`)

Foot-edge signature  = ankle_roll deviation (foot tips onto its edge when the
                       measured ankle_roll drifts from its commanded target).
Fixation-oscillation = base roll + base angular-velocity activity.
"""
import csv, glob, json, math, os

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(RUN_DIR, "_run_analysis.json")

R2D = 180.0 / math.pi
N_SAMPLES = 130  # downsample every time series to this many points

LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
# channels we keep full (downsampled) traces for: meas + tgt
TRACE_JOINTS = [
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_hip_pitch_joint", "right_hip_pitch_joint",
]


def quat_to_rpy(x, y, z, w):
    # roll (x), pitch (y)
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    return roll * R2D, pitch * R2D


def downsample(arr, n=N_SAMPLES):
    if len(arr) <= n:
        return [round(v, 2) for v in arr]
    step = len(arr) / n
    return [round(arr[min(len(arr) - 1, int(i * step))], 2) for i in range(n)]


def rms(vals):
    if not vals:
        return 0.0
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def classify_policy(meta, fname):
    bp = (meta or {}).get("box_policy", "")
    if "v30" in bp or "v30" in fname:
        return "v30 (deployed)"
    if "policy.npz" in bp or "x2_box_policy.npz" in bp:
        return "v27 (prior)"
    return "unknown"


def process(csv_path):
    base = os.path.basename(csv_path)
    meta_path = csv_path[:-4] + ".meta.json"
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)

    rows = []
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    if len(rows) < 20:
        return None

    t = [float(r["t_s"]) for r in rows]
    t0 = t[0]
    t = [x - t0 for x in t]
    phase = [r["phase"] for r in rows]

    # base roll/pitch from quaternion (roll also logged directly, but recompute
    # pitch consistently)
    roll = []
    pitch = []
    for r in rows:
        rr, pp = quat_to_rpy(
            float(r["base_quat_x"]), float(r["base_quat_y"]),
            float(r["base_quat_z"]), float(r["base_quat_w"]),
        )
        roll.append(rr)
        pitch.append(pp)
    angvel = [
        math.sqrt(float(r["base_ang_vel_x"]) ** 2 + float(r["base_ang_vel_y"]) ** 2
                  + float(r["base_ang_vel_z"]) ** 2)
        for r in rows
    ]

    def col(j, kind):
        k = f"{j}__{kind}"
        return [float(r[k]) * R2D for r in rows]

    # ----- traces (downsampled) -----
    traces = {"t": downsample(t), "roll": downsample(roll), "pitch": downsample(pitch),
              "angvel": downsample(angvel)}
    # phase index downsampled (as strings)
    ph_ds = []
    if len(phase) <= N_SAMPLES:
        ph_ds = phase
    else:
        step = len(phase) / N_SAMPLES
        ph_ds = [phase[min(len(phase) - 1, int(i * step))] for i in range(N_SAMPLES)]
    traces["phase"] = ph_ds

    for j in TRACE_JOINTS:
        traces[j + "__meas"] = downsample(col(j, "pos_meas"))
        traces[j + "__tgt"] = downsample(col(j, "tgt"))

    # ----- metrics (full resolution) -----
    # active window = exclude ramp/settle/done, i.e. where the box policy is doing work
    active_idx = [i for i, p in enumerate(phase)
                  if p in ("pickup", "carry", "setdown", "policy")]
    if not active_idx:
        active_idx = list(range(len(rows)))
    pickup_idx = [i for i, p in enumerate(phase) if p in ("pickup", "policy")]
    if not pickup_idx:
        pickup_idx = active_idx

    def leg_track_rmse(idx):
        errs = []
        for j in LEG_JOINTS:
            m = col(j, "pos_meas")
            g = col(j, "tgt")
            errs += [m[i] - g[i] for i in idx]
        return rms(errs)

    # per-joint tracking RMSE over active window
    per_joint = {}
    for j in LEG_JOINTS:
        m = col(j, "pos_meas")
        g = col(j, "tgt")
        per_joint[j] = round(rms([m[i] - g[i] for i in active_idx]), 2)

    # foot-edge metric: ankle_roll |measured| and tracking error during pickup
    def ankle_stat(side, idx):
        j = f"{side}_ankle_roll_joint"
        m = col(j, "pos_meas")
        g = col(j, "tgt")
        dev = [abs(m[i]) for i in idx]           # absolute roll of the foot
        err = [m[i] - g[i] for i in idx]         # commanded vs actual
        return round(rms(dev), 2), round(max(dev) if dev else 0, 2), round(rms(err), 2)

    lroll_rms, lroll_max, lroll_err = ankle_stat("left", pickup_idx)
    rroll_rms, rroll_max, rroll_err = ankle_stat("right", pickup_idx)

    def rng(vals, idx):
        s = [vals[i] for i in idx]
        return (max(s) - min(s)) if s else 0.0

    base_roll_std = round(rms([roll[i] - (sum(roll[k] for k in pickup_idx) / len(pickup_idx))
                               for i in pickup_idx]), 2)
    base_roll_range = round(rng(roll, pickup_idx), 2)
    base_pitch_range = round(rng(pitch, pickup_idx), 2)
    angvel_rms = round(rms([angvel[i] for i in pickup_idx]), 3)

    phases_present = []
    for p in phase:
        if p not in phases_present:
            phases_present.append(p)
    # hybrid runs expose setdown/done; pickup-only runs stay in "policy" the
    # whole time so completion can't be read from phase -> report None.
    if "hybrid" in base:
        completed = any(p in ("setdown", "done") for p in phase)
    else:
        completed = None

    metrics = {
        "leg_track_rmse_active": round(leg_track_rmse(active_idx), 2),
        "per_joint_rmse": per_joint,
        "ankle_roll_L_dev_rms": lroll_rms, "ankle_roll_L_dev_max": lroll_max,
        "ankle_roll_L_track_err": lroll_err,
        "ankle_roll_R_dev_rms": rroll_rms, "ankle_roll_R_dev_max": rroll_max,
        "ankle_roll_R_track_err": rroll_err,
        "foot_edge_score": round((lroll_rms + rroll_rms) / 2, 2),
        "base_roll_std": base_roll_std,
        "base_roll_range": base_roll_range,
        "base_pitch_range": base_pitch_range,
        "base_angvel_rms": angvel_rms,
        "peak_abs_roll": round(max(abs(x) for x in roll), 2),
        "peak_abs_pitch": round(max(abs(x) for x in pitch), 2),
    }

    return {
        "file": base,
        "timestamp": base[:15],
        "kind": "hybrid" if "hybrid" in base else "pickup",
        "policy": classify_policy(meta, base),
        "duration_s": round(t[-1], 1),
        "n_rows": len(rows),
        "phases": phases_present,
        "completed": completed,
        "box_policy": (meta or {}).get("box_policy", ""),
        "gain_scale": (meta or {}).get("gain_scale"),
        "leg_filter": (meta or {}).get("leg_filter"),
        "run_path": (meta or {}).get("run_path", ""),
        "metrics": metrics,
        "traces": traces,
    }


def sim_checkpoints():
    """Foot-tilt (edge) trend across saved sim rollouts in /tmp, if present.

    Uses numpy only if available; degrades gracefully.
    """
    try:
        import numpy as np
    except Exception:
        return []
    fs = sorted(glob.glob("/tmp/x2_box_v30_planted_recovery_iter*_rollout.npz"),
                key=os.path.getmtime)
    out = []

    def rpy(q):
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        return roll * R2D

    for f in fs:
        try:
            d = np.load(f, allow_pickle=True)
            meta = json.loads(str(d["_metadata_json"]))
            bn = meta.get("body_names") or meta.get("bodies")
            bq = d["body_quat_xyzw"]
            li = bn.index("left_ankle_roll_link")
            ri = bn.index("right_ankle_roll_link")
            lr = rpy(bq[:, li])
            rr = rpy(bq[:, ri])
            n = bq.shape[0]
            w = slice(int(n * 0.15), int(n * 0.75))

            def rms(a):
                return float(np.sqrt(np.mean(a ** 2)))
            it = f.split("iter")[-1].split("_")[0]
            out.append({
                "iter": int(it),
                "footTiltL": round(rms(lr[w]), 1),
                "footTiltR": round(rms(rr[w]), 1),
                "footTiltMaxR": round(float(np.abs(rr[w]).max()), 1),
            })
        except Exception:
            continue
    out.sort(key=lambda r: r["iter"])
    return out


def main():
    files = sorted(glob.glob(os.path.join(RUN_DIR, "*_box_*_x2_box_policy*.csv")))
    runs = []
    for f in files:
        try:
            r = process(f)
            if r:
                runs.append(r)
        except Exception as e:
            print("skip", os.path.basename(f), "->", e)
    runs.sort(key=lambda r: r["timestamp"])
    out = {"generated": "run_logs analysis", "n_runs": len(runs), "runs": runs,
           "simCheckpoints": sim_checkpoints()}
    with open(OUT, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    kb = os.path.getsize(OUT) / 1024
    print(f"wrote {OUT}  ({kb:.0f} KB, {len(runs)} runs)")
    # quick console summary
    print(f"{'file':44} {'policy':16} {'kind':7} {'edge':>5} {'rollσ':>6} {'legRMSE':>7} done")
    for r in runs:
        m = r["metrics"]
        print(f"{r['file']:44} {r['policy']:16} {r['kind']:7} "
              f"{m['foot_edge_score']:5.2f} {m['base_roll_std']:6.2f} "
              f"{m['leg_track_rmse_active']:7.2f} {r['completed']}")


if __name__ == "__main__":
    main()
