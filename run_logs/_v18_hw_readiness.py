"""Is v18 safe to put on the robot? Compare it against the v17 that was.

Four questions decide it, and each has a number that came off the hardware or off
the finding that followed it:

  1. Did the ankle_roll exploit come back? The scale went up 3x, and the cap it
     replaced existed to stop a sustained unreachable command. The mentor's test is
     the command gap, mean |target - achieved|: 33.3 deg while exploited, 4.6 deg
     while capped.
  2. Can the ankles now follow the reference? That is what the scale was for. On
     hardware v17 covered 0.65-0.74 of the reference range while hips and knees
     managed 0.85-0.98.
  3. Is anything over its torque limit? ankle_pitch exceeded 36 N-m in 4 of 7
     engaged v17 runs, up to 46.0 N-m, and has no override.
  4. Does it still chatter? The jitter that made v16 unrunnable was 28.6 mrad RMS
     above 5 Hz in sim, against 5.8 mrad in the reference.
"""

from __future__ import annotations

import json

import numpy as np

REF = ("/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/x2_31dof"
       "/whole_body_tracking/sub3_largebox_003_walk_feasible.npz")
RUNS = {
    "v17 i49000 (was on hw)": "/tmp/x2_box_walk_feasible_v17_iter49000_rollout.npz",
    "v18 i81499 (new)": "/tmp/x2_box_ankle_scale_v18_iter81499_rollout.npz",
}
LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
R2D = 180.0 / np.pi


def hp_rms(x, fps=50.0, cut=5.0):
    """RMS of everything above `cut` Hz, in mrad -- oscillation with the task removed."""
    f = np.fft.rfftfreq(len(x), 1.0 / fps)
    X = np.fft.rfft(x, axis=0)
    X[f < cut] = 0.0
    return float(np.sqrt((np.fft.irfft(X, n=len(x), axis=0) ** 2).mean()) * 1000.0)


def main():
    r = np.load(REF, allow_pickle=True)
    rjn = [str(x) for x in r["joint_names"]]
    rdof = np.asarray(r["joint_pos"])[:, 7:]

    out = {}
    for tag, path in RUNS.items():
        d = np.load(path, allow_pickle=True)
        m = json.loads(str(d["_metadata_json"]))
        jn = list(m["dof_names"])
        eff = np.asarray(m["effort_limits"], float)
        lo = np.asarray(m["dof_pos_lower_limits"], float)
        hi = np.asarray(m["dof_pos_upper_limits"], float)
        out[tag] = dict(
            jn=jn, eff=eff, lo=lo, hi=hi,
            tgt=np.asarray(d["dof_pos_target"]), pos=np.asarray(d["dof_pos"]),
            tau=np.asarray(d["torques_substep"]).reshape(-1, len(jn)),
            act=np.asarray(d["actions"]),
            n=len(d["dof_pos"]),
        )

    # ---- 1. the command gap on ankle_roll --------------------------------------
    print("1. ANKLE_ROLL COMMAND GAP  mean |target - achieved|")
    print("   (exploited 33.3 deg / capped 4.6 deg, measured on hardware)")
    for tag, s in out.items():
        gaps = []
        for side in ("left", "right"):
            i = s["jn"].index(f"{side}_ankle_roll_joint")
            gaps.append(np.abs(s["tgt"][:, i] - s["pos"][:, i]).mean() * R2D)
        print(f"   {tag:24s} left {gaps[0]:5.1f} deg   right {gaps[1]:5.1f} deg")

    # ---- 2. range of motion, measured / reference -------------------------------
    print("\n2. RANGE OF MOTION  measured / reference  (hw v17: ankles 0.65-0.74,"
          " hips+knees 0.85-0.98)")
    hdr = "   " + f"{'joint':<24}" + "".join(f"{t:>26}" for t in out)
    print(hdr)
    for base in LEG:
        for side in ("left", "right"):
            j = f"{side}_{base}_joint"
            if j not in rjn:
                continue
            rr = np.ptp(rdof[:, rjn.index(j)]) * R2D
            row = f"   {j:<24}"
            for tag, s in out.items():
                i = s["jn"].index(j)
                mr = np.ptp(s["pos"][:, i]) * R2D
                flag = "  <-- short" if mr / rr < 0.80 else ""
                row += f"{mr:8.1f}/{rr:5.1f} = {mr/rr:4.2f}{flag:>0}"
                row = f"{row:<{len(row)}}"
            print(row)

    # ---- 3. torque against the limit --------------------------------------------
    print("\n3. PEAK TORQUE as % of the effort limit  (ankle_pitch hit 128% on hw)")
    print(f"   {'joint':<24}" + "".join(f"{t:>24}" for t in out))
    for base in LEG:
        for side in ("left", "right"):
            j = f"{side}_{base}_joint"
            row = f"   {j:<24}"
            for tag, s in out.items():
                if j not in s["jn"]:
                    row += f"{'-':>24}"
                    continue
                i = s["jn"].index(j)
                pk = np.abs(s["tau"][:, i]).max()
                pct = 100.0 * pk / s["eff"][i]
                row += f"{pk:12.1f} N-m {pct:5.0f}%" + ("!" if pct > 95 else " ")
            print(row)

    # ---- 4. chatter --------------------------------------------------------------
    print("\n4. CHATTER  RMS of leg position targets above 5 Hz, in mrad")
    print("   (v16, which was unrunnable: 28.6 commanded.  reference: 5.8)")
    ridx = [rjn.index(f"{s}_{b}_joint") for b in LEG for s in ("left", "right")
            if f"{s}_{b}_joint" in rjn]
    print(f"   {'reference clip':24s} {hp_rms(rdof[:, ridx]):6.1f} mrad")
    for tag, s in out.items():
        idx = [s["jn"].index(f"{sd}_{b}_joint") for b in LEG for sd in ("left", "right")
               if f"{sd}_{b}_joint" in s["jn"]]
        print(f"   {tag:24s} {hp_rms(s['tgt'][:, idx]):6.1f} mrad commanded,"
              f" {hp_rms(s['pos'][:, idx]):6.1f} achieved")

    # ---- 5. how hard the policy is pushing, and into what -------------------------
    print("\n5. ANKLE_ROLL COMMAND vs THE MECHANICAL STOP (+-15.0 deg)")
    for tag, s in out.items():
        for side in ("left", "right"):
            i = s["jn"].index(f"{side}_ankle_roll_joint")
            t = s["tgt"][:, i]
            past = (t < s["lo"][i]) | (t > s["hi"][i])
            print(f"   {tag:24s} {side:5s} target {t.min()*R2D:+6.1f}..{t.max()*R2D:+6.1f} deg,"
                  f" past the stop on {100*past.mean():4.1f}% of frames,"
                  f" |a| max {np.abs(s['act'][:, i]).max():5.1f} mean {np.abs(s['act'][:, i]).mean():4.1f}")

    print("\n6. HOW FAR EACH ROLLOUT GOT")
    for tag, s in out.items():
        print(f"   {tag:24s} {s['n']} of {len(rdof)} frames")


if __name__ == "__main__":
    main()
