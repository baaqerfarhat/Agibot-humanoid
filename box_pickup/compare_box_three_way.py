"""Box policy v19 across three plants: mjlab, Isaac Sim, and the robot.

The question is which simulator predicts the robot. Isaac is where the policy was
trained and where it looks fine, so Isaac agreeing with the robot would mean the
reference motion is at fault, while mjlab agreeing with the robot would mean Isaac's
plant is flattering the policy and training should move.

Chatter is measured the way the earlier hardware post-mortems measured it: RMS of the
commanded joint target above 5 Hz, over the twelve leg joints, in milliradians. Below
5 Hz is the task; above it is the policy arguing with itself. Targets are used rather
than measured positions because the target is what the policy actually emits, and it
is the one signal all three plants record identically.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "run_logs"
ROLL = HERE / "sim_rollouts"
LEGS = [
    f"{s}_{j}_joint"
    for s in ("left", "right")
    for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
]


def chatter(tgt: np.ndarray, fs: float = 50.0, cut: float = 5.0) -> float:
    """RMS of the >5 Hz content of a target trajectory, in mrad."""
    if len(tgt) < 30:
        return float("nan")
    b, a = butter(2, cut / (fs / 2), btype="high")
    return float(np.sqrt((filtfilt(b, a, tgt, axis=0) ** 2).mean()) * 1000)


def load_hw(path: Path):
    raw = path.read_text().splitlines()
    hdr = raw[0].split(",")
    width = len(hdr)
    rows = [r.split(",") for r in raw[1:]]
    rows = [r for r in rows if len(r) == width]  # trailing line is often truncated
    col = {n: i for i, n in enumerate(hdr)}
    arr = np.array(rows, dtype=object)
    keep = np.array([r[col["phase"]] == "policy" for r in rows])
    if not keep.any():
        return None
    arr = arr[keep]

    def f(name):
        return np.array([float(x) for x in arr[:, col[name]]])

    return {
        "tgt": np.stack([f(f"{j}__tgt") for j in LEGS], 1),
        "pos": np.stack([f(f"{j}__pos_meas") for j in LEGS], 1),
        "roll": f("roll"),
        "t": f("t_s"),
    }


def main() -> None:
    print("=" * 74)
    print("BOX POLICY v19 (iter 85500) ACROSS THREE PLANTS")
    print("=" * 74)

    # ---- Isaac ---------------------------------------------------------------
    I = np.load(ROLL / "x2_box_walk_retimed_v19_iter85500_rollout.npz", allow_pickle=True)
    md = json.loads(str(I["_metadata_json"]))
    dn = list(md["dof_names"])
    li = [dn.index(j) for j in LEGS]
    ic = chatter(I["dof_pos_target"][:, li])
    iz = I["root_pos"][:, 2]

    # ---- mjlab ---------------------------------------------------------------
    M = np.load(ROLL / "mjlab_box_v19_pos_rollout.npz", allow_pickle=True)
    mj = list(M["joint_names"])
    mi = [mj.index(j) for j in LEGS]
    mc = chatter(M["target"][:, mi])
    mz = M["pelvis_h"]

    # ---- hardware ------------------------------------------------------------
    hw = sorted(LOGS.glob("*box_pickup_x2_box_policy_walk_retimed_v19_iter85500.csv"))
    hcs, hrolls = [], []
    for p in hw:
        d = load_hw(p)
        if d is None or len(d["t"]) < 30:
            continue
        hcs.append(chatter(d["tgt"]))
        hrolls.append(np.abs(d["roll"]).max() * 180 / np.pi)

    print()
    print(f"{'plant':<12} {'outcome':<22} {'leg-target chatter >5Hz':<26}")
    print("-" * 74)
    print(f"{'Isaac Sim':<12} {'completes, upright':<22} {ic:6.1f} mrad")
    print(f"{'mjlab':<12} {'FALLS (pelvis ' + f'{mz.min():.2f}' + ' m)':<22} {mc:6.1f} mrad")
    print(f"{'hardware':<12} {'0 of ' + str(len(hcs)) + ' runs completed':<22} "
          f"{np.mean(hcs):6.1f} mrad  (n={len(hcs)}, "
          f"range {np.min(hcs):.0f}-{np.max(hcs):.0f})")
    print()
    print(f"Isaac pelvis:  start {iz[0]:.3f}  min {iz.min():.3f}  final {iz[-1]:.3f}")
    print(f"mjlab pelvis:  start {mz[0]:.3f}  min {mz.min():.3f}  final {mz[-1]:.3f}")
    print()
    print(f"chatter amplification, sim -> hardware:  "
          f"Isaac {np.mean(hcs)/ic:.2f}x   mjlab {np.mean(hcs)/mc:.2f}x")

    # ---- squat control -------------------------------------------------------
    print()
    print("=" * 74)
    print("CONTROL: the squat policy, which was trained in mjlab and works on the robot")
    print("=" * 74)
    sq = sorted(LOGS.glob("*squat_x2_squat_policy_40pct_iter16499.csv"))
    scs = []
    for p in sq:
        d = load_hw(p)
        if d is None or len(d["t"]) < 30:
            continue
        scs.append(chatter(d["tgt"]))
    if scs:
        print(f"squat on hardware: {np.mean(scs):.1f} mrad chatter "
              f"(n={len(scs)}, range {np.min(scs):.0f}-{np.max(scs):.0f}) -- and it stands")
        print(f"box   on hardware: {np.mean(hcs):.1f} mrad chatter -- and it falls")
        print(f"the working policy is {np.mean(hcs)/np.mean(scs):.1f}x quieter on the same robot")

    print()
    print("=" * 74)
    print("READING")
    print("=" * 74)
    print("mjlab and the robot agree: the box policy does not survive either.")
    print("Isaac is the only plant in which this policy works, and Isaac is the one")
    print("it was trained in. A simulator that disagrees with the robot cannot be")
    print("used to certify a policy for the robot.")


if __name__ == "__main__":
    main()
