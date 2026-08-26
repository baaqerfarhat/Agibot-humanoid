#!/usr/bin/env python3
"""Export a holosoma box ckpt to npz and verify waist_pitch tracks +ref (not anti).

Pass criterion (perfect-obs open-loop), after mirroring the deploy HW clip:
  at motion frames where ref waist_pitch > 5 deg, the DEPLOYED target (raw
  action clipped to the +-18 deg hardware limit) must stay positive and within
  15 deg of the reference (sign + magnitude sanity).

Usage (holosoma conda env):

  python box_pickup/export_and_verify_waist.py \
      --checkpoint .../model_XXXXX.pt \
      --config     .../holosoma_config.yaml \
      --out        box_pickup/policy/x2_box_policy_v33.npz
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

R2D = 180.0 / math.pi
ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "agibot_control_functions" / "export_box_policy_npz.py"
MOTION = (
    Path("/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions")
    / "x2_31dof/whole_body_tracking/box_multispeed/sub3_largebox_003_mj_w_obj.npz"
)
if not MOTION.exists():
    MOTION = Path(
        "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/"
        "x2_31dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"
    )


def elu(x):
    return np.where(x > 0, x, np.exp(np.clip(x, -30, 0)) - 1)


def verify(npz_path: Path) -> bool:
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))
    jn = meta["joint_names"]
    wi = jn.index("waist_pitch_joint")
    mean, std = d["mean"], d["std"]
    Ws = [d[f"W{i}"] for i in range(int(d["n_layers"]))]
    bs = [d[f"b{i}"] for i in range(int(d["n_layers"]))]
    default = np.array(meta["default_joint_pos"], np.float32)
    ascale = np.array(meta["action_scale"], np.float32)
    ref_q = d["ref_joint_pos"]
    ref_v = d["ref_joint_vel"]
    ori6 = np.array([1, 0, 0, 1, 0, 0], np.float32)  # identity relative ori

    def act(obs):
        x = (obs - mean) / std
        for W, b in zip(Ws[:-1], bs[:-1]):
            x = elu(x @ W.T + b)
        return x @ Ws[-1].T + bs[-1]

    # Deploy clips every target to the hardware joint limit before it reaches
    # the robot (deploy_x2_box_pickup.py). waist_pitch HW limit is +-18 deg
    # (+-0.314 rad). Mirror that here so the gate scores the target the robot
    # ACTUALLY receives, not the raw pre-clip action. A raw +55 deg command
    # that clips to +18 deg == ref (mid-squat, ref pinned at the limit) is
    # benign; a -10 deg command in the hold (within limits, unclipped) is not.
    WAIST_HW_DEG = 18.0
    last = np.zeros(31, np.float32)
    print(f"{'f':>4} {'refW':>7} {'raw':>7} {'clip':>7} {'err':>7}")
    bad = 0
    checked = 0
    overshoot = 0
    for f in range(0, min(220, len(ref_q)), 5):
        obs = np.concatenate(
            [last, np.zeros(3, np.float32), ref_q[f] - default, ref_v[f], ref_q[f], ref_v[f], ori6]
        ).astype(np.float32)
        a = act(obs)
        last = a.astype(np.float32)
        tgt = a * ascale + default
        rw = ref_q[f, wi] * R2D
        tw_raw = tgt[wi] * R2D
        tw = float(np.clip(tw_raw, -WAIST_HW_DEG, WAIST_HW_DEG))  # deployed target
        flag = ""
        if abs(tw_raw) > WAIST_HW_DEG + 1.0:
            overshoot += 1
            flag = "  (raw>clip)"
        print(f"{f:4d} {rw:+7.1f} {tw_raw:+7.1f} {tw:+7.1f} {tw - rw:+7.1f}{flag}")
        # rw > 5 deg (was 10): the HOLD phase reference is only ~+8.6 deg but
        # the robot is bent forward under the box there -- a negative (lean
        # back) target is just as dangerous as in the deep squat, so it must be
        # verified too. The old 10 deg gate skipped the hold entirely and let a
        # -10 deg hold command slip through as a false PASS.
        if rw > 5.0:
            checked += 1
            # After the deploy clip: target must stay same sign as ref and
            # within 15 deg of it. Overshoot past the limit collapses onto
            # +18 deg == ref, so only genuine wrong-sign / large gaps fail.
            if tw < 0.0 or abs(tw - rw) > 15.0:
                bad += 1
    ok = checked > 0 and bad == 0
    print(
        f"\nwaist check (post-HW-clip): checked={checked} bad={bad} "
        f"raw_overshoot_frames={overshoot}  "
        f"waist_ascale={ascale[wi]:.3f} kp={meta['joint_stiffness'][wi]}"
    )
    print("PASS" if ok else "FAIL — do not deploy")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--motion", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-export", action="store_true")
    # The hybrid controller freezes the pickup here and splices its own carry walk
    # in. That only means anything for a clip with a hold phase: 211-311 was the
    # chest hold of the old in-place motion, and on the walking clip those frames
    # are mid-stride, so shipping the range would park the robot on one foot.
    # Absent it, the hybrid controller refuses to start rather than guessing.
    ap.add_argument("--hold-frames", type=int, nargs=2, default=None, metavar=("H0", "H1"))
    args = ap.parse_args()

    # Default to the clip the run ACTUALLY trained on, read from its own config.
    # This used to default to a hardcoded path; exporting a newer run with it
    # embeds the wrong reference trajectory in the deployed npz, so the robot
    # would track a motion the policy has never seen -- on hardware, not in a video.
    if args.motion is None:
        import yaml

        cfg = yaml.safe_load(open(args.config))
        rel = cfg["command"]["setup_terms"]["motion_command"]["params"]["motion_config"][
            "motion_file"
        ]
        cand = Path("/home/baaqer/baaqer_ws/holosoma/src/holosoma") / rel
        args.motion = str(cand if cand.exists() else MOTION)
        print(f"[export] motion from run config: {Path(args.motion).name}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_export:
        cmd = [
            sys.executable,
            str(EXPORT),
            "--checkpoint",
            args.checkpoint,
            "--config",
            args.config,
            "--motion",
            args.motion,
            "--out",
            str(out),
        ]
        if args.hold_frames:
            cmd += ["--hold-frames", str(args.hold_frames[0]), str(args.hold_frames[1])]
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd)

    ok = verify(out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
