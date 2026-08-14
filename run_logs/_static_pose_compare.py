#!/usr/bin/env python3
"""E2 analysis: measured static torque minus what the model says the pose needs.

The robot held a pose and reported `eff_meas` per joint. MuJoCo evaluates the same
pose at zero velocity, where `qfrc_bias` is pure gravity, giving the torque the
URDF believes that pose requires. The difference is the load residual: mass, CoM,
or -- for the arms -- force that never made it through the soft hand.

Reading the result:

  * a large residual on the LEGS and WAIST, roughly constant across poses
      -> mass / CoM error. Static, so online adaptation can absorb it, and
         `--mask waist` has both a real error signal and spare authority.
  * a residual only on the ARMS
      -> the hand-box coupling, which no amount of waist adaptation fixes.
         Model the contact compliance in training instead.
  * residual near zero everywhere
      -> the model is fine and the run-log waist error is dynamic lag, i.e. a
         bandwidth problem. Adaptation will not help; do not spend robot time on it.

Run with the mjlab venv, which is the one that has mujoco:

  ~/baaqer_ws/mjlab/.venv/bin/python \
      run_logs/_static_pose_compare.py <static_pose_id_*.csv>

Hold with empty hands. The sim model is not gripping a box, so a box on the robot
would show up as a residual that is really just the box's weight. If you also want
the box load on its own, do a second run holding it and pass the empty-handed log
as --baseline; the subtraction cancels the model error and leaves the box.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HOLO = os.environ.get("HOLOSOMA_ROOT", "/home/baaqer/baaqer_ws/holosoma")
XML_BOX = os.path.join(HOLO, "src/holosoma_retargeting/holosoma_retargeting"
                             "/models/x2/x2_31dof_w_largebox.xml")
XML_PLAIN = os.path.join(HOLO, "src/holosoma_retargeting/holosoma_retargeting"
                               "/models/x2/x2_31dof.xml")
GROUPS = (("legs", ("hip", "knee", "ankle")),
          ("waist", ("waist",)),
          ("arms", ("shoulder", "elbow", "wrist")))


def _joint_adr(model):
    adr = {}
    for k in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, k)
        if nm and model.jnt_type[k] == mujoco.mjtJoint.mjJNT_HINGE:
            adr[nm] = (model.jnt_qposadr[k], model.jnt_dofadr[k])
    return adr


def sim_static_hold(xml, joint_names, q_by_name, kp, kd, settle_s=6.0, avg_s=0.5):
    """Torque the model needs to hold this pose while STANDING ON THE GROUND.

    A floating-base gravity bias will not do here: it reports the torque to hold
    each limb against gravity as if the pelvis were pinned in the air, which for
    the legs is a few Nm instead of the body weight they actually carry through
    the feet. So the sim repeats the hardware experiment -- same pose, same PD
    gains, robot on the floor -- and reads back the torque the controller had to
    apply once everything stopped moving.

    Returns (tau, quality) where quality is the residual joint speed; a large
    value means it never settled and the torque should not be trusted.
    """
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    adr = _joint_adr(model)
    idx = [adr[n] for n in joint_names]
    qadr = np.array([a for a, _ in idx])
    dadr = np.array([d for _, d in idx])
    goal = np.array([q_by_name[n] for n in joint_names])

    data.qpos[:] = model.qpos0
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qadr] = goal
    data.qpos[2] = 1.0
    mujoco.mj_forward(model, data)
    # Drop the base so the lowest point of the robot starts just above the floor;
    # gravity then seats the feet without a damaging fall.
    data.qpos[2] -= float(data.geom_xpos[1:, 2].min()) - 0.01

    n_settle = int(settle_s / model.opt.timestep)
    n_avg = int(avg_s / model.opt.timestep)
    acc = np.zeros(len(joint_names))
    speed = 0.0
    for i in range(n_settle):
        tau = kp * (goal - data.qpos[qadr]) - kd * data.qvel[dadr]
        data.qfrc_applied[dadr] = tau
        mujoco.mj_step(model, data)
        if i >= n_settle - n_avg:
            acc += tau
            speed = max(speed, float(np.abs(data.qvel[dadr]).max()))
    return acc / n_avg, speed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--xml", default=XML_PLAIN,
                    help="Robot model. The default has no box, which is why the hold "
                         "is done with empty hands: a box the sim is not holding "
                         "would show up as a residual that is really just the box.")
    ap.add_argument("--baseline", default=None,
                    help="A second hold log to subtract, e.g. the empty-handed run "
                         "when this one carried the box. The difference is the box "
                         "load alone, with every model error cancelled out.")
    args = ap.parse_args()

    pol = json.loads(str(np.load(
        os.path.join(REPO, "box_pickup/policy/x2_box_policy_clean_iter9000.npz"),
        allow_pickle=True)["meta_json"]))
    kp = kd = None

    def load(name):
        nonlocal kp, kd
        p = name if os.path.isabs(name) else os.path.join(HERE, name)
        rows = list(csv.DictReader(open(p)))
        meta = json.load(open(p.replace(".csv", ".meta.json")))
        jn = meta.get("joint_names") or pol["joint_names"]
        if kp is None:
            kp = np.array(meta.get("joint_stiffness", pol["joint_stiffness"]), float)
            kd = np.array(meta.get("joint_damping", pol["joint_damping"]), float)
        settle = float(meta.get("settle_s", 1.5))
        out = {}
        for ph in sorted({r["phase"] for r in rows if r["phase"].startswith("hold")},
                         key=lambda s: int(s[4:])):
            sel = [r for r in rows if r["phase"] == ph]
            t0 = float(sel[0]["t_s"])
            sel = [r for r in sel if float(r["t_s"]) - t0 >= settle] or sel
            tau = np.array([[float(r[f"{n}__eff_meas"]) for n in jn] for r in sel])
            q = {n: float(np.mean([float(r[f"{n}__pos_meas"]) for r in sel]))
                 for n in jn}
            out[ph] = (tau.mean(axis=0), q)
        if not out:
            raise SystemExit(f"{name}: no hold phases -- was --engage passed?")
        return jn, out

    jn, holds = load(args.csv)

    if args.baseline:
        # Two hardware runs, so the model is not involved and must not be printed
        # as if it were: the sim is empty-handed and cannot predict a box delta.
        base = load(args.baseline)[1]
        print(f"box load = {os.path.basename(args.csv)} - "
              f"{os.path.basename(args.baseline)}\n")
        for ph, (meas, _) in holds.items():
            if ph not in base:
                continue
            delta = meas - base[ph][0]
            print(f"  pose {ph[4:]}:  " + "   ".join(
                f"{g} {np.abs(delta[[i for i, n in enumerate(jn) if any(k in n for k in ks)]]).mean():.2f}"
                for g, ks in GROUPS) + " Nm")
            for i in np.argsort(-np.abs(delta))[:6]:
                print(f"      {jn[i]:28s}{delta[i]:+8.2f} Nm")
        print("\nCompare this against the box's weight through the grip. Much less "
              "than that\nmeans the hands are slipping or the box is resting on the "
              "forearms instead.")
        return

    print(f"model: {os.path.basename(args.xml)}")
    print(f"\n{'pose':>6s}{'group':>8s}{'measured':>11s}{'model':>10s}"
          f"{'residual':>11s}{'ratio':>8s}")
    per_pose = {}
    for ph, (meas, q) in holds.items():
        model_tau, speed = sim_static_hold(args.xml, jn, q, kp, kd)
        if speed > 0.2:
            print(f"  [warn] pose {ph[4:]}: sim never settled "
                  f"(max|qvel| {speed:.2f}); treat its model column as approximate")
        per_pose[ph] = (meas, model_tau)
        for gname, keys in GROUPS:
            idx = [i for i, n in enumerate(jn) if any(k in n for k in keys)]
            m, g = np.abs(meas[idx]).mean(), np.abs(model_tau[idx]).mean()
            print(f"{ph[4:]:>6s}{gname:>8s}{m:11.2f}{g:10.2f}{m - g:+11.2f}"
                  f"{m / max(g, 1e-6):8.2f}x")
        print()

    print("largest per-joint residuals (measured - model), averaged over poses:")
    res = np.nanmean([m - g for m, g in per_pose.values()], axis=0)
    for i in np.argsort(-np.abs(res))[:12]:
        print(f"    {jn[i]:28s}{res[i]:+8.2f} Nm")

    def group(keys):
        idx = [i for i, n in enumerate(jn) if any(k in n for k in keys)]
        nm = np.mean([np.abs(m[idx]).mean() for m, _ in per_pose.values()])
        md = np.mean([np.abs(g[idx]).mean() for _, g in per_pose.values()])
        return np.abs(res[idx]).mean(), nm / max(md, 1e-6)

    # A small joint needs a small absolute residual to be badly wrong, and a big
    # one can be off by a couple of Nm and still be fine, so judge on both.
    (w, wr), (l, lr), (a, ar) = (group(("waist",)),
                                 group(("hip", "knee", "ankle")),
                                 group(("shoulder", "elbow", "wrist")))
    print(f"\nmean |residual|   waist {w:.2f} ({wr:.2f}x)   legs {l:.2f} ({lr:.2f}x)"
          f"   arms {a:.2f} ({ar:.2f}x)")
    wp = jn.index("waist_pitch_joint")
    print(f"waist_pitch alone: residual {res[wp]:+.2f} Nm")
    # A small joint is badly wrong at a small absolute residual and a big one can
    # be off by a couple of Nm and still be fine, hence both tests. These cutoffs
    # are a first read; the per-joint table above is the actual result.
    big = lambda v, r: v > 3.0 or r > 1.2
    print("verdict:", end=" ")
    if big(w, wr) or big(l, lr):
        print("load residual in the waist/legs, and it is static, so waist "
              "adaptation has something real to absorb. Run the adaptation A/B.")
    elif big(a, ar):
        print("residual is confined to the arms -- this is the hand-box coupling. "
              "Fix it in training with contact compliance, not with adaptation.")
    else:
        print("no meaningful residual -- the model matches the robot, so the waist "
              "error in the run logs is dynamic lag. Adaptation will not help; skip it.")


if __name__ == "__main__":
    main()
