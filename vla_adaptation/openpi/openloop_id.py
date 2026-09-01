"""Open-loop replay: is the fault identifiable when NO policy reacts to it?

The closed-loop residual failed to recover the fault even with a plant model explaining 98%
of translation motion. Two explanations compete: the plant model is inadequate, or the
faulted data is off-distribution because the policy compensates and drives the arm through
different states.

Replay separates them. A recorded command sequence is played back open loop -- no policy, no
feedback -- with and without the fault added. The commands are then IDENTICAL by construction,
so any difference in the achieved motion is the fault propagating through the plant, and
d(motion)/d(fault) is measurable directly. That derivative is the map an adaptive law needs.

No inference is involved, so this runs on CPU in seconds.
"""
from __future__ import annotations

import argparse, json, pathlib
import numpy as np
import main as lm
from so3 import rot_delta
from libero.libero import benchmark

OUT = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])


def replay(env, inits, init_idx, cmds, f):
    env.reset()
    obs = env.set_init_state(inits[init_idx])
    for _ in range(10):
        obs, *_ = env.step(lm.LIBERO_DUMMY_ACTION)
    D, X = [], []
    for a in cmds:
        # the log kept only the 6 arm dims; the env expects 7, so re-attach the gripper
        # channel (held open -- it plays no part in an arm-dim additive fault)
        a = np.concatenate([np.asarray(a, float)[:6], [-1.0]])
        a[:6] += f
        x0 = np.array(obs["robot0_eef_pos"], float)
        q0 = np.array(obs["robot0_eef_quat"], float)
        obs, _, done, _ = env.step(a.tolist())
        x1 = np.array(obs["robot0_eef_pos"], float)
        q1 = np.array(obs["robot0_eef_quat"], float)
        D.append(np.concatenate([x1 - x0, rot_delta(q0, q1)]))
        X.append(np.concatenate([x1, lm._quat2axisangle(q1)]))
        if done:
            break
    return np.array(D), np.array(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--probe", type=float, default=0.02,
                    help="central-difference magnitude for M. The default 0.02 sits in "
                         "the linear region; the experiments inject 0.05, where "
                         "translation does not respond linearly. Setting this to the "
                         "operating point measures M where it is actually used.")
    a = ap.parse_args()

    d = json.loads(a.log.read_text())
    cmds = np.array(d[0]["raw_a"])[: a.steps]        # a nominal episode's commands
    suite = benchmark.get_benchmark_dict()[a.suite]()
    task = suite.get_task(0)
    env, desc = lm._get_libero_env(task, lm.LIBERO_ENV_RESOLUTION, 7)
    inits = suite.get_task_init_states(0)

    base, _ = replay(env, inits, 45, cmds, np.zeros(6))
    print(f"replayed {len(base)} steps open loop\n")
    print(f"{'fault f':>9} " + " ".join(f"{n:>9}" for n in ["dx", "dy", "dz", "drx", "dry", "drz"]))
    rows = []
    for f_mag in (0.01, 0.02, 0.05, -0.05):
        f = np.full(6, f_mag)
        D, _ = replay(env, inits, 45, cmds, f)
        n = min(len(D), len(base))
        dd = (D[:n] - base[:n]).mean(0) / OUT          # motion change, in action units
        rows.append(dict(f=f_mag, d_motion=dd.tolist(), sens=(dd / f_mag).tolist()))
        print(f"{f_mag:>9.3f} " + " ".join(f"{v:>9.4f}" for v in dd))
    print("\nsensitivity d(motion)/df, per unit fault  (1.0 = fault passes straight through):")
    for r in rows:
        print(f"{r['f']:>9.3f} " + " ".join(f"{v:>9.3f}" for v in r["sens"]))

    # The rows above perturb ALL six dims at once, so each column is a SUM over inputs, not
    # a sensitivity. The map an adaptive law needs is the 6x6 matrix: perturb one input axis
    # at a time and read the whole output response.
    print(f"\nSENSITIVITY MATRIX  M[out, in] = d(motion_out)/d(fault_in), f = +-{a.probe}, central:")
    M = np.zeros((6, 6))
    for j in range(6):
        acc = []
        for sgn in (+1.0, -1.0):
            f = np.zeros(6); f[j] = sgn * a.probe
            D, _ = replay(env, inits, 45, cmds, f)
            n = min(len(D), len(base))
            acc.append((D[:n] - base[:n]).mean(0) / OUT / (sgn * a.probe))
        M[:, j] = np.mean(acc, axis=0)
    hdr = ["dx", "dy", "dz", "drx", "dry", "drz"]
    print("        " + " ".join(f"{h:>8}" for h in hdr) + "   <- fault applied to")
    for i in range(6):
        print(f"{hdr[i]:>6}  " + " ".join(f"{M[i, j]:>8.3f}" for j in range(6)))
    off = np.abs(M - np.diag(np.diag(M))).sum() / max(np.abs(M).sum(), 1e-9)
    print(f"\noff-diagonal share of |M| = {off:.2f}   (0 = decoupled, per-axis gains suffice)")
    print(f"diagonal: {np.round(np.diag(M), 3)}")
    print(f"condition number of M = {np.linalg.cond(M):.1f}   (large = ill-posed to invert)")
    a.out.write_text(json.dumps({"rows": rows, "M": M.tolist(), "probe": a.probe}, indent=1))


if __name__ == "__main__":
    main()
