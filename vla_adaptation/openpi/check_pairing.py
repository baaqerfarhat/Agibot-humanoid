#!/usr/bin/env python3
"""Does matched (task, init) or a matched seed actually reproduce the same scene?

Mahdi Taheri's correction (commit 8c06a40) established that the RoboCasa GR1 arms were never
paired: the simulator re-randomises the manipulated object, its placement and the task language at
every reset, so 45 of 86 bodies move between two resets with the same seed. The paired McNemar
counts and p-values for that embodiment were invalid and were replaced with unpaired Fisher tests.

That raises the obvious follow-on question, which a reviewer will certainly ask: if pairing was
wrong on the humanoid, why is it trusted on LIBERO and ALOHA, where every headline p-value in the
paper is a paired exact McNemar?

This script answers it by measurement rather than by argument. Run each check and compare states
directly.

  python openpi/check_pairing.py libero  --telemetry <cell_telemetry.jsonl>
  python openpi/check_pairing.py aloha                       # needs gym_aloha, MUJOCO_GL=egl

Result on this project's artifacts, 2026-09-07:

  LIBERO   env.reset(); env.set_init_state(inits[init])  (adaptive_law.py:390)
           writes the FULL simulator state, so the scene cannot drift between arms.
           Measured: 20 matched (task, init) keys, max ||dpos|| = 0.000e+00, max ||dquat|| = 0.000e+00.
           PAIRING VALID.

  ALOHA    env.reset(seed=self.seed + ep)                (aloha_adapt.py:160)
           Measured: same seed reproduces all 23 qpos DOF bit-identically; across seeds 100-105
           exactly two DOF move (16 and 17, the cube's x and y, by ~0.10 each) while the arms start
           at a fixed home pose. So episodes are distinct scenes, matched across arms.
           PAIRING VALID.

  GR1      env.reset() only, with per-reset randomisation.  PAIRING INVALID -- use Fisher.

The distinguishing mechanism is whether the runner writes the initial state (LIBERO) or seeds a
generator that the environment consumes deterministically (ALOHA), versus letting the environment
randomise freely (RoboCasa). Seeding alone is not sufficient in general -- RoboCasa is seeded too
and still moves 45 of 86 bodies -- which is why this is measured per embodiment rather than assumed.
"""
import argparse, json, sys
from collections import defaultdict

import numpy as np


def check_libero(path):
    first = defaultdict(dict)
    for line in open(path):
        d = json.loads(line)
        if d.get("type") != "step" or d.get("t") != 0:
            continue
        first[(d["task"], d["init"])].setdefault(d["arm"], d)
    dp, dq = [], []
    for key, arms in sorted(first.items()):
        if len(arms) < 2:
            continue
        a, b = (arms[k] for k in sorted(arms))
        dp.append(np.linalg.norm(np.array(a["position"]) - np.array(b["position"])))
        dq.append(np.linalg.norm(np.array(a["quaternion"]) - np.array(b["quaternion"])))
    if not dp:
        print("no matched (task, init) keys with two arms found"); return 1
    print(f"{len(dp)} matched (task, init) keys across arms")
    print(f"  max ||d position||   = {max(dp):.3e}")
    print(f"  max ||d quaternion|| = {max(dq):.3e}")
    ok = max(dp) < 1e-9 and max(dq) < 1e-9
    print("VERDICT:", "PAIRING VALID -- scene is deterministic given (task, init)" if ok
          else "PAIRING INVALID -- arms start from different states")
    return 0 if ok else 1


def check_aloha(seeds=range(100, 106)):
    import gymnasium as gym
    import gym_aloha  # noqa: F401

    def qpos(seed):
        env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
        env.reset(seed=seed)
        q = env.unwrapped._env.physics.data.qpos.copy()
        env.close()
        return q

    seeds = list(seeds)
    same = max(np.abs(qpos(s) - qpos(s)).max() for s in seeds[:3])
    Q = np.array([qpos(s) for s in seeds])
    spread = Q.max(axis=0) - Q.min(axis=0)
    varying = np.flatnonzero(spread > 1e-9)
    print(f"repeat of the same seed: max |d qpos| = {same:.3e}  ({Q.shape[1]} DOF)")
    print(f"across seeds {seeds[0]}-{seeds[-1]}: {len(varying)} of {Q.shape[1]} DOF vary")
    for i in varying:
        print(f"    dof {i}: spread {spread[i]:.4e}")
    ok = same < 1e-9 and len(varying) > 0
    print("VERDICT:", "PAIRING VALID -- same seed reproduces the scene, different seeds move it"
          if ok else
          "PROBLEM -- " + ("same seed does not reproduce the scene" if same >= 1e-9
                           else "the seed does not vary the scene at all, so episodes are repeats"))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["libero", "aloha"])
    ap.add_argument("--telemetry")
    a = ap.parse_args()
    if a.which == "libero":
        if not a.telemetry:
            ap.error("--telemetry is required for the libero check")
        sys.exit(check_libero(a.telemetry))
    sys.exit(check_aloha())
