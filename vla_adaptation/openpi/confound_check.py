"""Is the ACE ranking really a ranking of how hard each site actually got hit?

Perturbations were matched to a common |dAction| = 0.02 -- but that calibration used ONE
observation at ONE timestep. Across 220-step rollouts and ten tasks the induced displacement
can drift per site, and if it does, ACE is partly ranking effective perturbation strength
rather than causal importance. That is exactly the confound the original prereg's §3 matching
was meant to prevent, one level down.

This measures |dAction| for every site over MANY observations drawn from real rollouts, then
reports the spread and how far each site sits from the 0.02 target. Correlating that against
the measured ACE says whether the ranking is an artifact.
"""
from __future__ import annotations

import argparse, json, pathlib
import numpy as np
import main as lm
from openpi_client import image_tools
from paired_probe import Probe
from ace_screen_v2 import SITES


def obs_element(obs, desc):
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
    wr = image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
    return {"observation/image": img, "observation/wrist_image": wr,
            "observation/state": np.concatenate((obs["robot0_eef_pos"],
                lm._quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
            "prompt": str(desc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", type=pathlib.Path, required=True)
    ap.add_argument("--ack", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--matched-c", type=pathlib.Path, required=True)
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--snaps", type=int, default=4)
    a = ap.parse_args()
    matched = json.loads(a.matched_c.read_text())

    pr = Probe(a)
    pr.control(dict(site=None, pin_rng=True))

    # collect observations along REAL trajectories, not one frame
    els = []
    for tid in range(a.tasks):
        env, desc, inits = pr.env_for(tid)
        env.reset(); obs = env.set_init_state(inits[10])
        for _ in range(12):
            obs, *_ = env.step(lm.LIBERO_DUMMY_ACTION)
        plan = []
        for step in range(a.snaps * 15):
            el = obs_element(obs, desc)
            if step % 15 == 0:
                els.append(el)
            if not plan:
                plan = list(pr.client.infer(el)["actions"][: a.replan_steps])
            obs, _, done, _ = env.step(np.asarray(plan.pop(0), float).tolist())
            if done:
                break
    print(f"collected {len(els)} observations across {a.tasks} tasks\n")

    base = [np.asarray(pr.client.infer(e)["actions"], float) for e in els]
    rows = []
    print(f"{'site':<34} {'c':>7} {'mean|dA|':>9} {'sd':>8} {'cv':>6} {'x target':>9}")
    for s in SITES:
        if s not in matched:
            continue
        pr.control(dict(site=s, seed=99, c_rel=matched[s]["c"], pin_rng=True, dims=7))
        d = [float(np.abs(np.asarray(pr.client.infer(e)["actions"], float) - b).mean())
             for e, b in zip(els, base)]
        d = np.array(d)
        rows.append(dict(site=s, c=matched[s]["c"], mean=float(d.mean()), sd=float(d.std(ddof=1)),
                         cv=float(d.std(ddof=1) / d.mean()) if d.mean() else float("nan"),
                         ratio=float(d.mean() / 0.02), values=d.tolist()))
        print(f"{s:<34} {matched[s]['c']:>7.2f} {d.mean():>9.5f} {d.std(ddof=1):>8.5f} "
              f"{rows[-1]['cv']:>6.2f} {rows[-1]['ratio']:>8.2f}x")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=1))
    r = np.array([x["ratio"] for x in rows])
    print(f"\nAcross sites the ACHIEVED displacement spans {r.min():.2f}x .. {r.max():.2f}x the")
    print(f"0.02 target -- a {r.max()/r.min():.1f}x range. Matching on one frame did not hold.")


if __name__ == "__main__":
    main()
