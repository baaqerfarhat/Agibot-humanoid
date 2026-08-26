"""A selection criterion that answers the question the pipeline actually asks.

ACE ranks layers by how much a random perturbation moves the task metric. That is not what
"which layer should I adapt?" needs, and the data shows it: `action_out_proj/bias`, the one
site with a verified 100% repair, ranks 5th of 8 by ACE, and 54% of ACE's between-site
variance is explained by how hard each site happened to be perturbed.

The repair here is a CONSTANT action offset. A parameter change can implement it only if its
effect on the action is itself roughly constant across observations. A bias on the output
projection does that exactly -- the same delta shifts every action identically. A trunk
layer's effect depends on the input, so no single setting of it can act as a constant offset,
however "important" that layer is.

So score two scale-free things per site, from forward passes only -- no rollouts:

  consistency = || mean_obs dA || / mean_obs || dA ||     in [0,1]
      1.0 = the same parameter change moves every observation's action identically
      0.0 = the effect averages away; it cannot act as a constant offset

  alignment   = |< mean_obs dA , u* >| / (|| mean_obs dA || || u* ||)
      how much of the consistent part points along the direction the repair needs

Both are ratios, so neither depends on the perturbation scale -- which is the defect that
made ACE incomparable across layers in the first place.
"""
from __future__ import annotations

import argparse, json, pathlib
import numpy as np
import main as lm
from openpi_client import image_tools
from paired_probe import Probe
from ace_screen_v2 import SITES

NORM = ("/home/mtaheri/.cache/openpi/openpi-assets/checkpoints/pi05_libero/"
        "assets/physical-intelligence/libero/norm_stats.json")


def repair_direction():
    """u*: the normalised-action shift that cancels a +0.05 env-space offset on arm dims."""
    n = json.load(open(NORM))
    n = n.get("norm_stats", n)["actions"]
    scale = (np.array(n["q99"][:7]) - np.array(n["q01"][:7])) / 2.0
    u = np.zeros(7)
    u[:6] = -0.05 / scale[:6]
    return u / np.linalg.norm(u)


def repair_magnitude():
    """||u*|| in normalised action units -- the size of the shift, not just its direction."""
    n = json.load(open(NORM))
    n = n.get("norm_stats", n)["actions"]
    scale = (np.array(n["q99"][:7]) - np.array(n["q01"][:7])) / 2.0
    u = np.zeros(7); u[:6] = -0.05 / scale[:6]
    return float(np.linalg.norm(u))


def element(obs, desc):
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
    ap.add_argument("--probes", type=int, default=4)
    a = ap.parse_args()
    matched = json.loads(a.matched_c.read_text())
    u = repair_direction()

    pr = Probe(a)
    pr.control(dict(site=None, pin_rng=True))
    els = []
    for tid in range(a.tasks):
        env, desc, inits = pr.env_for(tid)
        env.reset(); obs = env.set_init_state(inits[10])
        for _ in range(12):
            obs, *_ = env.step(lm.LIBERO_DUMMY_ACTION)
        plan = []
        for step in range(a.snaps * 15):
            el = element(obs, desc)
            if step % 15 == 0:
                els.append(el)
            if not plan:
                plan = list(pr.client.infer(el)["actions"][: a.replan_steps])
            obs, _, done, _ = env.step(np.asarray(plan.pop(0), float).tolist())
            if done:
                break
    print(f"{len(els)} observations; repair direction u* = {np.round(u,3)}\n")
    base = [np.asarray(pr.client.infer(e)["actions"], float)[:, :7].mean(0) for e in els]

    rows = []
    print(f"{'site':<34} {'consistency':>12} {'alignment':>10} {'|mean dA|':>10}")
    for s in SITES:
        if s not in matched:
            continue
        cons, alis, mags = [], [], []
        for seed in range(a.probes):
            pr.control(dict(site=s, seed=500 + seed, c_rel=matched[s]["c"], pin_rng=True, dims=7))
            dA = np.array([np.asarray(pr.client.infer(e)["actions"], float)[:, :7].mean(0) - b
                           for e, b in zip(els, base)])
            mean_v = dA.mean(0)
            mag_mean = float(np.linalg.norm(mean_v))
            mean_mag = float(np.linalg.norm(dA, axis=1).mean())
            cons.append(mag_mean / mean_mag if mean_mag else 0.0)
            alis.append(abs(float(mean_v @ u) / mag_mean) if mag_mean else 0.0)
            mags.append(mag_mean)
        rows.append(dict(site=s, consistency=float(np.mean(cons)), consistency_sd=float(np.std(cons)),
                         alignment=float(np.mean(alis)), mean_dA=float(np.mean(mags))))
        print(f"{s:<34} {np.mean(cons):>12.3f} {np.mean(alis):>10.3f} {np.mean(mags):>10.5f}")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=1))
    print("\nGround truth: action_out_proj/bias is the site with a verified 100% repair.")
    best = max(rows, key=lambda r: r["consistency"])
    print(f"top by consistency: {best['site']}")


if __name__ == "__main__":
    main()
