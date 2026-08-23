"""How much does each site's perturbation actually move the ACTION, per unit of c?

The pre-registered screen equalises perturbations by RELATIVE PARAMETER NORM
(rho = c*||W||_F/sqrt(numel)), chosen so the ranking would not merely rank layer sizes.
It has the opposite failure. `action_out_proj/bias` has ||W||_F = 0.034 over 32 entries, so
c = 0.5 there is an absolute perturbation of 0.003, while the same c on `llm/mlp/linear`
(||W||_F = 193) is enormous -- and the oracle showed the bias site needs edits of 0.15-1.15
before behaviour changes at all.

So relative matching systematically under-probes sites whose parameters are small but whose
influence on the output is direct, which is exactly the site that admits a 100% repair.
This measures induced |dAction| per site so a screen can equalise by OUTPUT displacement --
the quantity the task actually sees -- instead.
"""
from __future__ import annotations

import argparse, json, pathlib
import numpy as np

import main as lm
from openpi_client import image_tools
from paired_probe import Probe
from ace_screen_v2 import SITES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", type=pathlib.Path, required=True)
    ap.add_argument("--ack", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--replan-steps", type=int, default=5)
    a = ap.parse_args()

    pr = Probe(a)
    env, desc, inits = pr.env_for(0)
    env.reset()
    obs = env.set_init_state(inits[8])
    for _ in range(12):
        obs, *_ = env.step(lm.LIBERO_DUMMY_ACTION)
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
    wr = image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
    el = {"observation/image": img, "observation/wrist_image": wr,
          "observation/state": np.concatenate((obs["robot0_eef_pos"],
              lm._quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
          "prompt": str(desc)}

    pr.control(dict(site=None, pin_rng=True))
    base = np.asarray(pr.client.infer(el)["actions"], float)
    rep = np.abs(np.asarray(pr.client.infer(el)["actions"], float) - base).max()
    print(f"pinned determinism: max|dA| on repeat = {rep:.2e}\n")
    print(f"{'site':<38} {'c':>5} {'rho':>10} {'mean|dA|':>10} {'|dA|/c':>10}")

    rows = []
    for s in SITES:
        pts = []
        for c in (0.1, 0.5):
            ack = pr.control(dict(site=s, seed=99, c_rel=c, pin_rng=True))
            d = float(np.abs(np.asarray(pr.client.infer(el)["actions"], float) - base).mean())
            pts.append(dict(c=c, rho=ack.get("rho"), dA=d))
            print(f"{s:<38} {c:>5} {ack.get('rho'):>10.3e} {d:>10.5f} {d/c:>10.5f}")
        rows.append(dict(site=s, points=pts, slope=float(np.mean([p["dA"] / p["c"] for p in pts]))))
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=1))

    sl = np.array([r["slope"] for r in rows])
    print(f"\n|dAction| per unit c spans {sl.min():.5f} .. {sl.max():.5f}"
          f"  -> {sl.max()/max(sl.min(), 1e-12):.0f}x across sites AT MATCHED RELATIVE NORM.")
    print("That spread is what the pre-registered screen was ranking.")


if __name__ == "__main__":
    main()
