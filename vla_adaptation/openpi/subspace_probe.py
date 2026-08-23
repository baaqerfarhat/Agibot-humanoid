"""Open-loop: how much signal does the task-relevant subspace recover?

LIBERO uses 7 of the model's 32 action dims; LiberoOutputs discards the rest as padding. A
perturbation spread over all 32 therefore puts 78% of its energy where it cannot matter.
This measures |dAction| for the same site, same rho, full-layer vs restricted to dims 0:7.
"""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np
import main as lm
from openpi_client import image_tools
from paired_probe import Probe

SITES = ["action_out_proj/bias", "action_out_proj/kernel", "action_in_proj/kernel"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", type=pathlib.Path, required=True)
    ap.add_argument("--ack", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--matched-c", type=pathlib.Path, required=True)
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--replan-steps", type=int, default=5)
    a = ap.parse_args()
    matched = json.loads(a.matched_c.read_text())

    pr = Probe(a)
    env, desc, inits = pr.env_for(0)
    env.reset(); obs = env.set_init_state(inits[8])
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
    out = []
    print(f"{'site':<26} {'dims':>6} {'seed':>5} {'|dA| all-32':>12} {'|dA| 0:7':>10} {'gain':>7}")
    for s in SITES:
        c = matched[s]["c"]
        for seed in (11, 22, 33):
            pr.control(dict(site=s, seed=seed, c_rel=c, pin_rng=True))
            d_all = float(np.abs(np.asarray(pr.client.infer(el)["actions"], float) - base).mean())
            pr.control(dict(site=s, seed=seed, c_rel=c, pin_rng=True, dims=7))
            d_sub = float(np.abs(np.asarray(pr.client.infer(el)["actions"], float) - base).mean())
            out.append(dict(site=s, seed=seed, c=c, dA_all=d_all, dA_sub=d_sub,
                            gain=d_sub / d_all if d_all > 0 else float("nan")))
            print(f"{s:<26} {'0:7':>6} {seed:>5} {d_all:>12.5f} {d_sub:>10.5f} "
                  f"{out[-1]['gain']:>7.2f}x")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))
    print("\nNote: at the SAME rho the restricted draw has 7/32 of the energy, so a gain")
    print("above sqrt(7/32) = 0.47 means the discarded dims were contributing nothing.")


if __name__ == "__main__":
    main()
