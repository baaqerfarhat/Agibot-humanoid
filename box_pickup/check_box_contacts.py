"""List every contact the box is involved in, and how far the feet are from it.

The box climbs 12 cm during a zero-action rollout with no hand contact reported, so
something other than the hands is pushing it. This dumps the raw MuJoCo contact list
rather than the sensor view, since the sensors only watch the pairs they were told to.

    cd ~/baaqer_ws/mjlab && uv run python <this>
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch


def main() -> None:
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-BoxPickup-Flat-X2")
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()

    model = env.sim.mj_model
    data = env.sim.mj_data if hasattr(env.sim, "mj_data") else None
    gname = lambda i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom{i}"

    box_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box/box_geom")
    print(f"box geom id {box_gid}")
    print(f"box geom contype {model.geom_contype[box_gid]} "
          f"conaffinity {model.geom_conaffinity[box_gid]}")

    # Geometry check: is the box overlapping the feet at spawn?
    zero = torch.zeros(1, env.action_manager.total_action_dim)
    for step in range(120):
        env.step(zero)
        if step % 20 and step != 119:
            continue
        d = env.sim.data
        # mjwarp keeps everything batched; pull world 0 back to numpy.
        ncon = int(np.asarray(d.nacon).reshape(-1)[0])
        # mujoco_warp packs the pair as contact.geom (..., 2) rather than geom1/geom2.
        pairs = np.asarray(d.contact.geom).reshape(-1, 2)
        dist = np.asarray(d.contact.dist).reshape(-1)
        hits = []
        for c in range(min(ncon, len(pairs))):
            if box_gid in pairs[c]:
                other = pairs[c][1] if pairs[c][0] == box_gid else pairs[c][0]
                hits.append(f"{gname(int(other))}(d={dist[c] * 1e3:+.1f}mm)")
        boxz = env.scene["box"].data.root_link_pos_w[0, 2].item()
        print(f"step {step:4d}  boxZ {boxz:.3f}  ncon {ncon:3d}  "
              f"box contacts: {', '.join(hits) if hits else 'none'}")

    env.close()


if __name__ == "__main__":
    main()
