"""Render VLA image observations for an Isaac rollout, by mirroring its state into MuJoCo.

Neither simulator can do this alone on this machine, and the gap is not the same gap:

    Isaac    runs the box task correctly but CANNOT RENDER -- Omniverse misparses the
             driver and refuses the RTX renderer, so every annotator returns shape (0,).
    MuJoCo   renders fine, but holosoma's MuJoCo backend CANNOT HOST THE BOX: it
             registers the robot and nothing else (`scene_count=0, individual_count=0`
             in simulator/mujoco/mujoco.py), and its scene manager has no object path.

So neither is made to do the other's job. Isaac keeps the physics and the scoring;
MuJoCo is handed the resulting state and asked only to draw it. The MJCF's nu=0 stops
mattering the moment nothing is being actuated.

Every rollout already carries what this needs -- `eval_adapt_isaac._rollout` records
dof_pos, root_pos, root_quat_xyzw, object_pos and object_quat_wxyz at every control
step, and the npz metadata carries dof_names, so joints map BY NAME rather than by a
column order that is not guaranteed to match the MJCF's.

That means the 95 rollouts already under adaptation/isaac_runs/ can be turned into VLA
observations offline -- no Isaac boot, and no GPU contention with a training run.

    python3 mj_mirror.py adaptation/isaac_runs/baseline_noisy/isaac_frozen_seed600.npz

Needs mujoco, which is not in the hssim env:
    PYTHONPATH=/home/mtaheri/.holosoma_deps/mjrender MUJOCO_GL=egl python3 ...
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import mujoco

from mj_cameras import X2_LARGEBOX, load_with_cameras

CAMERAS = ("ego", "agentview")


class MuJoCoMirror:
    """Holds a rendering-only copy of the scene and poses it from external state."""

    def __init__(self, dof_names, res: int = 224, cameras=CAMERAS, xml: str = X2_LARGEBOX):
        self.model = load_with_cameras(xml)
        self.data = mujoco.MjData(self.model)
        self.cameras = tuple(cameras)
        self.res = res
        self._renderer = None

        m = self.model
        self.qadr = np.empty(len(dof_names), int)
        for i, n in enumerate(dof_names):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, str(n))
            if jid < 0:
                raise KeyError(f"{n} is not a joint of {Path(xml).name}")
            if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                raise TypeError(f"{n} is not a hinge joint")
            self.qadr[i] = m.jnt_qposadr[jid]

        # the box rides a free joint; it is the only unnamed one in this MJCF
        box_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "largebox_link")
        free = [j for j in range(m.njnt)
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_bodyid[j] == box_body]
        if len(free) != 1:
            raise RuntimeError(f"expected one free joint on largebox_link, found {len(free)}")
        self.box_adr = int(m.jnt_qposadr[free[0]])
        self.root_adr = int(m.jnt_qposadr[0])

    @property
    def renderer(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.res, width=self.res)
        return self._renderer

    def pose(self, dof_pos, root_pos, root_quat_xyzw, object_pos=None, object_quat_wxyz=None):
        """Write one control step's state into the model's qpos."""
        q = self.data.qpos
        q[self.root_adr:self.root_adr + 3] = root_pos
        # Isaac reports the root as xyzw; MuJoCo free joints are wxyz.
        x, y, z, w = root_quat_xyzw
        q[self.root_adr + 3:self.root_adr + 7] = (w, x, y, z)
        q[self.qadr] = dof_pos
        if object_pos is not None:
            q[self.box_adr:self.box_adr + 3] = object_pos
            q[self.box_adr + 3:self.box_adr + 7] = object_quat_wxyz
        mujoco.mj_forward(self.model, self.data)

    def render(self) -> dict:
        """RGB from every camera at the current pose."""
        out = {}
        for nm in self.cameras:
            self.renderer.update_scene(self.data, camera=nm)
            out[nm] = np.asarray(self.renderer.render()).copy()
        return out

    def render_episode(self, rec: dict, stride: int = 1) -> dict:
        """Render a whole rollout. Returns {camera: (N, res, res, 3) uint8}."""
        n = len(rec["dof_pos"])
        has_box = "object_pos" in rec and len(rec["object_pos"]) == n
        frames = {c: [] for c in self.cameras}
        for t in range(0, n, stride):
            self.pose(rec["dof_pos"][t], rec["root_pos"][t], rec["root_quat_xyzw"][t],
                      rec["object_pos"][t] if has_box else None,
                      rec["object_quat_wxyz"][t] if has_box else None)
            for c, img in self.render().items():
                frames[c].append(img)
        return {c: np.stack(v) for c, v in frames.items()}

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def load_rollout(path) -> tuple[dict, dict]:
    """Read a saved rollout npz into (records, metadata)."""
    d = np.load(path, allow_pickle=True)
    rec = {k: d[k] for k in d.files if k != "_metadata_json"}
    md = json.loads(str(d["_metadata_json"])) if "_metadata_json" in d.files else {}
    return rec, md


def render_rollout(path, res: int = 224, stride: int = 1, cameras=CAMERAS):
    """Convenience: saved rollout -> {camera: frames}, plus its metadata."""
    rec, md = load_rollout(path)
    names = md.get("dof_names")
    if not names:
        raise KeyError(f"{path} has no dof_names in metadata; cannot map joints by name")
    mirror = MuJoCoMirror(names, res=res, cameras=cameras)
    try:
        return mirror.render_episode(rec, stride=stride), md
    finally:
        mirror.close()


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("npz", type=Path, help="a rollout from adaptation/isaac_runs/")
    ap.add_argument("--out", type=Path, default=Path("."), help="where to write images")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--sheet", type=int, default=8, help="tiles in the contact sheet")
    ap.add_argument("--mp4", action="store_true", help="also write one mp4 per camera")
    a = ap.parse_args()

    frames, md = render_rollout(a.npz, res=a.res, stride=a.stride)
    a.out.mkdir(parents=True, exist_ok=True)
    stem = a.npz.stem
    keys = ("survival_steps", "box_present", "diverged", "mode", "seed")
    print(f"{stem}: " + "  ".join(f"{k}={md.get(k)}" for k in keys if k in md))

    import imageio.v2 as iio
    for cam, arr in frames.items():
        idx = np.linspace(0, len(arr) - 1, min(a.sheet, len(arr))).round().astype(int)
        iio.imwrite(a.out / f"{stem}_{cam}.png", np.concatenate(arr[idx], axis=1))
        if a.mp4:
            iio.mimwrite(a.out / f"{stem}_{cam}.mp4", arr, fps=int(md.get("fps", 50)))
        print(f"  {cam:<10} {arr.shape} mean {arr.mean():.1f} -> {stem}_{cam}.png")


if __name__ == "__main__":
    main()
