"""Reproduce the ego-camera choice in mj_cameras.py, and re-check it if anything moves.

Scores candidate mounts on SEGMENTATION RENDERS over the real task motion -- what actually
lands on the 224x224 image -- because the two cheaper criteria both mislead here:

  frustum-only   likes mounting the camera on top of the skull; the geometry it then
                 stares into is its own head, which a frustum test cannot see.
  hands-in-view  unachievable by any mount: the hands grip the box, so the box occludes
                 them for most of the task. `--rays` shows this directly.

Usage (mujoco is not in the hssim env):
    PYTHONPATH=/home/mtaheri/.holosoma_deps/mjrender MUJOCO_GL=egl \
        python3 tune_ego_camera.py [--rays] [--sheet]

Columns: seen  = fraction of sampled frames with the box on screen
         box%  = mean share of pixels that are the box
         self% = mean share that is the robot's own body (wasted frame)
         clip  = share of box pixels touching the image border (cut off)
         off   = box centroid distance from image centre, 0 centred, 0.71 corner
Ranking adds one term those columns do not show; see `rank`.
"""
from __future__ import annotations

import argparse

import numpy as np
import mujoco

from mj_cameras import X2_LARGEBOX, EGO_POS, EGO_PITCH_DEG, EGO_FOVY, add_vla_cameras, _look_quat
from mj_motion import load_motion, qpos_at, check_order, phases

TARGETS = ["largebox_link", "left_hand_contact_link", "right_hand_contact_link"]


def _build(pos, pitch, fovy):
    spec = mujoco.MjSpec.from_file(X2_LARGEBOX)
    add_vla_cameras(spec, ego_pitch_deg=pitch, ego_pos=pos, ego_fovy=fovy)
    return spec.compile()


def score(pos, pitch, fovy, mo, frames, res=224):
    """Render the task and measure the ego image. Returns a metrics dict."""
    m = _build(pos, pitch, fovy)
    d = mujoco.MjData(m)
    box = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "largebox_link")
    body_of = np.append(m.geom_bodyid, -1)          # index -1 -> sentinel for background
    bf, sf, cf, of = [], [], [], []
    with mujoco.Renderer(m, height=res, width=res) as r:
        r.enable_segmentation_rendering()
        for t in frames:
            d.qpos[:] = qpos_at(mo, t)
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera="ego")
            seg = np.asarray(r.render())[:, :, 0]   # geom id per pixel, -1 = background
            gb = body_of[np.where(seg >= 0, seg, -1)]
            isbox = gb == box
            bf.append(isbox.mean())
            sf.append(((gb >= 0) & ~isbox & (gb != 0)).mean())   # body 0 = world/floor
            if isbox.any():
                edge = isbox[0].sum() + isbox[-1].sum() + isbox[:, 0].sum() + isbox[:, -1].sum()
                cf.append(edge / isbox.sum())
                ys, xs = np.nonzero(isbox)
                of.append(float(np.hypot(ys.mean() / res - 0.5, xs.mean() / res - 0.5)))
            else:
                cf.append(0.0)
                of.append(0.71)
    bf, sf, cf, of = map(np.asarray, (bf, sf, cf, of))
    return dict(pos=tuple(pos), pitch=pitch, fovy=fovy, seen=float((bf > 0.005).mean()),
                box=float(bf.mean()), self_=float(sf.mean()),
                clip=float(cf.mean()), off=float(of.mean()))


def rank(r):
    """Box on screen, centred, uncut, and neither buried in the robot nor swallowing it.

    The last term is the one a naive score misses. A VLA needs the surroundings as well as
    the object -- where the box is going, what else is on the floor -- so a mount that
    fills the frame with the carried box is worse than its centring alone suggests. Past a
    third of the image the box is crowding everything else out, and that is penalised.
    Without this the score prefers pitch 55, whose carry frames are nearly all box.
    """
    crowd = max(0.0, r["box"] - 1 / 3)
    return r["seen"] - 0.6 * r["self_"] - 0.3 * r["clip"] - 0.4 * r["off"] - 0.8 * crowd


def ray_visibility(mo, frames, pos=EGO_POS, pitch=EGO_PITCH_DEG, fovy=EGO_FOVY):
    """True (occlusion-aware) visibility of box and hands -- why hands cannot be a target."""
    m = _build(pos, pitch, fovy)
    d = mujoco.MjData(m)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in TARGETS]
    ty = np.tan(np.deg2rad(fovy) / 2)               # square image, so fovx == fovy
    gid = np.zeros(1, np.int32)
    seen = np.zeros(len(TARGETS))
    for t in frames:
        d.qpos[:] = qpos_at(mo, t)
        mujoco.mj_forward(m, d)
        cp, cR = d.cam_xpos[cam], d.cam_xmat[cam].reshape(3, 3)
        for k, b in enumerate(bid):
            rel = d.xpos[b] - cp
            pc = cR.T @ rel
            depth = -pc[2]
            if depth <= 0.05 or abs(pc[0]) > ty * depth or abs(pc[1]) > ty * depth:
                continue
            dist = np.linalg.norm(rel)
            mujoco.mj_ray(m, d, cp, rel / dist, None, 1, -1, gid)
            # Visible iff the first geom the ray meets belongs to the target body. A
            # distance test fails: a ray at a 0.6 m box's centre hits its face 0.3 m early.
            if gid[0] >= 0 and m.geom_bodyid[gid[0]] == b:
                seen[k] += 1
    return dict(zip(TARGETS, (seen / max(len(frames), 1)).round(3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=24, help="frame subsample for scoring")
    ap.add_argument("--rays", action="store_true", help="report true target visibility")
    ap.add_argument("--sheet", action="store_true", help="write ego_best.png at task phases")
    a = ap.parse_args()

    mo = load_motion()
    check_order(_build(EGO_POS, EGO_PITCH_DEG, EGO_FOVY), mo)
    frames = list(range(0, len(mo["jp"]), a.stride))
    print(f"scoring on {len(frames)} frames of the box-pickup clip")

    cands = [(EGO_POS, EGO_PITCH_DEG, EGO_FOVY), ((0.12, 0.0, 0.06), 35.0, 80.0)]  # tuned, previous
    for pitch in (40.0, 45.0, 50.0, 55.0):
        for pos in ((0.10, 0.0, 0.04), (0.12, 0.0, 0.04), (0.12, 0.0, 0.0), (0.15, 0.0, 0.04)):
            if (pos, pitch, 80.0) not in cands:
                cands.append((pos, pitch, 80.0))

    rows = sorted((score(p, pt, f, mo, frames) for p, pt, f in cands), key=rank, reverse=True)
    print(f"{'seen':>5} {'box%':>6} {'self%':>6} {'clip':>5} {'off':>5}  pitch fovy  pos")
    for r in rows:
        mark = "  <- mj_cameras.py" if (r["pos"], r["pitch"]) == (tuple(EGO_POS), EGO_PITCH_DEG) else ""
        print(f"{r['seen']:5.2f} {100*r['box']:6.1f} {100*r['self_']:6.1f} {r['clip']:5.2f}"
              f" {r['off']:5.2f}  {r['pitch']:5.0f} {r['fovy']:4.0f}  "
              f"[{r['pos'][0]:.2f},0,{r['pos'][2]:+.2f}]{mark}")
    top = rank(rows[0])
    mine = next(rank(r) for r in rows if (r["pos"], r["pitch"]) == (tuple(EGO_POS), EGO_PITCH_DEG))
    if top - mine > 0.02:   # the leading mounts sit within ~0.03 of each other; ignore noise
        print(f"\nNOTE: {rows[0]['pos']} pitch {rows[0]['pitch']:.0f} now scores {top:.3f} vs "
              f"{mine:.3f} for the committed mount -- worth re-checking mj_cameras.py.")

    if a.rays:
        print("\ntrue visibility at the committed mount (occlusion-aware):")
        for k, v in ray_visibility(mo, frames).items():
            print(f"  {k:<26} {v}")
        print("  hands stay low because they hold the box -- it is in front of them.")

    if a.sheet:
        import imageio.v2 as iio
        m = _build(EGO_POS, EGO_PITCH_DEG, EGO_FOVY)
        d = mujoco.MjData(m)
        tiles = []
        with mujoco.Renderer(m, height=224, width=224) as r:
            for t in phases(mo).values():
                d.qpos[:] = qpos_at(mo, t)
                mujoco.mj_forward(m, d)
                r.update_scene(d, camera="ego")
                tiles.append(np.asarray(r.render()))
        iio.imwrite("ego_best.png", np.concatenate(tiles, axis=1))
        print("\nwrote ego_best.png:", " ".join(phases(mo)))


if __name__ == "__main__":
    main()
