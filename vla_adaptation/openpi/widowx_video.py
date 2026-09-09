"""Side-by-side WidowX (SimplerEnv) video: the same seed, frozen-faulted vs corrected.
Same style as compare_video.py / gr1_video.py: the same seed (same scene, SimplerEnv seeds
its scene from env.reset(seed)) in both panels, the task language and the fault on the
frame, the live estimate on the corrected panel, and with --only-repaired only pairs where
the frozen policy fails and the corrected run succeeds on this render. Frames are the
benchmark's own 256x256 third-person camera -- exactly what the policy sees; nothing the
policy sees is changed. The law is widowx_adapt.episode's, copied so frames can be kept.
"""
from __future__ import annotations

import argparse, collections, json, pathlib
import numpy as np
import imageio.v2 as iio
from PIL import Image, ImageDraw
import widowx_adapt as WA

BAR = 74


def annotate(img, header, color, lines):
    im = Image.fromarray(img).resize((384, 384), Image.BILINEAR)
    canvas = Image.new("RGB", (384, 384 + BAR), color)
    canvas.paste(im, (0, BAR))
    d = ImageDraw.Draw(canvas)
    d.text((6, 4), header, fill=(255, 255, 255))
    for i, ln in enumerate(lines):
        d.text((6, 20 + 13 * i), ln, fill=(255, 255, 255))
    return np.asarray(canvas)


def rollout(A, ep, W, M_inv, M, scale, fvec, adapt, corr, gamma, dead, norm_r, clip, horizon, max_steps):
    obs = A.reset(ep); p = A.pose(obs); prompt = str(obs[WA.LANG_KEY])
    hist = collections.deque([np.zeros(6)] * (WA.K_FIR + 1), maxlen=WA.K_FIR + 1)
    f_hat = np.zeros(6); m = np.isin(np.arange(6), corr).astype(float); sel = m > 0
    plan = collections.deque(); frames = []; success = False
    for t in range(max_steps):
        if not plan:
            plan.extend(np.asarray(A.client.infer(A.policy_obs(obs))["actions"], float)[:horizon])
        a_cmd = np.asarray(plan.popleft(), float)
        a_corr = a_cmd.copy(); a_corr[:6] += (-f_hat * m) if adapt else 0.0
        a_exec = a_corr.copy(); a_exec[:6] += fvec
        obs, r, term, trunc, info = A.step(a_exec)
        p1 = A.pose(obs); y = WA.increment(p1, p); p = p1
        hist.appendleft(a_corr[:6].copy()); H = np.array(hist)
        pred = np.array([W[j, :WA.K_FIR + 1] @ H[:, j] + W[j, -1] for j in range(6)])
        res = y / scale - pred
        if adapt:
            e = res - M @ (f_hat * m); ne = float(np.linalg.norm(e[sel]))
            step = np.zeros(6) if ne < dead else (M_inv @ e) / (1.0 + (ne / norm_r) ** 2)
            f_hat = np.clip(f_hat + gamma * step * m, -clip, clip)
        frames.append((np.asarray(obs["video.image_0"], np.uint8), f_hat.copy(), t))
        success = success or bool(info.get("success", False))
        if term or trunc or success:
            break
    return frames, success, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=WA.DEFAULT_TASK)
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8005)
    ap.add_argument("--log", type=pathlib.Path, required=True); ap.add_argument("--openloop", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--episodes", default="0,1,2,3"); ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--fault-vec", required=True); ap.add_argument("--corr-dims", default="0,1,2")
    ap.add_argument("--gamma", type=float, default=0.08); ap.add_argument("--dead", type=float, default=0.001)
    ap.add_argument("--norm-r", type=float, default=0.009); ap.add_argument("--clip", type=float, default=0.03)
    ap.add_argument("--horizon", type=int, default=8); ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--fps", type=int, default=10); ap.add_argument("--title", default="")
    ap.add_argument("--only-repaired", action="store_true"); ap.add_argument("--max-clips", type=int, default=0)
    a = ap.parse_args()

    A = WA.WidowX(a.task, a.host, a.port, a.seed)
    scale = np.array(json.loads(a.log.with_name(a.log.stem + "_scale.json").read_text())["scale"])
    W, _ = WA.fit_plant(a.log, scale); M = np.array(json.loads(a.openloop.read_text())["M"]); M_inv = np.linalg.pinv(M)
    fvec = np.array([float(x) for x in a.fault_vec.split(",")]); corr = [int(x) for x in a.corr_dims.split(",")]
    shown = corr[:3]
    clips, kept = [], 0
    for ep in [int(x) for x in a.episodes.split(",")]:
        if a.max_clips and kept >= a.max_clips:
            break
        fL, okL, prompt = rollout(A, ep, W, M_inv, M, scale, fvec, False, corr, a.gamma, a.dead, a.norm_r, a.clip, a.horizon, a.max_steps)
        print(f"  seed {a.seed+ep}: frozen success={okL} steps={len(fL)}", flush=True)
        if okL:
            print("    frozen succeeded this render -> skipping"); continue
        fR, okR, _ = rollout(A, ep, W, M_inv, M, scale, fvec, True, corr, a.gamma, a.dead, a.norm_r, a.clip, a.horizon, a.max_steps)
        print(f"  seed {a.seed+ep}: corrected success={okR} steps={len(fR)}", flush=True)
        if a.only_repaired and not okR:
            print("    corrected run failed this render -> skipping"); continue
        kept += 1
        n = max(len(fL), len(fR))
        for k in range(n + a.fps):
            i, j = min(k, len(fL) - 1), min(k, len(fR) - 1)
            imL, _, tL = fL[i]; imR, fh, tR = fR[j]
            L = annotate(imL, "FROZEN  (uncorrected)", (150, 30, 30),
                         [f'"{prompt}"', a.title, f"step {tL}",
                          "SUCCESS" if (okL and k >= len(fL) - 1) else ("FAILED - timeout" if (not okL and k >= len(fL) - 1) else "")])
            R = annotate(imR, "ADAPTIVE  (online)", (25, 110, 45),
                         [f'"{prompt}"', a.title,
                          f"step {tR}   f_hat[{','.join(WA.POSE[d] for d in shown)}] = " + " ".join(f"{fh[d]:+.4f}" for d in shown),
                          "SUCCESS" if (okR and k >= len(fR) - 1) else ("FAILED - timeout" if (not okR and k >= len(fR) - 1) else "")])
            clips.append(np.concatenate([L, R], axis=1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    iio.mimwrite(a.out, clips, fps=a.fps, quality=8)
    print(f"wrote {a.out} ({len(clips)} frames, {len(clips)/a.fps:.1f} s)")


if __name__ == "__main__":
    main()
