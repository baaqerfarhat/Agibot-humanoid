"""Side-by-side GR1 humanoid video: the same seed, frozen-faulted vs corrected.
Same style as compare_video.py / aloha_video.py: the same seed in both panels, and with the
env rng reseeded per reset (gr1_adapt.GR1.reset, Sec 32.15) the same scene and object too;
the task language and the fault on the frame, the live
joint-space estimate on the corrected panel, and with --only-repaired only pairs the frozen
policy fails and the corrected run succeeds on this render. Frames are the benchmark's own
256x256 egocentric render; nothing the policy sees is changed.

For the identify-then-hold scheme pass --f-init <held 29-vector> --freeze-after 0: the
corrected panel then applies the stationary correction from step 0, which is what episodes
1..n of that scheme do.
"""
from __future__ import annotations

import argparse, collections, json, pathlib
import numpy as np
import imageio.v2 as iio
from PIL import Image, ImageDraw
import gr1_adapt as GA

BAR = 74
FRAME_KEY = "video.ego_view_pad_res256_freq20"


def annotate(img, header, color, lines):
    im = Image.fromarray(img).resize((384, 384), Image.BILINEAR)
    canvas = Image.new("RGB", (384, 384 + BAR), color)
    canvas.paste(im, (0, BAR))
    d = ImageDraw.Draw(canvas)
    d.text((6, 4), header, fill=(255, 255, 255))
    for i, ln in enumerate(lines):
        d.text((6, 20 + 13 * i), ln, fill=(255, 255, 255))
    return np.asarray(canvas)


def rollout(A, ep, W, M_inv, M, fvec, adapt, corr, clip, gamma, dead, norm_r, law="innov",
            f_init=None, freeze_after=None, horizon=16, max_steps=720):
    obs = A.reset(ep); q = A.q(obs); prompt = str(obs[GA.LANG_KEY])
    hist = collections.deque([q.copy()] * (GA.K_FIR + 1), maxlen=GA.K_FIR + 1)
    f_hat = np.zeros(GA.NJ) if f_init is None else np.asarray(f_init, float).copy()
    m = np.isin(np.arange(GA.NJ), corr).astype(float); sel = m > 0
    plan = collections.deque(); frames = []; success = False
    for t in range(max_steps):
        if not plan:
            plan.extend(np.asarray(A.client.infer(A.policy_obs(obs))["actions"], float)[:horizon])
        a_cmd = np.asarray(plan.popleft(), float)
        a_corr = a_cmd + ((-f_hat * m) if adapt else 0.0)
        obs, r, term, trunc, info = A.step(a_corr + fvec)
        q1 = A.q(obs); q = q1
        hist.appendleft(a_corr.copy()); H = np.array(hist)
        pred = np.array([W[j, :GA.K_FIR + 1] @ H[:, j] + W[j, -1] for j in range(GA.NJ)])
        res = q1 - pred
        if adapt and (freeze_after is None or t < freeze_after):
            if law == "innov":
                e = res - M @ (f_hat * m); ne = float(np.linalg.norm(e[sel]))
                step = np.zeros(GA.NJ) if ne < dead else (M_inv @ e) / (1.0 + (ne / norm_r) ** 2)
                f_hat = np.clip(f_hat + gamma * step, -clip, clip)
            else:
                est = M_inv @ res; nr = float(np.linalg.norm(res[sel]))
                if nr < dead: est = np.zeros(GA.NJ)
                est = est / (1.0 + (nr / norm_r) ** 2)
                f_hat = np.clip(f_hat + gamma * (est - f_hat), -clip, clip)
        frames.append((np.asarray(obs[FRAME_KEY], np.uint8), f_hat.copy(), t))
        success = success or bool(info.get("success", False))
        if term or trunc or success:
            break
    return frames, success, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=GA.DEFAULT_TASK)
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8004)
    ap.add_argument("--log", type=pathlib.Path, required=True); ap.add_argument("--openloop", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--episodes", default="0,1,2,3"); ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--fault-vec", required=True, help="29 values or arm:right:0.10")
    ap.add_argument("--corr-joints", default="arm:right")
    ap.add_argument("--clip", type=float, default=0.2); ap.add_argument("--gamma", type=float, default=0.08)
    ap.add_argument("--dead", type=float, default=0.013); ap.add_argument("--norm-r", type=float, default=0.11)
    ap.add_argument("--law", choices=["legacy", "innov"], default="innov")
    ap.add_argument("--fps", type=int, default=20); ap.add_argument("--title", default="")
    ap.add_argument("--only-repaired", action="store_true"); ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--freeze-after", type=int, default=None)
    ap.add_argument("--f-init", default=None, help="29 comma-separated values (the held estimate)")
    ap.add_argument("--horizon", type=int, default=16); ap.add_argument("--max-steps", type=int, default=720)
    a = ap.parse_args()

    def joints(spec):
        return GA.ARM[spec.split(":")[1]] if spec.startswith("arm:") else [int(x) for x in spec.split(",")]

    def fault(spec):
        if spec.startswith("arm:"):
            _, side, mag = spec.split(":"); f = np.zeros(GA.NJ); f[GA.ARM[side]] = float(mag); return f
        return np.array([float(x) for x in spec.split(",")])

    A = GA.GR1(a.task, a.host, a.port, a.seed)
    W, _ = GA.fit_plant(a.log); M = np.array(json.loads(a.openloop.read_text())["M"]); M_inv = np.linalg.pinv(M)
    fvec = fault(a.fault_vec); corr = joints(a.corr_joints); shown = corr[:3]
    f_init = np.array([float(x) for x in a.f_init.split(",")]) if a.f_init else None
    scheme = "held correction (identified in one episode)" if a.freeze_after == 0 and f_init is not None else "online"
    clips, kept = [], 0
    for ep in [int(x) for x in a.episodes.split(",")]:
        if a.max_clips and kept >= a.max_clips:
            break
        fL, okL, prompt = rollout(A, ep, W, M_inv, M, fvec, False, corr, a.clip, a.gamma, a.dead, a.norm_r, a.law,
                                  horizon=a.horizon, max_steps=a.max_steps)
        print(f"  seed {a.seed+ep}: frozen success={okL} steps={len(fL)}", flush=True)
        if okL:
            print("    frozen succeeded this render -> skipping"); continue
        fR, okR, promptR = rollout(A, ep, W, M_inv, M, fvec, True, corr, a.clip, a.gamma, a.dead, a.norm_r, a.law,
                             f_init=f_init, freeze_after=a.freeze_after, horizon=a.horizon, max_steps=a.max_steps)
        print(f"  seed {a.seed+ep}: corrected success={okR} steps={len(fR)}", flush=True)
        if a.only_repaired and not okR:
            print("    corrected run failed this render -> skipping"); continue
        kept += 1
        # each panel carries ITS OWN task language: the scene (and the object) is redrawn at
        # every reset, so the two rollouts of a seed are different draws (Sec 32.10)
        lang = prompt.split(": ", 1)[-1]; langR = promptR.split(": ", 1)[-1]
        n = max(len(fL), len(fR))
        for k in range(n + a.fps):
            i, j = min(k, len(fL) - 1), min(k, len(fR) - 1)
            imL, _, tL = fL[i]; imR, fh, tR = fR[j]
            L = annotate(imL, "FROZEN  (uncorrected)", (150, 30, 30),
                         [f'"{lang}"', a.title, f"step {tL}",
                          "SUCCESS" if (okL and k >= len(fL) - 1) else ("FAILED - timeout" if (not okL and k >= len(fL) - 1) else "")])
            R = annotate(imR, f"ADAPTIVE  ({scheme})", (25, 110, 45),
                         [f'"{langR}"', a.title,
                          f"step {tR}   f_hat[j{','.join(map(str, shown))}] = " + " ".join(f"{fh[d]:+.3f}" for d in shown),
                          "SUCCESS" if (okR and k >= len(fR) - 1) else ("FAILED - timeout" if (not okR and k >= len(fR) - 1) else "")])
            clips.append(np.concatenate([L, R], axis=1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    iio.mimwrite(a.out, clips, fps=a.fps, quality=8)
    print(f"wrote {a.out} ({len(clips)} frames, {len(clips)/a.fps:.1f} s)")


if __name__ == "__main__":
    main()
