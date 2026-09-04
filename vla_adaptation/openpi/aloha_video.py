"""Side-by-side ALOHA video: the same episode, frozen-faulted vs adaptively corrected.

Same style as compare_video.py for LIBERO -- identical seed in both panels, the prompt and
fault on the frame, the live joint-space estimate on the corrected panel, and (with
--only-frozen-fail) only episodes the frozen policy actually fails. Frames come from
gym_aloha's own renderer; nothing the policy sees is changed.
"""
from __future__ import annotations
import argparse, collections, json, pathlib
import numpy as np
import imageio.v2 as iio
from PIL import Image, ImageDraw
import aloha_adapt as AA

BAR = 74


def annotate(img, header, color, lines):
    h, w = img.shape[:2]
    canvas = Image.new("RGB", (w, h + BAR), color)
    canvas.paste(Image.fromarray(img), (0, BAR))
    d = ImageDraw.Draw(canvas)
    d.text((6, 4), header, fill=(255, 255, 255))
    for i, ln in enumerate(lines):
        d.text((6, 20 + 13 * i), ln, fill=(255, 255, 255))
    return np.asarray(canvas)


def rollout(A, ep, W, M_inv, fvec, adapt, corr, clip, gamma, dead, norm_r, f_init=None, freeze_after=None):
    """aloha_adapt.episode, but keeping every rendered frame."""
    obs = A.reset(ep); q = np.asarray(obs["agent_pos"], float)
    # History initialised at the CURRENT joint position (holding), not zeros. Zeros are a
    # valid history for delta commands (LIBERO) but mean target = 0 rad here: for the first
    # K_FIR steps the prediction was wildly wrong (|r| = 1.4), the normaliser zeroed every
    # update, and f_hat decayed toward zero by gamma per step -- a 1.6 cm dip at the start
    # of every episode, cold or warm (Sec 27.7).
    hist = collections.deque([q.copy()] * (AA.K_FIR + 1), maxlen=AA.K_FIR + 1)
    f_hat = np.zeros(AA.NJ) if f_init is None else np.asarray(f_init, float).copy()
    plan = collections.deque(); frames = []; success = False
    m = np.isin(np.arange(AA.NJ), corr).astype(float)
    for t in range(300):
        if not plan:
            plan.extend(np.asarray(A.client.infer(A.policy_obs(obs))["actions"], float)[:AA.HORIZON])
        a_cmd = np.asarray(plan.popleft(), float)
        a_corr = a_cmd + ((-f_hat * m) if adapt else 0.0)
        obs, r, term, trunc, info = A.env.step(a_corr + fvec)
        q1 = np.asarray(obs["agent_pos"], float)
        hist.appendleft(a_corr.copy()); H = np.array(hist)
        pred = np.array([W[j, :AA.K_FIR + 1] @ H[:, j] + W[j, -1] for j in range(AA.NJ)])
        res = q1 - pred
        if adapt and (freeze_after is None or t < freeze_after):
            est = M_inv @ res; nr = float(np.linalg.norm(res))
            if nr < dead: est = np.zeros(AA.NJ)
            est = est / (1.0 + (nr / norm_r) ** 2)
            f_hat = np.clip(f_hat + gamma * (est - f_hat), -clip, clip)
        frames.append((A.env.render(), f_hat.copy(), t))
        success = success or bool(info.get("is_success", False)) or (r >= 4)
        q = q1
        if term or trunc or success: break
    return frames, success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--log", type=pathlib.Path, required=True); ap.add_argument("--openloop", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--episodes", default="0,1,2,3", help="episode seeds (offsets from --seed)")
    ap.add_argument("--seed", type=int, default=200)
    ap.add_argument("--fault-vec", required=True); ap.add_argument("--corr-joints", default="0,1,2,3,4,5")
    ap.add_argument("--clip", type=float, default=0.15); ap.add_argument("--gamma", type=float, default=0.08)
    ap.add_argument("--dead", type=float, default=0.002); ap.add_argument("--norm-r", type=float, default=0.05)
    ap.add_argument("--fps", type=int, default=25); ap.add_argument("--title", default="")
    ap.add_argument("--only-frozen-fail", action="store_true")
    ap.add_argument("--freeze-after", type=int, default=None)
    ap.add_argument("--f-init", default=None, help="14 comma-separated values: start the corrected panel at a converged estimate (warm start)")
    a = ap.parse_args()
    A = AA.Aloha(a.host, a.port, a.seed)
    W, _ = AA.fit_plant(a.log); M_inv = np.linalg.pinv(np.array(json.loads(a.openloop.read_text())["M"]))
    fvec = np.array([float(x) for x in a.fault_vec.split(",")]); corr = [int(x) for x in a.corr_joints.split(",")]
    shown = [j for j in corr][:3]
    f_init = np.array([float(x) for x in a.f_init.split(',')]) if a.f_init else None
    clips = []
    for ep in [int(x) for x in a.episodes.split(",")]:
        fL, okL = rollout(A, ep, W, M_inv, fvec, False, corr, a.clip, a.gamma, a.dead, a.norm_r)
        print(f"  ep {ep}: frozen success={okL} steps={len(fL)}")
        if a.only_frozen_fail and okL:
            print("    frozen succeeded -> skipping"); continue
        fR, okR = rollout(A, ep, W, M_inv, fvec, True, corr, a.clip, a.gamma, a.dead, a.norm_r, f_init=f_init, freeze_after=a.freeze_after)
        print(f"  ep {ep}: adaptive success={okR} steps={len(fR)}")
        n = max(len(fL), len(fR))
        for k in range(n + a.fps):
            i, j = min(k, len(fL) - 1), min(k, len(fR) - 1)
            imL, _, tL = fL[i]; imR, fh, tR = fR[j]
            L = annotate(imL, "FROZEN  (uncorrected)", (150, 30, 30),
                         [f'"{AA.PROMPT}"', a.title, f"step {tL}",
                          "SUCCESS" if (okL and k >= len(fL) - 1) else ("FAILED" if (not okL and k >= len(fL) - 1) else "")])
            R = annotate(imR, "ADAPTIVE  (online)", (25, 110, 45),
                         [f'"{AA.PROMPT}"', a.title,
                          f"step {tR}   f_hat[j{','.join(map(str, shown))}] = " + " ".join(f"{fh[d]:+.3f}" for d in shown),
                          "SUCCESS" if (okR and k >= len(fR) - 1) else ("FAILED" if (not okR and k >= len(fR) - 1) else "")])
            clips.append(np.concatenate([L, R], axis=1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    iio.mimwrite(a.out, clips, fps=a.fps, quality=8)
    print(f"wrote {a.out} ({len(clips)} frames, {len(clips)/a.fps:.1f} s)")


if __name__ == "__main__":
    main()
