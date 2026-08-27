"""Side-by-side video: the same episode, frozen-faulted vs adaptively corrected.

Same task, same initial state, same policy, same fault. The only difference is whether the
adaptive law is running -- so the two panels are directly comparable frame by frame. The
right panel also shows the live fault estimate f_hat as it converges from zero.
"""
from __future__ import annotations

import argparse, collections, json, pathlib
import numpy as np
from PIL import Image, ImageDraw

import main as lm
from openpi_client import image_tools
from paired_probe import Probe
from gate_faults import apply_action_fault
from adaptive_law import fit_plant, OUT, K_FIR

BAR = 46


def annotate(img, title, colour, lines):
    im = Image.fromarray(img).resize((384, 384), Image.BILINEAR)
    canvas = Image.new("RGB", (384, 384 + BAR), colour)
    canvas.paste(im, (0, BAR))
    d = ImageDraw.Draw(canvas)
    d.text((8, 6), title, fill=(255, 255, 255))
    for i, ln in enumerate(lines):
        d.text((8, 20 + i * 12), ln, fill=(255, 255, 255))
    return np.asarray(canvas)


def rollout(pr, tid, init, sev, M_inv, W, gamma, adapt, dead, norm_r, clip, max_steps=220):
    env, desc, inits = pr.env_for(tid)
    env.reset(); obs = env.set_init_state(inits[init])
    plan, t = collections.deque(), 0
    hist = collections.deque([np.zeros(6)] * (K_FIR + 1), maxlen=K_FIR + 1)
    f_hat, frames = np.zeros(6), []
    done = False
    while t < max_steps + 10:
        if t < 10:
            obs, _, done, _ = env.step(lm.LIBERO_DUMMY_ACTION); t += 1; continue
        raw = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(raw, 224, 224))
        wr = image_tools.convert_to_uint8(image_tools.resize_with_pad(
            np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
        if not plan:
            plan.extend(pr.client.infer({
                "observation/image": img, "observation/wrist_image": wr,
                "observation/state": np.concatenate((obs["robot0_eef_pos"],
                    lm._quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                "prompt": str(desc)})["actions"][: pr.a.replan_steps])
        a_cmd = np.asarray(plan.popleft(), float)
        c = -f_hat if adapt else np.zeros(6)
        a_corr = a_cmd.copy(); a_corr[:6] += c
        a_exec = apply_action_fault(a_corr, "offset", sev, 6)
        x0 = np.array(obs["robot0_eef_pos"], float)
        r0 = lm._quat2axisangle(np.array(obs["robot0_eef_quat"], float))
        obs, _, done, _ = env.step(a_exec.tolist())
        x1 = np.array(obs["robot0_eef_pos"], float)
        r1 = lm._quat2axisangle(np.array(obs["robot0_eef_quat"], float))
        y = np.concatenate([x1 - x0, r1 - r0]) / OUT
        hist.appendleft(a_corr[:6])
        H = np.array(hist)
        pred = np.array([W[i, :K_FIR + 1] @ H[:, i] + W[i, -1] for i in range(6)])
        r = y - pred
        if adapt:
            est = M_inv @ r
            nr = float(np.linalg.norm(r))
            if nr < dead:
                est = np.zeros(6)
            est = est / (1.0 + (nr / norm_r) ** 2)
            f_hat = np.clip(f_hat + gamma * (est - f_hat), -clip, clip)
        frames.append((raw, f_hat.copy(), t))
        if done:
            break
        t += 1
    return frames, bool(done), str(desc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", type=pathlib.Path, required=True)
    ap.add_argument("--ack", type=pathlib.Path, required=True)
    ap.add_argument("--log", type=pathlib.Path, required=True)
    ap.add_argument("--openloop", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--episodes", default="4:45,8:45,9:45")
    ap.add_argument("--gamma", type=float, default=0.08)
    ap.add_argument("--dead", type=float, default=0.008)
    ap.add_argument("--norm-r", type=float, default=0.15)
    ap.add_argument("--clip", type=float, default=0.15)
    ap.add_argument("--sev", type=float, default=0.05)
    ap.add_argument("--fps", type=int, default=20)
    a = ap.parse_args()

    W = fit_plant(a.log)
    M_inv = np.linalg.pinv(np.array(json.loads(a.openloop.read_text())["M"]))
    pr = Probe(a)
    pr.control(dict(site=None, pin_rng=False))

    import imageio.v2 as iio
    clips = []
    for spec in a.episodes.split(","):
        tid, init = (int(x) for x in spec.split(":"))
        out = {}
        for adapt in (False, True):
            out[adapt] = rollout(pr, tid, init, a.sev, M_inv, W, a.gamma, adapt,
                                 a.dead, a.norm_r, a.clip)
            print(f"  task {tid} init {init}  adapt={adapt}  "
                  f"success={out[adapt][1]}  steps={len(out[adapt][0])}")
        (fL, okL, desc), (fR, okR, _) = out[False], out[True]
        n = max(len(fL), len(fR))
        for k in range(n + a.fps):                       # hold the last frame ~1 s
            i, j = min(k, len(fL) - 1), min(k, len(fR) - 1)
            imL, _, tL = fL[i]; imR, fh, tR = fR[j]
            L = annotate(imL, "FROZEN  (fault uncorrected)", (150, 30, 30),
                         [f"step {tL}", "SUCCESS" if (okL and k >= len(fL) - 1) else
                          ("FAILED" if (not okL and k >= len(fL) - 1) else "")])
            R = annotate(imR, "ADAPTIVE  (online correction)", (25, 110, 45),
                         [f"step {tR}   f_hat = [" + " ".join(f"{v:+.2f}" for v in fh[:3]) + " ...]",
                          "SUCCESS" if (okR and k >= len(fR) - 1) else
                          ("FAILED" if (not okR and k >= len(fR) - 1) else "")])
            clips.append(np.concatenate([L, R], axis=1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    iio.mimwrite(a.out, clips, fps=a.fps, quality=8)
    print(f"\nwrote {a.out}  ({len(clips)} frames, {len(clips)/a.fps:.1f}s)")


if __name__ == "__main__":
    main()
