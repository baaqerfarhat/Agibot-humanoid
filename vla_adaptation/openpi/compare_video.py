"""Side-by-side video: the same episode, frozen-faulted vs adaptively corrected.

Same task, same initial state, same policy, same fault. The only difference is whether the
adaptive law is running -- so the two panels are directly comparable frame by frame. The
right panel also shows the live fault estimate f_hat as it converges from zero.
"""
from __future__ import annotations

import argparse, collections, json, pathlib
import numpy as np
import textwrap
from PIL import Image, ImageDraw

import pathlib as _pl
import main as lm
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from paired_probe import Probe
from gate_faults import apply_action_fault
from adaptive_law import fit_plant, OUT, K_FIR

BAR = 74


def wide_env(task, resolution, seed, rec_cam, fovy):
    """Env with an EXTRA recording camera, widened.

    The policy must keep seeing the standard `agentview` at its trained fov -- widening that
    would change the policy's input and invalidate the comparison -- so the video is shot
    from a separate camera whose fovy we are free to open up.
    """
    bddl = _pl.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution,
                             camera_widths=resolution,
                             camera_names=["agentview", "robot0_eye_in_hand", rec_cam])
    env.seed(seed)
    return env, task.language


def set_fovy(env, cam, fovy):
    sim = env.env.sim if hasattr(env, "env") else env.sim
    sim.model.cam_fovy[sim.model.camera_name2id(cam)] = fovy


def annotate(img, title, colour, lines, prompt=""):
    im = Image.fromarray(img).resize((384, 384), Image.BILINEAR)
    canvas = Image.new("RGB", (384, 384 + BAR), colour)
    canvas.paste(im, (0, BAR))
    d = ImageDraw.Draw(canvas)
    d.text((8, 5), title, fill=(255, 255, 255))
    y = 18
    wrapped = textwrap.wrap(prompt, 62)[:2]           # the task the policy was given
    for j, ln in enumerate(wrapped):                  # quote the phrase, not each line
        txt = ('"' if j == 0 else " ") + ln + ('"' if j == len(wrapped) - 1 else "")
        d.text((8, y), txt, fill=(215, 215, 215)); y += 11
    for ln in lines:
        d.text((8, y), ln, fill=(255, 255, 255)); y += 12
    return np.asarray(canvas)


def rollout(pr, tid, init, sev, M_inv, W, gamma, adapt, dead, norm_r, clip, max_steps=None,
            rec_cam="agentview", rec_fovy=None, corr_dims=None):
    import paired_probe as _pp
    max_steps = max_steps or _pp.MAXS      # the suite's cap, not a spatial-only 220
    env, desc, inits = pr.env_for(tid)
    env.reset()
    if rec_fovy:
        set_fovy(env, rec_cam, rec_fovy)          # re-apply: reset can reload the model
    obs = env.set_init_state(inits[init])
    plan, t = collections.deque(), 0
    hist = collections.deque([np.zeros(6)] * (K_FIR + 1), maxlen=K_FIR + 1)
    f_hat, frames = np.zeros(6), []
    done = False
    while t < max_steps + 10:
        if t < 10:
            obs, _, done, _ = env.step(lm.LIBERO_DUMMY_ACTION); t += 1; continue
        key = f"{rec_cam}_image" if f"{rec_cam}_image" in obs else "agentview_image"
        raw = np.ascontiguousarray(obs[key][::-1, ::-1])
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
        if corr_dims is not None:
            m = np.zeros(6); m[list(corr_dims)] = 1.0
            c = c * m
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
    ap.add_argument("--rec-cam", default="agentview", help="camera used for the VIDEO only")
    ap.add_argument("--rec-fovy", type=float, default=None, help="widen it (45 = default)")
    ap.add_argument("--title", default="")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--corr-dims", default=None,
                    help="channels to correct, e.g. 3,4,5 for rotation only. The "
                         "earlier videos corrected all six, the configuration Sec 7.3 "
                         "shows is wrong for a uniform fault.")
    ap.add_argument("--only-frozen-fail", action="store_true",
                    help="skip episodes the frozen arm happens to pass")
    a = ap.parse_args()

    W = fit_plant(a.log)
    M_inv = np.linalg.pinv(np.array(json.loads(a.openloop.read_text())["M"]))
    cdims = [int(x) for x in a.corr_dims.split(',')] if a.corr_dims else None
    if cdims is not None:
        print(f'correction restricted to dims {cdims}')
    pr = Probe(a)
    pr.control(dict(site=None, pin_rng=False))
    if a.rec_fovy:                      # rebuild envs with the extra recording camera
        from libero.libero import benchmark as _bm
        suite = _bm.get_benchmark_dict()[a.suite]()
        pr._envs = {}
        for spec in a.episodes.split(","):
            tid = int(spec.split(":")[0])
            task = suite.get_task(tid)
            env, desc = wide_env(task, lm.LIBERO_ENV_RESOLUTION, 7, a.rec_cam, a.rec_fovy)
            set_fovy(env, a.rec_cam, a.rec_fovy)
            pr._envs[tid] = (env, desc, suite.get_task_init_states(tid))

    import imageio.v2 as iio
    clips = []
    for spec in a.episodes.split(","):
        tid, init = (int(x) for x in spec.split(":"))
        out = {}
        out[False] = rollout(pr, tid, init, a.sev, M_inv, W, a.gamma, False,
                             a.dead, a.norm_r, a.clip,
                             rec_cam=a.rec_cam, rec_fovy=a.rec_fovy)
        print(f"  task {tid} init {init}  frozen success={out[False][1]} "
              f"steps={len(out[False][0])}")
        if a.only_frozen_fail and out[False][1]:
            # the policy is stochastic, so a nominally-failing episode can succeed on a
            # given render. Showing such a clip demonstrates nothing, so skip it.
            print("    frozen succeeded this run -> skipping (nothing to show)")
            continue
        out[True] = rollout(pr, tid, init, a.sev, M_inv, W, a.gamma, True,
                            a.dead, a.norm_r, a.clip,
                            rec_cam=a.rec_cam, rec_fovy=a.rec_fovy, corr_dims=cdims)
        print(f"  task {tid} init {init}  adaptive success={out[True][1]} "
              f"steps={len(out[True][0])}")
        (fL, okL, desc), (fR, okR, _) = out[False], out[True]
        # show the channels the correction actually applies, not the first three:
        # with --corr-dims 3,4,5 the translation estimates are computed but never used,
        # and displaying them would misrepresent what the method is doing.
        shown = cdims if cdims is not None else [0, 1, 2]
        shown_lbl = ','.join(['x','y','z','rx','ry','rz'][d] for d in shown)
        n = max(len(fL), len(fR))
        for k in range(n + a.fps):                       # hold the last frame ~1 s
            i, j = min(k, len(fL) - 1), min(k, len(fR) - 1)
            imL, _, tL = fL[i]; imR, fh, tR = fR[j]
            L = annotate(imL, "FROZEN  (uncorrected)", (150, 30, 30),
                         [a.title, f"step {tL}", "SUCCESS" if (okL and k >= len(fL) - 1) else
                          ("FAILED - timeout" if (not okL and k >= len(fL) - 1) else "")],
                         prompt=desc)
            R = annotate(imR, "ADAPTIVE  (online)", (25, 110, 45),
                         [a.title,
                          f"step {tR}   f_hat[{shown_lbl}] = "
                          + " ".join(f"{fh[d]:+.3f}" for d in shown),
                          "SUCCESS" if (okR and k >= len(fR) - 1) else
                          ("FAILED - timeout" if (not okR and k >= len(fR) - 1) else "")],
                         prompt=desc)
            clips.append(np.concatenate([L, R], axis=1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    iio.mimwrite(a.out, clips, fps=a.fps, quality=8)
    print(f"\nwrote {a.out}  ({len(clips)} frames, {len(clips)/a.fps:.1f}s)")


if __name__ == "__main__":
    main()
