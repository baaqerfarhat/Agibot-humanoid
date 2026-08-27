"""Is the fault OBSERVABLE from the robot's own motion? -- the error an adaptive law needs.

Searching over task success ignores everything we know about the plant. LIBERO's arm runs an
OSC_POSE controller at 20 Hz with `control_delta=True` and
`output_max = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]`, so a commanded action unit is exactly 0.05 m
of end-effector translation (0.5 rad of rotation), and `robot0_eef_pos` is in the observation
every step. The commanded and the achieved motion are therefore both measurable, and their
difference is a direct estimate of an additive action fault:

    e_t = dx_t / 0.05  -  a_cmd,t          ->   f      (for a pure additive fault)

That is the error signal an adaptive law would drive to zero, and it needs no notion of task
success at all. What is NOT obvious is whether it survives real OSC tracking error, contact,
and joint limits -- which is what this measures, by logging commanded vs achieved motion over
episodes with the fault on and off.
"""
from __future__ import annotations

import argparse, collections, json, pathlib
import numpy as np

import main as lm
from so3 import rot_delta
from openpi_client import image_tools
from paired_probe import Probe
from gate_faults import apply_action_fault

OUT_MAX = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
FAULT, DIMS = "offset", 6


def log_episode(pr, tid, init, sev, max_steps=220):
    env, desc, inits = pr.env_for(tid)
    env.reset()
    obs = env.set_init_state(inits[init])
    plan, t, rec = collections.deque(), 0, []
    while t < max_steps + 10:
        if t < 10:
            obs, _, done, _ = env.step(lm.LIBERO_DUMMY_ACTION); t += 1; continue
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(
            np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
        wr = image_tools.convert_to_uint8(image_tools.resize_with_pad(
            np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
        if not plan:
            el = {"observation/image": img, "observation/wrist_image": wr,
                  "observation/state": np.concatenate((obs["robot0_eef_pos"],
                      lm._quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                  "prompt": str(desc)}
            plan.extend(pr.client.infer(el)["actions"][: pr.a.replan_steps])
        a_cmd = np.asarray(plan.popleft(), float)
        a_exec = apply_action_fault(a_cmd, FAULT, sev, DIMS)
        x0 = np.array(obs["robot0_eef_pos"], float)
        q0 = np.array(obs["robot0_eef_quat"], float)
        obs, _, done, _ = env.step(a_exec.tolist())
        x1 = np.array(obs["robot0_eef_pos"], float)
        q1 = np.array(obs["robot0_eef_quat"], float)
        rec.append(dict(a_cmd=a_cmd[:6].tolist(), a_exec=a_exec[:6].tolist(),
                        dx=(x1 - x0).tolist(), dr=rot_delta(q0, q1).tolist()))
        if done:
            break
        t += 1
    return rec, bool(done)


def analyse(recs, sev, label):
    A = np.array([r["a_cmd"] for e in recs for r in e])
    E = np.array([r["a_exec"] for e in recs for r in e])
    D = np.array([r["dx"] + r["dr"] for e in recs for r in e])
    ach = D / OUT_MAX                       # achieved motion, in action units
    err = ach - A                           # what an adaptive law would see
    true = E - A                            # the fault actually injected
    print(f"\n=== {label}  ({len(recs)} episodes, {len(A)} steps)")
    print(f"{'dim':<5} {'true fault':>11} {'mean e_t':>10} {'sd e_t':>9} {'SNR':>7}")
    for i, nm in enumerate(["dx", "dy", "dz", "drx", "dry", "drz"]):
        m, s = err[:, i].mean(), err[:, i].std()
        print(f"{nm:<5} {true[:, i].mean():>11.4f} {m:>10.4f} {s:>9.4f} "
              f"{abs(m)/max(s,1e-9):>7.2f}")
    return dict(label=label, sev=sev, n_steps=int(len(A)),
                true=true.mean(0).tolist(), mean_e=err.mean(0).tolist(),
                sd_e=err.std(0).tolist(),
                # raw pairs, so the command->motion map can be IDENTIFIED offline rather
                # than assumed from output_max (which is the target scaling, not the
                # realised one: nominal e_t is -0.175 on dx, so the arm achieves only a
                # fraction of the commanded delta within one 50 ms control step)
                raw_a=[r["a_cmd"] for e in recs for r in e],
                raw_exec=[r["a_exec"] for e in recs for r in e],
                raw_d=[r["dx"] + r["dr"] for e in recs for r in e],
                ep_len=[len(e) for e in recs])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", type=pathlib.Path, required=True)
    p.add_argument("--ack", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--host", default="0.0.0.0"); p.add_argument("--port", type=int, default=8000)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--episodes", type=int, default=3)
    a = p.parse_args()
    pr = Probe(a)
    pr.control(dict(site=None, pin_rng=False))     # frozen policy, no weight edit
    out = []
    for sev, lbl in ((0.0, "NOMINAL (no fault)"), (0.05, "FAULTED (+0.05 on arm dims)")):
        recs = []
        for k in range(a.episodes):
            r, ok = log_episode(pr, k % 10, 45 + k, sev)
            recs.append(r)
            print(f"  {lbl}: episode {k} -> {len(r)} steps, success={ok}")
        out.append(analyse(recs, sev, lbl))
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))
    n, f = out[0], out[1]
    print("\n=== IDENTIFIABILITY: faulted minus nominal, per dim")
    print(f"{'dim':<5} {'injected':>10} {'recovered':>10} {'ratio':>8}")
    for i, nm in enumerate(["dx", "dy", "dz", "drx", "dry", "drz"]):
        rec = f["mean_e"][i] - n["mean_e"][i]
        print(f"{nm:<5} {0.05:>10.4f} {rec:>10.4f} {rec/0.05:>8.2f}")


if __name__ == "__main__":
    main()
