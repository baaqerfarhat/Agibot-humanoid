"""Phase 0.3/0.4 -- the fault suite and headroom gate, per PREREG_OPENPI_ACE_SCREEN §1.

Faults are CLIENT-SIDE wrappers on the LIBERO env: no model surgery, and every fault is a
deterministic function of the action or the image, so a cell is exactly repeatable given
the same task and initial state. The policy is served frozen over openpi's websocket, the
same server that produced the 99.0% gate-0.1 baseline.

    fault       severities        applied to
    gain        0.7, 0.5, 0.3     multiplicative on the arm action
    offset      0.05, 0.10, 0.20  added to the arm action
    brightness  0.1, 0.2, 0.4     added to the agent-view image (fraction of full scale)

Gate, from the prereg: keep a cell iff the drop from the 99.0% nominal is >= 30 points AND
success stays > 5% (a floor-dead cell returns no search signal -- the walker's tq04 lesson).

INTERPRETATION, recorded because the prereg's wording admits two readings. It says
"per-joint action gain ... on the 7-DoF arm action", but LIBERO's 7-vector is 6 OSC deltas
plus a gripper channel that is effectively +-1. Scaling that 7th dim by 0.3 does not model
a weak joint, it disables grasping and would floor the cell for a reason unrelated to the
fault under study. So the arm faults hit dims 0-5 and leave the gripper alone; `--dims 7`
applies them to all seven if that reading is wanted instead. Every result records which.

This file never edits openpi. Run it with openpi's LIBERO client venv:

    cd /home/mtaheri/ws_AgibotX2/openpi
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=$PWD/third_party/libero:$PWD/examples/libero \
      examples/libero/.venv/bin/python <this file> --fault gain --severity 0.5

Note MUJOCO_EGL_DEVICE_ID indexes the OPPOSITE way from CUDA_VISIBLE_DEVICES on this box,
and CUDA_VISIBLE_DEVICES must stay UNSET for the client (the server owns the GPU).
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import logging
import pathlib
import time

import numpy as np
import tqdm

from libero.libero import benchmark
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

import main as libero_main   # examples/libero/main.py -- import-safe, helpers reused

ARM_DIMS_DEFAULT = 6
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
             "libero_10": 520, "libero_90": 400}
SEVERITIES = {"gain": [0.7, 0.5, 0.3], "offset": [0.05, 0.10, 0.20],
              "brightness": [0.1, 0.2, 0.4], "nominal": [0.0]}


def apply_action_fault(action, kind: str, sev: float, dims: int):
    """Faults on the commanded action. Deterministic, so a cell repeats exactly."""
    a = np.asarray(action, dtype=np.float64).copy()
    if kind == "gain":
        a[:dims] *= sev
    elif kind == "offset":
        a[:dims] += sev
    return a


def apply_image_fault(img, kind: str, sev: float):
    """Brightness bias on the agent-view image, as a fraction of full scale."""
    if kind != "brightness":
        return img
    return np.clip(img.astype(np.int16) + int(round(sev * 255)), 0, 255).astype(np.uint8)


def run_cell(args) -> dict:
    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    max_steps = MAX_STEPS[args.task_suite_name]
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    per_task, episodes, successes = [], 0, 0
    t_start = time.time()
    for task_id in tqdm.tqdm(range(suite.n_tasks), desc=f"{args.fault}@{args.severity}"):
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        env, task_description = libero_main._get_libero_env(
            task, libero_main.LIBERO_ENV_RESOLUTION, args.seed)
        task_successes = 0
        for episode_idx in range(args.num_trials_per_task):
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            done, t = False, 0
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, _, done, _ = env.step(libero_main.LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                    wrist = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist, args.resize_size, args.resize_size))
                    img = apply_image_fault(img, args.fault, args.severity)
                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist,
                            "observation/state": np.concatenate((
                                obs["robot0_eef_pos"],
                                libero_main._quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"])),
                            "prompt": str(task_description),
                        }
                        chunk = client.infer(element)["actions"]
                        action_plan.extend(chunk[: args.replan_steps])
                    action = apply_action_fault(action_plan.popleft(), args.fault,
                                                args.severity, args.dims)
                    obs, _, done, _ = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        successes += 1
                        break
                    t += 1
                except Exception as e:                      # noqa: BLE001
                    logging.error(f"task {task_id} ep {episode_idx}: {e}")
                    break
            episodes += 1
        per_task.append(dict(task_id=task_id, task=str(task_description),
                             n=args.num_trials_per_task, successes=task_successes))
        env.close()

    return dict(
        fault=args.fault, severity=args.severity, dims=args.dims,
        suite=args.task_suite_name, seed=args.seed,
        episodes=episodes, successes=successes,
        success_rate=successes / max(episodes, 1),
        per_task=per_task, wall_s=round(time.time() - t_start, 1),
        replan_steps=args.replan_steps, num_steps_wait=args.num_steps_wait,
    )


def main():
    p = argparse.ArgumentParser(description="Phase 0.4 fault suite / headroom gate")
    p.add_argument("--fault", required=True, choices=sorted(SEVERITIES))
    p.add_argument("--severity", type=float, required=True)
    p.add_argument("--dims", type=int, default=ARM_DIMS_DEFAULT,
                   help="how many leading action dims the arm faults touch (6 = no gripper)")
    p.add_argument("--task-suite-name", default="libero_spatial")
    p.add_argument("--num-trials-per-task", type=int, default=2)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--resize-size", type=int, default=224)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=pathlib.Path, required=True)
    a = p.parse_args()
    if a.fault != "nominal" and a.severity not in SEVERITIES[a.fault]:
        logging.warning(f"{a.severity} is not a pre-registered severity for {a.fault}")

    logging.basicConfig(level=logging.WARNING)
    res = run_cell(a)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"{res['fault']}@{res['severity']}  "
          f"{res['successes']}/{res['episodes']} = {100*res['success_rate']:.1f}%  "
          f"({res['wall_s']}s)  -> {a.out}")


if __name__ == "__main__":
    main()
