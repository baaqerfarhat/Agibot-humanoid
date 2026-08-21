"""The ACE screen of PREREG_OPENPI_ACE_SCREEN §2-§4, on the cell the gate left standing.

ACE_hat(l) = E[M | do(W_l + Delta_l)] - E[M | W_l], Delta_l ~ N(0, rho_l^2 I), every draw
scored by closed-loop LIBERO rollouts on the realised metric. rho_l is set per site so the
RELATIVE displacement is identical everywhere (prereg §3), which is what keeps the ranking
from being a ranking of layer sizes.

Budget, fixed in advance: 8 draws x 5 episodes = 40 episodes per site, 10 sites, plus a
20-episode frozen-faulted baseline shared across sites. All episodes carry the surviving
fault, offset @ 0.05 on arm dims 0-5.

Common random numbers: the (task, initial-state) pairs for draw d are the same for every
site, and disjoint from the gate's (the gate used initial states 0-1; the screen uses 2-5,
the baseline 6-7).

Every draw is VERIFIED live before any episode runs: the server acks the relative
displacement it actually applied, and a draw whose ack is missing or off-target aborts the
run rather than quietly scoring the unperturbed model. This is not paranoia -- openpi's
serving path freezes module state at wrap time, so the natural implementation of this
screen silently measures nothing (see ace_server.py).
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import time

import numpy as np

from libero.libero import benchmark
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wc

import main as libero_main
from gate_faults import apply_action_fault

SITES = ["action_out_proj/bias", "action_out_proj/kernel", "action_in_proj/kernel",
         "time_mlp_out/kernel", "expert/mlp_1/linear/L0", "expert/mlp_1/linear/L8",
         "expert/mlp_1/linear/L17", "llm/mlp/linear/L0", "llm/mlp/linear/L17",
         "img/MlpBlock_0/Dense_1/kernel/B26"]
N_DRAWS, EP_PER_DRAW, MAX_STEPS = 8, 5, 220
FAULT, SEVERITY, DIMS = "offset", 0.05, 6
C_REL = 0.02


def draw_episodes(d: int):
    """5 (task, init) pairs for draw d -- identical across sites, disjoint from the gate."""
    return [((d * EP_PER_DRAW + i) % 10, 2 + (d * EP_PER_DRAW + i) // 10)
            for i in range(EP_PER_DRAW)]


def baseline_episodes():
    return [(t, 6 + k) for k in range(2) for t in range(10)]


class Runner:
    def __init__(self, args):
        self.a = args
        self.suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        self.client = _wc.WebsocketClientPolicy(args.host, args.port)
        self._envs = {}

    def env_for(self, task_id):
        if task_id not in self._envs:
            task = self.suite.get_task(task_id)
            env, desc = libero_main._get_libero_env(
                task, libero_main.LIBERO_ENV_RESOLUTION, self.a.seed)
            self._envs[task_id] = (env, desc, self.suite.get_task_init_states(task_id))
        return self._envs[task_id]

    def set_control(self, site, draw, seed):
        """Ask the server for a perturbation and refuse to proceed until it confirms."""
        self.a.ack.unlink(missing_ok=True)
        self.a.control.write_text(json.dumps(
            dict(site=site, draw=draw, seed=seed)))
        # the server applies control on its next infer call, so poke it with one
        t0 = time.time()
        while True:
            self.probe()
            if self.a.ack.exists():
                ack = json.loads(self.a.ack.read_text())
                if ack.get("site") == site:
                    break
            if time.time() - t0 > 300:
                raise RuntimeError(f"server never acked control for {site} draw {draw}")
        if site is not None:
            rel = ack.get("applied_rel", 0.0)
            if not ack.get("ok") or abs(rel - C_REL) > 0.1 * C_REL:
                raise RuntimeError(
                    f"perturbation NOT live for {site} draw {draw}: applied_rel={rel}")
        return ack

    def probe(self):
        """One cheap inference so the server picks up the control file."""
        env, desc, inits = self.env_for(0)
        env.reset()
        obs = env.set_init_state(inits[0])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(
            np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(
            np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
        self.client.infer({"observation/image": img, "observation/wrist_image": wrist,
                           "observation/state": np.concatenate((
                               obs["robot0_eef_pos"],
                               libero_main._quat2axisangle(obs["robot0_eef_quat"]),
                               obs["robot0_gripper_qpos"])),
                           "prompt": str(desc)})

    def episode(self, task_id, init_idx) -> bool:
        env, desc, inits = self.env_for(task_id)
        env.reset()
        obs = env.set_init_state(inits[init_idx])
        plan, t, done = collections.deque(), 0, False
        while t < MAX_STEPS + self.a.num_steps_wait:
            try:
                if t < self.a.num_steps_wait:
                    obs, _, done, _ = env.step(libero_main.LIBERO_DUMMY_ACTION)
                    t += 1
                    continue
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                    np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
                wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                    np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
                if not plan:
                    chunk = self.client.infer({
                        "observation/image": img, "observation/wrist_image": wrist,
                        "observation/state": np.concatenate((
                            obs["robot0_eef_pos"],
                            libero_main._quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"])),
                        "prompt": str(desc)})["actions"]
                    plan.extend(chunk[: self.a.replan_steps])
                action = apply_action_fault(plan.popleft(), FAULT, SEVERITY, DIMS)
                obs, _, done, _ = env.step(action.tolist())
                if done:
                    return True
                t += 1
            except Exception as e:                                  # noqa: BLE001
                print(f"  [ep error task {task_id} init {init_idx}] {e}")
                return False
        return False

    def run_block(self, site, draw, eps, seed):
        ack = self.set_control(site, draw, seed)
        t0 = time.time()
        succ = sum(self.episode(t, i) for t, i in eps)
        return dict(site=site, draw=draw, seed=seed, n=len(eps), successes=succ,
                    success_rate=succ / len(eps), ack=ack, wall_s=round(time.time() - t0, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--control", type=pathlib.Path, required=True)
    p.add_argument("--ack", type=pathlib.Path, required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--only-baseline", action="store_true")
    a = p.parse_args()

    r = Runner(a)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if a.out.exists():
        done = {(d["site"], d["draw"]): d for d in json.loads(a.out.read_text())["blocks"]}
        print(f"resuming: {len(done)} blocks already recorded")

    blocks = list(done.values())

    def save():
        a.out.write_text(json.dumps(dict(
            fault=FAULT, severity=SEVERITY, dims=DIMS, c_rel=C_REL,
            n_draws=N_DRAWS, ep_per_draw=EP_PER_DRAW, blocks=blocks), indent=1))

    if (None, -1) not in done:
        print("=== baseline (frozen faulted, 20 episodes)")
        b = r.run_block(None, -1, baseline_episodes(), 0)
        print(f"    baseline {b['successes']}/{b['n']} = {100*b['success_rate']:.0f}%"
              f"  ({b['wall_s']}s)")
        blocks.append(b)
        save()
    if a.only_baseline:
        return

    for si, site in enumerate(SITES):
        for d in range(N_DRAWS):
            if (site, d) in done:
                continue
            seed = 1000 + si * 100 + d          # unique per (site, draw), fixed in advance
            b = r.run_block(site, d, draw_episodes(d), seed)
            print(f"{site:<38} draw {d}  {b['successes']}/{b['n']}"
                  f"  rel={b['ack'].get('applied_rel', 0):.4f}  ({b['wall_s']}s)")
            blocks.append(b)
            save()
    print("=== SCREEN COMPLETE")


if __name__ == "__main__":
    main()
