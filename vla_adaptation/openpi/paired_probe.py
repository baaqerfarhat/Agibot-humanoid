"""At what perturbation scale does a layer edit actually change BEHAVIOUR?

With the sampler RNG pinned an episode is deterministic, so baseline and perturbed runs on
the same initial state differ only by the weights. That turns the measurement from
"5-episode success rate, 96% of which is noise" into a paired comparison with no noise at
all, and lets a scale sweep answer the question the screen could not: how large must c be
before the perturbation moves the trajectory, let alone the outcome.
"""
from __future__ import annotations

import argparse, atexit, collections, json, pathlib, time
import numpy as np
from libero.libero import benchmark
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wc
import main as libero_main
from gate_faults import apply_action_fault

FAULT, SEV, DIMS, MAXS = "offset", 0.05, 6, 220
# Episode caps differ per suite; libero_10 is long-horizon (520 vs 220), which is the real
# stress test for an estimator that has to hold a correction for the whole episode.
SUITE_MAX = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
             "libero_10": 520, "libero_90": 400}


class Probe:
    def __init__(self, a):
        self.a = a
        name = getattr(a, "suite", None) or "libero_spatial"
        self.suite_name = name
        self.suite = benchmark.get_benchmark_dict()[name]()
        global MAXS
        MAXS = SUITE_MAX.get(name, 220)
        self.client = _wc.WebsocketClientPolicy(a.host, a.port)
        self._envs = {}
        # Close envs while EGL is still alive. Without this they are collected during
        # interpreter shutdown, after the EGL display is gone, and mujoco's context release
        # calls eglMakeCurrent on a dead display -> EGL_NOT_INITIALIZED. The results are
        # already written by then, so it is harmless, but it makes every completed run print
        # a traceback and look like a crash.
        atexit.register(self.close)


    def close(self):
        for env, _, _ in self._envs.values():
            try:
                env.close()
            except Exception:
                pass
        self._envs.clear()

    def env_for(self, tid):
        if tid not in self._envs:
            task = self.suite.get_task(tid)
            env, desc = libero_main._get_libero_env(task, libero_main.LIBERO_ENV_RESOLUTION, 7)
            self._envs[tid] = (env, desc, self.suite.get_task_init_states(tid))
        return self._envs[tid]

    def control(self, req):
        self.a.ack.unlink(missing_ok=True)
        self.a.control.write_text(json.dumps(req))
        t0 = time.time()
        while True:
            self.rollout(0, 8, probe_only=True)
            if self.a.ack.exists():
                return json.loads(self.a.ack.read_text())
            if time.time() - t0 > 300:
                raise RuntimeError("no ack")

    def rollout(self, tid, init, probe_only=False, faulted=True):
        """One deterministic episode. Returns (success, action trace)."""
        env, desc, inits = self.env_for(tid)
        env.reset()
        obs = env.set_init_state(inits[init])
        plan, t, acts = collections.deque(), 0, []
        while t < MAXS + 10:
            if t < 10:
                obs, _, done, _ = env.step(libero_main.LIBERO_DUMMY_ACTION); t += 1; continue
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
            wr = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
            el = {"observation/image": img, "observation/wrist_image": wr,
                  "observation/state": np.concatenate((
                      obs["robot0_eef_pos"], libero_main._quat2axisangle(obs["robot0_eef_quat"]),
                      obs["robot0_gripper_qpos"])), "prompt": str(desc)}
            if not plan:
                chunk = self.client.infer(el)["actions"]
                if probe_only:
                    return None, None
                plan.extend(chunk[: self.a.replan_steps])
            act = plan.popleft()
            acts.append(np.asarray(act, float).copy())
            a2 = apply_action_fault(act, FAULT, SEV if faulted else 0.0, DIMS)
            obs, _, done, _ = env.step(a2.tolist())
            if done:
                return True, np.array(acts)
            t += 1
        return False, np.array(acts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", type=pathlib.Path, required=True)
    p.add_argument("--ack", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--host", default="0.0.0.0"); p.add_argument("--port", type=int, default=8000)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--episodes", type=int, default=6)
    a = p.parse_args()

    pr = Probe(a)
    eps = [(t, 8) for t in range(a.episodes)]
    SITES = ["action_out_proj/bias", "action_in_proj/kernel", "llm/mlp/linear/L0"]
    SCALES = [0.02, 0.10, 0.50]

    # deterministic baseline, RNG pinned
    pr.control(dict(site=None, pin_rng=True))
    base = {}
    for tid, ini in eps:
        s, acts = pr.rollout(tid, ini)
        base[(tid, ini)] = (s, acts)
    print("baseline (pinned):", [int(base[e][0]) for e in eps],
          f"= {sum(base[e][0] for e in eps)}/{len(eps)}")

    # determinism check: repeat one episode
    s2, a2 = pr.rollout(*eps[0])
    d0 = np.abs(a2[:len(base[eps[0]][1])] - base[eps[0]][1]).max() if len(a2) else float("nan")
    print(f"determinism check: same episode twice -> max|dAction| = {d0:.2e}, success {int(s2)} vs {int(base[eps[0]][0])}")

    out = []
    for site in SITES:
        for c in SCALES:
            ack = pr.control(dict(site=site, seed=4242, c_rel=c, pin_rng=True))
            flips, divs = 0, []
            for e in eps:
                s, acts = pr.rollout(*e)
                b_s, b_a = base[e]
                n = min(len(acts), len(b_a))
                divs.append(float(np.abs(acts[:n] - b_a[:n]).max()) if n else 0.0)
                flips += int(bool(s) != bool(b_s))
            rec = dict(site=site, c=c, applied_rel=ack.get("applied_rel"),
                       flips=flips, n=len(eps), max_div=float(np.max(divs)),
                       mean_div=float(np.mean(divs)))
            out.append(rec)
            print(f"{site:<28} c={c:<5} rel={ack.get('applied_rel',0):.4f}  "
                  f"outcome flips {flips}/{len(eps)}  max|dA| {rec['max_div']:.4f}")
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(out, indent=1))
    print("=== PROBE DONE")


if __name__ == "__main__":
    main()
