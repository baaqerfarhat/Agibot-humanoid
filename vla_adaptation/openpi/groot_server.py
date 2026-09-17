"""Serve NVIDIA GR00T N1.7 (LIBERO finetune) behind the SAME websocket protocol the
experiments already speak -- the third backbone, after pi0.5 and OpenVLA-OFT.

Same argument as oft_server.py: the adaptive law needs `infer(obs) -> {"actions": chunk}`
and nothing else, so a new backbone is a new server and every experiment script, the plant
model and M stay untouched. GR00T is a different developer, a different VLM (Eagle) and a
different action head (flow-matching DiT over 16-step chunks) from both earlier backbones.

Conventions, each read from Isaac-GR00T's own LIBERO wrapper (gr00t/eval/sim/LIBERO/
libero_env.py) rather than assumed:
  images   256x256 LIBERO renders rotated 180 deg (`[::-1, ::-1]`), exactly what our client
           sends as observation/image_raw; the policy's processor does its own resize.
  state    the same 8 numbers as ours -- eef_pos(3), axis-angle(3), gripper_qpos(2) --
           split into named keys x,y,z,roll,pitch,yaw,gripper.
  actions  7-D per step (x,y,z,roll,pitch,yaw,gripper), 16-step chunk; the wrapper
           normalises the gripper from [0,1] to [-1,1], binarises, then sign-flips before
           env.step. Applied HERE so the returned chunk means what pi0.5's does: env-ready.
  batching the policy wants (B, T, ...) arrays with B = 1 and T = 1 for video and state,
           and language as [[str]].
Sampling control (added 2026-09-17 for the coupled-source studies): a control request {sampler_seed, episode, reset}
makes call i of episode e draw its flow-matching noise from torch seeded with sha256('groot|seed|e|i'); the call counter
resets at each control write; the ack carries the applied seed/episode, server pid, source and checkpoint-config
hashes; --calllog records every control application and inference (episode, call, derived seed, first action).
Gr00tPolicy.reset() is a no-op (the policy holds no episode state; the action chunk lives in the client).
Run from the Isaac-GR00T checkout with its own venv:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python groot_server.py --model-path checkpoints/GR00T-N1.7-LIBERO/libero_spatial \
      --port 8003 --control ... --ack ...
"""
from __future__ import annotations

import argparse, asyncio, hashlib, json, logging, os, pathlib, sys, time, traceback
import numpy as np

GROOT = pathlib.Path(os.environ.get("ISAAC_GROOT", "/home/mtaheri/ws_AgibotX2/Isaac-GR00T"))
sys.path.insert(0, str(GROOT))
OPENPI_CLIENT = pathlib.Path(os.environ.get(
    "OPENPI_CLIENT", "/home/mtaheri/ws_AgibotX2/openpi/packages/openpi-client/src"))
sys.path.insert(0, str(OPENPI_CLIENT))

from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402
from openpi_client import msgpack_numpy  # noqa: E402
import websockets.asyncio.server as _ws  # noqa: E402

STATE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
LANG_KEY = "annotation.human.action.task_description"


def normalize_gripper_action(action, binarize=True):
    """Copied from gr00t/eval/sim/LIBERO/libero_env.py: [0,1] -> [-1,1], then sign."""
    action = np.array(action, dtype=np.float64)
    action[..., -1] = 2 * (action[..., -1] - 0.0) / (1.0 - 0.0) - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action):
    action[..., -1] = action[..., -1] * -1.0
    return action


class GrootPolicy:
    def __init__(self, model_path: str, embodiment_tag: str = "LIBERO_PANDA"):
        self.policy = Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=model_path, device="cuda:0")
        self.policy.reset()

    @staticmethod
    def _img(obs, keyraw, key224):
        if keyraw in obs:
            return np.asarray(obs[keyraw], dtype=np.uint8)
        return np.asarray(obs[key224], dtype=np.uint8)

    def infer(self, obs: dict) -> dict:
        img = self._img(obs, "observation/image_raw", "observation/image")
        wr = self._img(obs, "observation/wrist_image_raw", "observation/wrist_image")
        st = np.asarray(obs["observation/state"], dtype=np.float32)          # (8,)
        state = {"x": st[0:1], "y": st[1:2], "z": st[2:3],
                 "roll": st[3:4], "pitch": st[4:5], "yaw": st[5:6], "gripper": st[6:8]}
        o = {"video": {"image": img[None, None], "wrist_image": wr[None, None]},   # (1,1,H,W,3)
             "state": {k: v[None, None].astype(np.float32) for k, v in state.items()},  # (1,1,D)
             "language": {LANG_KEY: [[str(obs["prompt"])]]}}
        act, _ = self.policy.get_action(o)
        cols = []
        for k in ACTION_KEYS:
            v = act[k] if k in act else act[f"action.{k}"]
            v = np.asarray(v, dtype=np.float64)
            v = v[0] if v.ndim == 3 else v                                    # (T, D)
            cols.append(v.reshape(v.shape[0], -1))
        chunk = np.concatenate(cols, axis=1)                                  # (T, 7)
        chunk = invert_gripper_action(normalize_gripper_action(chunk, binarize=True))
        return {"actions": chunk}


def derived_seed(sampler_seed: int, episode: int, call: int) -> int:
    """Deterministic 63-bit seed for (sampler_seed, episode, call): the GR00T analogue of ace_server's
    fold_in(fold_in(key(seed), episode), call). GR00T's only inference-time randomness is the flow-matching initial
    noise drawn with torch.randn on the global torch RNG (gr00t_n1d7.py get_action), so seeding the global torch/CUDA RNGs
    immediately before each get_action makes the draw a function of this triple."""
    h = hashlib.sha256(f"groot|{int(sampler_seed)}|{int(episode)}|{int(call)}".encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def seed_torch(seed: int):
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


async def serve(policy, host, port, control: pathlib.Path, ack: pathlib.Path, calllog: pathlib.Path | None, identity: dict):
    last = None
    holder = {"sampler_seed": None, "episode": 0, "call": 0, "pin_rng": False, "request": 0, "controls": 0}
    log_fh = open(calllog, "a", buffering=1) if calllog else None

    def apply_control():
        nonlocal last
        if not control.exists():
            return
        st = control.stat().st_mtime_ns
        if st == last:
            return
        last = st
        req = json.loads(control.read_text())
        if req.get("site") is None and req.get("bias_add") is None and req.get("combo") is None:
            holder["sampler_seed"] = req.get("sampler_seed")
            holder["episode"] = int(req.get("episode", 0)); holder["call"] = 0
            holder["pin_rng"] = bool(req.get("pin_rng", False)); holder["controls"] += 1
            reset_info = policy.policy.reset() if req.get("reset", False) or req.get("sampler_seed") is not None else None
            a = dict(site=None, draw=req.get("draw"), applied_rel=0.0, ok=True, pin_rng=holder["pin_rng"],
                     sampler_seed=holder["sampler_seed"], episode=holder["episode"], call_index_at_ack=0, reset_applied=reset_info is not None,
                     key_schedule=("sha256('groot|seed|episode|call')[:8] -> torch.manual_seed + cuda.manual_seed_all before every get_action; call index reset at each control write"
                                   if holder["sampler_seed"] is not None else ("torch seed 0 every call" if holder["pin_rng"] else "unpinned (global torch RNG state carried over)")),
                     control_index=holder["controls"], request_index_at_ack=holder["request"], wall=time.time(), **identity)
            ack.write_text(json.dumps(a))
            if log_fh:
                log_fh.write(json.dumps(dict(event="control", **a)) + "\n")
        else:
            ack.write_text(json.dumps(dict(site=None, ok=False, error="groot_server has no ACE sites")))

    def infer_seeded(obs):
        seed = None
        if holder["sampler_seed"] is not None:
            seed = derived_seed(holder["sampler_seed"], holder["episode"], holder["call"]); seed_torch(seed)
        elif holder["pin_rng"]:
            seed = 0; seed_torch(0)
        call = holder["call"]; holder["call"] += 1; holder["request"] += 1
        out = policy.infer(obs)
        if log_fh:
            log_fh.write(json.dumps(dict(event="infer", request=holder["request"], episode=holder["episode"], call=call, derived_seed=seed, wall=time.time(),
                                         first_action=[round(float(x), 6) for x in np.asarray(out["actions"])[0]])) + "\n")
        out["sampling"] = dict(episode=holder["episode"], call=call, derived_seed=seed)
        return out

    async def handler(ws):
        packer = msgpack_numpy.Packer()
        await ws.send(packer.pack({"backbone": "gr00t-n1.7", **identity}))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await ws.recv())
            except Exception:
                return
            apply_control()
            t0 = time.monotonic()
            try:
                out = infer_seeded(obs)
            except Exception:
                await ws.send(traceback.format_exc()); raise
            out["server_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000}
            await ws.send(packer.pack(out))

    async with _ws.serve(handler, host, port, compression=None, max_size=None) as srv:
        print(f"groot_server listening on {host}:{port}", flush=True)
        await srv.serve_forever()


def server_identity(model_path: str) -> dict:
    import torch
    cfg = pathlib.Path(model_path) / "config.json"
    return dict(server_pid=os.getpid(), server_source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                model_path=str(model_path), model_config_sha256=(hashlib.sha256(cfg.read_bytes()).hexdigest() if cfg.exists() else None),
                torch_version=torch.__version__, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), started_wall=time.time(),
                cudnn_deterministic=bool(torch.backends.cudnn.deterministic), cudnn_benchmark=bool(torch.backends.cudnn.benchmark))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--embodiment-tag", default="LIBERO_PANDA")
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--control", type=pathlib.Path, required=True)
    ap.add_argument("--ack", type=pathlib.Path, required=True)
    ap.add_argument("--calllog", type=pathlib.Path, default=None, help="append one JSON line per control application and per inference (episode, call, derived seed, first action)")
    ap.add_argument("--smoke", action="store_true", help="load, run one dummy inference, print shapes, exit")
    ap.add_argument("--selftest-seed", action="store_true", help="load, then check that identical observations with the same derived seed give identical actions and a different seed gives different actions; exit")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    import torch
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    pol = GrootPolicy(a.model_path, a.embodiment_tag)
    if a.smoke or a.selftest_seed:
        rng = np.random.default_rng(0)
        obs = {"observation/image_raw": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
               "observation/wrist_image_raw": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
               "observation/state": np.zeros(8, np.float32), "prompt": "pick up the black bowl"}
        if a.smoke:
            for i in range(3):
                t0 = time.monotonic(); out = pol.infer(obs); dt = time.monotonic() - t0
                print(f"smoke {i}: actions {out['actions'].shape} {out['actions'].dtype} {dt*1000:.0f} ms  first row {np.round(out['actions'][0], 3)}", flush=True)
            sys.exit(0)
        seed_torch(derived_seed(123, 0, 0)); a1 = pol.infer(obs)["actions"]
        seed_torch(derived_seed(123, 0, 0)); a2 = pol.infer(obs)["actions"]
        seed_torch(derived_seed(123, 0, 1)); a3 = pol.infer(obs)["actions"]
        seed_torch(derived_seed(124, 0, 0)); a4 = pol.infer(obs)["actions"]
        same = float(np.max(np.abs(a1 - a2))); diff_call = float(np.max(np.abs(a1 - a3))); diff_seed = float(np.max(np.abs(a1 - a4)))
        print(json.dumps(dict(selftest="groot seeded sampling", same_seed_max_abs_diff=same, next_call_max_abs_diff=diff_call, other_seed_max_abs_diff=diff_seed,
                              passed=bool(same == 0.0 and diff_call > 0 and diff_seed > 0), **server_identity(a.model_path))), flush=True)
        sys.exit(0 if (same == 0.0 and diff_call > 0 and diff_seed > 0) else 1)
    asyncio.run(serve(pol, a.host, a.port, a.control, a.ack, a.calllog, server_identity(a.model_path)))
