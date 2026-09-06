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
Run from the Isaac-GR00T checkout with its own venv:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python groot_server.py --model-path checkpoints/GR00T-N1.7-LIBERO/libero_spatial \
      --port 8003 --control ... --ack ...
"""
from __future__ import annotations

import argparse, asyncio, json, logging, os, pathlib, sys, time, traceback
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


async def serve(policy, host, port, control: pathlib.Path, ack: pathlib.Path):
    last = None

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
            ack.write_text(json.dumps(dict(site=None, draw=req.get("draw"), applied_rel=0.0,
                                           ok=True, pin_rng=bool(req.get("pin_rng", False)))))
        else:
            ack.write_text(json.dumps(dict(site=None, ok=False, error="groot_server has no ACE sites")))

    async def handler(ws):
        packer = msgpack_numpy.Packer()
        await ws.send(packer.pack({"backbone": "gr00t-n1.7"}))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await ws.recv())
            except Exception:
                return
            apply_control()
            t0 = time.monotonic()
            try:
                out = policy.infer(obs)
            except Exception:
                await ws.send(traceback.format_exc()); raise
            out["server_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000}
            await ws.send(packer.pack(out))

    async with _ws.serve(handler, host, port, compression=None, max_size=None) as srv:
        print(f"groot_server listening on {host}:{port}", flush=True)
        await srv.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--embodiment-tag", default="LIBERO_PANDA")
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--control", type=pathlib.Path, required=True)
    ap.add_argument("--ack", type=pathlib.Path, required=True)
    ap.add_argument("--smoke", action="store_true", help="load, run one dummy inference, print shapes, exit")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    pol = GrootPolicy(a.model_path, a.embodiment_tag)
    if a.smoke:
        rng = np.random.default_rng(0)
        obs = {"observation/image_raw": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
               "observation/wrist_image_raw": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
               "observation/state": np.zeros(8, np.float32), "prompt": "pick up the black bowl"}
        for i in range(3):
            t0 = time.monotonic(); out = pol.infer(obs); dt = time.monotonic() - t0
            print(f"smoke {i}: actions {out['actions'].shape} {out['actions'].dtype} "
                  f"{dt*1000:.0f} ms  first row {np.round(out['actions'][0], 3)}", flush=True)
        sys.exit(0)
    asyncio.run(serve(pol, a.host, a.port, a.control, a.ack))
