"""Serve GR00T N1.7 (SimplerEnv Bridge finetune) for the WidowX behind the experiments'
websocket protocol -- a fourth robot, a second simulator (SAPIEN), a benchmark with real-robot
counterparts (Bridge V2) that J-PARC also reports on.

Conventions from Isaac-GR00T's own wrapper (gr00t/eval/sim/SimplerEnv/simpler_env.py,
WidowXBridgeEnv):
  video.image_0   (T=1, 256, 256, 3) uint8, the third-person camera resized by the wrapper
  state           x, y, z, roll, pitch, yaw (Bridge-converted rpy), pad (0), gripper -- (1,1) each
  action          x, y, z, roll, pitch, yaw (deltas), gripper in [0,1]; the wrapper turns the
                  gripper into +-1 with a 15-step sticky rule -- that stays in the wrapper,
                  so this server returns the raw 8-step chunk (8, 7).
Protocol: obs {"wx/image": (256,256,3) uint8, "wx/state": (8,) float32 in the order above,
"prompt": str} -> {"actions": (8, 7)}.
Run: GROOT_HF_LOCAL_FIRST=1 CUDA_VISIBLE_DEVICES=1 <n17>/.venv/bin/python groot_widowx_server.py
     --model-path <n17>/checkpoints/GR00T-N1.7-SimplerEnv-Bridge --port 8005
"""
from __future__ import annotations

import argparse, asyncio, logging, os, pathlib, sys, time, traceback
import numpy as np

GROOT = pathlib.Path(os.environ.get("ISAAC_GROOT", "/home/mtaheri/ws_AgibotX2/Isaac-GR00T"))
sys.path.insert(0, str(GROOT))
OPENPI_CLIENT = pathlib.Path(os.environ.get(
    "OPENPI_CLIENT", "/home/mtaheri/ws_AgibotX2/openpi/packages/openpi-client/src"))
sys.path.insert(0, str(OPENPI_CLIENT))

from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402
from openpi_client import msgpack_numpy  # noqa: E402
import websockets.asyncio.server as _ws  # noqa: E402

STATE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
LANG_KEY = "annotation.human.action.task_description"


class WidowXPolicy:
    def __init__(self, model_path, embodiment_tag="SIMPLER_ENV_WIDOWX"):
        self.policy = Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=model_path, device="cuda:0")
        self.policy.reset()

    def infer(self, obs):
        img = np.asarray(obs["wx/image"], dtype=np.uint8)
        st = np.asarray(obs["wx/state"], dtype=np.float32).reshape(-1); assert st.shape[0] == 8, st.shape
        o = {"video": {"image_0": img[None, None]},
             "state": {k: st[i:i + 1][None, None] for i, k in enumerate(STATE_KEYS)},
             "language": {LANG_KEY: [[str(obs["prompt"])]]}}
        act, _ = self.policy.get_action(o)
        cols = []
        for k in ACTION_KEYS:
            v = np.asarray(act[k] if k in act else act[f"action.{k}"], dtype=np.float64)
            v = v[0] if v.ndim == 3 else v
            cols.append(v.reshape(v.shape[0], -1))
        return {"actions": np.concatenate(cols, axis=1)}          # (8, 7), gripper raw in [0,1]


async def serve(policy, host, port):
    async def handler(ws):
        packer = msgpack_numpy.Packer()
        await ws.send(packer.pack({"backbone": "gr00t-n1.7-bridge"}))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await ws.recv())
            except Exception:
                return
            t0 = time.monotonic()
            try:
                out = policy.infer(obs)
            except Exception:
                await ws.send(traceback.format_exc()); raise
            out["server_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000}
            await ws.send(packer.pack(out))
    async with _ws.serve(handler, host, port, compression=None, max_size=None) as srv:
        print(f"groot_widowx_server listening on {host}:{port}", flush=True)
        await srv.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8005)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    pol = WidowXPolicy(a.model_path)
    if a.smoke:
        rng = np.random.default_rng(0)
        obs = {"wx/image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
               "wx/state": np.zeros(8, np.float32), "prompt": "put the spoon on the towel"}
        for i in range(3):
            t0 = time.monotonic(); out = pol.infer(obs); dt = time.monotonic() - t0
            print(f"smoke {i}: actions {out['actions'].shape} {dt*1000:.0f} ms  first row {np.round(out['actions'][0], 4)}", flush=True)
        sys.exit(0)
    asyncio.run(serve(pol, a.host, a.port))
