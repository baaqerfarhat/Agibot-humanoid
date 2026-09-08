"""Serve GR00T N1.5-3B on the Fourier GR1 humanoid (RoboCasa GR1 tabletop) behind the SAME
websocket protocol the experiments speak -- the humanoid manipulator.

The GR1 embodiment is pretrained into the N1.5 base model ("humanoid robots with dexterous
hands using absolute joint space control"), so no finetuning is needed. The client
(gr1_adapt.py) does the robosuite/robocasa side and speaks joint space, like aloha_adapt.py.

Observation the policy wants (gr00t.model.policy.Gr00tPolicy, data config
fourier_gr1_arms_waist):
  video.ego_view                 (T=1, 256, 256, 3) uint8 -- the wrapper's bg-crop-pad render
  state.left_arm/right_arm       (1, 7)   absolute joint positions
  state.left_hand/right_hand     (1, 6)
  state.waist                    (1, 3)
  annotation.human.coarse_action [str]    "unlocked_waist: <task language>" for arms+waist
Returns action.<part> arrays of shape (16, D): absolute joint targets.

Protocol: obs dict with keys "gr1/video" (256,256,3 uint8), "gr1/state" (29,) ordered
[left_arm 7, right_arm 7, left_hand 6, right_hand 6, waist 3], "prompt" str;
reply {"actions": (16, 29)} in the same ordering.

Turing GPU: no flash-attn. The n1.5-release checkout is patched to SDPA (eagle config.json,
modeling_eagle2_5_vl.py, radio_model.py); smoke on 2026-09-07: 0.4 s per chunk.
Run: CUDA_VISIBLE_DEVICES=1 <n15>/.venv/bin/python groot15_server.py --model-path <n15>/checkpoints/GR00T-N1.5-3B --port 8004
"""
from __future__ import annotations

import argparse, asyncio, json, logging, os, pathlib, sys, time, traceback
import numpy as np

N15 = pathlib.Path(os.environ.get("ISAAC_GROOT_N15", "/home/mtaheri/ws_AgibotX2/Isaac-GR00T-n15"))
sys.path.insert(0, str(N15))
OPENPI_CLIENT = pathlib.Path(os.environ.get(
    "OPENPI_CLIENT", "/home/mtaheri/ws_AgibotX2/openpi/packages/openpi-client/src"))
sys.path.insert(0, str(OPENPI_CLIENT))

from gr00t.model.policy import Gr00tPolicy  # noqa: E402
from gr00t.experiment.data_config import DATA_CONFIG_MAP  # noqa: E402
from openpi_client import msgpack_numpy  # noqa: E402
import websockets.asyncio.server as _ws  # noqa: E402

PARTS = [("left_arm", 7), ("right_arm", 7), ("left_hand", 6), ("right_hand", 6), ("waist", 3)]
NJ = sum(d for _, d in PARTS)          # 29


class GR1Policy:
    def __init__(self, model_path: str, data_config: str = "fourier_gr1_arms_waist",
                 embodiment_tag: str = "gr1", denoising_steps=None):
        dc = DATA_CONFIG_MAP[data_config]
        self.policy = Gr00tPolicy(model_path=model_path, embodiment_tag=embodiment_tag,
                                  modality_config=dc.modality_config(),
                                  modality_transform=dc.transform(),
                                  denoising_steps=denoising_steps, device="cuda")

    def infer(self, obs: dict) -> dict:
        img = np.asarray(obs["gr1/video"], dtype=np.uint8)
        st = np.asarray(obs["gr1/state"], dtype=np.float32).reshape(-1)
        assert st.shape[0] == NJ, st.shape
        o = {"video.ego_view": img[None], "annotation.human.coarse_action": [str(obs["prompt"])]}
        i = 0
        for name, d in PARTS:
            o[f"state.{name}"] = st[i:i + d][None]; i += d
        act = self.policy.get_action(o)
        cols = [np.asarray(act[f"action.{name}"], dtype=np.float64).reshape(-1, d) for name, d in PARTS]
        return {"actions": np.concatenate(cols, axis=1)}          # (16, 29)


async def serve(policy, host, port, control: pathlib.Path | None, ack: pathlib.Path | None):
    last = None

    def apply_control():
        nonlocal last
        if control is None or not control.exists():
            return
        st = control.stat().st_mtime_ns
        if st == last:
            return
        last = st
        req = json.loads(control.read_text())
        ack.write_text(json.dumps(dict(site=None, draw=req.get("draw"), applied_rel=0.0,
                                       ok=True, pin_rng=bool(req.get("pin_rng", False)))))

    async def handler(ws):
        packer = msgpack_numpy.Packer()
        await ws.send(packer.pack({"backbone": "gr00t-n1.5-gr1"}))
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
        print(f"groot15_server listening on {host}:{port}", flush=True)
        await srv.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-config", default="fourier_gr1_arms_waist")
    ap.add_argument("--embodiment-tag", default="gr1")
    ap.add_argument("--denoising-steps", type=int, default=None)
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8004)
    ap.add_argument("--control", type=pathlib.Path, default=None)
    ap.add_argument("--ack", type=pathlib.Path, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    pol = GR1Policy(a.model_path, a.data_config, a.embodiment_tag, a.denoising_steps)
    if a.smoke:
        rng = np.random.default_rng(0)
        obs = {"gr1/video": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
               "gr1/state": np.zeros(NJ, np.float32), "prompt": "unlocked_waist: pick the cup"}
        for i in range(3):
            t0 = time.monotonic(); out = pol.infer(obs); dt = time.monotonic() - t0
            print(f"smoke {i}: actions {out['actions'].shape} {dt*1000:.0f} ms", flush=True)
        sys.exit(0)
    asyncio.run(serve(pol, a.host, a.port, a.control, a.ack))
