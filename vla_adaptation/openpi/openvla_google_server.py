"""Serve base OpenVLA-7B (openvla/openvla-7b, OXE-pretrained, no finetune) for the SimplerEnv
Google robot behind the experiments' websocket protocol -- a fifth robot and the second
backbone on the SimplerEnv benchmark. WidowX is not used with this policy: the published
SimplerEnv table puts base OpenVLA at 0% on every WidowX task (a competence floor, which the
protocol rejects as a control); on google_robot_move_near it reports about 46%.

Conventions (SimplerEnv's own OpenVLA wrapper, reproduced here):
  prompt      "In: What action should the robot take to {instruction}?\nOut:"
  unnorm_key  fractal20220817_data (google robot)
  action      7 = world_vector xyz (m), rotation delta rpy (rad) -> axis-angle, open_gripper in [0,1]
The GR00T SimplerEnv gym wrapper (simpler_env.py, Google branch) applies the relative/sticky
gripper rule itself from a raw [0,1] open-ness, so this server returns the raw value, as
groot_widowx_server.py does. Returns a (1, 7) chunk: [x, y, z, ax, ay, az, gripper].
Protocol: obs {"wx/image": (H,W,3) uint8, "wx/state": (8,), "prompt": str} -> {"actions": (1,7)}.
Run: CUDA_VISIBLE_DEVICES=1 <openvla-oft>/.venv/bin/python openvla_google_server.py
     --model-path <openvla-oft>/checkpoints/openvla-7b --port 8006
"""
from __future__ import annotations

import argparse, asyncio, logging, os, pathlib, sys, time, traceback
import numpy as np

OPENPI_CLIENT = pathlib.Path(os.environ.get(
    "OPENPI_CLIENT", "/home/mtaheri/ws_AgibotX2/openpi/packages/openpi-client/src"))
sys.path.insert(0, str(OPENPI_CLIENT))
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: E402
from transforms3d.euler import euler2axangle  # noqa: E402
from openpi_client import msgpack_numpy  # noqa: E402
import websockets.asyncio.server as _ws  # noqa: E402


class OpenVLAGoogle:
    def __init__(self, model_path, unnorm_key="fractal20220817_data", device="cuda:0"):
        self.proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to(device).eval()
        self.device = device; self.unnorm_key = unnorm_key
        assert unnorm_key in self.vla.norm_stats, list(self.vla.norm_stats)[:10]

    @torch.no_grad()
    def infer(self, obs):
        img = Image.fromarray(np.asarray(obs["wx/image"], dtype=np.uint8))
        prompt = f"In: What action should the robot take to {str(obs['prompt']).lower()}?\nOut:"
        inputs = self.proc(prompt, img).to(self.device, dtype=torch.bfloat16)
        raw = np.asarray(self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False), dtype=np.float64)
        ax, ang = euler2axangle(raw[3], raw[4], raw[5])
        act = np.concatenate([raw[:3], np.asarray(ax) * ang, raw[6:7]])
        return {"actions": act[None]}          # (1, 7), gripper raw open-ness in [0,1]


async def serve(policy, host, port):
    async def handler(ws):
        packer = msgpack_numpy.Packer()
        await ws.send(packer.pack({"backbone": "openvla-7b-base"}))
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
        print(f"openvla_google_server listening on {host}:{port}", flush=True)
        await srv.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--unnorm-key", default="fractal20220817_data")
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8006)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    pol = OpenVLAGoogle(a.model_path, a.unnorm_key)
    if a.smoke:
        rng = np.random.default_rng(0)
        obs = {"wx/image": rng.integers(0, 255, (256, 320, 3), dtype=np.uint8),
               "wx/state": np.zeros(8, np.float32), "prompt": "move blue plastic bottle near sponge"}
        for i in range(3):
            t0 = time.monotonic(); out = pol.infer(obs); dt = time.monotonic() - t0
            print(f"smoke {i}: actions {out['actions'].shape} {dt*1000:.0f} ms  row {np.round(out['actions'][0], 4)}", flush=True)
        sys.exit(0)
    asyncio.run(serve(pol, a.host, a.port))
