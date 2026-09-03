"""Serve OpenVLA-OFT behind the SAME websocket protocol the experiments already speak.

The adaptive law never touches the policy: it needs an `infer(obs) -> {"actions": chunk}`
endpoint and nothing else. So a second backbone is a second server, and every experiment
script, the plant model and M stay exactly as they are. This is the architecture-independence
test: if the law repairs OpenVLA-OFT with the pi0.5-identified calibration, nothing in the
method was pi0.5-specific.

Conventions reconciled here, each verified against the two codebases rather than assumed:
  images   both stacks rotate LIBERO renders 180 deg. Our client sends 224x224 pad-resized;
           OFT trains on a JPEG round-trip + lanczos3 resize from 256. If the client also
           sends the raw 256 render (observation/image_raw), OFT's own resize is applied so
           the input distribution matches training; otherwise the 224 image is used as-is.
  state    identical 8-D: eef_pos(3) + axis-angle(3) + gripper_qpos(2). OFT normalises it
           internally with the checkpoint's proprio stats.
  actions  OFT returns an 8-step chunk un-normalised to env units EXCEPT the gripper, which
           its eval binarises to +-1 and then sign-flips before env.step. That post-processing
           is applied HERE so the returned chunk means the same thing pi0.5's does: env-ready.
  control  the Probe's control/ack handshake (site=None) is honoured with the same ack shape
           ace_server writes, so paired_probe.Probe works unmodified.
"""
from __future__ import annotations
import argparse, asyncio, json, logging, pathlib, sys, time, traceback
import numpy as np

OFT = pathlib.Path("/home/mtaheri/ws_AgibotX2/openvla-oft")
sys.path.insert(0, str(OFT))
# TensorFlow is on OFT's inference path (center_crop -> tf.image.crop_and_resize) and by
# default claims ALL GPU memory on import -- on a card shared with a 15 GB torch model and
# another user's training job. It only resizes images here, so pin it to CPU first.
import tensorflow as _tf  # noqa: E402
_tf.config.set_visible_devices([], "GPU")
from experiments.robot.openvla_utils import (get_vla, get_processor, get_action_head,      # noqa: E402
                                             get_proprio_projector, get_vla_action,
                                             resize_image_for_policy)
from experiments.robot.robot_utils import normalize_gripper_action, invert_gripper_action  # noqa: E402
from prismatic.vla.constants import PROPRIO_DIM  # noqa: E402
from openpi_client import msgpack_numpy  # noqa: E402
import websockets.asyncio.server as _ws  # noqa: E402


class Cfg:
    """The subset of run_libero_eval.GenerateConfig the loaders read."""
    def __init__(self, ckpt, suite):
        self.model_family = "openvla"; self.pretrained_checkpoint = ckpt
        self.use_l1_regression = True; self.use_diffusion = False
        self.num_diffusion_steps_train = 50; self.num_diffusion_steps_inference = 50
        self.use_film = False; self.num_images_in_input = 2; self.use_proprio = True
        self.center_crop = True; self.num_open_loop_steps = 8; self.lora_rank = 32
        self.load_in_8bit = False; self.load_in_4bit = False
        self.unnorm_key = f"{suite}_no_noops"


class OFTPolicy:
    def __init__(self, ckpt, suite):
        self.cfg = Cfg(ckpt, suite)
        self.vla = get_vla(self.cfg)
        self.proc = get_processor(self.cfg)
        self.head = get_action_head(self.cfg, llm_dim=self.vla.llm_dim)
        self.proprio = get_proprio_projector(self.cfg, llm_dim=self.vla.llm_dim, proprio_dim=PROPRIO_DIM)

    @staticmethod
    def _img(obs, key224, keyraw):
        if keyraw in obs:                       # faithful path: OFT's own resize from 256
            return resize_image_for_policy(np.asarray(obs[keyraw], dtype=np.uint8), 224)
        return np.asarray(obs[key224], dtype=np.uint8)

    def infer(self, obs: dict) -> dict:
        o = {"full_image": self._img(obs, "observation/image", "observation/image_raw"),
             "wrist_image": self._img(obs, "observation/wrist_image", "observation/wrist_image_raw"),
             "state": np.asarray(obs["observation/state"], dtype=np.float64)}
        chunk = get_vla_action(self.cfg, self.vla, self.proc, o, str(obs["prompt"]),
                               action_head=self.head, proprio_projector=self.proprio,
                               use_film=False)
        acts = []
        for a in chunk:                          # the eval's process_action, per step
            a = normalize_gripper_action(np.asarray(a, dtype=np.float64), binarize=True)
            a = invert_gripper_action(a)
            acts.append(a)
        return {"actions": np.stack(acts)}       # (8, 7), env-ready


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
        # this backbone has no perturbable site: acknowledge, and refuse anything else loudly
        if req.get("site") is None and req.get("bias_add") is None and req.get("combo") is None:
            ack.write_text(json.dumps(dict(site=None, draw=req.get("draw"), applied_rel=0.0,
                                           ok=True, pin_rng=bool(req.get("pin_rng", False)))))
        else:
            ack.write_text(json.dumps(dict(site=None, ok=False, error="oft_server has no ACE sites")))

    async def handler(ws):
        packer = msgpack_numpy.Packer()
        await ws.send(packer.pack({"backbone": "openvla-oft"}))
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
        print(f"oft_server listening on {host}:{port}", flush=True)
        await srv.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="moojink/openvla-7b-oft-finetuned-libero-spatial")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--control", type=pathlib.Path, required=True)
    ap.add_argument("--ack", type=pathlib.Path, required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    pol = OFTPolicy(a.ckpt, a.suite)
    asyncio.run(serve(pol, a.host, a.port, a.control, a.ack))
