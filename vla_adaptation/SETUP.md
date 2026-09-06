# Setup

Cloning this repo is not enough to run anything. The experiments drive a π0.5 policy served
by [openpi](https://github.com/Physical-Intelligence/openpi), and neither that stack nor the
12 GB checkpoint is vendored here. This file lists exactly what is needed.

## You will need

| | |
|---|---|
| openpi | pinned at commit `15a9616` (what every result here was produced with) |
| checkpoint | `pi05_libero`, ~12 GB, downloaded to `$OPENPI_CACHE` |
| GPU | one, ~9 GB for the served policy. The experiments themselves are CPU |
| LIBERO | installed as openpi's `third_party/libero` |

## Two environments, not one

This is the detail most likely to waste your afternoon. openpi uses **separate virtualenvs**
for the model server and the simulation client, and they are not interchangeable:

| | python | needs |
|---|---|---|
| **server** — runs π0.5 | 3.11 | `jax` 0.5.3, `flax` 0.10.2, `numpy` 1.26 |
| **client** — runs these experiments | 3.8 | `robosuite` 1.4.1, `numpy` 1.22, `libero`, `openpi_client` |

The client venv has **no jax at all**. Running an experiment script with the server
interpreter, or vice versa, fails in ways that look like a bug in this code.

## 1. Install openpi at the pinned commit

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi && git checkout 15a9616
export OPENPI=$PWD
export OPENPI_CACHE=$HOME/.cache/openpi      # where the checkpoint lands
```

Follow openpi's own instructions to create both venvs and to install
`third_party/libero`. Verify:

```bash
$OPENPI/.venv/bin/python                -c "import jax, flax; print('server ok')"
$OPENPI/examples/libero/.venv/bin/python -c "import robosuite, libero; print('client ok')"
```

## 2. Start the policy server

`ace_server.py` (in this repo) serves π0.5 and additionally allows a named weight tensor to be
perturbed at runtime, which the ACE experiments need. It runs in the **server** venv:

```bash
cd $OPENPI
$OPENPI/.venv/bin/python /path/to/this/repo/openpi/ace_server.py \
  --port 8000 --control /tmp/ctl.json --ack /tmp/ack.json
```

The first launch downloads the checkpoint. Wait for it to listen on 8000 before continuing.

## 3. Run an experiment

From the **client** venv, with this repo's `openpi/` on the path:

```bash
cd $OPENPI
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0     # pick your GPU
export PYTHONPATH=$OPENPI/third_party/libero:$OPENPI/examples/libero:/path/to/this/repo/openpi
REPO=/path/to/this/repo

$OPENPI/examples/libero/.venv/bin/python $REPO/openpi/adaptive_law.py \
  --port 8000 --suite libero_spatial --episodes 20 --sev 0.05 \
  --gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.15 --corr-dims 3,4,5 \
  --log $REPO/results/phase05/error_signal_so3.json \
  --openloop $REPO/results/phase05/openloop_so3.json \
  --control /tmp/ctl.json --ack /tmp/ack.json --out /tmp/run.json

python $REPO/openpi/mcnemar.py /tmp/run.json          # plain python3 is fine here
```

`--control`/`--ack` are the files the server watches; they must match what you passed it.

Expect ~20 minutes for 20 paired episodes on `libero_spatial`.

## Reusing the shipped calibration, or redoing it

`--log` and `--openloop` supply the plant model and the sensitivity matrix `M`. The files in
`results/phase05/` were identified on this machine's healthy rollouts and are reused by
default. To re-identify them yourself:

```bash
# FIR plant, on fault-free rollouts
$CLIENT $REPO/openpi/error_signal.py --port 8000 --episodes 10 --out /tmp/plant.json
# M = d(motion)/d(fault), by open-loop replay; CPU, seconds
$CLIENT $REPO/openpi/openloop_id.py --log /tmp/plant.json --probe 0.02 --out /tmp/M.json
```

`--probe` sets the magnitude at which `M` is measured. It is deliberately exposed: `M` proved
magnitude-independent between 0.02 and 0.05 (docs §9), and that check is worth repeating on a
different plant rather than assumed.

## Analysis only, no GPU

`openpi/mcnemar.py` is pure standard library and runs against the shipped results with nothing
installed:

```bash
python3 openpi/mcnemar.py results/suites/*_rotonly_paired.json
```

That reproduces every p-value in the README from the stored per-episode outcomes.

## What you cannot reproduce from this repo

- **`hardware/`** analyses X2 humanoid logs (`$X2_RUN_LOGS`, 134 rollouts) that are **not
  shipped** — they are large and belong to a different project. The scripts and their outputs
  in `results/hardware/` are here for inspection; the inputs are not.
- **Exact episode outcomes.** π0.5 samples its action flow, so success counts move by roughly
  ±10 points run to run at n=20 (docs §8.7). Compare distributions and paired statistics, not
  individual numbers.

## Known issue

Every run prints an `EGL_NOT_INITIALIZED` traceback at exit, after results are written. It is
cosmetic — MuJoCo releasing its GL context during interpreter shutdown. Results are complete;
an `atexit` fix was attempted and did not resolve it.

## Second backbone: OpenVLA-OFT (optional)

`openpi/oft_server.py` serves OpenVLA-OFT behind the same websocket protocol, so every
experiment script runs unchanged against `--port 8001`. It needs the
[openvla-oft](https://github.com/moojink/openvla-oft) checkout on `sys.path` (edit the `OFT`
path at the top of the file), the `moojink/openvla-7b-oft-finetuned-libero-*` checkpoint
(~16 GB), and its **own** virtualenv — it pins `torch 2.2.0` and a forked `transformers`,
neither compatible with the π0.5 server venv.

`requirements-oft.txt` is the exact set that imports. Three things about it were not obvious
and each cost a rebuild:

- `python3.10 -m venv` may lack `ensurepip`; `--without-pip` plus `get-pip.py` works without root.
- `tensorflow-metadata` and `wandb` resolve by default to releases whose generated protos are
  **protobuf-6 gencode**, which no protobuf that TF 2.15 accepts can load. Pin
  `tensorflow-metadata==1.14.0` and `wandb==0.16.6`; protobuf then settles at 3.20.3.
- TensorFlow is on OFT's inference path (`center_crop`) and claims the whole GPU on import.
  The bridge pins it to CPU before importing anything from OFT. Do the same in any new script.

```bash
cd openvla-oft && python3.10 -m venv --without-pip .venv && .venv/bin/python get-pip.py
.venv/bin/pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -e . -e $OPENPI/packages/openpi-client -r /path/to/this/repo/requirements-oft.txt
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 .venv/bin/python /path/to/this/repo/openpi/oft_server.py \
  --port 8001 --control /tmp/oft_ctl.json --ack /tmp/oft_ack.json
```

## Second manipulator: ALOHA sim (optional)

`openpi/aloha_adapt.py` runs the law on bimanual ALOHA (`gym_aloha`, 14 absolute joint
targets) against openpi's own `pi0_aloha_sim` server — no bridge needed, it is an openpi
checkpoint. Client env: python 3.10, `gym-aloha`, `openpi-client`; `MUJOCO_GL=egl` is
required or dm_control fails at render with "an OpenGL platform library has not been loaded".

```bash
cd $OPENPI/examples/aloha_sim && python3.10 -m venv --without-pip .venv && .venv/bin/python get-pip.py
.venv/bin/pip install -r requirements.in -e $OPENPI/packages/openpi-client
$OPENPI/.venv/bin/python -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/pi0_aloha_sim')"   # 12 GB
CUDA_VISIBLE_DEVICES=1 $OPENPI/.venv/bin/python $OPENPI/scripts/serve_policy.py --env ALOHA_SIM --port 8002
MUJOCO_GL=egl PYTHONPATH=/path/to/this/repo/openpi .venv/bin/python /path/to/this/repo/openpi/aloha_adapt.py log --port 8002 --episodes 8 --out healthy.json
```

Read docs §26–§27 first. Three things differ from LIBERO and each cost a rerun:

- the plant is identified on **position**, not increment (an absolute-target servo absorbs
  an offset into steady-state position error and leaves nothing in Δq);
- every threshold (`--norm-r`, `--dead`, `--clip`) is a ratio against a **measured** scale,
  and the 14-joint residual norm is ~0.19 rad, not LIBERO's 0.034 — `--norm-r 0.4`, not 0.05;
- on a tight-margin task the correction must be **held stationary** while the task runs.
  `--identify-episodes 1` adapts during one sacrificial episode and then freezes the
  estimate; on transfer-cube that is the difference between 0/20 and 5/20 (the healthy
  rate), at the same 94% identification. A continuously updating correction fails there.

```bash
$CLIENT aloha_adapt.py run --port 8002 --episodes 20 --seed 200 --log healthy.json --openloop M.json \
  --fault-vec "0.02,0.02,0.02,0.02,0.02,0.02,0,0,0,0,0,0,0,0" --corr-joints 0,1,2,3,4,5 \
  --clip 0.08 --norm-r 0.4 --identify-episodes 1 --out repaired.json
```

## Third backbone: NVIDIA GR00T N1.7 (optional)

`openpi/groot_server.py` serves the official `nvidia/GR00T-N1.7-LIBERO` finetune behind the same
websocket protocol, so every script above runs unchanged with `--port 8003`. Conventions
(180-degree-rotated 256x256 cameras, 8-D state, 7-D action with the gripper normalised then
inverted) were read from Isaac-GR00T's own LIBERO wrapper, and a two-episode healthy run
through our client scored 2/2 before any fault was applied.

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T && cd Isaac-GR00T && uv sync      # python 3.12
.venv/bin/hf download nvidia/GR00T-N1.7-LIBERO --include "libero_spatial/config.json" \
    "libero_spatial/embodiment_id.json" "libero_spatial/model-*.safetensors" \
    "libero_spatial/model.safetensors.index.json" "libero_spatial/processor_config.json" \
    "libero_spatial/statistics.json" --local-dir checkpoints/GR00T-N1.7-LIBERO     # 6.9 GB
.venv/bin/python -m ensurepip && .venv/bin/python -m pip install websockets
GROOT_HF_LOCAL_FIRST=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u <repo>/openpi/groot_server.py \
    --model-path checkpoints/GR00T-N1.7-LIBERO/libero_spatial --port 8003 \
    --control /tmp/groot_control.json --ack /tmp/groot_ack.json
```

Three things that are not in the GR00T docs:
- The VLM backbone `nvidia/Cosmos-Reason2-2B` is a gated Hugging Face repo. Accept its licence
  once so it can be downloaded; after that `GROOT_HF_LOCAL_FIRST=1` makes GR00T load it from the
  cache without a network round-trip (and is required if no token is configured).
- On a pre-Ampere GPU (Turing and older) set `"use_flash_attention": false` in the checkpoint's
  `config.json`; the model then uses PyTorch SDPA attention. Inference is about 3.8 s per 16-step
  chunk on a Quadro RTX 8000, so use `--replan-steps 8` (GR00T's own evaluation horizon).
- `groot_server.py` imports `openpi_client` from the openpi checkout for the message format; set
  `OPENPI_CLIENT` if that checkout is elsewhere, and `ISAAC_GROOT` for the GR00T checkout.
