# Reproducing my Isaac Sim setup (X2 box pickup)

Everything needed to stand up the exact environment the v31 box-pickup policy was
trained and evaluated in, so adaptation work runs against the same physics, the same
reference motion and the same observation layout I used.

The end state is a rollout that matches `adaptation/FOR_MENTOR/isaac_v31_rollout.npz`:
734 control steps, box lifted and set down, no termination.

Budget roughly **90 minutes** and **30 GB of disk**; the Isaac Sim wheels are most of both.

---

## 0. Quickstart

```bash
git clone <this repo> Agibot-humanoid
cd Agibot-humanoid

# 1. Build the holosoma working copy (clone + X2 overlay + meshes)
./box_pickup/setup_holosoma_x2.sh ../holosoma

# 2. Install Isaac Sim 5.1 + IsaacLab 2.3 + holosoma into a conda env named `hssim`
../holosoma/scripts/setup_isaacsim.sh

# 3. Regenerate the reference motion clips (NOT in git - see section 2.4)
~/.holosoma_deps/miniconda3/envs/hssim/bin/python box_pickup/make_speed_variants.py

# 4. Unpack the v31 checkpoint (gitignored as *.pt, ships in the tarball)
tar -xzf x2_box_policy_share.tar.gz

# 5. Smoke test - should end with "termination: none (ran to max_steps)"
OMNI_KIT_ACCEPT_EULA=1 CUDA_VISIBLE_DEVICES=0 \
  ~/.holosoma_deps/miniconda3/envs/hssim/bin/python adaptation/dump_for_mentor.py \
  --out /tmp/my_rollout.npz
```

---

## 1. What I ran on

| | |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA TITAN RTX, 24 GB, driver 595.84 |
| Python | 3.11.15 (conda env `hssim`) |
| torch | 2.7.0+cu128 |
| Isaac Sim | 5.1.0.0 (pip, `isaacsim[all,extscache]`) |
| IsaacLab | v2.3.0 (tag), `isaaclab` 0.47.2 |
| holosoma | `amazon-far/holosoma` + the overlay in `box_pickup/holosoma_overlay/` |
| numpy | 1.26.0 |

Eval runs with `num_envs=1` and fits in well under 24 GB, so a smaller card is fine.
Training at 4096 envs is what actually needs the memory.

---

## 2. Install, step by step

### 2.1 Clone this repo

Nothing here needs LFS. The largest tracked file is `x2_box_policy_share.tar.gz` (32 MB).

### 2.2 Build the holosoma working copy

holosoma is **not** vendored. `box_pickup/holosoma_overlay/` holds only the files I
added or changed, and the setup script clones upstream and lays the overlay on top:

```bash
./box_pickup/setup_holosoma_x2.sh ../holosoma
```

It clones `amazon-far/holosoma`, rsyncs the overlay over it, and fills in the X2 STL
meshes from `mjlab/src/mjlab/asset_zoo/robots/x2/xmls/assets` (identical files, kept in
one place so the repo stays small).

**Put the clone at `../holosoma`, a sibling of this repo.** That is the default every
script assumes. Anywhere else works too, but then export `HOLOSOMA_ROOT` (section 4).

Re-run this script any time you `git pull` here — that is how overlay changes reach
your holosoma checkout.

The overlay is checked file-for-file against my working holosoma checkout, so what you
get after this step is the same tree I run, not an older snapshot of it. That includes
the v27–v33 changes to the PD gains, the domain randomization ranges and the reward
terms, which had drifted out of the overlay until now.

### 2.3 Install Isaac Sim, IsaacLab and holosoma

```bash
../holosoma/scripts/setup_isaacsim.sh
```

This is holosoma's own installer with one local change (it skips the `apt install` when
the build tools are already present, so it does not block on a sudo prompt on a headless
box). It is idempotent — it drops a sentinel file and returns immediately on re-runs.

What it does: installs Miniconda under `~/.holosoma_deps/miniconda3`, creates the
`hssim` env on Python 3.11, installs torch 2.7.0+cu128, `isaacsim[all,extscache]==5.1.0`
from the NVIDIA index, clones IsaacLab at tag `v2.3.0` into `~/.holosoma_deps/IsaacLab`
and runs `./isaaclab.sh --install`, then `pip install -e ../holosoma/src/holosoma`.

Everything lands in `~/.holosoma_deps` (26 GB on my machine, 23 GB of it the conda env).
Override the env name with `CONDA_ENV_NAME=... ` if you want.

There is no `conda activate` in any of my commands — I call the interpreter by absolute
path, `~/.holosoma_deps/miniconda3/envs/hssim/bin/python`. Activating works equally well.

### 2.4 Regenerate the reference motion clips

**This step is mandatory and easy to miss.** v31 tracks
`box_multispeed/box_speed100.npz`, and that directory is 12 MB of derived data that is
not in git. It is generated deterministically from the source clip that *is* in the
overlay:

```bash
~/.holosoma_deps/miniconda3/envs/hssim/bin/python box_pickup/make_speed_variants.py
```

That writes three clips into
`../holosoma/src/holosoma/holosoma/data/motions/x2_31dof/whole_body_tracking/box_multispeed/`.
Verify you got mine exactly:

| file | frames | md5 |
|---|---|---|
| `box_speed080.npz` | 879 | `16f8e8374dd6a1c18f066da22b135b92` |
| `box_speed100.npz` | 734 | `73bda589b2ee528617ead5bd16526e15` |
| `box_speed125.npz` | 617 | `83d2e0981e8ea17c1fff11c5dd16dfcd` |

`box_speed100.npz` is byte-identical to
`adaptation/FOR_MENTOR/v31_reference_box_speed100.npz`, the clip I sent you, so you can
also just copy that file into place instead of regenerating.

The three clips are time-scaled variants of the same pickup (0.8x, 1.0x, 1.25x) plus a
3 s static hold appended at the end. Training samples among all three; eval should pin
`box_speed100`.

### 2.5 Unpack the v31 checkpoint

`.gitignore` excludes `*.pt` except under `box_pickup/policy/`, so the v31 checkpoint
travels inside the tracked tarball:

```bash
tar -xzf x2_box_policy_share.tar.gz
md5sum x2_box_policy_share/checkpoint/model_202500.pt   # 89849d9fe6f957671bad821cd6832201
```

The scripts look for the checkpoint in the original training log directory first, then
fall back to this extracted copy, so no path editing is needed. `x2_box_policy_share/`
also carries the numpy-exported policies and the deploy scripts.

---

## 3. Smoke test

```bash
OMNI_KIT_ACCEPT_EULA=1 CUDA_VISIBLE_DEVICES=0 \
  ~/.holosoma_deps/miniconda3/envs/hssim/bin/python adaptation/dump_for_mentor.py \
  --out /tmp/my_rollout.npz
```

Headless, seed 42, observation noise off, starts at motion frame 0. Expect **734 steps
with no termination**, and a `termination_reason` of `none (ran to max_steps)` in the
metadata. The first Isaac launch spends several minutes compiling shaders before the
sim starts; later runs are much quicker.

Compare against my log to confirm you match:

```python
import json, numpy as np
mine  = np.load("adaptation/FOR_MENTOR/isaac_v31_rollout.npz", allow_pickle=True)
yours = np.load("/tmp/my_rollout.npz", allow_pickle=True)
print(json.loads(str(yours["_metadata_json"]))["termination_reason"])
print("actions:", np.abs(mine["actions"] - yours["actions"]).max())
print("box z  :", np.abs(mine["object_pos"][:, 2] - yours["object_pos"][:, 2]).max())
```

Re-running on my machine reproduces the log bitwise — both diffs come out at exactly
`0.0`. On different GPU or driver versions PhysX may drift slightly, so a small nonzero
difference is fine. A large one, or an early termination, means the motion clip or the
config is wrong; check section 2.4 first.

---

## 4. Where things live, and how to move them

| what | default location |
|---|---|
| holosoma checkout | `../holosoma` (sibling of this repo) |
| conda env | `~/.holosoma_deps/miniconda3/envs/hssim` |
| IsaacLab | `~/.holosoma_deps/IsaacLab` |
| v31 checkpoint | training log dir, else `x2_box_policy_share/checkpoint/model_202500.pt` |
| reference motion | `<holosoma>/src/holosoma/holosoma/data/motions/x2_31dof/whole_body_tracking/box_multispeed/box_speed100.npz` |
| exported numpy policy | `box_pickup/policy/x2_box_policy_v31.npz` |
| v31 training config | `adaptation/FOR_MENTOR/holosoma_config_v31_20260730_215012.yaml` |

`adaptation/paths.py` resolves all of this, and every entry is overridable by
environment variable, so nothing needs editing if your layout differs:

| variable | overrides |
|---|---|
| `HOLOSOMA_ROOT` | the holosoma checkout |
| `X2_CKPT` | the torch checkpoint |
| `X2_MOTION` | the reference clip used for eval |
| `X2_MOTION_DIR` | the directory the task config samples clips from |
| `X2_POLICY_NPZ` | the exported numpy policy |

Example:

```bash
HOLOSOMA_ROOT=/opt/holosoma X2_CKPT=/data/model_202500.pt \
  python adaptation/dump_for_mentor.py
```

---

## 5. Running the adaptation experiments

`adaptation/adapt_experiments_isaac.py` runs your `LayerAdapter` against this env across
several seeds and variants in a single Isaac session (startup dominates the runtime):

```bash
OMNI_KIT_ACCEPT_EULA=1 CUDA_VISIBLE_DEVICES=0 \
  ~/.holosoma_deps/miniconda3/envs/hssim/bin/python \
  adaptation/adapt_experiments_isaac.py --seeds 5 --steps 400
```

It sweeps `frozen` plus six adapter configurations (two gains x `gx_level` 1 vs the
PhysX Schur-complement inertia at `gx_level=2`, plus two error-mask ablations) and
reports survival, tracking error and box-handling outcome per variant. Useful flags:
`--only <names>` to run a subset, `--record-seed <n>` to dump a renderable NPZ,
`--obs-noise on|off`, `--dr no-push|none|all`.

Your adapter code is imported unmodified from `adaptation/ACC_ADAPTATION_PACKAGE/`.
The one change I make at the call site is the leak term: `-gamma*(W - W0)` instead of
`-gamma*W`, so the leak pulls back toward the trained weights rather than toward zero.

---

## 6. Rendering rollouts to video

Rendering is MuJoCo, not Isaac, and it needs a separate interpreter — `hssim` has no
mujoco. The in-repo mjlab checkout provides one:

```bash
cd mjlab && uv sync && uv pip install imageio pillow && cd ..
MJ=mjlab/.venv/bin/python

$MJ box_pickup/render_box_rollout.py rollout.npz out.mp4
$MJ box_pickup/render_side_by_side.py left.npz "FROZEN" right.npz "ADAPTED" out.mp4
```

Both render offscreen through EGL, so no display is required. They read the robot XML
from the holosoma checkout, because that is where the meshes get installed.

---

## 7. Things that will bite you

**The task config sets `motion_dir`, not `motion_file`, and `motion_dir` wins.** If you
write an eval script that passes `motion_file`, holosoma silently ignores it and hands
the env a random one of the three multispeed clips. This is the single most common way
to end up "diverging" against a policy that is actually fine. My scripts pin the clip
explicitly; `adaptation/FOR_MENTOR/ANSWERS.md` has the code path.

**Observation noise defaults to on.** It is training noise, and leaving it on made the
frozen baseline drop the box on 3 of 5 seeds, which looks like an adaptation win when it
is really a noisy baseline. Every comparison I report uses `--obs-noise off`.

**Push randomization is already off in eval.** holosoma disables the push randomizer
outside training, so you do not need to strip it, and adding pushes back is not an
apples-to-apples comparison against my numbers.

**`OMNI_KIT_ACCEPT_EULA=1` is required** or Isaac Sim will block waiting on the licence
prompt. All my commands set it inline.

**cuDNN is unusable on a driver-535 host, and only TRAINING notices.** The cuDNN 9.2
bundled with the torch cu128 wheels raises `CUDNN_STATUS_NOT_INITIALIZED` on any cuDNN
call when the driver is 535 / CUDA 12.2. Evaluation never hits it — the eval harness pins
`use_adaptive_timesteps_sampler=False`, and everything else in the policy path is
cuBLAS. Training does, through a 1-D smoothing `conv1d` in the adaptive sampler, and dies
seconds after the scene loads. Set `torch.backends.cudnn.enabled = False` before
importing the trainer; that conv is tiny, so the fallback costs nothing.

**Re-run `setup_holosoma_x2.sh` after pulling.** Overlay edits do not propagate to your
holosoma checkout on their own.

---

## 8. The policy contract

Short version, for wiring an adapter in. The full derivation, the per-term slice offsets
and the `base_ang_vel` frame issue are in `adaptation/FOR_MENTOR/ANSWERS.md`.

- **Observation: 164-dim**, the concatenation of six terms in **alphabetical order** —
  `actions`, `base_ang_vel`, `dof_pos`, `dof_vel`, `motion_command`, `motion_ref_ori_b`.
  Alphabetical ordering is holosoma's convention, not a coincidence; get it wrong and
  the policy still runs, it just walks into the floor.
- **Action: 31 joint targets**, applied as `target_q = action * action_scale + default_q`
  and tracked by PD at 500 Hz.
- **Rates**: 50 Hz control, 500 Hz physics, decimation 10.
- **The policy is blind.** No cameras, no box state in the observation. It tracks the
  reference clock, so the box has to start where the reference expects it.
