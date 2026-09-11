#!/usr/bin/env bash
# Stand up the ALOHA half of the openpi stack on lambda.arcl, under /data (home is 100% full).
#
# Only ALOHA: the gripper/normaliser experiment needs gym_aloha + pi0_aloha_sim (12 GB).
# The LIBERO half needs robosuite and a separate 12 GB checkpoint; install that separately.
#
# v4. The real blocker was ENOSPC, not the network: pip cached to a full root fs and
# reported it as a truncated download. All scratch now lives on /data.
# v3. v2 skipped the install because `import openpi` succeeded from v1's partial
# editable install; the guard now requires jax too. v1 reported "done" on a failed setup because each step ended in `|| echo WARN`. Every
# step here FAILS HARD and the script exits non-zero, so "done" means done. The pip install
# also retries: the nvidia_cudnn_cu12 wheel (571 MB) truncated at 362 MB on the first attempt.
set -uo pipefail

ROOT=/data/fxxie/vla
OPENPI=$ROOT/openpi
PIN=15a9616
CONDA=$HOME/miniconda3/bin/conda
export OPENPI_DATA_HOME=$ROOT/assets
# lambda's root filesystem (which holds $HOME and /tmp) is 100% full. pip caches downloads
# under ~/.cache/pip and builds in /tmp, so every install died with ENOSPC -- which pip
# reports as a truncated download ("not enough bytes received"), not as a disk error. Put
# every scratch path on /data, which has ~530G.
export TMPDIR=$ROOT/tmp
export PIP_CACHE_DIR=$ROOT/pipcache
export XDG_CACHE_HOME=$ROOT/cache
export CONDA_PKGS_DIRS=$ROOT/condapkgs
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$CONDA_PKGS_DIRS"

mkdir -p "$ROOT" "$OPENPI_DATA_HOME" || { echo "FATAL: cannot create $ROOT"; exit 1; }
exec > >(tee -a "$ROOT/setup.log") 2>&1
echo "=== setup v2 start $(date -Is) on $(hostname) ==="
df -h /data | tail -1

step() { echo; echo "--- [$(date +%H:%M:%S)] $* ---"; }
die()  { echo "FATAL: $*"; exit 1; }

step "1. openpi at $PIN"
if [ ! -d "$OPENPI/.git" ]; then
  git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI" || die "clone"
fi
git -C "$OPENPI" checkout "$PIN" 2>&1 | tail -1 || die "checkout $PIN"
git -C "$OPENPI" submodule update --init --recursive 2>&1 | tail -2
[ "$(git -C "$OPENPI" rev-parse --short HEAD)" = "$PIN" ] || die "HEAD is not $PIN"
echo "HEAD = $PIN OK"

step "2. server env (python 3.11)"
[ -d "$ROOT/envs/server" ] || "$CONDA" create -y -p "$ROOT/envs/server" python=3.11 2>&1 | tail -2
"$ROOT/envs/server/bin/python" -V || die "server env"

step "3. install openpi (retrying: a 571 MB cudnn wheel truncated on the first attempt)"
# Guard on jax, not openpi: `pip install -e .` registers the package even when its
# dependency downloads fail, so `import openpi` succeeding does NOT mean the install
# completed. v2 skipped the retry loop for exactly that reason.
if ! "$ROOT/envs/server/bin/python" -c "import openpi, jax" 2>/dev/null; then
  "$ROOT/envs/server/bin/pip" install -q --upgrade pip
  ok=0
  for attempt in 1 2 3 4 5; do
    echo "  install attempt $attempt"
    if ( cd "$OPENPI" && "$ROOT/envs/server/bin/pip" install \
           --retries 10 --timeout 120 -e . ) 2>&1 | tail -4; then
      if "$ROOT/envs/server/bin/python" -c "import openpi, jax" 2>/dev/null; then ok=1; break; fi
    fi
    echo "  attempt $attempt did not complete; retrying (pip cache keeps finished wheels)"
    sleep 10
  done
  [ "$ok" = 1 ] || die "openpi install failed after 5 attempts"
fi
"$ROOT/envs/server/bin/python" -c "
import openpi, jax
print('openpi ok; jax', jax.__version__, jax.devices())" || die "openpi/jax import"

step "4. client env (python 3.10, gym_aloha)"
[ -d "$ROOT/envs/aloha" ] || "$CONDA" create -y -p "$ROOT/envs/aloha" python=3.10 2>&1 | tail -2
"$ROOT/envs/aloha/bin/pip" install -q --upgrade pip
if ! "$ROOT/envs/aloha/bin/python" -c "import gym_aloha" 2>/dev/null; then
  REQ="$OPENPI/examples/aloha_sim/requirements.in"
  [ -f "$REQ" ] && "$ROOT/envs/aloha/bin/pip" install -q --retries 10 -r "$REQ" 2>&1 | tail -3
  "$ROOT/envs/aloha/bin/pip" install -q -e "$OPENPI/packages/openpi-client" 2>&1 | tail -2
fi
"$ROOT/envs/aloha/bin/python" -c "import gym_aloha; print('gym_aloha ok')" || die "gym_aloha"

step "5. pi0_aloha_sim checkpoint (~12 GB)"
CKPT="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_aloha_sim"
if [ ! -d "$CKPT" ]; then
  for attempt in 1 2 3; do
    echo "  download attempt $attempt"
    "$ROOT/envs/server/bin/python" -c "
from openpi.shared import download
print('at', download.maybe_download('gs://openpi-assets/checkpoints/pi0_aloha_sim'))" 2>&1 | tail -3
    [ -d "$CKPT" ] && break
    sleep 15
  done
fi
[ -d "$CKPT" ] || die "checkpoint not downloaded"
du -sh "$CKPT"

step "6. smoke: build the env exactly as aloha_adapt.py does"
# v1 used a bare gym.make and got KeyError 'agent_pos'. The repo uses obs_type=pixels_agent_pos.
MUJOCO_GL=egl "$ROOT/envs/aloha/bin/python" -c "
import gym_aloha, gymnasium as gym, numpy as np
e = gym.make('gym_aloha/AlohaTransferCube-v0', obs_type='pixels_agent_pos', render_mode='rgb_array')
o,_ = e.reset(seed=0)
assert 'agent_pos' in o and 'pixels' in o, list(o)
print('env ok; agent_pos dim', len(o['agent_pos']), '| cams', list(o['pixels']))
" || die "env smoke"

echo; echo "=== setup v2 DONE $(date -Is) ==="
df -h /data | tail -1
echo "server=$ROOT/envs/server  client=$ROOT/envs/aloha  ckpt=$CKPT"
