#!/usr/bin/env bash
# Stand up the LIBERO half of the openpi stack on lambda.arcl.
#
# The ALOHA half is already installed by setup_lambda_aloha.sh and shares $ROOT/openpi and
# the python 3.11 server env. This adds: the pi05_libero checkpoint (~12 GB), a python 3.8
# CLIENT env with robosuite 1.4.1 + libero, and third_party/libero on the path.
#
# Lessons already paid for by the ALOHA setup, applied here from the start:
#   - every scratch path goes on /data; lambda's root fs is 100% full and pip reports the
#     resulting ENOSPC as a truncated download, not as a disk error
#   - the completion sentinel is a real dependency (robosuite), never the package itself;
#     `pip install -e .` registers a package even when its dependencies fail
#   - every step fails hard, so "DONE" means done
#   - pgrep patterns use a character class so they cannot match this script's own command line
set -uo pipefail

ROOT=/data/fxxie/vla
OPENPI=$ROOT/openpi
CONDA=$HOME/miniconda3/bin/conda
export OPENPI_DATA_HOME=$ROOT/assets
export TMPDIR=$ROOT/tmp PIP_CACHE_DIR=$ROOT/pipcache XDG_CACHE_HOME=$ROOT/cache
export CONDA_PKGS_DIRS=$ROOT/condapkgs
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$CONDA_PKGS_DIRS" "$OPENPI_DATA_HOME"

exec > >(tee -a "$ROOT/setup_libero.log") 2>&1
echo "=== LIBERO setup start $(date -Is) ==="
df -h /data | tail -1
step() { echo; echo "--- [$(date +%H:%M:%S)] $* ---"; }
die()  { echo "FATAL: $*"; exit 1; }

[ -d "$OPENPI/.git" ] || die "openpi missing; run setup_lambda_aloha.sh first"

step "1. client env (python 3.8, robosuite 1.4.1 + libero)"
if [ ! -d "$ROOT/envs/libero" ]; then
  "$CONDA" create -y -p "$ROOT/envs/libero" python=3.8 2>&1 | tail -2 || die "conda create"
fi
"$ROOT/envs/libero/bin/python" -V || die "client env"
"$ROOT/envs/libero/bin/pip" install -q --upgrade pip

step "2. install the client requirements"
if ! "$ROOT/envs/libero/bin/python" -c "import robosuite" 2>/dev/null; then
  REQ="$OPENPI/examples/libero/requirements.txt"
  [ -f "$REQ" ] || REQ="$OPENPI/examples/libero/requirements.in"
  ok=0
  for attempt in 1 2 3; do
    echo "  attempt $attempt"
    # requirements.txt pins torch==1.11.0+cu113. CUDA-suffixed wheels live ONLY on
    # PyTorch's own index, never on PyPI, so a plain `pip install -r` fails with
    # "No matching distribution found" and reads like a bad pin rather than a missing
    # index. This is the third undocumented detail in this stack, after lerobot's
    # [tool.uv.sources] pin (which pip silently ignores) and ENOSPC-as-truncated-download.
    [ -f "$REQ" ] && "$ROOT/envs/libero/bin/pip" install --retries 10 --timeout 120 \
        --extra-index-url https://download.pytorch.org/whl/cu113 -r "$REQ" 2>&1 | tail -4
    "$ROOT/envs/libero/bin/pip" install -q --retries 10 -e "$OPENPI/packages/openpi-client" 2>&1 | tail -2
    if "$ROOT/envs/libero/bin/python" -c "import robosuite" 2>/dev/null; then ok=1; break; fi
    sleep 10
  done
  [ "$ok" = 1 ] || die "robosuite did not install after 3 attempts"
fi

step "3. install LIBERO's OWN requirements"
# NOT the same file as examples/libero/requirements.txt. third_party/libero declares
# bddl==1.0.1, easydict, robomimic, hydra, gym==0.25.2 and more; installing only openpi's
# client requirements leaves LIBERO importable but its envs subpackage broken, which
# surfaces one missing module at a time (bddl, then easydict, ...) rather than as one
# clear failure. Fifth undocumented detail in this stack.
LREQ="$OPENPI/third_party/libero/requirements.txt"
if [ -f "$LREQ" ]; then
  "$ROOT/envs/libero/bin/pip" install --retries 10 --timeout 120 \
      --extra-index-url https://download.pytorch.org/whl/cu113 -r "$LREQ" 2>&1 | tail -3
fi
# robosuite warns on every run without a private macro file; create it once.
"$ROOT/envs/libero/bin/python" \
  "$ROOT/envs/libero/lib/python3.8/site-packages/robosuite/scripts/setup_macros.py" 2>&1 | tail -1

step "4. install third_party/libero"
if ! "$ROOT/envs/libero/bin/python" -c "import libero" 2>/dev/null; then
  [ -d "$OPENPI/third_party/libero" ] || die "third_party/libero missing (submodule not checked out)"
  "$ROOT/envs/libero/bin/pip" install -q --retries 10 -e "$OPENPI/third_party/libero" 2>&1 | tail -3
fi
"$ROOT/envs/libero/bin/python" -c "
import robosuite, libero, numpy
print('  robosuite', robosuite.__version__, '| numpy', numpy.__version__, '| libero ok')" || die "client imports"

step "5. pi05_libero checkpoint (~12 GB)"
CKPT="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero"
if [ ! -d "$CKPT" ]; then
  for attempt in 1 2 3; do
    echo "  download attempt $attempt"
    "$ROOT/envs/server/bin/python" -c "
from openpi.shared import download
print('at', download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero'))" 2>&1 | tail -3
    [ -d "$CKPT" ] && break
    sleep 15
  done
fi
[ -d "$CKPT" ] || die "checkpoint not downloaded"
du -sh "$CKPT"

step "6. create LIBERO's config non-interactively"
# libero/libero/__init__.py PROMPTS on first import if its config file is absent
# ("Do you want to specify a custom path for the dataset folder? (Y/N)"), which is an
# EOFError under nohup with no stdin. Answer once so the config gets written with defaults.
# Fourth undocumented detail in this stack, after lerobot's [tool.uv.sources] pin, ENOSPC
# reported as a truncated download, and torch's CUDA wheels needing --extra-index-url.
echo "N" | MUJOCO_GL=egl "$ROOT/envs/libero/bin/python" -c "import libero.libero" 2>&1 | tail -2

step "7. smoke: build a LIBERO env in the client, as paired_probe.py does"
MUJOCO_GL=egl PYTHONPATH="$OPENPI/third_party/libero:$OPENPI/examples/libero:$ROOT/repo/openpi" \
  "$ROOT/envs/libero/bin/python" -c "
from libero.libero import benchmark
bd = benchmark.get_benchmark_dict()['libero_spatial']()
print('  suite ok, tasks:', bd.n_tasks)
" || die "libero env smoke"

echo; echo "=== LIBERO setup DONE $(date -Is) ==="
df -h /data | tail -1
echo "client=$ROOT/envs/libero  server=$ROOT/envs/server  ckpt=$CKPT"
