#!/usr/bin/env bash
# Generate a progress video for the latest (or a given) box-pickup checkpoint.
#
# Usage:
#   ./make_progress_video.sh                 # newest checkpoint of newest training run
#   ./make_progress_video.sh <run_dir>       # newest checkpoint of a specific run dir
#   ./make_progress_video.sh <model_*.pt>    # a specific checkpoint
#
# Pipeline (same as all previous progress videos):
#   1. eval_record_driver.py  (Isaac Sim, headless, GPU 0, demo mode: clean
#      rollout from t=0, no noise, no early termination) -> trajectory .npz
#   2. render_box_rollout.py  (MuJoCo offscreen)          -> .mp4
#
# Output: /home/baaqer/baaqer_ws/Agibot-humanoid/box_pickup/videos/x2_box_<run>_iter<N>.mp4
set -euo pipefail

LOGS=/home/baaqer/baaqer_ws/holosoma/logs/WholeBodyTracking
VIDEOS=/home/baaqer/baaqer_ws/Agibot-humanoid/box_pickup/videos
HSSIM_PY=/home/baaqer/.holosoma_deps/miniconda3/envs/hssim/bin/python
MJLAB_PY=/home/baaqer/baaqer_ws/mjlab/.venv/bin/python

# ---- resolve checkpoint ----
if [ $# -ge 1 ] && [[ "$1" == *.pt ]]; then
    CKPT="$1"
    RUN_DIR="$(dirname "$CKPT")"
else
    if [ $# -ge 1 ]; then
        RUN_DIR="$1"
    else
        RUN_DIR="$(ls -td "$LOGS"/*-locomotion | head -1)"
    fi
    CKPT="$(ls -t "$RUN_DIR"/model_*.pt | head -1)"
fi

RUN_NAME="$(basename "$RUN_DIR" | sed -E 's/^[0-9]{8}_[0-9]{6}-//; s/-locomotion$//; s/^x2_box_//')"
ITER="$(basename "$CKPT" .pt | sed 's/model_0*//')"
NPZ="/tmp/x2_box_${RUN_NAME}_iter${ITER}_rollout.npz"
MP4="$VIDEOS/x2_box_${RUN_NAME}_iter${ITER}.mp4"

echo "[video] run:        $RUN_NAME"
echo "[video] checkpoint: $CKPT (iteration $ITER)"

# ---- 1. record rollout (GPU 0 so it never disturbs training on GPU 1) ----
echo "[video] recording rollout (takes ~4 min, Isaac Sim startup dominates)..."
cd /home/baaqer/baaqer_ws/holosoma
OMNI_KIT_ACCEPT_EULA=1 CUDA_VISIBLE_DEVICES=0 "$HSSIM_PY" \
    src/holosoma/holosoma/eval_record_driver.py "$CKPT" "$NPZ" 400 demo \
    > /tmp/x2_box_video_eval.log 2>&1
[ -f "$NPZ" ] || { echo "[video] ERROR: rollout failed, see /tmp/x2_box_video_eval.log"; exit 1; }

# ---- 2. render to mp4 ----
echo "[video] rendering..."
cd /home/baaqer/baaqer_ws
MUJOCO_GL=egl "$MJLAB_PY" render_box_rollout.py "$NPZ" "$MP4" 2>/dev/null || true
[ -f "$MP4" ] || { echo "[video] ERROR: render failed"; exit 1; }

echo "[video] done: $MP4"
