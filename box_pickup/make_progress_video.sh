#!/usr/bin/env bash
# Generate a progress video for a whole-body-tracking training run
# (box pickup, slope crawl, ...).
#
# Usage:
#   ./make_progress_video.sh                 # interactive: lists detected runs, you pick one
#   ./make_progress_video.sh <run_dir>       # newest checkpoint of a specific run dir
#   ./make_progress_video.sh <model_*.pt>    # a specific checkpoint
#
# Pipeline (same as all previous progress videos):
#   1. eval_record_driver.py  (Isaac Sim, headless, demo mode: clean rollout
#      from t=0, no noise, no early termination) -> trajectory .npz
#      Runs on the GPU with the most free memory so it disturbs training least.
#   2. render_*_rollout.py    (MuJoCo offscreen)  -> .mp4
#      Renderer picked by run type: crawl runs get the slope-terrain renderer,
#      box runs get the box renderer.
#
# Output: /home/baaqer/baaqer_ws/Agibot-humanoid/box_pickup/videos/x2_<type>_<run>_iter<N>.mp4
set -euo pipefail

LOGS=/home/baaqer/baaqer_ws/holosoma/logs/WholeBodyTracking
VIDEOS=/home/baaqer/baaqer_ws/Agibot-humanoid/box_pickup/videos
HSSIM_PY=/home/baaqer/.holosoma_deps/miniconda3/envs/hssim/bin/python
MJLAB_PY=/home/baaqer/baaqer_ws/mjlab/.venv/bin/python
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- resolve checkpoint ----
if [ $# -ge 1 ] && [[ "$1" == *.pt ]]; then
    CKPT="$1"
    RUN_DIR="$(dirname "$CKPT")"
elif [ $# -ge 1 ]; then
    RUN_DIR="$1"
    CKPT="$(ls -t "$RUN_DIR"/model_*.pt | head -1 || true)"
else
    # ---- detect runs that have checkpoints, newest first ----
    mapfile -t RUNS < <(ls -td "$LOGS"/*-locomotion 2>/dev/null | while read -r r; do
        ls "$r"/model_*.pt >/dev/null 2>&1 && echo "$r"
    done)
    [ "${#RUNS[@]}" -gt 0 ] || { echo "No training runs with checkpoints found in $LOGS"; exit 1; }

    echo "Detected training runs:"
    for i in "${!RUNS[@]}"; do
        r="${RUNS[$i]}"
        name="$(basename "$r")"
        # NOTE: "|| true" guards against SIGPIPE (exit 141) under pipefail:
        # head closes the pipe after one line and ls dies mid-write on runs
        # with many checkpoints, which would silently kill the whole script.
        latest="$(ls -t "$r"/model_*.pt | head -1 || true)"
        iter="$(basename "$latest" .pt | sed -E 's/^model_0*//')"
        [ -n "$iter" ] || iter=0
        # mark a run as live only if a train_agent process uses its training
        # name AND this is the newest run dir for that name (restarted runs
        # share the name; only the newest one is actually being written to).
        base_name="$(echo "$name" | sed -E 's/^[0-9]{8}_[0-9]{6}-//; s/-locomotion$//')"
        live=""
        if pgrep -f "train_agent.py.*--training.name $base_name( |$)" >/dev/null 2>&1; then
            newest_for_name="$(ls -td "$LOGS"/*"-${base_name}-locomotion" 2>/dev/null | head -1 || true)"
            [ "$r" = "$newest_for_name" ] && live="  [TRAINING NOW]"
        fi
        printf '  %d) %s  (latest: iter %s)%s\n' "$((i + 1))" "$name" "$iter" "$live"
    done
    printf 'Which policy do you want a progress video of? [1-%d]: ' "${#RUNS[@]}"
    read -r CHOICE
    [[ "$CHOICE" =~ ^[0-9]+$ ]] && [ "$CHOICE" -ge 1 ] && [ "$CHOICE" -le "${#RUNS[@]}" ] \
        || { echo "Invalid choice."; exit 1; }
    RUN_DIR="${RUNS[$((CHOICE - 1))]}"
    CKPT="$(ls -t "$RUN_DIR"/model_*.pt | head -1 || true)"
fi

RUN_BASE="$(basename "$RUN_DIR" | sed -E 's/^[0-9]{8}_[0-9]{6}-//; s/-locomotion$//')"
ITER="$(basename "$CKPT" .pt | sed -E 's/^model_0*//')"
[ -n "$ITER" ] || ITER=0

# ---- pick renderer + naming by run type ----
if [[ "$RUN_BASE" == *crawl* ]]; then
    TYPE="crawl"
    RENDERER="$SCRIPTS_DIR/render_crawl_rollout.py"
    # Full palmflat slope crawl is ~19 s @ 50 Hz.
    STEPS=1000
    MOTION=""
else
    TYPE="box"
    RENDERER="$SCRIPTS_DIR/render_box_rollout.py"
    # Box runs (v29+) train on THREE speed variants of the clip; without a
    # motion override the demo recorder plays a RANDOM one. Pin the nominal
    # clip and record its FULL length (pickup + set-down + 3 s end hold) so
    # the set-down is never cut off (the old fixed 460 frames truncated it).
    MOTION=/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/x2_31dof/whole_body_tracking/box_multispeed/box_speed100.npz
    STEPS="$("$HSSIM_PY" -c "import numpy as np; print(np.load('$MOTION')['joint_pos'].shape[0] - 1)")"
fi
RUN_NAME="$(echo "$RUN_BASE" | sed -E "s/^x2_(box|crawl)_?//")"
[ -n "$RUN_NAME" ] || RUN_NAME="$RUN_BASE"
NPZ="/tmp/x2_${TYPE}_${RUN_NAME}_iter${ITER}_rollout.npz"
MP4="$VIDEOS/x2_${TYPE}_${RUN_NAME}_iter${ITER}.mp4"

# ---- pick the GPU with the most free memory for the eval rollout ----
EVAL_GPU="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -rn | head -1 | cut -d, -f1 | tr -d ' ' || true)"
[ -n "$EVAL_GPU" ] || EVAL_GPU=0

echo "[video] run:        $RUN_BASE  (type: $TYPE)"
echo "[video] checkpoint: $CKPT (iteration $ITER)"
echo "[video] eval GPU:   $EVAL_GPU"
[ -n "$MOTION" ] && echo "[video] motion:     $(basename "$MOTION") ($STEPS frames)"

# ---- 1. record rollout ----
echo "[video] recording rollout (takes ~4 min, Isaac Sim startup dominates)..."
cd /home/baaqer/baaqer_ws/holosoma
OMNI_KIT_ACCEPT_EULA=1 CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$HSSIM_PY" \
    src/holosoma/holosoma/eval_record_driver.py "$CKPT" "$NPZ" "$STEPS" demo ${MOTION:+"$MOTION"} \
    > "/tmp/x2_${TYPE}_video_eval.log" 2>&1
[ -f "$NPZ" ] || { echo "[video] ERROR: rollout failed, see /tmp/x2_${TYPE}_video_eval.log"; exit 1; }

# ---- 2. render to mp4 ----
echo "[video] rendering..."
RENDER_LOG="/tmp/x2_${TYPE}_video_render.log"
# EGL teardown often prints a harmless destructor error; keep the log so a
# real failure is visible. Prefer mjlab venv, fall back to hsretargeting.
if ! MUJOCO_GL=egl "$MJLAB_PY" "$RENDERER" "$NPZ" "$MP4" >"$RENDER_LOG" 2>&1; then
    if [ ! -f "$MP4" ]; then
        HS_PY=/home/baaqer/baaqer_ws/holosoma/.venv/hsretargeting/bin/python
        MUJOCO_GL=egl "$HS_PY" "$RENDERER" "$NPZ" "$MP4" >>"$RENDER_LOG" 2>&1 || true
    fi
fi
[ -f "$MP4" ] || { echo "[video] ERROR: render failed, see $RENDER_LOG"; exit 1; }

echo "[video] done: $MP4"
