#!/usr/bin/env bash
# Recreate the holosoma X2 box-pickup training setup on a new machine.
#
# Usage:  ./setup_holosoma_x2.sh [target_dir]     (default: ../../holosoma)
#
# 1. Clones amazon-far/holosoma (the upstream this work was built on).
# 2. Applies the overlay in ./holosoma_overlay (X2 robot/task configs, sampler
#    fix, hand-proximity rewards, eval/record driver, motion data, URDFs/XMLs).
# 3. Fills in the X2 STL meshes from mjlab's asset zoo in THIS repo (identical
#    files; they are not duplicated in git to keep the repo small).
#
# After this, follow holosoma's own installation docs (Isaac Sim / IsaacLab),
# then train with:
#   python src/holosoma/holosoma/train_agent.py exp:x2-31dof-wbt-w-object \
#       logger:disabled --training.num-envs 4096 --training.name x2_box
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"           # Agibot-humanoid
OVERLAY="$REPO_DIR/box_pickup/holosoma_overlay"
MESHES="$REPO_DIR/mjlab/src/mjlab/asset_zoo/robots/x2/xmls/assets"
TARGET="${1:-$REPO_DIR/../holosoma}"

if [ ! -d "$TARGET/.git" ]; then
    git clone https://github.com/amazon-far/holosoma.git "$TARGET"
fi

echo "[setup] applying X2 overlay -> $TARGET"
rsync -a "$OVERLAY/" "$TARGET/"

echo "[setup] installing X2 meshes from mjlab asset zoo"
mkdir -p "$TARGET/src/holosoma/holosoma/data/robots/x2/meshes" \
         "$TARGET/src/holosoma_retargeting/holosoma_retargeting/models/x2/meshes" \
         "$TARGET/src/holosoma_retargeting/holosoma_retargeting/models/x2/assets"
rsync -a "$MESHES/" "$TARGET/src/holosoma/holosoma/data/robots/x2/meshes/"
rsync -a "$MESHES/" "$TARGET/src/holosoma_retargeting/holosoma_retargeting/models/x2/meshes/"
rsync -a "$MESHES/" "$TARGET/src/holosoma_retargeting/holosoma_retargeting/models/x2/assets/"

echo "[setup] done. Training entry point:"
echo "  exp:x2-31dof-wbt-w-object   (box pickup, whole-body tracking)"
