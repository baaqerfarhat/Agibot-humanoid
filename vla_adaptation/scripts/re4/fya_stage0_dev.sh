#!/bin/bash
# Stage 0 development replays (old sources, new driver/format): state 39 (E1 v2 fit/qualification sources) and state 33
# (E1 v2 locked test sources, inspected -> development). No fresh data. CPU/EGL only, no policy server.
OPENPI=/home/mtaheri/ws_AgibotX2/openpi; REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation; R=$REPO/results/frozen_yet_adaptive_deadline_v1/stage0
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=$OPENPI/third_party/libero:$OPENPI/examples/libero:$REPO/openpi:$REPO/openpi/re4_theory
PY=$OPENPI/examples/libero/.venv/bin/python; B=$REPO/results/frozen_yet_adaptive_deadline_v1/configuration.json
for st in 39 33; do
  case $st in 39) LOG=$REPO/results/iclr_unified_v1/sources/e1_fitqual_spatial_init39.json;; 33) LOG=$REPO/results/iclr_unified_v1/sources/e1_test_spatial_init33.json;; esac
  echo "#### STAGE0 dev replay state $st  $(date +%T)"
  timeout 3600 $PY $REPO/openpi/re4_theory/fya_continuations.py --log $LOG --bundle $B --episodes all --out $R/dev_state${st}.json --split-label stage0_development_state${st} > $R/dev_state${st}.log 2>&1; echo "exit $?"
done
echo "STAGE0 DEV DONE $(date +%T)"
