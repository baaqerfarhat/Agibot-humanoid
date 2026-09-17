#!/bin/bash
# Crossed command-stream replay v1 (PREREG_FYA_CROSSED_REPLAY_V1.md): 18 eligible keys x 7 continuations, then score.
OPENPI=/home/mtaheri/ws_AgibotX2/openpi; REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation; R=$REPO/results/fya_crossed_replay_v1
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=$OPENPI/third_party/libero:$OPENPI/examples/libero:$REPO/openpi:$REPO/openpi/re4_theory
PY=$OPENPI/examples/libero/.venv/bin/python; mkdir -p $R/runs $R/analysis
echo "#### CROSSED collect  $(date +%T)"
timeout 7200 $PY $REPO/openpi/re4_theory/fya_crossed_replay.py --keys eligible --cells all --label crossed_v1_locked --out $R/runs/crossed_run.json > $R/runs/crossed_run.log 2>&1; echo "exit $?"
echo "#### CROSSED score  $(date +%T)"
cd $REPO && python3 openpi/re4_theory/fya_crossed_score.py $R/runs/crossed_run.json --out $R/analysis > $R/analysis/score.log 2>&1; echo "exit $?"
echo "CROSSED DONE $(date +%T)"
