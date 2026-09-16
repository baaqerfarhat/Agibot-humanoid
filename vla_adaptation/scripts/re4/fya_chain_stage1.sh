#!/bin/bash
# FrozenYet Adaptive recovery campaign (prereg_records/PREREG_FYA_RECOVERY_DEADLINE_V1.md), Stage 1 only (split 2026-09-16 13:50: Stage 2 runs on GPU 0 in fya_stage2_gpu0.sh; both cards shared, never Yujin's jobs).
# GPU 1 (EGL alive there; GPU 0's EGL is dead), policy server on port 8000, shared card with Yujin's training (never touched).
# Order: A) fresh sources (policy)  B) qualification: predict -> replay -> calibrate  C) test: predict (frozen) -> replay -> evaluate + score
#        D) Stage 2 reacting-policy bridge (policy)  E) STATUS.
set -u
SP=/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad
OPENPI=/home/mtaheri/ws_AgibotX2/openpi; REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation
R=$REPO/results/frozen_yet_adaptive_deadline_v1; B=$R/configuration.json; MODEL=$R/stage0/forecast_model.json
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=$OPENPI/third_party/libero:$OPENPI/examples/libero:$REPO/openpi:$REPO/openpi/re4_theory
PY=$OPENPI/examples/libero/.venv/bin/python; SRV=$OPENPI/.venv/bin/python
SERVER=/home/mtaheri/ws_AgibotX2/Agibot-humanoid/vla_adaptation/openpi/ace_server.py
CTL=$SP/ctl_fya.json; ACK=$SP/ack_fya.json; PORT=8000
mkdir -p $R/qualification $R/physical_test $R/predictions $R/reacting_policy $R/analysis $R/sources
stamp() { echo "#### $1  $(date +%T)"; }
server_up() {
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6   # shared card: leave room for the other user's training
  cd $OPENPI && CUDA_VISIBLE_DEVICES=1 nohup $SRV $SERVER --port $PORT --control $CTL --ack $ACK > $SP/pi05_server_fya_$1.log 2>&1 &
  local t=0; until grep -q "listening" $SP/pi05_server_fya_$1.log 2>/dev/null; do sleep 10; t=$((t+10)); if [ $t -gt 1500 ]; then echo "SERVER NOT READY"; return 1; fi; done
  cd $REPO; return 0
}
server_down() {  # kill by the python PID, never by $! (which is the subshell)
  local P; P=$(ps -eo pid,args | grep "[a]ce_server.py" | grep "port $PORT" | awk '{print $1}'); [ -n "$P" ] && kill $P; sleep 15; return 0
}
QSEEDS=$(cat $R/sampler_seeds_qualification.txt); TSEEDS=$(cat $R/sampler_seeds_physical_test.txt)

# ---------------- A (continued): wait for the running qualification collection, then the test sources ----------------
stamp "A wait for the qualification collection already running"
while pgrep -f "error_signal.py.*qualification_spatial_init9_10" > /dev/null; do sleep 20; done; echo "qualification collection finished"
stamp "A physical-test sources states 11,12 (seeds 82000+2t+r)"
timeout 7200 $PY -u $REPO/openpi/error_signal.py --port $PORT --control $CTL --ack $ACK --suite libero_spatial --healthy-only --init-base 11 --episodes 20 \
  --sampler-seeds "$TSEEDS" --out $R/sources/physical_test_spatial_init11_12.json > $R/sources/physical_test_spatial_init11_12.log 2>&1; echo "exit $?"
server_down; echo "RUN DONE FYA sources"
cd $REPO && sha256sum $R/sources/*.json > $R/sources/SHA256SUMS

# ---------------- B. qualification: predict BEFORE replay, replay, calibrate ----------------
stamp "B predict qualification (frozen model, nominal commands only)"
python3 $REPO/openpi/re4_theory/fya_forecast.py predict --model $MODEL --bundle $B --log $R/sources/qualification_spatial_init9_10.json --out $R/predictions/qualification_predictions.csv > $R/predictions/qualification_predict.log 2>&1; echo "exit $?"
sha256sum $R/predictions/qualification_predictions.csv $R/predictions/qualification_predictions.trajectories.json $MODEL $B > $R/predictions/FROZEN_BEFORE_QUALIFICATION.sha256
stamp "B replay qualification"
timeout 7200 $PY $REPO/openpi/re4_theory/fya_continuations.py --log $R/sources/qualification_spatial_init9_10.json --bundle $B --episodes all --out $R/qualification/run.json --split-label qualification > $R/qualification/run.log 2>&1; echo "exit $?"
stamp "B calibrate + score qualification"
python3 $REPO/openpi/re4_theory/fya_forecast.py calibrate --predictions $R/predictions/qualification_predictions.csv --run $R/qualification/run.json --out $R/qualification/calibration.json > $R/qualification/calibrate.log 2>&1; echo "exit $?"
python3 $REPO/openpi/re4_theory/fya_score.py $R/qualification/run.json --out $R/qualification/score > $R/qualification/score.log 2>&1; echo "exit $?"
echo "RUN DONE FYA qualification"

# ---------------- C. locked test: predict (frozen), replay, evaluate, score ----------------
stamp "C predict test (frozen before any test branch exists)"
python3 $REPO/openpi/re4_theory/fya_forecast.py predict --model $MODEL --bundle $B --log $R/sources/physical_test_spatial_init11_12.json --out $R/predictions/test_predictions.csv > $R/predictions/test_predict.log 2>&1; echo "exit $?"
sha256sum $R/predictions/test_predictions.csv $R/predictions/test_predictions.trajectories.json $R/qualification/calibration.json $MODEL $B > $R/predictions/FROZEN_BEFORE_TEST.sha256
stamp "C replay locked test"
timeout 7200 $PY $REPO/openpi/re4_theory/fya_continuations.py --log $R/sources/physical_test_spatial_init11_12.json --bundle $B --episodes all --out $R/physical_test/run.json --split-label locked_physical_test > $R/physical_test/run.log 2>&1; echo "exit $?"
stamp "C evaluate + score test"
python3 $REPO/openpi/re4_theory/fya_forecast.py evaluate --predictions $R/predictions/test_predictions.csv --calibration $R/qualification/calibration.json --run $R/physical_test/run.json --out $R/analysis/test_evaluation > $R/analysis/test_evaluate.log 2>&1; echo "exit $?"
python3 $REPO/openpi/re4_theory/fya_score.py $R/physical_test/run.json --out $R/analysis/test_score > $R/analysis/test_score.log 2>&1; echo "exit $?"
echo "RUN DONE FYA physical test"

echo "FYA STAGE 1 DONE $(date +%T)"
