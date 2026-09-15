#!/bin/bash
# Healthy source collections for E1/E2 (PREREG_UNIFIED_PARTITIONS.md), after the DC chain; then the
# E1/E2 tool smoke tests on the held-out log (simulator only, no server).
SP=/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad
OPENPI=/home/mtaheri/ws_AgibotX2/openpi; REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation; R=$REPO/results/iclr_unified_v1/sources
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=$OPENPI/third_party/libero:$OPENPI/examples/libero:$REPO/openpi:$REPO/openpi/re4_theory
PY=$OPENPI/examples/libero/.venv/bin/python
until grep -q "DC CHAIN DONE" $SP/dc_chain.log 2>/dev/null; do sleep 60; done; sleep 20
mkdir -p $R
cd $OPENPI && CUDA_VISIBLE_DEVICES=1 nohup $OPENPI/.venv/bin/python /home/mtaheri/ws_AgibotX2/Agibot-humanoid/vla_adaptation/openpi/ace_server.py --port 8000 --control $SP/ctl.json --ack $SP/ack.json > $SP/pi05_server_prep.log 2>&1 &
t=0; until grep -q "listening" $SP/pi05_server_prep.log 2>/dev/null; do sleep 10; t=$((t+10)); if [ $t -gt 1500 ]; then echo "SERVER NOT READY"; exit 1; fi; done
col() { local id=$1 base=$2 n=$3; echo "#### PREP $id  $(date +%T)"; timeout 14400 $PY -u $REPO/openpi/error_signal.py --port 8000 --control $SP/ctl.json --ack $SP/ack.json --suite libero_spatial --healthy-only --init-base $base --episodes $n --out $R/$id.json > $R/$id.log 2>&1; echo "exit $?"; echo "RUN DONE PREP $id"; return 0; }
col e2_fit_qual_spatial_init40_43 40 40
col e1_fitqual_spatial_init39 39 10
col e1_test_spatial_init34 34 10
P=$(ps -eo pid,args | grep "[a]ce_server.py" | awk '{print $1}'); [ -n "$P" ] && kill $P; sleep 15
echo "#### SMOKE e1  $(date +%T)"; timeout 1800 $PY $REPO/openpi/re4_theory/e1_continuations.py --log $REPO/results/heldout/error_signal_init25.json --episodes 0 --checkpoints 30 --horizon 12 --hold-estimate 0,0,0,0,0.04,0 --split-label smoke --out $SP/e1_smoke.json > $SP/e1_smoke.log 2>&1; echo "exit $?"; echo "RUN DONE SMOKE e1"
echo "#### SMOKE e2  $(date +%T)"; timeout 1800 $PY $REPO/openpi/re4_theory/e2_probe.py --log $REPO/results/heldout/error_signal_init25.json --episodes 0 --checkpoints 30 --horizon 20 --split-label smoke --out $SP/e2_smoke.json > $SP/e2_smoke.log 2>&1; echo "exit $?"; echo "RUN DONE SMOKE e2"
echo "PREP CHAIN DONE $(date +%T)"
