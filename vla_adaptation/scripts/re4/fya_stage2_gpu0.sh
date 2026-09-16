#!/bin/bash
# FrozenYet Adaptive recovery campaign (prereg_records/PREREG_FYA_RECOVERY_DEADLINE_V1.md), Stage 2 only, on GPU 0 (policy server CUDA device 0, port 8001; rendering on GPU 1 because GPU 0's EGL is dead). Shared with Yujin's training, never interrupted.
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
CTL=$SP/ctl_fya0.json; ACK=$SP/ack_fya0.json; PORT=8001
mkdir -p $R/qualification $R/physical_test $R/predictions $R/reacting_policy $R/analysis $R/sources
stamp() { echo "#### $1  $(date +%T)"; }
server_up() {
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6   # shared card: leave room for the other user's training
  cd $OPENPI && CUDA_VISIBLE_DEVICES=0 nohup $SRV $SERVER --port $PORT --control $CTL --ack $ACK > $SP/pi05_server_fya_$1.log 2>&1 &
  local t=0; until grep -q "listening" $SP/pi05_server_fya_$1.log 2>/dev/null; do sleep 10; t=$((t+10)); if [ $t -gt 1500 ]; then echo "SERVER NOT READY"; return 1; fi; done
  cd $REPO; return 0
}
server_down() {  # kill by the python PID, never by $! (which is the subshell)
  local P; P=$(ps -eo pid,args | grep "[a]ce_server.py" | grep "port $PORT" | awk '{print $1}'); [ -n "$P" ] && kill $P; sleep 15; return 0
}
QSEEDS=$(cat $R/sampler_seeds_qualification.txt); TSEEDS=$(cat $R/sampler_seeds_physical_test.txt)

# ---------------- D. Stage 2 reacting-policy bridge (policy inference) ----------------
stamp "D stage 2: server up"; server_up stage2 || exit 1
COMMON="--port $PORT --control $CTL --ack $ACK --suite libero_spatial --log $REPO/results/phase05/error_signal_so3.json --openloop $REPO/results/phase05/openloop_so3.json \
  --gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.05 --corr-dims 3,4,5 --norm-channels all --scenario-reset --manifest $R/stage2_manifest.json --episodes 20"
run2() {  # name, arms, law, fault-vec, onset(env steps), adapt-from(policy steps)
  local name=$1 arms=$2 law=$3 fvec=$4 onset=$5 af=$6
  stamp "D $name"
  timeout 7200 $PY -u $REPO/openpi/adaptive_law.py $COMMON --arms $arms --law $law --fault-vec $fvec --onset $onset --adapt-from $af \
    --out $R/reacting_policy/$name.json --telemetry $R/reacting_policy/${name}_telemetry.jsonl > $R/reacting_policy/$name.log 2>&1; echo "exit $?"
}
# prospective arm order: healthy off, nt, innovation; faulted innovation, off, nt; delayed nt, innovation
run2 healthy_off        frozen   legacy 0,0,0,0,0,0    40 30
run2 healthy_nt         adaptive legacy 0,0,0,0,0,0    40 30
run2 healthy_innovation adaptive innov  0,0,0,0,0,0    40 30
run2 fault_innovation   adaptive innov  0,0,0,0,0.05,0 40 30
run2 fault_off          frozen   legacy 0,0,0,0,0.05,0 40 30
run2 fault_nt           adaptive legacy 0,0,0,0,0.05,0 40 30
run2 delay_nt           adaptive legacy 0,0,0,0,0.05,0 40 40
run2 delay_innovation   adaptive innov  0,0,0,0,0.05,0 40 40
server_down; echo "RUN DONE FYA stage 2"
cd $REPO && for f in $R/reacting_policy/*_telemetry.jsonl; do gzip -f $f; done
echo "FYA STAGE 2 DONE $(date +%T)"
