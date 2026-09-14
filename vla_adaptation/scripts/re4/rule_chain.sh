#!/bin/bash
# Constants by the rule (prereg PREREG_RULE_CONSTANTS.md): innovation law + normaliser over the corrected
# channels, four faulted cells then four healthy controls, shipped calibration, own pi0.5 server.
SP=/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad
OPENPI=/home/mtaheri/ws_AgibotX2/openpi; REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation; R=$REPO/results/rule_constants
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=$OPENPI/third_party/libero:$OPENPI/examples/libero:$REPO/openpi
PY=$OPENPI/examples/libero/.venv/bin/python
cd $OPENPI && CUDA_VISIBLE_DEVICES=1 nohup $OPENPI/.venv/bin/python /home/mtaheri/ws_AgibotX2/Agibot-humanoid/vla_adaptation/openpi/ace_server.py --port 8000 --control $SP/ctl.json --ack $SP/ack.json > $SP/pi05_server_rule.log 2>&1 &
t=0; until grep -q "listening" $SP/pi05_server_rule.log 2>/dev/null; do sleep 10; t=$((t+10)); if [ $t -gt 1500 ]; then echo "SERVER NOT READY"; exit 1; fi; done
BASE="--port 8000 --control $SP/ctl.json --ack $SP/ack.json --gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.15 --scenario-reset --corr-dims 3,4,5 --law innov --norm-channels corrected --log $REPO/results/phase05/error_signal_so3.json --openloop $REPO/results/phase05/openloop_so3.json"
go() { local id=$1 suite=$2 n=$3 sev=$4; local d=$R/$id; mkdir -p $d; echo "#### RULE $id  $(date +%T)"
  timeout 21600 $PY -u $REPO/openpi/adaptive_law.py $BASE --suite $suite --episodes $n --sev $sev --timing $d/timing.jsonl --out $d/result.json > $d/run.log 2>&1
  echo "exit $?  $(grep -E ': [0-9]+/[0-9]+ =' $d/run.log | tr '\n' ' ')"
  (cd $REPO && python3 openpi/re4_record.py record $d/result.json --part rule_constants --run-id $id --cohort $suite --timing $d/timing.jsonl --root $R/.. >> $d/record.log 2>&1) || echo "RECORD FAILED $id"
  echo "RUN DONE RULE $id"; return 0; }
go faulted_libero_spatial libero_spatial 20 0.05
go faulted_libero_object  libero_object  20 0.05
go faulted_libero_goal    libero_goal    40 0.05
go faulted_libero_10      libero_10      40 0.05
go healthy_libero_spatial libero_spatial 20 0.0
go healthy_libero_object  libero_object  20 0.0
go healthy_libero_goal    libero_goal    40 0.0
go healthy_libero_10      libero_10      40 0.0
P=$(ps -eo pid,args | grep "[a]ce_server.py" | awk '{print $1}'); [ -n "$P" ] && kill $P
echo "RULE CHAIN DONE $(date +%T)"
