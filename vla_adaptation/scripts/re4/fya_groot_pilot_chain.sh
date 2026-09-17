#!/bin/bash
# GR00T healthy crossed pilot (PREREG_FYA_GROOT_HEALTHY_CROSSED_V1.md part B): server self-test, 12 source episodes on one process, diagonal replay.
SP=/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad
OPENPI=/home/mtaheri/ws_AgibotX2/openpi; REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation; GR=/home/mtaheri/ws_AgibotX2/Isaac-GR00T
R=$REPO/results/fya_groot_healthy_crossed_v1/pilot; S=$R/sources; M=$R/manifest.json; mkdir -p $S $R/runs $R/analysis
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=$OPENPI/third_party/libero:$OPENPI/examples/libero:$REPO/openpi:$REPO/openpi/re4_theory
PY=$OPENPI/examples/libero/.venv/bin/python; GPY=$GR/.venv/bin/python; CKPT=$GR/checkpoints/GR00T-N1.7-LIBERO/libero_spatial
CTL=$SP/ctl_groot.json; ACK=$SP/ack_groot.json; PORT=8003
stamp() { echo "#### $1  $(date +%T)"; }
stamp "GR00T self-test"; cd $GR && GROOT_HF_LOCAL_FIRST=1 CUDA_VISIBLE_DEVICES=1 $GPY $REPO/openpi/groot_server.py --model-path $CKPT --port $PORT --control $CTL --ack $ACK --selftest-seed > $R/selftest.log 2>&1; echo "selftest exit $?"; tail -1 $R/selftest.log | cut -c1-300
grep -q '"passed": true' $R/selftest.log || { echo "SELFTEST FAILED"; echo "GROOT PILOT DONE $(date +%T)"; exit 0; }
stamp "GR00T server up"; cd $GR && GROOT_HF_LOCAL_FIRST=1 CUDA_VISIBLE_DEVICES=1 nohup $GPY $REPO/openpi/groot_server.py --model-path $CKPT --port $PORT --control $CTL --ack $ACK --calllog $R/server_calllog.jsonl > $SP/groot_server_pilot.log 2>&1 &
t=0; until grep -q "listening" $SP/groot_server_pilot.log 2>/dev/null; do sleep 10; t=$((t+10)); if [ $t -gt 1800 ]; then echo "SERVER NOT READY"; exit 1; fi; done; cd $REPO
runsrc() { local name=$1 arms=$2
  stamp "$name (gr00t, shared process)"
  timeout 7200 $PY -u $REPO/openpi/adaptive_law.py --port $PORT --control $CTL --ack $ACK --suite libero_spatial --replan-steps 8 \
    --log $REPO/results/phase05/error_signal_so3.json --openloop $REPO/results/phase05/openloop_so3.json --gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.05 \
    --corr-dims 3,4,5 --norm-channels all --scenario-reset --manifest $M --episodes 4 --arms $arms --law legacy --baseline none --fault-vec 0,0,0,0,0,0 --onset 40 --adapt-from 30 \
    --out $S/$name.json --telemetry $S/${name}_telemetry.jsonl > $S/$name.log 2>&1; echo "exit $?"; }
T0=$(date +%s); runsrc healthy_off frozen; runsrc healthy_nt adaptive; runsrc healthy_off_dup frozen; echo "source seconds $(( $(date +%s) - T0 ))"
P=$(ps -eo pid,args | grep "[g]root_server.py" | grep "port $PORT" | awk '{print $1}'); [ -n "$P" ] && kill $P; sleep 10
cd $REPO && for f in $S/*_telemetry.jsonl; do gzip -f $f; done
stamp "pilot coupling"; python3 openpi/re4_theory/fya_source_coupling_report.py --src $S --a healthy_off --b healthy_nt --prefix 30 --out $R/analysis/coupling_off_vs_nt_prefix30.json > /dev/null 2>&1; python3 openpi/re4_theory/fya_source_coupling_report.py --src $S --a healthy_off --b healthy_off_dup --out $R/analysis/coupling_off_vs_dup_full.json > /dev/null 2>&1
stamp "pilot extraction"; python3 openpi/re4_theory/fya_crossed_extract.py --src $S --manifest $M --out $R --arms healthy_off,healthy_off,healthy_nt,healthy_off_dup --config-arm healthy_nt > $R/analysis/extraction.log 2>&1; echo "exit $?"; grep -o '"n_eligible": [0-9]*' $R/analysis/extraction.log
stamp "pilot diagonal replay"; T1=$(date +%s); timeout 3600 $PY openpi/re4_theory/fya_crossed_replay.py --root $R --keys all --cells diagonal --matrix healthy --label groot_pilot_diagonal --out $R/runs/pilot_diagonal.json > $R/runs/pilot_diagonal.log 2>&1; echo "exit $?"; echo "replay seconds $(( $(date +%s) - T1 ))"; grep "^key" $R/runs/pilot_diagonal.log
echo "GROOT PILOT DONE $(date +%T)"
