#!/bin/bash
# GR00T healthy crossed replication, part B (confirmatory; run only after the pilot passed and part B was registered):
# 40 keys (states 23,24,29,31, seed 89001), three arms on one server process, extraction, healthy matrix, scoring.
SP=/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad
OPENPI=/home/mtaheri/ws_AgibotX2/openpi; REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation; GR=/home/mtaheri/ws_AgibotX2/Isaac-GR00T
R=$REPO/results/fya_groot_healthy_crossed_v1; S=$R/sources; M=$R/manifest.json; mkdir -p $S $R/runs $R/analysis
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=$OPENPI/third_party/libero:$OPENPI/examples/libero:$REPO/openpi:$REPO/openpi/re4_theory
PY=$OPENPI/examples/libero/.venv/bin/python; GPY=$GR/.venv/bin/python; CKPT=$GR/checkpoints/GR00T-N1.7-LIBERO/libero_spatial
CTL=$SP/ctl_groot.json; ACK=$SP/ack_groot.json; PORT=8003
stamp() { echo "#### $1  $(date +%T)"; }
stamp "GR00T server up (confirmatory)"; cd $GR && GROOT_HF_LOCAL_FIRST=1 CUDA_VISIBLE_DEVICES=1 nohup $GPY $REPO/openpi/groot_server.py --model-path $CKPT --port $PORT --control $CTL --ack $ACK --calllog $R/server_calllog.jsonl > $SP/groot_server_confirm.log 2>&1 &
t=0; until grep -q "listening" $SP/groot_server_confirm.log 2>/dev/null; do sleep 10; t=$((t+10)); if [ $t -gt 1800 ]; then echo "SERVER NOT READY"; exit 1; fi; done; cd $REPO
runsrc() { local name=$1 arms=$2
  stamp "$name (gr00t, shared process)"
  timeout 21600 $PY -u $REPO/openpi/adaptive_law.py --port $PORT --control $CTL --ack $ACK --suite libero_spatial --replan-steps 8 \
    --log $REPO/results/phase05/error_signal_so3.json --openloop $REPO/results/phase05/openloop_so3.json --gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.05 \
    --corr-dims 3,4,5 --norm-channels all --scenario-reset --manifest $M --episodes 40 --arms $arms --law legacy --baseline none --fault-vec 0,0,0,0,0,0 --onset 40 --adapt-from 30 \
    --out $S/$name.json --telemetry $S/${name}_telemetry.jsonl --timing $S/${name}_timing.jsonl > $S/$name.log 2>&1; echo "exit $?"; }
runsrc healthy_off frozen; runsrc healthy_nt adaptive; runsrc healthy_off_dup frozen
P=$(ps -eo pid,args | grep "[g]root_server.py" | grep "port $PORT" | awk '{print $1}'); [ -n "$P" ] && kill $P; sleep 10
cd $REPO && for f in $S/*_telemetry.jsonl; do gzip -f $f; done
stamp "coupling"; python3 openpi/re4_theory/fya_source_coupling_report.py --src $S --a healthy_off --b healthy_nt --prefix 30 --out $R/analysis/coupling_off_vs_nt_prefix30.json > /dev/null 2>&1; python3 openpi/re4_theory/fya_source_coupling_report.py --src $S --a healthy_off --b healthy_off_dup --out $R/analysis/coupling_off_vs_dup_full.json > /dev/null 2>&1
stamp "extraction"; python3 openpi/re4_theory/fya_crossed_extract.py --src $S --manifest $M --out $R --arms healthy_off,healthy_off,healthy_nt,healthy_off_dup --config-arm healthy_nt > $R/analysis/extraction.log 2>&1; echo "exit $?"; grep -o '"n_eligible": [0-9]*' $R/analysis/extraction.log
stamp "replay (healthy matrix)"; timeout 14400 $PY openpi/re4_theory/fya_crossed_replay.py --root $R --suite libero_spatial --keys eligible --cells all --matrix healthy --label groot_healthy_crossed_v1_locked --out $R/runs/crossed_healthy.json > $R/runs/crossed_healthy.log 2>&1; echo "exit $?"
stamp "score"; python3 openpi/re4_theory/fya_crossed_score.py $R/runs/crossed_healthy.json --out $R/analysis --primary R1 --delta-I 1e-5 --seed 20260920 > $R/analysis/score.log 2>&1; echo "exit $?"
echo "GROOT CONFIRM DONE $(date +%T)"
