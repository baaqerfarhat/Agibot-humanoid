#!/bin/bash
# Healthy coupled replication v1 (PREREG_FYA_HEALTHY_COUPLED_V1.md): sources on both cards, extraction, healthy matrix, scoring.
source /home/mtaheri/ws_AgibotX2/vla-adaptation/scripts/re4/fya_sf_common.sh
R=$REPO/results/fya_healthy_coupled_v1; S=$R/sources; M=$R/manifest.json
until grep -q "SF DELAY DONE" $SP/sf_delay.log 2>/dev/null; do sleep 20; done; sleep 20
runsrc() { local gpu=$1 port=$2 name=$3 arms=$4 af=$5
  stamp "$name (gpu $gpu)"
  timeout 7200 $PY -u $REPO/openpi/adaptive_law.py --port $port --control $SP/ctl_sf$gpu.json --ack $SP/ack_sf$gpu.json --suite libero_spatial \
    --log $REPO/results/phase05/error_signal_so3.json --openloop $REPO/results/phase05/openloop_so3.json --gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.05 \
    --corr-dims 3,4,5 --norm-channels all --scenario-reset --manifest $M --episodes 40 --arms $arms --law legacy --fault-vec 0,0,0,0,0,0 --onset 40 --adapt-from $af \
    --out $S/$name.json --telemetry $S/${name}_telemetry.jsonl > $S/$name.log 2>&1; echo "exit $?"; }
stamp "HC servers up"; server_up 1 8000 hc1 || exit 1; server_up 0 8001 hc0 || exit 1
( runsrc 1 8000 healthy_off frozen 30; runsrc 1 8000 healthy_off_dup frozen 30 ) > $SP/hc_gpu1.log 2>&1 &
P1=$!
( runsrc 0 8001 healthy_nt adaptive 30 ) > $SP/hc_gpu0.log 2>&1 &
P0=$!
wait $P1 $P0; cat $SP/hc_gpu1.log $SP/hc_gpu0.log
server_down 8000; server_down 8001; cd $REPO && for f in $S/*_telemetry.jsonl; do gzip -f $f; done
stamp "HC extraction"; python3 $REPO/openpi/re4_theory/fya_crossed_extract.py --src $S --manifest $M --out $R --arms healthy_off,healthy_off,healthy_nt,healthy_off_dup --config-arm healthy_nt > $R/extraction.log 2>&1; echo "exit $?"
stamp "HC replay (healthy matrix)"; timeout 7200 $PY $REPO/openpi/re4_theory/fya_crossed_replay.py --root $R --keys eligible --cells all --matrix healthy --label healthy_coupled_v1_locked --out $R/runs/crossed_healthy.json > $R/runs/crossed_healthy.log 2>&1; echo "exit $?"
stamp "HC score"; python3 $REPO/openpi/re4_theory/fya_crossed_score.py $R/runs/crossed_healthy.json --out $R/analysis --primary R1 > $R/analysis/score.log 2>&1; echo "exit $?"
echo "HC DONE $(date +%T)"
