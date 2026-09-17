#!/bin/bash
# Delayed-NT increment (PREREG_FYA_STRONGER_FAULT_V1.md section 7): one arm on the 40 evaluation keys, split by state across the cards.
source /home/mtaheri/ws_AgibotX2/vla-adaptation/scripts/re4/fya_sf_common.sh
E=$R/evaluation; FV=0.10,0.10,0.10,0.10,0.10,0.10
runarm_delay() { local gpu=$1 port=$2 man=$3 name=$4
  stamp "$name (gpu $gpu)"
  timeout 7200 $PY -u $REPO/openpi/adaptive_law.py --port $port --control $SP/ctl_sf$gpu.json --ack $SP/ack_sf$gpu.json --suite libero_spatial \
    --log $REPO/results/phase05/error_signal_so3.json --openloop $REPO/results/phase05/openloop_so3.json --gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.05 \
    --corr-dims 3,4,5 --norm-channels all --scenario-reset --manifest $man --episodes 20 --arms adaptive --law legacy --fault-vec $FV --onset 40 --adapt-from 40 \
    --out $E/$name.json --telemetry $E/${name}_telemetry.jsonl > $E/$name.log 2>&1; echo "exit $?"; }
stamp "DELAY servers up"; server_up 1 8000 delay1 || exit 1; server_up 0 8001 delay0 || exit 1
( runarm_delay 1 8000 $R/eval_manifest_a.json eval_delay_nt_a ) > $SP/sf_delay_gpu1.log 2>&1 &
P1=$!
( runarm_delay 0 8001 $R/eval_manifest_b.json eval_delay_nt_b ) > $SP/sf_delay_gpu0.log 2>&1 &
P0=$!
wait $P1 $P0; cat $SP/sf_delay_gpu1.log $SP/sf_delay_gpu0.log
server_down 8000; server_down 8001; cd $REPO && for f in $E/eval_delay_nt_*_telemetry.jsonl; do [ -f $f ] && gzip -f $f; done
echo "SF DELAY DONE $(date +%T)"
