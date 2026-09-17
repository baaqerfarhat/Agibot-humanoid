#!/bin/bash
# Evaluation stage (locked): four arms on the 40 untouched keys, two per card; arm order registered here.
source /home/mtaheri/ws_AgibotX2/vla-adaptation/scripts/re4/fya_sf_common.sh
E=$R/evaluation; M=$R/eval_manifest.json; SEL=$(cat $R/development/SELECTED_FAULT.txt); NAME=$(echo $SEL | cut -d' ' -f1); FV=$(echo $SEL | cut -d' ' -f2)
[ "$NAME" = "NONE" ] && { echo "no fault selected; evaluation not run"; exit 0; }
stamp "EVAL servers up (fault $NAME $FV)"; server_up 1 8000 eval1 || exit 1; server_up 0 8001 eval0 || exit 1
( runarm 1 8000 $M eval_fault_nt adaptive legacy $FV $E; runarm 1 8000 $M eval_healthy_off frozen legacy 0,0,0,0,0,0 $E ) > $SP/sf_eval_gpu1.log 2>&1 &
P1=$!
( runarm 0 8001 $M eval_fault_off frozen legacy $FV $E; runarm 0 8001 $M eval_healthy_nt adaptive legacy 0,0,0,0,0,0 $E ) > $SP/sf_eval_gpu0.log 2>&1 &
P0=$!
wait $P1 $P0; cat $SP/sf_eval_gpu1.log $SP/sf_eval_gpu0.log
server_down 8000; server_down 8001; cd $REPO && for f in $E/*_telemetry.jsonl $R/development/*_telemetry.jsonl; do [ -f $f ] && gzip -f $f; done
echo "SF EVAL DONE $(date +%T)"
