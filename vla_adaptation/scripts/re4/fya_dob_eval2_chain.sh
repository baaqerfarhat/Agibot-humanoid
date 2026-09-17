#!/bin/bash
# DOB comparator amendment 1: in-process trios. Faulted trio on GPU 0 after the development selection; healthy trio on GPU 1 after HC3.
source /home/mtaheri/ws_AgibotX2/vla-adaptation/scripts/re4/fya_sf_common.sh
SF=$R; R=$REPO/results/fya_dob_comparator_v1; D=$R/development; E=$R/evaluation_v2; ME=$SF/eval_manifest.json; FV=0.10,0.10,0.10,0.10,0.10,0.10; mkdir -p $E
until [ -s $D/SELECTED_GAIN.txt ]; do sleep 30; done; AL=$(cat $D/SELECTED_GAIN.txt | tr -d ' \n'); echo "selected alpha $AL"
# cancel the first evaluation route (its chain is waiting for the delay repair); kill only that chain's bash processes
for p in $(ps -eo pid,args | grep "bash scripts/re4/fya_dob_chain.sh" | grep -v grep | awk '{print $1}'); do kill $p 2>/dev/null; done; sleep 3
runarm2() { local gpu=$1 port=$2 name=$3 arms=$4 base=$5 gam=$6 fvec=$7
  stamp "$name (gpu $gpu)"
  timeout 7200 $PY -u $REPO/openpi/adaptive_law.py --port $port --control $SP/ctl_sf$gpu.json --ack $SP/ack_sf$gpu.json --suite libero_spatial \
    --log $REPO/results/phase05/error_signal_so3.json --openloop $REPO/results/phase05/openloop_so3.json --gamma $gam --dead 0.008 --norm-r 0.15 --clip 0.05 \
    --corr-dims 3,4,5 --norm-channels all --scenario-reset --manifest $ME --episodes 40 --arms $arms --law legacy --baseline $base --fault-vec $fvec --onset 40 --adapt-from 30 \
    --out $E/$name.json --telemetry $E/${name}_telemetry.jsonl > $E/$name.log 2>&1; echo "exit $?"; }
# faulted trio on GPU 0 (one process)
( until ! ps -eo args | grep -q "[a]ce_server.py.*port 8001"; do sleep 20; done
  stamp "DOB2 faulted trio server up (gpu 0)"; server_up 0 8001 dob2f || exit 1
  runarm2 0 8001 fault_off frozen none 0.08 $FV; runarm2 0 8001 fault_nt adaptive none 0.08 $FV; runarm2 0 8001 fault_dob adaptive dob $AL $FV
  server_down 8001 ) > $SP/dob2_gpu0.log 2>&1 &
P0=$!
# healthy trio on GPU 1 after HC3
( until grep -q "HC3 DONE" $SP/hc3.log 2>/dev/null; do sleep 30; done; sleep 15
  stamp "DOB2 healthy trio server up (gpu 1)"; server_up 1 8000 dob2h || exit 1
  runarm2 1 8000 healthy_off frozen none 0.08 0,0,0,0,0,0; runarm2 1 8000 healthy_nt adaptive none 0.08 0,0,0,0,0,0; runarm2 1 8000 healthy_dob adaptive dob $AL 0,0,0,0,0,0
  server_down 8000 ) > $SP/dob2_gpu1.log 2>&1 &
P1=$!
wait $P0 $P1; cat $SP/dob2_gpu0.log $SP/dob2_gpu1.log; cd $REPO && for f in $E/*_telemetry.jsonl $D/*_telemetry.jsonl; do [ -f $f ] && gzip -f $f; done
stamp "DOB2 coupling + score"
for pair in "fault_off fault_nt" "fault_off fault_dob" "healthy_off healthy_nt" "healthy_off healthy_dob"; do set -- $pair; python3 $REPO/openpi/re4_theory/fya_source_coupling_report.py --src $E --a $1 --b $2 --prefix 30 --out $R/analysis/coupling_${1}_vs_${2}.json > /dev/null 2>&1; done
python3 $REPO/openpi/re4_theory/fya_comparator_score.py --manifest $ME --arm fault_off=$E/fault_off.json --arm fault_nt=$E/fault_nt.json --arm fault_dob=$E/fault_dob.json \
  --arm healthy_off=$E/healthy_off.json --arm healthy_nt=$E/healthy_nt.json --arm healthy_dob=$E/healthy_dob.json \
  --primary fault_nt-fault_dob --secondary fault_dob-fault_off,fault_nt-fault_off,healthy_dob-healthy_off,healthy_nt-healthy_off --out $R/analysis/score_v2.json > $R/analysis/score_v2.log 2>&1; echo "exit $?"
echo "DOB2 DONE $(date +%T)"
