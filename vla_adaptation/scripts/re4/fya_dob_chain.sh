#!/bin/bash
# DOB comparator (PREREG_FYA_DOB_COMPARATOR_V1.md): gain selection on GPU 0 now; evaluation arms after GPU 1 is released by the delay repair.
source /home/mtaheri/ws_AgibotX2/vla-adaptation/scripts/re4/fya_sf_common.sh
SF=$R; R=$REPO/results/fya_dob_comparator_v1; D=$R/development; E=$R/evaluation; MD=$SF/dev_manifest.json; ME=$SF/eval_manifest.json; FV=0.10,0.10,0.10,0.10,0.10,0.10
rundob() { local gpu=$1 port=$2 man=$3 name=$4 alpha=$5 fvec=$6 od=$7
  stamp "$name (gpu $gpu, alpha $alpha)"
  timeout 7200 $PY -u $REPO/openpi/adaptive_law.py --port $port --control $SP/ctl_sf$gpu.json --ack $SP/ack_sf$gpu.json --suite libero_spatial \
    --log $REPO/results/phase05/error_signal_so3.json --openloop $REPO/results/phase05/openloop_so3.json --gamma $alpha --dead 0.008 --norm-r 0.15 --clip 0.05 \
    --corr-dims 3,4,5 --norm-channels all --scenario-reset --manifest $man --episodes 40 --arms adaptive --law legacy --baseline dob --fault-vec $fvec --onset 40 --adapt-from 30 \
    --out $od/$name.json --telemetry $od/${name}_telemetry.jsonl > $od/$name.log 2>&1; echo "exit $?"; }
stamp "DOB dev server up (gpu 0)"; server_up 0 8001 dobdev || exit 1
for al in 0.02 0.08 0.20; do rundob 0 8001 $MD dev_dob_a${al}_healthy $al 0,0,0,0,0,0 $D; rundob 0 8001 $MD dev_dob_a${al}_fault $al $FV $D; done
server_down 8001
SEL=$(python3 - <<'PY'
import json
D="/home/mtaheri/ws_AgibotX2/vla-adaptation/results/fya_dob_comparator_v1/development"
best=None
for al in ("0.02","0.08","0.20"):
    h=json.load(open(f"{D}/dev_dob_a{al}_healthy.json"))["arms"]["adaptive"]["successes"]; f=json.load(open(f"{D}/dev_dob_a{al}_fault.json"))["arms"]["adaptive"]["successes"]
    score=(h+f)/40; hl=20-h; print(f"# alpha {al}: healthy {h}/20 fault {f}/20 mean {score:.3f}", file=__import__("sys").stderr)
    key=(score, -hl, -float(al))
    if best is None or key>best[0]: best=(key, al)
print(best[1])
PY
)
echo "SELECTED_GAIN $SEL"; echo "$SEL" > $D/SELECTED_GAIN.txt
until grep -q "DELAY REPAIR DONE" $SP/delay_repair.log 2>/dev/null; do sleep 30; done; sleep 10
stamp "DOB eval servers up"; server_up 1 8000 dobev1 || exit 1; server_up 0 8001 dobev0 || exit 1
( rundob 1 8000 $ME eval_dob_fault $SEL $FV $E ) > $SP/dob_gpu1.log 2>&1 &
P1=$!
( rundob 0 8001 $ME eval_dob_healthy $SEL 0,0,0,0,0,0 $E ) > $SP/dob_gpu0.log 2>&1 &
P0=$!
wait $P1 $P0; cat $SP/dob_gpu1.log $SP/dob_gpu0.log
server_down 8000; server_down 8001; cd $REPO && for f in $D/*_telemetry.jsonl $E/*_telemetry.jsonl; do [ -f $f ] && gzip -f $f; done
stamp "DOB score"
python3 $REPO/openpi/re4_theory/fya_comparator_score.py --manifest $ME --arm fault_off=$SF/evaluation/eval_fault_off.json --arm fault_nt=$SF/evaluation/eval_fault_nt.json --arm fault_dob=$E/eval_dob_fault.json \
  --arm healthy_off=$SF/evaluation/eval_healthy_off.json --arm healthy_nt=$SF/evaluation/eval_healthy_nt.json --arm healthy_dob=$E/eval_dob_healthy.json \
  --primary fault_nt-fault_dob --secondary fault_dob-fault_off,healthy_dob-healthy_off,fault_nt-fault_off --out $R/analysis/score.json > $R/analysis/score.log 2>&1; echo "exit $?"
echo "DOB DONE $(date +%T)"
