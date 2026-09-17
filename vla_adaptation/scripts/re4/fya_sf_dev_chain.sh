#!/bin/bash
# Development stage: candidate faulted-off arms split across the two cards, healthy off, then NT under the selected fault.
source /home/mtaheri/ws_AgibotX2/vla-adaptation/scripts/re4/fya_sf_common.sh
D=$R/development; M=$R/dev_manifest.json
stamp "DEV servers up"; server_up 1 8000 dev1 || exit 1; server_up 0 8001 dev0 || exit 1
( runarm 1 8000 $M dev_F1_off frozen legacy 0.05,0.05,0.05,0.05,0.05,0.05 $D; runarm 1 8000 $M dev_F3_off frozen legacy 0,0,0,0,0.15,0 $D ) > $SP/sf_dev_gpu1.log 2>&1 &
P1=$!
( runarm 0 8001 $M dev_F2_off frozen legacy 0,0,0,0,0.10,0 $D; runarm 0 8001 $M dev_healthy_off frozen legacy 0,0,0,0,0,0 $D ) > $SP/sf_dev_gpu0.log 2>&1 &
P0=$!
wait $P1 $P0; cat $SP/sf_dev_gpu1.log $SP/sf_dev_gpu0.log
SEL=$(python3 - <<'PY'
import json
D="/home/mtaheri/ws_AgibotX2/vla-adaptation/results/fya_stronger_fault_v1/development"
cands=[("F1","0.05,0.05,0.05,0.05,0.05,0.05"),("F2","0,0,0,0,0.10,0"),("F3","0,0,0,0,0.15,0")]
sel=None
for name,f in cands:
    d=json.load(open(f"{D}/dev_{name}_off.json")); s=d["arms"]["frozen_faulted"]["successes"]
    print(f"# {name} off {s}/20", file=__import__("sys").stderr)
    if sel is None and s<=10: sel=(name,f)
print(f"{sel[0]} {sel[1]}" if sel else "NONE NONE")
PY
)
echo "SELECTED $SEL"; echo "$SEL" > $D/SELECTED_FAULT.txt
NAME=$(echo $SEL | cut -d' ' -f1); FV=$(echo $SEL | cut -d' ' -f2)
if [ "$NAME" != "NONE" ]; then runarm 1 8000 $M dev_${NAME}_nt adaptive legacy $FV $D; fi
server_down 8000; server_down 8001; echo "SF DEV DONE $(date +%T)"
