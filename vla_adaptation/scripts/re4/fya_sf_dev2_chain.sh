#!/bin/bash
# Development amendment (PREREG_FYA_STRONGER_FAULT_V1.md §5): two stronger candidates on the same development keys,
# reusing the running servers; selection over F1..F5 in registered order; NT under the selected fault; then SF DEV DONE.
source /home/mtaheri/ws_AgibotX2/vla-adaptation/scripts/re4/fya_sf_common.sh
D=$R/development; M=$R/dev_manifest.json
grep -q listening $SP/pi05_server_sf_dev1.log || server_up 1 8000 dev1 || exit 1; grep -q listening $SP/pi05_server_sf_dev0.log || server_up 0 8001 dev0 || exit 1
stamp "DEV2 candidates F4 (gpu 1) and F5 (gpu 0)"
( runarm 1 8000 $M dev_F4_off frozen legacy 0.10,0.10,0.10,0.10,0.10,0.10 $D ) > $SP/sf_dev2_gpu1.log 2>&1 &
P1=$!
( runarm 0 8001 $M dev_F5_off frozen legacy 0,0,0,0,0.20,0 $D ) > $SP/sf_dev2_gpu0.log 2>&1 &
P0=$!
wait $P1 $P0; cat $SP/sf_dev2_gpu1.log $SP/sf_dev2_gpu0.log
SEL=$(python3 - <<'PY'
import json,sys
D="/home/mtaheri/ws_AgibotX2/vla-adaptation/results/fya_stronger_fault_v1/development"
cands=[("F1","0.05,0.05,0.05,0.05,0.05,0.05"),("F2","0,0,0,0,0.10,0"),("F3","0,0,0,0,0.15,0"),("F4","0.10,0.10,0.10,0.10,0.10,0.10"),("F5","0,0,0,0,0.20,0")]
sel=None
for name,f in cands:
    d=json.load(open(f"{D}/dev_{name}_off.json")); s=d["arms"]["frozen_faulted"]["successes"]; print(f"# {name} off {s}/20", file=sys.stderr)
    if sel is None and s<=10: sel=(name,f)
print(f"{sel[0]} {sel[1]}" if sel else "NONE NONE")
PY
)
echo "SELECTED $SEL"; echo "$SEL" > $D/SELECTED_FAULT.txt
NAME=$(echo $SEL | cut -d' ' -f1); FV=$(echo $SEL | cut -d' ' -f2)
if [ "$NAME" != "NONE" ]; then runarm 1 8000 $M dev_${NAME}_nt adaptive legacy $FV $D; fi
server_down 8000; server_down 8001; echo "SF DEV DONE $(date +%T)"
