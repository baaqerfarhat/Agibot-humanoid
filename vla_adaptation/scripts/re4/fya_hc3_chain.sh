#!/bin/bash
# Healthy coupled replication, amendment 2: all three source arms on ONE server process (GPU 1), after the delay repair.
source /home/mtaheri/ws_AgibotX2/vla-adaptation/scripts/re4/fya_sf_common.sh
R=$REPO/results/fya_healthy_coupled_v1; S=$R/sources_v3; M=$R/manifest.json; mkdir -p $S $R/runs_v3 $R/analysis_v3
until grep -q "DELAY REPAIR DONE" $SP/delay_repair.log 2>/dev/null; do sleep 30; done; sleep 15
runsrc() { local name=$1 arms=$2
  stamp "$name (gpu 1, shared process)"
  timeout 7200 $PY -u $REPO/openpi/adaptive_law.py --port 8000 --control $SP/ctl_sf1.json --ack $SP/ack_sf1.json --suite libero_spatial \
    --log $REPO/results/phase05/error_signal_so3.json --openloop $REPO/results/phase05/openloop_so3.json --gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.05 \
    --corr-dims 3,4,5 --norm-channels all --scenario-reset --manifest $M --episodes 40 --arms $arms --law legacy --fault-vec 0,0,0,0,0,0 --onset 40 --adapt-from 30 \
    --out $S/$name.json --telemetry $S/${name}_telemetry.jsonl > $S/$name.log 2>&1; echo "exit $?"; }
stamp "HC3 server up (gpu 1)"; server_up 1 8000 hc3 || exit 1
runsrc healthy_off frozen; runsrc healthy_nt adaptive; runsrc healthy_off_dup frozen
server_down 8000; cd $REPO && for f in $S/*_telemetry.jsonl; do gzip -f $f; done
stamp "HC3 coupling reports"
python3 $REPO/openpi/re4_theory/fya_source_coupling_report.py --src $S --a healthy_off --b healthy_nt --prefix 30 --out $R/analysis_v3/coupling_off_vs_nt_prefix30.json > /dev/null 2>&1
python3 $REPO/openpi/re4_theory/fya_source_coupling_report.py --src $S --a healthy_off --b healthy_off_dup --out $R/analysis_v3/coupling_off_vs_dup_full.json > /dev/null 2>&1
stamp "HC3 extraction"; python3 $REPO/openpi/re4_theory/fya_crossed_extract.py --src $S --manifest $M --out $R/v3 --arms healthy_off,healthy_off,healthy_nt,healthy_off_dup --config-arm healthy_nt > $R/analysis_v3/extraction.log 2>&1; echo "exit $?"; grep -o '"n_eligible": [0-9]*' $R/analysis_v3/extraction.log
stamp "HC3 replay"; timeout 7200 $PY $REPO/openpi/re4_theory/fya_crossed_replay.py --root $R/v3 --keys eligible --cells all --matrix healthy --label healthy_coupled_v1_amendment2_locked --out $R/runs_v3/crossed_healthy.json > $R/runs_v3/crossed_healthy.log 2>&1; echo "exit $?"
stamp "HC3 score"; python3 $REPO/openpi/re4_theory/fya_crossed_score.py $R/runs_v3/crossed_healthy.json --out $R/analysis_v3 --primary R1 > $R/analysis_v3/score.log 2>&1; echo "exit $?"
echo "HC3 DONE $(date +%T)"
