#!/bin/bash
# Recovery study P2 (PREREG_FYA_HEALTHY_LIBERO10_V1.md): healthy LIBERO-10 sources on one server process (GPU 1), extraction, healthy matrix, scoring.
source /home/mtaheri/ws_AgibotX2/vla-adaptation/scripts/re4/fya_sf_common.sh
R=$REPO/results/fya_recovery_study_v1/healthy_libero10; S=$R/sources; M=$R/manifest.json; mkdir -p $S $R/runs $R/analysis
runsrc() { local name=$1 arms=$2
  stamp "$name (gpu 1, shared process, libero_10)"
  timeout 14400 $PY -u $REPO/openpi/adaptive_law.py --port 8000 --control $SP/ctl_sf1.json --ack $SP/ack_sf1.json --suite libero_10 --replan-steps 5 \
    --log $REPO/results/phase05/error_signal_so3.json --openloop $REPO/results/phase05/openloop_so3.json --gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.05 \
    --corr-dims 3,4,5 --norm-channels all --scenario-reset --manifest $M --episodes 40 --arms $arms --law legacy --baseline none --fault-vec 0,0,0,0,0,0 --onset 40 --adapt-from 30 \
    --out $S/$name.json --telemetry $S/${name}_telemetry.jsonl > $S/$name.log 2>&1; echo "exit $?"; }
stamp "P2 server up (gpu 1)"; server_up 1 8000 hl10 || exit 1
runsrc healthy_off frozen; runsrc healthy_nt adaptive; runsrc healthy_off_dup frozen
server_down 8000; cd $REPO && for f in $S/*_telemetry.jsonl; do gzip -f $f; done
stamp "P2 coupling reports"
python3 $REPO/openpi/re4_theory/fya_source_coupling_report.py --src $S --a healthy_off --b healthy_nt --prefix 30 --out $R/analysis/coupling_off_vs_nt_prefix30.json > /dev/null 2>&1
python3 $REPO/openpi/re4_theory/fya_source_coupling_report.py --src $S --a healthy_off --b healthy_off_dup --out $R/analysis/coupling_off_vs_dup_full.json > /dev/null 2>&1
stamp "P2 extraction"; python3 $REPO/openpi/re4_theory/fya_crossed_extract.py --src $S --manifest $M --out $R --arms healthy_off,healthy_off,healthy_nt,healthy_off_dup --config-arm healthy_nt > $R/analysis/extraction.log 2>&1; echo "exit $?"; grep -o '"n_eligible": [0-9]*' $R/analysis/extraction.log
stamp "P2 replay (libero_10 healthy matrix)"; timeout 14400 $PY $REPO/openpi/re4_theory/fya_crossed_replay.py --root $R --suite libero_10 --keys eligible --cells all --matrix healthy --label healthy_libero10_v1_locked --out $R/runs/crossed_healthy.json > $R/runs/crossed_healthy.log 2>&1; echo "exit $?"
stamp "P2 score"; python3 $REPO/openpi/re4_theory/fya_crossed_score.py $R/runs/crossed_healthy.json --out $R/analysis --primary R1 --delta-I 1e-5 --seed 20260918 > $R/analysis/score.log 2>&1; echo "exit $?"
echo "P2 DONE $(date +%T)"
