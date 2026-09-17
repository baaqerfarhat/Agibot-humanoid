#!/bin/bash
# Healthy coupled replication v1, amendment: re-run healthy_nt on the GPU 1 server, then extraction, matrix, scoring.
source /home/mtaheri/ws_AgibotX2/vla-adaptation/scripts/re4/fya_sf_common.sh
R=$REPO/results/fya_healthy_coupled_v1; S=$R/sources; M=$R/manifest.json
stamp "HC2 server up (gpu 1)"; server_up 1 8000 hc2 || exit 1
stamp "healthy_nt (gpu 1)"
timeout 7200 $PY -u $REPO/openpi/adaptive_law.py --port 8000 --control $SP/ctl_sf1.json --ack $SP/ack_sf1.json --suite libero_spatial \
  --log $REPO/results/phase05/error_signal_so3.json --openloop $REPO/results/phase05/openloop_so3.json --gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.05 \
  --corr-dims 3,4,5 --norm-channels all --scenario-reset --manifest $M --episodes 40 --arms adaptive --law legacy --fault-vec 0,0,0,0,0,0 --onset 40 --adapt-from 30 \
  --out $S/healthy_nt.json --telemetry $S/healthy_nt_telemetry.jsonl > $S/healthy_nt.log 2>&1; echo "exit $?"
server_down 8000; cd $REPO && gzip -f $S/healthy_nt_telemetry.jsonl
stamp "HC2 extraction"; python3 $REPO/openpi/re4_theory/fya_crossed_extract.py --src $S --manifest $M --out $R --arms healthy_off,healthy_off,healthy_nt,healthy_off_dup --config-arm healthy_nt > $R/extraction.log 2>&1; echo "exit $?"; grep -o '"n_eligible": [0-9]*' $R/extraction.log
stamp "HC2 replay (healthy matrix)"; timeout 7200 $PY $REPO/openpi/re4_theory/fya_crossed_replay.py --root $R --keys eligible --cells all --matrix healthy --label healthy_coupled_v1_locked --out $R/runs/crossed_healthy.json > $R/runs/crossed_healthy.log 2>&1; echo "exit $?"
stamp "HC2 score"; python3 $REPO/openpi/re4_theory/fya_crossed_score.py $R/runs/crossed_healthy.json --out $R/analysis --primary R1 > $R/analysis/score.log 2>&1; echo "exit $?"
echo "HC2 DONE $(date +%T)"
