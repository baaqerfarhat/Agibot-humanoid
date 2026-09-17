#!/bin/bash
# Recovery study P1 (PREREG_FYA_CROSSED_STRONG_V1.md): stronger-fault NT crossed matrix on the archived 32 keys. Replays only.
source /home/mtaheri/ws_AgibotX2/vla-adaptation/scripts/re4/fya_sf_common.sh
R=$REPO/results/fya_recovery_study_v1/stronger_nt
stamp "P1 replay"; timeout 7200 $PY $REPO/openpi/re4_theory/fya_crossed_replay.py --root $R --keys eligible --cells all --matrix nt --ref-arm eval_healthy_off --m0-arm eval_fault_off --m1-arm eval_fault_nt \
  --label crossed_strong_v1_locked --out $R/runs/crossed_strong_nt.json > $R/runs/crossed_strong_nt.log 2>&1; echo "exit $?"
stamp "P1 score"; cd $REPO && python3 openpi/re4_theory/fya_crossed_score.py $R/runs/crossed_strong_nt.json --out $R/analysis --primary R1 --delta-I 1e-5 --seed 20260918 > $R/analysis/score.log 2>&1; echo "exit $?"
echo "P1 DONE $(date +%T)"
