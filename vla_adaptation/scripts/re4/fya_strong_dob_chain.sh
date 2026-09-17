#!/bin/bash
# Recovery study P3 (PREREG_FYA_CROSSED_DOB_V1.md): DOB diagonal pilot, then NT and DOB matrices on the 29 common keys, scoring and pairing.
source /home/mtaheri/ws_AgibotX2/vla-adaptation/scripts/re4/fya_sf_common.sh
R=$REPO/results/fya_recovery_study_v1/stronger_dob; mkdir -p $R/runs $R/analysis_nt $R/analysis_dob $R/pilot
until grep -q "P1 DONE" $SP/p1.log 2>/dev/null; do sleep 30; done
stamp "P3 DOB diagonal pilot (2 keys)"
timeout 3600 $PY $REPO/openpi/re4_theory/fya_crossed_replay.py --root $R/bundle_dob --keys 1_18_85001,2_18_85001 --cells diagonal --matrix dob --ref-arm eval_healthy_off --m0-arm fault_off --m1-arm fault_dob \
  --label p3_dob_pilot --out $R/pilot/pilot_dob.json > $R/pilot/pilot_dob.log 2>&1; echo "exit $?"; grep "^key" $R/pilot/pilot_dob.log
if grep "^key" $R/pilot/pilot_dob.log | grep -q "valid=False"; then echo "P3 PILOT FAILED: DOB diagonal does not reproduce the archive"; echo "P3 DONE $(date +%T)"; exit 0; fi
for m in nt dob; do
  stamp "P3 $m matrix (29 keys)"
  timeout 7200 $PY $REPO/openpi/re4_theory/fya_crossed_replay.py --root $R/bundle_$m --keys eligible --cells all --matrix $m --ref-arm eval_healthy_off --m0-arm fault_off --m1-arm fault_$m \
    --label crossed_strong_${m}_locked --out $R/runs/crossed_strong_$m.json > $R/runs/crossed_strong_$m.log 2>&1; echo "exit $?"
  cd $REPO && python3 openpi/re4_theory/fya_crossed_score.py $R/runs/crossed_strong_$m.json --out $R/analysis_$m --primary R1 --delta-I 1e-5 --seed 20260918 > $R/analysis_$m/score.log 2>&1; echo "score exit $?"
done
stamp "P3 pairing"; python3 $REPO/openpi/re4_theory/fya_pair_matrices.py --a $R/analysis_nt/source_effects.csv --b $R/analysis_dob/source_effects.csv --label-a NT --label-b DOB --primary T --delta 1e-5 --seed 20260918 --out $R/analysis_pair_nt_minus_dob.json > $R/pair.log 2>&1; echo "exit $?"
echo "P3 DONE $(date +%T)"
