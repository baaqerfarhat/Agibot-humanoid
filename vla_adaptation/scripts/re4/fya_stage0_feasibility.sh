#!/bin/bash
# Stage 0 forecast feasibility (PREREG_FYA_RECOVERY_DEADLINE_V1.md section 9): leave-one-state-out on the OLD sources.
#   fold A: fit on state 39 (E1 v2 pass1 + dev_state39 replay) -> predict the state-33 sources -> calibrate on dev_state33
#   fold B: fit on state 33 (E1 v2 pass2 + dev_state33 replay) -> predict the state-39 sources -> calibrate on dev_state39
# then "evaluate" each fold against its own calibration to see how many intervals could be decisive at that width
# (development analysis: the same partition calibrates and evaluates, so this over-states coverage; it is a feasibility
# check of the width, not a result). Finally fit the campaign model on all four development runs.
set -u
REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation; R=$REPO/results/frozen_yet_adaptive_deadline_v1; S=$R/stage0; B=$R/configuration.json
F=$REPO/openpi/re4_theory/fya_forecast.py; SRC=$REPO/results/iclr_unified_v1/sources
cd $REPO
python3 $F fit --dev results/iclr_unified_v1/E1_v2/pass1_fitqual.json $S/dev_state39.json.gz --out $S/foldA_model_fit39.json
python3 $F predict --model $S/foldA_model_fit39.json --bundle $B --log $SRC/e1_test_spatial_init33.json --out $S/foldA_pred_state33.csv > $S/foldA_predict.log 2>&1
python3 $F calibrate --predictions $S/foldA_pred_state33.csv --run $S/dev_state33.json.gz --out $S/foldA_calibration.json
python3 $F evaluate --predictions $S/foldA_pred_state33.csv --calibration $S/foldA_calibration.json --run $S/dev_state33.json.gz --out $S/foldA_eval > $S/foldA_eval.log 2>&1
python3 $F fit --dev results/iclr_unified_v1/E1_v2/pass2_test.json $S/dev_state33.json.gz --out $S/foldB_model_fit33.json
python3 $F predict --model $S/foldB_model_fit33.json --bundle $B --log $SRC/e1_fitqual_spatial_init39.json --out $S/foldB_pred_state39.csv > $S/foldB_predict.log 2>&1
python3 $F calibrate --predictions $S/foldB_pred_state39.csv --run $S/dev_state39.json.gz --out $S/foldB_calibration.json
python3 $F evaluate --predictions $S/foldB_pred_state39.csv --calibration $S/foldB_calibration.json --run $S/dev_state39.json.gz --out $S/foldB_eval > $S/foldB_eval.log 2>&1
# cross-fold: fold A's width applied to fold B's predictions on state 39 (a width calibrated on a DIFFERENT state)
python3 $F evaluate --predictions $S/foldB_pred_state39.csv --calibration $S/foldA_calibration.json --run $S/dev_state39.json.gz --out $S/crossAB_eval > $S/crossAB_eval.log 2>&1
python3 $F evaluate --predictions $S/foldA_pred_state33.csv --calibration $S/foldB_calibration.json --run $S/dev_state33.json.gz --out $S/crossBA_eval > $S/crossBA_eval.log 2>&1
# measured development benefits under the new driver (delay / cap / sign effects on old sources)
python3 $REPO/openpi/re4_theory/fya_score.py $S/dev_state39.json.gz --out $S/score_state39 > $S/score_state39.log 2>&1
python3 $REPO/openpi/re4_theory/fya_score.py $S/dev_state33.json.gz --out $S/score_state33 > $S/score_state33.log 2>&1
# campaign model: all development data
python3 $F fit --dev results/iclr_unified_v1/E1_v2/pass1_fitqual.json results/iclr_unified_v1/E1_v2/pass2_test.json $S/dev_state39.json.gz $S/dev_state33.json.gz --out $S/forecast_model.json
sha256sum $S/forecast_model.json $B > $S/FROZEN_MODEL.sha256
echo "FEASIBILITY DONE $(date +%T)"
