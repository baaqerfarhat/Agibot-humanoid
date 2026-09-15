#!/bin/bash
# E2 core (PREREG_E2_CORE.md): 120 keys x 7 arms on the manifests, innovation law, corrected-channel normaliser,
# predictors U and C from the E2 fit partition. Waits for the post-prep chain, runs its own server, checks the
# sampler schedule on a two-key smoke first.
SP=/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad
OPENPI=/home/mtaheri/ws_AgibotX2/openpi; REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation; R=$REPO/results/iclr_unified_v1/E2_core
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=$OPENPI/third_party/libero:$OPENPI/examples/libero:$REPO/openpi
PY=$OPENPI/examples/libero/.venv/bin/python; MAN=$REPO/results/iclr_unified_v1/manifests
until grep -q "POST PREP DONE" $SP/post_prep_chain.log 2>/dev/null; do sleep 60; done; sleep 10
mkdir -p $R
cd $OPENPI && CUDA_VISIBLE_DEVICES=1 nohup $OPENPI/.venv/bin/python /home/mtaheri/ws_AgibotX2/Agibot-humanoid/vla_adaptation/openpi/ace_server.py --port 8000 --control $SP/ctl.json --ack $SP/ack.json > $SP/pi05_server_e2.log 2>&1 &
t=0; until grep -q "listening" $SP/pi05_server_e2.log 2>/dev/null; do sleep 10; t=$((t+10)); if [ $t -gt 1500 ]; then echo "SERVER NOT READY"; exit 1; fi; done
FIT="--log $REPO/results/iclr_unified_v1/sources/e2_fit_qual_spatial_init40_43.json --calib-episodes $(seq -s, 0 29) --openloop $REPO/results/phase05/openloop_so3.json"
LAW="--law innov --norm-channels corrected --gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.15 --corr-dims 3,4,5 --scenario-reset"
BASE="--port 8000 --control $SP/ctl.json --ack $SP/ack.json $LAW $FIT"
C="--dc-gain 4=0.254"; ORACLE="--static-corr=-0.05,-0.05,-0.05,-0.05,-0.05,-0.05"
go() { local id=$1 suite=$2; shift 2; local d=$R/$id; mkdir -p $d; echo "#### E2 $id  $(date +%T)"
  timeout 36000 $PY -u $REPO/openpi/adaptive_law.py $BASE --suite $suite --manifest $MAN/${suite}_E2_core.json "$@" --telemetry $d/telemetry.jsonl --timing $d/timing.jsonl --out $d/result.json > $d/run.log 2>&1
  echo "exit $?  $(grep -E ': [0-9]+/[0-9]+ =|DC-gain' $d/run.log | tr '\n' ' ' | cut -c1-200)"
  (cd $REPO && python3 openpi/re4_record.py record $d/result.json --part E2_core --run-id $id --cohort $suite --timing $d/timing.jsonl --root $R/.. >> $d/record.log 2>&1) || echo "RECORD FAILED $id"
  gzip -f $d/telemetry.jsonl 2>/dev/null; echo "RUN DONE E2 $id"; return 0; }
# sampler-schedule smoke: two keys, frozen arm, then check the ack and the CSV
python3 - <<'PY'
import json; m=json.load(open("/home/mtaheri/ws_AgibotX2/vla-adaptation/results/iclr_unified_v1/manifests/libero_spatial_E2_core.json")); m["scenarios"]=m["scenarios"][:2]; m["n_scenarios"]=2
json.dump(m, open("/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad/smoke_manifest.json","w"))
PY
d=$R/smoke_schedule; mkdir -p $d; echo "#### E2 smoke_schedule  $(date +%T)"
timeout 1800 $PY -u $REPO/openpi/adaptive_law.py $BASE --suite libero_spatial --manifest $SP/smoke_manifest.json --sev 0.05 --arms frozen --out $d/result.json > $d/run.log 2>&1; echo "exit $?"
grep -q "fold_in" $SP/ack.json && echo "SCHEDULE OK $(grep -o '"sampler_seed": [0-9]*' $SP/ack.json | head -1)" || { echo "SCHEDULE FAILED"; cat $SP/ack.json | head -5; exit 1; }
for suite in libero_spatial libero_10; do
  go ${suite}_healthy_off     $suite --sev 0.0  --arms frozen
  go ${suite}_healthy_U       $suite --sev 0.0  --arms adaptive
  go ${suite}_healthy_C       $suite --sev 0.0  --arms adaptive $C
  go ${suite}_faulted_off     $suite --sev 0.05 --arms frozen
  go ${suite}_faulted_U       $suite --sev 0.05 --arms adaptive
  go ${suite}_faulted_C       $suite --sev 0.05 --arms adaptive $C
  go ${suite}_faulted_oracle  $suite --sev 0.05 --arms adaptive $ORACLE
done
P=$(ps -eo pid,args | grep "[a]ce_server.py" | awk '{print $1}'); [ -n "$P" ] && kill $P
echo "E2 CORE DONE $(date +%T)"
