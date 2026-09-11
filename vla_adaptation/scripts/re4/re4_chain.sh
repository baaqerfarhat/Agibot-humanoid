#!/bin/bash
# re4 evidence plan, Parts C and D on the pi0.5 LIBERO server (E piggybacks via --timing).
SP=/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad
OPENPI=/home/mtaheri/ws_AgibotX2/openpi; REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation; R=$REPO/results/re4_evidence
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=$OPENPI/third_party/libero:$OPENPI/examples/libero:$REPO/openpi
PY=$OPENPI/examples/libero/.venv/bin/python
SHIP="--log $REPO/results/phase05/error_signal_so3.json --openloop $REPO/results/phase05/openloop_so3.json"
HELD="--log $REPO/results/heldout/error_signal_init25.json --openloop $REPO/results/heldout/openloop_init25.json"
BASE="--port 8000 --control $SP/ctl.json --ack $SP/ack.json --gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.15 --scenario-reset"
ROT="--corr-dims 3,4,5"
ORACLE="--static-corr=-0.05,-0.05,-0.05,-0.05,-0.05,-0.05"
cd $OPENPI
go() { local part=$1 id=$2 cohort=$3; shift 3; local d=$R/$part/$id; mkdir -p $d
  echo "#### $part $id ($cohort)  $(date +%T)"
  timeout 21600 $PY -u $REPO/openpi/adaptive_law.py "$@" --timing $d/timing.jsonl --out $d/result.json > $d/run.log 2>&1
  echo "exit $?  $(grep -E ': [0-9]+/[0-9]+ =' $d/run.log | tr '\n' ' ')"
  ( cd $REPO && python3 openpi/re4_record.py record $d/result.json --part $part --run-id $id --cohort $cohort \
        --timing $d/timing.jsonl >> $d/record.log 2>&1 \
    && python3 openpi/re4_record.py timing $d/timing.jsonl --result $d/result.json >> $d/record.log 2>&1 ) \
    || echo "RECORD FAILED $id"
  echo "RUN DONE $part $id  $(date +%T)"
}
CELLS=("libero_spatial 20" "libero_object 20" "libero_goal 40" "libero_10 40")
for c in "${CELLS[@]}"; do set -- $c; go C_healthy shipped_$1 $1 $BASE $ROT $SHIP --suite $1 --episodes $2 --sev 0.0; done
for c in "${CELLS[@]}"; do set -- $c; go C_healthy heldout_$1 $1 $BASE $ROT $HELD --suite $1 --episodes $2 --sev 0.0; done
for c in "libero_spatial 20" "libero_10 40"; do set -- $c
  go D_baselines D0_method_$1       $1 $BASE $ROT $SHIP --suite $1 --episodes $2 --sev 0.05
  go D_baselines D1_static_k0_$1    $1 $BASE $ROT $SHIP --suite $1 --episodes $2 --sev 0.05 --fir-k 0
  go D_baselines D2_innov_$1        $1 $BASE $ROT $SHIP --suite $1 --episodes $2 --sev 0.05 --law innov
  go D_baselines D3a_oracle_rot_$1  $1 $BASE $ROT $SHIP --suite $1 --episodes $2 --sev 0.05 $ORACLE
done
go D_baselines D3b_oracle_all_libero_spatial libero_spatial $BASE $SHIP --suite libero_spatial --episodes 20 --sev 0.05 $ORACLE
echo "RE4 CHAIN CD DONE $(date +%T)"
