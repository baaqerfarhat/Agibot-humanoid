#!/bin/bash
# Collaborator's queue (docs/independent_adaptation/PRIORITY_QUEUE.md): Q2 pose tracking (PREREG_Q2_POSE_TRACKING.md),
# Q5 six-channel oracle (PREREG_Q5_SIXCHANNEL_ORACLE.md), Q6 pinned FIR/ARX/DC (PREREG_Q6_PINNED_FIR_ARX_DC.md).
# Runs the MERGED runner from the integrate/collab-q worktree (main's runner + the opt-in tracking/replay code, verified
# byte-identical with the new flags unset). Waits for the E1 v2 chain, runs its own server. Q2's OFF arm is E2's
# libero_10_faulted_off (same keys and schedule), not re-run.
SP=/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad
OPENPI=/home/mtaheri/ws_AgibotX2/openpi; MAIN=/home/mtaheri/ws_AgibotX2/vla-adaptation; REPO=$SP/wt-q; R=$MAIN/results/collab_q
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=$OPENPI/third_party/libero:$OPENPI/examples/libero:$REPO/openpi
PY=$OPENPI/examples/libero/.venv/bin/python; MAN=$MAIN/results/iclr_unified_v1/manifests
true  # GPU-0 variant: shared with the other user, no wait
mkdir -p $R
SERVER_PID=$(ps -eo pid,args | grep "[a]ce_server.py" | grep "port 8001" | awk '{print $1}')
until [ -f $R/q2_C_libero_10/result.json ]; do sleep 60; done; sleep 5
(cd $MAIN && python3 openpi/re4_record.py record $R/q2_C_libero_10/result.json --part collab_q --run-id q2_C_libero_10 --cohort libero_10 --timing $R/q2_C_libero_10/timing.jsonl --root $R/.. >> $R/q2_C_libero_10/record.log 2>&1); gzip -f $R/q2_C_libero_10/telemetry.jsonl 2>/dev/null; echo "RUN DONE Q q2_C_libero_10 (orphaned client, recorded here)"
SHIP="--log $MAIN/results/phase05/error_signal_so3.json --openloop $MAIN/results/phase05/openloop_so3.json"
CONST="--gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.15 --corr-dims 3,4,5 --scenario-reset"
BASE="--port 8001 --control $SP/ctl0.json --ack $SP/ack0.json $CONST $SHIP"
TRACK="--track-kappa 0.001 --track-mode position --track-anchor 0 --track-dims 3,5 --track-ref dc --track-leak 0.02 --track-obs tracked"
go() { local id=$1 suite=$2; shift 2; local d=$R/$id; mkdir -p $d; echo "#### Q $id  $(date +%T)"
  timeout 36000 $PY -u $REPO/openpi/adaptive_law.py $BASE --suite $suite "$@" --telemetry $d/telemetry.jsonl --timing $d/timing.jsonl --out $d/result.json > $d/run.log 2>&1
  echo "exit $?  $(grep -E ': [0-9]+/[0-9]+ =' $d/run.log | tr '\n' ' ' | cut -c1-160)"
  (cd $MAIN && python3 openpi/re4_record.py record $d/result.json --part collab_q --run-id $id --cohort $suite --timing $d/timing.jsonl --root $R/.. >> $d/record.log 2>&1) || echo "RECORD FAILED $id"
  gzip -f $d/telemetry.jsonl 2>/dev/null; echo "RUN DONE Q $id"; return 0; }
# Q2 primary cell: libero_10, E2 keys and schedule; OFF shared with E2
go q2_D_libero_10  libero_10 --manifest $MAN/libero_10_E2_core.json --sev 0.05 --arms adaptive --law innov --dc-constrain corrected $TRACK
go q2_HC_libero_10 libero_10 --manifest $MAN/libero_10_Q2_healthy20.json --sev 0.0 --arms adaptive --law innov --dc-constrain corrected
go q2_HD_libero_10 libero_10 --manifest $MAN/libero_10_Q2_healthy20.json --sev 0.0 --arms adaptive --law innov --dc-constrain corrected $TRACK
# Q2 secondary cell: spatial joint-5 torque, 40 keys, C and D only
go q2_C_spatial_joint5 libero_spatial --manifest $MAN/libero_spatial_Q2_joint5_40.json --sev 0.0 --joint-fault torque:5:5.0 --arms adaptive --law innov --dc-constrain corrected
go q2_D_spatial_joint5 libero_spatial --manifest $MAN/libero_spatial_Q2_joint5_40.json --sev 0.0 --joint-fault torque:5:5.0 --arms adaptive --law innov --dc-constrain corrected $TRACK
# Q5: six-channel oracle on libero_10, D3a's protocol with all six channels corrected; both arms
# Q6: pinned adaptive arms on the 8.2a keys (inits 45-49 then 44-42), frozen arm shared with a_method_libero10_n80
kill $SERVER_PID   # only our own GPU-0 server
echo "COLLAB Q2 GPU0 DONE $(date +%T)"
