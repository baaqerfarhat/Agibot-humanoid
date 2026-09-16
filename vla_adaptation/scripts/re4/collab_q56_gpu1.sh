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
true  # GPU 1 is free (E1 v2 done)
mkdir -p $R
cd $OPENPI && CUDA_VISIBLE_DEVICES=1 nohup $OPENPI/.venv/bin/python /home/mtaheri/ws_AgibotX2/Agibot-humanoid/vla_adaptation/openpi/ace_server.py --port 8000 --control $SP/ctl.json --ack $SP/ack.json > $SP/pi05_server_q.log 2>&1 &
t=0; until grep -q "listening" $SP/pi05_server_q.log 2>/dev/null; do sleep 10; t=$((t+10)); if [ $t -gt 1500 ]; then echo "SERVER NOT READY"; exit 1; fi; done
SHIP="--log $MAIN/results/phase05/error_signal_so3.json --openloop $MAIN/results/phase05/openloop_so3.json"
CONST="--gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.15 --corr-dims 3,4,5 --scenario-reset"
BASE="--port 8000 --control $SP/ctl.json --ack $SP/ack.json $CONST $SHIP"
TRACK="--track-kappa 0.001 --track-mode position --track-anchor 0 --track-dims 3,5 --track-ref dc --track-leak 0.02 --track-obs tracked"
go() { local id=$1 suite=$2; shift 2; local d=$R/$id; mkdir -p $d; echo "#### Q $id  $(date +%T)"
  timeout 36000 $PY -u $REPO/openpi/adaptive_law.py $BASE --suite $suite "$@" --telemetry $d/telemetry.jsonl --timing $d/timing.jsonl --out $d/result.json > $d/run.log 2>&1
  echo "exit $?  $(grep -E ': [0-9]+/[0-9]+ =' $d/run.log | tr '\n' ' ' | cut -c1-160)"
  (cd $MAIN && python3 openpi/re4_record.py record $d/result.json --part collab_q --run-id $id --cohort $suite --timing $d/timing.jsonl --root $R/.. >> $d/record.log 2>&1) || echo "RECORD FAILED $id"
  gzip -f $d/telemetry.jsonl 2>/dev/null; echo "RUN DONE Q $id"; return 0; }
# Q2 primary cell: libero_10, E2 keys and schedule; OFF shared with E2
# Q2 secondary cell: spatial joint-5 torque, 40 keys, C and D only
# Q5: six-channel oracle on libero_10, D3a's protocol with all six channels corrected; both arms
go q5_D3b_oracle_all_libero_10 libero_10 --episodes 40 --eval-init 45 --sev 0.05 --corr-dims 0,1,2,3,4,5 --static-corr=-0.05,-0.05,-0.05,-0.05,-0.05,-0.05
# Q6: pinned adaptive arms on the 8.2a keys (inits 45-49 then 44-42), frozen arm shared with a_method_libero10_n80
go q6_arx_libero10_n80 libero_10 --episodes 80 --eval-init 45 --sev 0.05 --pin-rng --arms adaptive --ar 1
go q6_dc_libero10_n80  libero_10 --episodes 80 --eval-init 45 --sev 0.05 --pin-rng --arms adaptive --dc-constrain corrected
P=$(ps -eo pid,args | grep "[a]ce_server.py" | awk '{print $1}'); [ -n "$P" ] && kill $P
echo "COLLAB Q5Q6 GPU1 DONE $(date +%T)"
