#!/bin/bash
# Six-channel correction on libero_10 (PREREG_SIX_CHANNEL_LIBERO10.md): faulted U6, faulted C6, six-channel oracle on GPU 1.
SP=/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad
OPENPI=/home/mtaheri/ws_AgibotX2/openpi; REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation; R=$REPO/results/six_channel
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=$OPENPI/third_party/libero:$OPENPI/examples/libero:$REPO/openpi
PY=$OPENPI/examples/libero/.venv/bin/python; MAN=$REPO/results/iclr_unified_v1/manifests
cd $OPENPI && CUDA_VISIBLE_DEVICES=1 nohup $OPENPI/.venv/bin/python /home/mtaheri/ws_AgibotX2/Agibot-humanoid/vla_adaptation/openpi/ace_server.py --port 8000 --control $SP/ctl.json --ack $SP/ack.json > $SP/pi05_server_six1.log 2>&1 &
SERVER_PID=$!; t=0; until grep -q "listening" $SP/pi05_server_six1.log 2>/dev/null; do sleep 10; t=$((t+10)); if [ $t -gt 1500 ]; then echo "SERVER NOT READY"; exit 1; fi; done
FIT="--log $REPO/results/iclr_unified_v1/sources/e2_fit_qual_spatial_init40_43.json --calib-episodes $(seq -s, 0 29) --openloop $REPO/results/phase05/openloop_so3.json --m-diag 0=0.257,1=0.263,2=0.291"
LAW="--law innov --norm-channels corrected --gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.15 --corr-dims 0,1,2,3,4,5 --scenario-reset"
BASE="--port 8000 --control $SP/ctl.json --ack $SP/ack.json $LAW $FIT --suite libero_10 --manifest $MAN/libero_10_E2_core.json"
go() { local id=$1; shift; local d=$R/$id; mkdir -p $d; echo "#### SIX $id  $(date +%T)"
  timeout 36000 $PY -u $REPO/openpi/adaptive_law.py $BASE "$@" --telemetry $d/telemetry.jsonl --timing $d/timing.jsonl --out $d/result.json > $d/run.log 2>&1
  echo "exit $?  $(grep -E ': [0-9]+/[0-9]+ =|M diagonal|DC-gain' $d/run.log | tr '\n' ' ' | cut -c1-220)"
  (cd $REPO && python3 openpi/re4_record.py record $d/result.json --part six_channel --run-id $id --cohort libero_10 --timing $d/timing.jsonl --root $R/.. >> $d/record.log 2>&1) || echo "RECORD FAILED $id"
  gzip -f $d/telemetry.jsonl 2>/dev/null; echo "RUN DONE SIX $id"; return 0; }
go libero_10_faulted_U      --sev 0.05 --arms adaptive
go libero_10_faulted_C      --sev 0.05 --arms adaptive --dc-gain 4=0.254
go libero_10_faulted_oracle --sev 0.05 --arms adaptive --static-corr=-0.05,-0.05,-0.05,-0.05,-0.05,-0.05
kill $SERVER_PID; echo "SIX GPU1 DONE $(date +%T)"
