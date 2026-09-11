#!/bin/bash
# re4 evidence plan, after Parts C and D: Part H (third-init M probe), then G.1 (decoder edit).
# G.1 edits the served decoder bias through the shared control file, so it must never overlap
# a rollout run; this script therefore waits for the C/D chain to finish.
SP=/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad
OPENPI=/home/mtaheri/ws_AgibotX2/openpi; REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation; R=$REPO/results/re4_evidence
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=$OPENPI/third_party/libero:$OPENPI/examples/libero:$REPO/openpi
PY=$OPENPI/examples/libero/.venv/bin/python
until grep -q "RE4 CHAIN CD DONE" $SP/re4_chain.log 2>/dev/null; do sleep 60; done
cd $OPENPI; mkdir -p $R/H_third_init $R/G_forensics
echo "#### H healthy log at init-base 5  $(date +%T)"
timeout 3600 $PY -u $REPO/openpi/error_signal.py --port 8000 --suite libero_spatial --episodes 10 --healthy-only --init-base 5 --control $SP/ctl.json --ack $SP/ack.json --out $R/H_third_init/error_signal_init5.json > $R/H_third_init/run.log 2>&1; echo "exit $?"
echo "#### H openloop M at init 5  $(date +%T)"
timeout 1800 $PY -u $REPO/openpi/openloop_id.py --log $R/H_third_init/error_signal_init5.json --probe 0.02 --probe-init 5 --out $R/H_third_init/openloop_init5.json >> $R/H_third_init/run.log 2>&1; echo "exit $?"
( cd $REPO && python3 openpi/re4_record.py calibration --log $R/H_third_init/error_signal_init5.json --openloop $R/H_third_init/openloop_init5.json --id third_init5 >> $R/H_third_init/run.log 2>&1 ) || echo "RECORD FAILED H"
echo "RUN DONE H_third_init"
if [ -f $REPO/openpi/g1_decoder.py ]; then
  echo "#### G.1 decoder realization  $(date +%T)"
  timeout 3600 $PY -u $REPO/openpi/g1_decoder.py --port 8000 --control $SP/ctl.json --ack $SP/ack.json --out $R/G_forensics/decoder_bound.json > $R/G_forensics/decoder_run.log 2>&1; echo "exit $?"
  echo "RUN DONE G1"
else echo "G.1 script absent, skipped"; fi
echo "RE4 POST DONE $(date +%T)"
