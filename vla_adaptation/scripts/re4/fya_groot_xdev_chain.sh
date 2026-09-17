#!/bin/bash
# Cross-device reproducibility control (PREREG_FYA_GROOT_HEALTHY_CROSSED_V1.md, part B addendum): healthy_off on the same 40 keys
# from a second GR00T server process on GPU 0 (rendering on GPU 1). Compared afterwards, whole-episode, to the GPU 1 healthy_off arm.
SP=/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad
OPENPI=/home/mtaheri/ws_AgibotX2/openpi; REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation; GR=/home/mtaheri/ws_AgibotX2/Isaac-GR00T
R=$REPO/results/fya_groot_healthy_crossed_v1; S=$R/xdevice_control; M=$R/manifest.json; mkdir -p $S
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=$OPENPI/third_party/libero:$OPENPI/examples/libero:$REPO/openpi:$REPO/openpi/re4_theory
PY=$OPENPI/examples/libero/.venv/bin/python; GPY=$GR/.venv/bin/python; CKPT=$GR/checkpoints/GR00T-N1.7-LIBERO/libero_spatial
CTL=$SP/ctl_groot0.json; ACK=$SP/ack_groot0.json; PORT=8004
stamp() { echo "#### $1  $(date +%T)"; }
stamp "GR00T server up on GPU 0 (cross-device control)"; cd $GR && GROOT_HF_LOCAL_FIRST=1 CUDA_VISIBLE_DEVICES=0 nohup $GPY $REPO/openpi/groot_server.py --model-path $CKPT --port $PORT --control $CTL --ack $ACK --calllog $S/server_calllog.jsonl > $SP/groot_server_xdev.log 2>&1 &
t=0; until grep -q "listening" $SP/groot_server_xdev.log 2>/dev/null; do sleep 10; t=$((t+10)); if [ $t -gt 1800 ]; then echo "SERVER NOT READY"; exit 1; fi; done; cd $REPO
stamp "healthy_off_gpu0 (gr00t, second process)"
timeout 21600 $PY -u $REPO/openpi/adaptive_law.py --port $PORT --control $CTL --ack $ACK --suite libero_spatial --replan-steps 8 \
  --log $REPO/results/phase05/error_signal_so3.json --openloop $REPO/results/phase05/openloop_so3.json --gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.05 \
  --corr-dims 3,4,5 --norm-channels all --scenario-reset --manifest $M --episodes 40 --arms frozen --law legacy --baseline none --fault-vec 0,0,0,0,0,0 --onset 40 --adapt-from 30 \
  --out $S/healthy_off_gpu0.json --telemetry $S/healthy_off_gpu0_telemetry.jsonl --timing $S/healthy_off_gpu0_timing.jsonl > $S/healthy_off_gpu0.log 2>&1; echo "exit $?"
P=$(ps -eo pid,args | grep "[g]root_server.py" | grep "port $PORT" | awk '{print $1}'); [ -n "$P" ] && kill $P; sleep 10
cd $REPO && gzip -f $S/healthy_off_gpu0_telemetry.jsonl
echo "GROOT XDEV DONE $(date +%T)"
