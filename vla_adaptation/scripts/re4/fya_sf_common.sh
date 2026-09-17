# shared definitions for the stronger-fault campaign (PREREG_FYA_STRONGER_FAULT_V1.md)
SP=/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad
OPENPI=/home/mtaheri/ws_AgibotX2/openpi; REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation; R=$REPO/results/fya_stronger_fault_v1
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=$OPENPI/third_party/libero:$OPENPI/examples/libero:$REPO/openpi:$REPO/openpi/re4_theory
PY=$OPENPI/examples/libero/.venv/bin/python; SRV=$OPENPI/.venv/bin/python; SERVER=/home/mtaheri/ws_AgibotX2/Agibot-humanoid/vla_adaptation/openpi/ace_server.py
stamp() { echo "#### $1  $(date +%T)"; }
server_up() {  # gpu port tag
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.42
  cd $OPENPI && CUDA_VISIBLE_DEVICES=$1 nohup $SRV $SERVER --port $2 --control $SP/ctl_sf$1.json --ack $SP/ack_sf$1.json > $SP/pi05_server_sf_$3.log 2>&1 &
  local t=0; until grep -q "listening" $SP/pi05_server_sf_$3.log 2>/dev/null; do sleep 10; t=$((t+10)); if [ $t -gt 1500 ]; then echo "SERVER NOT READY"; return 1; fi; done; cd $REPO; return 0
}
server_down() { local P; P=$(ps -eo pid,args | grep "[a]ce_server.py" | grep "port $1" | awk '{print $1}'); [ -n "$P" ] && kill $P; sleep 15; return 0; }
runarm() {  # gpu port manifest name arms law fault-vec out_dir
  local gpu=$1 port=$2 man=$3 name=$4 arms=$5 law=$6 fvec=$7 od=$8
  stamp "$name (gpu $gpu)"
  timeout 7200 $PY -u $REPO/openpi/adaptive_law.py --port $port --control $SP/ctl_sf$gpu.json --ack $SP/ack_sf$gpu.json --suite libero_spatial \
    --log $REPO/results/phase05/error_signal_so3.json --openloop $REPO/results/phase05/openloop_so3.json --gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.05 \
    --corr-dims 3,4,5 --norm-channels all --scenario-reset --manifest $man --episodes 20 --arms $arms --law $law --fault-vec $fvec --onset 40 --adapt-from 30 \
    --out $od/$name.json --telemetry $od/${name}_telemetry.jsonl > $od/$name.log 2>&1; echo "exit $?"
}
