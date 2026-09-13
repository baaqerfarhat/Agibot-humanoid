#!/bin/bash
# re4 theory Part 6 intervention B (prereg PREREG_RE4T_6_RY.md amendment): ARX plant, headline fault,
# rotation corrected; spatial n=20 (no-harm check) then libero_10 n=40. Waits for the telemetry batch,
# then runs its own pi0.5 server.
SP=/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad; OPENPI=/home/mtaheri/ws_AgibotX2/openpi; REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation; R=/home/mtaheri/ws_AgibotX2/vla-adaptation/results/re4_theory/6_ry
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=/home/mtaheri/ws_AgibotX2/openpi/third_party/libero:/home/mtaheri/ws_AgibotX2/openpi/examples/libero:/home/mtaheri/ws_AgibotX2/vla-adaptation/openpi
PY=/home/mtaheri/ws_AgibotX2/openpi/examples/libero/.venv/bin/python
until grep -q "RE4T TELE DONE" /tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad/re4t_tele.log 2>/dev/null; do sleep 60; done; sleep 20
cd /home/mtaheri/ws_AgibotX2/openpi && CUDA_VISIBLE_DEVICES=1 nohup /home/mtaheri/ws_AgibotX2/openpi/.venv/bin/python /home/mtaheri/ws_AgibotX2/Agibot-humanoid/vla_adaptation/openpi/ace_server.py --port 8000 --control /tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad/ctl.json --ack /tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad/ack.json > /tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad/pi05_server_ry.log 2>&1 &
t=0; until grep -q "listening" /tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad/pi05_server_ry.log 2>/dev/null; do sleep 10; t=$((t+10)); if [ $t -gt 1500 ]; then echo "SERVER NOT READY"; exit 1; fi; done
BASE="--port 8000 --sev 0.05 --gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.15 --corr-dims 3,4,5 --scenario-reset --ar 1 --log /home/mtaheri/ws_AgibotX2/vla-adaptation/results/phase05/error_signal_so3.json --openloop /home/mtaheri/ws_AgibotX2/vla-adaptation/results/phase05/openloop_so3.json --control /tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad/ctl.json --ack /tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad/ack.json"
go() { local id=$1 suite=$2 n=$3; local d=$R/$id; mkdir -p $d; echo "#### RY $id  $(date +%T)"; timeout 14400 $PY -u /home/mtaheri/ws_AgibotX2/vla-adaptation/openpi/adaptive_law.py $BASE --suite $suite --episodes $n --timing $d/timing.jsonl --out $d/result.json > $d/run.log 2>&1; echo "exit $?  $(grep -E ': [0-9]+/[0-9]+ =|ARX plant' $d/run.log | tr '\n' ' ')"; (cd /home/mtaheri/ws_AgibotX2/vla-adaptation && python3 openpi/re4_record.py record $d/result.json --part 6_ry --run-id $id --cohort $suite --timing $d/timing.jsonl --root /home/mtaheri/ws_AgibotX2/vla-adaptation/results/re4_theory > $d/record.log 2>&1); echo "RUN DONE RY $id"; }
go arx_spatial libero_spatial 20
go arx_libero_10 libero_10 40
P=$(ps -eo pid,args | grep "[a]ce_server.py" | awk '{print $1}'); [ -n "$P" ] && kill $P
echo "RE4T RY DONE $(date +%T)"
