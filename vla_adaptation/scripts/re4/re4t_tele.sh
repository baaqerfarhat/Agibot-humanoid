#!/bin/bash
# re4 theory plan, Parts 3/4/5 telemetry batch (prereg PREREG_RE4T_3_4_5_TELEMETRY.md). One GPU job at a time:
# waits for the Part 1 simulator replay to finish before starting the pi0.5 server.
SP=/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad; OPENPI=/home/mtaheri/ws_AgibotX2/openpi; REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation; R=/home/mtaheri/ws_AgibotX2/vla-adaptation/results/re4_theory/telemetry
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=/home/mtaheri/ws_AgibotX2/openpi/third_party/libero:/home/mtaheri/ws_AgibotX2/openpi/examples/libero:/home/mtaheri/ws_AgibotX2/vla-adaptation/openpi
PY=/home/mtaheri/ws_AgibotX2/openpi/examples/libero/.venv/bin/python
while ps -eo args | grep -q "[p]aired_rollout.py"; do sleep 20; done
mkdir -p $R; cd /home/mtaheri/ws_AgibotX2/openpi && CUDA_VISIBLE_DEVICES=1 nohup /home/mtaheri/ws_AgibotX2/openpi/.venv/bin/python /home/mtaheri/ws_AgibotX2/Agibot-humanoid/vla_adaptation/openpi/ace_server.py --port 8000 --control /tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad/ctl.json --ack /tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad/ack.json > /tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad/pi05_server.log 2>&1 &
SRV=$!; t=0; until grep -q "listening" /tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad/pi05_server.log 2>/dev/null; do sleep 10; t=$((t+10)); if [ $t -gt 1500 ]; then echo "SERVER NOT READY"; exit 1; fi; done
BASE="--port 8000 --suite libero_spatial --episodes 20 --gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.15 --corr-dims 3,4,5 --scenario-reset --log /home/mtaheri/ws_AgibotX2/vla-adaptation/results/phase05/error_signal_so3.json --openloop /home/mtaheri/ws_AgibotX2/vla-adaptation/results/phase05/openloop_so3.json --control /tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad/ctl.json --ack /tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad/ack.json"
go() { local id=$1; shift; local d=$R/$id; mkdir -p $d; echo "#### T $id  $(date +%T)"; timeout 14400 $PY -u /home/mtaheri/ws_AgibotX2/vla-adaptation/openpi/adaptive_law.py $BASE "$@" --telemetry $d/telemetry.jsonl --timing $d/timing.jsonl --out $d/result.json > $d/run.log 2>&1; echo "exit $?  $(grep -E ': [0-9]+/[0-9]+ =' $d/run.log | tr '\n' ' ')"; (cd /home/mtaheri/ws_AgibotX2/vla-adaptation && python3 openpi/re4_record.py record $d/result.json --part telemetry --run-id $id --cohort libero_spatial --timing $d/timing.jsonl --root /home/mtaheri/ws_AgibotX2/vla-adaptation/results/re4_theory > $d/record.log 2>&1); echo "RUN DONE T $id"; }
go T1_headline --sev 0.05
go T2_onset40 --sev 0.05 --onset 40
go T3_ramp60 --sev 0.05 --profile ramp --prof-p 60
go T4_innov --sev 0.05 --law innov
kill $SRV 2>/dev/null; sleep 5; P=$(ps -eo pid,args | grep "[a]ce_server.py" | awk '{print $1}'); [ -n "$P" ] && kill $P
echo "RE4T TELE DONE $(date +%T)"
