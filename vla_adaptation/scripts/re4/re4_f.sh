#!/bin/bash
# Part F, corrected chain: GR1 arms against the live GR1 server (pid 3389138), then the ALOHA arms.
# wait_ready ends with an explicit 'return 0': the earlier version returned the status of its last
# loop iteration (a failed [ ] test) whenever the server needed time to load, and the chain exited.
SP=/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad; OPENPI=/home/mtaheri/ws_AgibotX2/openpi; REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation; N15=/home/mtaheri/ws_AgibotX2/Isaac-GR00T-n15; RC=/home/mtaheri/ws_AgibotX2/robocasa-gr1; R=/home/mtaheri/ws_AgibotX2/vla-adaptation/results/re4_evidence/F_held_vs_continued
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=/home/mtaheri/ws_AgibotX2/vla-adaptation/openpi
wait_ready() { local log=$1 t=0; until grep -q "listening" $log 2>/dev/null; do sleep 10; t=$((t+10)); if [ $t -gt 1500 ]; then echo "SERVER NOT READY $log"; return 1; fi; done; return 0; }
mkdir -p $R
GSRV=3389138; wait_ready /tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad/gr1_server.log || exit 1
GR="/home/mtaheri/ws_AgibotX2/robocasa-gr1/.venv/bin/python -u /home/mtaheri/ws_AgibotX2/vla-adaptation/openpi/gr1_adapt.py run --port 8004 --episodes 30 --seed 100 --task gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env --log /home/mtaheri/ws_AgibotX2/vla-adaptation/results/gr1/screen_PosttrainPnPNovelFromPlateToPlateSplitA.json --openloop /home/mtaheri/ws_AgibotX2/vla-adaptation/results/gr1/openloop_arms_clean.json --fault-vec arm:right:0.10 --corr-joints arm:right --gamma 0.08 --dead 0.013 --norm-r 0.11 --clip 0.2 --f-init=0,0,0,0,0,0,0,0.1041,0.0972,0.0979,0.0987,0.0999,0.0932,0.0963,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
grun() { local id=$1; shift; mkdir -p $R/$id; echo "#### F $id  $(date +%T)"; (cd /home/mtaheri/ws_AgibotX2/robocasa-gr1 && timeout 36000 $GR "$@" --out $R/$id/result.json > $R/$id/run.log 2>&1); echo "exit $?  $(grep -E ': [0-9]+/[0-9]+ =' $R/$id/run.log | tr '\n' ' ')"; echo "RUN DONE F $id"; }
grun gr1_held --law innov --freeze-after 0
grun gr1_cont_innov --law innov --skip-frozen
grun gr1_cont_legacy --law legacy --skip-frozen
kill $GSRV 2>/dev/null; sleep 20
cd /home/mtaheri/ws_AgibotX2/openpi && CUDA_VISIBLE_DEVICES=1 nohup /home/mtaheri/ws_AgibotX2/openpi/.venv/bin/python /home/mtaheri/ws_AgibotX2/openpi/scripts/serve_policy.py --env ALOHA_SIM --port 8002 > /tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad/aloha_server2.log 2>&1 &
ASRV=$!; wait_ready /tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad/aloha_server2.log || exit 1
AL="/home/mtaheri/ws_AgibotX2/openpi/examples/aloha_sim/.venv/bin/python -u /home/mtaheri/ws_AgibotX2/vla-adaptation/openpi/aloha_adapt.py run --port 8002 --episodes 40 --seed 200 --log /home/mtaheri/ws_AgibotX2/vla-adaptation/results/aloha/healthy_log.json --openloop /home/mtaheri/ws_AgibotX2/vla-adaptation/results/aloha/openloop.json --fault-vec 0.02,0.02,0.02,0.02,0.02,0.02,0,0,0,0,0,0,0,0 --corr-joints 0,1,2,3,4,5 --gamma 0.08 --dead 0.002 --norm-r 0.4 --clip 0.08 --f-init=0.0193,0.0202,0.0190,0.0191,0.0192,0.0192,0,0,0,0,0,0,0,0"
arun() { local id=$1; shift; mkdir -p $R/$id; echo "#### F $id  $(date +%T)"; timeout 28800 $AL "$@" --timing $R/$id/timing.jsonl --out $R/$id/result.json > $R/$id/run.log 2>&1; echo "exit $?  $(grep -E ': [0-9]+/[0-9]+ =' $R/$id/run.log | tr '\n' ' ')"; echo "RUN DONE F $id"; }
arun aloha_held --freeze-after 0
arun aloha_cont_legacy --skip-frozen
arun aloha_cont_innov --skip-frozen --law innov
kill $ASRV 2>/dev/null
echo "RE4 F DONE $(date +%T)"
