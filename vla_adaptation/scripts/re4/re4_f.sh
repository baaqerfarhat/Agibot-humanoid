#!/bin/bash
# re4 evidence plan, Part F: held vs continued updating from a matched initial estimate, on ALOHA
# and GR1 (prereg_records/PREREG_RE4_F_HELD_VS_CONTINUED.md). Swaps policy servers on GPU 1, one
# at a time. Waits for the LIBERO chain and post-chain steps, and for the runner flags it needs.
SP=/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad
OPENPI=/home/mtaheri/ws_AgibotX2/openpi; REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation; R=$REPO/results/re4_evidence/F_held_vs_continued
N15=/home/mtaheri/ws_AgibotX2/Isaac-GR00T-n15; RC=/home/mtaheri/ws_AgibotX2/robocasa-gr1
flags_ready() { for f in aloha_adapt.py gr1_adapt.py; do grep -q -- '"--f-init"' $REPO/openpi/$f && grep -q -- '"--skip-frozen"' $REPO/openpi/$f || return 1; done; }
until grep -q "RE4 POST DONE" $SP/re4_post.log 2>/dev/null && flags_ready; do sleep 60; done
mkdir -p $R
wait_ready() { local log=$1 t=0; until grep -q "listening" $log 2>/dev/null; do sleep 10; t=$((t+10)); [ $t -gt 1200 ] && { echo "SERVER NOT READY $log"; return 1; }; done; }
echo "#### stopping the pi0.5 LIBERO server  $(date +%T)"; kill 3336432 2>/dev/null; sleep 15
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=$REPO/openpi
# ---------------- ALOHA ----------------
cd $OPENPI && CUDA_VISIBLE_DEVICES=1 nohup $OPENPI/.venv/bin/python $OPENPI/scripts/serve_policy.py --env ALOHA_SIM --port 8002 > $SP/aloha_server.log 2>&1 &
ASRV=$!; wait_ready $SP/aloha_server.log || exit 1
AL="$OPENPI/examples/aloha_sim/.venv/bin/python -u $REPO/openpi/aloha_adapt.py run --port 8002 --episodes 40 --seed 200 --log $REPO/results/aloha/healthy_log.json --openloop $REPO/results/aloha/openloop.json --fault-vec 0.02,0.02,0.02,0.02,0.02,0.02,0,0,0,0,0,0,0,0 --corr-joints 0,1,2,3,4,5 --gamma 0.08 --dead 0.002 --norm-r 0.4 --clip 0.08 --f-init=0.0193,0.0202,0.0190,0.0191,0.0192,0.0192,0,0,0,0,0,0,0,0"
arun() { local id=$1; shift; mkdir -p $R/$id; echo "#### F $id  $(date +%T)"; timeout 28800 $AL "$@" --out $R/$id/result.json > $R/$id/run.log 2>&1; echo "exit $?  $(grep -E ': [0-9]+/[0-9]+ =' $R/$id/run.log | tr '\n' ' ')"; echo "RUN DONE F $id"; }
arun aloha_held --freeze-after 0
arun aloha_cont_legacy --skip-frozen
arun aloha_cont_innov --skip-frozen --law innov
kill $ASRV 2>/dev/null; sleep 15
# ---------------- GR1 ----------------
cd $N15 && CUDA_VISIBLE_DEVICES=1 nohup $N15/.venv/bin/python -u $REPO/openpi/groot15_server.py --model-path $N15/checkpoints/GR00T-N1.5-3B --port 8004 > $SP/gr1_server.log 2>&1 &
GSRV=$!; wait_ready $SP/gr1_server.log || exit 1
GR="$RC/.venv/bin/python -u $REPO/openpi/gr1_adapt.py run --port 8004 --episodes 30 --seed 100 --task gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env --log $REPO/results/gr1/screen_PosttrainPnPNovelFromPlateToPlateSplitA.json --openloop $REPO/results/gr1/openloop_arms_clean.json --fault-vec arm:right:0.10 --corr-joints arm:right --gamma 0.08 --dead 0.013 --norm-r 0.11 --clip 0.2 --f-init=0,0,0,0,0,0,0,0.1041,0.0972,0.0979,0.0987,0.0999,0.0932,0.0963,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
grun() { local id=$1; shift; mkdir -p $R/$id; echo "#### F $id  $(date +%T)"; (cd $RC && timeout 36000 $GR "$@" --out $R/$id/result.json > $R/$id/run.log 2>&1); echo "exit $?  $(grep -E ': [0-9]+/[0-9]+ =' $R/$id/run.log | tr '\n' ' ')"; echo "RUN DONE F $id"; }
grun gr1_held --law innov --freeze-after 0
grun gr1_cont_innov --law innov --skip-frozen
grun gr1_cont_legacy --law legacy --skip-frozen
kill $GSRV 2>/dev/null
echo "RE4 F DONE $(date +%T)"
