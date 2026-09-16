#!/bin/bash
# E1 ALOHA, twenty fresh sources (PREREG_E1_ALOHA_CONTINUATIONS.md, extension): collect on the GPU-0 ALOHA server, then the sim-only passes.
SP=/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad
REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation; E=$REPO/results/iclr_unified_v1/E1_aloha_v2; PY=/home/mtaheri/ws_AgibotX2/openpi/examples/aloha_sim/.venv/bin/python
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=$REPO/openpi:$REPO/openpi/re4_theory
mkdir -p $E
t=0; until grep -q "listening\|Serving\|server" $SP/aloha_server_gpu0.log 2>/dev/null; do sleep 10; t=$((t+10)); if [ $t -gt 1800 ]; then echo "SERVER NOT READY"; exit 1; fi; done; sleep 5
echo "#### E1A2 collect 20 healthy sources (seeds 400-419)  $(date +%T)"; timeout 7200 $PY -u $REPO/openpi/aloha_adapt.py log --port 8002 --episodes 20 --seed 400 --out $REPO/results/aloha/healthy_log_e1_seed400.json > $E/collect.log 2>&1; echo "exit $?  $(grep -E 'successes|FIR R2' $E/collect.log | tr '\n' ' ' | cut -c1-160)"; echo "RUN DONE E1A2 collect"
P=$(ps -eo pid,args | grep "[s]erve_policy.py" | grep "port 8002" | awk '{print $1}'); [ -n "$P" ] && kill $P && echo "aloha server $P killed"
echo "#### E1A2 pass1 fit+qual (episodes 0-9)  $(date +%T)"; $PY $REPO/openpi/re4_theory/e1_aloha_continuations.py --log $REPO/results/aloha/healthy_log_e1_seed400.json --episodes 0,1,2,3,4,5,6,7,8,9 --checkpoints 60,120 --horizon 50 --split-label fit_qualification --out $E/pass1_fitqual.json > $E/pass1_fitqual.log 2>&1; echo "exit $?"; echo "RUN DONE E1A2 pass1"
HOLD=$(python3 - <<'PY'
import json,numpy as np
d=json.load(open("/home/mtaheri/ws_AgibotX2/vla-adaptation/results/iclr_unified_v1/E1_aloha_v2/pass1_fitqual.json")); vals=[]
for ep in d["episodes"]:
    if ep["episode"] < 6: continue
    for cp in ep["checkpoints"]:
        if cp.get("status")=="ok": vals.append(np.mean([s["f_hat"] for s in cp["branches"]["innovation_from_zero"][-20:]],axis=0))
h=np.median(np.array(vals),axis=0) if vals else np.zeros(14); print(",".join(f"{x:.5f}" for x in h))
PY
); echo "HOLD ESTIMATE $HOLD"; echo "$HOLD" > $E/hold_estimate.txt; sha256sum $E/hold_estimate.txt $E/pass1_fitqual.json > $E/FROZEN_BEFORE_TEST.sha256
echo "#### E1A2 pass2 locked test (episodes 10-19)  $(date +%T)"; $PY $REPO/openpi/re4_theory/e1_aloha_continuations.py --log $REPO/results/aloha/healthy_log_e1_seed400.json --episodes 10,11,12,13,14,15,16,17,18,19 --checkpoints 60,120 --horizon 50 --hold-estimate $HOLD --split-label locked_test --out $E/pass2_test.json > $E/pass2_test.log 2>&1; echo "exit $?"; echo "RUN DONE E1A2 pass2"
cd $REPO && python3 openpi/re4_theory/e1_aloha_score.py $E/pass1_fitqual.json --out $E/pass1_score.json > /dev/null 2>&1; python3 openpi/re4_theory/e1_aloha_score.py $E/pass2_test.json --out $E/pass2_score.json > /dev/null 2>&1; python3 openpi/re4_theory/e1_memory_model.py --pass1 $E/pass1_fitqual.json --pass2 $E/pass2_test.json --space joint --joints 0,1,2,3,4,5 --fit-key episode --fit-below 6 --out $E/memory_model_score.json > /dev/null 2>&1; echo "E1A2 DONE $(date +%T)"
