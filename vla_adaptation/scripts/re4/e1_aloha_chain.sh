#!/bin/bash
# E1 on ALOHA (PREREG_E1_ALOHA_CONTINUATIONS.md): simulator only, EGL on GPU 1. Pass 1 (fit+qual), hold estimate, pass 2 (test), scores.
SP=/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad
REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation; E=$REPO/results/iclr_unified_v1/E1_aloha; PY=/home/mtaheri/ws_AgibotX2/openpi/examples/aloha_sim/.venv/bin/python
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=$REPO/openpi:$REPO/openpi/re4_theory
mkdir -p $E
echo "#### E1A pass1 fit+qual (episodes 0-4)  $(date +%T)"; $PY $REPO/openpi/re4_theory/e1_aloha_continuations.py --episodes 0,1,2,3,4 --checkpoints 60,120 --horizon 50 --split-label fit_qualification --out $E/pass1_fitqual.json > $E/pass1_fitqual.log 2>&1; echo "exit $?"; echo "RUN DONE E1A pass1"
HOLD=$(python3 - <<'PY'
import json,numpy as np
d=json.load(open("/home/mtaheri/ws_AgibotX2/vla-adaptation/results/iclr_unified_v1/E1_aloha/pass1_fitqual.json")); vals=[]
for ep in d["episodes"]:
    if ep["episode"] < 3: continue
    for cp in ep["checkpoints"]:
        if cp.get("status")=="ok": vals.append(np.mean([s["f_hat"] for s in cp["branches"]["innovation_from_zero"][-20:]],axis=0))
h=np.median(np.array(vals),axis=0) if vals else np.zeros(14); print(",".join(f"{x:.5f}" for x in h))
PY
); echo "HOLD ESTIMATE $HOLD"; echo "$HOLD" > $E/hold_estimate.txt; sha256sum $E/hold_estimate.txt $E/pass1_fitqual.json > $E/FROZEN_BEFORE_TEST.sha256
echo "#### E1A pass2 locked test (episodes 5-7)  $(date +%T)"; $PY $REPO/openpi/re4_theory/e1_aloha_continuations.py --episodes 5,6,7 --checkpoints 60,120 --horizon 50 --hold-estimate $HOLD --split-label locked_test --out $E/pass2_test.json > $E/pass2_test.log 2>&1; echo "exit $?"; echo "RUN DONE E1A pass2"
cd $REPO && python3 openpi/re4_theory/e1_aloha_score.py $E/pass1_fitqual.json --out $E/pass1_score.json > /dev/null 2>&1; python3 openpi/re4_theory/e1_aloha_score.py $E/pass2_test.json --out $E/pass2_score.json > /dev/null 2>&1; echo "E1A DONE $(date +%T)"
