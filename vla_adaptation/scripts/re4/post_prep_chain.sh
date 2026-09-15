#!/bin/bash
# After the prep chain: E2 probe qualification (six declared checkpoints) and E1 pass 1 / pass 2. Simulator only.
SP=/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad
OPENPI=/home/mtaheri/ws_AgibotX2/openpi; REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation; R=$REPO/results/iclr_unified_v1
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=$OPENPI/third_party/libero:$OPENPI/examples/libero:$REPO/openpi:$REPO/openpi/re4_theory
PY=$OPENPI/examples/libero/.venv/bin/python; SRC=$R/sources
until grep -q "PREP CHAIN DONE" $SP/prep_chain.log 2>/dev/null; do sleep 60; done
mkdir -p $R/E2_probe $R/E1
# episode index in the init40-43 log = (state-40)*10 + task
echo "#### E2PROBE fit_cp30 (task0@40, task8@42)  $(date +%T)"; $PY $REPO/openpi/re4_theory/e2_probe.py --log $SRC/e2_fit_qual_spatial_init40_43.json --episodes 0,28 --checkpoints 30 --horizon 40 --split-label fit --out $R/E2_probe/fit_cp30.json > $R/E2_probe/fit_cp30.log 2>&1; echo "exit $?"
echo "#### E2PROBE fit_cp70 (task4@41)  $(date +%T)"; $PY $REPO/openpi/re4_theory/e2_probe.py --log $SRC/e2_fit_qual_spatial_init40_43.json --episodes 14 --checkpoints 70 --horizon 40 --split-label fit --out $R/E2_probe/fit_cp70.json > $R/E2_probe/fit_cp70.log 2>&1; echo "exit $?"
echo "#### E2PROBE qual_cp30 (task1@43, task9@43)  $(date +%T)"; $PY $REPO/openpi/re4_theory/e2_probe.py --log $SRC/e2_fit_qual_spatial_init40_43.json --episodes 31,39 --checkpoints 30 --horizon 40 --split-label qualification --out $R/E2_probe/qual_cp30.json > $R/E2_probe/qual_cp30.log 2>&1; echo "exit $?"
echo "#### E2PROBE qual_cp70 (task5@43)  $(date +%T)"; $PY $REPO/openpi/re4_theory/e2_probe.py --log $SRC/e2_fit_qual_spatial_init40_43.json --episodes 35 --checkpoints 70 --horizon 40 --split-label qualification --out $R/E2_probe/qual_cp70.json > $R/E2_probe/qual_cp70.log 2>&1; echo "exit $?"
echo "RUN DONE E2PROBE"
echo "#### E1 pass1 fit+qual (init39, episodes 0-9)  $(date +%T)"; $PY $REPO/openpi/re4_theory/e1_continuations.py --log $SRC/e1_fitqual_spatial_init39.json --episodes all --checkpoints 30,70 --horizon 100 --split-label fit_qualification --out $R/E1/pass1_fitqual.json > $R/E1/pass1_fitqual.log 2>&1; echo "exit $?"; echo "RUN DONE E1 pass1"
HOLD=$(python3 - <<'PY'
import json,numpy as np
d=json.load(open("/home/mtaheri/ws_AgibotX2/vla-adaptation/results/iclr_unified_v1/E1/pass1_fitqual.json"))
vals=[]
for ep in d["episodes"]:
    if ep["task"] < 6: continue                       # qualification sources only (tasks 6-9 at state 39)
    for cp in ep["checkpoints"]:
        if cp.get("status")=="ok": vals.append(np.mean([s["f_hat"] for s in cp["branches"]["innovation_from_zero"][-20:]],axis=0))
h=np.median(np.array(vals),axis=0) if vals else np.zeros(6); print(",".join(f"{x:.5f}" for x in h))
PY
); echo "HOLD ESTIMATE $HOLD"; echo "$HOLD" > $R/E1/hold_estimate.txt
echo "#### E1 pass2 locked test (init34)  $(date +%T)"; $PY $REPO/openpi/re4_theory/e1_continuations.py --log $SRC/e1_test_spatial_init34.json --episodes all --checkpoints 30,70 --horizon 100 --hold-estimate $HOLD --split-label locked_test --out $R/E1/pass2_test.json > $R/E1/pass2_test.log 2>&1; echo "exit $?"; echo "RUN DONE E1 pass2"
echo "POST PREP DONE $(date +%T)"
