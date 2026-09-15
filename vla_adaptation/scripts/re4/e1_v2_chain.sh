#!/bin/bash
# E1 v2 (supersession note in PREREG_E1_PHYSICAL_CONTINUATIONS.md): collect the fresh locked test sources (state 33),
# re-run fit/qualification with the corrected driver, freeze the hold estimate, the physical model and the PROSPECTIVE
# forecast, then run the locked test and score. Waits for the E2 core to release the server.
SP=/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/7b471ff2-72f5-4003-ac45-a286d3b67915/scratchpad
OPENPI=/home/mtaheri/ws_AgibotX2/openpi; REPO=/home/mtaheri/ws_AgibotX2/vla-adaptation; R=$REPO/results/iclr_unified_v1; E=$R/E1_v2
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=$OPENPI/third_party/libero:$OPENPI/examples/libero:$REPO/openpi:$REPO/openpi/re4_theory
PY=$OPENPI/examples/libero/.venv/bin/python
until grep -q "E2 CORE DONE" $SP/e2_core_chain.log 2>/dev/null; do sleep 60; done; sleep 20
mkdir -p $E
cd $OPENPI && CUDA_VISIBLE_DEVICES=1 nohup $OPENPI/.venv/bin/python /home/mtaheri/ws_AgibotX2/Agibot-humanoid/vla_adaptation/openpi/ace_server.py --port 8000 --control $SP/ctl.json --ack $SP/ack.json > $SP/pi05_server_e1v2.log 2>&1 &
t=0; until grep -q "listening" $SP/pi05_server_e1v2.log 2>/dev/null; do sleep 10; t=$((t+10)); if [ $t -gt 1500 ]; then echo "SERVER NOT READY"; exit 1; fi; done
echo "#### E1V2 collect test sources state 33  $(date +%T)"; timeout 7200 $PY -u $REPO/openpi/error_signal.py --port 8000 --control $SP/ctl.json --ack $SP/ack.json --suite libero_spatial --healthy-only --init-base 33 --episodes 10 --out $R/sources/e1_test_spatial_init33.json > $R/sources/e1_test_spatial_init33.log 2>&1; echo "exit $?"
P=$(ps -eo pid,args | grep "[a]ce_server.py" | awk '{print $1}'); [ -n "$P" ] && kill $P; sleep 15
echo "#### E1V2 pass1 fit+qual state 39 (driver v2)  $(date +%T)"; $PY $REPO/openpi/re4_theory/e1_continuations.py --log $R/sources/e1_fitqual_spatial_init39.json --episodes all --checkpoints 20,40 --horizon 50 --split-label fit_qualification_v2 --out $E/pass1_fitqual.json > $E/pass1_fitqual.log 2>&1; echo "exit $?"; echo "RUN DONE E1V2 pass1"
HOLD=$(python3 - <<'PY'
import json,numpy as np
d=json.load(open("/home/mtaheri/ws_AgibotX2/vla-adaptation/results/iclr_unified_v1/E1_v2/pass1_fitqual.json")); vals=[]
for ep in d["episodes"]:
    if ep["task"] < 6: continue
    for cp in ep["checkpoints"]:
        if cp.get("status")=="ok": vals.append(np.mean([s["f_hat"] for s in cp["branches"]["innovation_from_zero"][-20:]],axis=0))
h=np.median(np.array(vals),axis=0) if vals else np.zeros(6); print(",".join(f"{x:.5f}" for x in h))
PY
); echo "HOLD ESTIMATE $HOLD"; echo "$HOLD" > $E/hold_estimate.txt
cd $REPO && python3 openpi/re4_theory/e1_score.py $E/pass1_fitqual.json --out $E/pass1_score.json > /dev/null 2>&1
# physical model from fit/qualification ONLY (pass2 arg points at pass1 so the locked scoring block is empty; the frozen_model block is what we need)
python3 openpi/re4_theory/e1_memory_model.py --pass1 $E/pass1_fitqual.json --pass2 $E/pass1_fitqual.json --out $E/model_fitqual.json > /dev/null 2>&1; echo "model frozen exit $?"
echo "#### E1V2 forecast (before the locked pass)  $(date +%T)"; python3 openpi/re4_theory/e1_forecast.py --model $E/model_fitqual.json --hold $HOLD --out $E/forecast_v2.json; echo "exit $?"; sha256sum $E/forecast_v2.json $E/model_fitqual.json $E/hold_estimate.txt > $E/FROZEN_BEFORE_TEST.sha256; echo "RUN DONE E1V2 forecast"
echo "#### E1V2 pass2 locked test state 33  $(date +%T)"; $PY $REPO/openpi/re4_theory/e1_continuations.py --log $R/sources/e1_test_spatial_init33.json --episodes all --checkpoints 20,40 --horizon 50 --hold-estimate $HOLD --split-label locked_test_v2 --out $E/pass2_test.json > $E/pass2_test.log 2>&1; echo "exit $?"; echo "RUN DONE E1V2 pass2"
python3 openpi/re4_theory/e1_score.py $E/pass2_test.json --out $E/pass2_score.json > /dev/null 2>&1; python3 openpi/re4_theory/e1_memory_model.py --pass1 $E/pass1_fitqual.json --pass2 $E/pass2_test.json --out $E/memory_model_score.json > /dev/null 2>&1; echo "scored"
echo "E1V2 DONE $(date +%T)"
