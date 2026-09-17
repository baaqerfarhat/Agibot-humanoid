#!/bin/bash
# Delayed-NT repair (PREREG_FYA_DELAY_REPAIR_V1.md): full ordered manifest, one arm, GPU 1 after the healthy replication releases it.
source /home/mtaheri/ws_AgibotX2/vla-adaptation/scripts/re4/fya_sf_common.sh
E=$R/delay_repair; M=$R/eval_manifest.json; FV=0.10,0.10,0.10,0.10,0.10,0.10
until grep -q "HC2 DONE" $SP/hc2.log 2>/dev/null; do sleep 30; done; sleep 10
stamp "DELAY REPAIR server up (gpu 1)"; server_up 1 8000 delayrep || exit 1
stamp "eval_delay_nt_full (gpu 1)"
timeout 7200 $PY -u $REPO/openpi/adaptive_law.py --port 8000 --control $SP/ctl_sf1.json --ack $SP/ack_sf1.json --suite libero_spatial \
  --log $REPO/results/phase05/error_signal_so3.json --openloop $REPO/results/phase05/openloop_so3.json --gamma 0.08 --dead 0.008 --norm-r 0.15 --clip 0.05 \
  --corr-dims 3,4,5 --norm-channels all --scenario-reset --manifest $M --episodes 40 --arms adaptive --law legacy --fault-vec $FV --onset 40 --adapt-from 40 \
  --out $E/eval_delay_nt_full.json --telemetry $E/eval_delay_nt_full_telemetry.jsonl > $E/eval_delay_nt_full.log 2>&1; echo "exit $?"
server_down 8000; cd $REPO && gzip -f $E/eval_delay_nt_full_telemetry.jsonl
stamp "DELAY REPAIR coupling checks"
python3 $REPO/openpi/re4_theory/fya_source_coupling_report.py --src $R/evaluation --a eval_fault_nt --b ../delay_repair/eval_delay_nt_full --prefix 30 --out $E/coupling_prefix_vs_immediate_nt.json > $E/coupling1.log 2>&1 || true
python3 - <<'PY' > $E/coupling_report.json
import json,gzip,numpy as np,pathlib
R="/home/mtaheri/ws_AgibotX2/vla-adaptation/results/fya_stronger_fault_v1"
def eps(p):
    out={}
    with gzip.open(p,"rt") as fh:
        for line in fh:
            r=json.loads(line)
            if r.get("type")=="step" and r.get("phase")=="rollout": out.setdefault((r["task"],r["init"]),[]).append(r)
    for k in out: out[k].sort(key=lambda s:s["t"])
    return out
D=eps(f"{R}/delay_repair/eval_delay_nt_full_telemetry.jsonl.gz"); N=eps(f"{R}/evaluation/eval_fault_nt_telemetry.jsonl.gz"); O=eps(f"{R}/evaluation/eval_fault_off_telemetry.jsonl.gz")
rows=[]
for k in sorted(D):
    d=D[k]; n=N.get(k,[]); o=O.get(k,[])
    def gap(a,b,lo,hi):
        m=min(len(a),len(b),hi); 
        return float(max(np.abs(np.array(x["raw_action"])-np.array(y["raw_action"])).max() for x,y in zip(a[lo:m],b[lo:m]))) if m>lo else None
    rows.append(dict(task=k[0],init=k[1],prefix_gap_vs_nt=gap(d,n,0,30),prefix_gap_vs_off=gap(d,o,0,30),preactivation_gap_vs_off=gap(d,o,30,40),
                     first_correction_step=next((s["t"]-10 for s in d if np.any(np.abs(np.array(s["correction"]))>0)),None),
                     first_update_step=next((s["t"]-10 for s in d if s.get("update_applied")),None), len=len(d)))
print(json.dumps(dict(n=len(rows),prefix_exact_vs_nt=sum(r["prefix_gap_vs_nt"]==0.0 for r in rows),prefix_exact_vs_off=sum(r["prefix_gap_vs_off"]==0.0 for r in rows),
                      preactivation_exact_vs_off=sum(r["preactivation_gap_vs_off"]==0.0 for r in rows),first_correction_steps=sorted({r["first_correction_step"] for r in rows}),
                      first_update_steps=sorted({r["first_update_step"] for r in rows}),rows=rows),indent=1))
PY
stamp "DELAY REPAIR score"
python3 $REPO/openpi/re4_theory/fya_comparator_score.py --manifest $M --arm fault_off=$R/evaluation/eval_fault_off.json --arm fault_nt=$R/evaluation/eval_fault_nt.json --arm delay_nt=$E/eval_delay_nt_full.json \
  --primary delay_nt-fault_nt --secondary delay_nt-fault_off,fault_nt-fault_off --out $E/score.json > $E/score.log 2>&1; echo "exit $?"
echo "DELAY REPAIR DONE $(date +%T)"
