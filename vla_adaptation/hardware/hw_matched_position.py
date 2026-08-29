"""Within-session comparison: same day, same policy checkpoint, only gain_scale differs."""
import json, glob, csv, collections, numpy as np, pathlib, sys
sys.path.insert(0, "/home/mtaheri/ws_AgibotX2/Agibot-humanoid/vla_adaptation/hardware")
from hw_detect import load, design, fit_plant, residual, K
LOGS = "/home/mtaheri/ws_AgibotX2/Agibot-humanoid/run_logs"
joints = ["left_hip_pitch_joint","right_hip_pitch_joint","left_knee_joint",
          "right_knee_joint","left_hip_yaw_joint"]
recs=[]
for m in glob.glob(f"{LOGS}/*.meta.json"):
    try: d=json.load(open(m))
    except Exception: continue
    csvf=m.replace(".meta.json",".csv")
    if not pathlib.Path(csvf).exists(): continue
    recs.append(dict(f=csvf, g=d.get("gain_scale"), run=d.get("run_name",""),
                     date=d.get("created","")[:8]))
# group by (date, policy) so only gain_scale varies inside a group
grp=collections.defaultdict(list)
for r in recs:
    if r["g"] is not None: grp[(r["date"], r["run"])].append(r)
usable={k:v for k,v in grp.items() if len({x["g"] for x in v})>=2}
print(f"{len(recs)} logs; {len(usable)} (date, policy) groups contain >=2 gain_scale values\n")
for (date,run),v in sorted(usable.items()):
    gs=sorted({x["g"] for x in v})
    print(f"  {date}  {run[:46]:<46} gains {gs}  n={len(v)}")
if not usable:
    print("\nNO within-session gain_scale variation -> the comparison cannot be deconfounded"); raise SystemExit
# fit the plant on that session's OWN healthy runs
for (date,run),v in sorted(usable.items()):
    healthy=[x["f"] for x in v if x["g"]==1.0]
    if not healthy: 
        print(f"\n{date} {run[:40]}: no gain_scale=1.0 run in-session, skipped"); continue
    W=fit_plant(healthy, joints)
    print(f"\n=== {date}  {run[:50]}   plant fitted on {len(healthy)} in-session healthy run(s)")
    print(f"{'gain':>6} {'n':>3} " + " ".join(f"{j.split('_joint')[0][:11]:>12}" for j in joints))
    for g in sorted({x["g"] for x in v}):
        fs=[x["f"] for x in v if x["g"]==g]
        vals=[residual(f,W,joints) for f in fs]; vals=[x for x in vals if x]
        if not vals: continue
        row={j: float(np.mean([x[j] for x in vals if j in x])) for j in joints}
        print(f"{g:>6} {len(vals):>3} " + " ".join(f"{row.get(j,float('nan')):>12.5f}" for j in joints))
