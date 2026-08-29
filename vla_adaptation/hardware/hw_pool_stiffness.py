"""Within-session normalisation, then pool. Each session is normalised by its OWN
gain_scale=1.0 runs, which removes the session/checkpoint offset that confounded the first
attempt; the normalised ratios are then pooled across sessions."""
import json, glob, csv, collections, pathlib, numpy as np
LOGS="/home/mtaheri/ws_AgibotX2/Agibot-humanoid/run_logs"
JOINTS=["left_hip_pitch_joint","right_hip_pitch_joint","left_knee_joint","right_knee_joint"]
def load(p):
    rows=list(csv.reader(open(p))); hdr=rows[0]; D=[r for r in rows[1:] if len(r)==len(hdr)]
    idx={h:i for i,h in enumerate(hdr)}
    def col(n):
        i=idx[n]; return np.array([float(r[i]) if r[i] not in("","nan") else np.nan for r in D])
    return col, idx
def stiff(f):
    try: col,idx=load(f)
    except Exception: return None
    out={}
    for j in JOINTS:
        if f"{j}__eff_meas" not in idx: continue
        e,t,p=col(f"{j}__eff_meas"),col(f"{j}__tgt"),col(f"{j}__pos_meas")
        m=~(np.isnan(e)|np.isnan(t)|np.isnan(p))
        if m.sum()<60: continue
        err=(t-p)[m]; e=e[m]
        k=np.abs(err)>np.percentile(np.abs(err),60)
        if k.sum()<20: continue
        out[j]=float(np.polyfit(err[k],e[k],1)[0])
    return out
recs=[]
for m in glob.glob(f"{LOGS}/*.meta.json"):
    try: d=json.load(open(m))
    except Exception: continue
    c=m.replace(".meta.json",".csv")
    if pathlib.Path(c).exists() and d.get("gain_scale") is not None:
        recs.append(dict(f=c,g=float(d["gain_scale"]),run=d.get("run_name",""),date=d.get("created","")[:8]))
grp=collections.defaultdict(list)
for r in recs: grp[(r["date"],r["run"])].append(r)
pool=collections.defaultdict(list)
for key,v in grp.items():
    base_runs=[x["f"] for x in v if x["g"]==1.0]
    if not base_runs or len({x["g"] for x in v})<2: continue
    bs=[stiff(f) for f in base_runs]; bs=[x for x in bs if x]
    if not bs: continue
    base={j:np.mean([x[j] for x in bs if j in x]) for j in JOINTS if any(j in x for x in bs)}
    for x in v:
        s=stiff(x["f"])
        if not s: continue
        for j in s:
            if j in base and abs(base[j])>1e-6:
                pool[x["g"]].append(s[j]/base[j])
print("effective stiffness, normalised within session then pooled across sessions\n")
print(f"{'gain_scale':>10} {'n joints':>9} {'measured ratio':>16} {'expected':>9} {'error':>8}")
gs=sorted(pool)
xs,ys=[],[]
for g in gs:
    a=np.array(pool[g]); a=a[np.isfinite(a)]
    a=a[(a>0)&(a<5)]                       # drop degenerate fits
    if len(a)<3: continue
    print(f"{g:>10} {len(a):>9} {np.median(a):>16.3f} {g:>9.2f} {np.median(a)-g:>8.3f}")
    xs.append(g); ys.append(np.median(a))
if len(xs)>2:
    r=np.corrcoef(xs,ys)[0,1]; sl=np.polyfit(xs,ys,1)[0]
    print(f"\ncorrelation with true gain_scale: r = {r:+.3f}   slope = {sl:.2f}  (1.0 = perfect)")
