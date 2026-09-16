import numpy as np, itertools, time, json
from math import comb
from sim import run
FAULTS = ["healthy", "offset", "config", "friction", "oscill"]
MATCH = {"healthy": "const", "offset": "const", "config": "conf", "friction": "fric", "oscill": "const"}
QS, KS = [1e-6, 1e-5, 1e-4, 1e-3], [0.3, 1.0, 3.0]
ARMS = {"const k=0": ("const", [0.0]), "const k>0": ("const", KS),
        "state k=0": ("state", [0.0]), "state k>0": ("state", KS)}
def mcn(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)

t0 = time.time(); run("offset", regressor="state", kappa=1.0, E=600, seed=1); print(f"one dev run: {time.time()-t0:.1f}s")
best = {}
for arm, (reg, ks) in ARMS.items():            # identical grid & selection rule for every arm
    scores = {}
    for q, k in itertools.product(QS, ks):
        scores[(q, k)] = np.mean([run(f, regressor=reg, kappa=k, q=q, E=600, seed=101).mean() for f in FAULTS])
    best[arm] = max(scores, key=scores.get)
    print(f"tuned {arm:<10} -> q={best[arm][0]:g} kappa={best[arm][1]:g}  dev mean={scores[best[arm]]:.3f}")

print("\nEVALUATION (fresh seed, E=2000 paired episodes per cell)")
res = {}
hdr = ["frozen"] + list(ARMS) + ["matched k=0", "matched k>0"]
print(f"{'fault':<9}" + "".join(f"{h:>12}" for h in hdr))
for f in FAULTS:
    row = {"frozen": run(f, adapt=False, E=2000, seed=202)[0]}
    for arm, (reg, _) in ARMS.items():
        q, k = best[arm]; row[arm] = run(f, regressor=reg, kappa=k, q=q, E=2000, seed=202)[0]
    for tag, src in (("matched k=0", "const k=0"), ("matched k>0", "const k>0")):
        q, k = best[src]; row[tag] = run(f, regressor=MATCH[f], kappa=k, q=q, E=2000, seed=202)[0]
    res[f] = row
    print(f"{f:<9}" + "".join(f"{row[h].mean():>12.3f}" for h in hdr))

print("\nTHE QUESTION: contraction+state vs contraction+input-only (both k>0), paired")
for f in FAULTS:
    a, b = res[f]["const k>0"], res[f]["state k>0"]
    fx, bk = int((~a & b).sum()), int((a & ~b).sum())
    print(f"  {f:<9} const {a.mean():.3f}  state {b.mean():.3f}  diff {b.mean()-a.mean():+.3f}   state-only wins {fx:4d} / const-only wins {bk:4d}  p={mcn(fx,bk):.2g}")
print("\nWHAT CONTRACTION (tracking) ADDS, per regressor, paired")
for reg in ("const", "state"):
    for f in FAULTS:
        a, b = res[f][f"{reg} k=0"], res[f][f"{reg} k>0"]
        print(f"  {reg:<6}{f:<9} k=0 {a.mean():.3f} -> k>0 {b.mean():.3f}  ({b.mean()-a.mean():+.3f})")
json.dump({f: {h: float(v.mean()) for h, v in r.items()} for f, r in res.items()} | {"tuned": {a: list(v) for a, v in best.items()}},
          open("eval_exact_results.json", "w"), indent=1)
