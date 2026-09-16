import numpy as np, itertools
from math import comb
from sim import run
GE=(0.90,0.36); F=["healthy","offset","config","friction"]
def mcn(b,c):
    n=b+c; return 1.0 if n==0 else min(1.0,2*sum(comb(n,i) for i in range(min(b,c)+1))/2**n)
for ref_true in (True, False):
    lab = "reference uses TRUE gain (previous run)" if ref_true else "reference uses the ADAPTER's fitted gain (realistic)"
    print(f"\n=== {lab} ===")
    for reg in ("const","state"):
        # identical grid & selection for each arm, dev seed 303
        sc={(q,k):np.mean([run(f,regressor=reg,kappa=k,q=q,E=600,seed=303,gain_err=GE,ref_uses_model=not ref_true).mean() for f in F])
            for q,k in itertools.product([1e-4,1e-3],[0.0,0.1,0.3,1.0,3.0])}
        q,k=max(sc,key=sc.get)
        k0=max([kk for kk in sc if kk[1]==0.0],key=sc.get)
        print(f"  {reg}: tuned kappa={k:g} (q={q:g}); dev mean with kappa=0 {sc[k0]:.3f} vs tuned {sc[(q,k)]:.3f}")
        if k>0:
            for f in F:
                a=run(f,regressor=reg,kappa=0.0,q=k0[0],E=2000,seed=404,gain_err=GE,ref_uses_model=not ref_true)[0]
                b=run(f,regressor=reg,kappa=k,q=q,E=2000,seed=404,gain_err=GE,ref_uses_model=not ref_true)[0]
                fx,bk=int((~a&b).sum()),int((a&~b).sum())
                print(f"     {f:<9} k=0 {a.mean():.3f} -> k={k:g} {b.mean():.3f} ({b.mean()-a.mean():+.3f}) p={mcn(fx,bk):.2g}")
    # sensitivity of the const arm to kappa (offset + healthy), fixed q
    print("  kappa sensitivity, const, offset/healthy:", end="")
    for kk in (0.0,0.1,0.3,1.0,3.0):
        o=run("offset",regressor="const",kappa=kk,q=1e-4,E=1000,seed=505,gain_err=GE,ref_uses_model=not ref_true).mean()
        h=run("healthy",regressor="const",kappa=kk,q=1e-4,E=1000,seed=505,gain_err=GE,ref_uses_model=not ref_true).mean()
        print(f"  k={kk:g}:{o:.2f}/{h:.2f}", end="")
    print()
