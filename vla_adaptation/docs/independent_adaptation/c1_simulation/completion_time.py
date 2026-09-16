import numpy as np
from sim import run
GE=(0.90,0.36)
print("completion step (median over successful episodes; horizon 280) and success, E=1500 paired")
rows=[("frozen, healthy robot",            "healthy",dict(adapt=False)),
      ("no tracking, healthy",              "healthy",dict(regressor="const",kappa=0.0,q=1e-3,gain_err=GE)),
      ("tracking, TRUE-gain reference",     "healthy",dict(regressor="const",kappa=1.0,q=1e-4,gain_err=GE,ref_uses_model=False)),
      ("tracking, MODEL-gain reference",    "healthy",dict(regressor="const",kappa=3.0,q=1e-3,gain_err=GE,ref_uses_model=True)),
      ("tracking, MODEL ref, offset fault", "offset", dict(regressor="const",kappa=3.0,q=1e-3,gain_err=GE,ref_uses_model=True)),
      ("tracking, exact model, offset",     "offset", dict(regressor="const",kappa=1.0,q=1e-4)),
      ("no tracking, exact model, offset",  "offset", dict(regressor="const",kappa=0.0,q=1e-3))]
for lab,f,kw in rows:
    s,t=run(f,E=1500,seed=606,return_time=True,**kw); s=s[0]
    print(f"  {lab:<36} success {s.mean():.3f}   median completion step {np.median(t[s]) if s.any() else float('nan'):>6.0f}   p90 {np.percentile(t[s],90) if s.any() else float('nan'):>5.0f}")
# tighter horizon: what happens if the task must finish within 140 steps (LIBERO-like step limits)?
print("\nsame arms with the horizon halved to 140 steps (success only):")
import sim as S
S.N=140
for lab,f,kw in rows:
    s=run(f,E=1500,seed=606,**kw)[0]; print(f"  {lab:<36} success {s.mean():.3f}")
