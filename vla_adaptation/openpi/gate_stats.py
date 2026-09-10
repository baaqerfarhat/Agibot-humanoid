"""Healthy phantom statistics for the channel gate (prereg_records/PREREG_HEALTHY_GATE.md).

Reads an --estimate-only healthy run (sev 0) and writes, per channel, the mean b_i and the
across-episode standard deviation sd_i of the per-episode mean estimate over the last 50
steps (the Sec 29.2 statistic), with a floor on sd_i. Healthy data only: nothing here has
seen a fault. Usage: python gate_stats.py healthy_phantom.json --out phantom_stats.json
"""
import argparse, json, pathlib
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("run", type=pathlib.Path); ap.add_argument("--out", type=pathlib.Path, required=True)
ap.add_argument("--last", type=int, default=50); ap.add_argument("--sd-floor", type=float, default=0.002)
a = ap.parse_args()
d = json.loads(a.run.read_text())
arm = d["arms"]["adaptive"]
per_ep = []
for tr in arm["traj"]:
    # stored either as a list of per-step dicts (with f_hat) or as a list of f_hat vectors
    F = np.array([(s["f_hat"] if isinstance(s, dict) else s) for s in tr], float)
    if len(F) == 0:
        continue
    per_ep.append(F[-a.last:].mean(axis=0))
P = np.array(per_ep)
b = P.mean(axis=0); sd = P.std(axis=0, ddof=1); sd_f = np.maximum(sd, a.sd_floor)
out = dict(source=str(a.run), episodes=int(len(P)), last_steps=a.last, sd_floor=a.sd_floor,
           b=b.tolist(), sd_raw=sd.tolist(), sd=sd_f.tolist(),
           per_episode=P.tolist(), healthy_successes=arm["successes"], n=arm["n"])
a.out.write_text(json.dumps(out, indent=1))
names = ["x", "y", "z", "rx", "ry", "rz"]
print(f"{len(P)} healthy episodes; per-episode mean estimate over the last {a.last} steps")
for i, n in enumerate(names):
    print(f"  {n:2s}: b = {b[i]:+.4f}   sd = {sd[i]:.4f}  (used {sd_f[i]:.4f})   3 sd = {3*sd_f[i]:.4f}")
print(f"wrote {a.out}")
