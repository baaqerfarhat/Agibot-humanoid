"""Physical pre-test analysis. Replays are open loop with identical commands, so the fault-free frozen
arm A0a IS the healthy trajectory; each arm's pose deviation from it is the physical execution error."""
import json, sys, gzip, pathlib, numpy as np
import pathlib as _p; sys.path.insert(0, str(_p.Path(__file__).resolve().parents[3] / "openpi")); from so3 import rot_delta
OUT = np.array([.05,.05,.05,.5,.5,.5])
def load(p):
    p = pathlib.Path(p)
    op = gzip.open if p.suffix == ".gz" else open
    rows = [json.loads(l) for l in op(p, "rt")]
    eps = {}
    for r in rows[1:]:
        if r.get("type") == "step" and r.get("phase") != "warmup": eps.setdefault(r["episode"], []).append(r)
    for e in eps: eps[e].sort(key=lambda r: r["t"])
    return eps
def dev(ref, arm):   # per-step 6-D deviation in normalised units
    out = []
    for a, b in zip(ref, arm):
        dp = (np.array(b["position"]) - np.array(a["position"])) / OUT[:3]
        dq = rot_delta(np.array(a["quaternion"]), np.array(b["quaternion"])) / OUT[3:]
        out.append(np.r_[dp, dq])
    return np.array(out)
root = pathlib.Path(sys.argv[1]); arms = sys.argv[2].split(",")
_r = root / "A0a" / "telemetry.jsonl"
ref = load(_r if _r.exists() else _r.with_suffix(".jsonl.gz"))
np.set_printoptions(precision=3, suppress=True)
print(f"{'arm':<5} {'late |dev| (last 20 steps) x y z rx ry rz':<52} {'max |dev|':<44} f_hat/f (rot, late)")
for arm in arms:
    p = root / arm / "telemetry.jsonl"
    p = p if p.exists() else p.with_suffix(".jsonl.gz")
    if not p.exists(): print(f"{arm:<5} missing"); continue
    eps = load(p); L, Mx, S = [], [], []
    for e, rows in eps.items():
        D = np.abs(dev(ref[e], rows)); L.append(D[-20:].mean(0)); Mx.append(D.max(0))
        fh = np.array([r["f_hat"][:6] for r in rows[-20:]]).mean(0); ft = np.array(rows[-1]["f_true"][:6])
        S.append(np.where(np.abs(ft[3:]) > 0, fh[3:] / np.where(ft[3:] == 0, 1, ft[3:]), fh[3:]))
    print(f"{arm:<5} {np.mean(L,0)!s:<52} {np.mean(Mx,0)!s:<44} {np.mean(S,0)}")
