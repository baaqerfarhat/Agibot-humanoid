#!/usr/bin/env python3
"""re4 theory Part 8.2: score two seed-pinned runs as a paired decision cell.

Pairs the two runs' episodes by (task, init), counts the paired outcome table on the arm
requested for each run (default: adaptive vs adaptive), reports the gap and the exact McNemar
p-value, and, from the two telemetry logs, the step at which the executed commands first differ
by more than --tol (the pinning note in PREREG_RE4T_8_2_DECISION_CELLS.md: with one sampler key
per policy call the two runs draw identical actions until their trajectories diverge).
Also reports the frozen arms' agreement as a determinism check (same key, no correction).
"""
import argparse, csv, json, pathlib, re, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mcnemar import mcnemar_exact


def open_log(path):
    """Open a JSONL log that may be stored gzipped (large telemetry files are kept as .jsonl.gz)."""
    import gzip
    path = pathlib.Path(path)
    if path.exists():
        return open(path)
    gz = path.with_name(path.name + ".gz")
    if gz.exists():
        return gzip.open(gz, "rt")
    raise FileNotFoundError(path)


def outcomes(run_dir, arm):
    if (run_dir / "episodes.csv").exists():
        rows = list(csv.DictReader(open(run_dir / "episodes.csv")))
        return {(r["task"], r["init"]): int(r["outcome"]) for r in rows if r["arm"] == arm}
    # a run that crashed before its record was written: read the runner's per-episode lines
    pat = re.compile(r"\[(\w+)\] task (\d+) init (\d+): success=(True|False)")
    out = {}
    for line in open(run_dir / "run.log"):
        m = pat.search(line)
        if m and m.group(1) == arm:
            out[(m.group(2), m.group(3))] = int(m.group(4) == "True")
    return out


def commands(run_dir, arm):
    out = {}
    for line in open_log(run_dir / "telemetry.jsonl"):
        d = json.loads(line)
        if d.get("type") != "step" or d.get("phase") != "rollout" or d.get("arm") != arm:
            continue
        out.setdefault((str(d["task"]), str(d["init"])), {})[int(d["t"])] = np.asarray(d["command"], float)
    return out


def first_divergence(ca, cb, tol):
    res = {}
    for k in ca.keys() & cb.keys():
        a, b = ca[k], cb[k]; div = None
        for t in sorted(a.keys() & b.keys()):
            if np.max(np.abs(a[t] - b[t])) > tol:
                div = t; break
        res[k] = dict(first_divergence_step=div, common_steps=len(a.keys() & b.keys()), len_a=len(a), len_b=len(b))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=pathlib.Path); ap.add_argument("run_b", type=pathlib.Path)
    ap.add_argument("--arm-a", default="adaptive"); ap.add_argument("--arm-b", default="adaptive")
    ap.add_argument("--tol", type=float, default=1e-6); ap.add_argument("--gap-min", type=int, default=None,
                    help="registered minimum gap (a - b) for the prediction")
    ap.add_argument("--out", type=pathlib.Path); a = ap.parse_args()
    oa, ob = outcomes(a.run_a, a.arm_a), outcomes(a.run_b, a.arm_b); keys = sorted(oa.keys() & ob.keys())
    both = sum(oa[k] and ob[k] for k in keys); a_only = sum(oa[k] and not ob[k] for k in keys)
    b_only = sum(ob[k] and not oa[k] for k in keys); neither = len(keys) - both - a_only - b_only
    p = mcnemar_exact(a_only, b_only)
    div = first_divergence(commands(a.run_a, a.arm_a), commands(a.run_b, a.arm_b), a.tol)
    steps = [v["first_divergence_step"] for v in div.values()]
    fz = first_divergence(commands(a.run_a, "frozen_faulted"), commands(a.run_b, "frozen_faulted"), a.tol) \
        if (a.run_a / "telemetry.jsonl").exists() or (a.run_a / "telemetry.jsonl.gz").exists() else {}
    fo_a, fo_b = outcomes(a.run_a, "frozen_faulted"), outcomes(a.run_b, "frozen_faulted")
    out = dict(run_a=str(a.run_a), run_b=str(a.run_b), arm_a=a.arm_a, arm_b=a.arm_b, n_pairs=len(keys),
               a_successes=sum(oa[k] for k in keys), b_successes=sum(ob[k] for k in keys),
               table=dict(both=both, a_only=a_only, b_only=b_only, neither=neither), gap=a_only - b_only, mcnemar_exact_p=p,
               divergence=dict(n_with_telemetry=len(div), never_diverged=sum(s is None for s in steps),
                               median_step=float(np.median([s for s in steps if s is not None])) if any(s is not None for s in steps) else None,
                               steps=[s for s in steps]),
               frozen_check=dict(n_pairs=len(fo_a.keys() & fo_b.keys()),
                                 outcome_agreement=float(np.mean([fo_a[k] == fo_b[k] for k in fo_a.keys() & fo_b.keys()])) if fo_a.keys() & fo_b.keys() else None,
                                 never_diverged=sum(v["first_divergence_step"] is None for v in fz.values()), n_with_telemetry=len(fz),
                                 divergence_steps=sorted(v["first_divergence_step"] for v in fz.values() if v["first_divergence_step"] is not None)))
    if a.gap_min is not None:
        out["prediction"] = dict(gap_ge=a.gap_min, gap_holds=bool(out["gap"] >= a.gap_min), p_lt_0_05=bool(p < 0.05), passes=bool(out["gap"] >= a.gap_min and p < 0.05))
    js = json.dumps(out, indent=1)
    if a.out:
        a.out.write_text(js)
    print(js)


if __name__ == "__main__":
    main()
