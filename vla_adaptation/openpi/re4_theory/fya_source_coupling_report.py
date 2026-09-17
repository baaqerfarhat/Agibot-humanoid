#!/usr/bin/env python3
"""Full-episode source-coupling report (closed-loop EXPERIMENT_PLAN.md §3 item 1 and §4 checks).

Compares complete archived telemetry episodes between arms of the same (task, init, seed) key: lengths, raw actions,
positions, joint positions, orientation (SO(3) angle between recorded quaternions), success, and the control
acknowledgement (sampler schedule) when present. Reports exact-equality counts, first divergence step, maximum gaps
and every mismatching key. Read-only.

  --src DIR --a healthy_off --b healthy_off_dup            duplicate reproducibility (registered expectation: exact)
  --src DIR --a healthy_off --b healthy_nt --prefix 30     prefix coupling before any correction (exact expected)
"""
import argparse, gzip, json, pathlib
import numpy as np


def read_arm(path):
    eps = {}; acks = {}
    with gzip.open(path, "rt") if str(path).endswith(".gz") else open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("type") == "step" and r.get("phase") == "rollout":
                eps.setdefault((r["task"], r["init"]), []).append(r)
    for k in eps:
        eps[k].sort(key=lambda s: s["t"])
    return eps


def qangle(qa, qb):
    qa, qb = np.asarray(qa, float), np.asarray(qb, float); qa /= np.linalg.norm(qa); qb /= np.linalg.norm(qb)
    d = min(float(np.linalg.norm(qa - qb)), float(np.linalg.norm(qa + qb))); return float(4 * np.arcsin(min(1.0, d / 2)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", type=pathlib.Path, required=True); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--prefix", type=int, default=None, help="compare only the first N policy steps (coupling before correction); default: whole episodes")
    ap.add_argument("--out", type=pathlib.Path, required=True); a = ap.parse_args()
    A = read_arm(next(p for p in (a.src / f"{a.a}_telemetry.jsonl.gz", a.src / f"{a.a}_telemetry.jsonl") if p.exists()))
    B = read_arm(next(p for p in (a.src / f"{a.b}_telemetry.jsonl.gz", a.src / f"{a.b}_telemetry.jsonl") if p.exists()))
    resA = json.load(open(a.src / f"{a.a}.json")); resB = json.load(open(a.src / f"{a.b}.json"))
    okA = {(e["task"], e["init"]): e["ok"] for e in next(iter(resA["arms"].values()))["per_ep"]}; okB = {(e["task"], e["init"]): e["ok"] for e in next(iter(resB["arms"].values()))["per_ep"]}
    seedA = {(e["task"], e["init"]): e.get("sampler_seed") for e in next(iter(resA["arms"].values()))["per_ep"]}; seedB = {(e["task"], e["init"]): e.get("sampler_seed") for e in next(iter(resB["arms"].values()))["per_ep"]}
    rows = []
    for key in sorted(set(A) | set(B)):
        if key not in A or key not in B:
            rows.append(dict(task=key[0], init=key[1], status="missing in one arm")); continue
        x, y = A[key], B[key]; n = min(len(x), len(y)) if a.prefix is None else min(a.prefix, len(x), len(y))
        ra = np.array([s["raw_action"] for s in x[:n]]); rb = np.array([s["raw_action"] for s in y[:n]])
        d_raw = np.abs(ra - rb).max(1); first = int(np.argmax(d_raw > 0)) if np.any(d_raw > 0) else None
        pos = max(float(np.linalg.norm(np.array(s["position"]) - np.array(t["position"]))) for s, t in zip(x[:n], y[:n]))
        jnt = max(float(np.max(np.abs(np.array(s["joint_position"]) - np.array(t["joint_position"])))) for s, t in zip(x[:n], y[:n]))
        ang = max(qangle(s["quaternion"], t["quaternion"]) for s, t in zip(x[:n], y[:n]))
        exact = bool(first is None and pos == 0.0 and jnt == 0.0 and (a.prefix is not None or len(x) == len(y)))
        rows.append(dict(task=key[0], init=key[1], len_a=len(x), len_b=len(y), compared_steps=n, exact=exact, first_raw_divergence_step=first,
                         max_raw_action_gap=float(d_raw.max()), max_position_gap_m=pos, max_joint_gap_rad=jnt, max_angle_gap_rad=ang,
                         success_a=okA.get(key), success_b=okB.get(key), sampler_seed_a=seedA.get(key), sampler_seed_b=seedB.get(key)))
    n_exact = sum(r.get("exact", False) for r in rows)
    out = dict(src=str(a.src), arm_a=a.a, arm_b=a.b, prefix=a.prefix, n_keys=len(rows), n_exact=n_exact,
               mismatching_keys=[dict(task=r["task"], init=r["init"], first=r.get("first_raw_divergence_step"), max_raw=r.get("max_raw_action_gap"), max_pos_m=r.get("max_position_gap_m"), len=(r.get("len_a"), r.get("len_b"))) for r in rows if not r.get("exact", False)],
               max_raw_action_gap=max((r.get("max_raw_action_gap", 0.0) for r in rows), default=0.0), max_position_gap_m=max((r.get("max_position_gap_m", 0.0) for r in rows), default=0.0),
               success_a=f"{sum(bool(v) for v in okA.values())}/{len(okA)}", success_b=f"{sum(bool(v) for v in okB.values())}/{len(okB)}", rows=rows)
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
