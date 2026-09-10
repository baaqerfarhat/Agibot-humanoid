"""Actual end-effector displacement under each joint torque bias, from telemetry.

Reads `measured` (the achieved 6-vector: 3 translation, 3 rotation) from the estimate-only
probe telemetry. This is the quantity PREREG_JOINT_MAP.md prediction 2 needs. Computing the
split from the ESTIMATE f_hat instead is wrong and guaranteed uninformative, because Sec 29.2
already established joint faults are identified on translation and nowhere else -- that
mistake was made and corrected in the prereg amendment.

The registered prediction needs the translation/rotation split of the INDUCED DISPLACEMENT,
not of the estimate. `measured` is the 6-vector of achieved motion (3 translation, 3 rotation)
per step. Compare the faulted run against the healthy baseline over the same steps.
"""
import json, pathlib, math

import sys
d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results/jointmap")


def mean_measured(path, limit=200):
    tr = [0.0, 0.0, 0.0]; ro = [0.0, 0.0, 0.0]; n = 0
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") != "step":
                continue
            m = rec.get("measured")
            if not m or len(m) < 6:
                continue
            for i in range(3):
                tr[i] += m[i]; ro[i] += m[i + 3]
            n += 1
            if n >= limit:
                break
    if not n:
        return None
    return [x / n for x in tr], [x / n for x in ro], n


print(f"{'joint':>5} {'|trans|':>10} {'|rot|':>10} {'trans share':>12} {'steps':>7}")
rows = []
for j in range(7):
    p = d / f"probe_torque_{j}_tel.jsonl"
    if not p.exists():
        continue
    got = mean_measured(p)
    if not got:
        print(f"{j:5d}   (no measured records)"); continue
    tr, ro, n = got
    t = math.sqrt(sum(x * x for x in tr)); r = math.sqrt(sum(x * x for x in ro))
    tot = math.hypot(t, r)
    sh = t / tot if tot else 0.0
    rows.append((j, t, r, sh))
    print(f"{j:5d} {t:10.5f} {r:10.5f} {sh:11.3f} {n:7d}")
if rows:
    print(f"\nordering by translation share, best first: "
          f"{[j for j, *_, sh in sorted(rows, key=lambda x: -x[3])]}")
    print(f"ordering by total displacement, largest first: "
          f"{[j for j, t, r, _ in sorted(rows, key=lambda x: -math.hypot(x[1], x[2]))]}")
