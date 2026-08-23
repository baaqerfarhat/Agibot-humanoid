"""Score the paired ACE screen: does the estimator separate sites once the noise is gone?

Same primary as the pre-registered §4 -- one-way ANOVA over sites x draws plus eta^2 -- but
run on PAIRED ACE values, where each draw's number is a mean of deterministic -1/0/+1
differences against a shared baseline rather than an unpaired 5-episode success rate.
"""
from __future__ import annotations

import argparse, json, pathlib
import numpy as np
from ace_analyze import f_sf, TIER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=pathlib.Path)
    a = ap.parse_args()
    d = json.loads(a.path.read_text())
    base = d["baseline"]
    print(f"c_rel = {d['c_rel']}   paired, RNG pinned   baseline {sum(base)}/{len(base)}"
          f"   {d['draws']} draws x {d['episodes']} episodes\n")

    by = {}
    for b in d["blocks"]:
        by.setdefault(b["site"], []).append(b["ace"])
    sites = [s for s in TIER if s in by]
    groups = [np.array(by[s], float) for s in sites]
    complete = all(len(g) == d["draws"] for g in groups) and len(sites) == len(TIER)

    print(f"{'site':<38} {'tier':<14} {'ACE_hat':>9} {'sd':>7} {'n':>3}")
    rows = [(s, TIER[s], g.mean(), g.std(ddof=1) if len(g) > 1 else np.nan, len(g))
            for s, g in zip(sites, groups)]
    for s, t, m, sd, n in sorted(rows, key=lambda r: r[2]):
        print(f"{s:<38} {t:<14} {m:>+9.3f} {sd:>7.3f} {n:>3}")

    if len(groups) < 2 or min(len(g) for g in groups) < 2:
        print("\n(insufficient data yet)")
        return
    allv = np.concatenate(groups); grand = allv.mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    d1, d2 = len(groups) - 1, len(allv) - len(groups)
    eta2 = ssb / (ssb + ssw) if (ssb + ssw) > 0 else 0.0
    F = (ssb / d1) / (ssw / d2) if ssw > 0 else float("inf")
    p = f_sf(F, d1, d2)
    print(f"\nANOVA F({d1},{d2}) = {F:.3f}, p = {p:.4f}, eta^2 = {eta2:.3f}"
          f"{'' if complete else '   [PARTIAL]'}")
    print(f"PRIMARY {'PASSES' if (p < 0.05 and eta2 >= 0.25) else 'FAILS'}"
          f" (needs p < 0.05 AND eta^2 >= 0.25)")

    neg = sum(1 for r in rows if r[2] < 0)
    print(f"\nP3 check: ACE_hat < 0 at {neg}/{len(rows)} sites (theory predicted >= 7)")
    tiers = {}
    for s, t, m, sd, n in rows:
        tiers.setdefault(t, []).append(abs(m))
    order = sorted(tiers, key=lambda t: -np.mean(tiers[t]))
    print(f"P2 check: tier order by mean|ACE_hat|: {' > '.join(order)}")
    print(f"top |ACE_hat|: {max(rows, key=lambda r: abs(r[2]))[0]}")


if __name__ == "__main__":
    main()
