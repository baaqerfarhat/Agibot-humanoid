"""§4 primary endpoint for the ACE screen: does ACE_hat separate sites beyond draw noise?

Primary passes iff p < 0.05 AND eta^2 >= 0.25, both required (§4). Also reports the three
advance predictions of §5 so they are scored as stated rather than after the fact.

No scipy in reach here, so the F tail is computed from the regularised incomplete beta by
its continued fraction -- standard Lentz, checked against known values in --self-test.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

TIER = {"action_out_proj/bias": "interface", "action_out_proj/kernel": "interface",
        "action_in_proj/kernel": "interface", "time_mlp_out/kernel": "interface",
        "expert/mlp_1/linear/L0": "action expert", "expert/mlp_1/linear/L8": "action expert",
        "expert/mlp_1/linear/L17": "action expert",
        "llm/mlp/linear/L0": "VLM trunk", "llm/mlp/linear/L17": "VLM trunk",
        "img/MlpBlock_0/Dense_1/kernel/B26": "vision"}


def _betacf(a, b, x, itmax=300, eps=3e-12):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1e-30 if abs(d) < 1e-30 else d
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def f_sf(f, d1, d2):
    """P(F_{d1,d2} > f)."""
    if f <= 0:
        return 1.0
    return betainc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


def analyse(path: pathlib.Path):
    d = json.loads(path.read_text())
    blocks = d["blocks"]
    base = [b for b in blocks if b["site"] is None]
    if not base:
        raise SystemExit("no baseline block")
    M_base = base[0]["success_rate"]

    by_site = {}
    for b in blocks:
        if b["site"] is None:
            continue
        by_site.setdefault(b["site"], []).append(b["success_rate"])
    sites = [s for s in TIER if s in by_site]
    groups = [np.array(by_site[s], float) for s in sites]
    counts = {s: len(by_site[s]) for s in sites}
    complete = all(n == d["n_draws"] for n in counts.values()) and len(sites) == len(TIER)

    print(f"baseline (frozen faulted): {100*M_base:.0f}%   "
          f"sites {len(sites)}/{len(TIER)}   draws {sorted(set(counts.values()))}"
          f"{'' if complete else '   [PARTIAL -- not the primary]'}\n")

    print(f"{'site':<38} {'tier':<14} {'mean M':>7} {'ACE_hat':>9} {'sd':>7}")
    rows = []
    for s, g in zip(sites, groups):
        ace = g.mean() - M_base
        rows.append((s, TIER[s], g.mean(), ace, g.std(ddof=1) if len(g) > 1 else float("nan")))
    for s, t, m, a, sd in sorted(rows, key=lambda r: -abs(r[3])):
        print(f"{s:<38} {t:<14} {m:>7.3f} {a:>+9.3f} {sd:>7.3f}")

    # one-way ANOVA over sites
    if len(groups) < 2 or min(len(g) for g in groups) < 2:
        print("\n(too few sites/draws so far for the ANOVA -- rerun when the screen finishes)")
        return
    allv = np.concatenate(groups)
    grand = allv.mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    sst = ssb + ssw
    d1, d2 = len(groups) - 1, len(allv) - len(groups)
    eta2 = ssb / sst if sst > 0 else 0.0
    F = (ssb / d1) / (ssw / d2) if ssw > 0 else float("inf")
    p = f_sf(F, d1, d2)
    print(f"\nANOVA over {len(groups)} sites x {len(groups[0])} draws: "
          f"F({d1},{d2}) = {F:.3f}, p = {p:.4f}, eta^2 = {eta2:.3f}")
    verdict = "PASSES" if (p < 0.05 and eta2 >= 0.25) else "FAILS"
    print(f"PRIMARY {verdict} (needs p < 0.05 AND eta^2 >= 0.25)")
    if not complete:
        print("  ^ partial data: indicative only, not the pre-registered endpoint")

    # §5 predictions, scored as stated
    print("\n§5 predictions:")
    tiers = {}
    for s, t, m, a, sd in rows:
        tiers.setdefault(t, []).append(abs(a))
    order = sorted(tiers, key=lambda t: -np.mean(tiers[t]))
    print(f"  P2 tier order by mean|ACE_hat|: {' > '.join(order)}"
          f"   (predicted interface > action expert > VLM trunk)")
    neg = sum(1 for r in rows if r[3] < 0)
    print(f"  P3 ACE_hat < 0 at {neg}/{len(rows)} sites (predicted >= 7)")
    top = max(rows, key=lambda r: abs(r[3]))[0]
    print(f"  P4 top |ACE_hat| site: {top}"
          f"   (predicted NOT action_out_proj/bias)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("screen", nargs="?", type=pathlib.Path,
                    default=pathlib.Path("../results/ace_screen/screen.json"))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        # known values: P(F_{9,70} > 2.0) ~ 0.0527 ; P(F_{1,1} > 1) = 0.5
        print("f_sf(2.0, 9, 70) =", round(f_sf(2.0, 9, 70), 4), "(expect ~0.0527)")
        print("f_sf(1.0, 1, 1)  =", round(f_sf(1.0, 1, 1), 4), "(expect 0.5)")
        print("betainc(0.5,0.5,0.5) =", round(betainc(0.5, 0.5, 0.5), 4), "(expect 0.5)")
    else:
        analyse(a.screen)
