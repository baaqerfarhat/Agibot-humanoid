"""Paired analysis of a frozen-vs-adaptive result file.

The review of 2026-09-01 is right that Fisher's exact test is the wrong tool here. The two
arms are not independent samples: they run the SAME (task, init) episodes, with the same
policy and the same seeds, and differ only in whether the correction is applied. That is a
matched-pairs design, and the correct tests condition on the pairs.

Fisher throws the pairing away. It asks "are these two proportions different", when the
sharper question the design supports is "of the episodes where the two arms disagreed, how
lopsided is the disagreement". Discarding concordant pairs usually makes the test MORE
powerful, so this is not a concession -- it is the analysis the experiment earned.

Exact McNemar (binomial on the discordant pairs) is reported alongside a paired permutation
test, which makes no distributional assumption at all.
"""
import argparse, json, math, pathlib
from itertools import product


def _binom_p(k, n, p=0.5):
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar. b = frozen-only wins, c = adaptive-only wins."""
    n = b + c
    if n == 0:
        return 1.0
    obs = _binom_p(b, n)
    # Two-sided by the "sum of outcomes no more likely than observed" convention, which is
    # the exact analogue of Fisher's two-sided rule rather than a doubled one-tail.
    #
    # The comparison must be RELATIVE. An absolute tolerance is wrong here because these
    # probabilities span many orders of magnitude: at the pooled headline cell (b=1, c=51)
    # the observed probability is 1.15e-14, so an absolute `+ 1e-12` was 87x larger than the
    # quantity it toleranced and admitted k=2 and k=50 as well as the intended {0,1,51,52}.
    # That inflated the pooled p-value from 2.35e-14 to 6.12e-13 -- conservative, so no claim
    # was overstated, but wrong, and it is the number the abstract quotes. Individual cells
    # were unaffected because their observed probabilities (>= 6.1e-5) dwarf 1e-12.
    return min(1.0, sum(_binom_p(k, n) for k in range(n + 1)
                        if _binom_p(k, n) <= obs * (1.0 + 1e-9)))


def permutation_p(pairs, iters=200000, seed=0):
    """Paired permutation: flip each discordant pair's label with probability 1/2."""
    import random
    rng = random.Random(seed)
    disc = [(x, y) for x, y in pairs if x != y]
    if not disc:
        return 1.0
    obs = abs(sum(y - x for x, y in disc))
    n, hit = len(disc), 0
    for _ in range(iters):
        s = sum(rng.choice((-1, 1)) for _ in range(n))
        if abs(s) >= obs:
            hit += 1
    return (hit + 1) / (iters + 1)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(path, a="frozen_faulted", b="adaptive"):
    d = json.loads(pathlib.Path(path).read_text())
    arms = d["arms"]
    if a not in arms or b not in arms:
        raise SystemExit(f"{path}: arms are {list(arms)}")
    pa, pb = arms[a].get("per_ep"), arms[b].get("per_ep")
    if not pa or not pb:
        return None, arms[a]["successes"], arms[b]["successes"], arms[a]["n"]
    # Keying by (task, init) silently DROPS replicates: if an arm contains two episodes at the
    # same (task, init) -- which pooling two blocks of a single cell would produce -- the dict
    # comprehension keeps only the last and the pair count silently falls while `successes` and
    # `n` still report the full total. No stored result in this project is affected (315 arms
    # checked, all one episode per key), but the failure is silent, so it is now an error.
    for nm, rows in ((a, pa), (b, pb)):
        keys = [(e["task"], e["init"]) for e in rows]
        if len(set(keys)) != len(keys):
            raise SystemExit(
                f"{path}: arm {nm!r} has {len(rows)} episodes but only {len(set(keys))} distinct "
                f"(task, init) keys. Pairing by key would drop replicates. Score replicate blocks "
                f"separately, or pair positionally with an explicit replicate index.")
    ka = {(e["task"], e["init"]): e["ok"] for e in pa}
    kb = {(e["task"], e["init"]): e["ok"] for e in pb}
    keys = sorted(set(ka) & set(kb))
    return ([(int(ka[k]), int(kb[k])) for k in keys],
            arms[a]["successes"], arms[b]["successes"], arms[a]["n"])


def report(path, a, b):
    pairs, ka, kb, n = load(path, a, b)
    name = pathlib.Path(path).name
    print(f"\n=== {name} ===")
    if pairs is None:
        print(f"  {ka}/{n} -> {kb}/{n}   (+{100*(kb-ka)/n:.0f} points)")
        print("  NO per-episode record -- this file predates per_ep logging, so the pairing")
        print("  cannot be recovered and only an unpaired test is possible. Rerun to fix.")
        return
    both = sum(1 for x, y in pairs if x and y)
    nei = sum(1 for x, y in pairs if not x and not y)
    bb = sum(1 for x, y in pairs if x and not y)      # frozen only
    cc = sum(1 for x, y in pairs if y and not x)      # adaptive only
    lo_a, hi_a = wilson(ka, n)
    lo_b, hi_b = wilson(kb, n)
    print(f"  frozen   {ka}/{n} = {100*ka/n:.0f}%  [{100*lo_a:.0f}, {100*hi_a:.0f}]")
    print(f"  adaptive {kb}/{n} = {100*kb/n:.0f}%  [{100*lo_b:.0f}, {100*hi_b:.0f}]")
    print(f"  pairs: both {both}, neither {nei}, frozen-only {bb}, adaptive-only {cc}")
    print(f"  exact McNemar p = {mcnemar_exact(bb, cc):.5g}")
    print(f"  paired permutation p = {permutation_p(pairs):.5g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--a", default="frozen_faulted")
    ap.add_argument("--b", default="adaptive")
    g = ap.parse_args()
    for f in g.files:
        report(f, g.a, g.b)
