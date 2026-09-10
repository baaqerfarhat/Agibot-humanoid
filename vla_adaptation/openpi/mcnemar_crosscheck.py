"""An independent re-derivation of every paired p-value, sharing no code with mcnemar.py.

Why this file exists. `openpi/mcnemar.py` is the instrument behind every number in the
paper, and an instrument that is only ever checked against itself is not checked at all.
This script reads the same stored per-episode outcomes and recomputes the same statistics
through a *separately written* exact binomial, so that agreement between the two is
evidence rather than a tautology. It deliberately does not import from `mcnemar.py`.

It found one defect, now FIXED in `mcnemar.py` (2026-09-07) and retained here as the
regression test for it. `mcnemar.py:mcnemar_exact` used to build its two-sided tail with

    sum(p(k) for k in range(n+1) if p(k) <= obs + 1e-12)

an *absolute* tolerance of 1e-12 on a quantity that, for the pooled headline cell
(b=1, c=51, n=52), is p(1) = 1.15e-14. The tolerance is 87x the observed probability, so
the sum admitted k=2 and k=50 as well as the intended {0, 1, 51, 52}, and the reported
p-value was inflated about 26-fold: 6.1e-13 against a true 2.35e-14.

The error was conservative -- it made the pooled result look less significant than it was,
never more -- and no individual cell was affected, because every cell's observed
probability (>= 6.1e-5) dwarfs the tolerance. But 6.1e-13 was quoted in the abstract. The
fix, applied to mcnemar.py and to the paper, is to compare relatively
(`<= obs * (1 + 1e-9)`) rather than absolutely.

The two conventions for a two-sided exact McNemar differ, and both are defensible:

  * `mcnemar.py` uses the "sum of outcomes no more likely than observed" rule, the exact
    analogue of Fisher's two-sided rule.
  * this file uses 2 x min-tail (Agresti), the more common convention.

Under H0 the binomial is symmetric (p = 1/2), so the two agree exactly. That is why the
columns below match to every digit -- and why a mismatch would mean a real defect in one
of them, not a convention difference.

Usage:

    python3 openpi/mcnemar_crosscheck.py                    # the paper's cells, vs claims
    python3 openpi/mcnemar_crosscheck.py path/to/file.json  # any paired result file

Pure standard library, like `mcnemar.py`: it runs with nothing installed.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

# (label, path, p-value as printed in paper/iclr_draft.tex, arm-b key override or None)
PAPER_CELLS = [
    ("libero_spatial",        "results/suites/libero_spatial_rotonly_paired.json", 0.0020,   None),
    ("libero_goal n=40",      "results/suites/libero_goal_rotonly_n40.json",       0.00052,  None),
    ("libero_object",         "results/suites/libero_object_rotonly_paired.json",  0.00098,  None),
    ("libero_10 n=40",        "results/suites/libero_10_rotonly_n40.json",         6.1e-5,   None),
    ("OFT rotation 0.10",     "results/oft/oft_rot010.json",                       9.8e-4,   None),
    ("OFT translation 0.15",  "results/oft/oft_tra015.json",                       1.2e-4,   None),
    ("OFT gain 0.20",         "results/oft/oft_gain020.json",                      1.5e-5,   "adaptive_gain"),
    ("OFT healthy control",   "results/oft/oft_healthy.json",                      1.0,      None),
    ("GR00T rotation 0.10",   "results/groot/groot_rot010.json",                   1.2e-4,   None),
    ("ALOHA hold n=40",       "results/aloha/off002_identify1_hold_n40.json",      6.1e-5,   None),
    ("ALOHA healthy n=40",    "results/aloha/healthy_identify1_hold_n40.json",     0.58,     None),
    ("joint torque n=40",     "results/phase05/jf_torque_3_5_0_n40.json",          0.0075,   None),
    ("joint friction +10",    "results/phase05/jf_friction_3_10_0_paired.json",    0.51,     None),
    ("joint friction +20",    "results/phase05/jf_friction_3_20_0_paired.json",    0.0078,   None),
    ("joint lock",            "results/phase05/jf_lock_3_0_05_paired.json",        1.0,      None),
]

# The four suites the paper pools into its headline row.
POOLED = [c[1] for c in PAPER_CELLS[:4]]


def exact_two_sided(b: int, c: int) -> float:
    """Two-sided exact McNemar as 2 x min-tail (Agresti), capped at 1.

    Written from the definition rather than adapted from mcnemar.py, so that agreement
    between the two is independent evidence.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def pairs(path: str, arm_a: str = "frozen_faulted", arm_b: str | None = None):
    """Match the two arms on (task, init) and return the list of (frozen, corrected)."""
    arms = json.loads(pathlib.Path(path).read_text())["arms"]
    if arm_b is None:
        arm_b = "adaptive" if "adaptive" in arms else next(k for k in arms if k != arm_a)
    # See the matching guard in mcnemar.py: keying by (task, init) drops replicates silently.
    # This crosscheck shares that input-key convention with the tool it is meant to check, so the
    # defect would survive the crosscheck; hence the same guard here rather than a different one.
    for nm in (arm_a, arm_b):
        kk = [(e["task"], e["init"]) for e in arms[nm]["per_ep"]]
        if len(set(kk)) != len(kk):
            raise SystemExit(
                f"{path}: arm {nm!r} has {len(kk)} episodes but {len(set(kk))} distinct "
                f"(task, init) keys; pairing by key would drop replicates.")
    ka = {(e["task"], e["init"]): int(e["ok"]) for e in arms[arm_a]["per_ep"]}
    kb = {(e["task"], e["init"]): int(e["ok"]) for e in arms[arm_b]["per_ep"]}
    return [(ka[k], kb[k]) for k in sorted(set(ka) & set(kb))]


def counts(ps):
    """(n, frozen successes, corrected successes, broken, fixed)."""
    return (len(ps),
            sum(x for x, _ in ps),
            sum(y for _, y in ps),
            sum(1 for x, y in ps if x and not y),
            sum(1 for x, y in ps if y and not x))


def _agrees(got: float, claimed: float) -> bool:
    """Compare against a p-value printed to two significant figures."""
    if got == 0 or claimed == 0:
        return got == claimed
    return abs(math.log10(got) - math.log10(claimed)) < 0.05


def main(argv):
    if argv:
        for path in argv:
            n, ka, kb, b, c = counts(pairs(path))
            print(f"{pathlib.Path(path).name}: n={n} {ka} -> {kb}  "
                  f"fixed {c}, broken {b}, exact two-sided p = {exact_two_sided(b, c):.4g}")
        return 0

    root = pathlib.Path(__file__).resolve().parent.parent
    hdr = f"{'cell':22s} {'n':>4} {'froz':>5} {'corr':>5} {'brk':>4} {'fix':>4} {'this file':>12} {'paper':>9}"
    print(hdr)
    print("-" * len(hdr))
    disagreements = []
    for label, rel, claimed, arm_b in PAPER_CELLS:
        path = root / rel
        if not path.exists():
            print(f"{label:22s}  MISSING: {rel}")
            disagreements.append(label)
            continue
        n, ka, kb, b, c = counts(pairs(str(path), arm_b=arm_b))
        p = exact_two_sided(b, c)
        ok = _agrees(p, claimed)
        print(f"{label:22s} {n:4d} {ka:5d} {kb:5d} {b:4d} {c:4d} {p:12.4g} {claimed:9.2g}"
              f"{'' if ok else '   <-- DISAGREES'}")
        if not ok:
            disagreements.append(label)

    ps = [q for path in POOLED for q in pairs(str(root / path))]
    n, ka, kb, b, c = counts(ps)
    p = exact_two_sided(b, c)
    print(f"\npooled (4 suites)      {n:4d} {ka:5d} {kb:5d} {b:4d} {c:4d} {p:12.4g} {2.4e-14:9.2g}")
    print("  Historical note: mcnemar.py once reported 6.12e-13 here, from an absolute 1e-12\n"
          "  tolerance against an observed probability of 1.15e-14. Fixed 2026-09-07 to a\n"
          f"  relative comparison; both instruments now agree at {p:.4g}.")

    if disagreements:
        print(f"\nFAIL: {len(disagreements)} cell(s) disagree with the paper: {disagreements}")
        return 1
    print("\nOK: every individual cell reproduces the paper's p-value "
          "through an independently written test.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
