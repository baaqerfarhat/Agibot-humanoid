"""Apply PREREG_OPENPI_ACE_SCREEN §1's keep rule to the Phase 0.4 cells.

The rule, fixed in advance: keep a cell iff the drop from the 99.0% nominal is >= 30
percentage points AND success stays > 5%. The screen then runs on exactly ONE cell --
the survivor with the largest drop -- and if two are within 5 points, the actuation-gain
cell is taken. None of that is decided here; it was decided before the runs.
"""
from __future__ import annotations

import json
import pathlib
import sys

NOMINAL = 0.99          # the 500-episode gate-0.1 reference, not the 20-episode check
DROP_MIN = 0.30
FLOOR = 0.05
ORDER = {"gain": 0, "offset": 1, "brightness": 2}


def main(d: pathlib.Path):
    cells = []
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        if r["fault"] == "nominal":
            nom = r
            continue
        r["drop"] = NOMINAL - r["success_rate"]
        r["keep"] = r["drop"] >= DROP_MIN and r["success_rate"] > FLOOR
        cells.append(r)
    cells.sort(key=lambda r: (ORDER.get(r["fault"], 9), r["severity"]))

    print(f"nominal check (this client, 20 ep): {100*nom['success_rate']:.0f}%   "
          f"reference (500 ep): {100*NOMINAL:.0f}%\n")
    print(f"{'fault':<12} {'sev':>5} {'succ':>10} {'drop':>7}   verdict")
    for r in cells:
        why = "KEEP" if r["keep"] else ("floor-dead" if r["success_rate"] <= FLOOR
                                        else f"drop {100*r['drop']:.0f} < 30")
        print(f"{r['fault']:<12} {r['severity']:>5} {r['successes']:>3}/{r['episodes']:<3}"
              f" {100*r['success_rate']:>4.0f}% {100*r['drop']:>6.0f}   {why}")

    kept = [r for r in cells if r["keep"]]
    print()
    if not kept:
        print("KILL RULE: no cell passes. Per the prereg, Phase 0.4 reports "
              "'no usable fault cell on libero_spatial under client-side wrappers' "
              "and the ACE screen does NOT run. Severities are not re-swept.")
        print("The one permitted extension is a single screen of libero_10 (published 92.4)"
              " under the same document.")
        return
    best = max(kept, key=lambda r: r["drop"])
    tied = [r for r in kept if abs(r["drop"] - best["drop"]) <= 0.05]
    if len(tied) > 1 and any(r["fault"] == "gain" for r in tied):
        best = next(r for r in tied if r["fault"] == "gain")
        print(f"tie within 5 points among {[(r['fault'], r['severity']) for r in tied]}"
              " -> actuation-gain cell taken, per the prereg")
    print(f"SCREEN CELL: {best['fault']} @ {best['severity']} "
          f"({100*best['success_rate']:.0f}%, drop {100*best['drop']:.0f} points)")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                      else "../results/gate04"))
