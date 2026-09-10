#!/usr/bin/env python3
"""Recount the eight selected descriptor cells at the pinned review commit.

Uses only Git and the standard library. Reads no current result files, launches
no simulators, and performs no remote operation. The cluster sign-flip check is
explicitly post-hoc sensitivity analysis, not a new confirmatory claim.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess


SOURCE_COMMIT = "cda6c92357e884ef6e2b1195abda673fc18853ac"
FILES = (
    "results/sweep/spatial_j0.json",
    "results/sweep/spatial_j1.json",
    "results/sweep/spatial_j2.json",
    "results/sweep/spatial_j3.json",
    "results/sweep/spatial_j4.json",
    "results/descriptor/g14_spatial_j5.json",
    "results/sweep/spatial_j6.json",
    "results/descriptor/g11_object_j3_unclipped.json",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def probability(value):
    return {"numerator": value.numerator, "denominator": value.denominator,
            "decimal": float(value)}


def mcnemar(fixed, broken):
    from math import comb
    n = fixed + broken
    return probability(min(Fraction(1), Fraction(
        2 * sum(comb(n, k) for k in range(min(fixed, broken) + 1)), 2**n)))


def outcomes(arm, label):
    episodes = arm["per_ep"]
    require(arm["n"] == len(episodes) == 20, label + ": expected 20 episodes")
    paired = {}
    for episode in episodes:
        key = (episode["task"], episode["init"])
        require(key not in paired, label + ": duplicate task/init key")
        require(type(episode["ok"]) is bool, label + ": nonboolean outcome")
        paired[key] = episode["ok"]
    require(sum(paired.values()) == arm["successes"], label + ": success mismatch")
    return paired


def summarize(cells):
    names = ("paired_rows", "frozen_successes", "adaptive_successes", "fixed", "broken")
    result = {name: sum(cell[name] for cell in cells) for name in names}
    result["cell_count"] = len(cells)
    result["net_success_difference"] = result["adaptive_successes"] - result["frozen_successes"]
    return result


def recount(root):
    require(git(root, "rev-parse", SOURCE_COMMIT + "^{commit}").decode().strip()
            == SOURCE_COMMIT, "Pinned source commit does not resolve")
    cells, inputs = [], []
    clusters = defaultdict(int)
    for path in FILES:
        blob = git(root, "rev-parse", SOURCE_COMMIT + ":" + path).decode().strip()
        raw = git(root, "show", SOURCE_COMMIT + ":" + path)
        data = json.loads(raw)
        require(data.get("complete") is True, path + ": incomplete cell")
        require(set(data["arms"]) == {"frozen_faulted", "adaptive"}, path + ": unexpected arms")
        require(data.get("policy_rng_pinned") is False, path + ": changed RNG assumption")
        off = outcomes(data["arms"]["frozen_faulted"], path + ":frozen")
        on = outcomes(data["arms"]["adaptive"], path + ":adaptive")
        require(off.keys() == on.keys(), path + ": unequal paired episode keys")
        suite = data["args"]["suite"]
        fixed = sum(on[key] and not value for key, value in off.items())
        broken = sum(value and not on[key] for key, value in off.items())
        for (task, init), value in off.items():
            clusters[(suite, task, init)] += int(on[(task, init)]) - int(value)
        cells.append({"path": path, "suite": suite, "joint_fault": data["joint_fault"],
                      "paired_rows": len(off), "frozen_successes": sum(off.values()),
                      "adaptive_successes": sum(on.values()), "fixed": fixed, "broken": broken,
                      "exact_raw_mcnemar_p": mcnemar(fixed, broken)})
        inputs.append({"path": path, "git_blob_id": blob,
                       "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})

    distribution = Counter({0: 1})
    nonzero = [abs(value) for value in clusters.values() if value]
    for weight in nonzero:
        updated = Counter()
        for value, count in distribution.items():
            updated[value - weight] += count
            updated[value + weight] += count
        distribution = updated
    observed = abs(sum(clusters.values()))
    assignments = 2**len(nonzero)
    require(sum(distribution.values()) == assignments, "Sign-flip DP mass mismatch")
    tail = sum(count for value, count in distribution.items() if abs(value) >= observed)
    total = summarize(cells)
    return {
        "schema_version": 1,
        "source_commit": SOURCE_COMMIT,
        "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": inputs,
        "per_cell": cells,
        "per_suite": {suite: summarize([cell for cell in cells if cell["suite"] == suite])
                      for suite in sorted({cell["suite"] for cell in cells})},
        "descriptive_total": total,
        "naive_pooled_mcnemar": {
            "status": "arithmetic reproduction only; repeated scenarios are not independent rows",
            "p": mcnemar(total["fixed"], total["broken"])},
        "posthoc_cluster_sign_flip": {
            "status": "exploratory sensitivity analysis, not preregistered confirmation",
            "cluster_key": ["suite", "task", "init"],
            "statistic": "absolute sum of within-cluster adaptive-minus-frozen success differences",
            "cluster_count": len(clusters), "nonzero_cluster_count": len(nonzero),
            "observed_absolute_sum": observed,
            "tail_assignments": tail, "total_nonzero_sign_assignments": assignments,
            "exact_two_sided_p": probability(Fraction(tail, assignments)),
            "cluster_weights": [{"suite": key[0], "task": key[1], "init": key[2], "weight": value}
                                for key, value in sorted(clusters.items())],
            "assumptions": [
                "Scenario clusters are independent; dependence across joint conditions within a cluster is retained.",
                "Under the tested null, each entire cluster contrast is sign-exchangeable.",
                "The selected eight cells and fixed weighting define this descriptive panel, not a population of all tasks or robots."],
            "limitations": [
                "Post-hoc grouping/test choice; no correction for sequential development, bound selection, or repeated evaluation.",
                "Independent policy draws remain unmatched; fixed/broken labels are observed outcomes, not deterministic counterfactuals.",
                "Does not isolate composite feedback, learned features, geometry, or causal fault recovery."]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON receipt; otherwise print to stdout")
    args = parser.parse_args()
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    result = recount(root)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
        print(json.dumps({"receipt": str(args.output), "source_commit": SOURCE_COMMIT,
                          "total": result["descriptive_total"],
                          "posthoc_cluster_p": result["posthoc_cluster_sign_flip"]["exact_two_sided_p"]}, indent=2))
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
