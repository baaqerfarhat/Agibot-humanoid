"""Score a declared LIBERO or ALOHA follow-up from stored outcomes only.

The manifest names complete run files, their expected scenarios/configuration, and
every comparison in each multiplicity family. No directory scan or intersection of
episode sets can silently select a favorable subset. Scenario matching is a design
assumption that the experiment must verify; matching identifiers alone does not
prove matching physical scenes. Unpaired data are deliberately rejected here.

Usage:
    python3 -B openpi/score_joint_followup.py study_manifest.json
    python3 -B openpi/score_joint_followup.py study_manifest.json --json
    python3 -B openpi/score_joint_followup.py --example
    python3 -B openpi/score_joint_followup.py --selftest

Result paths are relative to the manifest. See --example for its schema. Error
metrics are not inferred from f_hat: a Cartesian estimate is not a ground-truth
joint torque. This scorer reports task outcomes, not estimator accuracy or safety
certification. It never launches a simulator or writes a result file.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    if __package__:
        from . import mcnemar, mcnemar_crosscheck
    else:
        import mcnemar, mcnemar_crosscheck
finally:
    sys.dont_write_bytecode = _bytecode


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(rows, label):
    require(isinstance(rows, list) and rows, f"{label}: nonempty scenario list required")
    result = []
    for row in rows:
        require(isinstance(row, (list, tuple)) and len(row) == 2
                and all(type(x) is int and x >= 0 for x in row),
                f"{label}: scenarios must be [nonnegative task, nonnegative init]")
        result.append(tuple(row))
    require(len(result) == len(set(result)), f"{label}: duplicate scenario key")
    return set(result)


def outcomes(arm, expected, label, seed_offset=None):
    require(isinstance(arm, dict), f"{label}: arm object required")
    rows = arm.get("per_ep")
    require(isinstance(rows, list), f"{label}: per_ep required")
    result = {}
    for row in rows:
        require(isinstance(row, dict), f"{label}: episode object required")
        task, init, ok = row.get("task"), row.get("init"), row.get("ok")
        require(type(task) is int and type(init) is int and task >= 0 and init >= 0,
                f"{label}: nonnegative integer task/init required")
        require(type(ok) in (bool, int) and ok in (False, True),
                f"{label}: ok must be boolean or 0/1")
        if seed_offset is not None:
            require(task == 0 and 0 <= init < len(expected),
                    f"{label}: ALOHA requires task=0 and local episode indices 0..n-1")
        key = (task, init if seed_offset is None else seed_offset + init)
        require(key not in result, f"{label}: duplicate scenario {key}")
        result[key] = bool(ok)
    require(set(result) == expected,
            f"{label}: scenario mismatch; missing={sorted(expected-set(result))}, "
            f"unexpected={sorted(set(result)-expected)}")
    require(type(arm.get("n")) is int and arm["n"] == len(result),
            f"{label}: n disagrees with per_ep")
    require(type(arm.get("successes")) is int
            and arm["successes"] == sum(result.values()),
            f"{label}: successes disagree with per_ep")
    return result


def exact_mcnemar(left_only, right_only):
    a = mcnemar.mcnemar_exact(left_only, right_only)
    b = mcnemar_crosscheck.exact_two_sided(left_only, right_only)
    require(math.isfinite(a) and 0 <= a <= 1
            and math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0),
            f"McNemar implementations disagree: {a} versus {b}")
    return a


def upper95(events, trials):
    """One-sided 95% Clopper-Pearson bound; no opportunities means no bound."""
    require(type(events) is int and type(trials) is int and 0 <= events <= trials,
            "invalid event count")
    if trials == 0:
        return None
    if events == trials:
        return 1.0
    if events == 0:
        return -math.expm1(math.log(0.05) / trials)
    # Log-space binomial probabilities avoid overflowing combinations at large n.
    def cdf(p):
        logs = [math.lgamma(trials + 1) - math.lgamma(i + 1)
                - math.lgamma(trials - i + 1) + i * math.log(p)
                + (trials - i) * math.log1p(-p) for i in range(events + 1)]
        maximum = max(logs)
        return math.exp(maximum) * math.fsum(math.exp(x - maximum) for x in logs)
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        if cdf(mid) > 0.05:
            lo = mid
        else:
            hi = mid
    return hi


def compare(left, right):
    require(set(left) == set(right) and left, "comparison needs identical nonempty scenario sets")
    n = len(left)
    lost = sum(left[k] and not right[k] for k in left)
    gained = sum(right[k] and not left[k] for k in left)
    baseline = sum(left.values())
    successes = sum(right.values())
    return dict(n=n, left_successes=baseline, right_successes=successes,
                gain_points=100 * (successes - baseline) / n,
                right_only=gained, left_only=lost,
                p=exact_mcnemar(lost, gained),
                regressions_all=lost / n,
                regression_opportunities=baseline,
                regressions_conditional=lost / baseline if baseline else None,
                regression_upper95_one_sided=upper95(lost, baseline))


def score(manifest, directory=Path("."), reader=None):
    """Validate all records and contrasts before returning any report."""
    reader = reader or (lambda path: path.read_bytes())
    require(isinstance(manifest, dict), "manifest must be an object")
    require(manifest.get("schema_version") == 1, "schema_version must be 1")
    require(isinstance(manifest.get("study"), str) and manifest["study"], "study is required")
    stage = manifest.get("stage")
    require(stage in ("development", "confirmation"), "stage must be development or confirmation")
    pairing = manifest.get("pairing")
    require(pairing in ("libero_set_init_state", "aloha_seeded_reset"),
            "only verified LIBERO set_init_state or ALOHA seeded-reset pairing is supported")
    aloha = pairing == "aloha_seeded_reset"
    suites = ("gym_aloha/AlohaTransferCube-v0",) if aloha else (
        "libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
    if "suites" in manifest:
        declared_suites = manifest["suites"]
        require("suite" not in manifest and not aloha and isinstance(declared_suites, list)
                and declared_suites and len(set(declared_suites)) == len(declared_suites)
                and all(suite in suites for suite in declared_suites),
                "multi-suite manifests require unique supported LIBERO suites and no single suite field")
        multisuite = True
    else:
        require(manifest.get("suite") in suites, "suite does not match the pairing mechanism")
        declared_suites, multisuite = [manifest["suite"]], False
    expected = keys(manifest.get("expected_keys"), "expected_keys")
    forbidden = manifest.get("forbidden_keys", [])
    if forbidden:
        forbidden = keys(forbidden, "forbidden_keys")
        require(not expected.intersection(forbidden), "evaluation overlaps forbidden calibration/development keys")
    require(isinstance(manifest.get("registration"), str) and manifest["registration"],
            "registration must identify the design record and its revision")
    if stage == "confirmation":
        require(manifest.get("locked_before_outcomes") is True,
                "confirmation requires a design locked before its outcomes")
        require(isinstance(manifest.get("selection_record"), str)
                and manifest["selection_record"], "confirmation requires selection_record")
        require(forbidden, "confirmation requires explicit forbidden calibration/development keys")
    specs = manifest.get("runs")
    require(isinstance(specs, list) and specs, "runs must be nonempty")
    runs, artifacts, used_paths = {}, [], set()
    for spec in specs:
        require(isinstance(spec, dict), "run specification must be an object")
        rid, cell = spec.get("id"), spec.get("cell")
        require(isinstance(rid, str) and rid and rid not in runs, "run IDs must be nonempty and unique")
        require(isinstance(cell, str) and cell, f"{rid}: cell identifier required")
        require(isinstance(spec.get("path"), str) and spec["path"], f"{rid}: path required")
        path = (directory / spec["path"]).resolve()
        require(path not in used_paths, f"{rid}: result path already declared")
        used_paths.add(path)
        raw = reader(path)
        require(isinstance(raw, bytes), f"{rid}: result reader must return bytes")
        record = json.loads(raw)
        require(isinstance(record, dict) and isinstance(record.get("args"), dict),
                f"{rid}: recorded args required")
        if "status" in record:
            require(record["status"] == "complete", f"{rid}: result status is not complete")
        if isinstance(record.get("pairing"), dict) and "valid_so_far" in record["pairing"]:
            require(record["pairing"]["valid_so_far"] is True, f"{rid}: recorded pairing check failed")
        args = record["args"]
        run_suite = spec.get("suite") if multisuite else manifest["suite"]
        require(run_suite in declared_suites, f"{rid}: run must declare a supported manifest suite")
        seed_offset = None
        if aloha:
            require(type(args.get("seed")) is int and args["seed"] >= 0,
                    f"{rid}: ALOHA base seed required")
            seed_offset = args["seed"]
        else:
            require(args.get("suite") == run_suite, f"{rid}: suite mismatch")
        require(type(args.get("episodes")) is int and args["episodes"] == len(expected),
                f"{rid}: args.episodes mismatch")
        required_args = spec.get("expected_args")
        require(isinstance(required_args, dict) and required_args,
                f"{rid}: expected_args must fix the intended condition/configuration")
        for key, value in required_args.items():
            require(key in args and args[key] == value, f"{rid}: expected args.{key}={value!r}, got {args.get(key)!r}")
        if "expected_source_hashes" in spec:
            expected_hashes = spec["expected_source_hashes"]
            require(isinstance(expected_hashes, dict) and expected_hashes, f"{rid}: nonempty source hashes required")
            require(isinstance(record.get("source_hashes"), dict), f"{rid}: recorded source hashes missing")
            for source, digest in expected_hashes.items():
                require(record["source_hashes"].get(source) == digest, f"{rid}: source hash mismatch for {source}")
        require(isinstance(record.get("arms"), dict), f"{rid}: arms required")
        require("frozen_faulted" in record["arms"] and "adaptive" in record["arms"],
                f"{rid}: frozen_faulted and adaptive arms required")
        arms = {name: outcomes(arm, expected, f"{rid}/{name}", seed_offset)
                for name, arm in record["arms"].items()}
        condition = {key: args.get(key) for key in
                     ("joint_fault", "joint_faults", "fault_vec", "gain", "profile",
                      "prof_p", "onset", "wrist_shift", "obs_offset", "task")}
        if not any(condition[key] for key in ("joint_fault", "joint_faults", "fault_vec", "gain")):
            condition["sev"] = args.get("sev")
        runs[rid] = dict(cell=cell, suite=run_suite, arms=arms, condition=condition,
                         shared_control_id=record.get("shared_control_id"))
        artifacts.append(dict(id=rid, cell=cell, suite=run_suite, path=str(path),
                              sha256=hashlib.sha256(raw).hexdigest()))
    for group in manifest.get("shared_control_groups", []):
        require(isinstance(group, list) and len(group) >= 2 and len(set(group)) == len(group)
                and all(rid in runs for rid in group), "invalid shared-control group")
        first = runs[group[0]]
        require(isinstance(first["shared_control_id"], str) and first["shared_control_id"],
                "shared-control ID missing")
        for rid in group[1:]:
            other = runs[rid]
            require((other["suite"], other["cell"], other["shared_control_id"], other["arms"]["frozen_faulted"])
                    == (first["suite"], first["cell"], first["shared_control_id"], first["arms"]["frozen_faulted"]),
                    "declared shared-control views disagree")
    comparisons = manifest.get("comparisons")
    require(isinstance(comparisons, list) and comparisons, "comparisons must be nonempty")
    rows, ids, contrast_keys = [], set(), set()
    for spec in comparisons:
        require(isinstance(spec, dict), "comparison specification must be an object")
        cid, family = spec.get("id"), spec.get("family")
        require(isinstance(cid, str) and cid and cid not in ids, "comparison IDs must be unique")
        ids.add(cid)
        require(isinstance(family, str) and family, f"{cid}: multiplicity family required")
        refs = []
        for side in ("left", "right"):
            ref = spec.get(side)
            require(isinstance(ref, dict) and ref.get("run") in runs,
                    f"{cid}: unknown {side} run")
            run = runs[ref["run"]]
            require(ref.get("arm") in run["arms"], f"{cid}: unknown {side} arm")
            refs.append((ref["run"], ref["arm"]))
        require(refs[0] != refs[1], f"{cid}: cannot compare an arm with itself")
        require(runs[refs[0][0]]["suite"] == runs[refs[1][0]]["suite"],
                f"{cid}: cross-suite comparison is not a matched treatment contrast")
        contrast_key = tuple(sorted(refs))
        require(contrast_key not in contrast_keys, f"{cid}: duplicate or reversed contrast")
        contrast_keys.add(contrast_key)
        require(runs[refs[0][0]]["cell"] == runs[refs[1][0]]["cell"],
                f"{cid}: cross-cell comparison is not a matched treatment contrast")
        require(runs[refs[0][0]]["condition"] == runs[refs[1][0]]["condition"],
                f"{cid}: recorded fault/task configurations differ between runs")
        row = compare(*(runs[rid]["arms"][arm] for rid, arm in refs))
        rows.append(dict(id=cid, family=family, suite=runs[refs[0][0]]["suite"],
                         left=list(refs[0]), right=list(refs[1]), **row))
    families = Counter(row["family"] for row in rows)
    if "family_sizes" in manifest:
        require(dict(families) == manifest["family_sizes"], "comparison families differ from declared global family sizes")
    for row in rows:
        row["family_tests"] = families[row["family"]]
        row["p_bonferroni"] = min(1.0, row["p"] * row["family_tests"])
        row["inference"] = ("right_higher" if row["gain_points"] > 0 else "left_higher") if row["p_bonferroni"] < .05 else "not_resolved"
        if stage == "development":
            row["inference"] = "exploratory_" + row["inference"]
    return dict(study=manifest["study"], stage=stage, registration=manifest["registration"],
                n_scenarios=len(expected), suites=declared_suites, artifacts=artifacts, comparisons=rows,
                caveat="Scenario identity must be independently checked. Not resolved is not equivalence. "
                       "A conditional harm bound is not a deployment safety guarantee.")


def render(report):
    lines = [f"{report['study']} — {report['stage']}; {report['n_scenarios']} matched scenarios per run",
             "Comparison: left→right; regressions are left successes lost by right."]
    for row in report["comparisons"]:
        opp, lost = row["regression_opportunities"], row["left_only"]
        bound = row["regression_upper95_one_sided"]
        conditional = (f"{lost}/{opp}; one-sided 95% upper bound {100*bound:.1f}%" if opp
                       else "undefined (no left-success opportunities)")
        lines.extend([f"{row['id']}: {row['left_successes']}/{row['n']}→{row['right_successes']}/{row['n']} "
                      f"({row['gain_points']:+.1f} points), right-only={row['right_only']}, left-only={lost}; "
                      f"McNemar p={row['p']:.6g}, Bonferroni({row['family_tests']})={row['p_bonferroni']:.6g}; {row['inference']}",
                      f"  regressions/all={lost}/{row['n']}; regressions/left successes={conditional}"])
    lines.append(report["caveat"])
    return "\n".join(lines)


def example():
    return dict(schema_version=1, study="composite_joint_followup", stage="development",
                suite="libero_spatial", pairing="libero_set_init_state",
                registration="PREREG_COMPOSITE_JOINT_FOLLOWUP.md at <revision>",
                expected_keys=[[task, 30] for task in range(10)],
                forbidden_keys=[[0, 25]],
                runs=[dict(id="candidate_j5", cell="torque5", path="candidate_j5.json",
                           expected_args=dict(joint_fault="torque:5:5.0", estimate_only=False))],
                comparisons=[dict(id="candidate_vs_frozen_j5", family="development",
                                  left=dict(run="candidate_j5", arm="frozen_faulted"),
                                  right=dict(run="candidate_j5", arm="adaptive"))])


def selftest():
    """Exercise inference and provenance rejection on synthetic, in-memory data."""
    import copy
    import unittest

    class Tests(unittest.TestCase):
        def fixture(self):
            manifest = example()
            expected = manifest["expected_keys"]
            def arm(wins):
                return dict(n=10, successes=wins,
                            per_ep=[dict(task=t, init=i, ok=k < wins)
                                    for k, (t, i) in enumerate(expected)])
            record = dict(args=dict(suite="libero_spatial", episodes=10,
                                    joint_fault="torque:5:5.0", estimate_only=False),
                          arms=dict(frozen_faulted=arm(1), adaptive=arm(9)))
            return manifest, record

        def score(self, manifest, record):
            return score(manifest, reader=lambda _: json.dumps(record).encode())

        def test_known_inference_and_bound(self):
            self.assertAlmostEqual(exact_mcnemar(9, 0), .00390625)
            self.assertAlmostEqual(upper95(0, 6), .39303776899708276)
            self.assertIsNone(upper95(0, 0))
            self.assertEqual(upper95(4, 4), 1.0)
            manifest, record = self.fixture()
            result = self.score(manifest, record)["comparisons"][0]
            self.assertEqual((result["left_only"], result["right_only"]), (0, 8))
            self.assertEqual(result["inference"], "exploratory_right_higher")
            record["arms"]["adaptive"]["per_ep"].reverse()
            self.assertEqual(self.score(manifest, record)["comparisons"][0], result)

        def test_bad_cohorts_rejected(self):
            for change in ("duplicate", "missing", "count", "successes", "condition"):
                manifest, record = self.fixture()
                arm = record["arms"]["adaptive"]
                if change == "duplicate":
                    arm["per_ep"][-1] = arm["per_ep"][0]
                elif change == "missing":
                    arm["per_ep"].pop()
                elif change == "count":
                    arm["n"] = 11
                elif change == "successes":
                    arm["successes"] = 8
                else:
                    record["args"]["joint_fault"] = "torque:3:5.0"
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.score(manifest, record)

        def test_confirmation_requires_lock_and_disjointness(self):
            manifest, record = self.fixture()
            manifest["stage"] = "confirmation"
            with self.assertRaises(ValueError):
                self.score(manifest, record)
            manifest.update(locked_before_outcomes=True, selection_record="dev-selection at revision")
            self.assertEqual(self.score(manifest, record)["stage"], "confirmation")
            manifest["forbidden_keys"] = [[0, 30]]
            with self.assertRaises(ValueError):
                self.score(manifest, record)

        def test_declared_family_and_cross_cell(self):
            manifest, record = self.fixture()
            second = copy.deepcopy(manifest["runs"][0])
            second.update(id="second", path="second.json")
            manifest["runs"].append(second)
            contrast = copy.deepcopy(manifest["comparisons"][0])
            contrast.update(id="head_to_head", left=dict(run="candidate_j5", arm="adaptive"),
                            right=dict(run="second", arm="adaptive"))
            manifest["comparisons"].append(contrast)
            rows = self.score(manifest, record)["comparisons"]
            self.assertTrue(all(row["family_tests"] == 2 for row in rows))
            self.assertEqual(rows[0]["p_bonferroni"], min(1.0, 2*rows[0]["p"]))
            manifest["runs"][1]["cell"] = "different_fault"
            with self.assertRaises(ValueError):
                self.score(manifest, record)

        def test_duplicate_contrast_and_unpaired_rejected(self):
            manifest, record = self.fixture()
            duplicate = copy.deepcopy(manifest["comparisons"][0])
            duplicate["id"] = "duplicate"
            manifest["comparisons"].append(duplicate)
            with self.assertRaises(ValueError):
                self.score(manifest, record)
            manifest, record = self.fixture()
            manifest["pairing"] = "same_seed_gr1"
            with self.assertRaises(ValueError):
                self.score(manifest, record)

        def test_aloha_seed_band_is_not_local_episode_index(self):
            manifest, record = self.fixture()
            manifest.update(pairing="aloha_seeded_reset", suite="gym_aloha/AlohaTransferCube-v0",
                            expected_keys=[[0, 1000+i] for i in range(10)])
            record["args"].pop("suite")
            record["args"]["seed"] = 1000
            for arm in record["arms"].values():
                for i, episode in enumerate(arm["per_ep"]):
                    episode.update(task=0, init=i)
            self.assertEqual(self.score(manifest, record)["n_scenarios"], 10)
            record["args"]["seed"] = 2000
            with self.assertRaisesRegex(ValueError, "scenario mismatch"):
                self.score(manifest, record)

    return unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path)
    parser.add_argument("--json", action="store_true", help="emit machine-readable scores and artifact hashes")
    parser.add_argument("--example", action="store_true", help="print a manifest example without writing files")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return 0 if selftest() else 1
    if args.example:
        print(json.dumps(example(), indent=2))
        return 0
    if args.manifest is None:
        parser.error("manifest is required unless --example or --selftest is used")
    try:
        raw = args.manifest.read_bytes()
        report = score(json.loads(raw), args.manifest.resolve().parent)
        report["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    except (ValueError, OSError) as exc:
        print(f"No scores produced: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, allow_nan=False) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
