"""Synthetic complete-allocation and rejection tests; never read live outcomes."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import pathlib
import tempfile
import unittest

try:
    from . import summarize_followup_confirmation as summary
except ImportError:
    import summarize_followup_confirmation as summary


def sha(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def fixture(robot="panda"):
    panda = robot == "panda"
    suites = summary.PANDA_SUITES if panda else ("gym_aloha/AlohaTransferCube-v0",)
    expected = [[t, i] for t in range(10) for i in (35, 36)] if panda else [[0, 3000+i] for i in range(20)]
    arms = ("legacy_translation", "weighted_full") if panda else ("legacy_correctedchannels", "weighted_full")
    manifest = dict(schema_version=1, study="synthetic_confirmation", stage="confirmation",
        pairing="libero_set_init_state" if panda else "aloha_seeded_reset",
        registration="synthetic locked allocation", locked_before_outcomes=True,
        selection_record="locked_selection.json", expected_keys=expected, forbidden_keys=[[0, 25]],
        runs=[], comparisons=[], shared_control_groups=[], family_sizes={
            "legacy_vs_off": 24 if panda else 13, "candidate_vs_off": 24 if panda else 13,
            "candidate_vs_legacy": 24 if panda else 13})
    manifest.update({"suites": list(suites)} if panda else {"suite": suites[0]})
    directory = pathlib.Path("/synthetic_confirmation")
    records = {}
    for suite in suites:
        for j in range(-1, 7 if panda else 12):
            label = "healthy" if j < 0 else f"joint{j}"
            cid = suite.replace("/", "_")+"_"+label
            torque = 0. if j < 0 else (5. if panda else [16., 32., 16., .2, 1., .4][j % 6])
            fault = f"torque:{max(j, 0)}:{torque}" if panda or j >= 0 else None
            def arm(name):
                rows = []
                for k, (task, init) in enumerate(expected):
                    # Primary has30 fixes /0 breaks across disjoint suites. Other fault
                    # cells deliberately disagree, so accidental cross-fault pooling fails.
                    if j == 5:
                        ok = k >= 10 if name == "off" else (False if name == arms[0] else k < 10)
                    else:
                        ok = name != arms[1]
                    state = dict(sha256=sha((suite, task, init)), pairing_valid=True,
                        max_abs_qpos=0., max_abs_qvel=0., nonstate_physics_mismatch=0.)
                    rows.append(dict(task=task, init=init if panda else k, ok=ok, initial_state=state))
                return dict(n=20, successes=sum(row["ok"] for row in rows), per_ep=rows)
            group = []
            for name in arms:
                rid = cid+"_"+name
                args = dict(suite=suite, episodes=20, stage="confirm", joint_fault=fault,
                    followup_arm=name, seed=3000, baseline="weighted_dob" if name == "weighted_full" else "none",
                    corr_dims="0,1,2,3,4,5" if name == "weighted_full" else "0,1,2", state_tolerance=1e-9)
                source_hashes = {"frozen_runner.py": sha("frozen"), "healthy_calibration.json": sha(suite)}
                spec = dict(id=rid, cell=cid, path=rid+".json", expected_args=copy.deepcopy(args),
                            expected_source_hashes=source_hashes.copy())
                if panda:
                    spec["suite"] = suite
                manifest["runs"].append(spec)
                record = dict(schema_version=1, compact_schema_version=1, status="complete", scoreable=True,
                    stage="confirm", study_id="study_"+cid, shared_control_id="off_"+cid,
                    args=args, source_hashes=source_hashes, source_study=dict(path=cid+"/study.json", sha256=sha(cid), bytes=100),
                    source_telemetry=dict(path=cid+"/telemetry.jsonl", sha256=sha(cid+"tel"), hashed=True,
                        hash_scope="whole_file", unchanged_during_hash=True, bytes=100),
                    collector_source=dict(path="collector.py", sha256=sha("collector")),
                    pairing=dict(valid_so_far=True, mechanism=manifest["pairing"], checked_states=60, tolerance=1e-9),
                    arms=dict(frozen_faulted=arm("off"), adaptive=arm(name)))
                records[(directory/spec["path"]).resolve()] = record
                group.append(rid)
            manifest["shared_control_groups"].append(group)
            legacy, weighted = group
            for family, left, right in (
                ("legacy_vs_off", (legacy, "frozen_faulted"), (legacy, "adaptive")),
                ("candidate_vs_off", (weighted, "frozen_faulted"), (weighted, "adaptive")),
                ("candidate_vs_legacy", (legacy, "adaptive"), (weighted, "adaptive"))):
                manifest["comparisons"].append(dict(id=cid+"_"+family, family=family,
                    left=dict(run=left[0], arm=left[1]), right=dict(run=right[0], arm=right[1])))
    if panda:
        manifest["planned_aggregates"] = []
        for primary in (True, False):
            declaration = dict(id=summary.PRIMARY if primary else summary.SECONDARY,
                role="prespecified_primary" if primary else "descriptive_secondary",
                comparison_ids=[s+"_joint5_"+("candidate_vs_legacy" if primary else "candidate_vs_off") for s in suites],
                expected_scenarios=60, scenario_identity=["suite", "task", "init"])
            if primary:
                declaration.update(family=summary.PRIMARY, family_tests=1)
            manifest["planned_aggregates"].append(declaration)
    return manifest, directory, records


class ConfirmationSummaryTests(unittest.TestCase):
    def run_summary(self, manifest, directory, records, **kwargs):
        return summary.summarize(manifest, directory,
            reader=lambda path: json.dumps(records[path], sort_keys=True).encode(), **kwargs)

    def test_primary_only_pools_three_disjoint_joint5_suites(self):
        manifest, directory, records = fixture()
        report = self.run_summary(manifest, directory, records)
        self.assertEqual((report["cells"], report["unique_rollout_assignments"]), (24, 1440))
        self.assertEqual(report["family_sizes"], summary.PANDA_FAMILIES)
        primary, secondary = report["aggregates"]
        self.assertEqual((primary["n"], primary["right_only"], primary["left_only"]), (60, 30, 0))
        self.assertEqual(len(set(map(tuple, primary["scenarios"]))), 60)
        self.assertAlmostEqual(primary["p"], 2.**-29)
        self.assertEqual((secondary["right_only"], secondary["left_only"]), (30, 30))
        self.assertNotIn("p", secondary)
        self.assertNotIn("p_bonferroni", secondary)
        self.assertIn("broken", summary.markdown(report))

    def test_incomplete_allocation_rejected_before_reading_outcomes(self):
        manifest, directory, _ = fixture()
        manifest["runs"].pop()
        calls = []
        with self.assertRaisesRegex(ValueError, "48 compact views"):
            summary.summarize(manifest, directory, reader=lambda p: calls.append(p))
        self.assertEqual(calls, [])

    def test_family_reduction_and_missing_contrast_rejected(self):
        for mutation in ("family", "contrast"):
            manifest, directory, records = fixture()
            if mutation == "family":
                manifest["family_sizes"]["candidate_vs_off"] = 23
            else:
                manifest["comparisons"].pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.run_summary(manifest, directory, records)

    def test_changed_pooled_fault_or_repeated_suite_rejected(self):
        for mutation in ("fault", "suite"):
            manifest, directory, records = fixture()
            ids = manifest["planned_aggregates"][0]["comparison_ids"]
            ids[0] = ids[0].replace("joint5", "joint3") if mutation == "fault" else ids[1]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.run_summary(manifest, directory, records)

    def test_source_hash_and_collected_bytes_tampering_rejected(self):
        for mutation in ("source", "view"):
            manifest, directory, records = fixture()
            first = manifest["runs"][0]
            if mutation == "source":
                records[directory/first["path"]]["source_hashes"]["frozen_runner.py"] = sha("changed")
            else:
                first["collected_view_sha256"] = sha("other bytes")
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.run_summary(manifest, directory, records)

    def test_partial_failed_pairing_and_telemetry_prefix_rejected(self):
        for mutation in ("partial", "pairing", "episode", "telemetry"):
            manifest, directory, records = fixture()
            first = records[directory/manifest["runs"][0]["path"]]
            if mutation == "partial":
                first["status"] = "running"
            elif mutation == "pairing":
                first["pairing"]["valid_so_far"] = False
            elif mutation == "episode":
                first["arms"]["adaptive"]["per_ep"][0]["initial_state"]["pairing_valid"] = False
            else:
                first["source_telemetry"]["hash_scope"] = "prefix_at_collection_start"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.run_summary(manifest, directory, records)

    def test_shared_off_physical_evidence_cannot_differ(self):
        manifest, directory, records = fixture()
        first = records[directory/manifest["runs"][0]["path"]]
        first["arms"]["frozen_faulted"]["per_ep"][0]["initial_state"]["sha256"] = sha("other physical state")
        with self.assertRaisesRegex(ValueError, "shared off copies disagree"):
            self.run_summary(manifest, directory, records)

    def test_supplied_score_report_is_recomputed_and_checked(self):
        manifest, directory, records = fixture()
        report = self.run_summary(manifest, directory, records)
        scores = copy.deepcopy(report["validated_score_report"])
        matched = self.run_summary(manifest, directory, records, scored_report=scores)
        self.assertTrue(matched["validation"]["supplied_score_report_matched"])
        for artifact in scores["artifacts"]:
            artifact["path"] = "/relocated/"+pathlib.Path(artifact["path"]).name
        relocated = self.run_summary(manifest, directory, records, scored_report=scores)
        self.assertTrue(relocated["validation"]["supplied_score_report_matched"])
        scores["comparisons"][0]["right_successes"] += 1
        with self.assertRaisesRegex(ValueError, "supplied score report"):
            self.run_summary(manifest, directory, records, scored_report=scores)

    def test_aloha_all_twelve_joints_are_descriptive_without_pooling(self):
        manifest, directory, records = fixture("aloha")
        report = self.run_summary(manifest, directory, records)
        self.assertEqual((report["cells"], report["unique_rollout_assignments"]), (13, 780))
        self.assertEqual(report["aggregates"], [])
        self.assertEqual({r["condition"] for r in report["per_cell"]}, {"healthy"}|{f"joint{j}" for j in range(12)})
        manifest["planned_aggregates"] = [dict(id="posthoc_pool")]
        with self.assertRaisesRegex(ValueError, "no pooled tests"):
            self.run_summary(manifest, directory, records)

    def test_standalone_figures_export_denominators_and_named_comparisons(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("Matplotlib unavailable")
        manifest, directory, records = fixture("aloha")
        report = self.run_summary(manifest, directory, records)
        with tempfile.TemporaryDirectory() as folder:
            paths = summary.plot(report, folder)
            self.assertEqual({p.suffix for p in paths}, {".svg", ".png"})
            png = next(p for p in paths if p.suffix == ".png")
            svg = next(p for p in paths if p.suffix == ".svg")
            self.assertTrue(png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            text = svg.read_text()
            self.assertIn("n=20", text)
            self.assertIn("weighted_full", text)
            self.assertIn("Nm", text)


if __name__ == "__main__":
    unittest.main()
