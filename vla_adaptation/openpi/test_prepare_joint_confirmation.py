"""Prospective preparation and global multi-suite inference tests; no rollouts."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

import prepare_joint_confirmation as prepare
import score_joint_followup as scorer
from openloop_id import nominal_commands


class Tests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        calibration = root/"calibration"
        source = root/"source"
        (source/"openpi").mkdir(parents=True)
        for name in (*prepare.RUNNER_SOURCES, "weighted_dob.py", "score_joint_followup.py",
                     "collect_followup_results.py", "prepare_joint_confirmation.py"):
            (source/"openpi"/name).write_text("# frozen test source "+name+"\n")
        (source/prepare.PREREG).parent.mkdir(parents=True)
        (source/prepare.PREREG).write_text("Prospective test registration.\n")
        selection = root/"selection.json"
        selection.write_text(json.dumps(dict(candidate="weighted_full", evidence="synthetic development selection")))
        for suite in prepare.SUITES:
            folder = calibration/suite
            folder.mkdir(parents=True)
            keys = [[task, 26] for task in range(10)]
            commands = np.arange(700, dtype=float).reshape(100, 7)/1000
            nominal = dict(sev=0, ep_len=[10]*10, n_steps=100, episode_keys=keys,
                per_ep=[dict(task=task, init=26, steps=10, ok=True) for task in range(10)],
                raw_a=commands[:, :6].tolist(), raw_exec=commands[:, :6].tolist(),
                raw_d=(commands[:, :6]*.03).tolist(), raw_cmd=commands.tolist(), raw_executed=commands.tolist())
            fir = dict(status="complete", suite=suite, reset_protocol="libero-reset-v1",
                       records=[nominal], calib_episodes=keys)
            path = folder/"fir.json"
            path.write_text(json.dumps(fir))
            _, command_source = nominal_commands(fir, 0, 8)
            original_path = f"/original/calibration/{suite}/fir.json"
            M = dict(status="complete", suite=suite, reset_protocol="libero-reset-v1", probe=.02,
                     M=np.eye(6).tolist(), probe_episodes=[[0, 25]], probe_task=0, probe_init=25,
                     source_calib_episodes=keys, command_source=command_source,
                     args=dict(log=original_path), source_hashes={original_path: prepare.digest(path)})
            (folder/"M.json").write_text(json.dumps(M))
        args = prepare.parser().parse_args(["--candidate", "weighted_full", "--calibration-root", str(calibration),
            "--source-root", str(source), "--snapshot-root", str(root/"snapshot"),
            "--remote-root", str(root/"remote"), "--out-dir", str(root/"prepared"),
            "--selection-record", str(selection), "--registration", "prospective synthetic lock", "--workers", "6"])
        return args

    def result_records(self, manifest, directory):
        def arm(wins):
            return dict(n=20, successes=wins, per_ep=[dict(task=task, init=initial, ok=index < wins)
                        for index, (task, initial) in enumerate(manifest["expected_keys"])])
        records = {}
        for spec in manifest["runs"]:
            candidate = spec["expected_args"]["followup_arm"] == "weighted_full"
            record = dict(args=copy.deepcopy(spec["expected_args"]), status="complete",
                source_hashes=copy.deepcopy(spec["expected_source_hashes"]),
                pairing=dict(valid_so_far=True), shared_control_id=spec["cell"]+":shared_off",
                arms=dict(frozen_faulted=arm(1), adaptive=arm(11 if candidate else 10)))
            records[(directory/spec["path"]).resolve()] = record
        return records

    def score(self, manifest, directory, records):
        return scorer.score(manifest, directory, reader=lambda path: json.dumps(records[path]).encode())

    def test_full_allocation_and_global_families(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            artifacts = prepare.build(args)
            launch, manifest, collector = (artifacts[name] for name in
                ("launch_manifest.json", "score_manifest.json", "collector_plan.json"))
            self.assertEqual(launch["allocation"]["total_rollouts"], 1440)
            self.assertEqual(len(launch["jobs"]), 24)
            self.assertEqual(len(manifest["runs"]), 48)
            self.assertEqual(len(manifest["comparisons"]), 72)
            self.assertEqual(launch["concurrency"], 6)
            self.assertEqual(launch["shared_policy_server"]["port"], 8000)
            self.assertEqual(launch["env"]["MUJOCO_EGL_DEVICE_ID"], "0")
            for suite in prepare.SUITES:
                cells = [job for job in launch["jobs"] if job["suite"] == suite]
                self.assertEqual({job["condition"] for job in cells}, {"healthy"} | {f"joint{i}" for i in range(7)})
                for cell in cells:
                    argv = cell["argv"]
                    self.assertEqual(argv[argv.index("--port")+1], "8000")
                    self.assertEqual(argv[argv.index("--control")+1], "/tmp/ctl.json")
                    self.assertEqual(argv[argv.index("--max-steps")+1], str(prepare.CAPS[suite]))
                    self.assertEqual(argv[argv.index("--eval-init")+1], "35")
            self.assertEqual(len(collector["collection_jobs"]), 24)
            records = self.result_records(manifest, args.out_dir)
            report = self.score(manifest, args.out_dir, records)
            self.assertEqual(len(report["comparisons"]), 72)
            self.assertTrue(all(row["family_tests"] == 24 for row in report["comparisons"]))
            legacy = next(row for row in report["comparisons"] if row["family"] == "legacy_vs_off")
            self.assertEqual(legacy["p"], .00390625)
            self.assertEqual(legacy["p_bonferroni"], .09375)
            self.assertEqual(legacy["inference"], "not_resolved")  # per-suite factor 8 would falsely resolve it
            primary = manifest["planned_aggregates"][0]
            self.assertEqual(primary["expected_scenarios"], 60)
            self.assertEqual(primary["family_tests"], 1)
            ids = {row["id"] for row in report["comparisons"]}
            self.assertTrue(set(primary["comparison_ids"]) <= ids)

    def test_suite_aliases_and_missing_global_comparisons_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            manifest = prepare.build(args)["score_manifest.json"]
            records = self.result_records(manifest, args.out_dir)
            altered = copy.deepcopy(manifest)
            for run in altered["runs"]:
                run["cell"] = "same_label_in_every_suite"
            spatial = next(run for run in altered["runs"] if run["suite"] == "libero_spatial")
            other = next(run for run in altered["runs"] if run["suite"] == "libero_object")
            altered["comparisons"][0].update(left=dict(run=spatial["id"], arm="adaptive"),
                                             right=dict(run=other["id"], arm="adaptive"))
            with self.assertRaisesRegex(ValueError, "cross-suite"):
                self.score(altered, args.out_dir, records)
            altered = copy.deepcopy(manifest)
            altered["comparisons"].pop()
            with self.assertRaisesRegex(ValueError, "global family sizes"):
                self.score(altered, args.out_dir, records)
            altered = copy.deepcopy(manifest)
            altered["runs"][0]["suite"] = "libero_object"
            with self.assertRaisesRegex(ValueError, "suite mismatch"):
                self.score(altered, args.out_dir, records)

    def test_shared_controls_and_frozen_hashes_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            manifest = prepare.build(args)["score_manifest.json"]
            records = self.result_records(manifest, args.out_dir)
            first = next(iter(records))
            damaged = copy.deepcopy(records)
            key = next(iter(damaged[first]["source_hashes"]))
            damaged[first]["source_hashes"][key] = "0"*64
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                self.score(manifest, args.out_dir, damaged)
            damaged = copy.deepcopy(records)
            damaged[first]["shared_control_id"] = "another_off_cohort"
            with self.assertRaisesRegex(ValueError, "shared-control views disagree"):
                self.score(manifest, args.out_dir, damaged)

    def test_calibration_overlap_and_wrong_donor_bytes_rejected(self):
        for mutation in ("evaluation_overlap", "source_hash", "command_hash", "incomplete"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                args = self.fixture(directory)
                path = args.calibration_root/"libero_spatial"/"M.json"
                M = json.loads(path.read_text())
                if mutation == "evaluation_overlap":
                    M.update(probe_init=35, probe_episodes=[[0, 35]])
                elif mutation == "source_hash":
                    M["source_hashes"][M["args"]["log"]] = "0"*64
                elif mutation == "command_hash":
                    M["command_source"]["command_sha256"] = "0"*64
                else:
                    M["status"] = "running"
                path.write_text(json.dumps(M))
                with self.assertRaises(ValueError):
                    prepare.build(args)

    def test_preflight_rejects_changed_inputs_and_existing_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            manifest = prepare.build(args)["launch_manifest.json"]
            for row in manifest["frozen_inputs"]:
                target = Path(row["remote"])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(row["local"], target)
            path = Path(directory)/"launch.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(prepare.verify_inputs(path)["prospective_cells"], 24)
            target = Path(manifest["frozen_inputs"][0]["remote"])
            original = target.read_bytes()
            target.write_bytes(original+b"changed")
            with self.assertRaisesRegex(ValueError, "changed frozen input"):
                prepare.verify_inputs(path)
            target.write_bytes(original)
            Path(manifest["jobs"][0]["out_dir"]).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "output already exists"):
                prepare.verify_inputs(path)

    def test_explicit_candidate_matches_selection_and_writes_reviewable_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            args.candidate = "legacy_translation"
            with self.assertRaisesRegex(ValueError, "distinct from legacy"):
                prepare.build(args)
            args.candidate = "weighted_translation"
            with self.assertRaisesRegex(ValueError, "selection record"):
                prepare.build(args)
            argv = ["--candidate", "weighted_full", "--calibration-root", str(args.calibration_root),
                    "--source-root", str(args.source_root), "--snapshot-root", str(args.snapshot_root),
                    "--remote-root", str(args.remote_root), "--out-dir", str(args.out_dir),
                    "--selection-record", str(args.selection_record), "--registration", args.registration]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(prepare.main(argv), 0)
            self.assertEqual({path.name for path in args.out_dir.iterdir()},
                             {"launch_manifest.json", "score_manifest.json", "collector_plan.json"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
