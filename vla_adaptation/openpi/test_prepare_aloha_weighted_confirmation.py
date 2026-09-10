"""ALOHA prospective allocation, explicit collector API, and frozen-input tests."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import collect_followup_results as collector
import prepare_aloha_weighted_confirmation as prepare
import score_joint_followup as scorer


class Tests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        source = root/"source"
        (source/"openpi").mkdir(parents=True)
        for name in prepare.SNAPSHOT_FILES:
            (source/"openpi"/name).write_text("# frozen test source "+name+"\n")
        prereg = source/prepare.REGISTRATION
        prereg.parent.mkdir(parents=True)
        prereg.write_text("Prospective ALOHA transfer registration.\n")
        healthy = root/"healthy.json"
        healthy.write_text(json.dumps([dict(u=np.zeros((10, 14)).tolist(), q=np.zeros((10, 14)).tolist()) for _ in range(10)]))
        provenance = root/"provenance.json"
        provenance.write_text(json.dumps(dict(data_sha256=prepare.digest(healthy), header=dict(
            args=dict(seed=2600, episodes=10, mode="log", fault_vec=None, gain=None)))))
        sensitivity = root/"sensitivity.json"
        sensitivity.write_text(json.dumps(dict(M=np.eye(14).tolist(), probe=.02)))
        reference = root/"reference.json"
        reference.write_text(json.dumps(dict(qualification=dict(allowed=True), model=dict(state_indices=list(range(6))),
            observer=dict(W=np.zeros((14, 8)).tolist(), M=np.eye(14).tolist(), Q=(.001*np.eye(14)).tolist(),
                R=(.01*np.eye(14)).tolist(), bias=None, dt=.02, gamma=.08, fit_episode_indices=list(range(6))),
            settings=dict(fit_episodes=list(range(6)), validation_episodes=list(range(6, 10))),
            source=dict(sha256=prepare.digest(healthy), sensitivity=dict(sha256=prepare.digest(sensitivity))))))
        rejected = root/"rejected.json"
        rejected.write_text(json.dumps(dict(qualification=dict(allowed=False, reasons=["held-out error"])) ))
        return prepare.parser().parse_args(["--source-root", str(source), "--snapshot-root", str(root/"snapshot"),
            "--remote-root", str(root/"remote"), "--out-dir", str(root/"prepared"), "--prereg", str(prereg),
            "--healthy-log", str(healthy), "--healthy-provenance", str(provenance),
            "--sensitivity-artifact", str(sensitivity), "--reference-artifact", str(reference),
            "--rejected-reference", str(rejected)])

    def write_artifacts(self, args, artifacts):
        args.out_dir.mkdir(parents=True)
        for name, value in artifacts.items():
            (args.out_dir/name).write_bytes(prepare.serialized(value))

    def collect_synthetic_cells(self, args, artifacts):
        plan, score = artifacts["collector_plan.json"], artifacts["score_manifest.json"]
        for cell in plan["cells"]:
            condition = cell["id"]
            def arm(wins):
                return dict(n=20, successes=wins, per_ep=[dict(task=0, init=i, actual_seed=3000+i,
                    ok=i < wins, initial_state=dict(sha256=f"{3000+i:064x}", pairing_valid=True)) for i in range(20)])
            spec = next(run for run in score["runs"] if run["cell"] == condition)
            source_dir = args.remote_root/"cells"/condition
            source_dir.mkdir(parents=True)
            telemetry = source_dir/"telemetry.jsonl"
            telemetry.write_text('{"type":"synthetic_header"}\n')
            study = dict(study_id=condition+":study", status="complete", stage="confirm", robot_joints=[],
                args=dict(cell["expected_args"], telemetry=str(telemetry)),
                source_hashes=spec["expected_source_hashes"],
                arm_configs={name: prepare.ARM_CONFIGS[name] for name in prepare.ARMS},
                shared_control_id=condition+":off", pairing=dict(mechanism="aloha_seeded_reset", valid_so_far=True, checked_states=60),
                schedule=[dict(task=0, init=i) for i in range(20)],
                arms=dict(off=arm(1), legacy_allchannels=arm(10), weighted=arm(11)))
            path = source_dir/"study.json"
            path.write_text(json.dumps(study))
            collector.collect(path, args.out_dir/"compact"/condition, hash_telemetry=True)

    def test_exact_allocation_and_explicit_collector_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            artifacts = prepare.build(args)
            self.write_artifacts(args, artifacts)
            launch = artifacts["launch_manifest.json"]
            self.assertEqual(launch["allocation"]["total_rollouts"], 780)
            self.assertEqual(launch["launch_argv"][1], str(args.snapshot_root/"openpi/launch_followup_grid.py"))
            self.assertIn(launch["launch_argv"][1], [row["remote"] for row in launch["frozen_inputs"]])
            self.assertEqual(len(launch["jobs"]), 13)
            self.assertEqual(len(artifacts["score_manifest.json"]["runs"]), 26)
            self.assertEqual(len(artifacts["score_manifest.json"]["comparisons"]), 39)
            for job in launch["jobs"]:
                argv = job["argv"]
                self.assertEqual(argv[argv.index("--arms")+1], "off,legacy_allchannels,weighted")
                self.assertEqual(argv[argv.index("--seed")+1], "3000")
                self.assertEqual(argv[argv.index("--episodes")+1], "20")
                self.assertEqual(argv[argv.index("--port")+1], "8002")
            self.assertEqual(artifacts["transfer_selection.json"]["sigma"], .1)
            self.collect_synthetic_cells(args, artifacts)
            generated = collector.generate_manifests(artifacts["collector_plan.json"], args.out_dir/"generated",
                                                     plan_directory=args.out_dir)
            self.assertEqual(set(generated), {family+".json" for family in prepare.FAMILIES})
            for manifest in generated.values():
                report = scorer.score(manifest, args.out_dir/"generated")
                self.assertEqual(len(report["comparisons"]), 13)
                self.assertTrue(all(row["family_tests"] == 13 for row in report["comparisons"]))
            report = scorer.score(artifacts["score_manifest.json"], args.out_dir)
            self.assertEqual(len(report["comparisons"]), 39)
            legacy = next(row for row in report["comparisons"] if row["family"] == "legacy_vs_off")
            self.assertEqual(legacy["p"], .00390625)
            self.assertEqual(legacy["p_bonferroni"], .05078125)
            self.assertEqual(legacy["inference"], "not_resolved")

    def test_changed_source_or_old_development_seed_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            artifacts = prepare.build(args)
            self.write_artifacts(args, artifacts)
            self.collect_synthetic_cells(args, artifacts)
            path = args.out_dir/"compact/healthy/weighted.json"
            record = json.loads(path.read_text())
            record["args"]["seed"] = 2900
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                scorer.score(artifacts["score_manifest.json"], args.out_dir)

    def test_calibration_links_and_failed_reference_preserved(self):
        for mutation in ("healthy_hash", "sensitivity_hash", "fit_split", "rejected_qualification"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                args = self.fixture(directory)
                path = args.rejected_reference if mutation == "rejected_qualification" else args.reference_artifact
                record = json.loads(path.read_text())
                if mutation == "healthy_hash":
                    record["source"]["sha256"] = "0"*64
                elif mutation == "sensitivity_hash":
                    record["source"]["sensitivity"]["sha256"] = "0"*64
                elif mutation == "fit_split":
                    record["observer"]["fit_episode_indices"] = [6, 7, 8, 9]
                else:
                    record["qualification"]["allowed"] = True
                path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    prepare.build(args)

    def test_preflight_freezes_inputs_and_rejects_existing_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            artifacts = prepare.build(args)
            self.write_artifacts(args, artifacts)
            manifest = artifacts["launch_manifest.json"]
            for row in manifest["frozen_inputs"]:
                target = Path(row["remote"])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(row["local"], target)
            path = args.out_dir/"launch_manifest.json"
            self.assertEqual(prepare.verify_inputs(path)["prospective_cells"], 13)
            target = Path(manifest["frozen_inputs"][0]["remote"])
            original = target.read_bytes()
            target.write_bytes(original+b"changed")
            with self.assertRaisesRegex(ValueError, "changed frozen input"):
                prepare.verify_inputs(path)
            target.write_bytes(original)
            Path(manifest["jobs"][0]["out_dir"]).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "outcome directory already exists"):
                prepare.verify_inputs(path)

    def test_capture_survives_checkout_changes_and_rejects_mid_lock_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            artifacts = prepare.build(args)
            original = Path(artifacts["launch_manifest.json"]["frozen_inputs"][0]["local"])
            original_bytes = original.read_bytes()
            original.write_bytes(original_bytes+b"changed during preparation")
            with self.assertRaisesRegex(ValueError, "changed while preparing"):
                prepare.write_frozen_plan(args, artifacts)
            self.assertFalse(args.out_dir.exists())
            original.write_bytes(original_bytes)
            captured = prepare.write_frozen_plan(args, artifacts)
            self.assertEqual(captured["captured_inputs"], len(prepare.SNAPSHOT_FILES)+7)
            original.write_bytes(b"later checkout edits")
            manifest = json.loads((args.out_dir/"launch_manifest.json").read_text())
            for row in manifest["frozen_inputs"]:
                self.assertEqual(prepare.digest(row["local"]), row["sha256"])
                self.assertIn("frozen_inputs", Path(row["local"]).parts)
            self.assertEqual(Path(manifest["frozen_inputs"][0]["local"]).read_bytes(), original_bytes)

    def test_launcher_imports_from_isolated_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root/"snapshot/openpi"
            snapshot.mkdir(parents=True)
            for name in prepare.SNAPSHOT_FILES:
                shutil.copyfile(Path(prepare.__file__).parent/name, snapshot/name)
            result = subprocess.run([sys.executable, str(snapshot/"launch_followup_grid.py"), "--help"],
                cwd=root, env=dict(os.environ, PYTHONPATH=str(snapshot)), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--manifest", result.stdout)
            self.assertIn("--status", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
