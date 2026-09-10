"""Synthetic outcome/provenance tests; no simulators, policies, or real scores."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import collect_followup_results as collect
import score_joint_followup as scoring


def fixture(layout="panda", condition="joint0"):
    if layout == "panda":
        names = collect.PANDA_ARMS
        config = {name: dict(adapt=name != "off", baseline="dob" if name == "dob_translation" else "none",
                    corr_dims=[0, 1, 2], correction_scale=.5 if name == "legacy_half" else 1.,
                    freeze_after=30 if name == "legacy_hold30" else None) for name in names}
        args = dict(episodes=10, suite="libero_spatial", joint_fault="torque:5:5.0",
                    gamma=.08, dead=.008, norm_r=.15, clip=.3, profile="step", onset=0)
        schedule = [dict(episode=i, task=i, init=30, arms=names) for i in range(10)]
    else:
        names = collect.COMPOSITE_ARMS if layout == "aloha_composite" else collect.PHYSICAL_ARMS
        config = {name: dict(adapt=name != "off", baseline="kalman" if name == "kalman" else
                    "dob" if name == "dob" else "composite" if name.startswith("composite_") else "none",
                    norm_channels="corrected" if "corrected" in name else "all") for name in names}
        args = dict(episodes=10, seed=2700 if layout == "aloha_composite" else 2900,
                    task=collect.ALOHA_TASK, suite=collect.ALOHA_TASK, gamma=.08, dead=.002,
                    norm_r=.4, clip=.08, law="legacy", deadzone_mode="zero", damping=0.,
                    profile="step", onset=0, gain=None,
                    corr=list(range(6)) if layout == "aloha_composite" else list(range(6))+list(range(7,13)))
        if layout == "aloha_composite":
            for values in config.values():
                values.update(tracking_rate=0., tracking_strength=0.)
            for name, strength, rate in (("composite_001", .01, 686923.6043281124),
                ("composite_005", .05, 3434618.021640562),
                ("composite_010", .1, 6869236.043281124),
                ("composite_025", .25, 17173090.10820281)):
                config[name].update(tracking_strength=strength, tracking_rate=rate)
            schedule = [dict(episode=i, task=0, init=i, condition=condition, arms=names)
                        for i in range(10) for condition in ("healthy", "offset")]
        else:
            torque = None if condition == "healthy" else [16.,32.,16.,.2,1.,.4][int(condition[5:])%6]
            args.update(condition=condition, joint_fault=None if torque is None else
                        f"torque:{int(condition[5:])}:{torque}", tracking_rate=0., fault_vec=None)
            schedule = [dict(episode=i, task=0, init=i, arms=names) for i in range(10)]
    study = dict(schema_version=1, study_id="fixture_"+layout+"_"+condition, stage="dev", status="complete",
                 args=args, arm_configs=config, source_hashes={"runner.py":"b"*64}, schedule=schedule,
                 pairing=dict(mechanism="libero_set_init_state" if layout == "panda" else "aloha_seeded_reset",
                              valid_so_far=True, checked_states=len(schedule)*len(names), tolerance=1e-9))
    def arms():
        values = {}
        for index, name in enumerate(names):
            wins = 3 + index%5
            rows = []
            for i in range(10):
                row = dict(task=i if layout == "panda" else 0, init=30 if layout == "panda" else i,
                    ok=i<wins, initial_state=dict(sha256="a"*64, qpos=[1.,2.], qvel=[0.,0.],
                    pairing_valid=True, exact_hash_match=True, physics=dict(fields={"model.body_pos":"c"*64})))
                if layout != "panda":
                    row["actual_seed"] = args["seed"]+i
                rows.append(row)
            values[name] = dict(n=10, successes=wins, per_ep=rows,
                                f_hat=[[9.]*14]*10, traj=[[[9.]*14]*300]*10, f_true=[])
        return values
    if layout == "aloha_composite":
        study["conditions"] = {name: dict(arms=arms(), shared_control_id=f"fixture_{name}:off",
                    fault_vec=[0.]*14 if name == "healthy" else [0.02]*6+[0.]*8)
                               for name in ("healthy", "offset")}
    else:
        study.update(arms=arms(), shared_control_id="fixture:off")
        if layout == "aloha_joint":
            study.update(robot_joints=[dict(joint=0)], physical_fault=None if condition == "healthy" else dict(joint=0))
    return study


class CollectionTests(unittest.TestCase):
    def test_scalar_joint_addresses_are_not_removed_as_state_arrays(self):
        value = dict(robot_joints=[dict(qpos=8, qvel=8)],
                     initial_state=dict(qpos=[1., 2.], qvel=[0., 0.], sha256="a"*64))
        compact = collect.trim_state(value)
        self.assertEqual(compact["robot_joints"], value["robot_joints"])
        self.assertEqual(compact["initial_state"]["omitted_arrays"], dict(qpos=2, qvel=2))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def save(self, study, label="run"):
        path = self.root / (label+".json")
        path.write_text(json.dumps(study))
        return path

    def test_panda_compaction_preserves_outcomes_hashes_and_effective_arms(self):
        study = fixture()
        source = self.save(study)
        telemetry = self.root / "large.jsonl"
        telemetry.write_bytes(b"test telemetry\n" * 200000)
        out = self.root / "compact"
        summary = collect.collect(source, out, telemetry=telemetry, hash_telemetry=True)
        self.assertEqual(summary["source_study"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(summary["source_telemetry"]["sha256"], hashlib.sha256(telemetry.read_bytes()).hexdigest())
        self.assertEqual(summary["source_telemetry"]["hash_scope"], "whole_file")
        for name in collect.PANDA_ARMS[1:]:
            view = json.loads((out / (name+".json")).read_text())
            self.assertEqual(view["arms"]["adaptive"]["n"], 10)
            self.assertNotIn("traj", view["arms"]["adaptive"])
            self.assertNotIn("f_hat", view["arms"]["adaptive"])
            state = view["arms"]["adaptive"]["per_ep"][0]["initial_state"]
            self.assertEqual(state["sha256"], "a"*64)
            self.assertNotIn("qpos", state)
            self.assertEqual(state["physics"]["fields"]["model.body_pos"], "c"*64)
            self.assertEqual(view["source_study"], summary["source_study"])
        held = json.loads((out / "legacy_hold30.json").read_text())
        self.assertEqual(held["args"]["freeze_after"], 30)
        half = json.loads((out / "legacy_half.json").read_text())
        self.assertEqual(half["args"]["correction_scale"], .5)
        self.assertLess((out / "summary.json").stat().st_size, source.stat().st_size/5)

    def test_incomplete_export_is_diagnostic_and_cannot_make_manifest(self):
        study = fixture()
        study["status"] = "failed"
        study["pairing"]["valid_so_far"] = False
        source = self.save(study)
        out = self.root / "compact"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            collect.collect(source, out)
        self.assertFalse(out.exists())
        summary = collect.collect(source, out, allow_incomplete=True)
        self.assertFalse(summary["scoreable"])
        self.assertEqual(list(out.iterdir()), [out / "summary.json"])
        with self.assertRaisesRegex(ValueError, "incomplete/diagnostic"):
            collect._load_summary(out / "summary.json")

    def test_counts_missing_scenarios_and_unhashed_states_are_rejected(self):
        for issue in ("count", "duplicate", "missing", "state", "pairing"):
            study = fixture()
            arm = study["arms"]["off"]
            if issue == "count": arm["successes"] += 1
            elif issue == "duplicate": arm["per_ep"][-1] = copy.deepcopy(arm["per_ep"][-2])
            elif issue == "missing": arm["per_ep"].pop(); arm["n"] -= 1
            elif issue == "state": arm["per_ep"][0]["initial_state"].pop("sha256")
            else: study["pairing"]["valid_so_far"] = False
            with self.subTest(issue=issue), self.assertRaises(ValueError):
                collect.collect(self.save(study, issue), self.root / (issue+"_out"))

    def test_unique_output_refusal_leaves_previous_collection(self):
        path = self.save(fixture())
        out = self.root / "compact"
        collect.collect(path, out)
        previous = (out / "summary.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "exists"):
            collect.collect(path, out)
        self.assertEqual((out / "summary.json").read_bytes(), previous)

    def test_composite_registered_families_round_trip_through_real_scorer(self):
        out = self.root / "compact"
        study = fixture("aloha_composite")
        collect.collect(self.save(study), out)
        plan = collect.preset_plan("aloha_composite", [out / "summary.json"], "lock 2 at hash")
        manifests = collect.generate_manifests(plan, self.root / "manifests")
        self.assertEqual(sorted(len(x["comparisons"]) for x in manifests.values()), [8,14,16])
        for manifest in manifests.values():
            report = scoring.score(manifest, self.root / "manifests")
            self.assertEqual(report["n_scenarios"], 10)
            self.assertTrue(all(row["family_tests"] == len(manifest["comparisons"])
                                for row in report["comparisons"]))
            self.assertTrue(all(row["inference"].startswith("exploratory_") for row in report["comparisons"]))

    def test_all_thirteen_physical_cells_are_required_and_score_39_26(self):
        summaries = []
        for condition in ["healthy"] + [f"joint{i}" for i in range(12)]:
            out = self.root / condition
            collect.collect(self.save(fixture("aloha_joint", condition), condition), out)
            summaries.append(out / "summary.json")
        with self.assertRaisesRegex(ValueError, "all thirteen"):
            collect.preset_plan("aloha_joint", summaries[:-1], "lock 3")
        plan = collect.preset_plan("aloha_joint", summaries, "lock 3")
        manifests = collect.generate_manifests(plan, self.root / "manifests")
        self.assertEqual(sorted(len(x["comparisons"]) for x in manifests.values()), [26,39])
        for manifest in manifests.values():
            report = scoring.score(manifest, self.root / "manifests")
            self.assertEqual(len(report["comparisons"]), len(manifest["comparisons"]))

    def test_changed_registered_parameters_or_arm_meaning_rejected(self):
        for issue in ("seed", "gamma", "corr", "baseline", "tracking_rate"):
            study = fixture("aloha_composite")
            if issue == "seed":
                study["args"]["seed"] += 1
                for cell in study["conditions"].values():
                    for arm in cell["arms"].values():
                        for row in arm["per_ep"]: row["actual_seed"] += 1
            elif issue == "gamma": study["args"]["gamma"] = .1
            elif issue == "corr": study["args"]["corr"] = [0,1,2]
            else: study["arm_configs"]["composite_001"][issue] = "none" if issue == "baseline" else 99.
            out = self.root / issue
            collect.collect(self.save(study, issue), out)
            plan = collect.preset_plan("aloha_composite", [out / "summary.json"], "lock 2")
            with self.subTest(issue=issue), self.assertRaises(ValueError):
                collect.generate_manifests(plan, self.root / (issue+"_manifests"))

    def test_custom_panda_plan_and_wrong_cohort_rejection(self):
        out = self.root / "compact"
        collect.collect(self.save(fixture()), out)
        plan = collect.example_plan()
        plan["cells"][0]["summary"] = str(out / "summary.json")
        manifests = collect.generate_manifests(plan, self.root / "manifests")
        self.assertEqual(sorted(len(x["comparisons"]) for x in manifests.values()), [5,6])
        for manifest in manifests.values():
            scoring.score(manifest, self.root / "manifests")
        plan["expected_keys"][0][1] = 31
        with self.assertRaisesRegex(ValueError, "planned scenarios"):
            collect.generate_manifests(plan, self.root / "wrong")
        self.assertFalse((self.root / "wrong").exists())

    def test_tampered_view_cannot_disagree_with_authoritative_collection(self):
        out = self.root / "compact"
        collect.collect(self.save(fixture()), out)
        path = out / "legacy_translation.json"
        view = json.loads(path.read_text())
        view["arms"]["adaptive"]["per_ep"][0]["ok"] = False
        path.write_text(json.dumps(view))
        plan = collect.example_plan()
        plan["cells"][0]["summary"] = str(out / "summary.json")
        with self.assertRaisesRegex(ValueError, "outcome differs"):
            collect.generate_manifests(plan, self.root / "manifests")


if __name__ == "__main__":
    unittest.main()
