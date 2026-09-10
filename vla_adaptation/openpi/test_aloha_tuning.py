"""Tuning-runner integrity and actual ALOHA episode parameter-plumbing tests."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import run_aloha_tuning as runner
from test_aloha_joint_followup import FakeEnvironment


def plan_fixture(**updates):
    plan = dict(schema_version=1, bank_sha256="0"*64, stage="tuning", seed=4000,
        episodes=2, max_steps=4, correction_indices=list(range(6)),
        conditions=[dict(name="healthy", kind="offset", fault_vec=[0.]*14),
                    dict(name="torque0", kind="torque", joint=0, torque=16.)],
        candidate_names=["off", "kf", "composite"], excluded_seeds=[3100, 3101])
    plan.update(updates)
    return runner.normalize_plan(plan)


def validated_fixture():
    model = dict(A=(np.eye(6)*.5).tolist(), B=(np.eye(6)*.5).tolist(),
        offset=np.zeros(6).tolist(), metric=np.eye(6).tolist(),
        state_indices=list(range(6)), command_indices=list(range(6)))
    shared = dict(Q=(np.eye(14)*1e-5).tolist(), R=(np.eye(14)*1e-3).tolist(),
                  P0=np.diag(np.linspace(2e-4, 9e-4, 14)).tolist(), clip=.06,
                  damping=.2, tracking_rate=0.)
    return dict(observer=dict(W=np.zeros((14, 8)).tolist(), M=np.eye(14).tolist()),
        reference_model=model, qualification=dict(reference=True, candidates=True),
        candidates=dict(off=dict(family="off"), kf=dict(family="kalman", **shared),
            composite=dict(dict(family="composite", **shared), tracking_rate=7.)),
        sources={}, correction_indices=list(range(6)), dt=.02,
        calibration_seeds=[3100, 3101], screened_steps=300)


class ValidationTests(unittest.TestCase):
    def test_correction_stream_preserves_chunks_and_rejects_duplicate_steps(self):
        raw = io.StringIO()
        stream = runner.CorrectionTelemetry(raw, "healthy/kf", 2, 14)
        row = json.dumps(dict(type="step", arm="healthy/kf", episode=2, t=0,
                              correction=[.1]+[0.]*13))+"\n"
        stream.write(row[:11])
        self.assertEqual(stream.steps, 0)
        stream.write(row[11:])
        self.assertEqual(raw.getvalue(), row)
        self.assertEqual(stream.steps, 1)
        self.assertEqual(stream.energy, .1**2)
        with self.assertRaisesRegex(RuntimeError, "episode/arm/order"):
            stream.write(row)

    def test_duplicate_names_unknown_fields_and_seed_overlap_fail(self):
        for update, message in [
            (dict(candidate_names=["off", "kf", "kf"]), "unique"),
            (dict(seed=3100), "overlap"),
            (dict(episodes=True), "integer"),
            (dict(hidden_gain=10), "unknown"),
            (dict(conditions=[dict(name="x", kind="torque", joint=12, torque=1)]), "0..11")]:
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, message):
                plan_fixture(**update)

    def test_candidate_covariance_and_qualification_are_not_silent_defaults(self):
        for field, value, message in [("P0", np.zeros((14, 14)).tolist(), "positive definite"),
                ("Q", np.eye(13).tolist(), "square"), ("R", None, "square"),
                ("qualified", False, "not qualified"), ("tracking_rate", 1., "only composite")]:
            prepared = validated_fixture()
            prepared["candidates"]["kf"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                runner.select_candidates(plan_fixture(), prepared)

    def test_duplicate_effective_configs_rejected(self):
        prepared = validated_fixture()
        prepared["candidates"]["another_kf"] = copy.deepcopy(prepared["candidates"]["kf"])
        with self.assertRaisesRegex(ValueError, "duplicate effective"):
            runner.select_candidates(plan_fixture(candidate_names=["off", "kf", "another_kf"]), prepared)

    def test_dynamic_oracle_is_rejected(self):
        prepared = validated_fixture()
        prepared["candidates"]["oracle"] = dict(family="oracle")
        plan = plan_fixture(candidate_names=["off", "oracle"],
            conditions=[dict(name="ramp", kind="offset", fault_vec=[.01]*14, profile="ramp")])
        with self.assertRaisesRegex(ValueError, "constant faults"):
            runner.select_candidates(plan, prepared)

    def test_confirmation_requires_hash_bound_selection(self):
        with self.assertRaisesRegex(ValueError, "selection_record"):
            plan_fixture(stage="confirmation")
        plan = plan_fixture(stage="confirmation", selection_record=dict(path="selection.json", sha256="a"*64))
        self.assertEqual(plan["selection_record"]["sha256"], "a"*64)

    def test_runtime_scope_rejects_unqualified_mask_horizon_dt_and_hidden_overlap(self):
        for field, value, message in [("correction_indices", [0, 1], "mask"),
                ("dt", .04, "interval"), ("screened_steps", 1, "horizon"),
                ("calibration_seeds", [4000], "overlap")]:
            prepared = validated_fixture()
            prepared[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                runner.validate_runtime(plan_fixture(), prepared)


class LifecycleTests(unittest.TestCase):
    def setup_study(self, directory, mismatch=False, source_change=False, exception=False, torque_only=False):
        args = SimpleNamespace(plan=Path(directory)/"plan.json", candidate_bank=Path(directory)/"bank.json",
                               out_dir=Path(directory), telemetry=Path(directory)/"telemetry.jsonl")
        plan = plan_fixture()
        if torque_only:
            plan["conditions"] = [plan["conditions"][1]]
        env = FakeEnvironment(mismatch=mismatch)
        aloha = SimpleNamespace(env=env, seed=plan["seed"])
        aloha.reset = lambda ep: aloha.env.reset(seed=aloha.seed+ep)[0]
        calls = []
        source = Path(directory)/"frozen.txt"
        source.write_text("original source")
        hashes = {str(source): runner.digest(source)}
        def episode(client, ep, **kwargs):
            calls.append(kwargs)
            client.reset(ep)
            trace = []
            for t in range(2):
                client.env.step(np.zeros(14))
                if exception:
                    raise RuntimeError("simulated rollout exception")
                kwargs["telemetry"].write(json.dumps(dict(type="step", arm=kwargs["arm"],
                    episode=ep, t=t, correction=[0.]*14))+"\n")
                trace.append(dict(f_hat=[.01]*14, f_true=[0.]*14))
            if source_change:
                source.write_text("modified after inference")
            return bool(kwargs["adapt"]), np.full(14, .01), trace
        law = SimpleNamespace(NJ=14, K_FIR=6, DT=.02, HORIZON=10, MAX_STEPS=300,
            TASK="gym_aloha/AlohaTransferCube-v0", episode=episode,
            write_telemetry=lambda stream, row: stream.write(json.dumps(row, default=runner.json_default)+"\n"))
        return args, aloha, env, law, plan, validated_fixture(), hashes, calls

    def test_shared_controls_full_pairing_interleaving_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            args, aloha, env, law, plan, validated, hashes, calls = self.setup_study(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                study = runner.execute(args, aloha, law, plan, validated, hashes)
            self.assertEqual(study["status"], "complete")
            self.assertTrue(study["completed_source_recheck"])
            self.assertEqual(study["pairing"]["checked_states"], 12)
            self.assertFalse(study["pairing"]["policy_rng_pinned"])
            self.assertEqual(len(calls), 12)
            self.assertIs(aloha.env, env)
            self.assertEqual(law.MAX_STEPS, 300)
            for condition in ("healthy", "torque0"):
                views = [json.loads((Path(directory)/f"{condition}_{name}.json").read_text()) for name in ("kf", "composite")]
                self.assertEqual(views[0]["shared_control_id"], views[1]["shared_control_id"])
                self.assertEqual(views[0]["arms"]["frozen_faulted"], views[1]["arms"]["frozen_faulted"])
                for arm in study["conditions"][condition]["arms"].values():
                    self.assertEqual([row["actual_seed"] for row in arm["per_ep"]], [4000, 4001])
                    self.assertEqual(arm["diagnostics"][0]["physical_steps"], 2)
                    self.assertEqual(arm["diagnostics"][0]["applied_correction_steps"], 2)
                    self.assertEqual(arm["diagnostics"][0]["applied_correction_energy"], 0.)
                    self.assertEqual(arm["diagnostics"][0]["force_limit"][0], 2)
            for call in calls:
                self.assertNotIn("joint_fault", call)
                np.testing.assert_array_equal(call["fvec"], np.zeros(14))
                self.assertIsNone(call["f_init"])
                self.assertIsNone(call["freeze_after"])
            kf = next(call for call in calls if call["baseline"] == "composite" and call["tracking_rate"] == 0)
            np.testing.assert_array_equal(kf["initial_covariance"], validated["candidates"]["kf"]["P0"])
            self.assertEqual(kf["damping"], .2)
            starts = [json.loads(line) for line in args.telemetry.read_text().splitlines()
                      if json.loads(line)["type"] == "scenario_start"]
            self.assertEqual([(row["episode"], row["condition"], row["arm"]) for row in starts],
                [(batch["episode"], batch["condition"], name) for batch in study["schedule"] for name in batch["arms"]])
            np.testing.assert_array_equal(env._env.physics.data.qfrc_applied, np.zeros(23))

    def test_pairing_drift_aborts_before_second_policy_rollout_and_restores(self):
        with tempfile.TemporaryDirectory() as directory:
            args, aloha, env, law, plan, validated, hashes, calls = self.setup_study(directory, mismatch=True, torque_only=True)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "pairing failed"):
                runner.execute(args, aloha, law, plan, validated, hashes)
            study = json.loads((Path(directory)/"study.json").read_text())
            self.assertEqual(study["status"], "failed")
            self.assertFalse(study["pairing"]["valid_so_far"])
            self.assertEqual(len(env.forces), 2)
            self.assertIs(aloha.env, env)
            np.testing.assert_array_equal(env._env.physics.data.qfrc_applied, np.zeros(23))

    def test_source_drift_retains_raw_telemetry_but_does_not_count_a_valid_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            args, aloha, env, law, plan, validated, hashes, calls = self.setup_study(directory, source_change=True)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "source changed"):
                runner.execute(args, aloha, law, plan, validated, hashes)
            study = json.loads((Path(directory)/"study.json").read_text())
            self.assertEqual(study["status"], "failed")
            self.assertEqual(sum(arm["n"] for cohort in study["conditions"].values() for arm in cohort["arms"].values()), 0)
            self.assertIn('"type": "physical_step"', args.telemetry.read_text())
            self.assertEqual(law.MAX_STEPS, 300)

    def test_rollout_exception_restores_torque_and_max_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            args, aloha, env, law, plan, validated, hashes, calls = self.setup_study(directory, exception=True, torque_only=True)
            with self.assertRaisesRegex(RuntimeError, "rollout exception"):
                runner.execute(args, aloha, law, plan, validated, hashes)
            self.assertIs(aloha.env, env)
            np.testing.assert_array_equal(env._env.physics.data.qfrc_applied, np.zeros(23))
            self.assertEqual(law.MAX_STEPS, 300)

    def test_profiled_torque_uses_declared_step_time_and_never_accumulates(self):
        import aloha_joint_fault
        env = FakeEnvironment()
        client = SimpleNamespace(env=env)
        condition = dict(kind="torque", joint=1, torque=32., onset=1, profile="ramp", prof_p=2.)
        with runner.condition_context(client, condition, aloha_joint_fault):
            client.env.reset(seed=1)
            for _ in range(5):
                client.env.step(np.zeros(14))
        self.assertEqual([float(force[1]) for force in env.forces], [0., 0., 16., 32., 32.])
        np.testing.assert_array_equal(env._env.physics.data.qfrc_applied, np.zeros(23))
        self.assertIs(client.env, env)

    def test_existing_outputs_rejected_before_bank_validation_or_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank = root/"bank.json"
            bank.write_text("{}")
            plan = plan_fixture(bank_sha256=runner.digest(bank))
            plan_path = root/"plan.json"
            plan_path.write_text(json.dumps(plan))
            (root/"study.json").write_text("preserve")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                runner.main(["--plan", str(plan_path), "--candidate-bank", str(bank), "--out-dir", directory])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual((root/"study.json").read_text(), "preserve")


class ActualEpisodeTests(unittest.TestCase):
    def test_real_episode_executes_every_family_and_keeps_torque_truth_out_of_observer(self):
        import aloha_adapt
        with tempfile.TemporaryDirectory() as directory:
            prepared = validated_fixture()
            for family in ("legacy", "dob", "rls", "integral_calibrated", "oracle"):
                prepared["candidates"][family] = dict(family=family)
            plan = plan_fixture(candidate_names=list(prepared["candidates"]), episodes=1)
            env = FakeEnvironment()
            client = SimpleNamespace(env=env, seed=plan["seed"],
                client=SimpleNamespace(infer=lambda _: {"actions": np.zeros((10, 14))}),
                policy_obs=lambda obs: obs)
            client.reset = lambda episode: client.env.reset(seed=client.seed+episode)[0]
            args = SimpleNamespace(plan=Path(directory)/"plan.json", candidate_bank=Path(directory)/"bank.json",
                out_dir=Path(directory), telemetry=Path(directory)/"telemetry.jsonl")
            with contextlib.redirect_stdout(io.StringIO()):
                study = runner.execute(args, client, aloha_adapt, plan, prepared, {})
            self.assertEqual(study["status"], "complete")
            self.assertEqual(study["pairing"]["checked_states"], 16)
            records = [json.loads(line) for line in args.telemetry.read_text().splitlines()]
            starts = [row for row in records if row["type"] == "step" and row["t"] == 0]
            self.assertEqual(len(starts), 16)
            for row in starts:
                np.testing.assert_array_equal(row["f_hat_before"], np.zeros(14))
                np.testing.assert_array_equal(row["f_true"], np.zeros(14))
            oracle = json.loads((Path(directory)/"torque0_oracle.json").read_text())
            self.assertTrue(oracle["args"]["oracle_diagnostic"])
            self.assertEqual(oracle["args"]["static_corr"], [.02]+[0.]*13)
            actual = [row for row in starts if row["arm"] == "torque0/oracle"][0]
            self.assertEqual(actual["correction"], [-.02]+[-0.]*13)
            diagnostic = study["conditions"]["torque0"]["arms"]["oracle"]["diagnostics"][0]
            self.assertEqual(diagnostic["applied_correction_steps"], plan["max_steps"])
            self.assertEqual(diagnostic["applied_correction_energy"], plan["max_steps"] * .02**2)
            for condition, cohort in study["conditions"].items():
                for name, arm in cohort["arms"].items():
                    rows = [row for row in records if row["type"] == "step" and row["arm"] == f"{condition}/{name}"]
                    measured = sum(float(np.asarray(row["correction"]) @ np.asarray(row["correction"])) for row in rows)
                    self.assertEqual(arm["diagnostics"][0]["applied_correction_energy"], measured)

    def test_matrix_P0_changes_first_kalman_gain_and_resets_each_episode(self):
        import aloha_adapt
        import composite_observer
        from test_aloha_composite import ContractingServo, calibration, recorded_episode
        servo = ContractingServo(length=4)
        options = calibration(servo)
        P0 = np.diag(np.linspace(1e-5, 5e-5, 14))
        options.update(initial_covariance=P0, baseline="composite", tracking_rate=0., damping=.3)
        _, rows = recorded_episode(servo, **options)
        _, repeated = recorded_episode(servo, **options)
        self.assertEqual(rows, repeated)
        _, expected = composite_observer.composite_step(np.zeros(14), rows[0]["r"], options["M"],
            Q=options["kf_q"]/aloha_adapt.DT, R=options["kf_r"], dt=aloha_adapt.DT,
            initial_covariance=P0, damping=.3, clip=options["clip"])
        np.testing.assert_allclose(rows[0]["gain"], expected["gain"], rtol=0, atol=1e-16)
        np.testing.assert_allclose(rows[0]["f_hat"], expected["prediction_increment"], rtol=0, atol=1e-16)
        self.assertEqual(rows[0]["estimator_state"]["updates"], 1)


if __name__ == "__main__":
    unittest.main()
