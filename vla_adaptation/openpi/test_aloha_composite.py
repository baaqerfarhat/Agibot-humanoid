"""Exercise ALOHA's actual episode and CLI plumbing without a simulator/server.

The fake plant is a contracting 14-dimensional position servo. These are wiring
and control-timing checks, not evidence about the real ALOHA dynamics.
Run: python3 -m unittest discover -s openpi -p test_aloha_composite.py
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

import aloha_adapt as aloha
import run_aloha_followup as followup


class ContractingServo:
    """An exact affine servo; policy chunks depend only on time and reset seed."""

    seed = 4100

    def __init__(self, length=35):
        self.env = self.client = self
        self.length = length
        self.A = np.diag(np.linspace(0.35, 0.65, aloha.NJ))
        self.B = np.eye(aloha.NJ) - self.A
        self.offset = np.linspace(-0.0005, 0.0005, aloha.NJ)

    def reset(self, ep):
        self.t = 0
        self.ep = ep
        self.q = np.linspace(-0.12, 0.13, aloha.NJ) + ep * 0.025
        return {"agent_pos": self.q.copy()}

    def policy_obs(self, obs):
        return obs

    def infer(self, obs):
        # Every arm sees identical raw policy commands, even after corrections.
        times = np.arange(self.t, self.t + aloha.HORIZON)[:, None]
        channels = np.arange(aloha.NJ)[None, :]
        return {"actions": 0.07 * np.sin(0.13 * times + 0.17 * channels)
                + 0.01 * self.ep}

    def step(self, command):
        self.q = self.A @ self.q + self.B @ command + self.offset
        self.t += 1
        return {"agent_pos": self.q.copy()}, 0, self.t == self.length, False, {}

    def reference_model(self):
        return dict(A=self.A[:6, :6], B=self.B[:6, :6], offset=self.offset[:6],
                    metric=np.eye(6), state_indices=list(range(6)),
                    command_indices=list(range(6)))


def calibration(servo):
    # Truncated exact impulse response, with the omitted DC tail placed on the
    # oldest command. It gives nontrivial FIR residuals even for this exact servo.
    alpha = np.diag(servo.A)
    W = np.zeros((aloha.NJ, aloha.K_FIR + 2))
    for k in range(aloha.K_FIR + 1):
        W[:, k] = (1 - alpha) * alpha ** k
    W[:, aloha.K_FIR] += alpha ** (aloha.K_FIR + 1)
    W[:, -1] = servo.offset / (1 - alpha)
    M = np.eye(aloha.NJ)
    # Dense SPD covariance exercises cross-channel state propagation, including
    # corrected to uncorrected coordinates. All baselines get the same matrices.
    mixing = np.eye(aloha.NJ) + 0.04 * np.ones((aloha.NJ, aloha.NJ))
    R = 0.0002 * mixing @ mixing.T
    Q = 0.08 ** 2 / (1 - 0.08) * R
    return dict(W=W, M=M, M_inv=M.copy(), kf_q=Q, kf_r=R,
                reference_model=servo.reference_model(), corr=list(range(6)),
                adapt=True, clip=0.3)


def recorded_episode(servo, ep=0, **kwargs):
    stream = io.StringIO()
    result = aloha.episode(servo, ep, telemetry=stream, **kwargs)
    return result, [json.loads(line) for line in stream.getvalue().splitlines()]


class EpisodeTests(unittest.TestCase):
    def test_tuned_initial_covariance_controls_startup_and_resets(self):
        servo = ContractingServo(length=4)
        options = calibration(servo)
        # M=I and Q=.08²/(1-.08)*R: P0=.08*R is the steady posterior.
        initial = 0.08 * options["kf_r"]
        _, rows = recorded_episode(servo, baseline="composite", tracking_rate=0,
                                   initial_covariance=initial, **options)
        _, repeated = recorded_episode(servo, ep=1, baseline="composite", tracking_rate=0,
                                       initial_covariance=initial, **options)
        for row in rows + repeated:
            np.testing.assert_allclose(row["gain"], 0.08*np.eye(aloha.NJ), atol=2e-15)
        _, diffuse = recorded_episode(servo, baseline="composite", tracking_rate=0,
                                      **options)
        self.assertGreater(np.min(np.diag(diffuse[0]["gain"])), 0.99)
        np.testing.assert_array_equal(initial, 0.08 * options["kf_r"])

    def test_rls_initial_covariance_is_not_silently_ignored(self):
        servo = ContractingServo(length=1)
        options = calibration(servo)
        for p0 in (0.01, 1.0):
            _, rows = recorded_episode(servo, baseline="rls", rls_p0=p0,
                                       rls_lambda=0.95, **options)
            predicted = p0 / 0.95
            np.testing.assert_allclose(rows[0]["gain"],
                predicted/(predicted+1)*np.eye(aloha.NJ), rtol=0, atol=2e-15)

    def test_zero_tracking_and_damping_match_full_kalman_trajectory(self):
        servo = ContractingServo()
        options = calibration(servo)
        # Include uncorrected disturbances and a time-varying injection, so this
        # also detects accidental observation masking or a stale state reset.
        options.update(fvec=np.linspace(-0.018, 0.026, aloha.NJ), profile="ramp",
                       prof_p=7, onset=3)
        kalman, krows = recorded_episode(servo, baseline="kalman", **options)
        composite, crows = recorded_episode(servo, baseline="composite",
                                            tracking_rate=0, damping=0, **options)
        self.assertEqual(len(krows), servo.length)
        self.assertEqual(kalman[0], composite[0])
        np.testing.assert_allclose(kalman[1], composite[1], rtol=0, atol=2e-15)
        for krow, crow in zip(krows, crows):
            for key in ("raw_action", "correction", "command", "measured", "r",
                        "f_hat_before", "f_hat", "gain"):
                np.testing.assert_allclose(krow[key], crow[key], rtol=0, atol=2e-15)
            np.testing.assert_allclose(krow["estimator_state"]["covariance"],
                                       crow["estimator_state"]["covariance"],
                                       rtol=0, atol=2e-15)
            np.testing.assert_array_equal(crow["tracking_increment"], np.zeros(aloha.NJ))
        self.assertEqual(crows[-1]["estimator_state"]["updates"], servo.length)

    def test_tracking_reads_actual_minus_reference_separately_from_residual(self):
        servo = ContractingServo(length=1)
        options = calibration(servo)
        # An intentionally wrong FIR gives a large healthy residual. It must not
        # become tracking error: the independent exact healthy reference is right.
        options["W"] = np.zeros_like(options["W"])
        _, healthy = recorded_episode(servo, baseline="composite",
                                      tracking_rate=1000, **options)
        row = healthy[0]
        self.assertGreater(np.linalg.norm(row["r"]), 0.05)
        np.testing.assert_allclose(row["tracking_error"], np.zeros(6), atol=2e-17)
        np.testing.assert_allclose(row["tracking_increment"], np.zeros(aloha.NJ), atol=1e-17)
        self.assertGreater(np.linalg.norm(row["prediction_increment"]), 0.05)

        fault = np.zeros(aloha.NJ)
        fault[0] = 0.02
        _, faulted = recorded_episode(servo, baseline="composite", tracking_rate=1000,
                                      fvec=fault, **options)
        row = faulted[0]
        expected_error = (servo.B @ fault)[:6]
        np.testing.assert_allclose(row["tracking_error"], expected_error, atol=2e-17)
        self.assertGreater(row["tracking_increment"][0], 0)
        np.testing.assert_allclose(row["tracking_increment"],
                                   np.asarray(row["tracking_gain"]) @ expected_error,
                                   atol=2e-17)
        # The estimate is updated AFTER the command that generated this error.
        np.testing.assert_array_equal(row["correction"], np.zeros(aloha.NJ))

    def test_mask_prevents_estimated_uncorrected_channels_changing_actions(self):
        servo = ContractingServo()
        options = calibration(servo)
        _, rows = recorded_episode(servo, baseline="composite", tracking_rate=1000,
                                   fvec=np.full(aloha.NJ, 0.015), **options)
        self.assertGreater(np.linalg.norm(np.asarray(rows[-1]["f_hat"])[6:]), 0.005)
        self.assertGreater(max(np.linalg.norm(row["tracking_increment"]) for row in rows), 0)
        for row in rows:
            correction = np.asarray(row["correction"])
            np.testing.assert_array_equal(correction[6:], np.zeros(8))
            np.testing.assert_array_equal(correction[:6], -np.asarray(row["f_hat_before"])[:6])
            np.testing.assert_array_equal(np.asarray(row["nominal_command"])[6:],
                                          np.asarray(row["raw_action"])[6:])

    def test_reference_and_observer_history_reset_at_episode_boundary(self):
        servo = ContractingServo(length=12)
        options = calibration(servo)
        first, rows1 = recorded_episode(servo, baseline="composite", tracking_rate=1000,
                                        fvec=np.full(aloha.NJ, 0.02), **options)
        _, rows2 = recorded_episode(servo, ep=1, baseline="composite", tracking_rate=1000,
                                    fvec=np.full(aloha.NJ, 0.02), **options)
        repeated, rows3 = recorded_episode(servo, baseline="composite", tracking_rate=1000,
                                           fvec=np.full(aloha.NJ, 0.02), **options)
        self.assertEqual(rows1, rows3)
        np.testing.assert_array_equal(first[1], repeated[1])
        model = options["reference_model"]
        expected = (model["A"] @ np.asarray(rows2[0]["joint_before"])[:6]
                    + model["B"] @ np.asarray(rows2[0]["raw_action"])[:6] + model["offset"])
        np.testing.assert_array_equal(rows2[0]["reference_position"], expected)
        self.assertEqual(rows2[0]["estimator_state"]["updates"], 1)
        np.testing.assert_array_equal(rows2[0]["f_hat_before"], np.zeros(aloha.NJ))
        # Reset must apply even when a caller deliberately carries only theta.
        _, carried = recorded_episode(servo, ep=1, baseline="composite", tracking_rate=1000,
                                       f_init=first[1], **options)
        np.testing.assert_array_equal(carried[0]["reference_position"], expected)
        self.assertEqual(carried[0]["estimator_state"]["updates"], 1)
        np.testing.assert_array_equal(carried[0]["f_hat_before"], first[1])

    def test_freeze_holds_estimate_while_reference_keeps_advancing(self):
        servo = ContractingServo(length=9)
        options = calibration(servo)
        _, rows = recorded_episode(servo, baseline="composite", tracking_rate=1000,
                                   fvec=np.full(aloha.NJ, 0.02), freeze_after=3, **options)
        model = options["reference_model"]
        reference = np.asarray(rows[0]["joint_before"])[:6]
        for t, row in enumerate(rows):
            reference = (model["A"] @ reference + model["B"] @ np.asarray(row["raw_action"])[:6]
                         + model["offset"])
            np.testing.assert_array_equal(row["reference_position"], reference)
            self.assertEqual(row["update_applied"], t < 3)
            if t >= 3:
                np.testing.assert_array_equal(row["f_hat"], rows[2]["f_hat"])


class CliQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_composite_validation import prepared_artifact
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.bound_artifact = prepared_artifact(temporary.name)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        servo = ContractingServo()
        options = calibration(servo)
        self.openloop = self.directory / "openloop.json"
        self.openloop.write_text(json.dumps(dict(M=options["M"].tolist())))
        self.artifact_path = self.directory / "reference.json"
        self.artifact = copy.deepcopy(self.bound_artifact)
        self.tracking_rate = self.artifact["tracking"]["candidates"][1]["tracking_rate"]
        self.options = options

    def invoke(self, artifact=None, extra_args=()):
        self.artifact_path.write_text(json.dumps(artifact or self.artifact, default=aloha.json_value))
        args = ["aloha_adapt.py", "run", "--out", str(self.directory / "unused.json"),
                "--log", str(self.directory / "unused_log.json"), "--openloop", str(self.openloop),
                "--baseline", "composite", "--reference-artifact", str(self.artifact_path),
                "--corr-joints", "0,1,2,3,4,5", "--tracking-rate", str(self.tracking_rate)] + list(extra_args)
        stderr = io.StringIO()
        with mock.patch.object(sys, "argv", args), mock.patch.object(aloha, "fit_plant",
                return_value=(self.options["W"], np.ones(aloha.NJ))), \
                mock.patch.object(aloha, "run_cli") as run_cli, contextlib.redirect_stderr(stderr):
            try:
                aloha.main()
            except SystemExit as error:
                self.assertEqual(error.code, 2)
                run_cli.assert_not_called()
                return stderr.getvalue(), None
        return stderr.getvalue(), run_cli.call_args

    def test_qualified_configuration_passes_exact_shared_calibration(self):
        error, call = self.invoke()
        self.assertEqual(error, "")
        self.assertIsNotNone(call)
        for key in ("W", "M", "Q", "R"):
            np.testing.assert_array_equal(call.kwargs["observer"][key], self.artifact["observer"][key])

    def test_unqualified_reference_is_rejected_before_simulator(self):
        artifact = copy.deepcopy(self.artifact)
        artifact["qualification"] = dict(allowed=False, reasons=["held-out error too large"])
        error, _ = self.invoke(artifact)
        self.assertIn("failed qualification", error)

    def test_mask_rate_and_damping_must_match_qualification(self):
        for extra, expected in [(("--corr-joints", "0,1,2"), "correction mask"),
                                (("--tracking-rate", "1001"), "qualified calibration grid"),
                                (("--damping", "0.1"), "dt/damping")]:
            with self.subTest(extra=extra):
                error, _ = self.invoke(extra_args=extra)
                self.assertIn(expected, error)

    def test_unqualified_candidate_is_rejected(self):
        artifact = copy.deepcopy(self.artifact)
        artifact["tracking"]["candidates"][1]["qualified_for_experiment"] = False
        error, _ = self.invoke(artifact)
        self.assertIn("qualified calibration grid", error)

    def test_cli_rejects_stale_model_and_covariance_before_simulator(self):
        for group, field in (("model", "B"), ("observer", "Q")):
            with self.subTest(group=group, field=field):
                artifact = copy.deepcopy(self.artifact)
                artifact[group][field] = (np.asarray(artifact[group][field])*2).tolist()
                error, call = self.invoke(artifact)
                self.assertIsNone(call)
                self.assertIn("qualification mismatch: "+group+"."+field, error)

    def test_weighted_baseline_does_not_require_composite_sources_or_qualification(self):
        artifact = copy.deepcopy(self.artifact)
        artifact["qualification"]["allowed"] = False
        artifact.pop("source")
        error, call = self.invoke(artifact, extra_args=("--baseline", "weighted_dob", "--tracking-rate", "0"))
        self.assertEqual(error, "")
        self.assertIsNotNone(call)


class FollowupStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_composite_validation import prepared_artifact
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.bound_artifact = prepared_artifact(temporary.name)

    class PhysicalServo(ContractingServo):
        def __init__(self, mismatch=False):
            super().__init__(length=4)
            self.unwrapped = self
            self._env = SimpleNamespace(physics=SimpleNamespace(data=SimpleNamespace(
                qpos=np.zeros(21), qvel=np.zeros(20))))
            self.resets = 0
            self.mismatch = mismatch

        def reset(self, ep):
            observation = super().reset(ep)
            self._env.physics.data.qpos[:14] = self.q
            self._env.physics.data.qpos[14:] = ep * 0.01
            self._env.physics.data.qvel[:] = 0
            if self.mismatch and self.resets:
                self._env.physics.data.qpos[-1] += 0.001  # cube only, absent in agent_pos
            self.resets += 1
            return observation

    def artifact(self, servo):
        return copy.deepcopy(self.bound_artifact)

    def arguments(self, directory):
        args = followup.parser().parse_args(["--out-dir", directory, "--episodes", "2",
                    "--arms", "off,kalman,composite_001", "--max-steps", "4"])
        args.arm_names = args.arms.split(",")
        args.condition_names = args.conditions.split(",")
        args.telemetry = args.out_dir / "telemetry.jsonl"
        return args

    def test_actual_episode_loop_interleaves_and_shares_off_once(self):
        with tempfile.TemporaryDirectory() as directory:
            servo = self.PhysicalServo()
            args = self.arguments(directory)
            servo.seed = args.seed
            previous_max_steps = aloha.MAX_STEPS
            with contextlib.redirect_stdout(io.StringIO()):
                study = followup.execute(args, servo, aloha, self.artifact(servo), {"synthetic": "hash"})
            self.assertEqual(aloha.MAX_STEPS, previous_max_steps)
            self.assertEqual(study["status"], "complete")
            self.assertEqual(servo.resets, 2 * 2 * 3)
            self.assertEqual(study["pairing"]["checked_states"], 12)
            self.assertTrue(study["pairing"]["valid_so_far"])
            records = [json.loads(line) for line in args.telemetry.read_text().splitlines()]
            starts = [row for row in records if row["type"] == "scenario_start"]
            self.assertEqual([(row["episode"], row["condition"], row["arm"]) for row in starts],
                [(batch["episode"], batch["condition"], name)
                 for batch in study["schedule"] for name in batch["arms"]])
            for condition in args.condition_names:
                views = [json.loads((args.out_dir / f"{condition}_{name}.json").read_text())
                         for name in args.arm_names if name != "off"]
                self.assertEqual(views[0]["shared_control_id"], views[1]["shared_control_id"])
                self.assertEqual(views[0]["arms"]["frozen_faulted"], views[1]["arms"]["frozen_faulted"])
                for arm in study["conditions"][condition]["arms"].values():
                    self.assertEqual(arm["n"], 2)
                    self.assertEqual([row["init"] for row in arm["per_ep"]], [0, 1])
                    self.assertEqual([row["actual_seed"] for row in arm["per_ep"]], [2700, 2701])
                off_steps = [row for row in records if row.get("type") == "step"
                             and row.get("arm") == f"{condition}/off"]
                self.assertEqual(len(off_steps), 8)
                for row in off_steps:
                    np.testing.assert_array_equal(row["correction"], np.zeros(14))
                    np.testing.assert_array_equal(row["f_true"], followup.FAULTS[condition])
            self.assertNotEqual(study["conditions"]["healthy"]["shared_control_id"],
                                study["conditions"]["offset"]["shared_control_id"])

    def test_full_cube_state_mismatch_aborts_before_second_policy_rollout(self):
        with tempfile.TemporaryDirectory() as directory:
            servo = self.PhysicalServo(mismatch=True)
            args = self.arguments(directory)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "pairing failed"):
                followup.execute(args, servo, aloha, self.artifact(servo), {})
            study = json.loads((args.out_dir / "study.json").read_text())
            self.assertEqual(study["status"], "failed")
            self.assertFalse(study["pairing"]["valid_so_far"])
            self.assertEqual(study["pairing"]["mismatch"]["max_abs_qpos"], 0.001)
            self.assertEqual(servo.t, 0)  # second reset happened; no policy action followed
            self.assertEqual(sum(arm["n"] for cohort in study["conditions"].values()
                                 for arm in cohort["arms"].values()), 1)

    def test_modified_noise_cannot_reuse_qualified_gain_flag(self):
        artifact = self.artifact(self.PhysicalServo())
        artifact["observer"]["Q"] = np.asarray(artifact["observer"]["Q"]) * 2
        with self.assertRaisesRegex(ValueError, "qualified tracking matrices"):
            followup.calibrated_configs(artifact, ["off", "composite_001"], aloha)

    def test_modified_model_cannot_reuse_qualified_gain_flag(self):
        artifact = self.artifact(self.PhysicalServo())
        artifact["model"]["A"] = np.asarray(artifact["model"]["A"]) * 1.1
        with self.assertRaisesRegex(ValueError, "qualification mismatch: model.A"):
            followup.calibrated_configs(artifact, ["off", "composite_001"], aloha)

    def test_existing_output_is_rejected_before_environment_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "study.json").write_text("preserve me")
            with mock.patch.object(aloha, "Aloha") as construct, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                followup.main(["--out-dir", directory])
            self.assertEqual(error.exception.code, 2)
            construct.assert_not_called()
            self.assertEqual((Path(directory) / "study.json").read_text(), "preserve me")


if __name__ == "__main__":
    unittest.main()
