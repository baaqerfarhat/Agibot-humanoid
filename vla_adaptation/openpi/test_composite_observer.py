"""Simulator-free checks of the composite observer's control/data contracts.

Run: python3 -m unittest discover -s openpi -p test_composite_observer.py
"""
import unittest
import contextlib
import io
import json
from pathlib import Path
import tempfile

import numpy as np

from composite_observer import (
    augmented_error_report,
    composite_step,
    contraction_report,
    fit_joint_reference,
    masked_tracking_map,
    qualify_reference,
    reference_step,
)


def healthy_episodes(A, B, offset, count=6, length=180, seed=19):
    rng = np.random.RandomState(seed)
    episodes = []
    for _ in range(count):
        q = rng.normal(size=len(A))
        commands, positions = [], []
        for _ in range(length):
            u = rng.normal(size=B.shape[1])
            q = A @ q + B @ u + offset
            commands.append(u.copy())
            positions.append(q.copy())
        episodes.append(dict(u=np.asarray(commands), q=np.asarray(positions)))
    return episodes


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[0.76, 0.08], [-0.04, 0.65]])
        self.B = np.array([[0.12, 0.025], [-0.02, 0.2]])
        self.offset = np.array([0.003, -0.002])
        self.episodes = healthy_episodes(self.A, self.B, self.offset)

    def test_post_step_log_alignment_and_episode_split(self):
        model, report = fit_joint_reference(
            self.episodes, [0, 1], validation_episode_indices=[4, 5], ridge=1e-10)
        np.testing.assert_allclose(model["A"], self.A, atol=1e-9)
        np.testing.assert_allclose(model["B"], self.B, atol=1e-9)
        np.testing.assert_allclose(model["offset"], self.offset, atol=1e-9)
        self.assertEqual(report["training_episode_indices"], [0, 1, 2, 3])
        self.assertEqual(report["training_samples"], 4 * 179)
        self.assertLess(report["validation"]["rollout_rmse"], 1e-8)
        allowed, reasons = qualify_reference(report, max_validation_rmse=1e-5)
        self.assertTrue(allowed, reasons)

    def test_training_validation_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            fit_joint_reference(self.episodes, [0, 1], fit_episode_indices=[0, 1],
                                validation_episode_indices=[1, 2])

    def test_explicit_precommand_positions_use_the_first_transition(self):
        # Recover the known pre-command initial position for each synthetic log.
        episodes = []
        for episode in self.episodes:
            q, u = episode["q"], episode["u"]
            first = np.linalg.solve(self.A, q[0] - self.B @ u[0] - self.offset)
            before = np.concatenate([first[None, :], q[:-1]])
            episodes.append(dict(q=q, u=u, q_before=before))
        model, report = fit_joint_reference(episodes, [0, 1], validation_episode_indices=[5])
        self.assertEqual(report["training_samples"], 5 * 180)
        self.assertFalse(report["first_unlogged_precommand_sample_dropped"])
        np.testing.assert_allclose(model["A"], self.A, atol=1e-6)
        episodes[0]["q_before"] = episodes[0]["q_before"].copy()
        episodes[0]["q_before"][7] += 0.1
        with self.assertRaisesRegex(ValueError, "consecutive"):
            fit_joint_reference(episodes, [0, 1], validation_episode_indices=[5])

    def test_validation_does_not_change_model_and_detects_shift(self):
        model, _ = fit_joint_reference(self.episodes, [0, 1], validation_episode_indices=[4, 5])
        changed = list(self.episodes[:4]) + healthy_episodes(
            self.A, self.B, self.offset + 0.12, count=2, seed=92)
        shifted, report = fit_joint_reference(changed, [0, 1], validation_episode_indices=[4, 5])
        for key in ("A", "B", "offset", "metric"):
            np.testing.assert_array_equal(shifted[key], model[key])
        allowed, reasons = qualify_reference(report, max_validation_rmse=0.02)
        self.assertFalse(allowed)
        self.assertTrue(any("held-out" in r for r in reasons))

    def test_no_validation_is_not_qualified(self):
        _, report = fit_joint_reference(self.episodes, [0, 1])
        allowed, reasons = qualify_reference(report, max_validation_rmse=0.1)
        self.assertFalse(allowed)
        self.assertIn("no held-out healthy reference validation", reasons)

    def test_per_state_threshold_cannot_hide_one_bad_joint(self):
        _, report = fit_joint_reference(self.episodes, [0, 1], validation_episode_indices=[5])
        report["validation"]["rollout_rmse"] = 0.004
        report["validation"]["rollout_rmse_per_state"] = np.array([0.001, 0.006])
        allowed, reasons = qualify_reference(report, max_validation_rmse=0.005)
        self.assertFalse(allowed)
        self.assertTrue(any("held-out" in r for r in reasons))

    def test_noncontractive_model_is_not_silently_stabilized(self):
        for A in (np.eye(2), np.diag([1.01, 0.5])):
            report = contraction_report(A)
            self.assertFalse(report["contractive"])
            self.assertIsNone(report["metric"])
            self.assertGreaterEqual(report["spectral_radius"], 1)

    def test_nonnormal_model_contracts_in_lyapunov_metric(self):
        A = np.array([[0.7, 1.4], [0, 0.6]])
        report = contraction_report(A)
        self.assertGreater(report["euclidean_factor"], 1)
        self.assertLess(report["contraction_factor"], 1)
        L = report["metric"]
        np.testing.assert_allclose(A.T @ L @ A - L, -np.eye(2), atol=1e-10)

    def test_rank_deficient_calibration_is_not_qualified(self):
        episodes = [dict(q=np.zeros((40, 2)), u=np.zeros((40, 2))) for _ in range(3)]
        _, report = fit_joint_reference(episodes, [0, 1], validation_episode_indices=[2])
        allowed, reasons = qualify_reference(report, max_validation_rmse=0.1)
        self.assertFalse(allowed)
        self.assertTrue(any("rank deficient" in r for r in reasons))

    def test_reference_uses_requested_command_coordinates(self):
        model = dict(A=np.array([[0.8]]), B=np.array([[0.2, -0.1]]),
                     offset=np.array([0.03]), command_indices=[0, 2])
        result = reference_step(model, np.array([0.4]), np.array([0.6, 1000, 0.7]))
        np.testing.assert_allclose(result, [0.8 * 0.4 + 0.2 * 0.6 - 0.1 * 0.7 + 0.03])
        D = masked_tracking_map(model, np.array([1., 1., 0.]))
        np.testing.assert_array_equal(D, [[0.2, 0., 0.]])


class CompositeTests(unittest.TestCase):
    def test_zero_tracking_matches_existing_kalman(self):
        from adaptive_law import estimator_step

        rng = np.random.RandomState(51)
        H = np.eye(6) + 0.03 * rng.normal(size=(6, 6))
        M_inv = np.linalg.inv(H)
        R = np.eye(6) * 0.015
        Q_step = np.eye(6) * 0.0002
        bias = np.linspace(-0.01, 0.01, 6)
        ours, baseline = np.zeros(6), np.zeros(6)
        state, baseline_state = None, None
        for _ in range(50):
            residual = rng.normal(size=6) * 0.1 + H @ bias
            baseline, base_diag = estimator_step(
                baseline, residual, M_inv, M=H, gamma=0.08, dead=0.002,
                norm_r=0.4, clip=0.3, bias=bias, baseline="kalman",
                state=baseline_state, kf_q=Q_step, kf_r=R)
            ours, diag = composite_step(
                ours, residual - H @ bias, H, state=state,
                Q=Q_step / 0.02, R=R, dt=0.02, tracking_rate=0,
                damping=0, clip=0.3)
            state, baseline_state = diag["estimator_state"], base_diag["estimator_state"]
            np.testing.assert_allclose(ours, baseline, atol=1e-13, rtol=1e-13)
            np.testing.assert_allclose(state["covariance"], baseline_state["covariance"],
                                       atol=1e-13, rtol=1e-13)

    def test_damping_and_process_noise_have_consistent_time_units(self):
        theta = np.array([0.08, -0.1])
        opts = dict(H=np.zeros((1, 2)), observation=np.zeros(1),
                    Q=np.diag([0.003, 0.001]), R=np.ones((1, 1)),
                    damping=0.7, tracking_rate=0, clip=None)
        one, one_diag = composite_step(theta, dt=0.2, **opts)
        many, state = theta.copy(), None
        for _ in range(10):
            many, diag = composite_step(many, state=state, dt=0.02, **opts)
            state = diag["estimator_state"]
        np.testing.assert_allclose(one, many, atol=1e-14)
        np.testing.assert_allclose(one_diag["estimator_state"]["covariance"],
                                   state["covariance"], atol=1e-14)

    def test_actual_minus_reference_sign_rejects_constant_fault(self):
        # No predictive observation in this test: tracking must genuinely supply
        # the feedback, and cannot pass by just reusing an observer innovation.
        theta, e, state = np.zeros(1), np.zeros(1), None
        fault = 0.08
        A, D = np.array([[0.8]]), np.array([[0.2]])
        L = contraction_report(A)["metric"]
        for _ in range(180):
            e = A @ e + D[:, 0] * (fault - theta[0])
            theta, diag = composite_step(
                theta, np.zeros(1), np.zeros((1, 1)), state=state,
                Q=np.zeros((1, 1)), R=np.ones((1, 1)), dt=0.02,
                tracking_rate=30, tracking_error=e, tracking_map=D,
                metric=L, clip=0.3)
            state = diag["estimator_state"]
        self.assertLess(abs(e[0]), 1e-8)
        self.assertLess(abs(theta[0] - fault), 1e-8)

    def test_augmented_dynamics_match_action_before_update_timing(self):
        A = np.array([[0.7, 0.05], [0.02, 0.65]])
        D = np.array([[0.2, 0.03], [-0.04, 0.12]])
        H = np.array([[0.9, -0.08], [0.03, 1.1]])
        theta, e = np.array([0.04, -0.02]), np.array([0.03, 0.01])
        L = contraction_report(A)["metric"]
        next_e = A @ e - D @ theta
        next_theta, diag = composite_step(
            theta, np.zeros(2), H, Q=np.eye(2) * 0.03,
            R=np.eye(2) * 0.2, dt=0.05, damping=0.1,
            tracking_error=next_e, tracking_map=D, metric=L,
            tracking_rate=12, clip=None)
        report = augmented_error_report(
            A, D, H, diag["estimator_state"]["covariance"], diag["gain"],
            dt=0.05, tracking_rate=12, damping=0.1, metric=L)
        np.testing.assert_allclose(report["matrix"] @ np.r_[e, theta],
                                   np.r_[next_e, next_theta], atol=1e-14)

    def test_contracting_servo_can_have_unstable_composite_gain(self):
        A, D = np.array([[0.8]]), np.array([[0.2]])
        L = contraction_report(A)["metric"]
        opts = dict(A=A, D=D, H=np.zeros((1, 1)), covariance=np.eye(1),
                    gain=np.zeros((1, 1)), dt=0.02, metric=L)
        self.assertTrue(augmented_error_report(tracking_rate=30, **opts)["schur_stable"])
        self.assertFalse(augmented_error_report(tracking_rate=10000, **opts)["schur_stable"])

    def test_mask_and_tracking_dimensions_are_explicit(self):
        H = np.eye(3)
        D = np.array([[0.2, 0., 0.]])
        theta, diag = composite_step(
            np.zeros(3), np.zeros(3), H, Q=np.eye(3) * 0.01,
            R=np.eye(3), dt=0.02, tracking_error=np.array([0.1]),
            tracking_map=D, tracking_rate=50)
        self.assertGreater(theta[0], 0)
        np.testing.assert_array_equal(theta[1:], [0, 0])
        with self.assertRaisesRegex(ValueError, "tracking map"):
            composite_step(np.zeros(3), np.zeros(3), H, Q=np.eye(3), R=np.eye(3),
                           dt=0.02, tracking_rate=1, tracking_error=np.zeros(3),
                           tracking_map=np.eye(2))

    def test_covariance_stays_positive_with_large_feedback_and_projection(self):
        theta, state = np.zeros(2), None
        for _ in range(50):
            theta, diag = composite_step(
                theta, np.ones(2), np.array([[1., 0.8], [0.8, 1.]]),
                state=state, Q=np.eye(2) * 0.01, R=np.eye(2) * 1e-4,
                dt=0.02, tracking_rate=10000, tracking_error=np.ones(2),
                tracking_map=np.eye(2), clip=0.08)
            state = diag["estimator_state"]
            self.assertGreater(np.linalg.eigvalsh(state["covariance"])[0], 0)
            self.assertLessEqual(np.max(np.abs(theta)), 0.08)
        self.assertTrue(diag["projection_active"])

    def test_bad_covariance_and_missing_tracking_signal_are_rejected(self):
        opts = dict(theta=np.zeros(2), observation=np.zeros(2), H=np.eye(2),
                    Q=np.eye(2), R=np.eye(2), dt=0.02)
        with self.assertRaisesRegex(ValueError, "position error"):
            composite_step(tracking_rate=1, **opts)
        opts["R"] = np.array([[1., 2.], [2., 1.]])
        with self.assertRaisesRegex(ValueError, "positive definite"):
            composite_step(**opts)


class PreparationTests(unittest.TestCase):
    def test_noise_fit_and_gain_grid_use_training_data_only(self):
        from prepare_composite_reference import fit_joint_fir_noise, tracking_grid

        A, B = np.diag([0.7, 0.8]), np.diag([0.3, 0.2])
        episodes = healthy_episodes(A, B, np.zeros(2), count=4)
        noise = fit_joint_fir_noise(episodes, [0, 1, 2])
        changed = list(episodes[:3]) + healthy_episodes(A, B, np.ones(2), count=1)
        changed_noise = fit_joint_fir_noise(changed, [0, 1, 2])
        np.testing.assert_array_equal(changed_noise["W"], noise["W"])
        np.testing.assert_array_equal(changed_noise["R"], noise["R"])
        model, _ = fit_joint_reference(episodes, [0, 1], validation_episode_indices=[3])
        grid = tracking_grid(model, np.eye(2), noise, [0, 1], [0., 0.05, 100.], 0.02)
        np.testing.assert_allclose(grid["steady_effective_gain"], np.eye(2) * 0.08)
        self.assertTrue(grid["candidates"][0]["allowed"])
        self.assertTrue(grid["candidates"][1]["allowed"])
        self.assertFalse(grid["candidates"][2]["allowed"])

    def test_cli_records_qualified_shared_observer_and_rejected_validation(self):
        from prepare_composite_reference import main, json_ready

        A, B = np.diag([0.7, 0.8]), np.diag([0.3, 0.2])
        episodes = healthy_episodes(A, B, np.zeros(2), count=3)
        with tempfile.TemporaryDirectory(prefix="composite_reference_test_") as directory:
            root = Path(directory)
            source, matrix, destination = root / "healthy.json", root / "M.json", root / "reference.json"
            source.write_text(json.dumps(json_ready(episodes)))
            matrix.write_text(json.dumps(dict(M=np.eye(2).tolist())))
            argv = ["--log", str(source), "--out", str(destination),
                    "--state-indices", "0,1", "--command-indices", "0,1",
                    "--fit-episodes", "0,1", "--validation-episodes", "2",
                    "--max-validation-rmse", "0.001", "--dt", "0.02",
                    "--openloop", str(matrix), "--correction-indices", "0,1"]
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(argv)
            self.assertEqual(status, 0)
            artifact = json.loads(destination.read_text())
            self.assertTrue(artifact["qualification"]["allowed"])
            self.assertEqual(set(("W", "M", "Q", "R")) - set(artifact["observer"]), set())
            self.assertEqual(artifact["observer"]["fit_episode_indices"], [0, 1])
            self.assertEqual(len(artifact["source"]["sha256"]), 64)
            np.testing.assert_allclose(artifact["observer"]["Q"],
                                       np.asarray(artifact["tracking"]["Q_rate"]) * 0.02)
            # Keep the same fixed threshold: shift ONLY the held-out episode.
            episodes[-1] = healthy_episodes(A, B, np.ones(2) * 0.1, count=1)[0]
            source.write_text(json.dumps(json_ready(episodes)))
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(argv)
            self.assertEqual(status, 2)
            rejected = json.loads(destination.read_text())
            self.assertFalse(rejected["qualification"]["allowed"])
            self.assertTrue(all(not row["qualified_for_experiment"]
                                for row in rejected["tracking"]["candidates"]))
            np.testing.assert_array_equal(rejected["observer"]["W"], artifact["observer"]["W"])


if __name__ == "__main__":
    unittest.main()
