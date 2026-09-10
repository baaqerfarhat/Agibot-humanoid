"""Independent allocation, filtering, and numeric-failure checks; NumPy only."""
import copy
import itertools
import unittest

import numpy as np

from weighted_dob import _solve_box_ridge, weighted_dob_step


def enumerate_two_dimensional_solution(M, target, R, bound, prior):
    """Independent optimum: augmented least squares on every face/edge/vertex."""
    chol = np.linalg.cholesky(R)
    A = np.vstack([np.linalg.solve(chol, M), np.diag(1/np.asarray(prior))])
    b = np.r_[np.linalg.solve(chol, target), np.zeros(2)]
    candidates = []
    for signs in itertools.product((-1, 0, 1), repeat=2):
        fixed = np.array(signs) != 0
        x = np.array(signs)*bound
        if np.any(~fixed):
            x[~fixed] = np.linalg.lstsq(A[:,~fixed], b-A[:,fixed]@x[fixed], rcond=None)[0]
        if np.all(np.abs(x) <= bound+1e-12):
            candidates.append((float(np.linalg.norm(A@x-b)**2), x.copy()))
    return min(candidates, key=lambda pair: pair[0])


class WeightedDobTests(unittest.TestCase):
    def step(self, theta, residual, M, **overrides):
        kwargs = dict(mask=np.ones(len(theta)), R=np.eye(len(residual)), gamma=.08, clip=.3)
        kwargs.update(overrides)
        return weighted_dob_step(theta, residual, M, **kwargs)

    def test_masked_joint_solution_avoids_discarded_rotation_cancellation(self):
        # The second coordinate cancels its own x effect through the first. Keeping
        # only that first coordinate from the joint inverse creates x error of 10.
        M = np.array([[1., 1.], [0., .01]])
        target = np.array([0., .1])
        inverse_then_mask = np.linalg.solve(M, target) * [1,0]
        self.assertGreater(np.linalg.norm(target-M@inverse_then_mask), 100*np.linalg.norm(target))
        estimate, diag = self.step(np.zeros(2), target, M, mask=[1,0], gamma=1)
        np.testing.assert_array_equal(estimate, np.zeros(2))
        np.testing.assert_array_equal(diag["predicted_compensated_filtered_residual"], target)
        # This remaining orthogonal component illustrates nonexpansion, not strict contraction.
        self.assertAlmostEqual(diag["optimizer"]["weighted_residual_norm"], np.linalg.norm(target))

    def test_unconstrained_solution_and_residual_ema_have_analytic_form(self):
        M = np.array([[1.2,.3],[-.2,.8]])
        R = np.array([[.02,.003],[.003,.01]])
        prior = np.array([.1,.2])
        L = np.linalg.solve(M.T@np.linalg.solve(R,M)+np.diag(1/prior**2),
                            M.T@np.linalg.inv(R))
        theta = np.zeros(2)
        state = None
        expected_filtered = np.zeros(2)
        for residual in (np.array([.08,-.03]), np.array([.02,.01]), np.zeros(2)):
            expected_filtered = .75*expected_filtered+.25*residual
            theta, diag = self.step(theta, residual, M, R=R, prior_std=prior, gamma=.25, state=state)
            np.testing.assert_allclose(theta, L@expected_filtered, rtol=0, atol=2e-16)
            np.testing.assert_allclose(diag["estimator_state"]["residual_estimate"], expected_filtered,
                                       rtol=0, atol=1e-16)
            state = diag["estimator_state"]
        self.assertEqual(state["updates"], 3)

    def test_constant_fault_correction_sign_and_removal(self):
        theta = np.zeros(3)
        state = None
        target = np.array([.04,-.02,.01])
        previous_error = np.linalg.norm(target)
        for _ in range(80):
            theta, diag = self.step(theta, target, np.eye(3), R=np.eye(3)*1e-5, gamma=.08, state=state)
            corrected = target-theta  # actual minus estimate is the commanded correction convention
            self.assertLessEqual(np.linalg.norm(corrected), previous_error+1e-14)
            previous_error = np.linalg.norm(corrected)
            state = diag["estimator_state"]
        self.assertLess(previous_error, .00012)
        start = np.linalg.norm(theta)
        for _ in range(80):
            theta, diag = self.step(theta, np.zeros(3), np.eye(3), R=np.eye(3)*1e-5, state=state)
            state = diag["estimator_state"]
        self.assertLess(np.linalg.norm(theta), start*.002)

    def test_bias_is_removed_in_residual_coordinates_before_filtering(self):
        M = np.array([[1.,2.],[-.5,.7]])
        bias = np.array([.04,-.08])
        theta, diag = self.step(np.zeros(2), M@bias, M, gamma=1, bias=bias)
        np.testing.assert_array_equal(theta, np.zeros(2))
        np.testing.assert_array_equal(diag["residual_estimate"], np.zeros(2))

    def test_empty_mask_filters_but_applies_no_correction(self):
        theta, diag = self.step(np.ones(6), np.arange(6)*.01, np.eye(6), mask=np.zeros(6), gamma=.2)
        np.testing.assert_array_equal(theta, np.zeros(6))
        np.testing.assert_allclose(diag["residual_estimate"], np.arange(6)*.002)
        self.assertEqual(diag["optimizer"]["status"], "optimal")
        self.assertEqual(diag["optimizer"]["kkt_residual"], 0)

    def test_active_box_solutions_match_independent_face_enumeration(self):
        rng = np.random.RandomState(13)
        active_cases = 0
        for _ in range(60):
            M = rng.normal(size=(3,2))
            factor = rng.normal(size=(3,3))
            R = factor@factor.T*.0001 + np.eye(3)*1e-6
            bound = rng.uniform(.015,.09,2)
            prior = rng.uniform(.05,.2,2)
            target = rng.normal(size=3)*.4
            expected_cost, expected = enumerate_two_dimensional_solution(M, target, R, bound, prior)
            actual, diag = self.step(np.zeros(2), target, M, R=R, gamma=1, clip=bound, prior_std=prior)
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=2e-11)
            self.assertAlmostEqual(diag["optimizer"]["objective"]/max(1,expected_cost),
                                   expected_cost/max(1,expected_cost), places=11)
            active_cases += bool(np.any(np.abs(actual) == bound))
        self.assertGreater(active_cases, 50)

    def test_random_correlated_covariances_and_twelve_active_inputs_pass_kkt(self):
        rng = np.random.RandomState(51)
        for active in (1,3,6,12):
            for _ in range(15):
                dimension = active+2
                M = rng.normal(size=(dimension,dimension))
                factor = rng.normal(size=(dimension,dimension))
                # Explicit floor turns a PSD sample-covariance construction PD.
                R = factor@factor.T*1e-5 + np.eye(dimension)*1e-7
                target = rng.normal(size=dimension)*.1
                mask = np.r_[np.ones(active),np.zeros(2)]
                initial = rng.normal(size=dimension)
                estimate, diag = self.step(initial, target, M, mask=mask, R=R, gamma=1, clip=.035)
                np.testing.assert_array_equal(estimate[-2:], np.zeros(2))
                self.assertTrue(np.all(np.abs(estimate) <= .035))
                opt = diag["optimizer"]
                self.assertEqual(opt["status"], "optimal")
                self.assertLessEqual(opt["kkt_scaled_residual"], opt["kkt_tolerance"])
                self.assertLessEqual(opt["objective"], opt["zero_objective"]+opt["objective_numeric_tolerance"])
                self.assertTrue(np.all(np.diff(opt["objective_history"]) <= opt["objective_numeric_tolerance"]))
                # Verify optimality in original residual coordinates independently.
                gradient = M[:,:active].T@np.linalg.solve(R,M@estimate-target)+estimate[:active]/.1**2
                for i,value in enumerate(estimate[:active]):
                    if value == -.035: self.assertGreaterEqual(gradient[i], -1e-7)
                    elif value == .035: self.assertLessEqual(gradient[i], 1e-7)
                    else: self.assertAlmostEqual(gradient[i], 0, delta=1e-7)

    def test_bad_covariance_parameters_and_explicit_nonconvergence(self):
        for R in (np.zeros((2,2)), np.diag([1.,-1.]), np.array([[1.,.2],[0.,1.]])):
            with self.subTest(R=R), self.assertRaises(ValueError):
                self.step(np.zeros(2), np.ones(2), np.eye(2), R=R)
        for option in (dict(gamma=0), dict(gamma=1.1), dict(clip=0), dict(prior_std=-1),
                       dict(mask=[1,.5]), dict(bias=[1]), dict(state=dict(residual_estimate=[1],updates=0))):
            with self.subTest(option=option), self.assertRaises(ValueError):
                self.step(np.zeros(2), np.ones(2), np.eye(2), **option)
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            _solve_box_ridge(np.eye(2), np.ones(2), np.ones(2), np.full(2,.1), max_iterations=0)

    def test_inputs_and_previous_state_are_immutable(self):
        theta=np.array([.03,-.02]); r=np.array([.08,.01]); M=np.array([[1.,.3],[.1,.8]])
        R=np.array([[.01,.001],[.001,.02]]); mask=np.array([1.,0.]); bias=np.array([.001,.002])
        state=dict(residual_estimate=np.array([.02,-.01]),updates=7)
        arrays=[theta,r,M,R,mask,bias,state["residual_estimate"]]
        before=[array.tobytes() for array in arrays]
        saved=copy.deepcopy(state)
        _,diag=self.step(theta,r,M,R=R,mask=mask,bias=bias,state=state)
        self.assertEqual(before,[array.tobytes() for array in arrays])
        self.assertEqual(state["updates"],saved["updates"])
        diag["estimator_state"]["residual_estimate"][0]=999
        self.assertEqual(before,[array.tobytes() for array in arrays])


if __name__ == "__main__":
    unittest.main()
