"""Meaningful recipe, nesting, eligibility and provenance regressions."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from composite_observer import composite_step
from prepare_composite_tuning import _steady, build_candidate_bank, validate_candidate_bank
from test_composite_validation import prepared_artifact


class CandidateBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        prepared_artifact(cls.root)
        cls.reference = cls.root / "reference.json"
        cls.specs = [
            dict(name="kf",family="kalman",p0="gamma_c"),
            dict(name="composite_zero",family="composite",p0="gamma_c"),
            dict(name="composite",family="composite",p0="gamma_c",active_strength=.05),
            dict(name="unsafe",family="composite",active_strength=100.),
            dict(name="isotropic",family="kalman",q_recipe="arm_isotropic",p0="steady"),
            dict(name="scalar",family="kalman",p0=.25),
            dict(name="matrix",family="kalman",p0=(.25*np.eye(14)).tolist()),
            dict(name="off",family="off"),
            dict(name="oracle",family="oracle"),
        ]
        cls.bank = build_candidate_bank(cls.reference,cls.specs,
            calibration_seeds=[100,101,102],screened_steps=40,covariance_steps=200)

    def test_reconstructs_sources_and_retains_rejected_but_excludes_them_from_runtime(self):
        result = validate_candidate_bank(self.bank)
        self.assertNotIn("unsafe",result["candidates"])
        self.assertIn("unsafe",[row["name"] for row in self.bank["candidates"]])
        self.assertTrue(next(row for row in self.bank["candidates"] if row["name"]=="unsafe")["qualification"]["reasons"])
        self.assertEqual(result["calibration_seeds"],[100,101,102])
        self.assertEqual(result["screened_steps"],40)
        self.assertEqual(result["correction_indices"],list(range(6)))
        self.assertEqual(result["dt"],.02)
        self.assertEqual(set(result["sources"]),{str(self.reference),str(self.root/"healthy.json"),str(self.root/"M.json")})

    def test_zero_tracking_nests_kalman_with_identical_initialization_and_projection(self):
        result = validate_candidate_bank(self.bank)
        kf = result["candidates"]["kf"]
        zero = result["candidates"]["composite_zero"]
        H = np.asarray(result["observer"]["M"])
        state_a = state_b = None
        theta_a = theta_b = np.zeros(14)
        rng = np.random.RandomState(62)
        for _ in range(30):
            observation = rng.normal(size=14)*.3
            outputs = []
            for config,state,theta in [(kf,state_a,theta_a),(zero,state_b,theta_b)]:
                outputs.append(composite_step(theta,observation,H,state=state,
                    Q=config["Q"]/.02,R=config["R"],dt=.02,
                    initial_covariance=config["P0"],tracking_rate=config["tracking_rate"],
                    damping=config["damping"],clip=config["clip"]))
            (theta_a,a),(theta_b,b) = outputs
            state_a,state_b = a["estimator_state"],b["estimator_state"]
            np.testing.assert_array_equal(theta_a,theta_b)
            np.testing.assert_array_equal(state_a["covariance"],state_b["covariance"])

    def test_scalar_and_matrix_initialization_are_equivalent(self):
        result = validate_candidate_bank(self.bank)["candidates"]
        np.testing.assert_array_equal(result["scalar"]["P0"],result["matrix"]["P0"])

    def test_matched_steady_prior_has_declared_bandwidth_from_first_update(self):
        result = validate_candidate_bank(self.bank)
        config = result["candidates"]["kf"]
        H = np.asarray(result["observer"]["M"])
        _,diag = composite_step(np.zeros(14),np.ones(14),H,Q=config["Q"]/.02,
            R=config["R"],dt=.02,initial_covariance=config["P0"])
        np.testing.assert_allclose(diag["effective_gain"],.08*np.eye(14),atol=1e-14)

    def test_isotropic_process_template_respects_arm_and_gripper_units(self):
        result = validate_candidate_bank(self.bank)["candidates"]
        base = json.loads(self.reference.read_text())
        inverse = np.linalg.inv(np.asarray(base["observer"]["M"]))
        C = inverse @ np.asarray(base["observer"]["R"]) @ inverse.T
        template = result["isotropic"]["Q"]*(1-.08)/.08**2
        arms = list(range(6))+list(range(7,13))
        np.testing.assert_allclose(np.diag(template)[arms],np.median(np.diag(C)[:6]))
        np.testing.assert_allclose(np.diag(template)[[6,13]],np.diag(C)[[6,13]])
        np.testing.assert_array_equal(template,np.diag(np.diag(template)))

    def test_exact_steady_covariance_matches_damped_correlated_core_recursion(self):
        H = np.array([[1.,.3],[-.1,.8]])
        Q = np.array([[.008,.002],[.002,.015]])
        R = np.array([[.07,-.01],[-.01,.02]])
        steady,gain,_,valid = _steady(H,Q,R,.02,.7,100)
        self.assertTrue(valid)
        state = None
        for _ in range(300):
            _,diag = composite_step(np.zeros(2),np.zeros(2),H,Q=Q/.02,R=R,
                dt=.02,damping=.7,state=state)
            state = diag["estimator_state"]
        np.testing.assert_allclose(state["covariance"],steady,rtol=1e-12,atol=1e-15)
        np.testing.assert_allclose(diag["gain"],gain,rtol=1e-12,atol=1e-15)

    def test_very_slow_noise_mode_has_valid_analytic_steady_solution(self):
        steady,_,_,valid = _steady(np.eye(2),np.diag([1e-16,.01]),np.eye(2),.02,0,100)
        self.assertTrue(valid)
        self.assertAlmostEqual(steady[0,0],1e-8,delta=1e-15)

    def test_matched_scalar_solution_handles_damping_and_independent_noise_scales(self):
        H = np.array([[1.,.3],[-.1,.8]])
        Rbase = np.array([[.07,-.01],[-.01,.02]])
        inverse = np.linalg.inv(H)
        Cbase = inverse@Rbase@inverse.T
        gamma,q_scale,r_scale = .2,1.7,.3
        Q = gamma**2/(1-gamma)*q_scale*Cbase
        R = r_scale*Rbase
        scale = gamma**2/(1-gamma)*q_scale/r_scale
        steady,gain,_,valid = _steady(H,Q,R,.02,.7,100,matched_scale=scale)
        self.assertTrue(valid)
        state = None
        for _ in range(300):
            _,diag = composite_step(np.zeros(2),np.zeros(2),H,Q=Q/.02,R=R,
                dt=.02,damping=.7,state=state)
            state = diag["estimator_state"]
        np.testing.assert_allclose(state["covariance"],steady,rtol=1e-12,atol=1e-15)
        np.testing.assert_allclose(diag["gain"],gain,rtol=1e-12,atol=1e-15)

    def test_matched_ill_conditioned_covariance_does_not_create_spurious_gain_coupling(self):
        H = np.array([[1.,.2,0.],[0.,.8,.1],[0.,0.,1.]])
        C = np.diag([1e-9,1e-5,1.])
        R = H@C@H.T
        gamma = .08
        scale = gamma**2/(1-gamma)
        steady,gain,_,valid = _steady(H,scale*C,R,.02,0,100,matched_scale=scale)
        self.assertTrue(valid)
        np.testing.assert_allclose(steady,gamma*C,rtol=1e-12,atol=1e-18)
        np.testing.assert_allclose(gain,gamma*np.linalg.inv(H),rtol=1e-14,atol=1e-18)
        np.testing.assert_allclose(gain@H,gamma*np.eye(3),rtol=1e-14,atol=1e-17)

    def test_active_normalization_uses_applied_output_coordinates(self):
        row = next(row for row in self.bank["candidates"] if row["name"]=="composite")
        reference = self.bank["reference"]["artifact"]
        P = np.asarray(row["screening"]["steady_covariance"])
        D = np.asarray(reference["tracking"]["tracking_map"])
        L = np.asarray(reference["model"]["metric"])
        operator = .02*row["parameters"]["tracking_rate"]*P@D.T@L@D
        self.assertAlmostEqual(np.linalg.norm(operator[:6,:6],2),.05,places=13)

    def test_changed_matrices_rates_and_clipping_cannot_keep_qualification(self):
        for field in ("Q","R","P0","tracking_rate","clip"):
            with self.subTest(field=field):
                edited = copy.deepcopy(self.bank)
                row = next(row for row in edited["candidates"] if row["name"]=="composite")
                value = np.asarray(row["parameters"][field])
                row["parameters"][field] = (value*1.7).tolist()
                with self.assertRaisesRegex(ValueError,"qualification mismatch"):
                    validate_candidate_bank(edited)

    def test_changed_recipe_cannot_reuse_old_matrices(self):
        edited = copy.deepcopy(self.bank)
        edited["candidate_specs"][0]["gamma"] = .2
        with self.assertRaisesRegex(ValueError,"qualification mismatch"):
            validate_candidate_bank(edited)

    def test_reconstruction_roundoff_does_not_replace_declared_runtime_bits(self):
        edited = copy.deepcopy(self.bank)
        row = next(row for row in edited["candidates"] if row["name"]=="composite")
        for key in ("Q","R","P0"):
            old = row["parameters"][key][0][0]
            row["parameters"][key][0][0] = float(np.nextafter(old,np.inf))
        row["parameters"]["tracking_rate"] = float(np.nextafter(row["parameters"]["tracking_rate"],np.inf))
        edited["observer"]["W"][0][0] = float(np.nextafter(edited["observer"]["W"][0][0],np.inf))
        edited["reference_model"]["A"][0][0] = float(np.nextafter(edited["reference_model"]["A"][0][0],np.inf))
        result = validate_candidate_bank(edited)
        for key in ("Q","R","P0"):
            self.assertEqual(result["candidates"]["composite"][key][0,0],row["parameters"][key][0][0])
        self.assertEqual(result["candidates"]["composite"]["tracking_rate"],row["parameters"]["tracking_rate"])
        self.assertEqual(result["observer"]["W"][0][0],edited["observer"]["W"][0][0])
        self.assertEqual(result["reference_model"]["A"][0][0],edited["reference_model"]["A"][0][0])

    def test_rejected_candidate_flag_cannot_be_overridden(self):
        edited = copy.deepcopy(self.bank)
        row = next(row for row in edited["candidates"] if row["name"]=="unsafe")
        row["qualification"] = dict(allowed=True,reasons=[])
        with self.assertRaisesRegex(ValueError,"qualification mismatch"):
            validate_candidate_bank(edited)

    def test_changed_embedded_reference_cannot_keep_source_hash(self):
        edited = copy.deepcopy(self.bank)
        edited["reference"]["artifact"]["model"]["B"][0][0] *= 5
        with self.assertRaisesRegex(ValueError,"qualification mismatch"):
            validate_candidate_bank(edited)

    def test_wrong_explicit_healthy_source_fails_without_implicit_fallback(self):
        path = self.root/"wrong.json"
        path.write_text("[]")
        with self.assertRaisesRegex(ValueError,"SHA256 mismatch"):
            validate_candidate_bank(self.bank,source_log=path)

    def test_portable_reference_and_source_copies_are_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"reference.json").write_bytes(self.reference.read_bytes())
            (root/"healthy.json").write_bytes((self.root/"healthy.json").read_bytes())
            (root/"M.json").write_bytes((self.root/"M.json").read_bytes())
            bank = copy.deepcopy(self.bank)
            bank["reference"]["path"] = "/missing/reference.json"
            path = root/"bank.json"
            path.write_text(json.dumps(bank))
            result = validate_candidate_bank(bank,bank_path=path,
                source_log=root/"healthy.json",source_sensitivity=root/"M.json")
            self.assertIn(str(root/"reference.json"),result["sources"])
            self.assertIn(str(path),result["sources"])

    def test_invalid_or_ambiguous_recipes_fail_before_rollouts(self):
        specs = [dict(name="bad",family="composite",p0=-1),
                 dict(name="bad",family="composite",p0=np.zeros((14,14)).tolist()),
                 dict(name="bad",family="kalman",active_strength=.1),
                 dict(name="bad",family="legacy",damping=.1),
                 dict(name="bad",family="dob",typo_gamma=.2),
                 dict(name="bad",family="kalman",gamma=1)]
        for spec in specs:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                build_candidate_bank(self.reference,[spec],screened_steps=40,covariance_steps=200)
        with self.assertRaisesRegex(ValueError,"duplicate candidate name"):
            build_candidate_bank(self.reference,[self.specs[0]]*2,screened_steps=40,covariance_steps=200)

    def test_source_bound_reference_is_rebuilt_not_accepted_from_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = json.loads(self.reference.read_text())
            reference["model"]["A"][0][0] = 1.5
            path = root/"reference.json"
            path.write_text(json.dumps(reference))
            with self.assertRaisesRegex(ValueError,"qualification mismatch"):
                build_candidate_bank(path,self.specs,screened_steps=40,covariance_steps=200)


if __name__ == "__main__":
    unittest.main()
