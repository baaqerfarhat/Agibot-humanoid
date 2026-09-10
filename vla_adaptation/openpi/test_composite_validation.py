"""Source/model mutation regressions for composite-only launch qualification."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from prepare_composite_reference import main as prepare, json_ready
from test_composite_observer import healthy_episodes
from validate_composite_reference import validate_composite_artifact


def prepared_artifact(directory):
    """A complete, portable synthetic calibration; no simulator or repo results."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    log, sensitivity, output = root / "healthy.json", root / "M.json", root / "reference.json"
    A = np.diag(np.linspace(0.5, 0.8, 14))
    episodes = healthy_episodes(A, np.eye(14)-A, np.zeros(14), count=3)
    log.write_text(json.dumps(json_ready(episodes)))
    sensitivity.write_text(json.dumps(dict(M=np.eye(14).tolist())))
    with contextlib.redirect_stdout(io.StringIO()):
        status = prepare(["--log", str(log), "--out", str(output), "--openloop", str(sensitivity),
            "--state-indices", "0,1,2,3,4,5", "--command-indices", "0,1,2,3,4,5",
            "--correction-indices", "0,1,2,3,4,5", "--fit-episodes", "0,1",
            "--validation-episodes", "2", "--max-validation-rmse", "0.001", "--dt", "0.02",
            "--tracking-strengths", "0,0.01"])
    if status:
        raise AssertionError("synthetic reference did not qualify")
    return json.loads(output.read_text())


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.artifact = prepared_artifact(cls.directory)

    def test_frozen_sources_and_runtime_matrices_reconstruct(self):
        result = validate_composite_artifact(self.artifact)
        self.assertTrue(result["validated"])
        self.assertEqual(result["sources"]["log"]["sha256"], self.artifact["source"]["sha256"])

    def test_changed_model_cannot_keep_qualification_even_if_tracking_copy_is_changed(self):
        for name in ("A", "B", "offset", "metric"):
            with self.subTest(name=name):
                edited = copy.deepcopy(self.artifact)
                edited["model"][name] = (np.asarray(edited["model"][name]) + 0.01).tolist()
                if name == "B":
                    edited["tracking"]["tracking_map"] = np.c_[edited["model"]["B"], np.zeros((6,8))].tolist()
                if name == "metric":
                    edited["tracking"]["metric"] = edited["model"]["metric"]
                with self.assertRaisesRegex(ValueError, "model."+name):
                    validate_composite_artifact(edited)

    def test_changed_shared_observer_and_gain_cannot_keep_flags(self):
        for name in ("W", "M", "Q", "R", "tracking_rate"):
            with self.subTest(name=name):
                edited = copy.deepcopy(self.artifact)
                if name == "tracking_rate":
                    edited["tracking"]["candidates"][1][name] *= 2
                else:
                    edited["observer"][name] = (np.asarray(edited["observer"][name])*2).tolist()
                with self.assertRaisesRegex(ValueError, "qualification mismatch"):
                    validate_composite_artifact(edited)

    def test_missing_source_fails_and_explicit_identical_portable_copy_passes(self):
        edited = copy.deepcopy(self.artifact)
        edited["source"]["path"] = "/missing/composite/healthy.json"
        edited["settings"]["log"] = "/missing/composite/healthy.json"
        with self.assertRaisesRegex(ValueError, "hash-matched log source"):
            validate_composite_artifact(edited)
        portable = self.directory / "healthy_portable.json"
        portable.write_bytes((self.directory / "healthy.json").read_bytes())
        result = validate_composite_artifact(edited, source_log=portable)
        self.assertEqual(result["sources"]["log"]["path"], str(portable))

    def test_explicit_wrong_source_never_falls_back_to_valid_implicit_source(self):
        wrong = self.directory / "wrong.json"
        wrong.write_text("[]")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            validate_composite_artifact(self.artifact, source_log=wrong)

    def test_fir_residual_mean_allows_observed_cross_numpy_cancellation_roundoff(self):
        # Actual portable fresh-reference discrepancy was 1.4297906055989332e-15
        # between NumPy 1.21.5 and 1.26.4. This diagnostic is not observer bias.
        edited = copy.deepcopy(self.artifact)
        mean = np.asarray(edited["noise"]["residual_mean"])
        edited["noise"]["residual_mean"] = (mean+1.4297906055989332e-15).tolist()
        result = validate_composite_artifact(edited)
        self.assertTrue(result["validated"])
        self.assertIsNone(edited["observer"]["bias"])

    def test_mean_roundoff_exception_does_not_accept_material_bias_or_covariance_changes(self):
        edited = copy.deepcopy(self.artifact)
        edited["noise"]["residual_mean"][0] += 1e-10
        with self.assertRaisesRegex(ValueError, "noise.residual_mean"):
            validate_composite_artifact(edited)
        for section,key in (("noise","R"),("observer","R"),("observer","Q")):
            with self.subTest(section=section,key=key):
                edited = copy.deepcopy(self.artifact)
                edited[section][key] = (np.asarray(edited[section][key])*1.00001).tolist()
                with self.assertRaisesRegex(ValueError, "qualification mismatch"):
                    validate_composite_artifact(edited)

    def test_mean_roundoff_exception_rejects_nonfinite_or_wrong_shape(self):
        for value in ([float("nan")]*14,[0.]*13):
            edited = copy.deepcopy(self.artifact)
            edited["noise"]["residual_mean"] = value
            with self.assertRaisesRegex(ValueError, "noise.residual_mean"):
                validate_composite_artifact(edited)


if __name__ == "__main__":
    unittest.main()
