"""Reconstruct composite qualification from hash-matched healthy source files.

This is a launch-time consistency guard, not a policy or simulator operation.
It does not modify calibration or pick gains. Portable copies may supply explicit
source paths; their bytes must match the hashes already stored in the artifact.
An artifact is not an authenticated preregistration: changing its sources,
settings AND hashes together still requires external provenance review.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _source(artifact, name, override=None, artifact_path=None):
    source = artifact["source"] if name == "log" else artifact["source"]["sensitivity"]
    expected = source["sha256"]
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("composite qualification requires the recorded %s SHA256" % name)
    candidates = [Path(override)] if override is not None else [Path(source["path"])]
    if override is None:
        declared = Path(artifact["settings"]["log" if name == "log" else "openloop"])
        if declared.is_absolute():
            candidates.append(declared)
        else:
            # The preparation CLI historically recorded repo-relative settings.
            candidates.append(Path(__file__).resolve().parents[1] / declared)
            if artifact_path is not None:
                candidates.append(Path(artifact_path).resolve().parent / declared)
    checked = []
    for candidate in dict.fromkeys(path.resolve() for path in candidates):
        if not candidate.is_file():
            checked.append("%s (missing)" % candidate)
            continue
        raw = candidate.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected:
            checked.append("%s (SHA256 mismatch)" % candidate)
            continue
        return json.loads(raw), dict(path=str(candidate), sha256=digest)
    raise ValueError("composite qualification cannot find hash-matched %s source; checked %s. "
                     "Supply --reference-source-%s with identical source bytes." %
                     (name, "; ".join(checked), "log" if name == "log" else "sensitivity"))


def _same(actual, expected, name):
    """Allow numerical-library roundoff; reject stale semantic/matrix contents."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError("composite qualification mismatch: " + name)
        for key, value in expected.items():
            if key not in actual:
                raise ValueError("composite qualification missing %s.%s" % (name, key))
            _same(actual[key], value, "%s.%s" % (name, key))
        return
    if expected is None or isinstance(expected, (str, bool)):
        if actual != expected:
            raise ValueError("composite qualification mismatch: " + name)
        return
    if isinstance(expected, (list, tuple)) and any(isinstance(value, (str, dict)) for value in expected):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError("composite qualification mismatch: " + name)
        for index, (left, right) in enumerate(zip(actual, expected)):
            _same(left, right, "%s[%d]" % (name, index))
        return
    try:
        left, right = np.asarray(actual, float), np.asarray(expected, float)
    except (TypeError, ValueError):
        raise ValueError("composite qualification mismatch: " + name)
    # Per-entry relative tolerance plus a very small matrix-scale roundoff floor.
    # R and Q can be tiny; an absolute 1e-8 covariance tolerance is inappropriate.
    floor = max(1e-24, float(np.max(np.abs(right))) * 1e-12) if right.size else 1e-24
    if left.shape != right.shape or not np.all(np.isfinite(left)) or not np.all(
            np.abs(left - right) <= 1e-7 * np.abs(right) + floor):
        raise ValueError("composite qualification mismatch: " + name)


def _same_fir_residual_mean(actual, expected, episodes, fit_indices):
    """Allow cancellation roundoff only in this unused OLS diagnostic.

    The per-channel FIR includes an intercept, so its training residual mean
    is theoretically zero. Subtracting two order-one positions can leave a
    mean around 1e-15 that changes across LAPACK/NumPy builds. Relative tolerance
    against that near-zero mean is inappropriate. The absolute floor below is
    64 float64 epsilons times the source-position magnitude (at least one),
    independently for each channel. This does not change any runtime matrix,
    observer bias, prediction error, or qualification threshold.
    """
    expected = np.asarray(expected, float)
    actual = np.asarray(actual, float)
    targets = np.concatenate([np.asarray(episodes[index]["q"], float) for index in fit_indices])
    scale = np.maximum(1.0, np.max(np.abs(targets), axis=0))
    floor = 64*np.finfo(np.float64).eps*scale
    if (actual.shape != expected.shape or expected.shape != scale.shape
            or not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected))
            or not np.all(np.abs(actual-expected) <= 1e-7*np.abs(expected)+floor)):
        raise ValueError("composite qualification mismatch: noise.residual_mean")


def validate_composite_artifact(artifact, *, artifact_path=None, source_log=None,
                                source_sensitivity=None):
    """Validate model, reference gate, shared observer and every declared gain.

    The public runners separately require their runtime mask/dt/damping/rate to
    equal the artifact's validated settings. No generic or weighted baseline
    needs this position-reference qualification.
    """
    import composite_observer as composite
    from prepare_composite_reference import fit_joint_fir_noise, tracking_grid

    try:
        if not artifact["qualification"]["allowed"]:
            raise ValueError("composite reference artifact failed qualification")
        settings = artifact["settings"]
        healthy, log_source = _source(artifact, "log", source_log, artifact_path)
        sensitivity, sensitivity_source = _source(artifact, "sensitivity", source_sensitivity, artifact_path)
        episodes = healthy.get("episodes") if isinstance(healthy, dict) else healthy
        model, report = composite.fit_joint_reference(episodes, settings["state_indices"],
            command_indices=settings["command_indices"], fit_episode_indices=settings["fit_episodes"],
            validation_episode_indices=settings["validation_episodes"], ridge=settings["ridge"])
        allowed, reasons = composite.qualify_reference(report, settings["max_validation_rmse"],
                                                       settings["max_condition_number"])
        if not allowed:
            raise ValueError("reconstructed composite reference failed qualification: " + str(reasons))
        _same(artifact["model"], model, "model")
        _same(artifact["report"], report, "report")
        _same(artifact["qualification"], dict(allowed=allowed, reasons=reasons), "qualification")
        noise = fit_joint_fir_noise(episodes, report["training_episode_indices"], settings["fir_lag"])
        M = np.asarray(sensitivity["M"], float)
        tracking = tracking_grid(model, M, noise, settings["correction_indices"],
            settings["tracking_strengths"], settings["dt"], gamma=settings["gamma"],
            damping=settings["damping"], covariance_steps=settings["covariance_steps"])
        _same(artifact["noise"], {key:value for key,value in noise.items()
                                if key != "residual_mean"}, "noise")
        _same_fir_residual_mean(artifact["noise"]["residual_mean"],noise["residual_mean"],
                               episodes,report["training_episode_indices"])
        _same(artifact["observer"], dict(W=noise["W"], M=M, Q=tracking["Q_step"], R=noise["R"],
            dt=settings["dt"], gamma=settings["gamma"], bias=None,
            fit_episode_indices=report["training_episode_indices"]), "observer")
        keys = ("H", "Q_step", "Q_rate", "R", "dt", "gamma", "damping", "correction_mask",
                "tracking_map", "metric", "steady_covariance", "steady_gain", "steady_effective_gain", "scale")
        _same(artifact["tracking"], {key: tracking[key] for key in keys}, "tracking")
        old_candidates, new_candidates = artifact["tracking"]["candidates"], tracking["candidates"]
        if len(old_candidates) != len(new_candidates):
            raise ValueError("composite qualification mismatch: candidate grid length")
        for old, new in zip(old_candidates, new_candidates):
            values = {key: new[key] for key in ("strength", "tracking_rate", "allowed", "reasons")}
            values.update(reference_qualified=True, qualified_for_experiment=bool(new["allowed"]))
            _same(old, values, "tracking.candidate")
            # The exact iteration count can differ across numerical libraries;
            # recomputed eligibility and the worst radius must still agree.
            if "maximum_spectral_radius" in new:
                _same(old["maximum_spectral_radius"], new["maximum_spectral_radius"], "tracking.maximum_spectral_radius")
        return dict(validated=True, sources=dict(log=log_source, sensitivity=sensitivity_source),
            checks="healthy source SHA256, refitted model/reference gate, training-only FIR/noise, shared matrices, gain reconstruction",
            limitation="internal consistency with declared sources/settings; external preregistration and artifact identity remain separately audited")
    except (KeyError, TypeError, np.linalg.LinAlgError) as error:
        raise ValueError("incomplete or invalid composite qualification artifact: %s" % error) from error
