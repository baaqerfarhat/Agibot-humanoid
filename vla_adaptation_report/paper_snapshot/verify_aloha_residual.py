#!/usr/bin/env python3
"""Reproduce historical healthy ALOHA FIR fit and raw-residual energy shares.

Run from any directory; no simulator or policy is loaded. This is an in-sample
description of the existing healthy log, not a held-out prediction assessment.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "openpi"))
import aloha_adapt  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signal_summary(values):
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "distinct_after_rounding_to_0.01": int(len(np.unique(np.round(values, 2)))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=ROOT / "results/aloha/healthy_log.json")
    parser.add_argument("--out", type=Path, default=ROOT / "paper/aloha_residual_receipt.json")
    args = parser.parse_args()
    log_path, output_path = args.log.resolve(), args.out.resolve()
    source_paths = [Path(__file__).resolve(), Path(aloha_adapt.__file__).resolve(), log_path]
    sources_before = {str(path): sha256(path) for path in source_paths}
    episodes = json.loads(log_path.read_text())
    W, fit_r2 = aloha_adapt.fit_plant(log_path)
    k, n = aloha_adapt.K_FIR, aloha_adapt.NJ
    residuals, targets, all_u, all_q, episode_samples = [], [], [], [], []
    for episode in episodes:
        u, q = np.asarray(episode["u"], float), np.asarray(episode["q"], float)
        if (u.ndim != 2 or u.shape != q.shape or u.shape[1] != n or
                len(u) <= k or not np.isfinite(u).all() or not np.isfinite(q).all()):
            raise ValueError("Expected finite, aligned 14-channel u/q episodes with FIR history")
        # q[t] is the measured position AFTER u[t]. Match fit_plant's exact
        # seven-tap newest-first design and exclude indices 0,...,5 per episode.
        predicted = np.column_stack([
            np.asarray([np.r_[u[t-k:t+1, j][::-1], 1.0]
                        for t in range(k, len(u))]) @ W[j]
            for j in range(n)
        ])
        residuals.append(q[k:] - predicted)
        targets.append(q[k:])
        all_u.append(u)
        all_q.append(q)
        episode_samples.append(len(u) - k)
    residual, target = np.concatenate(residuals), np.concatenate(targets)
    sse = np.square(residual).sum(axis=0)
    centered_energy = np.square(target - target.mean(axis=0)).sum(axis=0)
    recomputed_r2 = 1.0 - sse / np.maximum(centered_energy, 1e-12)
    if not np.allclose(recomputed_r2, fit_r2, rtol=1e-11, atol=1e-12):
        raise AssertionError("Residual reconstruction disagrees with actual fit_plant R2")
    shares = sse / sse.sum()
    arms = list(range(6)) + list(range(7, 13))
    u, q = np.concatenate(all_u), np.concatenate(all_q)
    sources_after = {str(path): sha256(path) for path in source_paths}
    if sources_after != sources_before:
        raise RuntimeError("An input or implementation source changed during reanalysis")
    receipt = {
        "schema_version": 1,
        "analysis": "Historical ALOHA healthy FIR in-sample residual decomposition",
        "sources_sha256": sources_before,
        "fit_function": {
            "module": "openpi/aloha_adapt.py",
            "name": "fit_plant",
            "first_line": inspect.getsourcelines(aloha_adapt.fit_plant)[1],
            "target": "measured post-command position q[t], not displacement",
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "reproduce": "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/verify_aloha_residual.py",
        "window": {
            "episodes": len(episodes),
            "fir_lags": k,
            "fir_taps": k + 1,
            "included_indices": "t=6,...,len(episode)-1 in every episode",
            "samples_per_episode": episode_samples,
            "total_samples": len(residual),
            "aggregation": "sum squared raw residuals across all included samples, then divide by total across 14 channels",
            "fit_and_evaluation_data": "same healthy episodes and same included samples (in-sample)",
        },
        "r2_per_channel": fit_r2.tolist(),
        "residual_squared_energy_per_channel": sse.tolist(),
        "residual_squared_energy_fraction_per_channel": shares.tolist(),
        "claims": {
            "right_gripper_channel": 13,
            "right_gripper_r2": float(fit_r2[13]),
            "arm_only_min_r2": float(fit_r2[arms].min()),
            "arm_only_max_r2": float(fit_r2[arms].max()),
            "right_gripper_squared_residual_percent": float(100 * shares[13]),
            "left_six_squared_residual_percent": float(100 * shares[:6].sum()),
        },
        "right_gripper_signals": {
            "window": "all logged steps, including initial six steps",
            "command": signal_summary(u[:, 13]),
            "measured_position": signal_summary(q[:, 13]),
        },
        "interpretation": [
            "The squared-energy percentage is a share of sum_t ||r_t||^2, not a percentage of ||r_t||.",
            "Raw coordinates combine arm radians and environment gripper coordinates; these shares describe the implemented unweighted norm, not unit-invariant importance.",
            "Neither held-out prediction accuracy nor causal suppression of fault repair follows from this healthy-data decomposition.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"receipt": str(output_path), "samples": len(residual), **receipt["claims"]}, indent=2))


if __name__ == "__main__":
    main()
