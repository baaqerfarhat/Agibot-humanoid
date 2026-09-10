"""Joint-fault diagnostics from recorded data and caller-controlled replay probes.

No simulator is imported or launched here. ``collect_local_probe`` accepts an evaluator
that MUST restore the same full simulator, controller, and random-generator state before
each call. It is deliberately not implemented as sim.set_state alone: OSC goals/history
can otherwise make a finite difference compare different experiments.

The response may be a one-step motion or a fixed-horizon trajectory flattened to a vector.
Choose its units/weights explicitly. A column-space residual describes that local response
and action interface, not whether a disturbance is globally "unmatched". Error reduction
on a replay is not a proof of contraction or task repair.

Examples (read stored artifacts only):
  python openpi/joint_diagnostics.py cells results/joint_map/cell_torque_{3,5}.json
  python openpi/joint_diagnostics.py telemetry results/jointmap/probe_torque_5_tel.jsonl
  python openpi/joint_diagnostics.py probe saved_local_probe.json
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import pathlib

import numpy as np


def _array(value, name, ndim=None):
    out = np.asarray(value, dtype=float)
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains nonfinite values")
    return out


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _mask(args, size):
    dims = args.get("corr_dims")
    if dims is None:
        return np.ones(size, dtype=bool)
    indices = [int(x) for x in dims.split(",")] if isinstance(dims, str) else list(dims)
    if len(set(indices)) != len(indices) or any(i < 0 or i >= size for i in indices):
        raise ValueError("invalid correction dimensions")
    return np.isin(np.arange(size), indices)


def cell_summary(path):
    """Paired outcomes, without treating Cartesian estimates as joint-torque truth."""
    data = json.loads(pathlib.Path(path).read_text())
    args = data.get("args", {})
    arms = data["arms"]
    keyed = {}
    summaries = {}
    for name in ("frozen_faulted", "adaptive"):
        arm = arms[name]
        outcomes = {}
        for row in arm["per_ep"]:
            key = (row["task"], row["init"])
            if key in outcomes:
                raise ValueError(f"{path}: duplicate episode key in {name}: {key}")
            outcomes[key] = bool(row["ok"])
        if len(outcomes) != arm["n"] or sum(outcomes.values()) != arm["successes"]:
            raise ValueError(f"{path}: inconsistent outcome totals for {name}")
        keyed[name] = outcomes
        summaries[name] = dict(successes=arm["successes"], episodes=arm["n"])
        if name == "adaptive" and arm.get("traj"):
            traces = [_array(t, "estimate trajectory", 2) for t in arm["traj"]]
            if len(traces) != arm["n"] or any(not len(t) for t in traces):
                raise ValueError("one nonempty estimate trajectory required per episode")
            mask = _mask(args, traces[0].shape[1])
            selected = np.concatenate([t[:, mask] for t in traces])
            clip = args.get("clip")
            summaries[name].update(
                median_estimate_peak_to_peak=np.median([np.ptp(t[:, mask], axis=0) for t in traces], axis=0),
                mean_final_estimate=np.mean([t[-1] for t in traces], axis=0),
                estimate_at_bound_fraction=(float(np.mean(np.abs(selected) >= float(clip) - 1e-8))
                                            if clip is not None and selected.size else None),
                note="Estimate trajectories alone do not measure applied corrections or model error.")
    if keyed["frozen_faulted"].keys() != keyed["adaptive"].keys():
        raise ValueError(f"{path}: arm episode sets differ")
    fixed = [key for key in keyed["adaptive"] if not keyed["frozen_faulted"][key] and keyed["adaptive"][key]]
    broken = [key for key in keyed["adaptive"] if keyed["frozen_faulted"][key] and not keyed["adaptive"][key]]
    opportunities = sum(keyed["frozen_faulted"].values())
    return dict(path=str(path), args=args, arms=summaries, fixed=fixed, broken=broken,
                regression_opportunities=opportunities,
                regression_rate=len(broken) / opportunities if opportunities else None)


def telemetry_summary(path):
    """Per-episode diagnostics; omit warmup and never subtract unrelated rollouts."""
    header = None
    groups = collections.defaultdict(list)
    for line in pathlib.Path(path).open():
        row = json.loads(line)
        if row.get("type") == "header":
            if header is not None:
                raise ValueError("multiple telemetry headers")
            header = row
        elif row.get("type") == "step" and row.get("phase") == "rollout":
            groups[(row["arm"], row["episode"], row["task"], row["init"])].append(row)
    if header is None or not groups:
        raise ValueError("one header and nonempty rollout telemetry required")
    config, args = header["config"], header["args"]
    inverse = _array(config["M_inv"], "M_inv", 2)
    bias = np.zeros(inverse.shape[0]) if config.get("bias") is None else _array(config["bias"], "bias", 1)
    summaries = []
    for key, rows in sorted(groups.items()):
        steps = [r["t"] for r in rows]
        if steps != sorted(set(steps)):
            raise ValueError(f"unordered or repeated rollout steps in {key}")
        residual = _array([r["r"] for r in rows], "residual", 2)
        measured = _array([r["measured"] for r in rows], "measured", 2)
        correction = _array([r["correction"] for r in rows], "correction", 2)[:, :inverse.shape[0]]
        estimate = _array([r["f_hat"] for r in rows], "estimate", 2)
        reading = residual @ inverse.T - bias
        mask = _mask(args, inverse.shape[0])
        selected = correction[:, mask]
        attempted = [r for r in rows if r.get("deadzone_fired") is not None]
        attenuation = [r["attenuation"] for r in rows if r.get("attenuation") is not None]
        clip = args.get("clip")
        width = min(30, len(rows))
        summaries.append(dict(
            arm=key[0], episode=key[1], task=key[2], init=key[3], rollout_steps=len(rows),
            success=any(bool(r.get("done")) for r in rows),
            residual_mean=residual.mean(axis=0), residual_rms=np.sqrt(np.mean(residual ** 2, axis=0)),
            equivalent_action_reading_mean=reading.mean(axis=0),
            equivalent_action_reading_std=reading.std(axis=0),
            measured_motion_mean=measured.mean(axis=0),
            correction_applied=bool(np.any(correction != 0)),
            correction_rms=float(np.sqrt(np.mean(np.sum(selected ** 2, axis=1)))),
            correction_peak_to_peak=np.ptp(correction, axis=0),
            correction_early_to_late=correction[-width:].mean(axis=0) - correction[:width].mean(axis=0),
            final_estimate=estimate[-1],
            estimate_at_bound_fraction=(float(np.mean(np.abs(estimate[:, mask]) >= float(clip) - 1e-8))
                                        if clip is not None and mask.any() else None),
            deadzone_fraction=(sum(bool(r["deadzone_fired"]) for r in attempted) / len(attempted)
                               if attempted else None),
            attenuation_median=float(np.median(attenuation)) if attenuation else None))
    return dict(path=str(path), args=args, episodes=summaries,
                notes=["M_inv r minus bias is an equivalent Cartesian action reading, not true joint torque.",
                       "Measured motion includes nominal policy motion; it is not induced fault displacement.",
                       "Different closed-loop trajectories are not same-state causal comparisons.",
                       "Only phase=rollout is scored; warmup is omitted."])


def bounded_correction(response_map, disturbance, bound):
    """Exact small box-constrained least squares by active-set enumeration (<=6 inputs)."""
    B = _array(response_map, "response_map", 2)
    d = _array(disturbance, "disturbance", 1)
    n = B.shape[1]
    limits = np.broadcast_to(_array(bound, "bound"), (n,))
    if B.shape[0] != len(d) or not 1 <= n <= 6 or np.any(limits < 0):
        raise ValueError("compatible response rows, one to six inputs, and nonnegative bounds required")
    best = None
    for status in itertools.product((-1, 0, 1), repeat=n):
        status = np.asarray(status)
        free = status == 0
        c = status * limits
        if free.any():
            c[free] = np.linalg.lstsq(B[:, free], -d - B[:, ~free] @ c[~free], rcond=None)[0]
        if np.any(np.abs(c) > limits + 1e-10):
            continue
        c = np.clip(c, -limits, limits)
        error = float(np.linalg.norm(d + B @ c))
        candidate = (error, float(np.linalg.norm(c)), c)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("no feasible box-constrained correction")
    return best[2], best[0]


def span_report(response_map, disturbance, bound=None):
    """Projection in the PROVIDED response coordinates; no implicit unit normalization."""
    B = _array(response_map, "response_map", 2)
    d = _array(disturbance, "disturbance", 1)
    if B.shape[0] != len(d):
        raise ValueError("response rows must match disturbance")
    correction = np.linalg.lstsq(B, -d, rcond=None)[0]
    remaining = d + B @ correction
    length = float(np.linalg.norm(d))
    result = dict(rank=int(np.linalg.matrix_rank(B)), singular_values=np.linalg.svd(B, compute_uv=False),
                  disturbance_norm=length, unconstrained_correction=correction,
                  unconstrained_remaining_norm=float(np.linalg.norm(remaining)),
                  unconstrained_remaining_fraction=float(np.linalg.norm(remaining)) / length if length else None)
    if bound is not None:
        c, error = bounded_correction(B, d, bound)
        result.update(bound=bound, bounded_correction=c, bounded_remaining_norm=error,
                      bounded_remaining_fraction=error / length if length else None)
    return result


def _response(evaluation):
    return _array(evaluation["response"], "response", 1)


def collect_local_probe(evaluate, *, dimensions=range(6), step=0.01, torque=5.0,
                        bound=0.3, candidate=None, metadata=None):
    """Evaluate local authority. Caller supplies an independently restored simulator.

    evaluate(action_delta_6, fault_torque, opposing_torque) -> dict(response=[...], ...).
    The two torques are separate arguments so the oracle exercises the caller's fault and
    cancellation paths. Every invocation, including repeats, MUST start from the identical
    full state and use identical subsequent open-loop policy commands and integration steps.
    Optional ``snapshot_id`` in evaluations is checked for agreement. A response-repeat
    discrepancy is reported and prevents calling the span result a causal intervention.
    """
    dimensions = tuple(dimensions)
    if not dimensions or len(set(dimensions)) != len(dimensions) or any(i not in range(6) for i in dimensions):
        raise ValueError("dimensions must be distinct Cartesian input indices 0..5")
    steps = np.broadcast_to(_array(step, "step"), (len(dimensions),))
    if np.any(steps <= 0) or not np.isfinite(torque):
        raise ValueError("positive finite probe steps and finite torque required")
    def run(delta, fault, opposite):
        result = evaluate(np.asarray(delta, float).copy(), float(fault), float(opposite))
        _response(result)
        return result
    zero = np.zeros(6)
    result = dict(schema_version=1, metadata=metadata or {}, dimensions=dimensions, steps=steps,
                  torque=torque, bound=bound, healthy=run(zero, 0, 0),
                  faulted=run(zero, torque, 0), torque_oracle=run(zero, torque, -torque),
                  plus=[], minus=[])
    for i, h in zip(dimensions, steps):
        delta = zero.copy(); delta[i] = h
        result["plus"].append(run(delta, torque, 0))
        result["minus"].append(run(-delta, torque, 0))
    result["faulted_repeat"] = run(zero, torque, 0)
    if candidate is not None:
        candidate = _array(candidate, "candidate", 1)
        if candidate.shape != (6,):
            raise ValueError("candidate must contain six Cartesian action corrections")
        result["candidate"] = dict(action=candidate, evaluation=run(candidate, torque, 0))
    report = analyze_local_probe(result)
    # The finite-difference model supplies an oracle candidate; a fresh nonlinear replay
    # determines whether that candidate actually reduces the disturbance.
    c = report["all_probed_inputs"].get("bounded_correction",
                                             report["all_probed_inputs"]["unconstrained_correction"])
    delta = zero.copy(); delta[list(dimensions)] = c
    result["local_oracle"] = dict(action=delta, evaluation=run(delta, torque, 0))
    return result


def analyze_local_probe(data):
    dimensions = list(data["dimensions"])
    steps = _array(data["steps"], "steps", 1)
    if len(dimensions) != len(steps) or len(set(dimensions)) != len(dimensions) or np.any(steps <= 0):
        raise ValueError("distinct dimensions and matching positive steps required")
    if len(data["plus"]) != len(steps) or len(data["minus"]) != len(steps):
        raise ValueError("one plus/minus response pair required per input dimension")
    evaluations = [data[k] for k in ("healthy", "faulted", "torque_oracle", "faulted_repeat")]
    evaluations += data["plus"] + data["minus"]
    evaluations += [data[k]["evaluation"] for k in ("candidate", "local_oracle") if k in data]
    responses = [_response(e) for e in evaluations]
    if any(r.shape != responses[0].shape for r in responses):
        raise ValueError("all replay responses must have the same shape")
    snapshots = [e.get("snapshot_id") for e in evaluations]
    if any(s is not None for s in snapshots) and (any(s is None for s in snapshots) or len(set(snapshots)) != 1):
        raise ValueError("replays identify different or missing initial snapshots")
    healthy, center = _response(data["healthy"]), _response(data["faulted"])
    d = center - healthy
    B = np.column_stack([(_response(p) - _response(m)) / (2 * h)
                         for p, m, h in zip(data["plus"], data["minus"], steps)])
    curvature = np.column_stack([(_response(p) + _response(m)) / 2 - center
                                 for p, m in zip(data["plus"], data["minus"])])
    result = dict(metadata=data.get("metadata", {}), dimensions=dimensions,
                  disturbance=d, response_map=B,
                  repeated_center_difference=float(np.linalg.norm(_response(data["faulted_repeat"]) - center)),
                  opposing_torque_error=float(np.linalg.norm(_response(data["torque_oracle"]) - healthy)),
                  probe_midpoint_error_norms=np.linalg.norm(curvature, axis=0),
                  snapshot_identity_recorded=all(s is not None for s in snapshots),
                  all_probed_inputs=span_report(B, d, data.get("bound")), subsets={}, replays={},
                  notes=["Projection uses the stated response coordinates and probe horizon.",
                         "Local response fit does not prove global matching, contraction, or task repair.",
                         "Check repeat, opposing-torque, midpoint and nonlinear replay errors before interpreting authority."])
    for label, indices in (("translation", (0, 1, 2)), ("rotation", (3, 4, 5))):
        columns = [i for i, dim in enumerate(dimensions) if dim in indices]
        if columns:
            bound = data.get("bound")
            if bound is not None:
                bound = np.broadcast_to(np.asarray(bound), (len(dimensions),))[columns]
            result["subsets"][label] = span_report(B[:, columns], d, bound)
    for name in ("candidate", "local_oracle"):
        if name not in data:
            continue
        replay = data[name]
        remaining = _response(replay["evaluation"]) - healthy
        length = float(np.linalg.norm(d))
        correction = _array(replay["action"], "replayed action", 1)
        predicted = d + B @ correction[dimensions]
        result["replays"][name] = dict(action=correction, remaining_norm=float(np.linalg.norm(remaining)),
            error_ratio=float(np.linalg.norm(remaining)) / length if length else None,
            linear_prediction_error=float(np.linalg.norm(remaining - predicted)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cells", "telemetry", "probe"))
    parser.add_argument("paths", nargs="+", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()
    if args.out is not None and args.out.resolve() in {p.resolve() for p in args.paths}:
        parser.error("--out must differ from input paths")
    analyze = {"cells": cell_summary, "telemetry": telemetry_summary,
               "probe": lambda p: analyze_local_probe(json.loads(p.read_text()))}[args.mode]
    report = dict(schema_version=1, mode=args.mode, reports=[analyze(p) for p in args.paths])
    payload = json.dumps(report, default=_json, indent=2, allow_nan=False) + "\n"
    if args.out is None:
        print(payload, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)


if __name__ == "__main__":
    main()
