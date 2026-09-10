"""Validate complete frozen ALOHA allocations, select candidates, and score tests.

The allocation manifest, candidate bank and every runtime source are byte-bound.
Study JSON is authoritative; per-comparison views are never pooled as new off
episodes. Aggregate inference swaps complete seed clusters across conditions.
Numerical ordering and statistical evidence are reported separately.
"""
from __future__ import annotations

import argparse
import datetime
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np

FAMILIES = ("legacy", "dob", "rls", "integral_calibrated", "kalman", "composite")
SECONDARY = ("legacy", "dob", "rls", "integral_calibrated")
MIN_DRAWS = 200000


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def _path(value, base, mapping):
    path = Path(mapping.get(str(value), str(value)))
    return path if path.is_absolute() else base / path


def _unique(values, label):
    require(isinstance(values, list) and bool(values) and len(values) == len(set(values)),
            label + " must be a nonempty list without duplicates")
    return values


def _state(row):
    state = row["initial_state"]
    require(state.get("pairing_valid") is True, "episode failed full-state pairing")
    arrays = [np.asarray(state[key], dtype="<f8") for key in ("qpos", "qvel")]
    require(all(x.ndim == 1 and len(x) > 0 and np.isfinite(x).all() for x in arrays),
            "invalid complete physical reset state")
    h = hashlib.sha256()
    h.update(json.dumps([list(x.shape) for x in arrays]).encode())
    for x in arrays:
        h.update(x.tobytes())
    require(state.get("sha256") == h.hexdigest(), "reset state hash mismatch")
    return arrays


def _energy(arm, index, config, plan, definition, metadata):
    trace = np.asarray(arm["traj"][index], float)
    require(trace.ndim == 2 and trace.shape[1] == 14 and np.isfinite(trace).all()
            and 0 < len(trace) <= plan["max_steps"], "invalid estimate trajectory")
    diagnostic = arm["diagnostics"][index]
    steps = diagnostic["physical_steps"]
    require(steps == len(trace), "physical/estimate step count mismatch")
    family = config["family"]
    if family == "off":
        correction = np.zeros_like(trace)
    elif family == "oracle":
        value = np.asarray(definition["fault_vec"], float).copy()
        if definition["kind"] == "torque":
            joint = metadata[definition["joint"]]
            value[joint["command_coordinate"]] = definition["torque"] / joint["command_to_torque_gain"]
        correction = np.tile(value, (steps, 1))
    else:
        correction = np.vstack([np.zeros(14), trace[:-1]])
        correction[:, [j for j in range(14) if j not in plan["correction_indices"]]] = 0
    reconstructed = float(np.sum(correction * correction))
    actual = diagnostic.get("applied_correction_energy")
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and np.isfinite(actual) and actual >= 0, "missing/invalid actual applied correction energy")
    require(diagnostic.get("applied_correction_steps") == steps, "applied correction step count mismatch")
    require(np.isclose(actual, reconstructed, rtol=1e-10, atol=1e-12),
            "applied correction energy does not reconstruct from command timing")
    return float(actual), steps


def load_allocation(manifest_path, source_map=None):
    """Reject partial, duplicated, stale or physically unpaired studies."""
    from run_aloha_tuning import normalize_plan, select_candidates
    manifest_path = Path(manifest_path).resolve()
    base, mapping = manifest_path.parent, source_map or {}
    manifest = read(manifest_path)
    require(manifest.get("schema_version") == 1, "unknown analysis manifest schema")
    require(manifest.get("phase") in ("search", "validation", "confirmation"), "unknown analysis phase")
    conditions = _unique(manifest.get("conditions"), "conditions")
    seeds = _unique(manifest.get("expected_seeds"), "expected_seeds")
    names = _unique(manifest.get("expected_candidates"), "expected_candidates")
    require("healthy" in conditions and len(conditions) > 1 and "off" in names,
            "allocation needs healthy, faults and a shared off arm")
    require(all(isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds), "invalid seeds")
    families = manifest.get("families", {})
    require(set(families) == set(FAMILIES), "all six method families must be declared")
    for family in FAMILIES:
        _unique(families[family], "family " + family)
    require(set().union(*(set(v) for v in families.values())) == set(names) - {"off", "oracle"},
            "family candidate allocation differs from executed candidates")
    bank_info = manifest["candidate_bank"]
    bank_path = _path(bank_info["path"], base, mapping)
    require(digest(bank_path) == bank_info["sha256"], "candidate bank hash mismatch")
    bank = read(bank_path)
    require(not set(seeds).intersection(bank.get("settings", {}).get("calibration_seeds", [])),
            "evaluation seeds overlap healthy calibration")
    candidates, recipes = {}, {}
    for row in bank["candidates"]:
        require(row["name"] not in candidates, "duplicate candidate bank name")
        if row["qualification"]["allowed"]:
            candidates[row["name"]] = dict(row["parameters"])
            recipes[row["name"]] = row.get("recipe", {})
    require(set(names) <= set(candidates), "unknown or rejected candidate in allocation")
    for family, members in families.items():
        for name in members:
            actual = candidates[name]["family"]
            alias = family == "composite" and actual == "kalman" and candidates[name]["tracking_rate"] == 0
            require(actual == family or alias, "invalid cross-family alias")
    if manifest["phase"] == "confirmation":
        require(all(len(v) == 1 for v in families.values()), "confirmation needs one candidate per family")
    required = manifest.get("sources", {})
    require(isinstance(required, dict) and bool(required), "manifest must bind required runtime sources")
    verified = {}

    def check_source(path, sha):
        if path in verified:
            require(verified[path] == sha, "conflicting source hashes across studies")
        else:
            require(digest(_path(path, base, mapping)) == sha, "runtime source hash mismatch: " + path)
            verified[path] = sha

    for path, sha in required.items():
        check_source(path, sha)
    registry_info = manifest.get("policy_server_registry")
    registry = None
    if registry_info:
        registry_path = _path(registry_info["path"], base, mapping)
        require(digest(registry_path) == registry_info["sha256"], "policy server registry hash mismatch")
        registry = read(registry_path)
        require(registry.get("schema_version") == 1 and registry.get("endpoints"), "invalid policy server registry")
    selection = manifest.get("selection_record")
    if manifest["phase"] == "confirmation":
        require(isinstance(selection, dict), "confirmation must bind prior selection")
    selection_content = None
    if selection:
        selection_path = _path(selection["path"], base, mapping)
        require(digest(selection_path) == selection["sha256"], "selection record hash mismatch")
        selection_content = read(selection_path)
        require(selection_content.get("kind") == "candidate_selection", "invalid prior selection record")
        require(selection_content["candidate_bank_sha256"] == bank_info["sha256"], "selection uses another bank")
        previous_seeds = selection_content.get("allocation", {}).get("expected_seeds")
        require(isinstance(previous_seeds, list), "prior selection lacks its seed allocation")
        require(not set(seeds).intersection(previous_seeds), "new stage reuses prior selection seeds")
        for family in FAMILIES:
            require(set(families[family]) <= set(selection_content["selected"][family]),
                    "allocation includes a candidate not selected in the previous stage")
    outcomes, energies, states, provenance = {}, {}, {}, []
    study_ids, paths_seen = set(), set()
    definitions_seen, endpoints_by_seed = {}, {}
    studies = manifest.get("studies", [])
    require(bool(studies), "manifest has no studies")
    for descriptor in studies:
        path = _path(descriptor["path"], base, mapping)
        require(str(path.resolve()) not in paths_seen, "duplicate study source")
        paths_seen.add(str(path.resolve()))
        study_bytes = path.read_bytes()
        study_sha = hashlib.sha256(study_bytes).hexdigest()
        study = json.loads(study_bytes)
        del study_bytes
        if "sha256" in descriptor:
            require(study_sha == descriptor["sha256"], "study artifact hash mismatch")
        require(study.get("status") == "complete" and study.get("completed_source_recheck") is True,
                "study is incomplete or lacks its final source check")
        identifier = study["study_id"]
        require(identifier not in study_ids, "duplicate study ID")
        study_ids.add(identifier)
        plan_path = _path(descriptor["plan_path"], base, mapping)
        require(digest(plan_path) == descriptor["plan_sha256"], "plan hash mismatch")
        plan = normalize_plan(read(plan_path))
        require(study["plan"] == plan, "recorded plan differs from frozen plan")
        require(plan["bank_sha256"] == bank_info["sha256"], "study plan uses another bank")
        expected_stage = "confirmation" if manifest["phase"] == "confirmation" else "tuning"
        require(plan["stage"] == study["stage"] == expected_stage, "study stage mismatch")
        for key in ("seed", "episodes", "max_steps"):
            require(study["args"].get(key) == plan[key], "runtime argument differs from plan: " + key)
        require(study["args"].get("corr") == plan["correction_indices"], "runtime correction interface differs")
        require(all(study["args"].get(key) is True for key in
            ("reset_estimate", "reset_covariance", "reset_reference")), "state is carried between episodes")
        if selection and manifest["phase"] == "confirmation":
            require(plan.get("selection_record", {}).get("sha256") == selection["sha256"],
                    "runner plan does not bind prior selection")
        local_names = plan["candidate_names"]
        local_seeds = list(range(plan["seed"], plan["seed"] + plan["episodes"]))
        local_conditions = [row["name"] for row in plan["conditions"]]
        require(set(local_names) <= set(names) and set(local_seeds) <= set(seeds)
                and set(local_conditions) <= set(conditions), "study outside frozen allocation")
        hashes = study["source_hashes"]
        require(all(hashes.get(key) == value for key, value in required.items()), "study omitted required source binding")
        require(hashes.get(study["args"]["plan"]) == descriptor["plan_sha256"], "study omitted its plan source hash")
        require(hashes.get(study["args"]["candidate_bank"]) == bank_info["sha256"], "study omitted bank source hash")
        for source, sha in hashes.items():
            check_source(source, sha)
        run_config_provenance = None
        endpoint = plan.get("metadata", {}).get("policy_endpoint")
        if endpoint is not None:
            require(manifest["phase"] in ("validation", "confirmation"),
                    "additional endpoint allocation is limited to later stages")
            require(isinstance(endpoint, dict) and isinstance(endpoint.get("host"), str)
                    and isinstance(endpoint.get("port"), int) and not isinstance(endpoint["port"], bool)
                    and 0 < endpoint["port"] < 65536, "invalid declared policy endpoint")
            registry_sha = plan["metadata"].get("policy_registry_sha256")
            require(isinstance(registry_sha, str) and len(registry_sha) == 64, "plan lacks policy registry binding")
            if registry is not None:
                require(registry_sha == registry_info["sha256"] and endpoint in registry["endpoints"],
                        "declared endpoint differs from bound policy registry")
            run_config_path = _path(descriptor.get("run_config_path", str(path.parent / "run_config.json")), base, mapping)
            require(run_config_path.is_file(), "missing run configuration for declared policy endpoint")
            run_config_bytes = run_config_path.read_bytes()
            run_config_sha = hashlib.sha256(run_config_bytes).hexdigest()
            run_config = json.loads(run_config_bytes)
            del run_config_bytes
            if "run_config_sha256" in descriptor:
                require(run_config_sha == descriptor["run_config_sha256"], "run configuration artifact hash mismatch")
            require(run_config.get("plan") == plan, "run configuration plan differs from frozen plan")
            require(run_config.get("source_hashes") == hashes, "run configuration source hashes differ from study")
            require(run_config.get("args", {}).get("host") == endpoint["host"]
                    and run_config["args"].get("port") == endpoint["port"],
                    "actual CLI endpoint differs from frozen policy endpoint")
            for key in ("plan", "candidate_bank", "telemetry"):
                require(run_config["args"].get(key) == study["args"][key],
                        "run configuration path differs from study: " + key)
            for seed in local_seeds:
                assignment = dict(endpoint=endpoint, registry_sha256=registry_sha)
                if seed in endpoints_by_seed:
                    require(endpoints_by_seed[seed] == assignment,
                            "conditions for one seed use different policy endpoints")
                endpoints_by_seed[seed] = assignment
            run_config_provenance = dict(path=str(run_config_path), sha256=run_config_sha,
                policy_endpoint=endpoint, policy_registry_sha256=registry_sha)
        # select_candidates reconstructs effective defaults and explicit matrices.
        validated = {"candidates": candidates}
        expected_configs = select_candidates(plan, validated)
        require(study["arm_configs"] == expected_configs, "runtime configuration differs from candidate bank")
        require(set(study["conditions"]) == set(local_conditions), "missing/extra study condition")
        expected_n = len(local_names) * len(local_conditions) * len(local_seeds)
        pairing = study["pairing"]
        require(pairing.get("valid_so_far") is True and pairing.get("checked_states") == expected_n
                and set(pairing.get("fields", [])) == {"full_qpos", "full_qvel"}, "incomplete full-state pairing audit")
        require(pairing["tolerance"] == plan["state_tolerance"], "pairing tolerance differs from plan")
        schedule_keys = []
        for batch in study["schedule"]:
            require(len(batch["arms"]) == len(set(batch["arms"])) and set(batch["arms"]) == set(local_names),
                    "invalid scheduled candidate order")
            schedule_keys.extend((batch["condition"], name, batch["seed"]) for name in batch["arms"])
        expected_keys = {(c, n, s) for c in local_conditions for n in local_names for s in local_seeds}
        require(len(schedule_keys) == len(set(schedule_keys)) and set(schedule_keys) == expected_keys,
                "schedule has duplicates or missing allocation rows")
        plan_definitions = {row["name"]: row for row in plan["conditions"]}
        for condition, cohort in study["conditions"].items():
            definition = cohort["definition"]
            require(definition == plan_definitions[condition], "condition changed after plan lock")
            require(cohort["shared_control_id"] == f"{identifier}:{condition}:off", "invalid shared off identity")
            if condition in definitions_seen:
                require(definitions_seen[condition] == definition, "condition differs across studies")
            definitions_seen[condition] = definition
            if condition == "healthy":
                require(definition["kind"] == "offset" and not np.any(definition["fault_vec"]),
                        "healthy condition injects a disturbance")
            require(set(cohort["arms"]) == set(local_names), "missing/extra candidate arm")
            for name, arm in cohort["arms"].items():
                require(arm["n"] == len(local_seeds) and len(arm["per_ep"]) == len(local_seeds), "incomplete arm")
                require(all(len(arm[field]) == len(local_seeds) for field in ("traj", "f_hat", "f_true", "diagnostics")),
                        "incomplete per-episode trajectory/diagnostics")
                successes, arm_seeds = 0, set()
                for i, row in enumerate(arm["per_ep"]):
                    seed = row["actual_seed"]
                    require(seed in local_seeds and seed not in arm_seeds, "duplicate/missing arm seed")
                    arm_seeds.add(seed)
                    require(row["task"] == 0 and row["init"] == row["episode"] == seed - plan["seed"],
                            "episode key and actual seed disagree")
                    require(isinstance(row["ok"], bool), "success outcome must be boolean")
                    key = (condition, name, seed)
                    require(key not in outcomes, "duplicate global condition/candidate/seed")
                    snapshot = _state(row)
                    if seed in states:
                        require(all(x.shape == y.shape and np.max(np.abs(x-y)) <= plan["state_tolerance"]
                            for x, y in zip(snapshot, states[seed])), "physical reset mismatch across arms/conditions/studies")
                    else:
                        states[seed] = snapshot
                    outcomes[key] = int(row["ok"])
                    energies[key] = _energy(arm, i, expected_configs[name], plan, definition, study["robot_joints"])
                    successes += int(row["ok"])
                require(successes == arm["successes"], "stored success total disagrees with outcomes")
        telemetry_path = _path(descriptor.get("telemetry_path", study["args"]["telemetry"]), base, mapping)
        with telemetry_path.open() as stream:
            header = json.loads(next(stream))
        require(header["study_id"] == identifier and header["source_hashes"] == hashes
                and header["plan"] == plan and header["config"]["arm_configs"] == study["arm_configs"],
                "telemetry header differs from authoritative study")
        telemetry_sha = digest(telemetry_path)
        if "telemetry_sha256" in descriptor:
            require(telemetry_sha == descriptor["telemetry_sha256"], "telemetry hash mismatch")
        require(digest(path) == study_sha, "study changed during analysis")
        if run_config_provenance:
            require(digest(run_config_path) == run_config_sha, "run configuration changed during analysis")
        provenance.append(dict(path=str(path), sha256=study_sha, study_id=identifier,
            plan_path=str(plan_path), plan_sha256=descriptor["plan_sha256"],
            telemetry_path=str(telemetry_path), telemetry_sha256=telemetry_sha,
            outcomes=expected_n, run_config=run_config_provenance))
    expected_keys = {(c, n, s) for c in conditions for n in names for s in seeds}
    require(set(outcomes) == expected_keys, "global allocation incomplete or contains extra episodes")
    for source, sha in verified.items():
        require(digest(_path(source, base, mapping)) == sha, "source changed during analysis: " + source)
    if registry_info:
        require(digest(registry_path) == registry_info["sha256"], "policy registry changed during analysis")
    for record in provenance:
        config = record["run_config"]
        if config:
            require(digest(config["path"]) == config["sha256"], "run configuration changed during analysis")
    return dict(manifest=manifest, manifest_path=str(manifest_path), manifest_sha256=digest(manifest_path),
        bank_sha256=bank_info["sha256"], candidates=candidates, recipes=recipes, outcomes=outcomes, energies=energies,
        provenance=provenance, verified_sources=verified, selection=selection_content)


def candidate_table(data, name):
    manifest, outcomes, energies = data["manifest"], data["outcomes"], data["energies"]
    seeds, conditions = manifest["expected_seeds"], manifest["conditions"]
    n = len(seeds)
    successes = {c: sum(outcomes[c, name, s] for s in seeds) for c in conditions}
    counts = {}
    for c in conditions:
        fixed = sum(outcomes[c, name, s] and not outcomes[c, "off", s] for s in seeds)
        broken = sum(outcomes[c, "off", s] and not outcomes[c, name, s] for s in seeds)
        counts[c] = dict(successes=successes[c], n=n, off_successes=sum(outcomes[c, "off", s] for s in seeds),
            observed_fixed=int(fixed), observed_broken=int(broken))
    faults = [c for c in conditions if c != "healthy"]
    score = Fraction(successes["healthy"], 2*n) + Fraction(sum(successes[c] for c in faults), 2*n*len(faults))
    energy = sum(energies[c, name, s][0] for c in conditions for s in seeds)
    steps = sum(energies[c, name, s][1] for c in conditions for s in seeds)
    return dict(candidate=name, runtime_family=data["candidates"][name]["family"],
        balanced_score=float(score), score_fraction=[score.numerator, score.denominator],
        unweighted_macro_success=sum(successes.values()) / (n*len(conditions)),
        healthy_regressions=counts["healthy"]["observed_broken"],
        per_step_applied_correction_energy=float(np.mean([energies[c, name, s][0]/energies[c, name, s][1]
            for c in conditions for s in seeds])), total_applied_correction_energy=energy,
        total_steps=steps, conditions=counts)


def selection_report(data, keep):
    require(data["manifest"]["phase"] in ("search", "validation"), "selection requires a tuning phase")
    require(keep in (1, 2), "keep must be one or two")
    expected_keep = 2 if data["manifest"]["phase"] == "search" else 1
    require(keep == expected_keep, "search retains two; validation retains one per family")
    ranked, selected = {}, {}
    for family, names in data["manifest"]["families"].items():
        rows = [candidate_table(data, name) for name in names]
        rows.sort(key=lambda r: (-Fraction(*r["score_fraction"]), r["healthy_regressions"],
                               r["per_step_applied_correction_energy"], r["candidate"]))
        require(len(rows) >= keep, "too few family candidates to apply selection rule")
        for rank, row in enumerate(rows, 1):
            row.update(rank=rank, eta_zero_alias=family == "composite" and row["runtime_family"] == "kalman")
        ranked[family] = rows
        selected[family] = [r["candidate"] for r in rows[:keep]]
    return dict(kind="candidate_selection", phase=data["manifest"]["phase"], keep=keep,
        selected=selected, ranked=ranked, candidate_bank_sha256=data["bank_sha256"],
        allocation={key: data["manifest"][key] for key in
            ("conditions", "expected_seeds", "expected_candidates", "families")},
        selection_rule="descending .5 healthy success + .5 mean fault success; then fewer observed healthy "
            "regressions versus paired off; then lower mean per-step applied correction energy; then lexical ID",
        interpretation="Ranking selects parameters from development data; it is not held-out superiority evidence.",
        deterministic_equivalences=deterministic_equivalences(data),
        composite_selected_prediction_only_aliases=[name for name in selected["composite"]
            if data["candidates"][name]["family"] == "kalman"])


def deterministic_equivalences(data):
    """Flag known cross-family identities without pooling independent rollouts."""
    pairs = []
    for kalman in data["manifest"]["families"]["kalman"]:
        recipe = data.get("recipes", {}).get(kalman, {})
        parameter = data["candidates"][kalman]
        if not (recipe.get("q_recipe") == "matched" and recipe.get("q_scale") == 1
                and recipe.get("r_scale") == 1 and recipe.get("p0") == "gamma_c"
                and parameter.get("damping", 0) == 0 and parameter.get("tracking_rate", 0) == 0):
            continue
        for dob in data["manifest"]["families"]["dob"]:
            comparator = data["candidates"][dob]
            if comparator["gamma"] == parameter["gamma"] and comparator["clip"] == parameter["clip"]:
                pairs.append(dict(kalman=kalman, dob=dob,
                    identity="P0=gamma*C and Q=gamma^2*C/(1-gamma) give K*M=gamma*I from the first update; "
                        "same gamma, clipping, zero estimate initialization and shared M yield the same mean recursion.",
                    policy_rollouts="Separate, unpinned policy draws are retained as preregistered.",
                    interpretation="Outcome differences between these equivalent controllers do not establish an intrinsic "
                        "Kalman-over-DOB method advantage; numerical implementation rounding may also differ."))
    return pairs


def sign_flip_test(differences, draws=MIN_DRAWS, random_seed=20260908):
    """Two-sided Monte Carlo whole-cluster sign flips, including the observed draw."""
    require(isinstance(draws, int) and draws >= MIN_DRAWS, "at least 200000 sign-flip draws required")
    d = np.asarray(differences, float)
    require(d.ndim == 1 and len(d) > 0 and np.isfinite(d).all(), "invalid cluster differences")
    rng, extreme = np.random.default_rng(random_seed), 0
    observed = abs(float(d.sum()))
    for start in range(0, draws, 4096):
        size = min(4096, draws-start)
        signs = rng.integers(0, 2, size=(size, len(d)), dtype=np.int8)*2-1
        values = np.abs(np.sum(signs*d, axis=1))
        extreme += int(np.count_nonzero(values >= observed - 1e-12))
    return dict(effect=float(d.mean()), p_two_sided=(extreme+1)/(draws+1),
        draws=draws, random_seed=random_seed, extreme_draws=extreme, seed_clusters=len(d),
        cluster_differences=d.tolist(), method="whole-seed sign flips across all conditions; plus-one correction")


def holm(values):
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    result, previous = [0.0]*len(values), 0.0
    for rank, i in enumerate(ordered):
        previous = max(previous, min(1.0, values[i]*(len(values)-rank)))
        result[i] = previous
    return result


def cluster_test(differences, denominator, draws=MIN_DRAWS, random_seed=20260908):
    """Exact integer sign-flip DP for balanced binary outcomes; MC fallback.

    For eight conditions d_s=(7*healthy_difference+sum(fault_differences))/14.
    The 2**30 assignments have at most 841 distinct signed integer sums, so
    integer-count dynamic programming gives an exact tail without enumeration.
    """
    d = np.asarray(differences, float)
    require(d.ndim == 1 and len(d) > 0 and np.isfinite(d).all(), "invalid cluster differences")
    require(isinstance(denominator, int) and denominator > 0, "invalid score denominator")
    scaled = d*denominator
    integer = np.rint(scaled).astype(np.int64)
    if (not np.allclose(scaled, integer, rtol=0, atol=1e-10)
            or sum(abs(int(x)) for x in integer) > 10000):
        return sign_flip_test(d, draws, random_seed)
    distribution = {0: 1}
    for raw in integer:
        value = int(raw)
        updated = {}
        for total, count in distribution.items():
            updated[total+value] = updated.get(total+value, 0) + count
            updated[total-value] = updated.get(total-value, 0) + count
        distribution = updated
    observed = abs(sum(int(x) for x in integer))
    extreme = sum(count for total,count in distribution.items() if abs(total) >= observed)
    assignments = 2**len(integer)
    require(sum(distribution.values()) == assignments, "integer sign-flip probability mass was lost")
    return dict(effect=float(sum(int(x) for x in integer)/(denominator*len(integer))),
        p_two_sided=extreme/assignments, exact=True, assignments=assignments,
        extreme_assignments=extreme, distinct_signed_sums=len(distribution),
        seed_clusters=len(integer), cluster_differences=d.tolist(),
        integer_cluster_differences=integer.tolist(), score_denominator=denominator,
        method="exact whole-seed sign flips using integer-count dynamic programming")


def cluster_bootstrap_interval(differences, random_seed=20260908, resamples=50000):
    """Percentile interval resampling paired whole-seed differences."""
    d = np.asarray(differences, float)
    require(d.ndim == 1 and len(d) > 0 and np.isfinite(d).all(), "invalid bootstrap cluster differences")
    require(resamples == 50000, "the declared bootstrap uses exactly 50000 resamples")
    rng = np.random.default_rng(random_seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 4096):
        size = min(4096, resamples-start)
        indices = rng.integers(0, len(d), size=(size, len(d)))
        means[start:start+size] = d[indices].mean(axis=1)
    lower, upper = np.quantile(means, [.025, .975])
    return dict(level=.95, lower=float(lower), upper=float(upper), resamples=resamples,
        random_seed=random_seed, seed_clusters=len(d),
        method="paired whole-seed percentile bootstrap of the balanced score difference",
        limitation="Approximate interval conditional on the observed seed distribution; a narrow or degenerate "
            "interval does not establish equivalence, and boundary/all-identical differences can understate uncertainty.")


def _mcnemar(broken, fixed):
    n = broken+fixed
    return min(1.0, 2*sum(math.comb(n, i) for i in range(min(broken, fixed)+1))/2**n) if n else 1.0


def confirmation_report(data, draws=MIN_DRAWS, random_seed=20260908):
    manifest = data["manifest"]
    require(manifest["phase"] == "confirmation", "held-out scoring requires confirmation phase")
    require(len(manifest["conditions"]) == 8, "confirmation requires all eight conditions")
    names = {family: manifest["families"][family][0] for family in FAMILIES}
    seeds, conditions, outcomes = manifest["expected_seeds"], manifest["conditions"], data["outcomes"]
    faults = [c for c in conditions if c != "healthy"]

    def compare(left, right, label, offset):
        differences = [.5*(outcomes["healthy", right, s]-outcomes["healthy", left, s]) +
            .5*sum(outcomes[c, right, s]-outcomes[c, left, s] for c in faults)/len(faults) for s in seeds]
        result = cluster_test(differences, 2*len(faults), draws, random_seed+offset)
        result.update(id=label, left=left, right=right,
            effect_direction="positive favors right", significant=result["p_two_sided"] < .05)
        result["confidence_interval"] = cluster_bootstrap_interval(differences, random_seed+offset)
        cells = []
        for condition in conditions:
            broken = sum(outcomes[condition, left, s] and not outcomes[condition, right, s] for s in seeds)
            fixed = sum(outcomes[condition, right, s] and not outcomes[condition, left, s] for s in seeds)
            cells.append(dict(condition=condition, left_successes=sum(outcomes[condition,left,s] for s in seeds),
                right_successes=sum(outcomes[condition,right,s] for s in seeds), n=len(seeds),
                observed_broken=int(broken), observed_fixed=int(fixed), p_exact=_mcnemar(broken, fixed)))
        adjusted = [min(1.0, 8*c["p_exact"]) for c in cells]
        for row, p in zip(cells, adjusted):
            row.update(p_bonferroni8=p, significant=p < .05)
        result["per_cell"] = cells
        return result

    primary = compare(names["kalman"], names["composite"], "composite_vs_kalman", 0)
    secondary = [compare(names[f], names["kalman"], "kalman_vs_"+f, i+1) for i,f in enumerate(SECONDARY)]
    for row, p in zip(secondary, holm([r["p_two_sided"] for r in secondary])):
        row.update(p_holm4=p, significant=p < .05)
    equivalents = deterministic_equivalences(data)
    for row in secondary:
        row["deterministically_equivalent_mean_update"] = any(
            pair["kalman"] == row["right"] and pair["dob"] == row["left"] for pair in equivalents)
        row["intrinsic_method_superiority_supported"] = (row["significant"] and row["effect"] > 0
            and not row["deterministically_equivalent_mean_update"])
    table = {name: candidate_table(data, name) for name in manifest["expected_candidates"]}
    numerical = sorted([dict(family=f, candidate=names[f], balanced_score=table[names[f]]["balanced_score"])
        for f in FAMILIES], key=lambda r: (-r["balanced_score"], r["family"]))
    return dict(kind="confirmation_analysis", phase="confirmation", selected=names, table=table,
        numerical_order=numerical, numerical_order_interpretation="Descriptive score order only; statistical "
            "ordering is limited to the explicitly tested contrasts. Unresolved differences are not ties/equivalence.",
        primary=primary, secondary=secondary,
        deterministic_equivalences=equivalents,
        statistical_families=dict(primary="one two-sided seed-cluster test, composite versus Kalman",
            secondary="four two-sided seed-cluster tests, Kalman versus legacy/DOB/RLS/calibrated integral; Holm4",
            per_cell="eight exact McNemar tests per predeclared method contrast, Bonferroni8 within that contrast; "
                "five distinct descriptive families, not one global 40-test guarantee"),
        composite_prediction_only=data["candidates"][names["composite"]]["family"] == "kalman",
        oracle_interpretation="Privileged diagnostic; excluded from estimated-method rankings.",
        limitations=["Inference is conditional on this fixed ALOHA task, fault panel and tuned candidate bank.",
            "Whole-seed swaps retain dependence across fault conditions sharing a reset.",
            "Exact sign-flip tails require the whole-seed exchangeability/symmetry null; arbitrary zero-mean "
                "asymmetric cluster differences do not give the same exact guarantee.",
            "Independent policy sampling contributes to observed fixed/broken outcomes.",
            "A selected zero-tracking alias provides no evidence of positive tracking advantage."])


def markdown(report):
    if report["kind"] == "candidate_selection":
        lines = ["# Frozen candidate selection", "", report["selection_rule"], "",
                 "| Family | Selected candidates |", "|---|---|"]
        lines += [f"| {family} | {', '.join(names)} |" for family,names in report["selected"].items()]
        lines += ["", report["interpretation"]]
    else:
        lines = ["# Complete held-out ALOHA tuning comparison", "",
                 "| Family | Candidate | Balanced score |", "|---|---|---:|"]
        lines += [f"| {r['family']} | {r['candidate']} | {r['balanced_score']:.4f} |" for r in report["numerical_order"]]
        lines += ["", report["numerical_order_interpretation"], "", "| Contrast | Right-minus-left score | Bootstrap 95% CI | p | Adjusted p |",
                  "|---|---:|---:|---:|---:|"]
        for row in [report["primary"]]+report["secondary"]:
            interval = row["confidence_interval"]
            lines.append(f"| {row['id']} | {row['effect']:+.4f} | [{interval['lower']:+.4f}, {interval['upper']:+.4f}] | {row['p_two_sided']:.6g} | {row.get('p_holm4', row['p_two_sided']):.6g} |")
        lines += ["", "Intervals use 50,000 whole-seed percentile bootstrap resamples; they do not establish equivalence. "
                  "P-values use exact whole-seed sign-flip dynamic programming when the balanced score has integer weights."]
        if report["composite_prediction_only"]:
            lines += ["", "The selected composite is a zero-tracking Kalman alias; this does not demonstrate a tracking benefit."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-map", type=Path)
    parser.add_argument("--draws", type=int, default=MIN_DRAWS)
    parser.add_argument("--random-seed", type=int, default=20260908)
    args = parser.parse_args(argv)
    require(not args.out_dir.exists(), "output directory exists; preserve the original analysis")
    data = load_allocation(args.manifest, None if args.source_map is None else read(args.source_map))
    phase = data["manifest"]["phase"]
    report = confirmation_report(data, args.draws, args.random_seed) if phase == "confirmation" else selection_report(data, 2 if phase == "search" else 1)
    report.update(schema_version=1, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        analysis_source_sha256=digest(Path(__file__)), manifest_path=data["manifest_path"],
        manifest_sha256=data["manifest_sha256"], candidate_bank_sha256=data["bank_sha256"],
        studies=data["provenance"], verified_runtime_sources=data["verified_sources"],
        validated_outcomes=len(data["outcomes"]))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    filename = "selection.json" if phase != "confirmation" else "analysis.json"
    (args.out_dir/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    (args.out_dir/"summary.md").write_text(markdown(report))
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
