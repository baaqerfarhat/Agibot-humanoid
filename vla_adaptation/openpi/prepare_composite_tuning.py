"""Build and reconstruct a source-bound bank of observer tuning candidates.

Candidate selection belongs to a separate development experiment. This module
uses healthy calibration only, retains rejected recipes, and never ranks task
outcomes. Q is covariance per STEP; R is per observation. The runtime composite
core receives Q/dt. Tracking strength is normalized on applied output channels.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np

import composite_observer as composite
from prepare_composite_reference import json_ready
from validate_composite_reference import _same, validate_composite_artifact


FAMILIES = ("off", "legacy", "dob", "rls", "integral_calibrated", "kalman", "composite", "oracle")
DEFAULTS = dict(gamma=.08, q_recipe="matched", q_scale=1., r_scale=1.,
                p0="diffuse_identity", active_strength=0., damping=0., clip=.08,
                dead=.002, norm_r=.05, norm_channels="all", ki=.02,
                rls_lambda=.95, rls_p0=1000.)
RECIPE_KEYS = set(DEFAULTS) | {"name", "family"}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(value, name, *, zero=False):
    value = float(value)
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError("%s must be finite and %s" % (name, "nonnegative" if zero else "positive"))
    return value


def _recipe(spec):
    if not isinstance(spec, dict):
        raise ValueError("candidate spec must be an object")
    if set(spec) - RECIPE_KEYS:
        raise ValueError("unknown candidate recipe keys: %s" % sorted(set(spec) - RECIPE_KEYS))
    name, family = spec.get("name"), spec.get("family")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("candidate name must use letters, numbers, underscores or hyphens")
    if family not in FAMILIES:
        raise ValueError("unknown candidate family: %s" % family)
    recipe = dict(DEFAULTS, **spec)
    for key in ("gamma", "q_scale", "r_scale", "clip", "norm_r", "ki", "rls_p0"):
        recipe[key] = _number(recipe[key], key)
    for key in ("active_strength", "damping", "dead"):
        recipe[key] = _number(recipe[key], key, zero=True)
    recipe["rls_lambda"] = _number(recipe["rls_lambda"], "rls_lambda")
    if not recipe["gamma"] < 1 or recipe["rls_lambda"] > 1:
        raise ValueError("gamma must be below one and rls_lambda at most one")
    if recipe["q_recipe"] not in ("matched", "arm_isotropic"):
        raise ValueError("q_recipe must be matched or arm_isotropic")
    if recipe["norm_channels"] not in ("all", "corrected"):
        raise ValueError("norm_channels must be all or corrected")
    if family != "composite" and recipe["active_strength"] != 0:
        raise ValueError("positive tracking strength requires composite family")
    if family not in ("kalman", "composite") and recipe["damping"] != 0:
        raise ValueError("damping is implemented only for kalman/composite")
    return recipe


def _p0(value, C, gamma, steady):
    n = len(C)
    if isinstance(value, str):
        choices = {"diffuse_identity": np.eye(n), "calibrated_c": C,
                   "gamma_c": gamma*C, "steady": steady}
        if value not in choices:
            raise ValueError("p0 must be diffuse_identity, calibrated_c, gamma_c, steady, a scalar or an SPD matrix")
        return choices[value].copy()
    if np.ndim(value) == 0:
        return np.eye(n) * _number(value, "p0")
    return composite._symmetric(value, "p0", n, positive_definite=True)


def _covariance_path(H, Q, R, dt, damping, initial, count):
    """No observation values enter covariance propagation or screening."""
    n = H.shape[1]
    P = initial.copy()
    decay = np.exp(-damping*dt)
    interval = dt if damping == 0 else -np.expm1(-2*damping*dt)/(2*damping)
    path = []
    for _ in range(count):
        prior = decay**2 * P + interval * Q/dt
        K = np.linalg.solve(H @ prior @ H.T + R, H @ prior).T
        remaining = np.eye(n)-K@H
        updated = remaining@prior@remaining.T + K@R@K.T
        updated = .5*(updated+updated.T)
        path.append((updated, K))
        P = updated
    return path


def _steady(H, Q, R, dt, damping, budget, matched_scale=None):
    # An exact scalar-mode DARE avoids falsely rejecting low process-noise
    # modes merely because diffuse-P0 iteration has not converged in 1000
    # steps. H is square/invertible and parameter decay is scalar here.
    inverse = np.linalg.inv(H)
    C = inverse @ R @ inverse.T
    C = .5*(C+C.T)
    decay = np.exp(-damping*dt)
    interval = dt if damping == 0 else -np.expm1(-2*damping*dt)/(2*damping)
    if matched_scale is not None:
        # Q_step is explicitly recipe-proportional to C. Avoid an ill-
        # conditioned eigen rotation followed by R^-1 merely to recover the
        # scalar gain: it can create platform-dependent ~1e-10 off-diagonals
        # that should be zero. This branch preserves the declared equations.
        q = _number(matched_scale,"matched process/measurement scale")*interval/dt
        b = q+1-decay**2
        p = 2*q/(b+np.sqrt(b*b+4*decay**2*q))
        P,K = p*C,p*inverse
        next_P,_ = _covariance_path(H,Q,R,dt,damping,P,1)[0]
        change = np.linalg.norm(next_P-P,2)/max(np.linalg.norm(P,2),1e-30)
        return P,K,float(change),bool(change < 1e-9)
    values, vectors = np.linalg.eigh(C)
    if np.any(values <= 0):
        raise ValueError("parameter measurement covariance must be positive definite")
    sqrt_C = (vectors*np.sqrt(values))@vectors.T
    inverse_sqrt_C = (vectors/np.sqrt(values))@vectors.T
    process = inverse_sqrt_C @ (interval*Q/dt) @ inverse_sqrt_C
    process = .5*(process+process.T)
    q, modes = np.linalg.eigh(process)
    if np.any(q <= 0):
        raise ValueError("whitened process covariance must be positive definite")
    b = q + 1-decay**2
    p = 2*q/(b+np.sqrt(b*b+4*decay**2*q))
    P = sqrt_C @ ((modes*p)@modes.T) @ sqrt_C
    P = .5*(P+P.T)
    K = np.linalg.solve(R,H@P).T
    next_P,_ = _covariance_path(H,Q,R,dt,damping,P,1)[0]
    change = np.linalg.norm(next_P-P,2)/max(np.linalg.norm(P,2),1e-30)
    return P,K,float(change),bool(change < 1e-9)


def _make_candidates(reference, specs, settings):
    observer = reference["observer"]
    model = reference["model"]
    H, Rbase = np.asarray(observer["M"], float), np.asarray(observer["R"], float)
    n = H.shape[1]
    inverse = np.linalg.inv(H)
    C = inverse @ Rbase @ inverse.T
    C = .5*(C+C.T)
    mask = np.zeros(n)
    indices = reference["settings"]["correction_indices"]
    mask[indices] = 1
    D = composite.masked_tracking_map(model, mask)
    A, L = np.asarray(model["A"]), np.asarray(model["metric"])
    dt = reference["settings"]["dt"]
    # Templates have parameter covariance units. R scaling is independent,
    # so q_scale/r_scale can change bandwidth rather than canceling by design.
    templates = {"matched": C}
    if n == 14:
        diagonal = np.diag(C).copy()
        arms = list(range(6)) + list(range(7, 13))
        diagonal[arms] = np.median(np.diag(C)[:6])
        templates["arm_isotropic"] = np.diag(diagonal)
    records, steady_cache, path_cache = [], {}, {}
    names = set()
    for spec in specs:
        recipe = _recipe(spec)
        name = recipe["name"]
        if name in names:
            raise ValueError("duplicate candidate name: " + name)
        names.add(name)
        if recipe["q_recipe"] not in templates:
            raise ValueError("arm_isotropic requires ALOHA's 14 parameter coordinates")
        gamma = recipe["gamma"]
        Q = gamma**2/(1-gamma) * recipe["q_scale"] * templates[recipe["q_recipe"]]
        R = recipe["r_scale"] * Rbase
        covariance_key = (recipe["q_recipe"], gamma, recipe["q_scale"], recipe["r_scale"], recipe["damping"])
        if covariance_key not in steady_cache:
            matched_scale = (gamma**2/(1-gamma)*recipe["q_scale"]/recipe["r_scale"]
                             if recipe["q_recipe"] == "matched" else None)
            steady_cache[covariance_key] = _steady(H,Q,R,dt,recipe["damping"],
                settings["covariance_steps"],matched_scale=matched_scale)
        Psteady, Ksteady, change, converged = steady_cache[covariance_key]
        P0 = _p0(recipe["p0"], C, gamma, Psteady)
        scale = float(np.linalg.norm((dt*Psteady@D.T@L@D)[np.ix_(indices,indices)],2))
        reasons = []
        rate = 0.0
        if recipe["active_strength"] > 0:
            if scale <= 0:
                reasons.append("zero tracking response on applied correction coordinates")
            else:
                rate = recipe["active_strength"]/scale
        parameters = {key: recipe[key] for key in ("family", "gamma", "clip", "dead", "norm_r",
            "norm_channels", "damping", "ki", "rls_lambda", "rls_p0")}
        parameters.update(Q=Q, R=R, P0=P0, tracking_rate=rate)
        screening = dict(scope="Fitted local unsaturated frozen-gain recursion; not nonlinear/VLA stability",
                         screened_steps=settings["screened_steps"], covariance_converged=converged,
                         steady_solution=("exact proportional-noise scalar DARE with one-step residual check"
                            if recipe["q_recipe"] == "matched" else
                            "exact whitened scalar-mode DARE with one-step residual check"),
                         steady_covariance=Psteady, steady_gain=Ksteady,
                         steady_effective_gain=Ksteady@H, active_scale=scale,
                         scale_definition="spectral norm of (dt*P_inf*D.T*L*D)[corrected,corrected]",
                         active_strength=recipe["active_strength"])
        if recipe["family"] in ("kalman", "composite"):
            if not converged:
                reasons.append("Riccati covariance did not converge within declared budget")
            key = covariance_key + (P0.tobytes(),)
            if key not in path_cache:
                path_cache[key] = _covariance_path(H,Q,R,dt,recipe["damping"],P0,settings["screened_steps"])
            path = path_cache[key] + [(Psteady,Ksteady)]
            radii = [composite.augmented_error_report(A,D,H,P,K,dt=dt,
                tracking_rate=rate,damping=recipe["damping"],metric=L)["spectral_radius"] for P,K in path]
            maximum = float(max(radii))
            screening.update(maximum_frozen_augmented_radius=maximum,
                             worst_covariance_step=int(np.argmax(radii))+1,
                             steady_state_included=True)
            if maximum >= settings["maximum_frozen_radius"]:
                reasons.append("non-Schur or insufficient-margin augmented dynamics during covariance transient/steady state")
        else:
            screening.update(maximum_frozen_augmented_radius=None,
                             worst_covariance_step=None, steady_state_included=False)
        records.append(dict(name=name, recipe=recipe, parameters=parameters,
            qualification=dict(allowed=not reasons, reasons=reasons), screening=screening))
    return records


def _settings(*, calibration_seeds=(), screened_steps=300, covariance_steps=1000):
    for name, value in (("screened_steps", screened_steps), ("covariance_steps", covariance_steps)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 2:
            raise ValueError(name + " must be an integer at least two")
    if covariance_steps < screened_steps:
        raise ValueError("covariance_steps must cover screened_steps")
    seeds = list(calibration_seeds)
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("calibration_seeds must be distinct nonnegative integers")
    return dict(calibration_seeds=seeds, screened_steps=screened_steps,
                covariance_steps=covariance_steps, maximum_frozen_radius=.999999)


def _assemble(reference, descriptor, specs, settings, checked):
    if not isinstance(specs, list) or not specs:
        raise ValueError("candidate specs must be a nonempty list")
    records = _make_candidates(reference, specs, settings)
    accepted = sum(row["qualification"]["allowed"] for row in records)
    return dict(schema_version=1, reference=dict(descriptor, artifact=reference),
        settings=settings, candidate_specs=specs, candidates=records,
        observer={key: reference["observer"][key] for key in ("W", "M")},
        reference_model=reference["model"], correction_indices=reference["settings"]["correction_indices"],
        dt=reference["settings"]["dt"], qualification=dict(reference_qualified=checked["validated"],
            allowed=bool(accepted), accepted_candidates=accepted,rejected_candidates=len(records)-accepted),
        interpretation=["All candidates use the same healthy calibration and qualified tracking metric.",
            "Q is per control step; R per sample; P0 is full parameter covariance.",
            "arm_isotropic uses equal process variance on twelve radian arm coordinates; grippers retain their own units.",
            "gamma is the nominal matched bandwidth only when q_scale=r_scale=1 and damping=0.",
            "calibration_seeds are declared provenance and must be kept disjoint from tuning/confirmation.",
            "A zero-tracking zero-damping composite with identical Q/R/P0/clip nests Kalman exactly.",
            "Retained rejected candidates must never be selected by a runner."])


def build_candidate_bank(reference_path, candidate_specs, *, source_log=None,
                         source_sensitivity=None, calibration_seeds=(),
                         screened_steps=300, covariance_steps=1000):
    reference_path = Path(reference_path).resolve()
    raw = reference_path.read_bytes()
    reference = json.loads(raw)
    checked = validate_composite_artifact(reference,artifact_path=reference_path,
        source_log=source_log,source_sensitivity=source_sensitivity)
    settings = _settings(calibration_seeds=calibration_seeds,screened_steps=screened_steps,
                         covariance_steps=covariance_steps)
    result = _assemble(reference, dict(path=str(reference_path),sha256=hashlib.sha256(raw).hexdigest()),
                       candidate_specs,settings,checked)
    result["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return json_ready(result)


def _reference_source(bank, bank_path):
    descriptor = bank["reference"]
    declared = Path(descriptor["path"])
    paths = [declared]
    if bank_path is not None:
        root = Path(bank_path).resolve().parent
        paths += [root/declared, root/declared.name]
    for path in dict.fromkeys(p.resolve() for p in paths):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() == descriptor["sha256"]:
            return json.loads(raw), path
    raise ValueError("candidate bank cannot find hash-matched base reference artifact")


def validate_candidate_bank(bank, *, bank_path=None, source_log=None, source_sensitivity=None):
    """Reconstruct once before launch; return only eligible runtime candidates.

    The caller should cache this result for a frozen bank, and rehash all returned
    source files at rollout boundaries. No cache is hidden in this function.
    """
    try:
        if bank.get("schema_version") != 1:
            raise ValueError("unsupported candidate bank schema")
        reference, reference_path = _reference_source(bank,bank_path)
        _same(bank["reference"]["artifact"],reference,"candidate_bank.reference.artifact")
        checked = validate_composite_artifact(reference,artifact_path=reference_path,
            source_log=source_log,source_sensitivity=source_sensitivity)
        raw_settings = bank["settings"]
        settings = _settings(**{k:raw_settings[k] for k in
            ("calibration_seeds","screened_steps","covariance_steps")})
        _same(raw_settings,settings,"candidate_bank.settings")
        expected = _assemble(reference,{k:bank["reference"][k] for k in ("path","sha256")},
                             bank["candidate_specs"],settings,checked)
        for key in ("observer","reference_model","correction_indices","dt","qualification"):
            _same(bank[key],expected[key],"candidate_bank."+key)
        if len(bank["candidates"]) != len(expected["candidates"]):
            raise ValueError("candidate bank candidate count mismatch")
        runtime = {}
        for old,new in zip(bank["candidates"],expected["candidates"]):
            for key in ("name","recipe","parameters","qualification"):
                _same(old[key],new[key],"candidate_bank.%s.%s" % (new["name"],key))
            # Worst step can differ at numerical ties; compare its physical
            # eligibility and maximum radius, not an arbitrary tie-break index.
            _same(old["screening"],{k:v for k,v in new["screening"].items()
                if k != "worst_covariance_step"},"candidate_bank.%s.screening" % new["name"])
            if new["qualification"]["allowed"]:
                # Reconstruction checks eligibility and consistency; execution
                # uses the exact declared numerical values, not LAPACK-specific
                # last-bit differences from the reconstruction platform.
                parameters = dict(old["parameters"])
                for key in ("Q", "R", "P0"):
                    parameters[key] = np.asarray(old["parameters"][key],float).copy()
                runtime[new["name"]] = parameters
        if not runtime:
            raise ValueError("candidate bank has no qualified candidates")
        sources = {str(reference_path):bank["reference"]["sha256"]}
        sources.update({value["path"]:value["sha256"] for value in checked["sources"].values()})
        if bank_path is not None:
            sources[str(Path(bank_path).resolve())] = _digest(bank_path)
        return dict(observer=bank["observer"],reference_model=bank["reference_model"],
            candidates=runtime,qualification=bank["qualification"],sources=sources,
            correction_indices=bank["correction_indices"],dt=bank["dt"],
            calibration_seeds=settings["calibration_seeds"],screened_steps=settings["screened_steps"])
    except (KeyError,TypeError,np.linalg.LinAlgError) as error:
        raise ValueError("invalid candidate bank: %s" % error) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--candidates",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--source-log",type=Path)
    parser.add_argument("--source-sensitivity",type=Path)
    parser.add_argument("--calibration-seeds",default="")
    parser.add_argument("--screened-steps",type=int,default=300)
    parser.add_argument("--covariance-steps",type=int,default=1000)
    args = parser.parse_args(argv)
    inputs = [args.reference,args.candidates,args.source_log,args.source_sensitivity]
    if args.out.resolve() in [p.resolve() for p in inputs if p is not None]:
        parser.error("--out must differ from every input")
    specs = json.loads(args.candidates.read_text())
    if isinstance(specs,dict):
        specs = specs["candidates"]
    result = build_candidate_bank(args.reference,specs,source_log=args.source_log,
        source_sensitivity=args.source_sensitivity,
        calibration_seeds=[int(v) for v in args.calibration_seeds.split(",") if v.strip()],
        screened_steps=args.screened_steps,covariance_steps=args.covariance_steps)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.out),**result["qualification"])))
    return 0 if result["qualification"]["allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
