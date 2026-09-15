#!/usr/bin/env python3
"""E2 core scorer, v2 (PREREG_E2_CORE.md; corrections from iclr2027/EXPERIMENTS_ASAP.md, 2026-09-15).

Pairs the seven arms of a suite on the key (task, init, sampler_seed) and computes the registered
primary metrics with task-clustered bootstrap intervals (whole tasks resampled with replacement,
every key of a task kept). Completeness is enforced: every registered arm must be present with
exactly the manifest's keys; duplicates, missing and extra keys are listed and the confirmation
denominator is never a silent intersection (--allow-partial reports partial arms as such).

Metric 1 (H1): per adaptive episode, over its last 50 VALID steps (valid = not done), the signed
rotation-y error of the observation z = M^-1 r and of the estimate against the injected fault
(+0.05 faulted, 0 healthy). Telemetry episodes are joined to the sampler seed through the ordered
manifest (episode ordinal -> scenario) and checked against episodes.csv. Episodes shorter than
50 valid steps are reported; an empty valid window yields no bias value and is listed. The
relative reduction C vs U is bootstrapped as a ratio: numerator AND denominator recomputed in
every task-resampled replicate.
Metric 2 (H2): success-rate contrasts with task-clustered intervals, both/fixed/broken/neither,
exact McNemar (descriptive), the -5 point healthy target, oracle gaps.
"""
import argparse, collections, csv, gzip, json, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
from mcnemar import mcnemar_exact
ARMS = ("healthy_off", "healthy_U", "healthy_C", "faulted_off", "faulted_U", "faulted_C", "faulted_oracle")


def read_outcomes(d):
    rows = list(csv.DictReader(open(d / "episodes.csv"))); out, dup = {}, []
    for r in rows:
        k = (r["task"], r["init"], r["policy_seed"])
        if k in out:
            dup.append(k)
        out[k] = int(r["outcome"])
    return out, dup


def open_log(d):
    p = d / "telemetry.jsonl"; g = d / "telemetry.jsonl.gz"
    if p.exists():
        return open(p)
    if g.exists():
        return gzip.open(g, "rt")
    return None


def observation_bias(d, M_inv, fault_ry, manifest, last=50):
    fh = open_log(d)
    if fh is None:
        return {}
    per = {}
    for line in fh:
        s = json.loads(line)
        if s.get("type") != "step" or s.get("phase") != "rollout" or s.get("arm") != "adaptive" or s.get("r") is None:
            continue
        per.setdefault(int(s["episode"]), []).append(s)
    out = {}
    for ordinal, steps in per.items():
        sc = manifest["scenarios"][ordinal]                     # ordinal -> declared scenario (task, init, sampler seed)
        assert str(sc["task"]) == str(steps[0]["task"]) and str(sc["init"]) == str(steps[0]["init"]), f"manifest/telemetry mismatch at episode {ordinal}"
        key = (str(sc["task"]), str(sc["init"]), f"schedule:{sc['sampler_seed']}")
        valid = [s for s in steps if not s.get("done")]; seg = valid[-last:]
        rec = dict(n_steps=len(steps), n_valid=len(valid), short=len(valid) < last, empty=len(seg) == 0)
        if seg:
            z = np.array([M_inv @ np.asarray(s["r"], float) for s in seg]); fhat = np.array([s["f_hat"] for s in seg]); corr = np.array([s["correction"] for s in seg])
            rec.update(obs_err_ry=float(z[:, 4].mean() - fault_ry), est_err_ry=float(fhat[:, 4].mean() - fault_ry),
                       remaining_ry=float(fault_ry + corr[:, 4].mean()), corr_norm=float(np.linalg.norm(corr[:, 3:6], axis=1).mean()),
                       clipped_fraction=float(np.mean(np.abs(np.abs(fhat[:, 3:6]) - 0.15) < 1e-6)))
        out[key] = rec
    return out


def cluster_boot(keys, stat, n=10000, seed=0):
    """keys: list of task ids per observation, stat(list_of_selected_indices) -> float; whole-task resampling."""
    rng = np.random.default_rng(seed); tasks = sorted(set(keys)); by = {t: [i for i, k in enumerate(keys) if k == t] for t in tasks}
    boots = []
    for _ in range(n):
        pick = rng.choice(tasks, len(tasks), replace=True); idx = [i for t in pick for i in by[t]]; boots.append(stat(idx))
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root", type=pathlib.Path); ap.add_argument("--suite", required=True)
    ap.add_argument("--manifest", type=pathlib.Path, default=None); ap.add_argument("--openloop", default=str(HERE.parent / "results/phase05/openloop_so3.json"))
    ap.add_argument("--allow-partial", action="store_true"); ap.add_argument("--out", type=pathlib.Path); a = ap.parse_args()
    manifest = json.loads((a.manifest or (HERE.parent / f"results/iclr_unified_v1/manifests/{a.suite}_E2_core.json")).read_text())
    expected = sorted({(str(s["task"]), str(s["init"]), f"schedule:{s['sampler_seed']}") for s in manifest["scenarios"]})
    M_inv = np.linalg.pinv(np.array(json.loads(pathlib.Path(a.openloop).read_text())["M"]))
    receipt = dict(suite=a.suite, expected_keys=len(expected), arms={}); oc = {}
    for arm in ARMS:
        d = a.root / f"{a.suite}_{arm}"
        if not (d / "episodes.csv").exists():
            receipt["arms"][arm] = dict(status="missing"); continue
        o, dup = read_outcomes(d); ks = set(o)
        receipt["arms"][arm] = dict(status="complete" if ks == set(expected) and not dup else "incomplete", n=len(o), duplicates=[list(k) for k in dup],
                                    missing=[list(k) for k in expected if k not in ks], extra=[list(k) for k in ks if k not in set(expected)])
        oc[arm] = o
    complete = [arm for arm in ARMS if receipt["arms"][arm].get("status") == "complete"]
    if len(complete) < len(ARMS) and not a.allow_partial:
        print(json.dumps(dict(receipt=receipt, error="not all registered arms complete; pass --allow-partial for a provisional report"), indent=1)); sys.exit(2)
    keys = expected
    out = dict(receipt=receipt, provisional=len(complete) < len(ARMS), n_keys=len(keys), success={arm: (sum(oc[arm][k] for k in keys), len(keys)) for arm in complete})
    tasks_of = [k[0] for k in keys]
    def contrast(x, y):
        if x not in complete or y not in complete:
            return None
        diff = np.array([oc[x][k] - oc[y][k] for k in keys], float)
        b = sum(oc[x][k] and oc[y][k] for k in keys); xo = sum(oc[x][k] and not oc[y][k] for k in keys); yo = sum(oc[y][k] and not oc[x][k] for k in keys)
        return dict(rate_difference=float(diff.mean()), task_clustered_ci95=cluster_boot(tasks_of, lambda idx: float(np.mean(diff[idx]))),
                    both=b, x_only=xo, y_only=yo, neither=len(keys) - b - xo - yo, mcnemar_exact_p_descriptive=mcnemar_exact(yo, xo))
    out["task"] = {n: contrast(x, y) for n, (x, y) in dict(C_minus_U_faulted=("faulted_C", "faulted_U"), U_minus_off_faulted=("faulted_U", "faulted_off"), C_minus_off_faulted=("faulted_C", "faulted_off"),
                                                       oracle_minus_C=("faulted_oracle", "faulted_C"), oracle_minus_U=("faulted_oracle", "faulted_U"),
                                                       U_minus_off_healthy=("healthy_U", "healthy_off"), C_minus_off_healthy=("healthy_C", "healthy_off"), C_minus_U_healthy=("healthy_C", "healthy_U")).items()}
    for k in ("U_minus_off_healthy", "C_minus_off_healthy"):
        if out["task"][k]:
            out["task"][k]["fails_no_harm_target_minus_5_points"] = bool(out["task"][k]["rate_difference"] < -0.05)
    ob = {}
    for arm, fr in (("faulted_U", 0.05), ("faulted_C", 0.05), ("healthy_U", 0.0), ("healthy_C", 0.0)):
        if arm in complete:
            ob[arm] = observation_bias(a.root / f"{a.suite}_{arm}", M_inv, fr, manifest)
    out["per_episode"] = {arm: {"|".join(k): v for k, v in d.items()} for arm, d in ob.items()}
    if "faulted_U" in ob and "faulted_C" in ob:
        u, c = ob["faulted_U"], ob["faulted_C"]
        common = [k for k in keys if k in u and k in c and not u[k]["empty"] and not c[k]["empty"]]
        au_v = np.array([abs(u[k]["obs_err_ry"]) for k in common]); ac_v = np.array([abs(c[k]["obs_err_ry"]) for k in common]); tk = [k[0] for k in common]
        au, ac = float(au_v.mean()), float(ac_v.mean())
        ratio = lambda idx: float(1 - ac_v[idx].mean() / au_v[idx].mean()) if au_v[idx].mean() > 0 else 0.0
        red_ci = cluster_boot(tk, ratio); abs_ci = cluster_boot(tk, lambda idx: float((au_v[idx] - ac_v[idx]).mean()))
        out["observation"] = dict(n_episodes=len(common), excluded_empty=dict(U=sum(u[k]["empty"] for k in keys if k in u), C=sum(c[k]["empty"] for k in keys if k in c)),
                                  short_episodes=dict(U=sum(u[k]["short"] for k in common), C=sum(c[k]["short"] for k in common)),
                                  abs_obs_bias_ry=dict(U=au, C=ac), relative_reduction=(au - ac) / au if au > 0 else None,
                                  relative_reduction_task_clustered_ci95_ratio_bootstrap=red_ci, absolute_reduction_task_clustered_ci95=abs_ci,
                                  uninformative_U_below_0_005=bool(au < 0.005),
                                  abs_est_err_ry=dict(U=float(np.mean([abs(u[k]["est_err_ry"]) for k in common])), C=float(np.mean([abs(c[k]["est_err_ry"]) for k in common]))),
                                  signed_mean_obs_err_ry=dict(U=float(np.mean([u[k]["obs_err_ry"] for k in common])), C=float(np.mean([c[k]["obs_err_ry"] for k in common]))),
                                  remaining_ry=dict(U=float(np.mean([u[k]["remaining_ry"] for k in common])), C=float(np.mean([c[k]["remaining_ry"] for k in common]))),
                                  clipped_fraction=dict(U=float(np.mean([u[k]["clipped_fraction"] for k in common])), C=float(np.mean([c[k]["clipped_fraction"] for k in common]))),
                                  prediction_1=dict(reduction_ge_25pct=bool(au > 0 and (au - ac) / au >= 0.25), ratio_ci_excludes_zero=bool(red_ci[0] > 0)))
    for arm in ("healthy_U", "healthy_C"):
        if arm in ob:
            v = [x for x in ob[arm].values() if not x["empty"]]
            out.setdefault("healthy_observation", {})[arm] = dict(n=len(v), mean_abs_obs_ry=float(np.mean([abs(x["obs_err_ry"]) for x in v])), max_abs_est_ry=float(max(abs(x["est_err_ry"]) for x in v)))
    js = json.dumps(out, indent=1)
    if a.out:
        a.out.write_text(js)
    print(json.dumps({k: v for k, v in out.items() if k != "per_episode"}, indent=1)[:5000])


if __name__ == "__main__":
    main()
