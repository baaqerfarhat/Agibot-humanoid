"""Which channels decide how much the ALOHA law is allowed to learn?

The correction can act on six arm joints while its deadzone and normaliser see all 14
residuals. A scratch calculation put almost all the healthy prediction error in the right
gripper: a channel we never corrected was setting the scale of every arm update. That is
a claim about the measured residual, not something to assume from the robot's anatomy.

Fit the POSITION plant from aloha_adapt.py and pool squared residuals before taking shares.
--fit-log defaults to the healthy log; --eval-log defaults to the fit log. Different logs
fit ONCE on all fit-log episodes, then score the eval log without re-fitting: healthy fit /
faulted eval matches the deployed configuration. Same-log scoring is a re-fit diagnostic,
holding out whole episodes by default (--in-sample fits and scores all episodes together).
To re-fit a faulted log explicitly, use --fit-log FAULTED (eval defaults to that same log).
The legacy --log option now aliases --eval-log, so it does not silently re-fit faulted data.

K_FIR=6 means SEVEN taps, including the current command, plus an unregularised intercept.
adaptive_law.py's increment/ridge model is different.
The first six steps of each episode are omitted, exactly as in fit_plant: the logs do not
contain the initial position needed to reconstruct the online law's initial history.

Deadzone occupancy uses the law's strict nr < dead comparison (default dead=0.002).
Attenuation is reported at the selected --norm-r (default 0.05) and the headline-run 0.4.
These compare channel sets on the same scored residuals, not a replay of adaptation,
M_inv, clipping, or the trajectory changes a different gate would cause. Closing the gate
zeros est; the deployed update still decays f_hat toward zero.
"""
from __future__ import annotations

import argparse, json, pathlib
import numpy as np

# Copied from aloha_adapt.py, not imported: its simulator imports are unnecessary here.
# --selftest checks these constants and compares fitted coefficients to its actual function.
NJ, K_FIR, NORM_R = 14, 6, 0.05
DEAD, HEADLINE_NORM_R = 0.002, 0.4
GRIPPERS = (6, 13)
CORR_JOINTS = tuple(range(6))
DEFAULT_LOG = pathlib.Path(__file__).resolve().parents[1] / "results/aloha/healthy_log.json"
CHANNEL_NAMES = [f"left_arm_{j}" for j in range(6)] + ["left_gripper"] + [
    f"right_arm_{j}" for j in range(6)] + ["right_gripper"]


def validate_episodes(data):
    if not isinstance(data, list) or not data:
        raise ValueError("log must be a nonempty list of episodes")
    episodes = []
    for i, ep in enumerate(data):
        if not isinstance(ep, dict) or "u" not in ep or "q" not in ep:
            raise ValueError(f"episode {i}: expected u and q arrays")
        u, q = np.asarray(ep["u"], float), np.asarray(ep["q"], float)
        if u.ndim != 2 or u.shape[1] != NJ or q.shape != u.shape:
            raise ValueError(f"episode {i}: u and q must have matching (steps, {NJ}) shapes")
        if len(u) <= K_FIR:
            raise ValueError(f"episode {i}: need at least {K_FIR + 1} steps")
        if not np.isfinite(u).all() or not np.isfinite(q).all():
            raise ValueError(f"episode {i}: u and q must be finite")
        episodes.append(dict(u=u, q=q))
    return episodes


def design_episode(ep):
    """The exact regressor and target in aloha_adapt.fit_plant, within one episode."""
    u, q = ep["u"], ep["q"]
    X = np.array([[np.r_[u[t - K_FIR:t + 1, j][::-1], 1.0]
                   for t in range(K_FIR, len(u))] for j in range(NJ)])
    return X, q[K_FIR:]


def fit_plant(designs):
    W = np.zeros((NJ, K_FIR + 2))
    for j in range(NJ):
        X = np.concatenate([x[j] for x, _ in designs])
        Y = np.concatenate([y[:, j] for _, y in designs])
        W[j], *_ = np.linalg.lstsq(X, Y, rcond=None)
    return W


def prediction_residuals(episodes, in_sample=False, eval_episodes=None):
    """Separate eval data always use one plant fitted only on all supplied fit episodes."""
    if eval_episodes is not None and in_sample:
        raise ValueError("--in-sample applies only when fit and eval logs are the same")
    split = eval_episodes is not None
    if not split and not in_sample and len(episodes) < 2:
        raise ValueError("leave-one-episode-out needs at least two episodes; use --in-sample for one")
    designs = [design_episode(ep) for ep in episodes]
    eval_designs = [design_episode(ep) for ep in eval_episodes] if split else designs
    all_indices = list(range(len(episodes)))
    W_all = fit_plant(designs) if split or in_sample else None
    residuals, folds = [], []
    for i, (X, Y) in enumerate(eval_designs):
        train = all_indices if split or in_sample else [k for k in all_indices if k != i]
        W = W_all if split or in_sample else fit_plant([designs[k] for k in train])
        pred = np.column_stack([X[j] @ W[j] for j in range(NJ)])
        residuals.append(Y - pred)
        folds.append(dict(evaluation_episode=i, training_episodes=train,
                          training_steps=sum(len(designs[k][1]) for k in train),
                          evaluation_steps=len(Y)))
    return np.concatenate(residuals), folds


def distribution(values):
    """Finite-only statistics, with singular cases counted rather than hidden or JSON NaN."""
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    out = dict(count=int(values.size), finite_count=int(finite.size),
               positive_infinity_count=int(np.isposinf(values).sum()),
               undefined_count=int(np.isnan(values).sum()))
    keys = ("min", "p01", "p05", "p25", "median", "p75", "p95", "max")
    out.update(dict(zip(keys, map(float, np.percentile(finite, [0, 1, 5, 25, 50, 75, 95, 100]))))
               if finite.size else dict.fromkeys(keys))
    out["mean"] = float(finite.mean()) if finite.size else None
    return out


def scalar_ratio(numerator, denominator):
    if denominator > 0:
        return dict(value=float(numerator / denominator), status="finite")
    return dict(value=None, status="infinite" if numerator > 0 else "undefined")


def summarize_residuals(residuals, corr_joints, norm_r, dead=DEAD):
    with np.errstate(over="raise", invalid="raise"):
        energy = (residuals ** 2).sum(axis=0)
    total = float(energy.sum())
    shares = energy / total if total > 0 else [None] * NJ
    channels = [dict(index=j, name=CHANNEL_NAMES[j], squared_residual=float(energy[j]),
                     share=float(shares[j]) if total > 0 else None) for j in range(NJ)]
    channels.sort(key=lambda ch: (-ch["squared_residual"], ch["index"]))

    def group(indices):
        ss = float(energy[list(indices)].sum())
        return dict(indices=list(indices), squared_residual=ss,
                    share=ss / total if total > 0 else None)

    grippers, corrected = group(GRIPPERS), group(corr_joints)
    nr_all = np.linalg.norm(residuals, axis=1)
    nr_corr = np.linalg.norm(residuals[:, corr_joints], axis=1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        norm_ratio = nr_all / nr_corr       # positive / 0 = inf; 0 / 0 = undefined

    def attenuation(rho):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            att_all = 1.0 / (1.0 + (nr_all / rho) ** 2)
            att_corr = 1.0 / (1.0 + (nr_corr / rho) ** 2)
            att_ratio = att_corr / att_all
        return dict(norm_r=float(rho), selected=rho == norm_r,
                    headline_run_setting=rho == HEADLINE_NORM_R,
                    attenuation_all=distribution(att_all),
                    attenuation_corrected=distribution(att_corr),
                    attenuation_ratio_corrected_over_all=distribution(att_ratio))

    attenuations = [attenuation(rho) for rho in dict.fromkeys((norm_r, HEADLINE_NORM_R))]
    distributions = dict(nr_all=distribution(nr_all), nr_corrected=distribution(nr_corr),
                         norm_ratio_all_over_corrected=distribution(norm_ratio),
                         **{key: value for key, value in attenuations[0].items()
                            if key.startswith("attenuation_")})

    def occupancy(norm):
        below = int(np.count_nonzero(norm < dead))
        stats = distribution(norm)
        return dict(below_count=below, step_count=len(norm), fraction_below=below / len(norm),
                    **{key: stats[key] for key in ("min", "p01", "median")})

    return dict(total_squared_residual=total, channels=channels, grippers=grippers,
                corrected_channels=corrected,
                gripper_to_corrected_energy_ratio=scalar_ratio(
                    grippers["squared_residual"], corrected["squared_residual"]),
                distributions=distributions, attenuation_by_norm_r=attenuations,
                deadzone_occupancy=dict(dead=float(dead), comparison="nr < dead",
                                        nr_all=occupancy(nr_all), nr_corrected=occupancy(nr_corr)))


def analyze(data, corr_joints=CORR_JOINTS, norm_r=NORM_R, in_sample=False, *,
            eval_data=None, dead=DEAD):
    episodes = validate_episodes(data)
    eval_episodes = validate_episodes(eval_data) if eval_data is not None else None
    corr_joints = tuple(corr_joints)
    if (not corr_joints or len(set(corr_joints)) != len(corr_joints)
            or any(not isinstance(j, (int, np.integer)) or j < 0 or j >= NJ for j in corr_joints)):
        raise ValueError("corr-joints must be distinct integer indices in 0..13, e.g. 0,1,2,3,4,5")
    if not np.isfinite(norm_r) or norm_r <= 0:
        raise ValueError("norm-r must be finite and strictly positive")
    if not np.isfinite(dead) or dead < 0:
        raise ValueError("dead must be finite and nonnegative")
    residuals, folds = prediction_residuals(episodes, in_sample, eval_episodes)
    if not np.isfinite(residuals).all():
        raise ValueError("nonfinite fitted residuals")
    split = eval_episodes is not None
    mode = ("fixed-fit/eval" if split else
            "same-log-refit/in-sample" if in_sample else "same-log-refit/leave-one-episode-out")
    evaluation_count = len(eval_episodes) if split else len(episodes)
    fit_note = ("Fit ONCE on all fit-log episodes; score eval-log episodes without re-fitting. "
                "Healthy fit / faulted eval matches the deployed fit/eval configuration." if split else
                "Same-log re-fit diagnostic: fit and score all episodes together (in-sample)." if in_sample else
                "Same-log re-fit diagnostic: fit other episodes separately for each held-out episode.")
    return dict(schema_version=2, mode=mode, fit_description=fit_note,
                fit_episode_count=len(episodes), episode_count=evaluation_count,
                evaluated_steps=len(residuals),
                omitted_warmup_steps=K_FIR * evaluation_count, norm_r=float(norm_r), dead=float(dead),
                model=dict(source="openpi/aloha_adapt.py:fit_plant", nj=NJ, k_fir=K_FIR,
                           taps=K_FIR + 1, regressor="[u[t,j], ..., u[t-6,j], 1.0]",
                           target="q[t,j] (position)", intercept=True,
                           solver="numpy.linalg.lstsq(rcond=None)", regularization=None,
                           deviation=None),
                notes=[fit_note,
                       "Fold training indices refer to fit-log; evaluation indices refer to eval-log.",
                       "Score t >= 6 independently per episode; initial online history is not logged.",
                       "Shares are fractions of pooled squared residual, not means of episode shares.",
                       "Distribution statistics use finite steps only; singular ratios are counted.",
                       "Headline stored runs use rho=0.4; rho=0.05 is the CLI/early-run setting.",
                       "Attenuation and gate occupancy compare the same residuals; no closed-loop replay.",
                       "The strict deadzone zeros est; f_hat still decays toward zero when it fires."],
                folds=folds, **summarize_residuals(residuals, corr_joints, norm_r, dead))


def stored_run_settings():
    """Read the actual stored args; retain filenames so the rho labels are auditable."""
    runs = []
    for path in sorted(DEFAULT_LOG.parent.glob("*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and isinstance(data.get("args"), dict):
            args = data["args"]
            runs.append(dict(file=path.name, norm_r=args.get("norm_r"), dead=args.get("dead")))
    return dict(source="results/aloha/*.json args", headline_norm_r=HEADLINE_NORM_R, runs=runs)


def print_report(report):
    def number(v):
        return "undefined" if v is None else f"{v:.9g}"

    def percent(v):
        return "undefined" if v is None else f"{100 * v:.9f}%"

    print(f"Fit log:  {report['fit_log']}")
    print(f"Eval log: {report['eval_log']}")
    print(f"Mode: {report['mode']}; {report['fit_episode_count']} fit episodes; "
          f"{report['episode_count']} eval episodes; "
          f"{report['evaluated_steps']} scored steps; {report['omitted_warmup_steps']} warmup steps omitted")
    print(report["fit_description"])
    print("Plant: q[t,j] ~ u[t,j] ... u[t-6,j] + intercept; per-channel lstsq, no ridge")
    print(f"Selected rho={report['norm_r']:g}; headline stored runs use rho={HEADLINE_NORM_R:g}.")
    runs = report["stored_run_settings"]["runs"]
    for key in ("norm_r", "dead"):
        for value in sorted({row[key] for row in runs if row[key] is not None}):
            files = [row["file"] for row in runs if row[key] == value]
            detail = f" ({', '.join(files)})" if key == "norm_r" and value == NORM_R else ""
            print(f"Stored args: {key}={value:g} in {len(files)}/{len(runs)} runs{detail}")
    print(f"Total squared residual: {report['total_squared_residual']:.12g}")
    print(f"\n{'index':>5}  {'channel':<16} {'squared residual':>18} {'share of total':>17}")
    for ch in report["channels"]:
        print(f"{ch['index']:5d}  {ch['name']:<16} {ch['squared_residual']:18.10g} {percent(ch['share']):>17}")
    print(f"\n{'group':<32} {'squared residual':>18} {'share of total':>17}")
    for label, key in (("Grippers [6, 13]", "grippers"), ("Corrected", "corrected_channels")):
        g = report[key]
        label = label if key == "grippers" else f"Corrected {g['indices']}"
        print(f"{label:<32} {g['squared_residual']:18.10g} {percent(g['share']):>17}")
    ratio = report["gripper_to_corrected_energy_ratio"]
    print("Gripper / corrected squared-residual ratio: " +
          (number(ratio["value"]) if ratio["status"] == "finite" else ratio["status"]))

    gate = report["deadzone_occupancy"]
    print(f"\nDeadzone occupancy; strict nr < dead={gate['dead']:g}; this fit/eval configuration")
    print(f"{'norm':<18} {'below / steps':>16} {'fraction below':>17} {'min':>14} {'p01':>14} {'median':>14}")
    for key in ("nr_all", "nr_corrected"):
        row = gate[key]
        count = f"{row['below_count']}/{row['step_count']}"
        values = " ".join(f"{number(row[k]):>14}" for k in ("min", "p01", "median"))
        print(f"{key:<18} {count:>16} {percent(row['fraction_below']):>17} {values}")
    print("Below threshold: est is zeroed; f_hat still decays. Occupancy excludes the first six steps/episode.")

    def print_distribution(label, d):
        values = " ".join(f"{number(d[k]):>12}" for k in ("mean", "min", "p05", "median", "p95", "max"))
        print(f"{label:<35} {values}")
        if d["finite_count"] != d["count"]:
            print(f"  finite={d['finite_count']}/{d['count']}; "
                  f"+inf={d['positive_infinity_count']}; undefined={d['undefined_count']}")

    print("\nPer-step norm distributions")
    columns = f"{'quantity':<35} {'mean':>12} {'min':>12} {'p05':>12} {'median':>12} {'p95':>12} {'max':>12}"
    print(columns)
    for label, key in (("nr all", "nr_all"), ("nr corrected", "nr_corrected"),
                       ("nr all / corrected", "norm_ratio_all_over_corrected")):
        print_distribution(label, report["distributions"][key])
    for row in report["attenuation_by_norm_r"]:
        labels = (["selected"] if row["selected"] else []) + (
            ["headline stored-run setting"] if row["headline_run_setting"] else ["diagnostic radius"])
        print(f"\nAttenuation = 1/(1+(nr/rho)^2); rho={row['norm_r']:g} ({'; '.join(labels)})")
        print(columns)
        for label, key in (("attenuation all", "attenuation_all"),
                           ("attenuation corrected", "attenuation_corrected"),
                           ("attenuation corrected / all", "attenuation_ratio_corrected_over_all")):
            print_distribution(label, row[key])
    print("Statistics pool scored steps (finite values only). The paired attenuation ratio measures")
    print("the factor increase from using corrected-channel norms on these residuals; no law replay.")


def selftest():
    """Known residuals orthogonal to a full-rank plant, then a separate leakage trap."""
    rng = np.random.default_rng(917)
    weights = np.array([1, 2, 3, 5, 7, 11, 43, 13, 17, 19, 23, 29, 31, 701], float) * 1e-6
    W_true = rng.normal(size=(NJ, K_FIR + 2))
    W_true[:, -1] += np.arange(NJ) + 2.0
    data, expected_residuals, expected_energy = [], [], np.zeros(NJ)
    for e, length in enumerate((67, 91, 128)):
        n = length - K_FIR
        u = rng.normal(size=(length, NJ)) + (e + 1) * np.linspace(-2, 2, NJ)
        q = np.full_like(u, 1e6)  # warmup must never enter the fit or the score
        noise = np.empty((n, NJ))
        variances = weights * (1 + e * np.linspace(0.2, 1.8, NJ))
        for j in range(NJ):
            # Independent column construction, not design_episode; every tap matters.
            X = np.column_stack([u[K_FIR - lag:length - lag, j]
                                 for lag in range(K_FIR + 1)] + [np.ones(n)])
            Q, _ = np.linalg.qr(X, mode="reduced")
            z = rng.normal(size=n)
            z -= Q @ (Q.T @ z)
            noise[:, j] = z * np.sqrt(n * variances[j] / (z @ z))
            q[K_FIR:, j] = X @ W_true[j] + noise[:, j]
        data.append(dict(u=u, q=q))
        expected_residuals.append(noise)
        expected_energy += n * variances
    expected_residuals = np.concatenate(expected_residuals)
    designs = [design_episode(ep) for ep in data]
    for ep, (X, Y) in zip(data, designs):
        np.testing.assert_array_equal(X[:, 0, :-1], ep["u"][K_FIR::-1].T)
        np.testing.assert_array_equal(X[:, :, -1], 1.0)
        np.testing.assert_array_equal(Y, ep["q"][K_FIR:])
    for in_sample in (False, True):
        actual, folds = prediction_residuals(data, in_sample)
        np.testing.assert_allclose(actual, expected_residuals, rtol=1e-9, atol=2e-12)
        for fold in folds:
            train = [designs[k] for k in fold["training_episodes"]]
            np.testing.assert_allclose(fit_plant(train), W_true, rtol=1e-10, atol=2e-12)
        report = analyze(data, in_sample=in_sample)
        got = {ch["index"]: ch for ch in report["channels"]}
        np.testing.assert_allclose([got[j]["squared_residual"] for j in range(NJ)], expected_energy,
                                   rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose([got[j]["share"] for j in range(NJ)],
                                   expected_energy / expected_energy.sum(), rtol=1e-9)
        np.testing.assert_allclose(report["corrected_channels"]["share"],
                                   expected_energy[:6].sum() / expected_energy.sum(), rtol=1e-9)
        np.testing.assert_allclose(report["grippers"]["share"],
                                   expected_energy[[6, 13]].sum() / expected_energy.sum(), rtol=1e-9)
        a = np.sqrt((expected_residuals ** 2).sum(axis=1))
        c = np.sqrt((expected_residuals[:, :6] ** 2).sum(axis=1))
        direct = dict(nr_all=a, nr_corrected=c, norm_ratio_all_over_corrected=a / c,
                      attenuation_all=1 / (1 + (a / NORM_R) ** 2),
                      attenuation_corrected=1 / (1 + (c / NORM_R) ** 2),
                      attenuation_ratio_corrected_over_all=(1 + (a / NORM_R) ** 2) / (1 + (c / NORM_R) ** 2))
        for key, vals in direct.items():
            stats = report["distributions"][key]
            np.testing.assert_allclose(stats["mean"], vals.mean(), rtol=1e-9)
            np.testing.assert_allclose([stats[k] for k in ("min", "p01", "p05", "p25", "median", "p75", "p95", "max")],
                                       np.percentile(vals, [0, 1, 5, 25, 50, 75, 95, 100]), rtol=1e-9)
        assert [row["norm_r"] for row in report["attenuation_by_norm_r"]] == [NORM_R, HEADLINE_NORM_R]
        headline = report["attenuation_by_norm_r"][1]
        np.testing.assert_allclose(headline["attenuation_all"]["median"],
                                   np.median(1 / (1 + (a / HEADLINE_NORM_R) ** 2)), rtol=1e-9)
        np.testing.assert_allclose(headline["attenuation_corrected"]["median"],
                                   np.median(1 / (1 + (c / HEADLINE_NORM_R) ** 2)), rtol=1e-9)
        json.dumps(report, allow_nan=False)
    print("PASS: seven taps, intercept, episode boundaries, known pooled shares and per-step factors")

    # Constant inputs leave only an intercept. Unequal lengths make training-row weights
    # observable; different episode offsets make any held-out leakage change the answer.
    sizes, offsets = np.array([9, 17, 26]), np.array([-0.7, 0.2, 1.6])
    leakage_data = [dict(u=np.zeros((int(n) + K_FIR, NJ)),
                         q=np.full((int(n) + K_FIR, NJ), v)) for n, v in zip(sizes, offsets)]
    for in_sample in (False, True):
        expected = []
        for i, (n, v) in enumerate(zip(sizes, offsets)):
            keep = np.ones(3, dtype=bool) if in_sample else np.arange(3) != i
            intercept = np.sum(sizes[keep] * offsets[keep]) / sizes[keep].sum()
            expected.append(np.full((int(n), NJ), v - intercept))
        actual, _ = prediction_residuals(leakage_data, in_sample)
        np.testing.assert_allclose(actual, np.concatenate(expected), rtol=1e-12, atol=1e-12)
    print("PASS: leave-one-episode-out excludes held-out targets; in-sample and row weights differ")

    # Full-rank healthy fit and distinct faulted commands/targets: a fixed offset is
    # retained by healthy fitting but is exactly absorbed by the faulted intercept.
    # Use different fit/eval episode counts and lengths to expose accidental data reuse.
    healthy, faulted = [], []
    offset = np.zeros(NJ)
    offset[[0, 13]] = [0.125, 0.25]
    for dest, lengths, shift in ((healthy, (53, 71), np.zeros(NJ)),
                                  (faulted, (43, 59, 83), offset)):
        for length in lengths:
            u = rng.normal(size=(length, NJ))
            q = np.full_like(u, 1e6)
            for j in range(NJ):
                X = np.column_stack([u[K_FIR - lag:length - lag, j]
                                     for lag in range(K_FIR + 1)] + [np.ones(length - K_FIR)])
                q[K_FIR:, j] = X @ W_true[j] + shift[j]
            dest.append(dict(u=u, q=q))
    actual, folds = prediction_residuals(healthy, eval_episodes=faulted)
    expected = np.tile(offset, (sum(len(ep["u"]) - K_FIR for ep in faulted), 1))
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=2e-12)
    assert all(fold["training_episodes"] == [0, 1] and fold["training_steps"] == 112 for fold in folds)
    assert [fold["evaluation_steps"] for fold in folds] == [37, 53, 77]
    for in_sample in (False, True):
        refitted, _ = prediction_residuals(faulted, in_sample)
        np.testing.assert_allclose(refitted, 0, atol=2e-12)
    split_report = analyze(healthy, eval_data=faulted)
    assert split_report["mode"] == "fixed-fit/eval"
    assert (split_report["fit_episode_count"], split_report["episode_count"]) == (2, 3)
    np.testing.assert_allclose(split_report["total_squared_residual"], 167 * (0.125 ** 2 + 0.25 ** 2))
    np.testing.assert_allclose(split_report["distributions"]["nr_corrected"]["median"], 0.125)
    assert analyze(faulted)["mode"] == "same-log-refit/leave-one-episode-out"
    assert analyze(faulted, in_sample=True)["mode"] == "same-log-refit/in-sample"
    analyze(healthy[:1], eval_data=faulted[:1])  # a fixed split needs no held-out fit episode
    json.dumps(split_report, allow_nan=False)
    print("PASS: healthy fit / separate faulted eval retains offset; explicit faulted re-fit absorbs it")

    # Known norms test strict inequality at the threshold, zeros, and both channel sets.
    known = np.zeros((5, NJ))
    known[1, 13] = DEAD
    known[2, 0] = DEAD / 2
    known[3, 0] = DEAD
    known[4, [0, 13]] = [0.003, 0.004]
    gate_fit = [dict(u=np.zeros((10, NJ)), q=np.zeros((10, NJ)))]
    gate_eval = [dict(u=np.zeros((11, NJ)), q=np.vstack([np.full((K_FIR, NJ), 1e6), known]))]
    gate_report = analyze(gate_fit, eval_data=gate_eval)
    gate = gate_report["deadzone_occupancy"]
    assert gate["dead"] == DEAD and gate["comparison"] == "nr < dead"
    for key, count, stats in (("nr_all", 2, [0, 0.00004, 0.002]),
                              ("nr_corrected", 3, [0, 0, 0.001])):
        row = gate[key]
        assert row["below_count"] == count and row["step_count"] == 5
        assert row["fraction_below"] == count / 5
        np.testing.assert_allclose([row[k] for k in ("min", "p01", "median")], stats)
    disabled = analyze(gate_fit, eval_data=gate_eval, dead=0)["deadzone_occupancy"]
    assert disabled["nr_all"]["below_count"] == disabled["nr_corrected"]["below_count"] == 0
    custom_gate = analyze(gate_fit, eval_data=gate_eval, corr_joints=(13,))["deadzone_occupancy"]
    assert custom_gate["nr_corrected"]["fraction_below"] == 3 / 5
    headline_only = analyze(gate_fit, eval_data=gate_eval, norm_r=HEADLINE_NORM_R)
    assert len(headline_only["attenuation_by_norm_r"]) == 1
    json.dumps(gate_report, allow_nan=False)
    print("PASS: deadzone occupancy, min/p01/median, strict threshold, zero threshold and custom channels")

    residuals = np.zeros((3, NJ))
    residuals[1, 13] = 2
    residuals[2, [0, 13]] = [3, 4]
    edge = summarize_residuals(residuals, CORR_JOINTS, 1.0)
    d = edge["distributions"]
    assert d["norm_ratio_all_over_corrected"]["finite_count"] == 1
    assert d["norm_ratio_all_over_corrected"]["positive_infinity_count"] == 1
    assert d["norm_ratio_all_over_corrected"]["undefined_count"] == 1
    np.testing.assert_allclose(d["norm_ratio_all_over_corrected"]["mean"], 5 / 3)
    np.testing.assert_allclose(d["attenuation_all"]["mean"], (1 + 1 / 5 + 1 / 26) / 3)
    np.testing.assert_allclose(d["attenuation_corrected"]["mean"], (1 + 1 + 1 / 10) / 3)
    np.testing.assert_allclose(edge["gripper_to_corrected_energy_ratio"]["value"], 20 / 9)
    custom = summarize_residuals(residuals, (13,), 1.0)
    np.testing.assert_allclose(custom["corrected_channels"]["share"], 20 / 29)
    zero = summarize_residuals(np.zeros((2, NJ)), CORR_JOINTS, NORM_R)
    assert all(ch["share"] is None for ch in zero["channels"])
    assert zero["gripper_to_corrected_energy_ratio"]["status"] == "undefined"
    assert zero["distributions"]["norm_ratio_all_over_corrected"]["mean"] is None
    only_gripper = summarize_residuals(residuals[1:2], CORR_JOINTS, NORM_R)
    assert only_gripper["gripper_to_corrected_energy_ratio"]["status"] == "infinite"
    json.dumps([edge, custom, zero, only_gripper], allow_nan=False)
    for bad, kwargs in (([], {}), (data[:1], {}), ([dict(u=[[0]], q=[[0]])], {}),
                        ([dict(u=np.zeros((6, NJ)), q=np.zeros((6, NJ)))], {}),
                        ([dict(u=np.full((7, NJ), np.nan), q=np.zeros((7, NJ)))], {}),
                        (data, dict(corr_joints=())), (data, dict(corr_joints=(0, 0))),
                        (data, dict(corr_joints=(14,))), (data, dict(norm_r=0)),
                        (data, dict(norm_r=float("nan"))), (data, dict(dead=-0.001)),
                        (data, dict(dead=float("nan"))), (data, dict(dead=float("inf"))),
                        (data, dict(eval_data=[])), (data, dict(eval_data=data, in_sample=True))):
        try:
            analyze(bad, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input was accepted")
    analyze(data[:1], in_sample=True)
    print("PASS: custom correction set, singular ratios, strict JSON and input validation")

    # Check against the real source without importing its simulator dependencies or writing
    # a temporary log. Execute only fit_plant, with an in-memory pathlib stand-in.
    import ast, types
    source = pathlib.Path(__file__).with_name("aloha_adapt.py")
    tree = ast.parse(source.read_text())
    constants = next(node for node in tree.body if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Tuple)
                     and [n.id for n in node.targets[0].elts] == ["NJ", "HORIZON", "K_FIR", "DT"])
    nj, _, k_fir, _ = ast.literal_eval(constants.value)
    assert (NJ, K_FIR) == (nj, k_fir), "aloha_adapt constants changed"
    for flag, expected_default in (("--norm-r", NORM_R), ("--dead", DEAD)):
        cli_arg = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                       and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == flag)
        assert expected_default == ast.literal_eval(next(kw.value for kw in cli_arg.keywords if kw.arg == "default"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "fit_plant")
    payload = json.dumps([dict(u=ep["u"].tolist(), q=ep["q"].tolist()) for ep in data])
    memory_pathlib = types.SimpleNamespace(Path=lambda _: types.SimpleNamespace(read_text=lambda: payload))
    namespace = dict(np=np, json=json, pathlib=memory_pathlib, NJ=nj, K_FIR=k_fir)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
    reference_W, _ = namespace["fit_plant"]("memory")
    np.testing.assert_allclose(fit_plant(designs), reference_W, rtol=1e-12, atol=1e-12)
    print("PASS: constants and fitted coefficients match aloha_adapt.py without simulator imports")
    print("selftest: all assertions passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-log", type=pathlib.Path, default=DEFAULT_LOG,
                    help="training episode JSON (default: results/aloha/healthy_log.json)")
    ap.add_argument("--eval-log", "--log", dest="eval_log", type=pathlib.Path,
                    help="evaluation JSON (default: fit-log); --log is a legacy alias for eval-log")
    ap.add_argument("--corr-joints", default="0,1,2,3,4,5", help="comma-separated corrected indices (default: 0..5)")
    ap.add_argument("--norm-r", type=float, default=NORM_R,
                    help="selected normaliser radius (default: 0.05); also report headline-run rho=0.4")
    ap.add_argument("--dead", type=float, default=DEAD, help="deadzone threshold (default: 0.002; strict nr < dead)")
    ap.add_argument("--in-sample", action="store_true",
                    help="same-log diagnostic: fit all episodes and score them (default: leave-one-episode-out)")
    ap.add_argument("--json", type=pathlib.Path, help="write a machine-readable report (shares are fractions)")
    ap.add_argument("--selftest", action="store_true", help="run synthetic assertions and source parity check, then exit")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    try:
        corr = tuple(int(j) for j in a.corr_joints.split(","))
        fit_log = a.fit_log.resolve()
        eval_log = a.eval_log.resolve() if a.eval_log is not None else fit_log
        # samefile also catches hard-linked aliases, beyond resolve's symlink handling.
        same_log = fit_log.samefile(eval_log)
        if a.json is not None and (a.json.resolve() in (fit_log, eval_log) or
                                  (a.json.exists() and any(a.json.samefile(p) for p in (fit_log, eval_log)))):
            raise ValueError("--json must not overwrite either input log")
        fit_data = json.loads(fit_log.read_text())
        eval_data = None if same_log else json.loads(eval_log.read_text())
        report = analyze(fit_data, corr, a.norm_r, a.in_sample, eval_data=eval_data, dead=a.dead)
        report.update(log=str(eval_log), fit_log=str(fit_log), eval_log=str(eval_log),
                      stored_run_settings=stored_run_settings())
        if a.json is not None:
            a.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, TypeError, FloatingPointError) as exc:
        ap.error(str(exc))
    print_report(report)


if __name__ == "__main__":
    main()
