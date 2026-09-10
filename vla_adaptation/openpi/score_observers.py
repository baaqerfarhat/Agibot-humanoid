"""Score the six-arm, n=20 matched-observer evaluation from stored JSON only.

Require complete matched data before reporting any scores. Every paired p-value is
checked through both preregistered McNemar implementations. Regression denominators
are frozen-success opportunities within the same validated episode cohort.

Estimation summaries use terminal estimates on CORR, not trajectories: overall MAE
and RMSE, max_j |mean_episode(error_j)|, max_j mean_episode(|error_j|), and
max_episode,j |error_j|. Static-correction arms have no estimation score because
their saved f_hat is an unused shadow estimate.

    python3 openpi/score_observers.py <dir-of-eval-jsons>
    python3 openpi/score_observers.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

# Import only the two stdlib scoring modules, without creating sibling bytecode files.
_dont_write_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    if __package__:
        from . import mcnemar, mcnemar_crosscheck
    else:
        import mcnemar
        import mcnemar_crosscheck
finally:
    sys.dont_write_bytecode = _dont_write_bytecode

ARMS = [("P_proposed", "the paper's law, published constants"),
        ("D_dob", "plain calibrated DOB (P minus deadzone and normalizer)"),
        ("R_rls", "recursive least squares"),
        ("K_kalman", "random-walk Kalman"),
        ("I_integral", "calibrated integral (expected to rail)"),
        ("O_oracle", "oracle, exact fault")]
EXPECTED_EPISODES = 20
TRUE_FAULT = [0.0, 0.0, 0.0, 0.10, 0.10, 0.10]
CORR = (3, 4, 5)


def require(condition, message):
    """Input checks must remain active even under python -O."""
    if not condition:
        raise ValueError(message)


def exact_two_sided(b, c):
    """Use both registered instruments; do not maintain a third implementation."""
    p = mcnemar.mcnemar_exact(b, c)
    crosscheck = mcnemar_crosscheck.exact_two_sided(b, c)
    if not (math.isfinite(p) and math.isfinite(crosscheck)
            and 0 <= p <= 1 and 0 <= crosscheck <= 1
            and math.isclose(p, crosscheck, rel_tol=1e-12, abs_tol=0.0)):
        raise ValueError(
            f"McNemar implementations disagree for b={b}, c={c}: "
            f"mcnemar.py={p!r}, mcnemar_crosscheck.py={crosscheck!r}")
    return p


def upper95(broken, opportunities):
    require(0 <= broken <= opportunities, "invalid regression counts")
    if broken == opportunities:
        return 1.0  # Includes zero opportunities: no evidence of safety.
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        tail = sum(math.comb(opportunities, i) * mid ** i * (1 - mid) ** (opportunities - i)
                   for i in range(broken + 1))
        lo, hi = (mid, hi) if tail > 0.05 else (lo, mid)
    return hi


def regression_cell(broken, opportunities):
    bound = upper95(broken, opportunities)
    rate = (f"{100 * broken / opportunities:.1f}%" if opportunities
            else "n/a (no opportunities)")
    return (f"{broken}/{opportunities} = {rate}; "
            f"95% upper bound {100 * bound:.1f}%")


def require_same_set(actual, expected, label):
    actual, expected = set(actual), set(expected)
    require(actual == expected,
            f"{label}: missing {sorted(expected - actual)!r}; "
            f"unexpected {sorted(actual - expected)!r}")


def episode_outcomes(arm, label):
    require(isinstance(arm, dict), f"{label}: expected an arm object")
    episodes = arm.get("per_ep")
    require(isinstance(episodes, list), f"{label}: missing per_ep list")
    outcomes = {}
    for i, episode in enumerate(episodes):
        require(isinstance(episode, dict), f"{label}: per_ep[{i}] must be an object")
        task, init, ok = (episode.get(k) for k in ("task", "init", "ok"))
        require(type(task) is int and type(init) is int and task >= 0 and init >= 0,
                f"{label}: per_ep[{i}] needs nonnegative integer task and init")
        require(type(ok) in (bool, int) and ok in (0, 1),
                f"{label}: per_ep[{i}].ok must be boolean or 0/1")
        key = (task, init)
        require(key not in outcomes, f"{label}: duplicate (task, init) key {key}")
        outcomes[key] = int(ok)
    require(len(outcomes) == EXPECTED_EPISODES,
            f"{label}: expected {EXPECTED_EPISODES} unique episodes, got {len(outcomes)}")
    require(type(arm.get("n")) is int and arm["n"] == len(outcomes),
            f"{label}: recorded n must equal actual n={len(outcomes)}")
    require(type(arm.get("successes")) is int
            and arm["successes"] == sum(outcomes.values()),
            f"{label}: recorded successes disagree with per_ep")
    return outcomes


def require_vector(vector, label):
    require(isinstance(vector, list) and len(vector) == len(TRUE_FAULT),
            f"{label}: expected a six-coordinate vector")
    require(all(type(x) in (int, float) and math.isfinite(x) for x in vector),
            f"{label}: coordinates must be finite numbers")


def is_static_correction(args):
    require(isinstance(args, dict) and "static_corr" in args,
            "recorded args.static_corr is required to identify static correction")
    correction = args["static_corr"]
    if correction is None or correction == "":
        return False
    if isinstance(correction, str):
        try:
            correction = [float(x) for x in correction.split(",")]
        except ValueError:
            raise ValueError("args.static_corr must contain six finite numbers") from None
    require_vector(correction, "args.static_corr")
    return True


def load_record(data):
    require(isinstance(data, dict), "expected a result object")
    args = data.get("args")
    static = is_static_correction(args)
    require(type(args.get("episodes")) is int and args["episodes"] == EXPECTED_EPISODES,
            f"args.episodes must declare {EXPECTED_EPISODES} episodes")
    arms = data.get("arms")
    require(isinstance(arms, dict), "missing arms object")
    require_same_set(arms, ("frozen_faulted", "adaptive"), "recorded arm set")
    frozen = episode_outcomes(arms["frozen_faulted"], "frozen_faulted")
    corrected = episode_outcomes(arms["adaptive"], "adaptive")
    require_same_set(corrected, frozen, "frozen/corrected episode keys")
    keys = sorted(frozen)
    estimates = None
    if not static:
        estimates = arms["adaptive"].get("f_hat")
        require(isinstance(estimates, list) and len(estimates) == len(keys),
                f"adaptive.f_hat must contain one endpoint per episode (n={len(keys)})")
        for i, vector in enumerate(estimates):
            require_vector(vector, f"adaptive.f_hat[{i}]")
    return keys, frozen, corrected, estimates


def load(path):
    try:
        return load_record(json.loads(pathlib.Path(path).read_text()))
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None


def errs(fh):
    """Endpoint MAE, RMSE, max absolute mean bias, worst-coordinate MAE, max error."""
    require(isinstance(fh, list) and len(fh) > 0, "endpoint f_hat must not be empty")
    for i, vector in enumerate(fh):
        require_vector(vector, f"f_hat[{i}]")
    rows = [[v[i] - TRUE_FAULT[i] for i in CORR] for v in fh]
    flat = [x for row in rows for x in row]
    mae = math.fsum(abs(x) for x in flat) / len(flat)
    rmse = math.hypot(*flat) / math.sqrt(len(flat))
    mean_bias = [math.fsum(row[j] for row in rows) / len(rows) for j in range(len(CORR))]
    coord_mae = [math.fsum(abs(row[j]) for row in rows) / len(rows)
                 for j in range(len(CORR))]
    return mae, rmse, max(abs(x) for x in mean_bias), max(coord_mae), max(map(abs, flat))


def load_study(directory):
    directory = pathlib.Path(directory)
    require(directory.is_dir(), f"{directory}: evaluation directory does not exist")
    require_same_set((p.name for p in directory.glob("*.json")),
                     (f"{name}.json" for name, _ in ARMS),
                     f"{directory}: expected the six declared evaluation arm files")
    return {name: load(directory / f"{name}.json") for name, _ in ARMS}


def head_to_head(name, proposed, other):
    require_same_set(other, proposed, f"P_proposed/{name} episode keys")
    n = len(proposed)
    pw = sum(proposed[k] and not other[k] for k in proposed)
    ow = sum(other[k] and not proposed[k] for k in proposed)
    p = exact_two_sided(pw, ow)
    verdict = ("P better" if p < 0.05 and pw > ow else
               f"{name} better" if p < 0.05 else f"NOT RESOLVED at n={n}")
    return (f"  P vs {name:12s} n={n}, P-only {pw:2d}, {name}-only {ow:2d}, "
            f"p = {p:.4f}   {verdict}")


def report(rows):
    """Build the entire report before printing, so any failure emits no partial scores."""
    require_same_set(rows, (name for name, _ in ARMS), "evaluation arm set")
    keys = rows["P_proposed"][0]
    n = len(keys)
    require(n == EXPECTED_EPISODES, f"expected {EXPECTED_EPISODES} episodes, got {n}")
    for name, (ks, frozen, corrected, _) in rows.items():
        require(len(ks) == len(set(ks)), f"{name}: duplicate episode keys")
        require_same_set(ks, keys, f"P_proposed/{name} episode keys")
        require_same_set(frozen, keys, f"{name}/frozen episode keys")
        require_same_set(corrected, keys, f"{name}/corrected episode keys")

    header = (f"{'arm':12s} {'frozen':>7} {'corrected':>10} {'fixed':>6} {'brk':>4} {'p':>10} "
              f"{'MAE':>9} {'RMSE':>9} {'max_abs_coord_mean_bias':>23} "
              f"{'worst_coord_MAE':>16} {'max_endpoint_coord_error':>24}")
    lines = [f"Complete evaluation: six arms, n={n} matched episodes per arm.",
             "Every p-value agrees between mcnemar.py and mcnemar_crosscheck.py.",
             "Endpoint estimation errors on corrected channels (3, 4, 5):",
             "  max_abs_coord_mean_bias = max_j |mean_episode(error_j)|; "
             "worst_coord_MAE = max_j mean_episode(|error_j|);",
             "  max_endpoint_coord_error = max_episode,j |error_j|.",
             header, "-" * len(header)]
    regressions = []
    for name, _ in ARMS:
        _, frozen, corrected, estimates = rows[name]
        br = sum(frozen[k] and not corrected[k] for k in keys)
        fx = sum(corrected[k] and not frozen[k] for k in keys)
        p = exact_two_sided(br, fx)
        if estimates is None:
            es = "n/a (static correction) [all five estimation metrics]"
        else:
            require(len(estimates) == n, f"{name}: endpoint count differs from n={n}")
            es = "%9.6f %9.6f %23.6f %16.6f %24.6f" % errs(estimates)
        frozen_successes = sum(frozen[k] for k in keys)
        corrected_successes = sum(corrected[k] for k in keys)
        lines.append(f"{name:12s} {frozen_successes:4d}/{n:<2d} {corrected_successes:7d}/{n:<2d} "
                     f"{fx:6d} {br:4d} {p:10.4g} {es}")
        regressions.append(f"  {name:12s} {regression_cell(br, frozen_successes)}")

    lines.append("\nHEAD-TO-HEAD against the proposed law, paired on (task, init):")
    for name, _ in ARMS:
        if name != "P_proposed":
            lines.append(head_to_head(name, rows["P_proposed"][2], rows[name][2]))
    lines.append("\nRegression rate against OPPORTUNITIES (episodes the frozen policy solved):")
    lines.extend(regressions)
    return "\n".join(lines)


def selftest():
    """Synthetic JSON and CLI fixtures; no simulator, evaluation data, or file writes."""
    import contextlib
    import copy
    import io
    import random
    import unittest
    from unittest import mock

    def fixture():
        arms = {}
        for tag, divisor in (("frozen_faulted", 3), ("adaptive", 2)):
            per_ep = [dict(task=i % 10, init=45 + i // 10, ok=i % divisor == 0)
                      for i in range(EXPECTED_EPISODES)]
            arms[tag] = dict(per_ep=per_ep, n=len(per_ep),
                             successes=sum(e["ok"] for e in per_ep),
                             f_hat=[TRUE_FAULT[:] for _ in per_ep])
        return dict(args=dict(episodes=EXPECTED_EPISODES, static_corr=None), arms=arms)

    class ScorerTests(unittest.TestCase):
        def run_cli(self, documents):
            stdout, stderr = io.StringIO(), io.StringIO()
            paths = [pathlib.Path("/synthetic") / f"{name}.json" for name in documents]
            with mock.patch.object(pathlib.Path, "is_dir", return_value=True), \
                 mock.patch.object(pathlib.Path, "glob", return_value=paths), \
                 mock.patch.object(pathlib.Path, "read_text",
                                   lambda path: json.dumps(documents[path.stem])), \
                 contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                status = main(["/synthetic"])
            return status, stdout.getvalue(), stderr.getvalue()

        def assert_cli_rejected(self, documents, reason):
            status, stdout, stderr = self.run_cli(documents)
            self.assertNotEqual(status, 0)
            self.assertEqual(stdout, "")
            self.assertIn(reason, stderr)

        def test_duplicate_and_mismatched_keys(self):
            for tag in ("frozen_faulted", "adaptive"):
                data = fixture()
                eps = data["arms"][tag]["per_ep"]
                eps[-1] = copy.deepcopy(eps[0])
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    load_record(data)
            data = fixture()
            data["arms"]["adaptive"]["per_ep"][0]["init"] = 99
            with self.assertRaisesRegex(ValueError, "episode keys"):
                load_record(data)

        def test_cross_file_keys(self):
            documents = {name: fixture() for name, _ in ARMS}
            for arm in documents["D_dob"]["arms"].values():
                arm["per_ep"][0]["init"] = 99
            self.assert_cli_rejected(documents, "P_proposed/D_dob episode keys")

        def test_arm_set_and_episode_counts(self):
            documents = {name: fixture() for name, _ in ARMS}
            del documents["O_oracle"]
            self.assert_cli_rejected(documents, "O_oracle.json")
            documents["O_oracle"] = fixture()
            documents["sweep_point"] = fixture()
            self.assert_cli_rejected(documents, "unexpected ['sweep_point.json']")
            for change, reason in (("arm", "recorded arm set"),
                                   ("count", "expected 20 unique episodes"),
                                   ("declared", "args.episodes"),
                                   ("n", "recorded n"),
                                   ("successes", "recorded successes")):
                data = fixture()
                adaptive = data["arms"]["adaptive"]
                if change == "arm":
                    del data["arms"]["adaptive"]
                elif change == "count":
                    adaptive["per_ep"].pop()
                elif change == "declared":
                    data["args"]["episodes"] = 10
                elif change == "n":
                    adaptive["n"] = 19
                else:
                    adaptive["successes"] += 1
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, reason):
                    load_record(data)

        def test_missing_or_invalid_estimates(self):
            for value in (None, [], [TRUE_FAULT[:]] * 19,
                          [[0.0] * 5] * 20, [[float("nan")] * 6] * 20,
                          [[float("inf")] * 6] * 20):
                documents = {name: fixture() for name, _ in ARMS}
                documents["I_integral"]["arms"]["adaptive"]["f_hat"] = value
                self.assert_cli_rejected(documents, "f_hat")
            data = fixture()
            del data["arms"]["adaptive"]["f_hat"]
            with self.assertRaisesRegex(ValueError, "f_hat"):
                load_record(data)

        def test_both_mcnemar_implementations(self):
            rng = random.Random(20260907)
            cases = [(0, 0), (0, 20), (20, 0), (10, 10), (1, 51), (51, 1)]
            cases.extend((rng.randrange(101), rng.randrange(101)) for _ in range(200))
            for b, c in cases:
                first = mcnemar.mcnemar_exact(b, c)
                second = mcnemar_crosscheck.exact_two_sided(b, c)
                self.assertTrue(math.isclose(first, second, rel_tol=1e-12, abs_tol=0.0))
                self.assertEqual(exact_two_sided(b, c), first)
            self.assertEqual(exact_two_sided(1, 51), 106 / 2 ** 52)

        def test_mcnemar_disagreement_fails_before_output(self):
            documents = {name: fixture() for name, _ in ARMS}
            with mock.patch.object(mcnemar_crosscheck, "exact_two_sided", return_value=0.123):
                self.assert_cli_rejected(documents, "McNemar implementations disagree")
            with mock.patch.object(mcnemar_crosscheck, "exact_two_sided", return_value=6.12e-13):
                with self.assertRaisesRegex(ValueError, "implementations disagree"):
                    exact_two_sided(1, 51)

        def test_sign_cancelling_endpoint_errors(self):
            errors = [0.091296, 0.014994, -0.039430, 0.0]
            estimates = [[truth + (error if j in CORR else 0.0)
                          for j, truth in enumerate(TRUE_FAULT)] for error in errors]
            mae, rmse, bias, coord_mae, max_error = errs(estimates)
            for got, expected in zip((bias, coord_mae, max_error),
                                     (0.016715, 0.036430, 0.091296)):
                self.assertAlmostEqual(got, expected, places=12)
            self.assertLess(bias, coord_mae)
            self.assertLess(coord_mae, max_error)
            self.assertAlmostEqual(mae, 0.036430, places=12)
            self.assertAlmostEqual(rmse, math.sqrt(sum(x * x for x in errors) / 4))

        def test_static_correction_and_metadata(self):
            for value in ("0,0,0,-0.10,-0.10,-0.10", "0,0,0,0,0,0", [0.0] * 6):
                data = fixture()
                data["args"]["static_corr"] = value
                data["arms"]["adaptive"].pop("f_hat")
                self.assertTrue(is_static_correction(data["args"]))
                self.assertIsNone(load_record(data)[3])
            self.assertFalse(is_static_correction(fixture()["args"]))
            for args in (None, {}, dict(static_corr="bad"), dict(static_corr="0,0"),
                         dict(static_corr="nan,0,0,0,0,0")):
                with self.assertRaises(ValueError):
                    is_static_correction(args)
            documents = {name: fixture() for name, _ in ARMS}
            documents["O_oracle"]["args"]["static_corr"] = "0,0,0,-.1,-.1,-.1"
            # An unused shadow estimate must never determine the oracle metric.
            documents["O_oracle"]["arms"]["adaptive"]["f_hat"] = [[999.0] * 6] * 20
            status, stdout, stderr = self.run_cli(documents)
            self.assertEqual((status, stderr), (0, ""))
            oracle_line = next(line for line in stdout.splitlines() if line.startswith("O_oracle"))
            self.assertIn("n/a (static correction)", oracle_line)
            self.assertNotIn("999", oracle_line)

        def test_zero_opportunity_bound(self):
            self.assertEqual(upper95(0, 0), 1.0)
            self.assertEqual(regression_cell(0, 0),
                             "0/0 = n/a (no opportunities); 95% upper bound 100.0%")
            self.assertAlmostEqual(upper95(0, 20), 1 - 0.05 ** (1 / 20))

        def test_complete_report_and_actual_n(self):
            documents = {name: fixture() for name, _ in ARMS}
            # Baseline outcomes may differ between files, but use the same episode keys.
            frozen = documents["D_dob"]["arms"]["frozen_faulted"]
            for episode in frozen["per_ep"]:
                episode["ok"] = False
            frozen["successes"] = 0
            status, stdout, stderr = self.run_cli(documents)
            self.assertEqual((status, stderr), (0, ""))
            self.assertIn("six arms, n=20", stdout)
            self.assertIn("0/0 = n/a (no opportunities); 95% upper bound 100.0%", stdout)
            self.assertIn("3/7 = 42.9%; 95% upper bound", stdout)
            self.assertIn("max_abs_coord_mean_bias", stdout)
            self.assertIn("worst_coord_MAE", stdout)
            self.assertIn("max_endpoint_coord_error", stdout)
            self.assertEqual(stdout.count("NOT RESOLVED at n=20"), 5)
            short = {(0, i): 0 for i in range(7)}
            self.assertIn("NOT RESOLVED at n=7", head_to_head("test", short, short))

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ScorerTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="results/observers")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    try:
        output = report(load_study(args.directory))
    except (OSError, ValueError, OverflowError) as exc:
        print(f"ERROR: refusing to score incomplete or invalid evaluation: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
