#!/usr/bin/env python3
"""Prospective physical-benefit forecast for the FrozenYet Adaptive recovery campaign
(EXPERIMENT_PLAN.md sections 2-4; RESEARCH_DECISION_20260916.md, theory A).

Four sub-commands, run in this order and frozen between them:

  fit        Fit the forecast model on OLD development continuation runs (E1 v2 pass1/pass2 format or fya format):
               * physical response: signed finite-memory linear model e_k = sum_{j<L} G_j d_{k-j} (end-effector translation
                 deviation from the healthy continuation, m; d = remaining command disturbance f + c on all six channels,
                 normalised units), L = 20, ridge 1e-6 (e1_memory_model.fit_memory);
               * residual model for the observer simulation: r_k = M f + (M - G_fit) c_k + nu_k, with nu_k drawn as WHOLE
                 healthy residual sequences from the development pool (they already carry the healthy mean b_h, which the
                 model stores for reference only; keeps autocorrelation and cross-channel correlation; no test data enters);
               * simple predictive controls selected on the same data: forecast mean |fhat - f| on the corrected channels
                 and forecast mean residual norm, reported next to the energy forecast.
  predict    For each source of a fresh log (nominal commands only; no branch data is read) and each scenario x arm, simulate
             the deployed observer update (adaptive_law.estimator_step for NT and innovation, cap, delay) under n_draws
             residual-noise draws. The update function is the deployed one, but the residual it sees comes from the
             simplified model above, not from the FIR predictor run on the actual command history: this is a
             simulation of a reduced loop, not of the deployed feedback loop. Map remaining disturbance to physical
             deviation, and write predictions.csv with the
             forecast trajectory norm ||Ehat||_Q (m sqrt(s)) and forecast energy Jhat = ||Ehat||_Q^2 (m^2 s), Q = dt I, plus
             the estimator-error / residual controls. The file is hashed; freeze it before the corresponding run is read.
  calibrate  On the QUALIFICATION run: per source, s_i = max over all compared branches of ||E - Ehat||_Q; the width rule is
             epsilon = max_i s_i (declared before qualification; the 90th percentile is reported as a secondary width).
             Writes calibration.json. If the model, metric or rule changes after inspection, the partition is development.
  evaluate   On the LOCKED test run with frozen predictions and calibration: for each source, scenario and law A,
             L_i = max(0, ||Ehat_i||_Q - eps)^2, U_i = (||Ehat_i||_Q + eps)^2, predicted benefit interval
             [L_off - U_A, U_off - L_A]; label benefit (lower > 0), harm (upper < 0), inconclusive; joint source coverage
             (all compared branches within eps), individual coverage, decisive counts, false-help keys, predicted-harm errors,
             forecast error, and the same evaluation for the simple controls (sign of predicted benefit from estimator error).

All physical quantities are translation only; angular energy is scored separately by fya_score.py. The forecast conditions
on the registered scenario fault (available offline); the deployed observer never sees it.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import adaptive_law as AL
from e1_memory_model import fit_memory, predict_memory, L
from fya_continuations import DEFAULT_SCENARIOS, ARMS, LAW, branch_specs

DT = 0.05


def read_json(path):
    """json or gzip-compressed json (campaign runs are stored .json.gz in the repository)."""
    import gzip
    path = pathlib.Path(path)
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    return json.loads(path.read_text())


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def healthy_name(br):
    return "healthy_off" if "healthy_off" in br else "healthy"


def dev_series(run):
    """(e, d) pairs for every non-healthy branch and the healthy residual sequences of a development run (either format)."""
    d = read_json(run); pairs = []; healthy_res = []
    for ep in d["episodes"]:
        cps = ep.get("checkpoints") or ([ep] if ep.get("status") == "ok" else [])
        for cp in cps:
            if cp.get("status") != "ok":
                continue
            br = cp["branches"]; h = br[healthy_name(br)]
            healthy_res.append(np.array([s["residual"] for s in h]))
            for name, b in br.items():
                if name.startswith("healthy"):
                    continue
                e = np.array([np.array(s["ee_pos"]) - np.array(hh["ee_pos"]) for s, hh in zip(b, h)])
                dd = np.array([s["remaining_disturbance"] for s in b])
                pairs.append((e, dd))
    return pairs, healthy_res, d["meta"]


def simulate(scenario, arm, W, M, M_inv, mask, consts, healthy_cap, H, noise, b_h):
    """Forward observer simulation under the residual model; returns d (H x 6), fhat (H x 6), residual (H x 6)."""
    K = W.shape[1] - 2; G_fit = np.diag(W[:, :K + 1].sum(1))
    f = np.asarray(scenario["fault"], float) if scenario is not None else np.zeros(6)
    cap = float(scenario["cap"]) if scenario is not None else float(healthy_cap)
    delay = int(scenario.get("delay", 0)) if scenario is not None else 0
    adapt = arm in ("nt", "innovation"); law = LAW.get(arm)
    f_hat = np.zeros(6); state = None; D, FH, RR = [], [], []
    ref_corr = -np.clip(f, -cap, cap) * mask
    for k in range(H):
        if arm == "reference":
            c = ref_corr if k >= delay else np.zeros(6)
        elif adapt:
            c = -f_hat * mask
        else:
            c = np.zeros(6)
        # noise[k] is a whole healthy residual sequence from the development pool and already carries the healthy
        # mean b_h; adding b_h again doubled the expected healthy residual (found in review 2026-09-16 14:10, fixed
        # before the test forecast was written; the qualification forecast was regenerated with this code).
        r = M @ f + (M - G_fit) @ c + noise[k]
        D.append(f + c); FH.append(f_hat.copy()); RR.append(r)
        if adapt and k >= delay:
            f_hat, diag = AL.estimator_step(f_hat, r, M_inv, gamma=consts["gamma"], dead=consts["dead"], norm_r=consts["norm_r"], clip=cap,
                                            mask=mask, norm_channels=consts["norm_channels"], law=law, M=M, state=state)
            state = diag.get("estimator_state")
    return np.array(D), np.array(FH), np.array(RR)


def qnorm(E):
    return float(np.sqrt(DT * np.sum(np.sum(np.asarray(E) ** 2, 1))))


def cmd_fit(a):
    pairs, healthy, metas = [], [], []
    for run in a.dev:
        p, h, m = dev_series(run); pairs += p; healthy += h; metas.append(dict(run=str(run), sha256=sha(run), n_pairs=len(p), n_healthy_sequences=len(h)))
    G = fit_memory(pairs)
    b_h = np.mean(np.concatenate(healthy, 0), 0)
    # in-sample fit quality (development only)
    err = [np.sum(np.linalg.norm(e - predict_memory(G, d), axis=1)) for e, d in pairs]
    model = dict(kind="fya_forecast model", L=L, ridge=1e-6, G_memory=G.tolist(), b_h=b_h.tolist(),
                 healthy_residual_pool=[h.tolist() for h in healthy], pool_note="whole healthy residual sequences (H x 6) of the development runs",
                 development_runs=metas, in_sample_integrated_abs_error_m_step=dict(median=float(np.median(err)), mean=float(np.mean(err))),
                 units=dict(e="m", d="normalised action units", residual="normalised action units"))
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(model)); print("wrote", a.out, "sha256", sha(a.out), "pairs", len(pairs), "healthy sequences", len(healthy))


def load_bundle(p):
    B = json.loads(pathlib.Path(p).read_text()); W = np.array(B["W"], float); M = np.array(B["M"], float)
    return B, W, M, np.linalg.pinv(M), AL.correction_mask([int(x) for x in B["corr_dims"]]), dict(gamma=B["gamma"], dead=B["dead"], norm_r=B["norm_r"], norm_channels=B["norm_channels"])


def cmd_predict(a):
    model = json.loads(a.model.read_text()); G = np.array(model["G_memory"]); b_h = np.array(model["b_h"]); pool = [np.array(x) for x in model["healthy_residual_pool"]]
    B, W, M, M_inv, mask, consts = load_bundle(a.bundle)
    scenarios = json.loads(a.scenarios.read_text()) if a.scenarios else DEFAULT_SCENARIOS
    log = json.loads(a.log.read_text()); rec = log["records"][0] if isinstance(log, dict) else log[0]
    lens = rec["ep_len"]; keys = rec["episode_keys"]; cmds_all = np.asarray(rec["raw_cmd"], float); starts = np.cumsum([0] + list(lens[:-1]))
    rng = np.random.default_rng(a.seed); H = a.horizon; cd = [int(x) for x in B["corr_dims"]]
    rows = []
    for ei, n in enumerate(lens):
        tid, init = int(keys[ei][0]), int(keys[ei][1])
        eligible = a.checkpoint + H <= n
        seg = cmds_all[starts[ei] + a.checkpoint:starts[ei] + a.checkpoint + H] if eligible else np.zeros((0, 7))
        nominal_sha = hashlib.sha256(np.ascontiguousarray(seg).tobytes()).hexdigest()
        # the same noise draws for every branch of a source (common random numbers)
        draws = [pool[i][:H] for i in rng.integers(0, len(pool), a.draws)]
        for name, sc, arm in branch_specs(scenarios, a.healthy_cap):
            if name == "healthy_duplicate":
                continue
            Es, Js, EE, RN = [], [], [], []
            f = np.asarray(sc["fault"], float) if sc else np.zeros(6)
            for nz in draws:
                D, FH, RR = simulate(sc, arm, W, M, M_inv, mask, consts, a.healthy_cap, H, nz, b_h)
                e = predict_memory(G, D); Es.append(e); Js.append(DT * float(np.sum(np.sum(e ** 2, 1))))
                EE.append(float(np.mean(np.abs(FH[:, cd] - f[cd])))); RN.append(float(np.mean(np.linalg.norm(RR, axis=1))))
            Ehat = np.mean(Es, 0)
            rows.append(dict(source=ei, task=tid, init=init, eligible=eligible, branch=name, scenario=(sc["id"] if sc else "healthy"), arm=arm,
                             Ehat_qnorm=qnorm(Ehat), Jhat=qnorm(Ehat) ** 2, Jhat_draw_mean=float(np.mean(Js)), Jhat_draw_sd=float(np.std(Js)),
                             endpoint_hat_m=float(np.linalg.norm(Ehat[-1])), ctrl_est_err=float(np.mean(EE)), ctrl_residual_norm=float(np.mean(RN)),
                             nominal_continuation_sha256=nominal_sha, model_sha256=sha(a.model), bundle_sha256=sha(a.bundle), draws=a.draws, seed=a.seed))
        print(f"source {ei} (task {tid}, init {init}) eligible={eligible}: forecast written for {len([r for r in rows if r['source']==ei])} branches", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # store the forecast trajectories too (needed for the calibration score), next to the csv
    traj_path = a.out.with_suffix(".trajectories.json")
    # recompute compactly: keep only Ehat per (source, branch)
    traj = {}
    rng = np.random.default_rng(a.seed)
    for ei, n in enumerate(lens):
        eligible = a.checkpoint + H <= n
        draws = [pool[i][:H] for i in rng.integers(0, len(pool), a.draws)]
        for name, sc, arm in branch_specs(scenarios, a.healthy_cap):
            if name == "healthy_duplicate":
                continue
            Es = [predict_memory(G, simulate(sc, arm, W, M, M_inv, mask, consts, a.healthy_cap, H, nz, b_h)[0]) for nz in draws]
            traj[f"{ei}|{name}"] = np.mean(Es, 0).tolist()
    traj_path.write_text(json.dumps(dict(horizon=H, checkpoint=a.checkpoint, trajectories=traj)))
    print("wrote", a.out, "sha256", sha(a.out), "and", traj_path, "sha256", sha(traj_path))


def measured_E(run):
    d = read_json(run); out = {}
    for src in d["episodes"]:
        if src.get("status") != "ok":
            continue
        h = src["branches"]["healthy_off"]
        for name, b in src["branches"].items():
            out[f"{src['episode']}|{name}"] = np.array([np.array(s["ee_pos"]) - np.array(hh["ee_pos"]) for s, hh in zip(b, h)])
    return out, d


def cmd_calibrate(a):
    traj = json.loads(a.predictions.with_suffix(".trajectories.json").read_text())["trajectories"]
    E, d = measured_E(a.run); per_source = {}
    for key, Em in E.items():
        ei, name = key.split("|")
        if name == "healthy_duplicate" or key not in traj:
            continue
        s = qnorm(Em - np.array(traj[key])); per_source.setdefault(int(ei), {})[name] = s
    scores = {ei: max(v.values()) for ei, v in per_source.items()}
    vals = np.array(list(scores.values()))
    pred_norm = {f"{ei}|{name}": qnorm(np.array(traj[f"{ei}|{name}"])) for ei in per_source for name in per_source[ei]}
    healthy_err = [per_source[ei][n] for ei in per_source for n in per_source[ei] if n.startswith("healthy")]
    aff_a = float(max(healthy_err)) if healthy_err else 0.0
    aff_rho = float(max(max(per_source[ei][n] - aff_a, 0.0) / max(pred_norm[f"{ei}|{n}"], 1e-12) for ei in per_source for n in per_source[ei] if not n.startswith("healthy")))
    cal = dict(rule_max="epsilon = max over qualification sources of the per-source maximum ||E - Ehat||_Q across compared branches",
               rule_affine="epsilon_i = a + rho * ||Ehat_i||_Q with a = max healthy-branch error and rho = max over non-healthy branches of (error - a)_+ / ||Ehat_i||_Q",
               epsilon=float(vals.max()), epsilon_p90_secondary=float(np.percentile(vals, 90)), affine_a=aff_a, affine_rho=aff_rho, n_sources=len(vals), per_source_max=scores,
               per_source_branch=per_source, pred_norm=pred_norm, predictions_sha256=sha(a.predictions), run_sha256=sha(a.run), units="m sqrt(s)")
    a.out.write_text(json.dumps(cal, indent=1)); print(json.dumps({k: v for k, v in cal.items() if k not in ("per_source_branch",)}, indent=1))


def cmd_evaluate(a):
    cal = json.loads(a.calibration.read_text()); eps = float(cal["epsilon"]) if a.epsilon is None else a.epsilon
    aff_a, aff_rho = float(cal.get("affine_a", 0.0)), float(cal.get("affine_rho", 0.0))
    width = (lambda pn: aff_a + aff_rho * pn) if a.width == "affine" else (lambda pn: eps)
    pred = {(int(r["source"]), r["branch"]): r for r in csv.DictReader(open(a.predictions))}
    traj = json.loads(a.predictions.with_suffix(".trajectories.json").read_text())["trajectories"]
    E, d = measured_E(a.run); meta = d["meta"]; scen = [s["id"] for s in meta["scenarios"]]
    tol = a.zero_tol
    rows = []; cover_ind = []; joint = {}
    for src in d["episodes"]:
        if src.get("status") != "ok":
            continue
        ei = src["episode"]; ok_all = True
        for name in meta["branches"]:
            if name == "healthy_duplicate":
                continue
            inside = qnorm(E[f"{ei}|{name}"] - np.array(traj[f"{ei}|{name}"])) <= width(qnorm(np.array(traj[f"{ei}|{name}"])))
            cover_ind.append(inside); ok_all &= inside
        joint[ei] = ok_all
        for sc in scen:
            po = pred[(ei, f"{sc}__off")]; Jo = DT * float(np.sum(np.sum(E[f"{ei}|{sc}__off"] ** 2, 1)))
            no = float(po["Ehat_qnorm"]); eo = width(no); Lo, Uo = max(0.0, no - eo) ** 2, (no + eo) ** 2
            for law in ("nt", "innovation", "reference"):
                pa = pred[(ei, f"{sc}__{law}")]; Ja = DT * float(np.sum(np.sum(E[f"{ei}|{sc}__{law}"] ** 2, 1)))
                na = float(pa["Ehat_qnorm"]); ea = width(na); La, Ua = max(0.0, na - ea) ** 2, (na + ea) ** 2
                lo, hi = Lo - Ua, Uo - La; label = "benefit" if lo > 0 else ("harm" if hi < 0 else "inconclusive")
                B = Jo - Ja; actual = "benefit" if B > tol else ("harm" if B < -tol else "within_tol")
                ctrl_label = "benefit" if float(pa["ctrl_est_err"]) < float(po["ctrl_est_err"]) - 1e-12 else "no_benefit"
                rows.append(dict(source=ei, task=src["task"], init=src["init"], scenario=sc, law=law, width_rule=a.width, eps_off=eo, eps_A=ea,
                                 Jhat_off=float(po["Jhat"]), Jhat_A=float(pa["Jhat"]), pred_lower=lo, pred_upper=hi, pred_label=label,
                                 J_off=Jo, J_A=Ja, B=B, actual=actual, forecast_error_B=(float(po["Jhat"]) - float(pa["Jhat"])) - B,
                                 both_covered=bool(qnorm(E[f"{ei}|{sc}__off"] - np.array(traj[f"{ei}|{sc}__off"])) <= eo and qnorm(E[f"{ei}|{sc}__{law}"] - np.array(traj[f"{ei}|{sc}__{law}"])) <= ea),
                                 false_help=bool(label == "benefit" and B <= tol), harm_error=bool(label == "harm" and B >= -tol),
                                 ctrl_est_err_label=ctrl_label, ctrl_est_err_correct=bool((ctrl_label == "benefit") == (B > tol))))
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "evaluation_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    decisive_sources = sorted({r["source"] for r in rows if r["pred_label"] != "inconclusive" and r["law"] in ("nt", "innovation")})
    # registered point-forecast criteria (prereg section 9): sign agreement on cells with |B| > tol, per law, and median |error|/|B|
    sign_agree, rel_err = {}, {}
    for law in ("nt", "innovation", "reference"):
        rr = [r for r in rows if r["law"] == law and abs(r["B"]) > tol]
        sign_agree[law] = float(np.mean([((r["Jhat_off"] - r["Jhat_A"]) > 0) == (r["B"] > 0) for r in rr])) if rr else None
        rel_err[law] = float(np.median([abs(r["forecast_error_B"]) / abs(r["B"]) for r in rr])) if rr else None
    summary = dict(width_rule=a.width, epsilon_max_rule=eps, affine_a=aff_a, affine_rho=aff_rho, zero_tolerance=tol,
                   point_forecast_sign_agreement=sign_agree, point_forecast_median_rel_error=rel_err,
                   point_forecast_target_ge_0_80=dict((law, bool(sign_agree[law] is not None and sign_agree[law] >= 0.80)) for law in ("nt", "innovation")), n_sources=len(joint), joint_coverage=int(sum(joint.values())), individual_coverage=f"{int(sum(cover_ind))}/{len(cover_ind)}",
                   decisive_sources_nt_or_innovation=decisive_sources, n_decisive=len(decisive_sources),
                   labels={f"{sc}/{law}": {lab: int(sum(1 for r in rows if r['scenario'] == sc and r['law'] == law and r['pred_label'] == lab)) for lab in ("benefit", "harm", "inconclusive")} for sc in scen for law in ("nt", "innovation", "reference")},
                   false_help=[dict(source=r["source"], task=r["task"], init=r["init"], scenario=r["scenario"], law=r["law"], B=r["B"], pred_lower=r["pred_lower"]) for r in rows if r["false_help"]],
                   predicted_harm_errors=[dict(source=r["source"], scenario=r["scenario"], law=r["law"], B=r["B"]) for r in rows if r["harm_error"]],
                   interval_width_median=float(np.median([r["pred_upper"] - r["pred_lower"] for r in rows])),
                   forecast_error_B=dict(median=float(np.median([r["forecast_error_B"] for r in rows])), mean_abs=float(np.mean([abs(r["forecast_error_B"]) for r in rows]))),
                   control_est_err_accuracy={law: float(np.mean([r["ctrl_est_err_correct"] for r in rows if r["law"] == law])) for law in ("nt", "innovation")},
                   targets=dict(joint_coverage_ge_18_of_20=bool(sum(joint.values()) >= 18), decisive_ge_10=bool(len(decisive_sources) >= 10), false_help_le_1=bool(len([r for r in rows if r["false_help"]]) <= 1)),
                   predictions_sha256=sha(a.predictions), calibration_sha256=sha(a.calibration), run_sha256=sha(a.run))
    (a.out / "evaluation_summary.json").write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit"); p.add_argument("--dev", nargs="+", required=True); p.add_argument("--out", type=pathlib.Path, required=True); p.set_defaults(fn=cmd_fit)
    p = sub.add_parser("predict"); p.add_argument("--model", type=pathlib.Path, required=True); p.add_argument("--bundle", type=pathlib.Path, required=True)
    p.add_argument("--log", type=pathlib.Path, required=True); p.add_argument("--scenarios", type=pathlib.Path, default=None); p.add_argument("--healthy-cap", type=float, default=0.05)
    p.add_argument("--checkpoint", type=int, default=30); p.add_argument("--horizon", type=int, default=50); p.add_argument("--draws", type=int, default=64); p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", type=pathlib.Path, required=True); p.set_defaults(fn=cmd_predict)
    p = sub.add_parser("calibrate"); p.add_argument("--predictions", type=pathlib.Path, required=True); p.add_argument("--run", type=pathlib.Path, required=True); p.add_argument("--out", type=pathlib.Path, required=True); p.set_defaults(fn=cmd_calibrate)
    p = sub.add_parser("evaluate"); p.add_argument("--predictions", type=pathlib.Path, required=True); p.add_argument("--calibration", type=pathlib.Path, required=True); p.add_argument("--run", type=pathlib.Path, required=True)
    p.add_argument("--epsilon", type=float, default=None); p.add_argument("--zero-tol", type=float, default=1e-9); p.add_argument("--width", choices=["max", "affine"], default="affine")
    p.add_argument("--out", type=pathlib.Path, required=True); p.set_defaults(fn=cmd_evaluate)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
