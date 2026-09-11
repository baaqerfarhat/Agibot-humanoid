"""G.3 predictor diagnostics (CPU only, stored logs only, numpy only).

For each healthy log the published joint-space cells were fitted on, compare the shipped
per-joint FIR position predictor against persistence (y_t = y_{t-1}) on
  (a) absolute position R^2 (the number the paper reports),
  (b) persistence absolute R^2,
  (c) incremental-motion R^2 on dy_t = y_t - y_{t-1},
  (d) per joint and pooled, in-sample (shipped W, fitted on all episodes) and
      leave-one-episode-out (fit on the other episodes, evaluate on the held-out one).

Log convention (aloha_adapt.episode / gr1_adapt.episode): log["u"][t] is the command applied
at step t, log["q"][t] the joint position measured AFTER that step. fit_plant regresses
q[t] on (u[t], u[t-1], ..., u[t-6], 1) over rows t >= K_FIR of every episode. Persistence
predicts q[t] by q[t-1] (the position measured before the step), on the same rows.
"""
import ast, hashlib, json, pathlib, sys, tempfile
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = pathlib.Path(__file__).resolve().parent
SCRATCH = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(tempfile.mkdtemp())
sys.path.insert(0, str(ROOT / "openpi"))
import aloha_adapt  # module level is stdlib + numpy + constants only; no simulator import

def gr1_fit_plant():
    """gr1_adapt.py imports gymnasium/robocasa at module level, so it is not imported.
    Its fit_plant is extracted verbatim from the source with ast and executed with the
    module's own constants (NJ=29, K_FIR=6)."""
    src = (ROOT / "openpi/gr1_adapt.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fit_plant")
    consts = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id in ("NJ", "K_FIR"):
            consts[n.targets[0].id] = ast.literal_eval(n.value)
    ns = dict(json=json, pathlib=pathlib, np=np, **consts)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "gr1_adapt.py", "exec"), ns)
    return ns["fit_plant"], consts, ast.get_source_segment(src, fn)

GR1_FIT, GR1_CONSTS, GR1_SRC = gr1_fit_plant()
K = aloha_adapt.K_FIR
assert GR1_CONSTS["K_FIR"] == K

DATASETS = [
    dict(name="aloha_transfer_cube", robot="ALOHA", fit=aloha_adapt.fit_plant, nj=aloha_adapt.NJ,
         log="results/aloha/healthy_log.json", joints=list(range(0, 6)), hz=50,
         used_by="every results/aloha/off*.json and healthy*.json cell (args.log)", role="primary"),
    dict(name="gr1_plate_to_plate", robot="GR1", fit=GR1_FIT, nj=GR1_CONSTS["NJ"],
         log="results/gr1/screen_PosttrainPnPNovelFromPlateToPlateSplitA.json", joints=list(range(7, 14)), hz=20,
         used_by="results/gr1/p2p_*.json cells (args.log)", role="primary"),
    dict(name="gr1_tray_to_plate", robot="GR1", fit=GR1_FIT, nj=GR1_CONSTS["NJ"],
         log="results/gr1/t2p_healthy_log.json", joints=list(range(7, 14)), hz=20,
         used_by="results/gr1/t2p_*.json cells (args.log)", role="primary"),
    dict(name="gr1_can_to_drawer_left_arm", robot="GR1", fit=GR1_FIT, nj=GR1_CONSTS["NJ"],
         log="results/gr1/healthy_log.json", joints=list(range(0, 7)), hz=20,
         used_by="results/gr1/left015_*.json cells (record sec 32.1-32.5; not a paper table cell)",
         role="supplementary"),
]

def rows(ep_list, W, j):
    """Target, FIR prediction, previous position, and the position two steps back for joint j
    over the fit_plant rows t >= K of each episode."""
    Y, P, Y1, Y2 = [], [], [], []
    for ep in ep_list:
        u = np.asarray(ep["u"], float)[:, j]; q = np.asarray(ep["q"], float)[:, j]
        for t in range(K, len(u)):
            x = np.r_[u[t - K:t + 1][::-1], 1.0]
            Y.append(q[t]); P.append(x @ W[j]); Y1.append(q[t - 1]); Y2.append(q[t - 2])
    return tuple(np.asarray(a) for a in (Y, P, Y1, Y2))

def r2(y, yhat):
    st = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - yhat) ** 2).sum() / max(st, 1e-12))

def stats(Y, P, Y1, Y2):
    dy = Y - Y1
    dfir = P - Y1
    dcv = Y1 - Y2                       # constant-velocity extrapolation (no command used)
    sse_fir = float(((Y - P) ** 2).sum())
    sse_per = float((dy ** 2).sum())
    return dict(
        n=int(len(Y)),
        r2_abs_fir=r2(Y, P),
        r2_abs_persistence=r2(Y, Y1),
        r2_inc_fir=r2(dy, dfir),
        r2_inc_persistence=r2(dy, np.zeros_like(dy)),
        r2_inc_const_velocity=r2(dy, dcv),
        skill_vs_persistence=float(1 - sse_fir / max(sse_per, 1e-18)),
        rms_abs_err_fir=float(np.sqrt(sse_fir / len(Y))),
        rms_abs_err_persistence=float(np.sqrt(sse_per / len(Y))),
        rms_dy=float(np.sqrt((dy ** 2).mean())),
        std_y=float(Y.std()),
        corr_inc_fir=float(np.corrcoef(dy, dfir)[0, 1]) if dy.std() > 0 and dfir.std() > 0 else None,
        # pooled accumulators
        _sse=dict(abs_fir=sse_fir, abs_per=sse_per, inc_fir=float(((dy - dfir) ** 2).sum()),
                  inc_per=sse_per, inc_cv=float(((dy - dcv) ** 2).sum()),
                  sst_abs=float(((Y - Y.mean()) ** 2).sum()), sst_inc=float(((dy - dy.mean()) ** 2).sum())),
    )

def pooled(per_joint):
    s = {k: sum(pj["_sse"][k] for pj in per_joint.values()) for k in next(iter(per_joint.values()))["_sse"]}
    return dict(
        definition="1 - sum_j SSE_j / sum_j SST_j, each joint centred on its own mean",
        r2_abs_fir=1 - s["abs_fir"] / s["sst_abs"], r2_abs_persistence=1 - s["abs_per"] / s["sst_abs"],
        r2_inc_fir=1 - s["inc_fir"] / s["sst_inc"], r2_inc_persistence=1 - s["inc_per"] / s["sst_inc"],
        r2_inc_const_velocity=1 - s["inc_cv"] / s["sst_inc"],
        skill_vs_persistence=1 - s["abs_fir"] / s["abs_per"],
        n=sum(pj["n"] for pj in per_joint.values()))

def strip(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}

def run(ds):
    path = ROOT / ds["log"]
    raw = path.read_bytes(); eps = json.loads(raw)
    W, r2_ship = ds["fit"](path)
    arm14 = [float(x) for x in r2_ship[:14]]
    out = dict(robot=ds["robot"], role=ds["role"], log=ds["log"], md5=hashlib.md5(raw).hexdigest(),
               used_by=ds["used_by"], control_hz=ds["hz"], joints=ds["joints"],
               n_episodes=len(eps), episode_lengths=[len(e["u"]) for e in eps],
               episode_success=[bool(e.get("success")) for e in eps],
               shipped_fit_plant_r2_all_joints=[float(x) for x in r2_ship],
               shipped_r2_corrected_joints_range=[float(min(r2_ship[ds["joints"]])), float(max(r2_ship[ds["joints"]]))],
               shipped_r2_first14_range=[min(arm14), max(arm14)],
               fir_dc_gain_corrected_joints={f"j{j}": float(W[j, :K + 1].sum()) for j in ds["joints"]})
    # (a)-(c) in-sample, shipped W
    pj = {}
    for j in ds["joints"]:
        s = stats(*rows(eps, W, j))
        assert abs(s["r2_abs_fir"] - r2_ship[j]) < 1e-9, (ds["name"], j, s["r2_abs_fir"], r2_ship[j])
        pj[f"j{j}"] = s
    out["in_sample"] = dict(note="shipped W = fit_plant(full log); evaluated on the same rows it was fitted on",
                            per_joint={k: strip(v) for k, v in pj.items()}, pooled=pooled(pj))
    # (d) leave-one-episode-out with the runner's own fit_plant on the remaining episodes
    acc = {f"j{j}": [] for j in ds["joints"]}
    folds = []
    for e in range(len(eps)):
        train = [x for i, x in enumerate(eps) if i != e]
        tmp = SCRATCH / f"{ds['name']}_loeo_{e}.json"
        tmp.write_text(json.dumps(train))
        We, _ = ds["fit"](tmp)
        fold = {}
        for j in ds["joints"]:
            r = rows([eps[e]], We, j); acc[f"j{j}"].append(r)
            fold[f"j{j}"] = r
        fp = {k: stats(*v) for k, v in fold.items()}
        folds.append(dict(held_out_episode=e, length=len(eps[e]["u"]), success=bool(eps[e].get("success")),
                          pooled=pooled(fp)))
    pjh = {k: stats(*(np.concatenate([r[i] for r in v]) for i in range(4))) for k, v in acc.items()}
    out["held_out_loeo"] = dict(
        note=("leave-one-episode-out: for each episode, W is refit by the runner's fit_plant on the other "
              f"{len(eps)-1} episodes and evaluated on the held-out one; R^2 is computed on the "
              "concatenated held-out predictions (all episodes, each predicted by a model that never saw it)"),
        per_joint={k: strip(v) for k, v in pjh.items()}, pooled=pooled(pjh), per_fold=folds)
    return out

def main():
    res = dict(
        item="RE4 evidence plan G.3 -- predictor diagnostics for the joint-space robots",
        script="results/re4_evidence/G_forensics/predictor_diagnostics.py",
        fit_plant_provenance=dict(
            aloha="imported openpi/aloha_adapt.py: fit_plant (module level is stdlib + numpy only)",
            gr1=("openpi/gr1_adapt.py imports gymnasium/robocasa at module level, so fit_plant was "
                 "extracted verbatim with ast and executed with the module's NJ=29, K_FIR=6; its body is "
                 "token-identical to aloha_adapt.fit_plant apart from the docstring"),
            K_FIR=K),
        definitions=dict(
            rows="t >= K_FIR (=6) of every episode, the exact rows fit_plant uses",
            fir="q_hat[t] = sum_{k=0..6} h_k u[t-k] + c   (u = command applied at step t, q[t] measured after it)",
            persistence="q_hat[t] = q[t-1]",
            increment="dy[t] = q[t] - q[t-1]; FIR increment = q_hat_FIR[t] - q[t-1]; persistence increment = 0",
            const_velocity="extra command-free baseline: dy_hat[t] = q[t-1] - q[t-2]",
            r2="1 - SSE / sum (y - mean y)^2 on the quantity named",
            skill_vs_persistence=("1 - SSE_FIR / SSE_persistence on absolute position; because the FIR "
                                  "position error equals its increment error, this is also the uncentred "
                                  "incremental R^2"),
        ),
        datasets={ds["name"]: run(ds) for ds in DATASETS})
    (OUT / "predictor_diagnostics.json").write_text(json.dumps(res, indent=1))
    for n, d in res["datasets"].items():
        for split in ("in_sample", "held_out_loeo"):
            p = d[split]["pooled"]
            print(f"{n:28s} {split:14s} absFIR {p['r2_abs_fir']:.5f} absPers {p['r2_abs_persistence']:.5f} "
                  f"incFIR {p['r2_inc_fir']:.4f} incPers {p['r2_inc_persistence']:.4f} "
                  f"incCV {p['r2_inc_const_velocity']:.4f} skill {p['skill_vs_persistence']:.4f}")
        for k, v in d["held_out_loeo"]["per_joint"].items():
            print(f"     {k:4s} LOEO absFIR {v['r2_abs_fir']:.5f} absPers {v['r2_abs_persistence']:.5f} "
                  f"incFIR {v['r2_inc_fir']:.4f} incPers {v['r2_inc_persistence']:.4f} incCV {v['r2_inc_const_velocity']:.4f} "
                  f"rms_dy {v['rms_dy']:.5f} rmsFIR {v['rms_abs_err_fir']:.5f}  | in-sample incFIR "
                  f"{d['in_sample']['per_joint'][k]['r2_inc_fir']:.4f} absFIR {d['in_sample']['per_joint'][k]['r2_abs_fir']:.5f}")

if __name__ == "__main__":
    main()
