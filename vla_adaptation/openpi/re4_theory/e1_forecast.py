#!/usr/bin/env python3
"""Prospective coupled prediction for E1 v2 (PREREG_E1_PHYSICAL_CONTINUATIONS.md, supersession note):
forecast the adaptation transient and the physical deviation of each branch BEFORE the locked test
branches are opened, using only frozen quantities -- the injected fault, the deployed observer
constants, the frozen predictor U (its tap sums G_fit) and sensitivity M, and the physical-response
model fitted on the fit sources (memory G, geometric lambda/B, neutral B). No realised test
correction, estimate or residual enters the forecast.

Residual model (record 50, the correction-direction feedback): with the world responding at M and
the predictor at G_fit, r_k = M f - (M - G_fit) c_k + 0 (nominal-command term dropped: its mean is
taken as zero), c_k = mask * f_hat_k. The observer recursion is the deployed adaptive_law.estimator_step.
The remaining disturbance d_k = f + c_k (c_k negative) drives the physical model.
Branches: faulted_off (c = 0), legacy_from_zero, innovation_from_zero, hold_supplied (c = -mask*hold),
exact_cancellation (c = -mask*f -> d = 0 on the mask).
Output: per branch the forecast d_k, f_hat_k, and e_k (ee deviation, m) for H steps under each physical
model, plus the frozen decision quantities (integrated |e| per branch, the ordering).
"""
import argparse, json, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "re4_theory"))
import adaptive_law as AL
from e1_memory_model import predict_memory, predict_geometric, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=pathlib.Path, required=True, help="memory_model_score.json of the fit/qualification pass (frozen_model block)")
    ap.add_argument("--receipt", type=pathlib.Path, default=HERE.parent / "results/iclr_unified_v1/E2_core/RECEIPT.json", help="frozen U and M")
    ap.add_argument("--hold", required=True, help="the supplied hold estimate (6 comma values)"); ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--fault-vec", default="0,0,0,0,0.05,0"); ap.add_argument("--corr-dims", default="3,4,5")
    ap.add_argument("--gamma", type=float, default=0.08); ap.add_argument("--dead", type=float, default=0.008); ap.add_argument("--norm-r", type=float, default=0.15); ap.add_argument("--clip", type=float, default=0.15)
    ap.add_argument("--out", type=pathlib.Path, required=True); a = ap.parse_args()
    fm = json.loads(a.model.read_text())["frozen_model"]; rc = json.loads(a.receipt.read_text())
    U = np.array(rc["predictor"]["U"]); M = np.array(rc["sensitivity"]["M"]); G_fit = np.diag(U[:, :AL.K_FIR + 1].sum(1)); M_inv = np.linalg.pinv(M)
    f = np.array([float(x) for x in a.fault_vec.split(",")]); hold = np.array([float(x) for x in a.hold.split(",")]); cd = [int(x) for x in a.corr_dims.split(",")]
    mask = AL.correction_mask(cd); H = a.horizon
    G = np.array(fm["G_memory"]); lam = fm["lam_geometric"]; Bg = np.array(fm["B_geometric"]); Bn = np.array(fm["B_neutral"])
    out = dict(frozen_inputs=dict(model=str(a.model), receipt=str(a.receipt), G_fit_diag=np.diag(G_fit).tolist(), M_diag=np.diag(M).tolist(), fault=f.tolist(), hold=hold.tolist(),
                                  residual_model="r_k = M f - (M - G_fit) c_k; nominal-command term taken as zero-mean", horizon=H), branches={})
    for name in ("faulted_off", "legacy_from_zero", "innovation_from_zero", "hold_supplied", "exact_cancellation"):
        f_hat = np.zeros(6); state = None; D, FH = [], []
        for k in range(H):
            if name == "faulted_off":
                c = np.zeros(6)
            elif name == "hold_supplied":
                c = -hold * mask
            elif name == "exact_cancellation":
                c = -f * mask
            else:
                c = -f_hat * mask
            d = f + c; D.append(d.copy()); FH.append(f_hat.copy())
            if name in ("legacy_from_zero", "innovation_from_zero"):
                # r = G_true(u_c + f) - G_fit(u_c) with u_c = a - f_hat, i.e. r = M f + (M - G_fit) c for the applied correction c
                # (c = -f_hat on the mask). The first frozen forecast (forecast_v2.json, 14:47) carried the opposite sign on the
                # feedback term, which sends the legacy estimate to 2.7 f instead of 0.61 f; corrected 14:52 before any locked
                # trajectory was read (pass 2 was running, unread) -- both forecasts are kept and scored.
                r = M @ f + (M - G_fit) @ c
                f_hat, diag = AL.estimator_step(f_hat, r, M_inv, gamma=a.gamma, dead=a.dead, norm_r=a.norm_r, clip=a.clip, mask=mask, norm_channels="all",
                                                law=("innov" if name.startswith("innovation") else "legacy"), M=M, state=state); state = diag.get("estimator_state")
        D = np.array(D)
        e_mem = predict_memory(G, D); e_geo = predict_geometric(lam, Bg, D); e_neu = predict_geometric(1.0, Bn, D)
        out["branches"][name] = dict(remaining_disturbance=D.tolist(), f_hat=np.array(FH).tolist(),
                                     forecast=dict(memory=dict(e=e_mem.tolist(), integrated_abs_m_step=float(np.sum(np.linalg.norm(e_mem, axis=1))), endpoint_m=float(np.linalg.norm(e_mem[-1]))),
                                                   geometric=dict(integrated_abs_m_step=float(np.sum(np.linalg.norm(e_geo, axis=1))), endpoint_m=float(np.linalg.norm(e_geo[-1]))),
                                                   neutral=dict(integrated_abs_m_step=float(np.sum(np.linalg.norm(e_neu, axis=1))), endpoint_m=float(np.linalg.norm(e_neu[-1])))),
                                     final_f_hat_ry=float(FH[-1][4]), final_remaining_ry=float(D[-1][4]))
    order = sorted(out["branches"], key=lambda n: -out["branches"][n]["forecast"]["memory"]["integrated_abs_m_step"])
    out["decision"] = dict(registered="innovation forecast integrated ee deviation within 30 % of the measured median locked source; ordering faulted > innovation ~ legacy > hold > exact observed",
                           forecast_ordering_memory_model=order, forecast_integrated=[(n, round(out["branches"][n]["forecast"]["memory"]["integrated_abs_m_step"], 4)) for n in order])
    a.out.write_text(json.dumps(out, indent=1)); print(json.dumps(out["decision"], indent=1)); print("innovation forecast: final f_hat_ry %.4f, remaining %.4f" % (out["branches"]["innovation_from_zero"]["final_f_hat_ry"], out["branches"]["innovation_from_zero"]["final_remaining_ry"]))


if __name__ == "__main__":
    main()
