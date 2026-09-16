"""Reconstruct the descriptor loop's two gain operators from a stored sweep result.
The Riccati recursion is data-independent for a constant basis, so both operators follow from the
stored model and args alone. Prediction gain: residual -> coefficient. Tracking gain:
tracking error -> coefficient (h P+ B^-1 Phi^T L). Run from the repository root."""
import json, sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))  # repository root
from learned_adaptation import identify
from learned_adaptation.libero_adapter import RobotController
from learned_adaptation.loop import forgetting_discretisation
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/sweep/spatial_j6.json"))
a = d["args"]; ident = identify.IdentifiedModel.from_dict(d["model"])
rc = RobotController(ident, np.zeros((3, 6)), np.zeros((6, 6)), correction_limit=a["correction_limit"],
                     tracking_scale=a["tracking_scale"], process_variance=a["process_variance"],
                     initial_variance=a["initial_variance"], decay=a["decay"], noise_scale=a["noise_scale"])
L, phi = rc.loop, rc.basis; h = L.dt; P = L.covariance.copy()
for k in range(301):
    fz, qd = forgetting_discretisation(L.decay, L.process_weight, h)
    Pm = fz @ P @ fz.T + qd; psi = h * phi; S = psi @ Pm @ psi.T + h * L.measurement_intensity
    G = np.linalg.solve(S.T, (Pm @ psi.T).T).T; ikh = np.eye(P.shape[0]) - G @ psi
    P = ikh @ Pm @ ikh.T + G @ (h * L.measurement_intensity) @ G.T; P = (P + P.T) / 2
    T = h * P @ (np.linalg.solve(L.model.B, phi).T @ L.tracking_metric)
    if k in (0, 20, 300):
        print(f"step {k:>3}: ||prediction gain|| {np.linalg.norm(G, 2):.4g}  "
              f"||tracking gain|| {np.linalg.norm(T, 2):.4g}  ratio {np.linalg.norm(G, 2) / np.linalg.norm(T, 2):.3g}")
