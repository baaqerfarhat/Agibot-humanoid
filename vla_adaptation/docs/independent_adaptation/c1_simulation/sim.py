"""Track 1 ground-truth simulation: contraction + state-dependent vs contraction + input-only
adaptive control, on a plant with the MEASURED properties of LIBERO's Panda under OSC_POSE:
  - increment dynamics first-order per axis, poles 0.83 / 0.93 (rx, ry, record 48 part 2.2)
  - position integrates increments: same-command contraction rate 1.00 (record 48 part 1)
  - a frozen replanning policy closes a position loop (P-control, 5-step chunks)
Arms differ in ONE factor at a time: regressor (constant vs state basis) and tracking (kappa).
All arms share the same Kalman-type estimator on the exact one-step innovation.
"""
import numpy as np

LAM = np.array([0.83, 0.93])          # measured increment poles (rx, ry)
G1 = 1.0 - LAM                        # one-step input gain (unit DC gain)
N, REPLAN, HOLD = 280, 5, 8
KPOL, UMAX, CMAX = 0.12, 0.5, 0.25

def features(kind, p, y, u=None, eps=0.05):
    u = np.zeros_like(p) if u is None else u
    """Regressor Phi(s): (E,2,m).  s = (measured position p, measured increment y)."""
    E = p.shape[0]; I = np.broadcast_to(np.eye(2), (E, 2, 2))
    if kind == "const":                      # input-only: constant action-equivalent offset
        return I.copy()
    blocks = [I]
    if kind in ("state", "fric"):            # input-direction feature (friction / deadband)
        blocks.append(np.einsum("ei,ij->eij", np.tanh(u / 0.05), np.eye(2)))
    if kind in ("state", "conf"):            # configuration features, normalized to O(1)
        blocks.append((p[:, 0] / 5.0)[:, None, None] * I); blocks.append((p[:, 1] / 5.0)[:, None, None] * I)
    return np.concatenate(blocks, axis=2)

def fault_fn(name, p, v, t, rng_phase, u=None):
    E = p.shape[0]
    if name == "healthy": return np.zeros((E, 2))
    if name == "offset":  return np.full((E, 2), 0.05)
    if name == "config":                     # rotates with configuration, like J(q)^+T tau
        phi = rng_phase + 0.26 * p[:, 0]
        return 0.08 * np.stack([np.cos(phi), np.sin(phi)], 1)
    if name == "friction":                   # Coulomb-like, opposes actual motion
        return -0.06 * np.tanh(u / 0.03)
    if name == "oscill":                     # time-varying, not a function of state
        return 0.06 * np.sin(2 * np.pi * t / 80.0)[None, None] * np.ones((E, 2))
    raise ValueError(name)

def run(fault, regressor="const", kappa=0.0, adapt=True, E=2000, seed=0, sigma=0.01,
        tasks=1, carry=False, q=1e-5, p0=0.02, washout=0.02, gain_err=(1.0, 1.0), ref_uses_model=True, return_time=False):
    rng = np.random.default_rng(seed)
    m = features(regressor, np.zeros((1, 2)), np.zeros((1, 2))).shape[2]
    prior = np.full(m, p0); prior[2:] = p0 / 10.0
    theta = np.zeros((E, m)); P = np.broadcast_to(np.diag(prior), (E, m, m)).copy()
    succ_all = []
    phase = rng.uniform(0, 2 * np.pi, E)                 # per-chain fault orientation
    for task in range(tasks):
        centre = rng.uniform(-5, 5, (E, 2))              # each task in its own workspace region
        goals = centre[:, None, :] + rng.uniform(-2.5, 2.5, (E, 3, 2))
        tol = rng.uniform(0.10, 0.45, E)                 # task-dependent tolerance
        p = centre + rng.normal(0, 0.3, (E, 2)); v = np.zeros((E, 2))
        y = v + rng.normal(0, sigma, (E, 2))
        pr, vr = p.copy(), v.copy()                      # healthy reference (same commands)
        if not carry or task == 0:
            theta[:] = 0; P[:] = np.diag(prior)
        else:
            P += np.diag(prior) * 0.25                    # keep the estimate, reopen covariance
        wp = np.zeros(E, int); hold = np.zeros(E, int); done = np.zeros(E, bool); tdone = np.full(E, N)
        u_nom = np.zeros((E, 2))
        for k in range(N):
            if k % REPLAN == 0:                          # frozen policy replans from observation
                g = goals[np.arange(E), np.minimum(wp, 2)]
                u_nom = np.clip(KPOL * (g - p), -UMAX, UMAX) + rng.normal(0, 0.01, (E, 2))
            Phi = features(regressor, p, y, u_nom)
            c = np.clip(np.einsum("eij,ej->ei", Phi, theta), -CMAX, CMAX) if adapt else 0.0
            u = u_nom - c
            d = fault_fn(fault, p, v, np.array(k), phase, u)
            v_new = LAM * v + G1 * (u + d)
            p = p + v_new; v = v_new
            y_new = v + rng.normal(0, sigma, (E, 2))
            vr = LAM * vr + G1 * (np.asarray(gain_err) if ref_uses_model else 1.0) * u_nom; pr = pr + vr + washout * (p - pr)   # reference from the ADAPTER'S model
            if adapt:
                # exact one-step innovation: zeta = G1*d + noise;  H = G1*Phi
                zeta = y_new - LAM * y - G1 * np.asarray(gain_err) * u
                H = G1[None, :, None] * Phi
                S = np.einsum("eim,emn,ejn->eij", H, P, H) + np.eye(2) * sigma**2 * (1 + LAM**2)
                K = np.einsum("emn,ejn,eji->emi", P, H, np.linalg.inv(S))
                theta = theta + np.einsum("emi,ei->em", K, zeta - np.einsum("eim,em->ei", H, theta))
                P = P - np.einsum("emi,eij,ejn->emn", K, H, P) + np.diag(prior / p0) * q
                if kappa:                                # composite tracking term: P Phi^T L e_p
                    ep = p - pr
                    theta = theta + kappa * np.einsum("emn,ein,ei->em", P, Phi, ep)
            y = y_new
            g = goals[np.arange(E), np.minimum(wp, 2)]
            inside = np.linalg.norm(p - g, axis=1) < tol
            hold = np.where(inside & ~done, hold + 1, 0)
            adv = hold >= HOLD
            wp = wp + adv; hold[adv] = 0; newly = (wp >= 3) & ~done; tdone[newly] = k; done |= wp >= 3
        succ_all.append(done.copy()); last_t = tdone
    return (np.array(succ_all), last_t) if return_time else np.array(succ_all)

if __name__ == "__main_v1__":
    import sys, itertools
    faults = ["healthy", "offset", "config", "friction", "oscill"]
    arms = [("frozen", dict(adapt=False)),
            ("const  k=0", dict(regressor="const", kappa=0.0)),
            ("const  k>0", dict(regressor="const", kappa=3.0)),
            ("state  k=0", dict(regressor="state", kappa=0.0)),
            ("state  k>0", dict(regressor="state", kappa=3.0))]
    print(f"{'fault':<10}" + "".join(f"{a:>12}" for a, _ in arms))
    for f in faults:
        row = []
        for name, kw in arms:
            s = run(f, seed=11, **kw)
            row.append(s.mean())
        print(f"{f:<10}" + "".join(f"{x:>12.3f}" for x in row))
