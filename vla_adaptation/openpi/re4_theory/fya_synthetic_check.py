#!/usr/bin/env python3
"""Stage 0 checks for the FrozenYet Adaptive recovery campaign (EXPERIMENT_PLAN.md section 2, items 4-5):

  A. delay / cap / reference timing of fya_continuations.run_branch on a linear fake plant:
       - for k < tau the adaptive estimate is exactly zero, no update is applied, corrections are zero;
       - at k = tau the correction is still zero (pre-update estimate) and the first update is applied;
       - at k = tau + 1 the correction is nonzero;
       - the reference branch corrects from k = tau with clip(f, -C, C) on the mask and never before;
       - the cap bounds every applied correction (|c_i| <= C) and the estimate;
       - sent = intended + correction, simulator input = sent + fault, for every step and branch;
       - histories advance during the delay (the predictor's residual at k = tau uses the last K+1 sent commands).
  B. the scalar memory/authority/delay floor  g [F S_N(lambda) - C S_{N-tau}(lambda)]_+  against an exact
     finite-horizon minimax optimisation (LP over the causal adapter: common corrections before tau, sign-specific
     after), on a grid of (lambda, tau, C/F, N). Checks the conditional formula only, not a robot bound.

Exit status 0 iff every assertion holds. Output: a JSON receipt (--out) with the checked grid and maxima.
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import adaptive_law as AL
from fya_continuations import run_branch, branch_specs, DEFAULT_SCENARIOS


def fake_plant(G_true, K):
    """Linear FIR world: y_k = sum_j G_true[j] * u_{k-j} (elementwise per channel) + small deterministic wobble."""
    hist = [np.zeros(6) for _ in range(K + 1)]; state = dict(k=0)

    def step(cmd7):
        u = np.asarray(cmd7, float)[:6]
        hist.insert(0, u); del hist[K + 1:]
        y = sum(G_true[j] * hist[j] for j in range(len(hist)))
        y = y + 1e-4 * np.sin(0.3 * state["k"] + np.arange(6)); state["k"] += 1
        return y, False

    def phys():
        return dict(q=[0.0] * 7, v=[0.0] * 7, ee_pos=[0.0, 0.0, 0.0], ee_mat=[1, 0, 0, 0, 1, 0, 0, 0, 1], goal_pos=[0, 0, 0],
                    goal_mat=[1, 0, 0, 0, 1, 0, 0, 0, 1], objects={}, contact=False, sim_time=0.0)
    return step, phys


def check_branches(out):
    K = AL.K_FIR; rng = np.random.default_rng(0)
    # a predictor W whose taps sum to a value different from M on r_y, as in the deployed configuration
    G_true = [np.array([0.30, 0.27, 0.13, 0.25, 0.28, 0.24]) * w for w in (0.5, 0.25, 0.12, 0.06, 0.04, 0.02, 0.01)]
    W = np.zeros((6, K + 2))
    for j in range(K + 1):
        W[:, j] = G_true[j] * np.array([1, 1, 1, 1, 0.55, 1])       # r_y under-modelled on purpose
    M = np.diag([0.2969, 0.2716, 0.1265, 0.2526, 0.2759, 0.2436]); M_inv = np.linalg.pinv(M)
    mask = AL.correction_mask([3, 4, 5]); consts = dict(gamma=0.08, dead=0.008, norm_r=0.15, norm_channels="all")
    seg = rng.normal(scale=0.2, size=(50, 7)); hist0 = [rng.normal(scale=0.2, size=6) for _ in range(K + 1)]
    receipts = []
    for name, sc, arm in branch_specs(DEFAULT_SCENARIOS, 0.05):
        step, phys = fake_plant(G_true, K)
        for _ in range(K + 1):                                       # bring the fake plant's history to hist0
            pass
        recs = run_branch(name, sc, arm, seg, hist0, step_fn=step, phys_fn=phys, W=W, M=M, M_inv=M_inv, mask=mask, K=K, consts=consts, healthy_cap=0.05)
        f = np.asarray(sc["fault"], float) if sc else np.zeros(6); cap = sc["cap"] if sc else 0.05; tau = sc.get("delay", 0) if sc else 0
        for r in recs:
            k = r["k"]; c = np.array(r["requested_correction"]); fb = np.array(r["fhat_before"]); fa = np.array(r["fhat_after"])
            assert np.allclose(np.array(r["sent"]), np.array(r["intended"]) + c), (name, k, "sent != intended + correction")
            assert np.allclose(np.array(r["simulator_input"]), np.array(r["sent"]) + f), (name, k, "simulator input != sent + fault")
            assert np.allclose(np.array(r["remaining_disturbance"]), f + c), (name, k)
            assert np.all(np.abs(c) <= cap + 1e-12), (name, k, "cap violated", c)
            assert np.all(np.abs(fa) <= cap + 1e-12), (name, k, "estimate outside the box", fa)
            if arm in ("nt", "innovation"):
                if k < tau:
                    assert not r["update_enabled"] and not r["update_applied"] and np.all(fb == 0) and np.all(fa == 0) and np.all(c == 0), (name, k, "delay violated")
                    assert r["delay_active"]
                if k == tau:
                    assert r["update_enabled"] and np.all(fb == 0) and np.all(c == 0), (name, k, "first enabled step must still act on a zero estimate")
                if k == tau + 1 and sc is not None:
                    assert np.any(c != 0), (name, k, "correction should be nonzero one step after the first update")
                assert np.allclose(c, -fb * mask), (name, k, "correction must come from the pre-update estimate")
            elif arm == "reference":
                expect = -np.clip(f, -cap, cap) * mask if k >= tau else np.zeros(6)
                assert np.allclose(c, expect), (name, k, "reference correction", c, expect)
                assert np.all(fa == 0) and not r["update_applied"]
            else:
                assert np.all(c == 0) and np.all(fa == 0)
        # histories advance during the delay: the residual at k = tau must differ from the residual that a
        # zero-history predictor would give (i.e. the predictor used the last K+1 sent commands)
        if arm in ("nt", "innovation") and tau > 0:
            r_tau = np.array(recs[tau]["residual"]); y_tau = np.array(recs[tau]["measured"])
            Hh = np.array([np.array(recs[tau - j]["sent"]) for j in range(0, min(tau, K) + 1)] + hist0)[:K + 1]
            pred = np.array([W[i, :K + 1] @ Hh[:, i] + W[i, -1] for i in range(6)])
            assert np.allclose(r_tau, y_tau - pred), (name, "history did not advance during the delay")
        receipts.append(dict(branch=name, steps=len(recs), first_update_k=next((r["k"] for r in recs if r["update_applied"]), None),
                             first_nonzero_correction_k=next((r["k"] for r in recs if np.any(np.array(r["requested_correction"]) != 0)), None),
                             max_abs_correction=float(max(np.abs(np.array(r["requested_correction"])).max() for r in recs))))
    out["branch_timing"] = receipts
    return True


def S(m, lam):
    return float(sum(lam ** j for j in range(m)))


def floor_formula(g, F, C, lam, N, tau):
    return g * max(F * S(N, lam) - C * S(N - tau, lam), 0.0)


def floor_exact(g, F, C, lam, N, tau):
    """Exact minimax over causal adapters: corrections c_k common to both signs for k < tau, sign-specific after.
    p_N = g sum_k lam^(N-1-k) (f - c_k). Minimise max_{f = +-F} |p_N| by LP (variables: common c, c+, c-, t)."""
    from scipy.optimize import linprog
    w = np.array([lam ** (N - 1 - k) for k in range(N)]); n_pre = tau; n_post = N - tau
    # variables: c_pre (n_pre), c_plus (n_post), c_minus (n_post), t
    nv = n_pre + 2 * n_post + 1; cobj = np.zeros(nv); cobj[-1] = 1.0
    A = []; b = []
    for sign in (+1.0, -1.0):
        cp = np.zeros(nv); cp[:n_pre] = -w[:n_pre]
        if sign > 0:
            cp[n_pre:n_pre + n_post] = -w[n_pre:]
        else:
            cp[n_pre + n_post:n_pre + 2 * n_post] = -w[n_pre:]
        const = sign * F * w.sum()
        # p/g = const + cp . x  ;  |p/g| <= t  -> const + cp.x - t <= 0 and -(const + cp.x) - t <= 0
        row = cp.copy(); row[-1] = -1.0; A.append(row); b.append(-const)
        row = -cp.copy(); row[-1] = -1.0; A.append(row); b.append(const)
    bounds = [(-C, C)] * (nv - 1) + [(0, None)]
    res = linprog(cobj, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method="highs")
    assert res.success, res.message
    return g * float(res.fun)


def check_floor(out):
    grid = []; worst = 0.0
    for lam in (0.0, 0.5, 0.663, 0.9, 1.0):
        for N in (10, 50):
            for tau in sorted({0, 1, min(10, N), N // 2, N}):
                for ratio in (0.0, 0.5, 1.0, 1.5):
                    F, C, g = 0.05, 0.05 * ratio, 0.3
                    a = floor_formula(g, F, C, lam, N, tau); b = floor_exact(g, F, C, lam, N, tau)
                    worst = max(worst, abs(a - b)); grid.append(dict(lam=lam, N=N, tau=tau, C_over_F=ratio, formula=a, exact_lp=b))
                    assert abs(a - b) <= 1e-9 + 1e-7 * max(abs(a), 1.0), (lam, N, tau, ratio, a, b)
    out["floor_grid"] = grid; out["floor_max_abs_discrepancy"] = worst
    # the two special forms quoted in the research note
    lam, N, tau, F, g = 1.0, 50, 10, 0.05, 0.3
    assert abs(floor_formula(g, F, F, lam, N, tau) - g * F * tau) < 1e-12
    lam = 0.663
    assert abs(floor_formula(g, F, F, lam, N, tau) - g * F * lam ** (N - tau) * S(tau, lam)) < 1e-12
    return True


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=pathlib.Path, default=None); a = ap.parse_args()
    out = dict(check="fya_synthetic_check", status="pending")
    check_branches(out); check_floor(out); out["status"] = "all assertions passed"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "floor_grid"}, indent=1))


if __name__ == "__main__":
    main()
