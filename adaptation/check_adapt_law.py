#!/usr/bin/env python3
"""Audit the ACC adaptation law: input map, leak term, and gain tuning.

Runs in the mentor's MuJoCo reference env (fast) so the law itself can be
inspected in isolation, then reports which term dominates the weight change.

  python check_adapt_law.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent / "ACC_ADAPTATION_PACKAGE"
sys.path.insert(0, str(PKG))

from ace_adapt import AdaptConfig, ExportedPolicy, LayerAdapter  # noqa: E402


class InstrumentedAdapter(LayerAdapter):
    """LayerAdapter that logs the learning and leak contributions separately.

    `leak_to_w0` switches the sigma-modification from -gamma*W (decay toward
    zero, as shipped) to -gamma*(W - W0) (decay toward the frozen weights).
    """

    def __init__(self, *a, leak_to_w0: bool = False, **kw):
        self.leak_to_w0 = leak_to_w0
        super().__init__(*a, **kw)
        self.learn_norm = 0.0
        self.leak_norm = 0.0

    def update(self, joint_error, dt):
        from ace_adapt import _elu_jacobian

        self.step += 1
        if self.diverged or self.step <= self.cfg.engage_step or self._cache is None:
            return
        a, z = self._cache
        L, layer = self.pol.n_layers, self.cfg.layer

        d = self.delta_L(joint_error)
        for l in range(L - 1, layer, -1):
            d = _elu_jacobian(a[l - 1]) * (self.W[l].T @ d)

        learn = self.cfg.gain * np.outer(d, z[layer])
        if self.leak_to_w0:
            leak = -self.cfg.leak * (self.W[layer] - self.pol.W0[layer])
        else:
            leak = -self.cfg.leak * self.W[layer]

        self.learn_norm += dt * float(np.linalg.norm(learn))
        self.leak_norm += dt * float(np.linalg.norm(leak))

        self.W[layer] = self.W[layer] + dt * (learn + leak)

        if not np.isfinite(self.W[layer]).all() or self.weight_drift > self.cfg.max_weight_drift:
            self.diverged = True
            self.W[layer] = self.pol.W0[layer].copy()

    def leak_alignment(self) -> float:
        """cos angle between (W - W0) and -W0. ~1 means the drift is pure decay."""
        dW = (self.W[self.cfg.layer] - self.pol.W0[self.cfg.layer]).ravel()
        w0 = -self.pol.W0[self.cfg.layer].ravel()
        n = np.linalg.norm(dW) * np.linalg.norm(w0)
        return float(dW @ w0 / n) if n > 0 else 0.0


def main() -> None:
    import run_mujoco_demo as R

    pol = ExportedPolicy(R.POLICY)
    env = R.BoxPickupEnv(pol)

    print("=== INPUT MAP (gx_level=1: dtau/da = Kp * diag(action_scale)) ===")
    kp_adapter = np.array(pol.meta["joint_stiffness"])
    kp_plant_mj = kp_adapter * R.GAIN_SCALE
    print(f"  adapter Kp (meta stiffness)     : {kp_adapter[:4]} ...")
    print(f"  MuJoCo plant Kp (x{R.GAIN_SCALE} gain scale): {kp_plant_mj[:4]} ...")
    print(f"  -> mentor env understates Kp by  {R.GAIN_SCALE}x (adapter is given unscaled kp)")
    print(f"  MuJoCo leg target filter        : {R.LEG_FILTER}  -> leg authority x{1 - R.LEG_FILTER:.2f}")
    print(f"  MuJoCo rate limit               : {R.MAX_JOINT_STEP} rad/step")
    print("  Isaac plant: target = a*scale + default, NO filter, NO rate limit")
    print(f"  -> effective leg d(target)/d(a): MuJoCo {1 - R.LEG_FILTER:.2f}x vs Isaac 1.00x")

    seeds = list(range(600, 608))
    variants = [
        ("shipped  (leak -> 0)", dict(leak_to_w0=False), {}),
        ("leak -> W0 (sigma-mod)", dict(leak_to_w0=True), {}),
        ("leak = 0", dict(leak_to_w0=False), dict(leak=0.0)),
    ]

    print("\n=== LEAK TERM: Wdot = Gamma*delta*z^T - gamma*W ===")
    for label, kw, cfg_over in variants:
        rows = []
        for s in seeds:
            cfg = AdaptConfig(**cfg_over)
            ad = InstrumentedAdapter(pol, cfg, joint_names=pol.meta["joint_names"], **kw)
            env.reset(seed=s)
            ad.reset()
            surv, trk, err = R.rollout(env, ad, 300)
            rows.append((surv, trk, err, ad.weight_drift, ad.learn_norm, ad.leak_norm,
                         ad.leak_alignment()))
        a = np.array(rows, dtype=float)
        print(f"\n  {label}")
        print(f"    survival  {a[:,0].mean()*env.ctrl_dt:5.2f}s   legErr {a[:,2].mean():6.2f} deg   "
              f"|dW| {a[:,3].mean():.3f}")
        print(f"    integrated |learn| {a[:,4].mean():.4f}   |leak| {a[:,5].mean():.4f}   "
              f"leak share {a[:,5].mean()/max(a[:,4].mean()+a[:,5].mean(),1e-9):.0%}")
        print(f"    cos(dW, -W0) = {a[:,6].mean():+.3f}   (near +1 => drift is mostly decay toward zero)")


if __name__ == "__main__":
    main()
