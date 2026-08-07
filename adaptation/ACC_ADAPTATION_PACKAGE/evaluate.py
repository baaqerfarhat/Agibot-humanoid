"""Paired frozen-vs-adapted evaluation with an exact sign test and a matched random null.

    python evaluate.py                  # 32 seeds, frozen vs adapted
    python evaluate.py --null           # + displacement-matched random control (20 draws)
    python evaluate.py --seeds 16 --seed0 700

Survival and tracking error are reported SEPARATELY and both matter — a combined
"steps before falling OR losing tracking" score lets a tracking gain read as a survival gain.

The matched null is not optional if you intend to claim the ADAPTATION DIRECTION is doing the
work. The frozen policy here is fragile enough that random perturbation of a layer can by itself
improve things, so beating frozen is necessary but not sufficient.
"""
from __future__ import annotations

import argparse

import numpy as np

from ace_adapt import AdaptConfig, ExportedPolicy, LayerAdapter
from run_mujoco_demo import POLICY, BoxPickupEnv, rollout


def sign_test(a, b):
    """Exact binomial sign test on paired samples; returns (wins, n, p)."""
    from scipy.stats import binomtest
    d = np.asarray(a) - np.asarray(b)
    nz = d[d != 0]
    if len(nz) == 0:
        return 0, 0, 1.0
    w = int((nz > 0).sum())
    return w, len(nz), float(binomtest(w, len(nz), 0.5).pvalue)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--seed0", type=int, default=600)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--gain", type=float, default=3e-4)
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--null", action="store_true", help="run the matched random control")
    ap.add_argument("--null-draws", type=int, default=20)
    args = ap.parse_args()

    pol = ExportedPolicy(POLICY)
    env = BoxPickupEnv(pol)
    cfg = AdaptConfig(layer=args.layer, gain=args.gain)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    print(f"seeds {seeds[0]}..{seeds[-1]} ({len(seeds)}), layer {cfg.layer}, Gamma {cfg.gain}")
    print(f"cap {args.max_steps} steps = {args.max_steps*env.ctrl_dt:.1f} s\n")

    fr, ad, drifts, ndiv = [], [], [], 0
    for s in seeds:
        env.reset(seed=s)
        fr.append(rollout(env, None, args.max_steps))
        env.reset(seed=s)
        adapter = LayerAdapter(pol, cfg, joint_names=pol.meta["joint_names"])
        adapter.reset()
        ad.append(rollout(env, adapter, args.max_steps))
        drifts.append(adapter.weight_drift)
        ndiv += int(adapter.diverged)

    fr = np.array(fr, dtype=float)      # columns: survival, tracked, legErr
    ad = np.array(ad, dtype=float)
    dt = env.ctrl_dt

    print(f"{'metric':<24} {'frozen':>10} {'adapted':>10} {'change':>10}")
    print(f"{'survival (steps)':<24} {fr[:,0].mean():>10.1f} {ad[:,0].mean():>10.1f} "
          f"{100*(ad[:,0].mean()/fr[:,0].mean()-1):>9.1f}%")
    print(f"{'survival (s)':<24} {fr[:,0].mean()*dt:>10.2f} {ad[:,0].mean()*dt:>10.2f}")
    print(f"{'never fell':<24} {int((fr[:,0]>=args.max_steps).sum()):>10d} "
          f"{int((ad[:,0]>=args.max_steps).sum()):>10d}   of {len(seeds)}")
    print(f"{'leg tracking error (deg)':<24} {fr[:,2].mean():>10.2f} {ad[:,2].mean():>10.2f} "
          f"{100*(ad[:,2].mean()/fr[:,2].mean()-1):>9.1f}%")
    print(f"{'|dW| median':<24} {'-':>10} {np.median(drifts):>10.3f}")
    print(f"{'diverged':<24} {'-':>10} {ndiv:>10d}")

    print("\npaired exact sign tests (adapted vs frozen)")
    for name, col, better_is_high in (("survival", 0, True), ("leg tracking error", 2, False)):
        a, b = (ad[:, col], fr[:, col]) if better_is_high else (-ad[:, col], -fr[:, col])
        w, n, p = sign_test(a, b)
        print(f"  {name:<22} better {w}/{n}, p = {p:.4f}")

    if args.null:
        print(f"\nmatched random null, {args.null_draws} draws at |dW| = {np.median(drifts):.3f}")
        mag = float(np.median(drifts))
        means_err, means_surv = [], []
        for d in range(args.null_draws):
            rng = np.random.default_rng(90_000 + d)
            g = rng.normal(size=pol.W0[cfg.layer].shape)
            dW = g / np.linalg.norm(g) * mag
            errs, survs = [], []
            for s in seeds:
                env.reset(seed=s)
                adapter = LayerAdapter(pol, cfg, joint_names=pol.meta["joint_names"])
                adapter.reset()
                adapter.W[cfg.layer] = pol.W0[cfg.layer] + dW
                adapter.cfg.gain = 0.0            # perturbation only, no adaptation
                r = rollout(env, adapter, args.max_steps)
                survs.append(r[0]); errs.append(r[2])
            means_surv.append(np.mean(survs)); means_err.append(np.mean(errs))
            print(f"  draw {d:2d}: survival {np.mean(survs):6.1f}  legErr {np.mean(errs):6.2f}")
        ms, me = np.array(means_surv), np.array(means_err)
        beats = int((me <= ad[:, 2].mean()).sum())
        print(f"\n  null legErr : mean {me.mean():.2f}, range {me.min():.2f}..{me.max():.2f}")
        print(f"  adapted     : {ad[:,2].mean():.2f}   frozen: {fr[:,2].mean():.2f}")
        print(f"  draws matching or beating the adaptation: {beats}/{args.null_draws} "
              f"-> empirical p = {(beats+1)/(args.null_draws+1):.3f} "
              f"(floor {1/(args.null_draws+1):.3f})")
        if beats == 0:
            print("  => the ADAPTATION DIRECTION carries the result, not the perturbation size.")


if __name__ == "__main__":
    main()
