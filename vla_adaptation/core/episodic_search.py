"""Model-agnostic episodic intervention search — the adaptation method that survived.

Distilled from the X2 walker studies (see ../prereg_records/ for the pre-registered runs
and ../docs/WHEN_ADAPTATION_WORKS.md for the six conditions). The method: adapt a small
(dim <= ~20) parameter vector theta — typically one layer's bias or a diagonal rescale of
the output layer — by CEM on the REALISED task metric, one candidate per (fresh) episode
batch. No gradients anywhere, so it runs on any frozen model (ONNX, served-over-websocket,
black box).

Lessons baked in (each cost a failed pre-registered run to learn):

1. ``seeds_per_gen >= 2``. Scoring candidates on a SINGLE episode makes top-k selection
   VARIANCE-SEEKING: outcome variance grows with ||theta||, lucky large candidates outrank
   honest small ones, and the refit walks the searched norm past the fix scale. Measured:
   V1 (1 seed/cand) failed at +0.11 m with ||settled|| 0.104 -> 0.199; V2 (2 seeds/cand,
   the only change) passed at +1.38 m with ||settled|| 0.156. (PREREG_ONLINE_TQ05*.md)
2. The deployable estimate is ``settled`` = elite-mean of the FINAL generation — the same
   estimator for treatment and control. ``best_ever`` is a selection object; deployment has
   no held-out evaluator to recognise luck, so never report best_ever as the method.
3. The matched control is the SAME loop with ``refit=False`` (rule 11: null the search,
   not just the result) — identical budget, identical estimator, only the learning differs.
4. ``ace_arm`` (mean effect of random interventions) is a CLASS-EFFECT detector, NOT a
   searchability screen: measured positive in exactly 1 of 4 cells while the search
   succeeded in all 4. Use headroom + reachability oracles as gates instead.
5. Report the induced action displacement per arm (rule 6); if candidates are clipped by a
   deployment envelope, measure the clipped ceiling FIRST — the envelope can select the
   function class before the search does (measured: multiplicative-fault ceiling 5.8%
   clipped vs 100.0% via direct layer edit).

The episode callable owns everything environment-specific::

    def episode(theta: np.ndarray, seed: int) -> float:   # realised score, higher better
        ...

Self-test: ``python episodic_search.py`` reproduces the variance-seeking mechanism on a
synthetic heteroscedastic objective in ~2 s (no dependencies beyond numpy).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

EpisodeFn = Callable[[np.ndarray, int], float]


def run_search(episode: EpisodeFn, dim: int, gen_seeds: Sequence[Sequence[int]],
               pop: int = 10, elites: int = 3, sigma0: float = 0.04,
               clip: float | None = None, refit: bool = True,
               rng: np.random.Generator | None = None) -> dict:
    """Across-episode CEM (refit=True) or its matched selection control (refit=False).

    ``gen_seeds``: one sequence of FRESH seeds per generation (len >= 2 each — lesson 1);
    all candidates of a generation share its seeds (common random numbers within a
    generation, never across). Returns settled (lesson 2), best_ever, and the history.
    """
    rng = rng or np.random.default_rng(0)
    mu, sigma = np.zeros(dim), np.full(dim, sigma0)
    best = (-np.inf, np.zeros(dim))
    history = []
    settled = mu
    for gen, seeds in enumerate(gen_seeds):
        pool = rng.normal(mu, sigma, size=(pop, dim))
        if clip is not None:
            pool = np.clip(pool, -clip, clip)
        vals = []
        for th in pool:
            v = float(np.mean([episode(th, s) for s in seeds]))
            vals.append(v)
            if v > best[0]:
                best = (v, th.copy())
        idx = np.argsort(vals)[::-1][:elites]
        settled = pool[idx].mean(0)          # elite-mean: the deployable estimate
        if refit:
            mu, sigma = settled, pool[idx].std(0) + 1e-3
        history.append({"gen": gen, "seeds": list(map(int, seeds)),
                        "mean": float(np.mean(vals)),
                        "elite": float(np.mean([vals[i] for i in idx])),
                        "norm_settled": float(np.linalg.norm(settled))})
    return {"settled": settled, "best_ever": best[1], "best_ever_score": best[0],
            "history": history}


def ace_arm(episode: EpisodeFn, dim: int, scale: float, seeds: Sequence[int],
            draws: int = 8, rng: np.random.Generator | None = None) -> dict:
    """Mean effect of norm-matched RANDOM interventions vs theta=0 (the ACC-2026 ACE
    estimator). A class-effect detector only — see lesson 4."""
    rng = rng or np.random.default_rng(10)
    base = float(np.mean([episode(np.zeros(dim), s) for s in seeds]))
    effs = []
    for _ in range(draws):
        th = rng.normal(0.0, 1.0, dim)
        th *= scale * np.sqrt(dim) / np.linalg.norm(th)
        effs.append(float(np.mean([episode(th, s) for s in seeds])) - base)
    return {"mean": float(np.mean(effs)), "sd": float(np.std(effs)), "effects": effs,
            "base": base}


def heldout(episode: EpisodeFn, arms: dict[str, np.ndarray | None],
            seeds: Sequence[int]) -> dict:
    """Evaluate settled arms (and the frozen baseline: theta=None) on disjoint seeds."""
    out = {}
    for name, th in arms.items():
        t = np.zeros(1) if th is None else th          # theta=None means frozen
        vals = [episode(np.zeros_like(t) if th is None else th, s) for s in seeds]
        out[name] = {"mean": float(np.mean(vals)), "per_seed": list(map(float, vals))}
    return out


# ----------------------------------------------------------------------------------------
# Self-test: reproduces the variance-seeking mechanism (lesson 1) synthetically, as a
# STATISTICAL ordering over 30 repetitions: mean ||settled|| is larger with 1 seed per
# candidate than with 3, on a bowl whose noise sd grows with ||theta||. The dramatic
# single-run version is the walker record itself (PREREG_ONLINE_TQ05: 0.199 vs 0.156).
# ----------------------------------------------------------------------------------------
def _selftest():
    dim, tstar = 20, np.full(20, 0.03)                  # ||t*|| = 0.134, the "fix scale"
    tnorm = np.linalg.norm(tstar)

    def episode(th, seed):
        # seed-and-theta-dependent noise whose sd grows with ||theta|| (heteroscedastic,
        # as on the robot); the rng must depend on theta too, else common seeds cancel it.
        r = np.random.default_rng([seed, int(1e6 * abs(th.sum()))])
        noise = r.normal(0.0, 0.05 + 8.0 * np.linalg.norm(th))
        return -float(np.sum((th - tstar) ** 2)) * 20.0 + noise

    reps = 30
    means = {}
    for spg in (1, 3):
        norms = []
        for rep in range(reps):
            seeds = [[9000 * rep + g * spg + j for j in range(spg)] for g in range(8)]
            out = run_search(episode, dim, seeds, sigma0=0.04, clip=0.10,
                             rng=np.random.default_rng(rep))
            norms.append(out["history"][-1]["norm_settled"])
        means[spg] = (float(np.mean(norms)), float(np.std(norms)))
        print(f"seeds/gen={spg}:  ||settled|| = {means[spg][0]:.3f} +/- {means[spg][1]:.3f}"
              f"  over {reps} reps   (fix scale {tnorm:.3f})")
    verdict = "REPRODUCED" if means[1][0] > means[3][0] else "not reproduced this build"
    print(f"variance-seeking (1-seed norm > 3-seed norm): {verdict}")


if __name__ == "__main__":
    _selftest()
