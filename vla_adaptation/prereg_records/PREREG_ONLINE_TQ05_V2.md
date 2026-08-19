# Pre-registration — online tq05, V2: one change, derived from V1's measured failure mode

**Written 2026-08-18 17:50 PDT, BEFORE the run.** V1 (`PREREG_ONLINE_TQ05.md`) failed its
primary at +0.11 m with a measured mechanism: single-episode scoring makes top-3-of-10
selection **variance-seeking** — ‖settled‖ walked 0.104 → 0.199, past the known fix scale
(0.141–0.149), while the never-refit control stayed at prior scale and scored +0.93 m.

## The one change

**Each candidate is scored on the MEAN of 2 fresh seeds** (both common to the generation's
candidates). Everything else is byte-identical to V1: pop 10 × 6 generations, elites 3,
sigma0 0.04, u_max 0.10, settled = elite-mean of the final generation, same arms
(cem_online, random_search never-refit control, frozen), held-out 3000–3005.

Budget: 6 gens × 10 pop × 2 seeds = **120 deployment episodes** per adaptive arm
(V1 was 60). Deployment seeds 5100–5111, disjoint from V1's 5000–5005 and from held-out.
No seed is ever revisited across generations — the online (no-replay) property is kept.

## Why this is the mechanism-derived change and not a rescue

Averaging over 2 seeds is exactly what the offline run (which passed at +1.70 m) did, and it
attacks the measured pathology directly: candidate-seed interaction luck — which grows with
‖r‖ and drove the variance-seeking refit — averages down by √2, in BOTH the selection and
the settled estimate, without touching any other constant.

## Primary endpoint

Held-out distance, cem_online settled − frozen **> +1.0 m** (unchanged from V1).

## Predictions, in advance

1. Primary passes (offline analog passed at +1.70 m with the same per-candidate averaging).
2. `‖settled‖` no longer overshoots: final value in **[0.10, 0.18]** (fix scale), not ~0.20.
3. cem_online settled ≥ random_search settled (learning beats selection once selection is
   not noise-dominated).

## Decision rule

| outcome | reading |
|---|---|
| primary passes AND prediction 2 holds | online layer-bias adaptation works at ~120 episodes; V1-vs-V2 is the measured price of selection noise |
| primary passes, ‖settled‖ still ~0.20 | it works for a different reason than claimed — report both, no further variants |
| **primary fails** | **the online chapter closes at the ~100-episode scale.** The paper reports the boundary: test-time optimisation with per-candidate averaging works offline; strictly-online selection does not convert at this budget. No V3. |

## Stated in advance

- No rescue runs at other pop/iters/sigma0/seed schedules. No V3 on failure.
- This is the second bite at one cell; the multiple-comparison cost is recorded here and
  must be reported with any positive.

---

## Results — 2026-08-18 23:00 PDT. **PRIMARY PASSES. The online chapter is open.**

Raw: `outputs/online_const_tq05_v2.json`, log `outputs/online_const_tq05_v2.log`.

| arm (held-out 3000–3005) | steps | dist | ‖r‖ | vs frozen |
|---|---|---|---|---|
| frozen | 157.5 | +3.41 m | — | — |
| **cem_online settled** | **224.8** | **+4.79 m** | **0.156** | **+1.38 m — PASS at +1.0** |
| random_search settled | 164.3 | +3.47 m | 0.124 | +0.06 m |
| cem_best_ever | 207.3 | +4.51 m | 0.158 | +1.10 m |
| random_best_ever | 219.2 | +4.97 m | 0.222 | +1.56 m |

All three pre-registered predictions hold:

1. **Primary +1.38 m > +1.0 m** (offline analog was +1.70 m).
2. **‖settled‖ = 0.156 ∈ [0.10, 0.18]** — the fix scale. Full trajectory 0.104 → 0.139 →
   0.140 → 0.158 → 0.159 → 0.156; V1's variance-seeking divergence (→0.199) is gone. The
   V1→V2 pair is the measured price of selection noise: one change (mean-of-2-seeds
   scoring), failure → pass.
3. **Learning beats selection at identical budget and estimator: +1.33 m** (cem settled
   +4.79 vs random settled +3.47). Deployed-stream regret: cem's per-generation mean ≥ the
   random arm's on the same seed pairs in **6/6 generations**.

Decision-rule row: *"primary passes AND prediction 2 holds → online layer-bias adaptation
works at ~120 episodes; V1-vs-V2 is the measured price of selection noise."*

### Carry with the number, always

- This is the **second bite** at the cell (V1 failed, V2 declared in advance and passed) —
  the multiple-comparison cost is recorded here and travels with the claim.
- `random_best_ever` (+1.56 m, ‖r‖ 0.222) beat everything — one lucky large draw out of 60.
  The pre-registered contrast is between SETTLED estimators, which is what an online
  algorithm can actually commit to without held-out evaluation (deployment has none); the
  best-ever columns are selection objects and say the class landscape still contains lucky
  members. Report both.
- The claim: **online, across-episode, gradient-free adaptation of one layer's bias (mlp.6)
  recovers a torque fault at deployment — 120 fresh-seed episodes, no replay, no privileged
  information — recovering ~55% of headroom (+1.38 of 2.50 m) on held-out seeds.**

### Deviations

None. Constants byte-identical to V1 except the pre-registered change.
