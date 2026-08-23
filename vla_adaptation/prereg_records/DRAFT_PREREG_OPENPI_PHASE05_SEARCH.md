# DRAFT pre-registration — Phase 0.5 on openpi: the first adaptation run on a VLA

**DRAFT — not in force until signed off.** Written 2026-08-21, while the ACE screen is
still running and **before any screen result is known**. That timing is the point: §1 below
fixes how the edit site is chosen from the screen's outcome, so the choice cannot be made
after seeing the ranking.

## Established, not assumed

- **The cell.** `libero_spatial`, π0.5-LIBERO frozen, client-side fault `offset @ 0.05` on
  arm dims 0–5 — the only cell to survive the Phase 0.4 gate (`PREREG_OPENPI_ACE_SCREEN` §1).
- **Headroom.** Nominal 99.0% (500 episodes). Faulted frozen: 11/20 = 55% at the gate,
  8/20 = 40% at the screen baseline on disjoint initial states — **19/40 = 47.5% pooled**,
  so roughly **51 points** of headroom, and the arm is neither floor-dead nor at ceiling.
- **The class is reachable in principle, and attenuated.** §0 measured on trained weights
  that a bias edit β lands as **0.34·(−β)** on the action, not the analytic −β: the trained
  flow attenuates a bias edit about 3×. Any repair scale must be read through that factor.
- **The intervention is small next to the sampler.** §0b: at c = 0.02 a layer perturbation
  moves the action ~0.0003 against a 0.019 sampler noise floor.

## 1. The edit site, fixed before the screen reports

Per `PREREG_OPENPI_ACE_SCREEN` §6, and stated here in advance:

| screen outcome | site searched here |
|---|---|
| primary passes | the top-ranked site by \|ACE_hat\|, **if** it admits a ≤20-dim parameterisation |
| primary passes, top site is a trunk/expert matrix | **`action_out_proj/bias`** — a trunk site needs a basis first, which is a separate experiment, not this one |
| primary fails | **`action_out_proj/bias`** (§6's "proceeds directly to a tier-1 search, which needs no layer ranking") |

**The search dimension is the ≤7 model action dims that map to LIBERO's arm channels**, not
all 32 — the remaining dims are unused by this embodiment and searching them would inflate
the dimension for nothing.

## 2. The gate before the search — RUN, and it passed at the ceiling

**Run 2026-08-22, before the search. Result: the class is fully reachable.**
`results/oracle/FINDINGS_ONLINE_REPAIR.md`.

The scalar β ≈ 0.147 written in the first version of this draft was **wrong**, and is
corrected here. It ignored normalisation: π0.5 emits normalised actions and `Unnormalize`
runs afterwards, so a bias edit lives in normalised units while the fault is applied in env
units. With quantile norm each dim has scale `(q99−q01)/2`, and those differ 7× across the
arm channels — a uniform +0.05 env offset is 3.0% of the action range on translation and
19.5% on `drx`. The repair is therefore a per-dim vector

    β_i = (0.05 / scale_i) / 0.34        spanning 0.157 … 1.149

| k on β | faulted | result |
|---|---|---|
| −1.0 | yes | 0.0% |
| 0.0 | yes | 46.7% (floor) |
| 0.5 | yes | 86.7% |
| **1.0** | yes | **100.0%** |
| 1.5 | yes | 20.0% |
| 1.0 | **no** | 6.7% |

**Ceiling = 100%**, so the §2 gate ("oracle recovers ≥ 70% of headroom") passes outright and
the search is licensed. Three things this fixes in the design below:

1. **σ₀ is now set from a measured basin, not a guess.** k = 1.5 scores 20% — worse than no
   repair — so the useful range is roughly k ∈ [0.4, 1.2]. σ₀ = 0.3·β_oracle would put over
   a third of the initial population past k = 1.3. **σ₀ is reduced to 0.15·β_oracle.**
2. **The search is parameterised in ENV-ACTION units, not raw bias units.** In env units the
   target is isotropic (−0.05 on all six dims); in bias units it spans 7.3× and an isotropic
   CEM is ill-conditioned exactly where the fault is largest. The search vector is `u` in env
   units, mapped to the bias by `β_i = −u_i / (scale_i · 0.34)`.
3. **A specificity control is added to §4**: the settled edit applied to the UNFAULTED policy.
   The oracle edit takes a healthy policy from 99% to 6.7%, so a settled edit that does *not*
   damage the healthy policy is not the inverse and would need explaining.

## 3. Method

Across-episode CEM on the edit vector, exactly the protocol the walker validated:

- population 10 × 6 generations, elites 3, **σ₀ = 0.15·β_oracle** (§2, measured basin)
- **≥ 2 fresh episodes per candidate**, both common to the generation's candidates —
  single-episode scoring is variance-seeking and measurably fails (`PREREG_ONLINE_TQ05` → `_V2`)
- **fresh seeds only, no replay**; no initial state is revisited across generations
- settled estimate = **elite-mean of the final generation**, for every arm

## 4. Arms

| arm | what it isolates |
|---|---|
| `cem_online` | the method |
| `never_refit` | selection without learning — same loop, refit disabled |
| `random_norm_matched` | edits of the same norm, no search |
| `frozen_faulted` | the 47.5% floor |
| `oracle` | §2's analytic inverse — the ceiling, measured at **100%** |
| `settled_unfaulted` | specificity: the settled edit on the HEALTHY policy |

## 5. Budget

Oracle 20 + cem_online 120 + never_refit 120 + random 40 + held-out settled 40×3 = **420
episodes**, ~4 h at the measured ~35 s/episode. Held-out initial states are disjoint from
the gate's (0–1), the screen's (2–5) and the screen baseline's (6–7).

## 6. Primary endpoint

**`cem_online` settled held-out success − `frozen_faulted` > +15 points**, and beating
`never_refit` by any margin. Both required: the first is the effect, the second is that it
came from learning rather than from selection luck.

## 7. Predictions, stated in advance

1. ~~The oracle clears 70% of headroom.~~ **Confirmed before the search: 100%.**
2. `cem_online` passes the primary. It is a ≤7-dim search on a class known to be reachable.
3. `never_refit` recovers a non-trivial share anyway — selection alone was worth +1.33 m of
   +1.38 m on the walker, and that near-tie is the result most likely to repeat here.
4. **A positive result here is weak evidence for the general claim.** Fault class = edit
   class is the easy case; the gate did not supply the state-dependent multiplicative cell
   that would test the hard one. Stated now so it is not discovered in review.

## 8. Stated in advance

- No rescue runs at another sigma0, population, or generation count.
- If the oracle fails its own gate (§2), this document stops there and reports it.
- **Not committed in this draft:** a gradient arm (REINFORCE through the flow sampler with
  exploration noise, same cell, same budget, same metric) as a declared head-to-head against
  `cem_online`. It belongs here rather than in a later document if it is wanted, because a
  same-budget comparison is only clean when both arms are declared together.
