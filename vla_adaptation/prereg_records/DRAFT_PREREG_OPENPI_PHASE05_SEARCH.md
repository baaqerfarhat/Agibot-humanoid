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

## 2. The gate before the search — reachability, measured not argued

The framework's order is headroom *then reachability*, and reachability here is cheap and
has an analytic answer, so it is measured first and reported whatever it says.

The fault adds +0.05 to arm dims 0–5. Cancelling it needs an action shift of −0.05, which
through §0's measured 0.34 attenuation needs a bias edit of **β ≈ 0.05 / 0.34 ≈ 0.147** on
the corresponding dims. **Oracle arm: set that edit, run 20 episodes, report the recovery.**

- Oracle recovers **≥ 70% of headroom** → the class is reachable; the search runs.
- Oracle recovers **< 30%** → the ceiling, not the search, is the binding constraint. Report
  that and do **not** run a 400-episode search against a ceiling that is not there. This is
  the walker's envelope-selects-the-class result applied before the spend, not after.
- Between the two → run, and report the search against the measured ceiling rather than
  against full headroom.

## 3. Method

Across-episode CEM on the edit vector, exactly the protocol the walker validated:

- population 10 × 6 generations, elites 3, sigma0 chosen as 0.3·β_oracle
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
| `oracle` | §2's analytic inverse — the ceiling |

## 5. Budget

Oracle 20 + cem_online 120 + never_refit 120 + random 40 + held-out settled 40×3 = **420
episodes**, ~4 h at the measured ~35 s/episode. Held-out initial states are disjoint from
the gate's (0–1), the screen's (2–5) and the screen baseline's (6–7).

## 6. Primary endpoint

**`cem_online` settled held-out success − `frozen_faulted` > +15 points**, and beating
`never_refit` by any margin. Both required: the first is the effect, the second is that it
came from learning rather than from selection luck.

## 7. Predictions, stated in advance

1. The oracle clears 70% of headroom. The fault class and the edit class coincide exactly
   here — this is the most favourable geometry the gate could have handed us.
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
