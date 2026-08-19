# Pre-registration — ONLINE across-episode adaptation of the output-layer bias on `tq05`

**Written 2026-08-18 ~16:30 PDT, BEFORE the run.** Results will be appended below; deviations
recorded as deviations.

## What this converts, and why it is the next step

`PREREG_CONST_TQ05.md` confirmed (primary passed, +1.70 m held-out) that a CEM over a constant
20-d residual — which is **exactly the bias of `mlp.6`**, since the output layer is linear —
finds ~68% of the fault's headroom from realised episode returns alone. That run was *test-time
optimisation*: candidates were scored by re-running two fixed train seeds, i.e. the same
environment draw was replayed, which a deployed robot cannot do.

This experiment removes that privilege. **Every episode uses a fresh seed, in a fixed deployment
order, and no seed is ever revisited.** If the search still finds the fix, the claim becomes:
*online (across-episode) adaptation of one layer's bias recovers the fault at deployment, with no
privileged information* — which is the paper's on-scope object (a layer, adapted online), obtained
by the only mechanism that has survived measurement (realised-metric search), and deployable on a
gradient-free frozen model (ONNX/VLA case).

## Design (constants copied from `const_adapt.py`, not tuned)

- Fault `tq05` = `torque_limit_scale 0.5`, onset after `n_pre=100` nominal steps, `n_post=300`.
- Class: constant residual `r ∈ R^20`, clip `u_max = 0.10` (≡ `mlp.6` bias delta).
- CEM: `pop 10 × iters 6`, elites 3, `sigma0 0.04` — **60 deployment episodes** total.
- **Common-seed-per-generation:** each generation's 10 candidates are scored on ONE fresh seed
  (5000+gen for `cem_online`); ranking within a generation therefore compares candidates under
  identical conditions, while no seed is ever reused across generations.
- **Settled estimate for every arm: the elite-mean (top-3) of the FINAL generation.** (For CEM
  this equals `mu_final`.) Best-ever is recorded but is a selection object, not the estimate.
- Arms, all on the same seed schedule 5000–5005 (one per generation):
  1. `cem_online` — distribution refit each generation (learning);
  2. `random_search` — identical protocol, sampling distribution NEVER refit (stays
     `N(0, sigma0)`): the selection-without-learning control;
  3. `frozen` — no residual, same 6 seeds (deployment baseline stream).
- Held-out evaluation: settled θ of arms 1 and 2, plus frozen, on seeds **3000–3005** (disjoint
  from the deployment stream and from the original train seeds).

## Primary endpoint

**Held-out distance, `cem_online` settled θ − frozen > +1.0 m** (same threshold as the confirmed
offline result; predicted +1.0…+1.7 m).

## Secondary endpoints (report, no gate)

- `cem_online` settled vs `random_search` settled on held-out seeds (learning vs selection at
  identical budget, identical estimator).
- Deployed-stream regret: per-generation mean episode score of each adaptive arm vs `frozen` on
  the same seed (was the robot already better DURING adaptation?).
- `‖r‖` trajectory across generations vs the known fix scale (0.14–0.17).

## Decision rule

| outcome | reading |
|---|---|
| primary passes AND cem > random | online layer-bias adaptation works; learning contributes beyond selection |
| primary passes, cem ≈ random | the class + blind search suffices online; report as class effect (consistent with the 20-draw control, p=0.095) |
| primary fails, offline result stands | single-episode scoring noise breaks selection at this budget — condition E at the episode level; report the budget floor |

## Stated in advance

- No rescue runs at other `sigma0`, pop, iters, or seed schedules.
- 6 distinct environment draws is thin exposure; the offline result transferred from 2, and the
  fix is a property of the fault, so transfer is expected — if train/held-out diverge sharply,
  that is reported, not tuned away.
- The random-search arm is the matched control demanded by rule 11 (null the SEARCH): identical
  budget, identical estimator, only the refit differs.

---

## Results — 2026-08-18 17:44 PDT. **PRIMARY FAILS — and the failure mode is measured.**

Raw: `outputs/online_const_tq05.json`, log `outputs/online_const_tq05.log`.

| arm (held-out 3000–3005) | steps | dist | ‖r‖ | vs frozen |
|---|---|---|---|---|
| frozen | 157.5 | +3.41 m | — | — |
| **cem_online settled** | 165.7 | +3.52 m | **0.199** | **+0.11 m — FAIL at +1.0** |
| random_search settled | 196.8 | +4.34 m | 0.072 | +0.93 m |
| cem_best_ever | 198.5 | +4.22 m | 0.192 | +0.81 m |
| random_best_ever | 187.8 | +3.89 m | 0.175 | +0.48 m |

Decision-rule row that fires: **"primary fails, offline result stands → single-episode
scoring noise breaks selection at this budget — condition E at the episode level."**
No arm reaches +1.0 m; the offline result (2-seed-averaged scoring, +1.70 m) stands.

### The mechanism, visible in the settled-norm trajectory

`‖settled‖` under refit: 0.104 → 0.142 → 0.187 → 0.194 → 0.200 → 0.199 — **monotone growth
past the known fix scale** (offline CEM 0.141; horizon oracle 0.149). The control, which
never refits, stays at prior scale (0.072–0.114) and lands +0.93 m.

Reading: top-3-of-10 selection on a SINGLE episode is **variance-seeking** — outcome variance
grows with ‖r‖, so a lucky big candidate out-scores an honest small one, the refit chases it,
and μ walks past the fix. Selection noise does not merely slow online learning; it biases the
searched norm upward. The offline run avoided this only by averaging each candidate over 2
seeds.

### What this changes

The online claim as pre-registered here is dead at this budget. The mechanism prescribes ONE
change — average each candidate over 2 fresh seeds — which is pre-registered as
`PREREG_ONLINE_TQ05_V2.md` (120 deployment episodes) with the explicit falsifier that a
second failure closes the online chapter at the ~100-episode scale and the paper reports the
budget floor: test-time optimisation with per-candidate averaging works; strictly-online
single-visit selection does not.

### Deviations

None. No rescue runs; V2 is a new document with one mechanism-derived change.
