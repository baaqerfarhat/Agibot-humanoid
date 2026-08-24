# Pre-registration — ACE under the corrected protocol, confirmatory

**Written 2026-08-23, BEFORE the run.** The exploratory result this tests
(`LESSONS_ADAPTATION.md` §9h) is on 4 of 10 sites at a scale chosen after seeing the
original null, so it cannot count as confirmatory no matter how small its p-value. This
document fixes the protocol, the sites, the budget and the endpoint in advance, on
initial states never used before.

## Established, not assumed

- **The pre-registered screen failed as declared.** `PREREG_OPENPI_ACE_SCREEN` §4:
  F(9,70) = 0.294, p = 0.974, η² = 0.036. That stands and is not revisited here.
- **Four measurement faults were then identified and measured** (§9d, §9g): unpaired
  scoring (96% of a 5-episode score is sampler noise); a perturbation scale below the
  threshold of behaviour (0 of 6 outcome flips at c = 0.02); relative-parameter-norm
  matching (induced |ΔAction| spans 30× across sites at matched c); and 78% of each draw
  spent on the 25 action dims LIBERO discards as padding.
- **Exploratory re-measurement with those fixed** gave F(3,44) = 8.751, p = 0.0001,
  η² = 0.374 on 4 sites — the result under test here.
- **`llm/mlp/linear/L17` produces exactly zero change in the action** at 50% of its own
  norm, at two scales. It cannot be output-matched because its output response is zero.
  It is **excluded and reported as a finding**, not silently dropped: a whole trunk layer
  with no causal path to the action. 9 sites remain.

## 1. Protocol, fixed

- **Paired and deterministic.** Sampler RNG pinned; a repeated episode gives
  `max|ΔAction| = 0.0` exactly. One deterministic baseline serves every site.
- **Output-matched scale.** Per-site `c` from `results/ace_v2/matched_c.json`, chosen so
  every site induces a common `|ΔAction| = 0.02`. Range 0.74 … 21.2.
- **Task subspace.** Perturbations restricted to the 7 action dims LIBERO uses.
- **Antithetic variates for variance reduction.** Each draw runs +Δ and −Δ; the ACE value
  is `(M₊ + M₋)/2 − M_base`. This is a variance-reduction technique for estimating
  `E[M(W+Δ)]`, **not** a change of estimand: the quantity scored is the ACC estimator
  unchanged. The first-order contrast `M₊ − M₋` is recorded alongside as a secondary.
- **Held-out initial states 10–17**, disjoint from the gate (0–1), the original screen
  (2–5), its baseline (6–7) and every exploratory run (8–9).

**Budget: 9 sites × 8 draws × 2 arms × 10 episodes = 1440 episodes.** Ten episodes span all
ten tasks, identical across every site and draw (common random numbers).

## 2. Primary endpoint

One-way ANOVA of ACE over 9 sites × 8 draws (df = 8, 63), plus `η² = SS_between/SS_total`.

**Primary passes iff p < 0.05 AND η² ≥ 0.25** — the same two bars as the original screen,
unchanged, so the two runs are directly comparable.

## 3. Predictions, stated in advance

1. **Primary passes.** The exploratory η² was 0.374; even at 0.25 the power here is ample.
2. **Tier ordering: interface > action expert > VLM trunk** by mean ACE. This is the
   original prereg's prediction P2, which its own run could not resolve.
3. **`action_out_proj/bias` is NOT uniquely top-ranked.** It was statistically tied with
   `action_in_proj/kernel` (p = 0.86) in the exploratory run. If it now separates cleanly,
   that is a stronger claim than the exploratory data supports and is reported as a surprise.
4. **The first-order secondary does NOT discriminate** (exploratory η² = 0.068). If it does
   here, the exploratory comparison was underpowered rather than decisive.

## 4. Decision rule

| outcome | reading |
|---|---|
| primary passes, P2 holds | ACE ranks layers on a 3.35B VLA under a declared protocol, and the null in the original screen was a measurement artifact. The four fixes become the standard protocol for this project. |
| primary passes, P2 fails | ACE discriminates but not along the interface/trunk axis; the ranking is reported as an empirical object. |
| **primary fails** | The exploratory result was overfitted to its 4 sites or its scale choice. The corrected protocol does **not** generalise, ACE stays retired as a selection stage, and §9h's retraction is itself retracted. |

## 5. Stated in advance

- **No rescue runs.** Not at another target `|ΔAction|`, another draw count, another episode
  count, another site list. One protocol, fixed above.
- **This tests ranking, not repairability.** ACE pointing at a tier does not mean a search
  there succeeds; that claim was falsified five times and is not revived.
- The excluded site is excluded for a measured reason recorded before the run, and its
  exclusion is reported with the result rather than as an implementation detail.

---

## Results

*(appended after the run)*
