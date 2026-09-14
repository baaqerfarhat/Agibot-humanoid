# Preregistration: seed-pinned repeats of the decision cells and the paired integral baseline (re4 theory plan, Part 8.2; 2026-09-13)

**Written before the runs.** Two comparisons were left unresolved at their registered size:
the static observer's gap on libero_10 (D.1: 9/40 against the method's 15/40, p = 0.07) and the
integral baseline, which the paper compares across cohorts rather than paired.

**8.2a Static observer vs method, n = 80, policy sampling pinned.** libero_10, tasks 0–9 ×
initial states 45–52, uniform +0.05, rotation corrected, headline constants; two runs under the
same protocol (`--scenario-reset`, section-0 records): the method (K = 6) and the static observer
(`--fir-k 0`). The policy server's sampler is pinned per episode (`pin_rng`, the ACE control
already supports it) so both runs see the same action draws given the same observations; this
removes the sampling noise the Part D comparison carried. Statistic: paired by (task, init),
method-only vs static-only counts, exact McNemar. Prediction: method − static ≥ 6 of 80 with
p < 0.05 (the D.1 gap of 6/40 scaled). Refutation: gap ≤ 3 or p ≥ 0.05 at n = 80 reports "FIR
memory does not resolve on the long-horizon suite at n = 80".

**8.2b Paired integral baseline (D.4).** libero_spatial, the headline 20 scenarios, translation
0.15 fault with translation corrected (the cell the paper's integral comparison uses), two runs:
the method and the integral baseline at its best gain (`--baseline integral --ki 0.005`), pinned
sampling. Prediction: method − integral ≥ 8 of 20 with p < 0.05 (the paper's 19/20 vs 7/20 cross-
cohort gap is 12). Refutation: gap ≤ 4 reports the integral comparison as unresolved when paired.

**Pinning note.** `pin_rng` fixes the flow sampler's noise to one key per policy call; with the
same observation sequence the two runs draw identical actions until their trajectories diverge,
so the pairing is on (task, init, draw) up to the first divergence, which is recorded per episode
(the step at which the executed commands first differ by more than 1e-6).

---

## Amendment before the rerun (2026-09-13, 10:05): initial states 42–49, not 45–52

LIBERO stores 50 initial states per task (indices 0–49). The registered "45–52" does not exist:
the first n = 80 run crashed at episode 51 (task 0, init 50) after 50 episodes, and the second
cell had started and would have crashed the same way; both were stopped (the partial
telemetry is kept unscored in `a_method_libero10_n80_partial_crash_init50/`). The runner now
wraps downward from `eval_init − 1` when it runs out of states, so n = 80 on ten tasks uses
inits 45–49 and then 44, 43, 42 (eight per task, none used by the shipped calibration except
45, which every shipped cell shares; the held-out calibration's states 25–34 are untouched).
Nothing else changes: statistic, band and refutation stand as registered. Run order: 8.2a
method, 8.2a static, 8.2b method, 8.2b integral, all on the server already running.

## Outcome, 8.2a (2026-09-13; `8_statistics/a_method_libero10_n80`, `a_static_libero10_n80`, `score_8_2a_method_vs_static.json`)

libero_10, ten tasks × inits 42–49, uniform +0.05, rotation corrected, pinned sampler
(`pinned:key0`), 80 pairs each run. Frozen 0/80 in both runs, outcomes identical pair by pair,
executed commands identical (within 1e-6 per channel, over the shared recorded steps) to the last step in 61 of 80 frozen episodes (the other 19 diverge at
a policy-call boundary, steps 10–150: the pinned sampler is exact up to the GPU's own
nondeterminism). The adaptive arms diverge at step 10–11 in every pair, the first update.

- Method (K = 6): **10/80**. Static observer (K = 0): **9/80**. Paired table: both 5, method-only
  5, static-only 4, neither 66. **Gap +1, exact McNemar p = 1.0.**
- **Registered refutation (primary): "FIR memory does not resolve on the long-horizon suite at
  n = 80."** The Part D.1 gap of 6/40 (record 44) was sampling noise at the cell level; with the
  draw pinned and the pairs doubled the two observers are indistinguishable on libero_10.
- Noted, not registered: on the four inits the shipped cell used (45–48) the method scores
  5/40 here against 15/40 unpinned (record 36). Pinning one key per call changes the absolute
  level (the same noise realisation every call is one policy, not the policy's average); the
  paired comparison is unaffected, the absolute level is not comparable across pinned and
  unpinned cells.

## Outcome, 8.2b (2026-09-13; `8_statistics/b_method_spatial_tra015`, `b_integral_spatial_tra015`, `score_8_2b_method_vs_integral.json`)

libero_spatial, the headline 20 scenarios, translation +0.15 with translation corrected, pinned
sampler, method vs the integral baseline at its best gain (k_i = 0.005). Frozen 2/20 in both
runs, identical pair by pair, commands identical within 1e-6 to the end in 17 of 20 frozen episodes; the
adaptive arms diverge at step 11 in every pair (the first update).

- Method **18/20**; integral **3/20**. Paired table: both 3, method-only 15, integral-only 0,
  neither 2. **Gap +15, exact McNemar p = 6.1×10⁻⁵.** Prediction (gap ≥ 8, p < 0.05) **holds**;
  the paper's cross-cohort 19/20 vs 7/20 is reproduced as a paired result, larger.
- Estimates (median last-50-step mean, % of the 0.15 fault): method 112 / 56 / 125 on x / y / z;
  integral 18 / 41 / 163. The integral baseline's x estimate never arrives and its z overshoots,
  which is the picture Part D.4 gave unpaired (record 44).

**8.2 summary.** The pinned design does what it was registered to do: frozen arms agree pair
by pair, the adaptive arms are paired exactly up to the first update, and both comparisons
are settled at the registered level — one for the method (integral baseline, +15 of 20,
p = 6×10⁻⁵) and one against a claim (FIR memory over a static observer on libero_10, +1 of 80,
p = 1). Pinned and unpinned absolute levels are not comparable (libero_10 method 5/40 pinned on
inits 45–48 against 15/40 unpinned).
