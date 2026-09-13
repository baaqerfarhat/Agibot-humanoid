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
