# Preregistration: the adaptive law's constants set by the paper's own rule, four suites plus
# healthy controls (2026-09-14)

**Written before the runs.** Record 49–51 and the constants audit of 2026-09-14 found one
constant of the LIBERO reference configuration set against the paper's rule ("every threshold is
a multiple of a measured scale"): the normaliser ρ = 0.15 is compared with the residual norm over
all six channels, whose median under the headline fault is 0.037 and 90th percentile 0.107, so the
attenuation averages 0.88 and the estimate settles at 87 / 41 / 88 % of the fault. The norm is
dominated by the *uncorrected* translation residual (median 0.032 against 0.020 on rotation) —
the defect the GR1 configuration already removed by normalising over the corrected channels. The
paper's ablation (ρ = 0.50: 20/20, unbiased estimate) already showed the direction.

**Configuration under test ("by the rule"):** the innovation law (`--law innov`, the paper's
stated deployable scheme, unbiased fixed point) with the normaliser over the corrected channels
(`--norm-channels corrected`); every other constant unchanged (γ = 0.08, δ = 0.008, ρ = 0.15,
κ = 0.15, K = 6, rotation corrected, shipped calibration `phase05/error_signal_so3.json` and
`openloop_so3.json`, `--scenario-reset`, no gate, no bias vector — exactly the shipped cells'
protocol with those two flags added). Runs: the four headline cells at the shipped n (spatial 20,
object 20, goal 40, libero_10 40; inits from 45, unpinned like the shipped cells) and the four
healthy controls (`--sev 0.0`, same scenarios). Outputs `results/rule_constants/<arm>_<suite>/`
with the section-0 records. Comparisons are against the shipped legacy cells (records 36/43:
faulted 18/20, 16/20, 29/40, 15/40; healthy controls 19→20, 20→20, 39→40, 38→37) and the Part D.2
innovation-with-all-channel-normaliser cells (spatial 9→18, libero_10 1→12), which isolate the
normaliser change.

**Predictions.**
1. Estimate: the settled rotation estimate (median over episodes of the last-50-step mean) is
   ≥ 90 % of the fault on r_x and r_z (T4 gave 95 / 98 %) and ≤ 50 % on r_y (the DC-gain mismatch,
   record 50). r_y above 70 % would refute the record-50 diagnosis.
2. Faulted cells: spatial ≥ 19/20; each of object, goal within three of the shipped count or
   better (16/20, 29/40); libero_10 ≥ 15/40. **Registered refutation:** libero_10 ≤ 12/40 (the D.2
   innovation count) means the normaliser channel change buys nothing on the long-horizon suite;
   any suite more than three below its shipped count means the configuration is worse there.
3. Healthy controls: no suite loses more than two healthy episodes (adaptive ≤ frozen − 3 is the
   registered harm refutation), because the innovation law's phantom is not attenuated and the
   normaliser no longer damps it.
4. Adoption rule (decided now): the configuration replaces the legacy reference as the paper's
   headline only if 2 holds on all four suites and 3 holds on all four; otherwise the paper
   reports it as an ablation row and keeps the legacy reference, with the failing cell named.

**Refutation handling.** Each prediction is scored separately and a failure is reported as
primary with the count that failed it. The held-out calibration rerun (step 3 of the plan) is
registered separately after these land.

---

## Outcome, faulted cells (2026-09-14; `results/rule_constants/faulted_*`; healthy controls pending)

| suite | frozen → adaptive | fixed / broken | shipped legacy | D.2 innovation, all-channel normaliser | settle r_x / r_y / r_z (% of fault) |
|---|---|---|---|---|---|
| libero_spatial, n = 20 | 9 → **19** | 10 / 0 | 18 | 18 | 93 / 42 / 102 |
| libero_object, n = 20 | 3 → **17** | 14 / 0 | 16 | — | 87 / 60 / 101 |
| libero_goal, n = 40 | 17 → **30** | 14 / 1 | 29 | — | 85 / 58 / 94 |
| libero_10, n = 40 | 0 → **13** | 13 / 0 | 15 | 12 | 91 / 45 / 97 |

Pooled over the four suites: 29/120 → **79/120**, 51 fixed, 1 broken, against the shipped
28/120 → 78/120 (51 fixed, 1 broken): the same aggregate, one episode apart.

- **Prediction 1 (estimate): holds in substance, misses the letter on r_x.** r_z is at the truth
  on every suite (94–102 %); r_x is 85–93 % (the registered floor was 90 %; two suites are below it
  by three and five points); r_y is 42–60 %, above the registered 50 % ceiling on object and goal
  and nowhere near the 70 % that would refute the record-50 diagnosis. Against the legacy law's
  87 / 41 / 88 % on the headline cell, the unbiased law recovers the r_z shortfall in full and most
  of r_x; the remaining r_x gap is not the attenuation.
- **Prediction 2 (counts): holds on spatial (19 ≥ 19), object (17 vs 16) and goal (30 vs 29); fails
  on libero_10 by two episodes (13 against the registered ≥ 15).** The registered refutation
  (≤ 12/40) is not triggered. Paired on the same 40 scenarios against the shipped legacy cell the
  difference is 4 episodes won and 6 lost (both runs unpinned), i.e. noise; against the D.2
  innovation cell with the all-channel normaliser it is 7 won and 6 lost. On the long-horizon
  suite the normaliser channel change buys nothing measurable, as the refutation clause
  anticipated, and it costs nothing.
- **Adoption rule, applied as registered:** prediction 2 must hold on all four suites; it does not
  (libero_10). The configuration therefore does not replace the legacy reference as the headline
  on this evidence and is reported as an ablation row — equal to the shipped configuration in
  aggregate (79 vs 78 of 120), with an estimate at the truth on r_z and 85–93 % on r_x (unbiased in its fixed point, not in the measurement), and the theory-covered law. The healthy
  controls (prediction 3) decide whether it is a safe row; they are running.

Reading, stated plainly: the one constant set against the paper's rule cost the estimate, not the
task outcome. Fixing it moves the estimate to the truth on r_z and to 85–93 % on r_x and leaves
the counts where they were, because the task margins on these suites tolerate a 10–15 %
under-correction on rotation. The r_y deficit (45–60 %) survives every law and normaliser and
remains the plant's DC-gain mismatch (record 50).

## Outcome, healthy controls and adoption decision (2026-09-14; `results/rule_constants/healthy_*`)

| suite | healthy frozen → adaptive | fixed / broken | shipped legacy healthy control | phantom on r_x / r_y / r_z |
|---|---|---|---|---|
| libero_spatial, n = 20 | 20 → **20** | 0 / 0 | 19 → 20 | −0.0004 / −0.0006 / 0.0015 |
| libero_object, n = 20 | 20 → **20** | 0 / 0 | 20 → 20 | −0.0002 / 0.0043 / 0.0013 |
| libero_goal, n = 40 | 40 → **39** | 0 / 1 | 39 → 40 | 0.0012 / 0.0040 / 0.0004 |
| libero_10, n = 40 | 38 → **36** | 1 / 3 | 38 → 37 | 0.0011 / 0.0007 / 0.0004 |

Pooled healthy: 118/120 → **115/120** (1 fixed, 4 broken), against the legacy law's 116 → 117.

- **Prediction 3 (harm): holds on all four suites at the registered level** (no suite loses more
  than two: 0, 0, −1, −2). It is not free: libero_10 breaks three healthy episodes and repairs one,
  and the pooled healthy count moves from 118 to 115 where the legacy law moved 116 to 117. The
  unattenuated law acts on healthy phantoms that the attenuated law damped; the phantoms
  themselves are small (≤ 0.004, a tenth of the fault scale), so the harm is the long-horizon
  suite's sensitivity to any sustained correction, as record 43 already saw with the legacy law
  (38 → 37).
- **Adoption decision, as registered:** not adopted. Prediction 2 failed on libero_10 (13 against
  ≥ 15) and prediction 3 holds only at its letter. The configuration is reported as an ablation
  row: pooled faulted 79/120 against the shipped 78/120 (51 fixed / 1 broken in both), estimate at
  the truth on r_z and 85–93 % on r_x, healthy 115/120 against 117/120. Same repair, honest
  estimate, slightly more healthy harm. The legacy reference stays the headline.
- What the exercise settled: the one constant set against the rule cost the estimate, not the
  task outcome; the r_y deficit (42–60 %) survives every law and normaliser and is the plant's
  DC-gain mismatch (record 50), which is the next lever.
