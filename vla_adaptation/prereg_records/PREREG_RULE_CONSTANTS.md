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
