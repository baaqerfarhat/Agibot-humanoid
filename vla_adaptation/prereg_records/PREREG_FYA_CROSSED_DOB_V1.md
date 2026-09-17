# PREREG: matched NT and DOB crossed matrices under the stronger fault (recovery study P3), v1

Registered 2026-09-17 (committed before any pilot or matrix cell was replayed). Implements
`papers/frozen_yet_adaptive_recovery_study/EXPERIMENT_PLAN.md` §5. Root `results/fya_recovery_study_v1/stronger_dob/`.

**Sources (archived, inspected).** Reference: the original stronger-fault healthy off arm
(`results/fya_stronger_fault_v1/evaluation/eval_healthy_off`). M0 / M1: the DOB comparator's in-process faulted trio
(`results/fya_dob_comparator_v1/evaluation_v2/fault_off`, `fault_nt`, `fault_dob`; α = .08, mask {3,4,5}, cap .05,
+.10 on all six coordinates from policy step 30). Symlinked bundle `sources/`; extractions `bundle_nt/` (audit sha
1c6d994ea1c69bff) and `bundle_dob/` (audit sha 05b8cd5056fe4a1a), each with the configuration from its own M1 header (NT: law legacy,
baseline none; DOB: law legacy, baseline dob, γ = α = .08, bias none). **29 keys eligible for both matrices**
(identical sets): excluded are 8/18 (M1 prefix mismatch, both laws), 2/19, 5/19, 6/19, 9/19 (healthy reference
prefix mismatch), 0/19, 0/21, 0/22, 3/19, 3/21, 3/22 (fewer than 80 steps). Task outcomes of the subset (reference
29/29, off 2/29, NT 9/29, DOB 8/29) are a subset; **the task comparison stays NT 14/40 vs DOB 14/40, 1 win / 1 loss**.

**Implementation added before registration.** `fya_crossed_replay.py --matrix dob` (sha f88ab2229e8c1da3) runs the causal
update with `baseline="dob"` (`clip((1−α) f̂ + α M⁻¹ r)`, no gate, no attenuation) and refuses a non-DOB header; the
extractor records baseline/bias; a synthetic check confirms the DOB recurrence equals its closed form and differs
from NT under the same residuals. **Pilot (registered):** the DOB diagonal cells (ref, J00, J11, fresh routes) on the
first two eligible keys of different tasks must reproduce the archived fault_dob path within the registered
tolerances (1e-8; corrections ≤ 1e-8) before any crossed cell is collected; a failure stops P3 and is reported.

**Protocol.** As in `PREREG_FYA_CROSSED_STRONG_V1.md` (prefix 30, window 50, fault +.10 × 6, seven branches, cell
order rotated by key index, fidelity rules, continuation semantics); two full matrices, NT and DOB, on the same
29 keys with the same M0 stream and reference. Scoring per matrix: `fya_crossed_score.py` (sha 8aaa407b195872cb; R1 primary,
margin 1e-5, seed 20260918; the scorer now also reports S = D0 − R1). **Primary of P3 (paired across matrices,
common fidelity-valid keys): T_NT − T_DOB** (total path benefit difference), equal-task mean, task-clustered
bootstrap 10,000 draws seed 20260918, margin 1e-5 m² s; `fya_pair_matrices.py` (sha 78d06b63c380133f). Secondary paired
differences: D0, R1, I, J10, J11, J01. No sign expectation is registered: task totals agree, so a resolved physical
difference in either direction is the informative outcome; an unresolved difference is reported as such and does
not establish equivalence. Chain `scripts/re4/fya_strong_dob_chain.sh` (starts after P1 has finished; CPU replay
only). Outcome appended below.

## Outcome (pilot 02:57; matrices 03:00–03:25; `pilot/`, `runs/`, `analysis_nt/`, `analysis_dob/`, `analysis_pair_nt_minus_dob.json`)

**Pilot passed.** DOB diagonal cells on keys 1/18 and 2/18 reproduce the archived fault_dob path exactly (ref, J00,
J11 including corrections and estimates, fresh routes: all gaps 0.0), so the DOB replay path is verified before any
crossed cell was read.

**Fidelity.** Both matrices: 27 of 29 keys exact; the same two keys fail as in P1 (task 7 state 18 and task 8
state 22, live-process divergences, fresh and restored routes agreeing), excluded from both. Common
fidelity-valid set: **27 keys across all ten tasks**.

| Matrix (27 keys) | R1 | D0 | T | I |
|---|---|---|---|---|
| NT | 1.29e-3 [−2.8e-3, 5.1e-3] unresolved | 1.4e-4 [−1.2e-4, 4.0e-4] | 1.43e-3 [−2.9e-3, 5.4e-3] | −2.2e-4 [−4.9e-4, 1.8e-5] |
| DOB | 2.60e-3 [7.5e-5, 5.6e-3] resolved positive (lower limit at the margin's scale) | 9.6e-5 [−1.9e-4, 3.9e-4] | 2.69e-3 [4.2e-5, 6.0e-3] | −9.7e-5 [−3.2e-4, 1.1e-4] |

**Primary, paired T(NT) − T(DOB): −1.27e-3, 95 % interval [−3.5e-3, +6.2e-5], 9 keys positive / 18 negative →
unresolved.** Secondary paired differences: R1 −1.31e-3 [−3.6e-3, −7.5e-6] (8/19), I −1.3e-4 [−2.6e-4, −1.5e-5],
D0 +4.3e-5 [−5e-7, 7.3e-5] (17/10), J10 −4.3e-5, J11 +1.27e-3 [−5.6e-5, 3.5e-3], J01 +1.18e-3 [−4.7e-5, 3.4e-3].

**Reading.** With identical task totals (14/40 each), the two laws' physical decompositions are close: NT's direct
term is marginally larger (its J10 is 4e-5 lower), DOB's stream term marginally larger (the DOB-generated stream
lands about 1.2e-3 closer to the healthy reference than the NT-generated one, on 19 of 27 keys), and the registered
primary total-benefit difference is not resolved (its upper limit sits at +6e-5, essentially zero). These are
marginal, heavy-tailed differences under the task-breaking fault where, as P1 showed, the fifty-step window resolves
little; they do not establish a mechanism difference between the laws and do not alter the task comparison. Both
matrices are reported; neither was selected.
