# PREREG: FrozenYet Adaptive recovery campaign, deadline v1 (Panda / LIBERO-Spatial, pi0.5)

Registered 2026-09-16, before any fresh source was collected. Implements
`papers/frozen_yet_adaptive/EXPERIMENT_PLAN.md` (sections 2–7) and the scope decision in
`papers/frozen_yet_adaptive/RESEARCH_DECISION_20260916.md`. Campaign root
`results/frozen_yet_adaptive_deadline_v1/` (`configuration.json`, `used_state_ledger.json`, `partition.csv`,
`sampler_seeds_*.txt`, `stage2_manifest.json`, `stage0/`, `sources/`, `predictions/`, `qualification/`,
`physical_test/`, `reacting_policy/`, `analysis/`, `STATUS.md`). Chain: `scripts/re4/fya_chain.sh`.
Code: `openpi/re4_theory/fya_continuations.py` (driver), `fya_forecast.py` (fit / predict / calibrate /
evaluate), `fya_score.py` (measured costs and benefits), `fya_synthetic_check.py` (Stage 0 checks),
`fya_bundle.py` (configuration bundle); `openpi/adaptive_law.py --adapt-from` and
`openpi/error_signal.py --sampler-seeds` (Stage 2 and source collection).

## 1. Question

Is the physical benefit of the evaluated execution adapter (NT and innovation observers on the
rotation support) predictable before a fixed-command continuation is run, and how do an imposed
adaptation delay, a halved correction cap and a reversed fault sign change that benefit? Stage 2 asks
whether the delay pattern survives a reacting policy. Nothing here is a task-success or
robot-level stability certificate.

## 2. Stage 0 (existing data only; completed before this registration was committed)

| Item | Result | File |
|---|---|---|
| Retrospective source-level energies (E1 v2 Panda, E1 ALOHA) | reproduced by the writing session (`papers/frozen_yet_adaptive/evidence/retrospective_benefit_energy.json`): NT/innovation reduce mean translation energy by 40.9 % / 40.4 % on Panda, 9/10 and 8/10 sources improved | that file |
| Deployed predictor bundle | W = ridge FIR (K = 6) on the shipped healthy log (state 45), r_y tap sum .1028; M = probed diagonal (r_y .2759); γ .08, δ .008, ρ .15, all-channel normalisation, support {3,4,5}. Sha256 `7ee10b1f…69df`. This is the headline / E1 v2 calibration, NOT the E2 partition predictor U (.1458) nor the constrained C. | `configuration.json` |
| Healthy residual bias b_h under W (old E1 v2 runs, 1,800 healthy steps) | rotation channels ≈ 0 (r_y +.0005 / −.0001), translation z +.009; step sd r_y .009–.010; the all-channel residual norm (translation sd .02–.035) always exceeds the .008 deadzone | Stage 0 script output, record §63 |
| Residual model r = M f + (M − G_fit) c + b_h | mean error on the faulted branches −.0015 on r_y, rms .010 (noise level): the correction-direction feedback term is the right structure; the nominal-mismatch term is modelled as sampled healthy residual sequences | same |
| Scalar memory/authority/delay floor | formula `g [F S_N(λ) − C S_{N−τ}(λ)]_+` equals the exact LP minimax on a grid of λ ∈ {0,.5,.663,.9,1}, N ∈ {10,50}, τ ∈ {0,1,10,N/2,N}, C/F ∈ {0,.5,1,1.5}: max discrepancy 2.7e-11 | `stage0/synthetic_check.json` |
| Driver timing checks | delay: estimate zero and no update for k < τ, first update at k = τ, first nonzero correction at k = τ+1; reference corrects from k = τ with clip(f, −C, C); cap bounds every correction and estimate; sent = intended + correction; simulator input = sent + fault; histories advance during the delay | same |
| Snapshot fidelity, new driver | duplicate-restore gap 0.0 and fresh-prefix-replay gap 0.0 rad on the smoke sources; physics fingerprint identical from both routes | `stage0/dev_state39.log` |
| Length rule audit | checkpoint 30 + horizon 50 needs ≥ 80 commands: 63 of 70 old Spatial sources qualify (90 %) | Stage 0 output |

Stage 0 feasibility of the forecast (leave-one-state-out on the old sources, appended in §9 before
launch) decides only whether a certificate is *promised*; the campaign runs either way, with the
narrowed claim stated in §7.

## 3. Frozen conditions

LIBERO-Spatial, ten tasks, pi0.5 checkpoint as served by `ace_server.py` (hash in the server log),
scenario reset `libero-reset-v1`, ten warm-up steps, then the recorded healthy nominal commands.
Checkpoint at policy step **30**, horizon **N = 50** steps (2.5 s at 20 Hz). Fault on normalised
rotation-y command coordinate 4, F = .05. Correction support {3,4,5}. Caps are componentwise
projection boxes on the estimate (and hence on the correction).

| Scenario | Fault | Cap C | Delay τ |
|---|---:|---:|---:|
| `reference` | +.05 | .05 | 0 |
| `delay10` | +.05 | .05 | 10 steps (.5 s) |
| `cap_half` | +.05 | .025 | 0 |
| `sign_reverse` | −.05 | .05 | 0 |

Branches per source: for each scenario **off / nt / innovation / reference**; healthy **off / nt /
innovation** (cap .05, no delay) and a **duplicate healthy off**: 20 branches, 1,000 transitions per
source. Delay semantics: updates suppressed for transitions k = 0…τ−1 with the estimate held at zero
and the command histories advancing; the first enabled update at k = τ supplies the estimate used at
k = τ+1; the reference corrects from k = τ with clip(f, −C, C) on the support. The adapter command
u_k = a_k − c_k and the faulted simulator input u_k + f_k are logged separately; corrections are
never reconstructed from the faulted input. Saturation (|input| > 1) is logged as physical-model
error. The wrapper's done flag is recorded and not used as an outcome. No resets at replans (there
are none: fixed commands).

## 4. Sources, partitions, states, seeds

Used-state ledger: `used_state_ledger.json` (union of three scans over `results/`; conservative).
Free Spatial states: 4, 9–15, 17–19, 21–24, 29, 31. Allocation, frozen:

| Partition | States | Sampler seeds (schedule `base + 2·task + r`, r = state index) | Episodes |
|---|---|---|---:|
| qualification | 9, 10 | 81000 + 2t + r | 20 |
| physical test (locked) | 11, 12 | 82000 + 2t + r | 20 |
| Stage 2 bridge | 13, 14 | 83001 for every key | 20 keys |
| replacements (ordered) | 15, 17, 18, 19, 21, 22, 23, 24, 29, 31 | same schedule, next free base | as needed |

Collection order is `partition.csv` (state-major, task-minor). Eligibility: ≥ 80 recorded healthy
commands (checkpoint + horizon) and both fidelity gaps ≤ 1e-8 rad. An ineligible source is listed
with its reason; a replacement state is drawn from the ordered list only if fewer than **16** of 20
sources in a partition are eligible (expected exclusions ≈ 2 per partition are reported as
missingness instead, so the 20-source design is kept without re-collection unless it degrades below
16). Task success of a source is recorded and is not a selection criterion. No source enters the
fitting pool after its outcome is viewed; the qualification partition is used only for calibration.

## 5. Forecast, calibration and decision rules (frozen before qualification)

Model (`stage0/forecast_model.json`, fitted on old development data only: E1 v2 pass 1 (state 39)
and pass 2 (state 33) plus their replays under the new driver, `stage0/dev_state{39,33}.json`):
physical response e_k = Σ_{j<20} G_j d_{k−j} (translation deviation from the matched healthy
continuation, m; d = remaining disturbance f + c); observer forecast by simulating the deployed
laws on r_k = M f + (M − G_fit) c_k + b_h + ν_k with ν drawn as whole healthy residual sequences from
the development pool (64 draws, seed 2026, common random numbers across branches of a source).
Forecast trajectory Ê = mean over draws. Primary cost J_p = .05 Σ_k ‖e_k‖² (m² s), Q = .05 I,
radius ε in m√s. Angular energy J_r (SO(3) rotation vectors, rad² s) and endpoints are reported
separately.

Calibration (qualification partition): s_i = max over the 19 compared branches of ‖E − Ê‖_Q for
source i; **ε = max_i s_i** over the eligible qualification sources (the 90th percentile is reported
as a secondary width, not used for decisions). Predictions for the qualification sources are written
and hashed before the qualification replay runs; predictions for the test sources are written and
hashed before any test branch exists (`predictions/FROZEN_BEFORE_*.sha256`).

Test evaluation, per source, scenario and law A ∈ {nt, innovation, reference}:
L = max(0, ‖Ê‖_Q − ε)², U = (‖Ê‖_Q + ε)², predicted benefit interval [L_off − U_A, U_off − L_A];
label benefit / harm / inconclusive. Measured B = J_off − J_A; zero tolerance 1e-9 m² s (development
duplicate energies are exactly 0).

Descriptive campaign targets (not population guarantees): joint source coverage ≥ 18/20; decisive
NT-or-innovation prediction on ≥ 10/20 sources; ≤ 1 false-help source. All three are reported
regardless. An all-inconclusive forecast is reported as "no useful prospective certificate"; a
confident forecast that misses is reported as such.

Simpler predictive controls (selected on old data): forecast mean |f̂ − f| on the support and forecast
mean residual norm; evaluated by sign agreement with the measured benefit.

Measured contrasts (task-clustered bootstrap over the ten tasks, 10,000 resamples, seed 1909; the
source-level bootstrap is secondary): B per scenario and law; B(delay10) − B(reference),
B(cap_half) − B(reference), B(sign_reverse) − B(reference); J(scenario, law) − J(reference, law);
healthy harm J_p(healthy_nt), J_p(healthy_innovation).

Registered expectations (directional, from the theory note; refutation is reported as primary):
E1 delay10 increases the adapted cost relative to reference for both laws (integrator memory);
E2 cap_half increases the adapted cost and the reference (capped) branch retains ≥ half the off
deviation; E3 the linear forecast covers sign_reverse no worse than reference (a failure here
flags an asymmetry the model cannot represent); E4 the reference branch is below both online laws in
every scenario.

## 6. Stage 2 reacting-policy bridge

Twenty keys (tasks 0–9 × states 13, 14, sampler seed 83001 pinned per key across arms), pi0.5
reacting normally, `--scenario-reset`. Common unadapted prefix: policy steps 0–29 (env steps
10–39). At policy step 30 (env step 40): fault +.05 on r_y begins in the faulted conditions
(`--onset 40`), and the adaptive arms start correcting and updating (`--adapt-from 30`); the delayed
condition starts adaptation at policy step 40 (`--adapt-from 40`). Cap .05, γ .08, δ .008, ρ .15,
all-channel normalisation, support {3,4,5}, same plant log and M as the bundle. Arms: off / NT /
innovation under healthy, faulted and faulted+delayed conditions. The faulted off arm is identical
under the delayed condition and is **aliased**, not re-run (160 policy episodes instead of the plan's
180; the alias is stated in the manifest). Prospective arm order: healthy off, nt, innovation;
faulted innovation, off, nt; delayed nt, innovation. Outcomes: success per key, paired wins/losses
(exact McNemar, descriptive), individual healthy regressions, cap/gate activity, estimate transients,
telemetry, runtime. Intention-to-treat: a key that terminates before step 30 is reported as unexposed.
Stage 2 supports only an empirical pattern claim; fixed-command bounds are not carried over.

Registered expectations: S1 faulted NT and innovation each exceed faulted off (the headline effect at
cap .05 from a mid-episode onset); S2 delayed adaptation is not better than immediate adaptation (the
direction of the memory argument; a tie is possible at this n); S3 healthy NT/innovation lose no more
than three individually successful off keys each.

## 7. Stop rules and claims

If the Stage 0 leave-one-state-out check (§9) or the qualification calibration yields ε so wide that
no test interval can be decisive, the paper reports the forecast as inconclusive and keeps the measured
delay/cap/sign effects and the Stage 2 pattern under a narrowed claim. No refit on the test partition,
no arm removal, no retuning, no sample extension inside the locked study. Every count, including
failures, is reported.

## 8. Budget and provenance

40 fresh policy episodes (sources) + 800 branches × 50 steps (+ 40 fresh-prefix fidelity
continuations) + 160 policy episodes (Stage 2). Hashes: bundle, model, predictions, sources
(`sources/SHA256SUMS`), driver and `adaptive_law.py`. GPU 1 shared with another user's training
(never interrupted); policy server port 8000.

## 9. Stage 0 forecast feasibility (old sources; appended 2026-09-16 13:50, before any fresh source was collected)

Leave-one-state-out on the replays of the old sources under the new driver (`stage0/dev_state39.json`,
9 eligible sources; `stage0/dev_state33.json`, 8 eligible; one and two sources excluded by the length
rule). Fold A: fit on state 39, predict and calibrate on state 33. Fold B: the reverse. Cross-folds apply
one fold's width to the other's predictions. Files `stage0/fold{A,B}_*`, `stage0/cross{AB,BA}_eval/`.

| Quantity | Fold A | Fold B |
|---|---:|---:|
| ε, max rule (m√s) | .0194 | .0208 |
| affine rule a / ρ | .0082 / 2.44 | .0065 / 1.65 |
| median ‖Ê_off‖_Q vs measured ‖E_off‖_Q (m√s) | .0068 / .0098 | .0086 / .0095 |
| median relative trajectory error ‖E − Ê‖/‖Ê‖ (non-healthy) | 1.04 | 0.81 |
| joint source coverage (own width) | 8/8 | 9/9 |
| decisive intervals, either rule | **0** | **0** |
| median forecast error of B (m² s) vs median B ≈ 6e-5 | −2.8e-5 | −1.2e-5 |
| sign accuracy of the estimator-error control (nt / innovation) | .91 / .88 | .94 / .92 |

**Outcome of the Stage 0 stopping rule.** The trajectory-radius certificate is infeasible with this
model: per-trajectory errors are of the order of the prediction itself, so every benefit interval
contains zero under both width rules, on both folds, even though the point forecast of the energy
difference has small median error and the sign of the forecast agrees with the measured sign on ≥ 88 %
of cells. Per EXPERIMENT_PLAN.md §2 (stopping rule) and RESEARCH_DECISION §A, the paper will **not
promise a certificate**. The campaign runs unchanged under the following narrowed, frozen claims:

1. The interval labels are still computed and reported for the locked test under both width rules
   (`--width max` and `--width affine`, calibrated on the fresh qualification partition); the registered
   expectation is that they remain inconclusive. Meeting the "≥ 10/20 decisive" target would be a surprise
   and is reported as such.
2. The prospective **point forecast** of B (frozen before the test replay) is evaluated by (a) sign
   agreement with the measured B on cells whose |B| exceeds the zero tolerance, target ≥ 80 % over the
   nt and innovation cells of the four scenarios (240 cells minus exclusions), and (b) the median of
   |forecast error| / |B|, reported without a target. The estimator-error control is evaluated the same
   way. Neither is a certificate; both are calibrated on the qualification partition first (the
   qualification report states their accuracy there before the test is opened).
3. The measured delay / cap / sign contrasts (E1–E4 in §5) and the healthy harm are the primary
   fixed-command results; Stage 2 is the empirical bridge under §6 and §7.

Development observations on the old sources (not results; they inform the wording of E1–E4 only):
delay10 raised the adapted cost on state 39 (J contrast +1.6e-5 [1e-6, 3.5e-5] m² s, both laws) and
directionally on state 33 (CI includes zero); cap_half raised the reference's cost on both states
(+2.8e-5 [1.7e-5, 4.1e-5]; +7.6e-5 [2.8e-5, 1.4e-4]) but barely the online laws, whose estimates settle
near .025 anyway; the sign reversal changed neither the benefit fraction (≈ .54 vs .64/.73) nor the
reference's exact cancellation; healthy false updates cost J_p ≈ 1.1–2.1e-5 m² s per source, 15–20 %
of the faulted off energy.

Chain launched after this section and the code (`fya_forecast.py --width`) were committed.

**Execution note (13:52, before any Stage 1 outcome was read).** The user allowed both cards; the chain was split without changing any registered setting: Stage 1 continues on GPU 1 (`scripts/re4/fya_chain_stage1.sh`, same server and seeds) and Stage 2 runs concurrently on GPU 0 (`scripts/re4/fya_stage2_gpu0.sh`, server port 8001, rendering on GPU 1). Stage 2 therefore does not wait for the Stage 1 qualification result; under §9 it runs as the narrowed empirical bridge in either case.

**Correction (14:10–14:17, before the qualification run file existed and before any test forecast).** Review
found that `fya_forecast.py` added the healthy residual mean b_h to noise sequences that already carry it,
doubling the expected healthy residual in the observer simulation (rotation channels ≈ 1e-3, translation
z ≈ .018 after doubling; it enters the all-channel normaliser). Fixed by dropping the explicit b_h term.
The §9 feasibility folds and the first qualification forecast used the doubled bias; those files are kept
as `predictions/qualification_predictions_v1_doublebias.*` and the §9 numbers stand as recorded (the
fold conclusions do not change qualitatively: the bias is small next to the sampled residual sequences).
The qualification forecast was regenerated with the corrected code at 14:16:51 from the nominal commands
only, while the qualification replay was still writing; no replay output was read. Hashes in
`predictions/FROZEN_BEFORE_QUALIFICATION.sha256` now include the corrected script. The test forecast is
written by the chain with the corrected code. The forecast remains a reduced-loop simulation (deployed
update function on a simplified residual model), not an exact simulation of the deployed FIR loop; the
docstring now says so.

## 10. Outcome, Stage 1 (locked physical test scored 2026-09-16 14:33; `analysis/test_score/`, `analysis/test_evaluation{,_max}/`)

**Sources.** 40/40 fresh episodes succeeded. Qualification: 18/20 eligible by length (task 0 at states 9 and 10:
75 and 70 commands), one more (task 3, state 10) invalid by the fidelity rule → **17** calibrated. Test: 20/20
eligible, one (task 7, state 12) invalid by the fidelity rule → **19** scored. Both fidelity failures are of the
same kind: duplicate-restore gap exactly 0 and identical physics fingerprints at the checkpoint, but the
fresh-prefix replay diverges over the continuation (2.3e-6 and 1.5e-4 rad max joint gap): solver warm-start
sensitivity on a contact-rich continuation, not an indexing error. They are excluded per the 1e-8 rule and
reported; no replacement was drawn (≥ 16 per partition).

**Forecast (frozen before the replays; corrected code).** Interval certificate: **0 decisive intervals** under
both width rules, as registered in §9. Coverage 19/19 (affine, a = .010, ρ = 3.83) and 16/19 (max, ε = .0257:
the three misses are all `sign_reverse` cells of sources 7, 8, 18). Point forecast of B: sign agreement
**nt .882, innovation .868, reference .947** (target ≥ .80 met for both laws; on the qualification partition .90 /
.88 / .99), median |error|/|B| .69 / .70 / .66. The estimator-error control reaches .882 / .868, the same as the
energy forecast. Median relative trajectory error stays ≈ 1.0–1.5, so the forecast ranks and signs but does not
bound.

**Measured (19 sources, task-clustered 95 % intervals, J_p in m² s).** J_off median 1.19e-4 (endpoint 13.7 mm).

| Scenario | B nt | B innovation | B reference | sources + / − (nt) | rel. reduction nt / innov / ref |
|---|---|---|---|---|---|
| reference | 8.8e-5 [4.7e-5, 1.3e-4] | 9.0e-5 [4.8e-5, 1.3e-4] | 1.47e-4 [1.0e-4, 1.9e-4] | 18 / 1 | .61 / .62 / 1.00 |
| delay10 | 6.1e-5 [2.9e-5, 9.2e-5] | 6.2e-5 [3.1e-5, 9.2e-5] | 1.07e-4 [5.4e-5, 1.5e-4] | 15 / 4 | .45 / .41 / .78 |
| cap_half | 7.7e-5 [4.4e-5, 1.1e-4] | 7.5e-5 [3.8e-5, 1.1e-4] | 9.8e-5 [5.3e-5, 1.4e-4] | 18 / 1 | .55 / .58 / .75 |
| sign_reverse | 1.62e-4 [4.8e-5, 3.3e-4] | 1.63e-4 [4.6e-5, 3.3e-4] | 2.47e-4 [1.2e-4, 4.4e-4] | 16 / 3 | .59 / .64 / 1.00 |

Endpoints (median, mm): off 13.7 → nt 7.8 / innovation 7.8 / reference 0.0 (reference scenario); delayed
reference 2.7; capped reference 5.7. Healthy false-update cost J_p: nt 1.22e-5 [5.0e-6, 2.1e-5], innovation
1.44e-5 [5.8e-6, 2.5e-5] (≈ 10–12 % of the faulted off energy). No saturated step in any branch.

**Registered expectations.**
- **E1 holds.** delay10 raises the adapted cost: J(delay10) − J(reference) = +2.7e-5 [9.5e-6, 4.4e-5] (nt),
  +2.8e-5 [1.1e-5, 4.7e-5] (innovation), 18/19 sources each; the delayed reference loses +4.0e-5 [2.2e-5, 6.9e-5].
- **E2 holds for the cost, fails for the retained fraction.** cap_half raises the adapted cost: +1.05e-5
  [1.8e-6, 1.9e-5] (nt), +1.5e-5 [5.3e-6, 2.5e-5] (innovation), 14/19 sources; but the capped reference retains
  a **quarter** of the off energy (median J/J_off .25; endpoint .42 of off), not "≥ half": halving the remaining
  disturbance quarters the quadratic cost. The clause was mis-specified in energy units and is recorded as failed.
- **E3 fails.** The linear forecast covers sign_reverse worse than reference: the three max-rule misses are all
  sign_reverse cells, the adaptive branches' relative trajectory error rises from 1.24–1.29 to 1.48–1.53, and the
  measured off cost itself is 30 % larger under −.05 than under +.05 (1.57e-4 vs 1.19e-4): the physical response
  to a rotation-y bias is not sign-symmetric at this magnitude.
- **E4 holds** on reference and sign_reverse (19/19 sources), 18/19 on delay10, 16/19 on cap_half (sources 7, 15,
  19: with the cap binding, an online law that settles near .025 anyway can match the capped reference).

Stage 2 outcome follows in §11.

**Caveat on the point-forecast criterion (added 14:50 after inspecting `figures/fya_forecast.pdf`).** The
registered sign target is met, but trivially: the frozen forecast of B varies only between 3.0e-5 and 4.6e-5
across the 19 sources (sd 5e-6) while the measured B has sd 1.5e-4, the correlation between forecast and
measured B is 0.19, and the sign agreement (.882 for NT) equals the base rate of positive benefit exactly
(an "always benefit" rule scores .882). The forecast does not condition on the source's nominal commands
(the physical model is driven only by the remaining disturbance, which the reduced loop reproduces almost
identically for every source), so it cannot discriminate sources. The registered ≥ .80 target was
mis-specified without a base-rate comparison; the honest reading is that neither the interval certificate
nor the point forecast predicts *which* source benefits, and the same holds for the estimator-error control.
What the campaign establishes is the measured effect structure (E1, E2 cost, E4) and the sign asymmetry (E3).

## 11. Outcome, Stage 2 (reacting-policy bridge, GPU 0, complete 15:16; `reacting_policy/`, `analysis/stage2_summary.json`, `analysis/stage2_telemetry.json`)

All 160 policy episodes ran (8 arms × 20 keys; the faulted off arm is aliased for the delayed condition as
registered). Telemetry confirms the protocol: fault live from env step 40 (policy step 30), estimate exactly zero
before enablement in every adaptive arm, delayed arms enabled at policy step 40.

| Condition | off | NT | innovation |
|---|---:|---:|---:|
| healthy | 20/20 | 19/20 (lost task 8, state 14) | 20/20 |
| faulted (+.05 r_y from step 30, cap .05) | **20/20** | 20/20 | 20/20 |
| faulted + 10-step adaptation delay | 20/20 (alias) | 20/20 | 20/20 |

**The registered fault does not degrade task success for the reacting policy on Spatial**: the frozen policy
completes every key with the r_y bias live for the last ≈ 78 policy steps of each episode. All paired
comparisons are 0 wins / 0 losses; the bridge is at the ceiling and carries no task-level information about
delay or adaptation.

- **S1 fails** (faulted NT / innovation do not exceed faulted off: equal at 20/20).
- **S2 holds trivially** (delayed = immediate = 20/20).
- **S3 holds** (healthy losses: NT 1, innovation 0).

Estimator behaviour (telemetry, medians over keys): faulted arms reach half the fault 8.5–9 steps after
enablement and settle at r_y ≈ .036 [.031, .044], 72 % of the fault, spending 1–4 % of enabled steps at the
cap; the delayed arms settle at the same level. Healthy arms carry a **phantom r_y estimate of ≈ .015**
[.005, .021] under the reacting policy, a third of the fault and far above the fixed-command healthy residual
bias (≈ 5e-4): replanning-induced command changes that the FIR does not predict are attributed to a fault.
Episode lengths 103–108 policy steps (median), no unexposed key.

**Reading.** A single-channel rotation bias of .05 introduced mid-episode is inside the policy's tolerance on
this suite (the headline uniform six-channel bias from step 0 is not: 8/20 frozen). The bridge therefore does not
test the delay pattern at task level; a larger or multi-channel mid-episode fault would be needed, and that is a
new registration, not an extension of this one. What Stage 2 does establish is the healthy phantom magnitude
under replanning and the unchanged 20/20 healthy innovation / 19/20 NT under a .05 cap.

Campaign closed 15:20. Both servers stopped; no further runs.
