# Draft and experiment audit — 2026-09-08

The draft needed updating. It now distinguishes the historical command-interface
results from the completed physical-fault confirmations and the new estimator
tuning study. **There is no reason identified by this audit to rerun the 2,220
new physical-fault confirmation rollouts or the 5,424 estimator evaluation
rollouts merely to report their results.** Stronger causal, held-out-transfer or
method-ranking claims require additional evidence.

## What needs new experiments

| Priority and purpose | What to run | Why existing results are insufficient |
|---|---|---|
| Recommended next: isolate composite tracking feedback | Freeze the selected composite configuration and compare tracking on/off with identical calibration, Q/R, P0, damping, clipping, mask and reference. Include an otherwise identical zero-damping control to isolate damping. Use fresh scenario seeds, healthy and all declared fault conditions, a fixed primary score and seed-clustered analysis. | The selected composite and Kalman differ in bandwidth, damping and clipping. Their comparison answers which selected configuration performed better, not the causal contribution of tracking. The completed result is unresolved, not evidence that composite must win. |
| Required for a **fresh held-out shared-calibration LIBERO headline** | Collect one declared healthy spatial FIR/M calibration on disjoint scenarios, freeze it across all four suites and the claimed backbones, then evaluate healthy/off/corrected on untouched scenarios with current full-state reset/source checks and fixed masks/parameters. | The historical M protocol probes task 0/init 45, which appears in the spatial evaluation. Exact original FIR scenario provenance is not embedded. Fresh Panda confirmation instead calibrates separately per suite, so it cannot validate unchanged-calibration transfer. |
| Required if retaining historical friction/lock efficacy as confirmed evidence | Rerun the elbow friction +10/+20 and ±0.05-rad lock cells with healthy/off/corrected controls, fault restoration on every exit, and model/full-state fingerprints before every episode. Expand to other joints only for broader claims. | The historical cached-environment runner could leave friction and joint limits active. Recreating a fault from that modified model can compound nominal severity; archived outcomes lack snapshots to exclude this. The new torque confirmations do not validate these old model-changing faults. |
| Required if replacing old ALOHA tables with the **new settings** | Rerun the claimed old severity and continuous/identify-then-hold comparisons using one fresh declared calibration and controlled configurations. Isolate all-channel versus corrected-channel gating/normalization with other parameters fixed if attributing benefit to gripper exclusion. | Old ALOHA used all-channel normalization and historical calibration. Selected `legacy_04` uses corrected channels **and** different gain/clip settings. The runner default remains `all`. Adding the option or changing prose cannot update historical measurements. |

These are requirements **for the specified claims**, not reasons to erase valid
historical counts. The revised draft labels historical settings and limitations;
it does not present these unperformed experiments as complete. This audit did not
launch new jobs.

## What needs only correction or reanalysis

- Keep the reproducible historical four-suite **28/120 → 78/120** result, with
  calibration/reset limitations. Its 51 rescues/1 regression give exact
  McNemar p = 2.3536728e-14; all four suite contrasts survive Bonferroni-4.
  Remove the undocumented **690 paired / 275 fixed / 7 broken** aggregate until
  a complete inclusion manifest establishes its denominator and reuse rules.
- Keep the already corrected GR1 comparisons: continuous **8/30 versus its own
  frozen 1/30**, Fisher p = .02569. The later pooled-control comparison is
  explicitly unpaired and retrospective; **21/30 held versus 64/90 healthy**
  does not establish equivalence. Fresh same-scenario GR1 confirmation would
  strengthen the claim but is unnecessary to reproduce these labeled results.
- Keep historical ALOHA **15/40** identify-then-hold and **17/40** healthy,
  p = .75390625. The former includes one failed identification episode and
  39 held episodes; zero variation applies only after identification.
- Describe legacy normalization as scaling the **target**, with an attenuated
  fixed point; below its gate the estimate decays. The gripper commands and
  measured positions are continuous-valued. For the refitted eight-log healthy
  FIR after excluding the first six steps per episode, right-gripper R² is
  .768, squared-residual share 93.63%, and corrected-left-six share .091%.
  The earlier 93.8%/.089% decomposition uses a different window convention.
  None of these decompositions identifies the cause of a task failure. The
  [reanalysis script](verify_aloha_residual.py) and [hashed receipt](aloha_residual_receipt.json)
  reproduce the in-sample figures using the actual `aloha_adapt.fit_plant` on
  2,205 samples. Raw-coordinate shares depend on the arm/gripper units.
- Replace the repairability “proposition” by a diagnostic hypothesis. Variation
  alone ignores bias, direction, timing and task dynamics. Fitted-reference
  contraction is not a proof for the complete healthy robot/VLA closed loop.
- Add the matched-residual pilot (19/17/17/18/15/20 successes), while retaining
  that the raw-error integral comparison cannot isolate the update law. All
  estimated pilot arms retain projection.
- Pairing on a common physical scenario remains useful with independent policy
  sampling. Rescues/regressions are observed differences, not deterministic
  counterfactuals under identical policy randomness. Historical nominal keys
  cannot retrospectively certify full physical-state pairing.

## Completed evidence now included in the draft

**Physical faults:** Panda confirmation has 1,440 rollouts across all seven
joints, three suites and healthy controls. The registered joint-5 comparison
improves **15/60 legacy → 41/60 weighted**, p = 2.556e-6; off is 36/60. Weighted
harms Object joint 3 versus legacy (**16/20 → 2/20**, Bonferroni-24 p = .01245).
The separate 780-rollout ALOHA weighted transfer test covers all twelve arm
joints and healthy controls: faulted totals **60/240 off, 70/240 legacy,
66/240 weighted**, with no resolved Bonferroni-13 cell. Healthy weighted falls
from off 11/20 to 6/20. This ALOHA study reused historical sensitivity and
does not establish fully held-out calibration or composite transfer.

**Estimator tuning:** 2,608 search + 896 validation + 1,920 confirmation =
5,424 evaluation rollouts. Fresh ALOHA fit/qualification seeds 3100–3129 are
separate from search 3200–3203, validation 3300–3307 and confirmation 3400–3429.
The valid scope is Transfer Cube, left-six correction, healthy plus seven fault
conditions. Confirmation balanced scores are original **50.00%**, RLS **43.33%**,
DOB **42.14%**, Kalman **36.90%**, composite **35.95%**, integral **31.90%**.
Composite minus Kalman is **−0.95 percentage points**, whole-seed 95% bootstrap
interval **[−9.52, +7.62]**, exact p = .86261. This is not an established total
ranking. Current full selected parameters, calibration and analysis families
are now stated in the draft; old tables retain their original settings.
The sign-flip test's exactness assumes whole-seed exchangeability/sign symmetry;
it is not an exact test for every arbitrary zero-mean distribution.

The composite tracking increment concentrates over 99.39% of its squared energy on corrected coordinates
on joint 1 in every confirmation condition. It lowers reference-position RMS
without establishing improved task success. This motivates the matched ablation
and direction-specific modeling investigation; it does not diagnose a proven
bug or justify changing the confirmation result.

## Further studies, rather than reruns of completed work

- A calibrated **Q=0 static Kalman** variant needs information-growth
  qualification; it was omitted by the current finite bank's eligibility rule.
  RLS with forgetting one does not make this the same calibrated comparison.
- Directional tracking preconditioning or a new controller-response model changes
  the method. Qualify it first, tune on development data and use fresh confirmation.
  Reusing current confirmation to choose it would turn those data into development.
- Broader composite claims require other qualified joints, ALOHA tasks, robots
  and policy backbones. The twelve-joint ALOHA reference failed qualification;
  the all-joint weighted experiment is not a composite experiment.
- A claim that weighted improvement comes specifically from rotation support or
  regularization needs controlled support-by-allocator comparisons. The current
  candidate changes both. Physical authority probes already show why authority
  alone is insufficient; more torque episodes cannot replace this ablation.
- The real-robot paired repair study is still pending. Existing X2 diagnostic
  logs support their stated detection analysis, not completed hardware repair.

## Sources and manuscript changes

- [Updated draft](iclr_draft.tex) and [compiled PDF](iclr_draft.pdf).
- [Completed tuning and diagnosis](../report/COMPOSITE_TUNING.md),
  [frozen protocol](../results/composite_tuning/PREREGISTRATION.md),
  [independent confirmation statistics](../results/composite_tuning/audit/confirmation_independent_statistics_v1/README.md).
- [Physical-fault follow-up](../report/FOLLOWUP_STATUS.md),
  [mechanism investigation](../report/JOINT_FAULT_MECHANISM.md),
  [Panda confirmation](../results/joint_followup/confirmation_plan/analysis/summary.md),
  [final ALOHA weighted confirmation](../results/composite_followup/aloha_weighted_confirmation_plan/analysis_final/summary.md).
- Historical sensitivity protocol: `git show f9d27fb:openpi/openloop_id.py`
  (task 0/init 45); archived `results/phase05/openloop.json` and
  `results/phase05/error_signal_so3.json` have incomplete scenario provenance.
- New source: [composite recursion](../openpi/composite_observer.py),
  [ALOHA runner](../openpi/aloha_adapt.py),
  [Panda runner](../openpi/run_joint_followup.py).

The update revises the abstract, contribution claims, law semantics, calibration
and pairing caveats, physical-fault and ALOHA interpretation, limitations and
reproducibility statement. It adds the implemented composite recursion, tuning
protocol, confirmation results and complete all-joint/settings tables. It does
not claim a fully audited submission-ready literature section; bibliography
metadata and final paper length still need an editorial pass.

Subsequent manuscript revision: the paper now has seven main-text pages and 26
pages total, with proofs and extended results in the appendix. The final build
has no undefined references/citations, overfull boxes or BibTeX warnings. The
previous missing-author metadata warnings were corrected in the
[reference audit](REFERENCE_AUDIT.md). See the [ICLR revision notes](ICLR_REVISION_NOTES.md)
for the new framing, conditional theory and reproducible layout checks. These
editorial/theoretical changes do not replace the experiments identified above.
