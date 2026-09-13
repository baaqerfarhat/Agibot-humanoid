# ICLR manuscript revision — 2026-09-08

The revised [manuscript](iclr_draft.tex) has **9 pages of main text** and a
**30-page complete PDF**, including excluded statements, references and appendices.
The [official ICLR 2027 author instructions](https://iclr.cc/Conferences/2027/AuthorGuidelines)
set a nine-page initial main-text limit, excluding references and appendices.
The local style, bibliography style and
supporting template files match the official downloadable files byte for byte.
No margins, font sizes or line spacing were changed to meet the limit.

## Oral/award writing study and resulting revision

The [complete ICLR 2023–2026 writing study](style_study/ICLR_WRITING_STUDY.md)
contains 793 indexed papers, all 14 publicly named Outstanding Paper winners and
20 honorable mentions, a structural pass over every PDF, and 34 targeted writing
profiles. The [PDF index](style_study/CORPUS_INDEX.md) links the downloaded copies.
The corpus contains 23,618 PDF pages and 5.78 GB of indexed PDF bytes; bulk files
remain outside git in `/home/fengze/online_adaptation_research/iclr_2023_2026/`.
Private candidate/shortlist identities are not inferred. The 2023 alternate and
archived versions are labeled separately from official 2024–2026 proceedings.

The resulting manuscript revision leads with the gap between identifying a
disturbance and correcting its effect. The introduction gives the motivating
counterevidence and maps contributions to verifiable results. Related work is
organized around adaptation location, learned information, closest physical-fault
evidence and control objectives. The method distinguishes fixed offline fits
from evolving online state and annotates the composite prediction/tracking terms.
Experiment headings now communicate the findings, with diagnostics distinguished
from task outcomes. Implementation debugging details already documented in the
tuning appendix are removed from the main tracking paragraph.

Independent review checked the revised claims and equation, retained calibration
and comparison qualifications, and inspected the rendered main figures/tables.
All four main figures retain their original data and reproduction scripts;
their explanatory roles are documented in the study. No experimental results,
proof claims or citation metadata were changed to strengthen the narrative.
The final build remains exactly nine main pages, with backmatter starting at
page 10 and 30 pages in the complete PDF.

## New main-branch changes reconciled

The merge includes main commits `c367fb2` and `96555f3`: GR1 scene-generator
reseeding, its documentation, and the smaller demonstration video. The audited
manuscript and historical unpaired Fisher comparisons are retained. The GR1
appendix now states the recorded innovation update and settings, and distinguishes
future reset-controlled comparisons from the historical cohorts. A nested-wrapper
mock checks the actual reset function; it does not validate the full simulator.
The [merge receipt](../docs/merge_audit_gr1_reset_20260908.json) records this limit.

The appendix convergence figure is regenerated from stored LIBERO, ALOHA and
GR1 traces, with readable panel titles and explicit source provenance. The GR1
90–112% range is the final-50-step held average from one identification episode,
not a claim that every coordinate has converged by step 100. These historical
traces and selected video clips add no new benchmark outcomes.

## Scientific framing

The title is now **From Residual Identification to Task Repair in Frozen
Vision–Language–Action Policies**. The main paper addresses three distinct
requirements: sufficient residual information, sufficient correction authority,
and stable estimator–controller interaction. Successful task execution adds a
separate trajectory-robustness requirement.

This framing explains the implementation and both favorable and harmful results.
It does not present a new name for an existing observer as the contribution,
assert that composite must beat Kalman, or imply that weighting fixes every
physical fault. J-PARC's earlier empirical distinction between local correction
and task recovery is acknowledged. The more specific contribution is the explicit
conditions and mismatch bounds, their connection to this implementation, and the
new controlled confirmations.

## Theory added

| Result | Meaning and limit |
|---|---|
| Linear cancellation characterization | A rule based on `z=Hf` can cancel `Gf` through `Dc` exactly iff `ker(H) ⊆ ker(G)` and `range(G) ⊆ range(D)`. With bounded inputs, the particular required response must also be feasible. This is an algebraic local characterization, not a nonlinear task theorem. |
| Correction-authority floor | A weighted projection separates missing input directions from losses due to correction bounds. Full parameter identification does not eliminate either. |
| Physical improvement under map/proxy uncertainty | Bounds distinguish the nominal allocator objective from actual physical error. A stronger inequality compares correction against the same uncertain baseline. Required uncertainty bounds are not certified by the current controller. |
| Exact composite recursion and conditional stability | The analysis respects old-estimate actions, next-position feedback, posterior covariance, damping, time-varying faults and projection. A common quadratic metric yields a disturbance bound; individually stable frozen matrices do not establish that condition. |
| Counterexamples | Exact inversion followed by masking can worsen error by √101; a small relative map error can nearly double physical error despite a >98% nominal-objective reduction; positive tracking feedback can destabilize a stable scalar healthy model/Kalman observer. These are constructed examples, not fitted explanations of individual robot failures. |
| Conditional task transfer | A successful full-trajectory tube and a valid lifting bound would connect state-error control to success. The paper explicitly states that the experiments do not establish those extra assumptions. |

Complete statements and proofs are in [theory_appendix.tex](theory_appendix.tex)
and [authority_appendix.tex](authority_appendix.tex). Main-text statements remain
self-contained; the appendix supplies the derivations and qualifications.
The arguments use standard linear algebra and stability tools. They are not
presented as a new general adaptive-control theory.

## What moved to the appendix

The main text retains the method, essential theory, experimental splits,
calibration caveats, two confirmation tables, central harm findings, and the
unresolved composite–Kalman result. The appendix retains the original detailed
implementation, historical severity/time-profile/backbone/ALOHA/GR1 results,
convergence and recoverability figures, all-joint tables, selected tuning settings,
per-condition counts, and hardware protocol. Proofs precede these details.

A new vector [interface diagram](fig_interface.pdf) separates command faults from
joint faults and shows the predictor's use of corrected pre-fault commands.
Its source is [make_interface_figure.py](make_interface_figure.py).

## Reference and figure revision

The expanded related work covers generalist robot policies, adaptive augmentation,
classical observers, and learned adaptation. MAGIC-VFM and HMAC are discussed in
the main text using their published records, alongside Neural-Fly,
control-oriented meta-learning and automated mirror descent. Additional policy
references include RT-2, Octo and Diffusion Policy; LIBERO and the ALOHA/ACT paper
identify the evaluation environments. ALOHA's ACT paper is background for the
platform, not a claim that ACT is the policy evaluated here.

The [citation source archive](citation_sources/) retains retrieved citation
exports and source records with URLs and hashes. The [reference audit](REFERENCE_AUDIT.md)
distinguishes official proceedings, publisher-deposited metadata and preprints;
it records any remaining metadata-source exceptions. These are not attributed
to Google Scholar unless actually retrieved from that service.

The symbolic interface figure combines the execution/estimation loop with authentic
Panda and ALOHA scenes and a quantitative historical joint-5 failure panel.
The [theory figure](fig_theory_analysis.pdf) visualizes correction authority and
the scalar composite stability boundary. The [results figure](fig_results_analysis.pdf)
shows all Panda joint/suite outcomes and the complete tuned ALOHA comparison,
including the primary paired interval. The enlarged video pair moved to the
appendix. A fourth [tracking diagnostic](fig_tracking_analysis.pdf) separates
reference tracking from known-offset identification and explains why lower
reference error does not establish better task success.
The [figure audit](FIGURE_AUDIT.md) records exact sources, metrics,
interpretation limits and reproduction commands. These additions contribute
analysis and explanations, not extra benchmark trials or a new estimator ranking.

## Validation and reproducibility

- Independent code/data review reproduced the main confirmation counts, score,
  primary test/interval and multiplicity corrections. It also checked the policy
  scope and distinguished vector RMS telemetry from per-coordinate qualification.
- Independent mathematical review checked both proof appendices. The executable
  [theory checks](verify_interface_theory.py) cover 80 actual composite recursion
  steps, the exact scalar threshold, both allocator counterexamples, and 128
  robust-bound cases. Maximum recurrence discrepancy is 8.9e-16. The
  [receipt](interface_theory_receipt.json) records source hashes and makes clear
  that finite numerical checks are not proofs or robot qualification.
- The [reference audit](REFERENCE_AUDIT.md) documents published citation exports,
  primary preprint records and remaining source exceptions. The manuscript cites
  28 works, up from 16; the bibliography database contains 50 entries and passes
  the standalone BibTeX check without warnings. The archive preserves source
  metadata so publication versions, author lists and formatting changes can be checked.
- The final build has no undefined citations/references or overfull boxes.
  Representative main/theory pages and the interface figure were visually checked.
- The [layout check](check_submission_layout.py) counts all pages before the
  explicit backmatter boundary, including deferred main-text floats. It checks
  the unchanged official style hashes and writes a
  [layout receipt](submission_layout_receipt.json).

Rebuild and validate from the repository root:

```sh
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error iclr_draft.tex
python check_submission_layout.py --require-main-pages 9
```

Optional reproduction of the numerical theory checks, from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/verify_interface_theory.py
```

## Remaining scientific work

This is a stronger, page-compliant draft, not a certification of conference
acceptance. The key evidence gaps remain those in the
[experiment audit](EXPERIMENT_RERUN_AUDIT.md): fresh shared held-out LIBERO
calibration/transfer, a matched-parameter composite tracking/damping ablation,
and renewed friction/lock checks if retaining those efficacy claims. Broader
composite generalization and real-robot repair need new evidence. A quantitative
comparison to the closest learned repair method under matched scenarios would
also strengthen the experimental positioning; bibliography or theory additions
cannot substitute for that comparison.

If the proposed uncertainty certificate becomes a deployed algorithm, its bounds
must be estimated or justified, then independently evaluated. Current confirmation
data cannot be reused to tune that new method and still be called untouched test
data. No new policy/simulator experiments were launched for this manuscript revision.
