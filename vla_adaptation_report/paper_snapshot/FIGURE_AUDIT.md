# Figure evidence and interpretation

The main paper contains four complementary figures. Their purpose is to explain
the method, distinguish its assumptions, and show the complete confirmation results.
No new simulator outcomes were collected for this figure revision.

| Figure | Content | Source and scope |
| --- | --- | --- |
| 1 — Interface and examples | A symbolic signal-flow diagram, Panda off/online scenes, an ALOHA held-correction scene, and historical joint-5 success counts | The scene [provenance](overview_assets/README.md) distinguishes methods, fault settings and render timing. The joint-5 chart comes from `results/joint_map/cell_torque_5.json`; its 12/20 versus 3/20 result uses continuous translation-only correction under a +5 N m torque. It is not an image of one of those failing episodes. |
| 2 — Two barriers to repair | Correction geometry and the spectral radius of a scalar composite loop | [Generator](make_theory_figure.py) and [receipt](theory_figure_receipt.json). These are exact constructed examples already stated in the proofs, not fitted explanations of a robot failure. The scalar gain boundary 16.2 is not an ALOHA parameter recommendation. |
| 3 — Complete confirmation results | All Panda joint/suite success changes and the tuned ALOHA estimator comparison | [Generator](make_results_figure.py) and [receipt](results_figure_receipt.json). The script reconstructs outcomes and checks the independent statistics: 1,440 Panda and 1,920 tuned-ALOHA confirmation rollouts. Controls are shared as recorded, not counted twice because they appear in two comparisons. |
| 4 — Tracking diagnostic | All eight condition-level reference-RMS pairs and the exact-offset reference/parameter error contrast | [Generator](make_tracking_figure.py) and [receipt](tracking_figure_receipt.json). Metrics are reconstructed from stored per-episode diagnostics. Each controller follows its own raw-command reference; the plot does not isolate tracking feedback from other selected parameter differences. |
| Appendix video figure | Larger view of the selected Panda pair | Original [video figure provenance](video_assets/README.md). It adds no new benchmark trial and is not a composite-versus-Kalman comparison. |
| Appendix estimator traces | LIBERO, ALOHA and GR1 historical fault estimates | [Generator](generate_convergence_figure.py) and [receipt](convergence_figure_receipt.json). Three pinned historical logs; each panel retains its own protocol, units and aggregation. This is not a three-robot task-success comparison. |

## Figure 3 metrics

Panda heatmaps show `100 × (observed rescues − observed regressions) / 20`,
relative to the shared correction-off cohort. Every healthy/joint cell appears
for each of the three suites and both correction rules. Color indicates magnitude,
not significance. Different arms use independent policy samples, so paired outcome
changes are observed differences rather than deterministic counterfactuals.

Weighted full has positive differences in 14 cells, negative differences in four,
and zero net differences in six. Legacy translation has seven positive, sixteen
negative and one zero. These descriptive counts include healthy conditions.
All four weighted negative differences occur on Object (joints 0, 1, 3 and 5).
The registered joint-5 comparison remains weighted versus legacy across three
suites; the heatmap does not redefine that primary comparison.

The ALOHA score is half healthy success plus half the mean success rate across
seven fault conditions, evaluated on 30 confirmation seeds per condition.
The figure includes all six selected estimated families and both controls in a
fixed family order. The only interval plotted is the stored primary paired
whole-seed bootstrap interval for composite minus Kalman. Independent error bars
for individual arms are not inferred from that paired interval. Oracle is a
privileged diagnostic, not a task-success upper bound.

## Figure 4 metric

The RMS is `sqrt(mean_episode(mean_last100_steps(||six_joint_error||²)))`.
All 30 episodes contribute 100 steps per arm/condition. This is a pooled vector
RMS, not an arithmetic mean of episode RMS values or a per-coordinate prediction
RMSE. On the exact offset, composite decreases reference RMS by 21.38% but
increases parameter RMS by 12.68% relative to Kalman. These telemetry differences
do not establish a task-success benefit or identify the effect of tracking alone.

## Reproduction and checks

The appendix trace figure uses the first 60 post-update LIBERO samples (indices
0–59), with pointwise medians and interquartile ranges across 20 episodes.
The final plotted rotation medians are 0.04083, 0.02431 and 0.04215. They are
not all rising near the end of that window. ALOHA shows one 300-update
identification episode, whose final left-arm estimate is 0.01835–0.01907 rad.
GR1 shows one 164-update identification episode; the held correction is its
final-50-update mean, 0.08996–0.11159 rad. This mean does not establish sustained
convergence by step 100. The generator verifies the carried estimate in every
subsequent recorded held episode. The GR1 log predates scene-generator reseeding
and is separate from the newly merged demonstration video.

From the repository root:

```sh
python paper/extract_overview_assets.py
python paper/make_interface_figure.py
python paper/make_theory_figure.py
python paper/make_results_figure.py
python paper/make_tracking_figure.py
python paper/generate_convergence_figure.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error iclr_draft.tex
python check_submission_layout.py --require-main-pages 9
```

The extraction scripts verify archived video/renderer versions and exact scene
crop pixels. Quantitative receipts bind inputs, calculations and exported figures.
The theory figure checks its values against the existing proof-verification
receipt. Main-text claims about cell signs and the theoretical vectors/stability
curve also received an independent read-only check. Final visual inspection is
performed on the typeset pages, including text size, labels and figure placement.
Heatmap cells use vector fills so the exported PDF preserves their colors;
the PDF itself is rendered and checked in addition to the PNG preview.

The layout check counts all pages before the explicit backmatter boundary,
including deferred figures and tables, and requires exactly nine main-text pages.
References, excluded statements and appendices follow that boundary. Official
style files, margins, font sizes and line spacing are unchanged.
