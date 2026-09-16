# Evaluation of `paper_integrated/` (FrozenYet Adaptive), 16 Sep 2026

Manuscript state evaluated: `paper_integrated/main.tex` and its inputs as of 16 Sep 09:26
(build receipt: status passed, 9 main pages, 35 total, 515 evidence hashes verified; re-run of
`python3 build.py --check-only` today reproduces that receipt). The evaluation is written as an
ICLR reviewer would write it, then followed by a number audit and a ranked list of fixes.

## 1. Verdict in one paragraph

Soundness is the paper's strength and framing is its risk. Every quantitative claim I traced
(Section 3) reconciles with the score files and the record, the statistics are preregistered and
task-clustered, and negative results are reported as primary. The problem is that the title,
abstract and contribution 1 lead with a **two-path architecture whose defining component, the
independently calibrated correction-response path, is not evaluated anywhere in the paper**. The
paper says so itself three times (`experiments.tex` line 8, `discussion.tex` second paragraph,
`method.tex` "proposed extension"). ICLR reviewers routinely score "the proposed method is not
evaluated" as a contribution-level defect regardless of how honest the text is. Expected scores
as it stands: soundness 4, presentation 2–3, contribution 2–3, overall 5 (borderline). One
targeted experiment (Section 4, item 1) moves contribution 1 from proposal to evidence and is the
single change most likely to lift the paper to a 6.

## 2. Reviewer-style assessment

### Strengths
- **Breadth with matched controls.** Four LIBERO suites, three Panda backbones (π0.5, OpenVLA-OFT,
  GR00T N1.7), three further interfaces (ALOHA, GR1, WidowX), each with healthy controls, faulted
  frozen floors, and identification episodes excluded and stated. Few adaptation papers report
  healthy regressions at all; this one lists the lost keys.
- **Mechanism separation.** The predictor-consistency experiment (E2), the correction-support
  oracle (six channels), and the physical continuations (Panda and ALOHA) test three different
  things (observation, authority, physical memory) and the text keeps them apart. The E2 negative
  on LIBERO-10 (identification quality does not determine task recovery) is a real finding.
- **Statistics.** Pinned sampler schedules where it matters, exact McNemar as descriptive,
  task-clustered bootstrap for inference, preregistered decision rules with refutation thresholds.
- **Theory is honest.** The joint tube (Theorem th:gated-smallgain) is stated as conditional, its
  premises are named, and the text says they were not qualified on the robots.

### Weaknesses a reviewer will raise
1. **The headline contribution is unevaluated.** "Its two prediction paths separate nominal
   behavior from the response to corrective actions" (abstract) is the thesis, and the only
   evidence for it is an identity (th:architecture-residual) plus the statement that the
   shared-predictor specialization couples (G−W)D_m into the observer. No experiment fits H_j,
   no experiment compares H=W against an independent H. The discussion even concedes that for
   linear predictors the two-path form "is also a predictor with explicit correction-history
   features", which invites the question of why it was not simply run.
2. **The theory has measured constants that the paper withholds.** The record measured the
   position-only contraction on Panda OSC_POSE at λ̂ = 1.00 (the tube is vacuous, the small-gain
   condition fails) and the ALOHA servo at λ ≈ 0.66; GR1's servo poles are 0.72–0.77. The main
   text only says the contracting premise "is not established for the evaluated VLA loops". A
   reviewer who sees a conditional theorem with no attempt to measure its constants will call it
   decorative. Reporting the measured λ̂ turns the theorem into a diagnosis (Panda integrates,
   ALOHA forgets), which is exactly what the continuation experiments then show.
3. **Presentation density.** One figure in the main text (the interface schematic), four tables,
   and ~90 lines of theory. Almost every paragraph ends in a caveat, so the claims are hard to
   find. The three result figures already produced (`results/iclr_unified_v1/figures/`:
   Panda-vs-ALOHA continuation contrast, E2 mechanism, LIBERO-10 mask/authority) are not used.
   The appendix's only figure is a convergence plot.
4. **Citation coverage.** 27 of 58 bibliography entries are cited. LIBERO (`libero`) itself is
   uncited, as are the adaptive-control anchors already in the bib (`astrom1995`, `narendra1989`,
   `sastry1989`, `slotine1991`, `hovakimyan2010`, `lohmiller1998contraction`, `ljung1999`) and
   the 2026 VLA-adaptation entries (`correctvla2026`, `healthvla2026`, `vitar2026`, `orpa2026`).
   The theory section cites two works; the experiments section cites none.
5. **Simulation only.** All task-repair results are simulated; hardware appears only as
   diagnostic logs. The abstract does not say "simulated" until the LIBERO sentence.
6. **Development-tuned mask and gains.** The rotation-only mask and the observer constants were
   selected on faulted development data (stated in the protocol paragraph). A reviewer will ask
   for the disjoint-state numbers to carry more weight than they do now (68/120 vs 78/120 is in
   the table, which is good, but the abstract quotes both without saying which is cleaner).
7. **Healthy cost of the adopted extensions.** Six channels: healthy 56→47 (U6) and 56→53 (C6);
   gated: 54, with faulted 32/60 in the inconclusive band. The paper reports this correctly, but
   the abstract's "bounded proprioceptive feedback" reads as safe while the best faulted
   configuration (41/60) costs six healthy episodes.

### Smaller points
- The abstract quotes the Panda continuation (13.30→6.68/6.50 mm) from ten Spatial episodes and
  eighteen checkpoints; say "ten held-out Spatial episodes" and consider quoting the ALOHA
  numbers next to it, since the two-robot contrast is the interesting part.
- "Robot-specific calibration also supports recovery after preliminary fault identification on
  simulated ALOHA, GR1, and WidowX interfaces" gives no number; 0/39→15/39, 1/27→17/27,
  0/20→14/20 fit in one clause.
- Transfer table denominators (17/39, 20/27, 14/20) differ from the record's cohort sizes
  (40, 30, 23) because identification episodes are excluded. The paragraph before the table says
  this, but the caption should too; a reviewer comparing with the appendix's matched
  initialization table (n=40, n=30) will otherwise flag an inconsistency.
- The theory appendix's "Prospective calibration" paragraph is the recipe for the missing
  experiment (fit H_j from matched replay pairs with a signed correction sequence). It should
  either be executed or explicitly labelled as future work in the contribution list.

## 3. Number audit (all reconcile)

| Paper statement | Source | Result |
|---|---|---|
| Headline 28/120→78/120, disjoint 34/120→68/120; healthy 116→117, 118→118 | record §§ headline cohorts, README tally | matches |
| Transfer: ALOHA 17/39, 0/39, 15/39 | `results/aloha/{healthy,off002}_identify1_hold_n40.json`: healthy frozen 17/40 with episode 0 a failure, adaptive 15/40 with episode 0 (identification) a failure | matches after episode-0 exclusion |
| Transfer: GR1 20/27, 1/27, 17/27 and 15/27, 1/27, 13/27 | record §32.18–32.19 (held episodes 3–29 of 30) | matches |
| E2 Spatial 35.5 % [21.9, 47.6]; LIBERO-10 −33.4 % [−135.6, 39.8]; C−U +11.7 [0, 25], p=.143; healthy 60/58/60 and 56/53/52 | `results/iclr_unified_v1/E2_core/RECEIPT.json`, `score_*.json`, record §56 | matches |
| Six channels: faulted 35/41/57, healthy 47/53 (off 56); C6−C +26.7 [8.3, 46.7]; C6−U6 +10.0 [−1.7, 21.7] p=.210; oracle−C6 26.7 [15.0, 38.3] 16/0; U6 healthy −15.0 [−26.7, −3.3] 10 lost/1 rescued; C6 −5.0 [−16.7, 3.3] 6/3 | `results/six_channel/score_libero_10_six.json` | matches |
| Gate: healthy 54/60 (3 lost, 1 gained), faulted 32/60 | record §62 | matches |
| Q6: FIR .267/.295, ARX .333/.337, DC .613/.715; 10/8/14 of 80; p=.774/.263/.388 | `results/collab_q/score_q6_*.json`; recomputed mean/median from `q6_{arx,dc}_libero10_n80/telemetry.jsonl.gz` (.333/.337, .613/.715) | matches |
| Panda continuation 13.30→6.68/6.50/3.42 mm, eighteen checkpoints, ten sources | `results/iclr_unified_v1/E1_v2/pass2_score.json` (126 rows = 7 branches × 18) | matches |
| ALOHA continuation .0491485→.0031339/.0019746/.0024046 rad; ratio 1.012; remaining .002816/.001414 (94.3 %/97.1 %); λ=.663; paired −1.5489 [−1.5640, −1.5267], −1.5935 [−1.5987, −1.5870] | `results/iclr_unified_v1/E1_aloha_v2/{pass2_score,memory_model_score}.json` | matches |

Nothing in the main text or the two appendices contradicts the record. The two earlier
fabrication incidents (Q2 joint-5 counts, six-channel oracle contrast) are not present in this
manuscript.

## 4. Ranked fixes before 25 Sep

1. **Evaluate the two-path predictor, at least on Panda.** The infrastructure exists: the E1
   replay driver produces matched healthy branches from full-state snapshots; add a branch with a
   known signed correction sequence c_k (both signs, two amplitudes) on the ten state-33 sources,
   fit H_j by `th:intervention-fit`, then run the sixty E2 LIBERO-10 keys and the sixty Spatial
   keys with H≠W against the existing C arm (same observer, same tuning). Replay collection is
   CPU-side MuJoCo; the online arms are ~120 policy episodes, one GPU-evening. Prereg the
   decision rule first (target: r_y observation bias no worse than C, healthy losses ≤ 3). Even a
   null result converts contribution 1 from proposal to evidence.
2. **Put the measured contraction constants in Section 4.** One sentence: Panda position-only
   λ̂ = 1.00 (integrator; tube vacuous), ALOHA 0.66, GR1 poles 0.72–0.77. Then the theorem's
   "when contraction cannot be established" clause has a concrete referent and the continuation
   contrast (growth 1.8× vs saturation 1.01×) reads as its consequence.
3. **Add two result figures to the main text**: the Panda/ALOHA continuation contrast and the
   LIBERO-10 authority figure (rotation-only → six channels → oracle). Both exist as PDFs. Trade
   the appendix-length prose in the six-channel paragraph for the figure.
4. **If item 1 is not run, reframe.** Title and abstract lead with the evaluated architecture
   ("calibrated execution feedback for frozen VLAs"), contribution 1 becomes "an analysis that
   isolates the correction-response pathway and a testable two-path extension", and the abstract's
   second sentence drops "its two prediction paths".
5. **Cite what is in the bib.** LIBERO, the adaptive-control anchors, contraction (Lohmiller &
   Slotine), the 2026 VLA-adaptation papers in related work, and the policy/benchmark papers in
   the experiments section.
6. **Abstract edits**: say "simulated" once up front; give one number for the transfer sentence;
   state which calibration (disjoint) is the cleaner headline.
7. **Captions**: transfer table caption states "identification episodes excluded (39, 27, 20)".

## 5. What was not evaluated

Page-format compliance beyond the receipt (9 main pages), the anonymisation statement and code
link (parked), and the writing session's pending items (evidence manifest after the Q6
amendment, hand-off absorption) were not re-checked here.
