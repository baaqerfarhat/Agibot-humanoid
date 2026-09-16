# Which manuscript to submit to ICLR 2027 — assessment (2026-09-16, before the abstract deadline of 18 Sep)

Three candidates are in the repository at be5853d / 0896a67; all three are consistent with the
records and the raw runs (numbers cross-checked against records 36–59).

| | A. `paper/iclr_draft.tex` | B. `iclr2027/main.tex` | C. `paper_integrated/main.tex` |
|---|---|---|---|
| title | Frozen Yet Adaptive (original story, rewritten abstract 15 Sep) | Predictor Consistency and Physical Recovery for Frozen VLA Policies | Frozen Yet Adaptive: Calibrated Feedback for VLA Policies |
| centre of gravity | breadth of empirical repair: four suites, three backbones, ALOHA, GR1, WidowX, fault families, hardware appendix | the predictor–observer feedback theory, with the experiments as tests of it | the empirical paper's structure with the controlled studies (E2, E1 v2, Q5, Q6) and a short conditional theory section |
| main text | 9 pages, 19 total; 12 result subsections | 9 pages, 41 total (four theory appendices) | 8 pages, 27 total |
| theory | two empirical propositions + an appendix of measured constants | stationary-bias identity, conditional Lyapunov test, recovery-weighted information bound, finite-horizon bounds, joint Lyapunov and causal-selection appendices | the residual-feedback identity, scalar equilibria, authority and finite-horizon bound; proofs in one appendix |
| controlled evidence for the theory | appendix only (E2/E1/Q results in the "latest experiments" appendix) | E2 table and E1 v2 in the main text | E2 table, E1 v2, Q5 and Q6 in the main text |
| GR00T / GR1 / WidowX | full main-text subsections with tables | one transfer sentence + GR1 row of the update table; WidowX in the appendix | one transfer table in the main text (OFT, GR00T N1.7, ALOHA, GR1 ×2, WidowX) |
| honesty of claims | high after the audit; abstract now hedged | high; the abstract says what failed | high; abstract states the failed criteria |
| build / checks | 9 pages, layout checker passes | 9 pages, receipt passed, 41 pages total | 8 pages, receipt passed; its evidence manifest predates the Q6 prereg amendment of 16 Sep 00:40 (rebuild needed by the writing session) |

## How a reviewer would read each

**A** reads as a systems paper with a very large evidence base and a modest theory. Its strengths
are breadth (three backbones, four robots, two simulators), paired protocols and preregistration,
and the failure-mode sections. Its weakness at ICLR is novelty framing: the method is a
disturbance observer with classical robustness modifications, the paper says the contribution is
the calibrated interface, and the theory is two propositions. The 15 Sep abstract rewrite made it
more accurate and less distinctive. Likely reviewer response: "solid, well-audited, but what is
new?" — a borderline paper whose fate depends on whether reviewers value the evidence base.

**B** reads as a theory paper with experiments. Its strengths are a clear question (why does
accurate prediction not give accurate estimation, and accurate estimation not give recovery), the
feedback-path analysis, and the fact that its controlled tests were run and reported including
the negative ones. Its weaknesses are the ones its own readiness note lists: the prospective
physical prediction failed (E1 v2 forecast −51 %), the E2 task effect is unresolved, several
appendices (recovery-weighted information bound, joint Lyapunov, causal selection) have no robot
evidence, and 41 pages of appendix invite the question of how much of the theory is load-bearing.
Likely reviewer response: "the analysis is careful, the decisive experiment is missing" — a paper
that can be rejected for promising a theory of recovery that its own results do not certify.

**C** reads as the empirical paper with the controlled studies pulled into the main text and the
theory reduced to what the experiments actually test. Its strengths: the headline table, a
transfer table with the humanoid and WidowX rows in the main text, the E2 table with healthy
controls, the six-channel oracle (38/40 vs 22/40) as a clean authority result, the continuation
study reported with its failed criteria, and one theory section whose every statement is
exercised by an experiment (residual feedback → E2; authority → Q5; memory → E1). It is the only
one of the three whose theory and evidence are in proportion. Its weaknesses: 8 pages leaves a
page unused; the discussion is short; the transfer table's "Healthy" column mixes off-policy
healthy with matched controls (labelled); and the main text's controlled-study paragraphs are
dense. It has fewer citations in the method/theory sections than B.

## Recommendation

**Submit C.** It makes the claim the evidence supports (calibrated execution feedback repairs
frozen VLAs; identification, authority and memory are separate, measured limits), it puts the
humanoid and the second-backbone results where reviewers will see them, and it cannot be
attacked for a theory it does not test. Use the ninth page for the discussion: the mask result
(Q5) as the concrete next step, the healthy-harm boundary, and the hardware scope.

Take from A into C: the fault-family table with the profile results (compact), the hardware
protocol appendix (one paragraph), and the "constants as ratios of measured scales" remark.
Take from B into C: nothing beyond what C already imports; keep B's long appendices out.

If the abstract has to be written before the choice is final, C's abstract is the right one: it
contains the headline numbers, the transfer sentence, the two controlled results with their
directions and one failed criterion, and no promise.

## Before submission (whichever is chosen)
- Regenerate C's evidence manifest and receipt after the 16 Sep Q6 amendment (its `build.py
  --check-only` now fails on the changed prereg bytes); the Q6 rotation-y target is reported as
  "median 0.715 / mean 0.613, target missed under the mean" in both C and the record.
- Decide the libero_10 sentence: with Q5 the ceiling is the rotation mask; C says this in one
  clause, A and B in an appendix.
- Anonymised supplement: C's package is self-contained (style, figures, bib, build), the raw
  evidence lives in this repository.
