# Proposed text for the ICLR 2027 manuscript (queue items Q3, Q4, Q6)

**Status: proposed, not applied.** `iclr2027/` is the collaborator's manuscript. Nothing here edits
it; each file names the exact place it would go. Apply only with his agreement.

**Provenance.** Two independent Claude tracks drafted Q3 and Q4 separately (dual track W1, 2026-09-15)
and were converged here. Q6's relabel is Track 1's; its numbers are checked against committed
artifacts. Track 2's versions of Q3 and Q4 were adopted because they are stricter than Track 1's (see
below); Track 1's independent checks are recorded as confirmations.

## Files

| file | item | goes into |
|---|---|---|
| `Q4_fir_arx_proposition.tex` | FIR/ARX equivalence, with proof | new subsection after `\label{uniapp:dc}` in `theory_appendix.tex` |
| `Q4_main_pointer.tex` | one-sentence pointers | `experiments.tex` (`exp:boundaries`) and `theory_main.tex` after `uni:maps` |
| `Q3_smallgain_scope.tex` | corollary kept; scope paragraph replaced | `theory_main.tex`, the corollary block (lines 109–122 at HEAD) |
| `Q3_appendix_gr1.tex` | GR1 constants for the corollary | new appendix subsection `uniapp:gr1` |
| `Q6_descriptor_relabel.tex` | descriptor paragraph | `empirical_appendix.tex`, lines 222–233 |
| `Q1_dc_pilot.tex` | DC pilot as a causal test (healthy control pending) | `experiments.tex`, `exp:boundaries` |
| `Q6_selection_ledger.md` | development-selection ledger (59 entries) | a new appendix table; its reviewer-facing list belongs in the limitations |
| `checks/` | the computations behind every number | run from the repository root |

## What converged, and why Track 2's text was adopted

- **Q4.** Both tracks derived the same identity independently: the ARX innovation is the input-only
  residual passed through the invertible filter A(q⁻¹). Track 1 checked it on a second-order SISO plant
  (6.5e-16); Track 2 on a MIMO plant (1e-14, stacked map det 1, identical Fisher information) and on the
  robot's healthy logs, where the filtered FIR residual explains 92–100 % of the ARX innovation's
  variance per channel (`checks/q4_real_log_check.py`). Track 2's statement is stricter in two places
  Track 1's was loose: equivalence holds in information (σ-algebra, Fisher), not for a fixed-gain
  estimator's per-step variance (381x per sample at λ = .93, 2.2x at the estimator bandwidth, 1x at
  DC); and the stationary estimate is bracketed, [R, 1/(2−R)] with R the fitted-to-probed DC ratio,
  according to how much of the fault the policy absorbs — the measured r_y settles .32–.45 lie in
  [.36, .61]. The DC bias of closed-loop identification is an empirical fact, not part of the theorem.
- **Q3.** Both tracks give the same scope: the corollary is empty on the Panda (σ = 1 in pose) and
  Theorem 1 is the operative statement there. Track 2 additionally measured every GR1 constant but
  one: σ = .771, β = .282, m = .023, ε = .028 (median) / .057 (p90) rad. The condition reduces to
  ℓ < (1−σ)(1−m)/β = .79 for a constant gain; ℓ is not measurable from stored logs, so no certificate
  is claimed.

## Errors in the current manuscript that these proposals fix (both tracks agree; checked by Track 1)

1. `theory_appendix.tex` §`uniapp:dc`: "State dependence, FIR/ARX memory, normalization, and deadzones
   further change the stationary relation." Predictor memory does not change the stationary relation,
   only the transient; replacement sentence in `Q4_fir_arx_proposition.tex`'s header.
2. `empirical_appendix.tex` line 267: "the ratio of fitted to probed gain is not a general fixed-point
   law" is misleading — it is the fixed point when the policy restores healthy motion and one end of the
   [R, 1/(2−R)] bracket otherwise.
3. `theory_main.tex`: the corollary is presented without saying its premise is measured false on the
   primary robot; `Q3_smallgain_scope.tex` fixes this.
4. Minor: in the corollary, `q < 1` is implied by `σ < 1` and `βt < (1−σ)(1−q)` because β, t ≥ 0.
5. `empirical_appendix.tex` descriptor paragraph: the composite term was inert and the Object joint 3
   figure depends on a post-hoc cap (12/20 at the registered cap); `Q6_descriptor_relabel.tex`.

6. **Development selection on confirmation data (the ledger, `Q6_selection_ledger.md`).** Ten entries a
   reviewer would treat as selection on confirmation data. The load-bearing one, confirmed by Track 1 in
   the collaborator's own provenance map (`results/re4_evidence/A_provenance/implementation_map.md`
   l.266): γ, δ and ρ were tuned on tasks 0–9 at init 45 and tasks 0–4 at init 46, "the same (task, init)
   band later used to evaluate the headline spatial sample", and the rotation-only mask was chosen on
   those same scenarios. Others: the ALOHA and GR1 cohorts' constants and identify-then-hold choices were
   made on seeds reused for evaluation; WidowX γ = 0.2 was chosen after its outcome; projection boxes are
   set knowing the fault magnitude; the DC pilot ran on confirmation scenarios without its registered
   healthy control. The held-out calibration cohort (34/120 → 68/120) is the manuscript's answer to the
   first two, and should be presented as such.
7. **Protocol inconsistencies (ledger, confirmed by Track 1 for the first).** The GR00T N1.7 rows ran
   `--replan-steps 8` (record §30 l.2507; provenance map C6) while the protocol appendix says five
   actions execute between replans on the Panda. The OFT and GR00T healthy controls used the rotation mask,
   so their +0.15 translation cells lack a mask-matched control. `PREREG_UNIFIED_PARTITIONS.md` declares
   Spatial states 40–43 unused, but three run families used them.

One softening by Track 1 of Track 2's pointer: "This discrepancy is the whole effect" became "This
discrepancy sets the estimate's bias", since the DC mismatch does not explain the libero_10 shortfall
(the rotation-only oracle caps it at 22/40; see `PREREG_Q5_SIXCHANNEL_ORACLE.md`).
