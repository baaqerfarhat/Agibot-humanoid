# re4 evidence plan — runs, logs, and artifacts to close the paper's stated gaps

**Target document.** The re4 submission ("Embodiment Adaptation of Vision-Language-Action
Models from Proprioceptive Residuals", `Paper/re4` in the ICLR2027 working tree; 9 main pages,
appendices A–G). Its theory is final. What remains is **evidence**: the paper currently hedges
every place where a record, control, or measurement is missing, and each hedge is quoted below
next to the artifact that removes or updates it.

**How to use this file on the evaluation machine.**
Work top to bottom inside each part; parts are ordered by value per hour. Every run follows the
**Global logging contract** (§0). Every new comparison gets a prereg file in
`prereg_records/` *before* it is run, in the style of `PREREG_HEALTHY_GATE.md` and
`PREREG_HELDOUT_CALIBRATION.md` (statistic, threshold, and refutation condition written first).
All outputs go under `results/re4_evidence/<part>/...` and are synced back on this branch.
Where a command is written with `<runner>`, substitute the LIBERO/ALOHA/GR1 entry point used
for the existing records; the flags named here (`--estimate-only`, `--init-base`, `--openloop`,
`--log`) are the ones already in records 37–39.

**Already done — do not redo, only export.** Record 37 (healthy-only channel gate) and
record 39 (held-out calibration, four cells) already close two of re4's stated gaps. Part B
only asks for their per-episode artifacts in the standard schema.

---

## 0. Global logging contract (applies to every run below)

For **each run**, fill and commit:

1. `reproduction/audit/run_configuration_template.json` → one filled copy per run at
   `results/re4_evidence/<part>/<run_id>/run_configuration.json`. Required fields that the
   audit found missing historically — none may be left null:
   - **code path**: external action subtraction vs native `action_out_proj/bias` edit;
   - **gate convention**: full hold (`chi=0`) vs zero-observation leakage (`chi=1, s=0`);
   - **clipping order** relative to the correction; projection box; units per channel;
   - **update law** (attenuation / innovation), gains, deadzone, normalizer `P_r` and its
     channel set; FIR order `K`; snapshot/application rule (chunk size, `tau_k`);
   - calibration id + SHA of the calibration arrays used; `reset_protocol`;
   - `policy_rng_pinned: true/false` (record the seed if true);
   - source SHAs of every script invoked (schema v2, as in the held-out log).
2. One row per episode in `reproduction/audit/episode_record_template.csv` schema:
   (task, init, seed, arm, outcome, fault family/magnitude, fall/violation flags), at
   `results/re4_evidence/<part>/<run_id>/episodes.csv`. **Paired cells must share
   (task, init) lists across arms**, and the pairing must be reconstructable from the CSV
   alone — this is what "matched healthy controls for these exact samples are unavailable"
   means today; the CSV removes it.
3. Timing fields of `reproduction/audit/timing_and_error_budget_schema.md` wherever Part E
   applies.

Calibration arrays themselves (FIR `H_l`, `c`, `S`, `b_cal`, per interface) are committed once
under `results/re4_evidence/calibration/` with SHAs referenced by every run that uses them.
Paper hedge removed: *"Original evaluation code, calibration arrays, and episode traces remain
unavailable; reported VLA outcomes were not independently rerun."*

---

## Part A — Provenance backfill (no physics; ~half a day)

Close the implementation-identity hedges by inspection of the code and existing configs.
Product: `results/re4_evidence/A_provenance/implementation_map.md` with one row per cohort
(headline 4 suites, severity map, ablations, gain cells, joint-level cells, OFT/N1.7 transfer,
ALOHA, GR1, WidowX):

| cohort | code path (external vs native-bias) | gate convention | clipping order | K | update law + constants | calibration id |

- re4 hedge to remove: *"the early report explicitly specifies native bias editing, whereas the
  later draft gives an additive action-map description without documenting a code transition.
  It also gives both leakage and full-hold descriptions of an inactive deadzone. The available
  sources do not resolve those choices for each cohort."*
- Also record `K` for ALOHA and GR1 (re4 table currently prints "K unrecorded"), total probe
  and reset counts for the sensitivity measurements, the fault-development selection budget,
  and complete task/seed identity lists for the four primary samples.
- If a cohort's config genuinely cannot be recovered, say so in the map — the paper keeps its
  hedge for that cohort and drops it for the others.

## Part B — Export of the two finished audit results (no new physics)

1. **Held-out calibration (record 39).** Export per-episode CSVs for the four held-out cells
   and the calibration arrays for init-25 into the §0 schema. re4 will then report the
   two-headline form (shipped 28→78/120; held-out 34→68/120, `libero_10` primary as held-out).
2. **Healthy-only channel gate (record 37).** Export the phantom stats and the gate decision
   table. re4 hedge updated: *"Correction directions were informed by fault diagnostics,
   without a documented held-out selection split."* → gate is preregistered, healthy-only,
   and reproduces the mask.

## Part C — Matched healthy controls for the primary cells (~1 GPU-day)

For the exact 120 (task, init) pairs of the headline table, both with the shipped and the
held-out calibration where applicable:

1. **Healthy, frozen** (no fault, no adaptation): 120 episodes → healthy competence on the
   same samples. Removes: *"Healthy counts for the exact four primary samples are absent from
   the available summaries."* and the caption hedge quoted in §0.2.
2. **Healthy, adaptation active** (no fault): 120 episodes, paired against 1 → the
   regression-under-health check at the primary scale (fixed/broken counts, exact McNemar).

## Part D — The three decisive missing baselines (~2 GPU-days)

re4 names them in one sentence: *"A calibrated static observer, a matched innovation-law
comparison, and known-fault correction through the same interface remain the decisive missing
baselines."* One prereg each, all on the **headline spatial cohort** (n=20 pairs, same
(task, init) lists, same seeds policy) plus `libero_10` (the hard suite):

1. **Calibrated static observer**: identical pipeline with `K=0` (static gain from the same
   healthy log). Tests whether FIR memory earns its place.
2. **Matched attenuation vs innovation**: identical cohort, constants scaled per the paper's
   ratio rule; report paired fixed/broken between laws. Closes: *"Its net effect on task
   success requires a matched comparison with innovation normalization."*
3. **Known-fault correction through the same interface**: inject `f`, apply exact `-f`
   externally (no estimator). This is the correction-authority oracle for the *evaluated*
   interface. Closes: *"known-fault correction through the same interface"* and gives the
   denominator for "fraction of the oracle's recovery achieved".
4. Optional, same cohort: best integral gain re-run paired on the identical (task, init) list,
   so the integral comparison in re4 §4.2 becomes paired instead of cross-cohort.

## Part E — Timing, latency, and recovery time (~half a GPU-day, with Part C/D runs)

Instrument (do not re-run separately; piggyback on Parts C–D):

- per-step wall-clock: policy call, adapter compute, end-to-end loop; report median / p95 /
  p99 / max and deadline misses at the interface rate (LIBERO and one of ALOHA/GR1);
- sensor-to-command delay as synchronized timestamps (the schema file defines the fields);
- **recovery time**: first step after fault onset at which executed-action error stays below a
  prespecified threshold for a prespecified window (write both numbers in the prereg), reported
  in seconds using real timestamps.
- Removes: *"The experiments do not measure median, p95, p99, or maximum adapter latency;
  synchronized sensor-to-command delay; deadline misses; or a predefined sustained
  executed-action error threshold for recovery."* and *"latency and sustained
  execution-recovery time were not measured."*

## Part F — Held-vs-continued isolation on ALOHA and GR1 (~1 GPU-day)

re4: *"Because initialization and timing both differ, these results do not isolate the effect
of pausing updates."* Design (prereg first):

- Arms, all initialized from the **same** prior-identification estimate: (i) held; (ii)
  continued updating; (iii) continued with the innovation law. Same episode lists as the
  existing held cells (ALOHA n=40; GR1 plate-0.10 n=30).
- Report paired fixed/broken held-vs-continued. This isolates "pausing" from "initialization".

## Part G — Estimator forensics (no or little physics)

1. **Decoder realization bound** (Appendix E of re4 gives the exact estimator): at fixed
   observation and decoder draw, measure decoded action with and without the native bias edit;
   report `zeta` and the fitted `\|D-J_bP_b\|` and quadratic term against Eq. (gen-decoder-bound).
   Removes: *"The final-action displacement induced by the decoder edit was not measured."*
2. **Centering convention**: state, per cohort, the exact averaging quantity, units, and
   subtraction order actually executed (raw-residual mean vs settled attenuated estimate — the
   two differ by the `s0(1-s0)b0` term derived in re4 App. B). One paragraph in the Part A map.
3. **ALOHA/GR1 predictor diagnostics**: persistence predictor `y_{k+1}=y_k` R² and
   incremental-motion prediction error alongside the shipped absolute-position R². Removes:
   *"high position R² alone does not identify command-induced motion"* as an open caveat.
4. **Units check**: confirm the "LIBERO's .019 cm per-step correction change" figure and its
   units (cm vs rad; coincidence with ALOHA's .019 rad static correction is suspicious).

## Part H — Optional, only if time remains

- Repeat M probes at a third initial state (prediction: translation block varies ~20%,
  rotation within ~8%, condition number ~3): turns record 39's two-point comparison into a
  three-point statement about configuration dependence.
- A second severity for the healthy-only gate to show mask stability.

---

## Acceptance checklist (what "done" means for the paper)

| re4 sentence (verbatim) | closed by |
|---|---|
| "reported VLA outcomes were not independently rerun" | §0 + Parts B–D reruns with full logs |
| "Matched healthy controls for these exact samples are unavailable" | Part C.1 |
| "the decisive missing baselines" | Part D.1–D.3 |
| "these results do not isolate the effect of pausing updates" | Part F |
| "without a documented held-out selection split" | Part B.2 (record 37) |
| "$K$ unrecorded" (ALOHA, GR1) | Part A |
| "The available sources do not resolve those choices for each cohort" | Part A |
| "latency and sustained execution-recovery time were not measured" | Part E |
| "The final-action displacement induced by the decoder edit was not measured" | Part G.1 |
| "Healthy counts for the exact four primary samples are absent" | Part C.1 |
| headline table single-calibration reporting | Part B.1 (held-out two-number headline) |

**Refutation handling.** Where a prereg threshold is crossed (as happened for `libero_10` under
held-out calibration), the paper reports the refutation as primary, exactly as record 39 does.
Nothing in this plan may be reported only-if-positive.

**Sync.** Commit results incrementally per part on this branch with messages
`re4 evidence: <part> <cohort>`; the writing side folds them into `Paper/re4` and rebuilds.
