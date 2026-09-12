# re4 evidence — index, status, and deviations from the plan

Plan: `docs/RE4_EVIDENCE_PLAN.md`. Every run below was preregistered first
(`prereg_records/PREREG_RE4_*.md`) and is recorded under the section-0 contract by
`openpi/re4_record.py`: `run_configuration.json` (no null fields), `episodes.csv` (pairing
reconstructable from the CSV alone), `timing.jsonl` + `timing_summary.json` where timed.
GPU scripts: `scripts/re4/`.

| part | folder | status |
|---|---|---|
| 0 calibration arrays | `calibration/` | done: `shipped_init45`, `heldout_init25` (FIR taps H_l, bias c, DC gain, M, M⁻¹, SHAs); `third_init5` |
| A provenance map (+ G.2 centering) | `A_provenance/` | done (record 42) |
| B held-out calibration and healthy gate exports | `B_heldout/`, `B_gate/` | done (records 37, 39) |
| C matched healthy controls, both calibrations | `C_healthy/` | done, both predictions confirmed (record 43) |
| D baselines (method rerun, K=0 static observer, innovation law, known-fault oracle) | `D_baselines/` | done (record 44) |
| E latency and recovery | every C/D (and ALOHA F) run's `timing_summary.json` | LIBERO done (17 runs); ALOHA with Part F |
| F held vs continued from a matched estimate (ALOHA, GR1) | `F_held_vs_continued/` | running |
| G.1 decoder-bias realisation | `G_forensics/decoder_bound.json` | done, refutation on realisation (record 46) |
| G.3 predictor diagnostics, G.4 units check | `G_forensics/` | done |
| H M at a third initial state | `H_third_init/` | done, refutation (record 45) |

## Deviations from the plan, stated

1. **Templates absent.** `reproduction/audit/run_configuration_template.json`,
   `episode_record_template.csv` and `timing_and_error_budget_schema.md` are not on this
   machine (nor is `Paper/re4`). The fields were taken from the plan's own list; fields that do
   not apply say why instead of being null.
2. **Protocol for new runs.** Every new LIBERO run uses `--scenario-reset` (forces cleared,
   cached env seeded per scenario, fingerprint), which the historical headline cells did not;
   comparisons across runs are therefore same-protocol only within `re4_evidence/`.
3. **D.0 added.** The headline law is rerun under that protocol with full logs, so every
   baseline has a same-protocol comparator and the headline has an independent rerun.
4. **F arm naming on GR1.** The plan's arms are (ii) "continued updating" and (iii) "continued
   with the innovation law". The GR1 cell's own law is already innovation, so on GR1 (ii) is
   innovation and (iii) legacy; (ii) always isolates pausing under the published law.
5. **G.1 definitions are operational.** re4's Appendix E symbols are not here; J, D, Q and ζ are
   defined in the prereg and in `openpi/g1_decoder.py`'s docstring for the writing side to map.
6. **E on a second robot** is ALOHA (50 Hz, 20 ms deadline), via a timing hook added to
   `aloha_adapt.py` for the Part F runs.
7. **Wall-clock is a simulator's.** Policy and simulator times are this machine's GPU/CPU times,
   not a real-time controller's; the adapter's own compute (≈ 0.2 ms median on LIBERO) is the
   number the latency claim needs.

## Corrections this work produced in the project's own documents

G.3/G.4 found five figures in the project's paper and report that the stored data do not
support; each was recomputed independently and corrected (record §41).
