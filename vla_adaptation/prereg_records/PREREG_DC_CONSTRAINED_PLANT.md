# Preregistration: the plant's DC gain constrained to the probed sensitivity on the corrected
# channels — the r_y lever (2026-09-14)

**Written before the runs.** Record 50: every plant fitted on the closed-loop healthy log (FIR
K = 6, K = 20, ARX) recovers an r_y DC gain of 0.10–0.12 where the open-loop probe gives 0.28 and
the onset transient (record 49) confirms the probe; on every rotation channel the estimate settles
near (fitted gain)/(probed gain) — r_x 0.90, r_y 0.37, r_z 1.02 — because the correction is
executed at the probed gain and predicted at the fitted one, feeding −(M − G_fit)·f̂ back into the
residual. The lever that follows: fit the plant with its command-tap sum constrained to the probed
M entry on the channels the law corrects (`--dc-constrain corrected`, a constrained ridge through
its KKT system; the intercept is free, the tap shape is refitted). On the shipped log this moves
the rotation tap sums 0.228 / 0.103 / 0.248 → 0.253 / 0.276 / 0.244; the translation plant is
unchanged (its probed z entry is a known state-45 artefact and is not used).

**Configuration:** the shipped legacy reference with only the plant changed (γ = 0.08, δ = 0.008,
ρ = 0.15 over all six channels, κ = 0.15, K = 6, rotation corrected, shipped calibration,
`--scenario-reset`, unpinned, inits from 45) — so the comparison is the shipped cells themselves
(spatial 8–9 → 18/20, libero_10 0 → 15/40) and the D.3a oracle (19/20, 22/40). Runs, in order:
`dc_faulted_libero_spatial` (n = 20), `dc_faulted_libero_10` (n = 40), `dc_healthy_libero_spatial`
(n = 20, `--sev 0`): the constrained plant fits the healthy log worse on r_y by construction, so
its healthy residual and phantom must be measured. Outputs `results/dc_plant/<id>/`.

**Predictions.**
1. r_y estimate (median over episodes of the last-50-step mean) ≥ 70 % of the fault on both
   suites (legacy law and ρ = 0.15 attenuate r_x, r_z to 87–88 %; r_y from 41 % to the same band
   is the mechanism's claim). **Refutation: r_y < 60 %** — the DC-gain reading of record 50 is
   wrong or incomplete.
2. If 1 holds: libero_10 ≥ 18/40 (toward the oracle's 22). **Registered refutation: libero_10 ≤
   16/40 with r_y ≥ 70 %** — r_y identification is not what limits the long-horizon suite.
3. No-harm: spatial faulted within 3 of 18/20; healthy spatial loses at most 2 episodes
   (adaptive ≥ frozen − 2) and the healthy r_y phantom stays below 0.01 (a fifth of the fault).
   A healthy loss ≥ 3 reports the constrained plant as unsafe on a healthy robot.

**Refutation handling.** Each prediction scored separately, failures primary with the number.

---

## Outcome, faulted cells (2026-09-14; `results/dc_plant/dc_faulted_*`; healthy control pending)

Constrained rotation tap sums on the shipped log: 0.228 / 0.103 / 0.248 → 0.253 / 0.276 / 0.244
(the probed diagonal); translation unchanged.

| suite | frozen → adaptive | fixed / broken | shipped legacy (same scenarios) | oracle D.3a | settle r_x / r_y / r_z (% of fault) |
|---|---|---|---|---|---|
| libero_spatial, n = 20 | 7 → **17** | 11 / 1 | 18 | 19 | 94 / **82** / 86 |
| libero_10, n = 40 | 0 → **18** | 18 / 0 | 15 | 22 | 96 / **70.5** / 89 |

- **Prediction 1 (r_y ≥ 70 %): holds on both suites** — 82 % on spatial (every episode passes
  70 % within 18–40 steps) and 70.5 % on libero_10 (median; IQR 50–85 %), from 41 % and 45 %
  under the unconstrained plant. The refutation band (< 60 %) is not entered. The record-50
  reading — the estimate settles at (fitted gain)/(probed gain) — is confirmed by intervention:
  raising the fitted r_y gain to the probed value doubles the r_y estimate with nothing else
  changed. r_x and r_z stay at 86–96 % (the legacy law's attenuation at ρ = 0.15).
- **Prediction 2 (libero_10 ≥ 18/40): holds at the letter** — 18/40, 18 fixed, 0 broken. Paired on
  the same 40 scenarios against the shipped legacy cell (15/40) it is 10 won / 7 lost; against the
  known-fault oracle (22/40) 4 won / 8 lost; against the rule-constants cell (13/40) 13 won / 8
  lost. All runs unpinned, so these are between-run differences at n = 40: the direction is
  toward the oracle, the size is inside the suite's noise (record 36: ± 11 points at n = 20).
  The registered refutation (≤ 16/40 with r_y ≥ 70 %) is not triggered.
- **Spatial no-harm: holds** (17 within 3 of 18); one episode broken.

**E2 decision rule (docs/UNIFIED_PLAN_EXECUTION.md), applied:** r_y ≥ 70 % on both suites → the
consistency intervention is worth its confirmation. E2's probe qualification and the 840-rollout
core go ahead once the healthy control below passes and the probe tool is validated; the pilot
is reported as a pilot (historical probe, legacy law, unpinned, shipped scenarios).

## Outcome, healthy control and summary (2026-09-14; `results/dc_plant/dc_healthy_libero_spatial`)

Healthy spatial, n = 20: frozen 20 → adaptive **19** (0 fixed, 1 broken); healthy rotation phantom
median 0.0006 / −0.0030 / 0.0016 on r_x / r_y / r_z. **Prediction 3 holds at its letter** (one
episode lost, allowed two; the median r_y phantom is a third of the 0.01 bound). Reported with
it: one healthy episode's r_y estimate ran to |0.078| over its last 50 steps, above the fault
size — the constrained plant fits the healthy log worse on r_y by construction, and on that
episode the residual it left was acted on until the projection bound; that is the lost episode.
The legacy plant's healthy control on the same scenarios lost none (record 43: 19 → 20).

**Summary.** The intervention does what record 50 said it would: raising the fitted r_y gain to
the probed value doubles the r_y estimate (41 → 82 % spatial, 45 → 70.5 % libero_10) with nothing
else changed, and libero_10 moves from 15 to 18 of 40 (paired 10 won / 7 lost, unpinned) toward
the oracle's 22; spatial is unchanged within noise (17 vs 18) and the healthy control loses one
episode to a runaway phantom. All three predictions hold; none of the count effects is resolved
at n ≤ 40. This is the pilot for E2 of `iclr2027/EXPERIMENT_PLAN.md`; E2's own design (probe-
qualified constraint, innovation law, fresh partitions, three sampler seeds, 120 paired keys,
healthy arms with a −5-point no-harm target) is what can resolve them, and it goes ahead.
