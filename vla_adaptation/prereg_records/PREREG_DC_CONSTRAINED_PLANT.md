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
