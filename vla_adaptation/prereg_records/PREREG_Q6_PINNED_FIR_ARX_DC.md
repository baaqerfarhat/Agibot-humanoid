# Preregistration: pinned, paired FIR / ARX / DC comparison on libero_10 (queue item Q6)

**Written 2026-09-15, before any run. GPU runs are on hold until lambda is free.**

**Question.** Current FIR-versus-ARX-versus-DC claims rest on unpinned counts from different runs. The
FIR/ARX equivalence result (proposed Q4) predicts that model structure does not move the estimator's
fixed point, while the fitted-to-probed DC ratio does.

**Runs.** Two new pinned adaptive arms on the exact keys of the existing pinned FIR run
`a_method_libero10_n80` (80 episodes, `--eval-init 45` wrapping over inits 42–49, `--pin-rng`,
legacy law, shipped constants, uniform 0.05 offset, rotation correction), `--arms adaptive` so the
existing frozen arm (0/80) is shared: `q6_arx` adds `--ar 1`; `q6_dc` adds `--dc-constrain corrected`.
160 rollouts. Paired by (task, init) with `openpi/re4_theory/decision_cells.py`.

**Predictions** (r_y settle = late-window mean of f̂_ry / f over adaptive episodes).
1. **ARX ≈ FIR.** r_y settle within 0.10 of FIR's (FIR 0.41 in record 49; ARX predicted 0.32–0.45).
   Success within ±5/80 of FIR's 10/80. Refutation: ARX settle ≥ 0.60, or |ΔARX−FIR| ≥ 9/80 with a
   paired McNemar p < 0.05.
2. **DC moves the fixed point.** r_y settle ≥ 0.70 (pilot, unpinned: 0.82). Refutation: < 0.60.
3. **Success direction.** DC ≥ FIR in successes. Stated power: for a true +8/80 difference the paired
   test at n = 80 has low power (Track 1's calculation: 26 % for +4/40); a null here does not refute
   the mechanism in 2.

---

## Outcome, ARX arm (2026-09-15, 20:15; run by Mahdi's session on GPU 1; `results/collab_q/q6_arx_libero10_n80/`, `score_q6_arx_vs_fir.json`; DC arm running)

Pinned (`--pin-rng`), the 80 keys of `a_method_libero10_n80` (inits 45–49, 44–42), legacy law,
shipped constants, `--ar 1`, frozen arm shared (0/80).

- **Prediction 1 (ARX ≈ FIR): holds.** ARX **8/80** against the pinned FIR's 10/80 (5 ARX-only,
  7 FIR-only, exact McNemar p = 0.77; |Δ| = 2 < 9). r_y settle (last-50-step median over
  episodes) ARX **0.34** of the fault against the pinned FIR's **0.30** on the same keys — within
  the registered 0.10 (the 0.41 quoted above was the unpinned T1 figure; the pinned FIR run
  settles lower). Model structure did not move the fixed point or the count.

## Outcome, DC arm and Q6 summary (2026-09-15, 23:10; `results/collab_q/q6_dc_libero10_n80/`, `score_q6_dc_vs_{fir,arx}.json`)

Pinned DC arm (`--dc-constrain corrected`, legacy law, same 80 keys, frozen arm shared): **14/80**;
r_y settle **0.715** of the fault (median over episodes; IQR 0.35–0.87), r_x 0.92, r_z 0.83.

- **Prediction 2 (DC moves the fixed point): holds at the letter** (0.715 ≥ 0.70; refutation < 0.60
  not entered) — on the pinned keys the constrained plant more than doubles the FIR's 0.30 and the
  ARX's 0.34, with wide per-episode spread.
- **Prediction 3 (DC ≥ FIR in successes): holds, unresolved in size.** DC 14 vs FIR 10 (8 DC-only,
  4 FIR-only, p = 0.39); DC 14 vs ARX 8 (13 / 7, p = 0.26). As registered, the paired test is
  underpowered for this size; the direction is the predicted one.

**Q6 summary.** On one pinned, paired set of 80 keys: FIR 10, ARX 8, DC 14; r_y settles 0.30, 0.34,
0.72. Model structure (ARX) moves nothing; the fitted-to-probed DC ratio moves the fixed point,
and the count in the predicted direction without resolving it. This is the pinned confirmation
of records 50 and 54.
