# Hand-off for `paper_integrated/` (2026-09-16): what landed since its last build, with numbers and paths

All items are committed on `main`; raw runs carry result.json, episodes.csv, run_configuration.json,
telemetry (gzipped) and record.log. Records are in `docs/ADAPTIVE_CONTROL_VLA.md` §§58–62.

## 1. The rotation mask caps libero_10, and six channels lift it (records 58, 61; `PREREG_Q5_SIXCHANNEL_ORACLE.md`, `PREREG_SIX_CHANNEL_LIBERO10.md`)
| libero_10, faulted | rotation mask {3,4,5} (E2 keys) | six-channel mask {0..5} (same keys) |
|---|---|---|
| off / U / C / oracle | 0 / 18 / 25 / 36 | 0 / **35** / **41** / **57** of 60 |
| healthy off / U / C | 56 / 53 / 52 | 56 / 47 / 53 |
- C6 − C(rotation) = +16 episodes (22 won / 6 lost, p = 0.004); oracle − C6 +26.7 points [+15.0, +38.3];
  C6 − U6 +10.0 points [−1.7, +21.7]. Healthy U6 −15.0 points [−26.7, −3.3], C6 −5.0 [−16.7, +3.3].
- Registered decision: adoption refused on the healthy criterion (U6 10 lost, C6 6 lost); the translation
  phantom is z ≈ 0.019 median with episodes at the 0.15 clip. M's translation diagonal was the E2 probes'
  (0.257 / 0.263 / 0.291), not the shipped state-45 values.
- Q5 (D3a keys, unpinned): six-channel oracle 38/40 vs rotation-only 22/40 (17 to 1).
- Figure `results/iclr_unified_v1/figures/libero10_mask.pdf`; scores `results/six_channel/score_libero_10_six.json`.

## 2. Six channels under the healthy-only gate (record 62; `PREREG_SIX_CHANNEL_GATE.md`)
Gate statistics from 20 healthy estimate-only episodes on the unused state 39: 3-sd thresholds
0.107 / 0.049 / 0.066 on x / y / z (at or above the 0.05 fault on x and z), 0.013–0.018 on rotation.
Healthy C6 + gate **54/60** (3 lost, 1 gained; translation open on 10–12 % of healthy steps).
Faulted C6 + gate: **32/60** (registered ≥ 36 adopts, ≤ 30 refutes: inconclusive, not adopted); vs ungated 41: 6 / 15,
p = 0.078; vs rotation-mask C 25: 18 / 11, p = 0.27; translation open under the fault 31 / 57 / 34 % on x / y / z
(thresholds on x and z exceed the fault). The three configurations on the same keys, faulted / healthy: rotation
mask 25 / 52, six channels 41 / 53, six channels gated 32 / 54.

## 3. E1 on ALOHA, twenty fresh sources (record 60; `PREREG_E1_ALOHA_CONTINUATIONS.md`; `results/iclr_unified_v1/E1_aloha_v2/`)
Ten locked sources, twenty checkpoints, 50-step continuations, +0.02 rad on the left arm: faulted
deviation saturates at the offset's norm (endpoint 1.01× its step-20 value; Panda 1.8×); both laws
cancel 94–97 % within 50 steps (Panda 55 %); geometric decay λ = 0.66 beats pure accumulation by
0.88 rad·step with source-level intervals far from zero (Panda: accumulation as good, λ → 1). One
registered letter fails: the innovation branch's endpoint-gap/remaining ratio 1.43 vs 1.30, at a
0.0014 rad remainder. Figure `results/iclr_unified_v1/figures/e1_contrast_panda_aloha.pdf` (Panda
E1 v2 left, ALOHA right). The eight-episode first cut (record 60) is not pooled with it.

## 4. E2 mechanism figure
`results/iclr_unified_v1/figures/e2_mechanism.pdf`: r_y estimate (median, IQR) and remaining r_y
disturbance over time for U and C on both suites, outcome counts and the two bias statistics annotated.

## 5. Matched healthy-adaptive controls for the transfer table
`results/iclr_unified_v1/transfer_healthy_controls.md`: OFT 20 → 19, GR00T N1.7 18 → 18, ALOHA
17 → 14 (held phantom), GR1 18 → 19 and 15 → 16, WidowX 15 → 16, with protocols and files.

## 6. Two corrections to absorb
- Q6's fixed-point statistic: median r_y settle 0.715 (holds ≥ 0.70) and mean 0.613 (misses); the prereg
  wording admitted both, the record and the manuscripts use the stricter mean (`PREREG_Q6_PINNED_FIR_ARX_DC.md`).
- Q2 joint-5 cell: D 30 vs C 27 of 40, paired on the full key 7 D-only / 4 C-only, p = 0.55 (the first
  numbers written into the prereg were not the scorer's; corrected in place).
- The package's evidence manifest predates the Q6 amendment: `build.py --check-only` fails on the changed
  prereg bytes until the receipt is regenerated (the rebuild needs your cached fonts; mine lacked cmex9).

## 7. Suggested placement in C
Section "Physical deviation and correction authority": the six-channel table (item 1) replaces the
one-sentence Q5 mention; the gate result (item 2) as the next sentence; the ALOHA panel (item 3) as the
second half of the continuation paragraph with the contrast figure; item 4 in the predictor section;
item 5 into the transfer table's Healthy column with the caption caveat removed. The ninth page is free.
