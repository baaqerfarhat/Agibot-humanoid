# re4 theory evidence — index, status, and deviations from the plan

Plan: `docs/RE4_THEORY_EVIDENCE_PLAN.md`. Every part is preregistered in
`prereg_records/PREREG_RE4T_*.md` (statistic, band, refutation written first; outcomes appended
after). Tools: `openpi/re4_theory/`. GPU chains: `scripts/re4/re4t_*.sh`. Runs follow the
section-0 logging contract of the first plan (`openpi/re4_record.py`).

| part | folder | status |
|---|---|---|
| T.0 paired-rollout driver | `openpi/re4_theory/paired_rollout.py` | done; bit-identical restore verified |
| 1 sampled contraction and input sensitivity (Panda) | `1_contraction/` | done: λ̂ = 1.00, marginal, not contracting (record 48) |
| 2.1 servo poles (GR1, ALOHA) | `2_metric/metric_certificate_{gr1,aloha}.json` | done: GR1 right arm certified; ALOHA refuted on most joints |
| 2.2 Panda tracking metric | `2_metric/metric_certificate_panda.json` | done: all six axes certified |
| 3 tube propagation on real episodes | `telemetry/T1_headline/part_3.json` | done: coverage 1.00 but vacuous (bound 98× the measurement; λ̂ = 1) — record 49 |
| 4.1–4.4 estimator recursions | `telemetry/T{2,1,1,3}_*/part_4.x.json` | done: 4.2, 4.3, 4.4 confirmed; 4.1's registered statistic fails (noise), episode-mean response right on r_x, r_z and 3× on r_y — record 49 |
| 5 small-gain constants | `telemetry/T1_headline/part_5.json` | done: fails, a = 1 − λ̂ = 9e-5 (margin −9.5e-4) — record 49 |
| 6 r_y as a theory test | `6_ry/` | diagnosis done (model error); ARX intervention running |
| 7 composite on GR1 (stretch) | — | not started; the first-order channel it needs is certified (2.1) |
| 8.1 task-clustered bootstrap | `8_statistics/clustered_bootstrap.json` | done |
| 8.2 seed-pinned decision cells, paired integral | `8_statistics/` | queued behind Part 6 |
| 8.3 recovery vs geometric decay | — | settled by Part 1: no decay time exists when λ̂ = 1 |

## Deviations from the plan, stated

1. **Part 1's metric list was fixed in advance** (joint, joint+velocity, end-effector, end-effector
   in the 2.2 metric). λ̂ = 1.00 in all four; per the registered rule no further metric was tried.
2. **Part 2.1 on ALOHA used the healthy logs, not step probes**, because the stored probe files hold
   only M. The first-order route fails there at 50 Hz; a sim-only step probe would separate model
   from excitation and is not run in this pass.
3. **Part 6's intervention was changed before any run**, from r_y excitation to an ARX plant, after
   the CPU diagnosis ruled out excitation and sensitivity. The amendment is in the prereg with the
   diagnosis that motivated it.
4. **Part 3's "nominal replay"** is the FIR prediction from the corrected commands, i.e. the logged
   residual, in the Part 2.2 metric; with λ̂ = 1 the tube recursion is driven by L and η only.
5. **Part 8.2's pinning** uses the server's `pin_rng` (one sampler key per call), so runs are paired
   on (task, init, draw) up to their first divergence, recorded per episode from telemetry.
