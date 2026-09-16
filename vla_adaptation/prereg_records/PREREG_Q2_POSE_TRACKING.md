# Preregistration: pose-referenced tracking in closed loop (queue item Q2)

**Written 2026-09-15, before any run. Requires (a) lambda to be free and (b) the collaborator's
agreement, because it uses his runner and shares the scenario keys of his E2 confirmation manifest.**

**Question.** The isolating replay test (DUAL_TRACK_C2.md) showed a leaky position-mode tracking term
on r_x and r_z cutting the retained orientation offset by 22 % and 41 % in open loop. The frozen policy
does not compensate r_x or r_z (two instruments), so the effect should survive closed loop. Does it,
and at what healthy cost? Task success is secondary: at most about +4/40 is available on libero_10 and
the paired test has 11 % power at n = 40 for that effect.

**Cells and arms.** Primary cell: libero_10, uniform command offset 0.05, the 60 keys of
`results/iclr_unified_v1/manifests/libero_10_E2_core.json` with its sampler-seed schedule.
- OFF: frozen, shared (reuse E2's if run on the same keys).
- C: `--law innov --dc-constrain corrected`, shipped constants, `--arms adaptive`.
- D: C plus `--track-kappa 0.001 --track-mode position --track-anchor 0 --track-dims 3,5
  --track-ref dc --track-leak 0.02 --track-obs tracked`.
- HC, HD: C and D with `--sev 0` on 20 of the keys (the first two per task).
Secondary cell: spatial joint-5 torque (`--joint-fault torque:5:5.0`), 40 keys, C and D only.
Budget ≈ 60·3 + 40 + 40·2 = 300 rollouts, plus OFF if not shared. Telemetry on for every arm.

**Primary endpoint (defined before any data).** For every adaptive episode, recompute offline from its
telemetry the leaky pose excess e_p on r_x and r_z exactly as the tracker defines it (reference =
the DC-constrained plant driven by `raw_action`, leak 0.02, anchored at the first rollout step) — the
same computation for C, where no tracker runs, as for D. Statistic: the late-window mean |e_p| over
steps ≥ 100 (or the last half of shorter episodes), per channel, averaged over episodes; ratio
R = D / C per channel on libero_10.

**Predictions.**
1. **R ≤ 0.80 on r_z and on r_x** (replay: 0.59 and 0.78). **Refutation: R ≥ 0.95 on r_z.** An r_x
   ratio between 0.80 and 0.95 with r_z ≤ 0.80 is reported as a partial confirmation.
2. **Healthy cost:** HD loses at most 2 of 20 episodes against HC, and HD's late |e_p| on r_x, r_z
   exceeds HC's by at most 0.02 normalised units (replay: +0.006).
3. **Specificity:** the r_y settle (untracked) is within ±0.10 between C and D.
4. **Success (secondary, power stated above):** D − C on libero_10 within [−2, +4] of 60. A difference
   ≥ +7 with a task-clustered interval excluding zero is registered as a surprise, not a confirmation
   of mechanism. Joint 5: D − C within ±3 of 40 (predicted null; its fault is translation-dominant).

**Analysis.** Paired by scenario key; per-channel ratios with a whole-episode bootstrap interval;
success by exact McNemar and the task-clustered bootstrap used in record 51.

---

## Outcome, primary cell (2026-09-15, 18:46; run by Mahdi's session on GPU 0, shared; `results/collab_q/q2_{C,D}_libero_10/`, `score_q2_libero_10_partial.json`; healthy and joint-5 cells pending)

libero_10, the 60 E2 keys with their sampler schedule; OFF shared with E2 (`libero_10_faulted_off`,
0/60). C = innovation law, all-channel normaliser, `--dc-constrain corrected`; D = C + the
position-mode tracker on r_x, r_z (κ 0.001, leak 0.02, anchor 0, obs tracked). The primary
endpoint was recomputed offline from both arms' telemetry with the tracker's own definition
(`openpi/re4_theory/q2_score.py`), late window steps ≥ 100 (median episode 520 steps).

| channel | late |e_p| C | late |e_p| D | R = D / C | whole-episode 95 % interval |
|---|---|---|---|---|
| r_x | 0.0505 | 0.0461 | **0.913** | [0.796, 1.052] |
| r_z | 0.0802 | 0.0654 | **0.815** | [0.696, 0.943] |

- **Prediction 1: neither confirmed nor refuted at the letter.** r_z = 0.815 misses the registered
  ≤ 0.80 by 0.015 and stays well below the 0.95 refutation (its interval excludes 1: a real
  ≈ 18 % reduction); r_x = 0.91 sits in the 0.80–0.95 band with its interval including 1. The
  replay pre-test's 0.59 / 0.78 is not reached in closed loop; the tracked r_z offset falls by a
  fifth, the r_x offset by a tenth at most.
- **Prediction 3 (specificity): holds.** r_y settle 0.0434 (C) vs 0.0435 (D), untouched.
- **Prediction 4 (success, secondary): holds.** D 27/60 vs C 25/60, +2 (7 D-only, 5 C-only,
  McNemar p = 0.77, task-clustered interval [−6.7, +13.3] points), inside the registered
  [−2, +4]; no surprise.
- Prediction 2 (healthy cost) waits on HC/HD, running.

Reading before the healthy cell: the tracking term does in closed loop a smaller version of what
it did in replay — it trims the retained r_z pose offset by about a fifth without touching r_y or
task success. Against Q5 (the libero_10 gap is translation, which this term cannot reach) it is a
mechanism result, not a task lever.

## Outcome, healthy cell (2026-09-15, 19:38; `q2_HC_libero_10`, `q2_HD_libero_10`, 20 keys; `score_q2_libero_10.json`)

HC 18/20, HD 18/20 (1 lost, 1 gained); late |e_p| HC 0.0151 / 0.0144, HD 0.0147 / 0.0155 on r_x / r_z
(+0.001 at most). **Prediction 2 holds:** the tracker costs nothing on healthy episodes at this
gain and leak. Joint-5 cell running.

## Outcome, joint-5 cell and Q2 summary (2026-09-15, 20:38; `q2_{C,D}_spatial_joint5`, 40 keys; `score_q2_joint5.json`)

Spatial, +5 N·m joint-5 torque, C 27/40, D **30/40** (paired on (task, init, sampler seed): D-only 7,
C-only 4, exact McNemar p = 0.55; *correction 21:55: the numbers first written here, 5 / 2 / 0.45, were
not from the scorer — the scorer's own first output had collapsed the 40 keys to 20 (task, init) pairs;
the 40-key pairing is in `score_q2_joint5.json`*):
D − C = +3, at the edge of the registered ±3 null band — **prediction 4's joint-5 clause holds**
(predicted null; the fault is translation-dominant).

**Q2 summary.** Primary endpoint neither confirmed nor refuted (r_z 0.815 [0.70, 0.94], r_x 0.91
[0.80, 1.05]); specificity holds; healthy cost none (18/20 both, excess unchanged); success null
on both cells (+2/60, +3/40). The pose-tracking term trims the retained r_z offset by about a
fifth in closed loop with no cost and no task effect — a mechanism result. Its task ceiling on
libero_10 is set by the rotation mask (Q5: six-channel oracle 38/40), which tracking on rotation
channels cannot move. All Q2 runs on GPU 0 shared with another user's training; the arms' keys and
sampler schedule are E2's.
