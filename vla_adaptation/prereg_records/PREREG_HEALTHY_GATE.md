# Preregistration: a channel gate from healthy data alone (2026-09-10)

**Written before the healthy phantom is measured and before the gated cell runs.** Filled-in
numbers below are marked TBD and will be entered from the phantom run before the gated cell
starts; the rule and the constants that are not phantom-derived are fixed here.

## The objection this answers

Dual-track audit §4.5 (`report/DUAL_TRACK_AUDIT.md`): the headline cells correct rotation
only, and that restriction came from a separation test run on faulted rollouts. The paper's
claim "no faulted data" is therefore not true end to end. Correcting all six channels on the
uniform fault works on one suite of four (record §19.3), so the restriction is load-bearing.

## The rule

Per channel i, at every control step t of every episode, with the healthy phantom's
across-episode mean b_i and standard deviation sd_i measured on healthy episodes only:

    corrected_i(t) = 1  if  |f_hat_i(t) − b_i| > k · sd_i,  else 0,     k = 3

No hysteresis, no memory across episodes; the mask is re-evaluated every step. The applied
correction is −(f_hat ⊙ corrected). The estimator itself runs on all six channels exactly as
in the headline cell (legacy law, gamma 0.08, dead 0.008, norm_r 0.15, clip 0.15, replan 5,
plant `results/phase05/error_signal_so3.json`, M `results/phase05/openloop_so3.json`). Nothing
in the rule uses a faulted rollout.

**Phantom statistic.** 20 healthy episodes, `--estimate-only`, all six channels, same
constants; b_i and sd_i are the mean and sd over episodes of the per-episode mean estimate
over the last 50 steps (the §29.2 statistic). A floor of sd_i >= 0.002 is applied so a
channel with a near-zero phantom sd cannot be gated open by numerical noise.

## The cell

π0.5, `libero_spatial`, uniform +0.05 on all six action dimensions, 20 paired (task, init)
scenarios, inits 45/46 (the headline cell's scenarios), frozen-faulted vs gated-adaptive,
plus a healthy arm with the gate running (the healthy control).

## Predictions, registered

1. **Primary.** The gated cell repairs at least 14/20 (frozen ~8/20 on this cell historically),
   with at most 1 broken. If it reaches the rotation-only headline (18/20, 10 fixed / 0 broken)
   the "no faulted data" claim becomes true end to end. Below 14/20 the claim is withdrawn
   from the paper and replaced by the stated limitation that the channel restriction needs
   one faulted separation test.
2. **Mechanism.** Rotation channels open the gate on >80% of steps after step 20; translation
   channels open on <20% of steps. Under the uniform fault the record says translation is
   under-identified (an estimate near the healthy phantom), so k=3 on translation should stay
   mostly closed. If translation opens often AND the cell still repairs, the earlier claim
   that translation correction hurts here (§19.3) is what needs revisiting.
3. **Healthy control.** With the gate running on healthy episodes, the correction is applied
   on <5% of steps and the healthy rate is unchanged (paired against healthy-no-law on the
   same scenarios; 0 or 1 discordant either way).

## What would refute what

- Gated cell < 14/20 or >= 3 broken: the gate does not replace the faulted separation test.
- Rotation gated open < 50% of steps: the phantom sd on rotation is too large for k=3 to
  admit a 0.05 fault, i.e. the healthy phantom is not a usable reference; report and stop.
- Healthy control loses >= 3 episodes: the gate opens spuriously; report as harm.

No constants will be changed after the phantom numbers are read. If k must change, that is a
second preregistration, not an amendment.

---

## Outcome (appended 2026-09-10, after the runs; nothing above was edited)

Phantom (20 healthy scenarios, healthy 20/20 both arms): b = (+0.015, −0.003, +0.029, +0.000,
−0.000, +0.001), sd = (0.039, 0.015, 0.040, 0.003, 0.013, 0.002 floor).

- **Prediction 1: confirmed.** Gated cell 19/20 vs frozen 9/20, 10 fixed / 0 broken,
  p = 0.0020 (threshold was ≥ 14/20 with ≤ 1 broken; headline was 18/20).
- **Prediction 2: half refuted.** After step 20, rx and rz open on 100 % of steps, ry on 28 %
  (predicted > 80 % for rotation: false for ry). x 17 %, z 20 %, y 36 % (predicted < 20 % for
  translation: false for y). The cell repaired anyway; the record §37 says what that means.
- **Prediction 3: half refuted.** Healthy no-law 20/20, gated 19/20, 0 fixed / 1 broken,
  p = 1.0: within the registered "0 or 1 discordant". But the gate opened on 30 % of healthy
  steps (any channel), not < 5 %: the 3 sd threshold was calibrated on a 50-step mean and is
  read on the instantaneous estimate. Recorded in §37 with the one lost episode.

No constant was changed. k = 3 stays.
