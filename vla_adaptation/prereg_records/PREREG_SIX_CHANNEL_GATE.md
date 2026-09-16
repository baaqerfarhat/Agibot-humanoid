# Preregistration: the six-channel configuration under the healthy-only channel gate (2026-09-16, before the runs)

**Why.** Record 61: correcting all six channels online lifts libero_10 from 25 to 41 of 60 (C6) but
loses 6 (C6) to 10 (U6) healthy episodes in 60 through the translation phantom (z ≈ 0.019 median,
episodes at the clip). Record 37 built a gate from healthy data alone for exactly this: channel i
is corrected at a step iff |f̂_i − b_i| > 3·max(sd_i, 0.002), with b and sd the mean and
across-episode sd of the healthy estimate over the last 50 steps.

**Design.** Three runs on GPU 1 after the six-channel oracle arm, own server.
1. *Gate statistics*: 20 healthy estimate-only episodes (`--estimate-only --sev 0`) under the C6
   configuration (innovation law, six-channel mask, E2 fit partition, `--dc-gain 4=0.254`, probed
   translation M entries) on **libero_10 state 39, ten tasks × two sampler seeds** (manifest
   `libero_10_gate_healthy20.json`, state 39 unused by any stored run; seeds 2000+), then
   `gate_stats.py --last 50 --sd-floor 0.002` → `results/six_channel/phantom_stats_six.json`.
   Healthy data only; the evaluation keys (states 41, 40) are untouched.
2. *Healthy C6 + gate* and 3. *faulted C6 + gate* on the 60 E2 libero_10 keys with their sampler
   schedule (`--gate-stats … --gate-k 3`, which replaces the fixed mask), frozen arms shared with
   E2. 140 rollouts. Gate occupancy per channel is recorded in the trajectories.

**Predictions.**
1. Healthy C6 + gate loses **≤ 3 of 60** against healthy off (56); refutation ≥ 6 lost (the gate
   does not hold the translation phantom on this suite either).
2. Faulted C6 + gate **≥ 36/60**; refutation ≤ 30/60 (the gate closes translation so often that
   the six-channel gain is given back; the occupancy will say so).
3. Gate occupancy after step 20: translation channels open on < 30 % of healthy steps and > 60 %
   of faulted steps.
**Decision.** If 1 and 2 hold, the gated six-channel configuration is the paper's libero_10
result (25 → ≥ 36 with healthy within three); otherwise the ungated numbers of record 61 stand as
a demonstration of authority with its cost stated.

---

## Outcome, gate statistics and the healthy arm (2026-09-16, 07:30; `results/six_channel/{libero_10_gatestats_healthy20,phantom_stats_six.json,libero_10_healthy_C_gate}`; faulted arm running)

Gate statistics (20 healthy estimate-only episodes, state 39, six-channel C6 configuration): healthy
phantom means b = −0.015 / −0.011 / +0.013 / +0.001 / −0.007 / +0.000 and across-episode sd 0.036 /
0.016 / 0.022 / 0.006 / 0.005 / 0.005 on x / y / z / r_x / r_y / r_z, so the 3-sd thresholds are
**0.107 / 0.049 / 0.066** on translation — at or above the 0.05 fault on x and z — and 0.018 / 0.014 /
0.013 on rotation. Stated before the arms ran: a healthy-only gate cannot separate a 0.05 translation
fault from the healthy phantom on x and z on this suite; the faulted arm tests what that costs.

- **Prediction 1 (healthy ≤ 3 lost): holds.** Healthy C6 + gate **54/60** against 56 (3 lost, 1 gained;
  ungated C6 lost 6, U6 lost 10). Healthy phantom under the gate: z +0.014 median, translation channels
  open on **10–12 %** of steps after step 20 (registered < 30 %: the healthy half of prediction 3 holds),
  rotation channels on 4–20 %.
- Prediction 2 and the faulted half of 3 wait on the faulted arm.

## Outcome, faulted arm and decision (2026-09-16, 09:18; `libero_10_faulted_C_gate`)

Faulted C6 + gate: **32/60**. Against the ungated C6 (41): 6 gated-only, 15 ungated-only (p = 0.078) —
the gate gives back nine of the sixteen episodes six channels had gained; against the rotation-mask C
(25): 18 / 11 (p = 0.27), still +7. Estimates 81 / 71 / 123 / 88 / 78 / 95 % of the fault.

- **Prediction 2 (≥ 36): not reached; refutation (≤ 30) not entered** — 32 sits in the inconclusive band.
- **Prediction 3, faulted half (translation open > 60 %): fails on x and z, holds on y** — open fractions
  after step 20 are 0.31 / 0.57 / 0.34 on x / y / z (rotation 0.94–0.98). The thresholds on x (0.107)
  and z (0.066) exceed the 0.05 fault, so the gate keeps those channels closed on two thirds of faulted
  steps; that is where the nine episodes went.

**Decision, as registered:** predictions 1 holds, 2 inconclusive, 3 split → the gated configuration is
**not adopted**; the ungated numbers of record 61 stand as the demonstration of authority with their
healthy cost, and the gated arm is reported alongside as what a healthy-only gate can and cannot do
here: it holds the healthy loss to three (from six) and keeps seven of the sixteen extra faulted
successes, because a healthy-data threshold cannot be below a phantom that is as large as the fault.
Not registered, noted for the paper's limitations: the pair (faulted 32, healthy 54) improves on the
rotation-mask pair (25, 52) on both counts and would be the deployment choice among the three; the
registered bands, written for a larger effect, do not resolve it at n = 60.
