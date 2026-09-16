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
