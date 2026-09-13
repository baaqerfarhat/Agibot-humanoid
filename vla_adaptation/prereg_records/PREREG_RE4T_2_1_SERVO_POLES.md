# Preregistration: per-joint first-order servo poles on GR1 and ALOHA (re4 theory plan, Part 2.1; 2026-09-13)

**Written before the fit.** The stored open-loop probe files hold only the sensitivity matrix
M, not per-step responses, so the poles are fitted from the healthy logs' per-step commands and
positions (`results/gr1/screen_PosttrainPnPNovelFromPlateToPlateSplitA.json`, 6 episodes at
20 Hz, 29 joints; `results/aloha/healthy_log.json`, 8 episodes at 50 Hz, 14 joints), then checked
on held-out episodes. A fresh sim-only step probe is a later validation, not this item.

**Model.** Per joint j, first-order lag on the absolute target: q_{k+1} − q_k = a_j Δt (u_k − q_k),
fitted by least squares over all steps of the fitting episodes; λ̂_j = 1 − a_j Δt is the sampled
pole; metric M_c = I (per-joint). Leave-one-episode-out: fit on the other episodes, evaluate on
the held-out one; report the one-step R² on increments and the free-running k-step prediction
error (k = 10) from the true start state.

**Domain.** A joint enters the certificate if λ̂_j ∈ (0, 1) on every fold and the held-out
one-step increment R² ≥ 0.5; joints that fail are listed as outside the domain with the reason.

**Predictions.**
1. GR1 right arm (joints 7–13): λ̂ ∈ [0.3, 0.95] on all seven; ALOHA arm joints (0–5, 7–12):
   λ̂ ∈ [0.5, 0.98] on all twelve (a 50 Hz servo is closer to unity per step).
2. Grippers (ALOHA 6, 13; GR1 hands 14–25) are outside the domain or have R² < 0.5: they are
   not position servos on a continuous target.
3. Held-out increment R² ≥ 0.5 on every arm joint that enters the domain; free-running 10-step
   error ≤ 3× the one-step error.

**Refutation.** Any arm joint with λ̂ ≥ 1 or ≤ 0 on a fold, or held-out R² < 0.5, means the
first-order-lag route does not certify that channel; it is reported outside the domain and the
paper keeps "unverified" for it. No alternative model is fitted post hoc.

**Artifact.** `results/re4_theory/2_metric/metric_certificate_<robot>.json`: per-joint a, λ̂,
fold spread, held-out R², domain membership, source hashes.

---

## Outcome (appended 2026-09-13, after the fit; nothing above was edited)

**GR1 (20 Hz, 6 episodes).** The seven right-arm joints are all certified: poles 0.724–0.766,
fold spread ≤ 0.02, held-out one-step increment R² 0.74–0.95, 10-step free-running error
1.5–3.4× the one-step error (joint 12 at 3.4 just misses prediction 3's 3×). Waist yaw (26) is
certified (pole 0.764, R² 0.75). The left arm (0–6) fits the same poles (0.75–0.78) but its
held-out R² is 0.13–0.45, outside the domain: it barely moves in this task, so the increments
are noise-sized; the pole is plausible but not certified. All twelve hand joints are outside
(poles 0.99–1.00, R² ≤ 0.17), as predicted. **Predictions 1 and 2 confirmed for GR1; 3 confirmed
on six of seven arm joints.**

**ALOHA (50 Hz, 8 episodes): prediction 1 refuted.** Only joints 1, 7, 8 (and the continuous
left gripper 6) are certified. Joint 5 fits a pole of 4.3 (a = −165/s: the wrist barely moves and
u − q is noise-sized), joint 12 a pole of 1.1, and joints 0, 2, 3, 4, 9, 10, 11, 13 have held-out
R² from −2.5 to 0.42. A first-order lag on the absolute target does not describe this 50 Hz servo
from closed-loop policy data: the increments per 20 ms step are one to two orders of magnitude
below the command-to-position gap, and the FIR of the paper (six taps, R² 0.99 on position but
0.53 on increments, G.3) is the better description. Per the refutation rule no other model is
fitted here; the ALOHA channel keeps "unverified" on this route. A sim-only step probe on the
ALOHA servo (a later item) would test whether the failure is the model or the excitation.

Artifacts: `results/re4_theory/2_metric/metric_certificate_{gr1,aloha}.json`.
