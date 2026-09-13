# Preregistration: held vs continued updating with matched initialisation (re4 Part F, 2026-09-11)

Written before any run. re4: *"Because initialization and timing both differ, these results do not
isolate the effect of pausing updates."* In the stored cells the continued arm starts from zero and
the held arm from an identified estimate, so the comparison mixes two things. Here every arm starts
**every episode** from the same prior-identification estimate f0, with no carry across episodes;
the arms differ only in what happens after step 0.

**ALOHA** (transfer-cube, frozen π0). Fault +0.02 rad on left-arm joints 0–5, corrected joints 0–5,
the stored cell's constants (γ 0.08, dead 0.002, ρ 0.4, clip 0.08), seed 200, 40 episodes, the
stored plant and M (`results/aloha/healthy_log.json`, `openloop.json`).
f0 = (0.0193, 0.0202, 0.0190, 0.0191, 0.0192, 0.0192) on joints 0–5, zeros elsewhere: the
episode-0 identification stored in `results/aloha/off002_identify1_hold_n40.json`.

**GR1** (plate-to-plate, frozen GR00T N1.5). Fault +0.10 rad on right-arm joints 7–13, corrected
joints 7–13, the stored cell's constants (innovation law, γ 0.08, dead 0.013, ρ 0.11, clip 0.2),
seed 100, 30 episodes, the stored plant and M. f0 = (0.1041, 0.0972, 0.0979, 0.0987, 0.0999,
0.0932, 0.0963) on joints 7–13, zeros elsewhere: the three-episode window median stored in
`results/gr1/p2p_right010_hold3w_paired.json`.

**Arms**, all from f0: (i) held (`--freeze-after 0`); (ii) continued updating with the cell's own
law; (iii) continued updating with the other law. On ALOHA (ii) is the legacy law and (iii) the
innovation law, as the plan names them; on GR1 the cell's own law is already the innovation law,
so (ii) is innovation and (iii) legacy. This is a stated deviation from the plan's wording, made
so that (ii) always isolates pausing under the published law. The frozen-faulted arm is run once
per robot, alongside (i). New runner flags: `--f-init` (every episode starts from the given
vector, no carry) and `--skip-frozen`; both additive.

**Statistic.** Held vs each continued arm, paired by episode key, fixed/broken and exact McNemar.

**Predictions.**
1. ALOHA: held exceeds continued-legacy by ≥ 5 paired successes with p < 0.05. The record's
   mechanism is that the updating law's within-episode movement (0.43 cm) consumes a sub-centimetre
   margin; with the initial estimate matched, any remaining gap is due to updating alone.
2. ALOHA: held exceeds continued-innovation by ≥ 3; the innovation law removes the attenuation
   bias but still moves within the episode.
3. GR1: held exceeds continued (innovation, the cell's own law) by ≥ 4.
4. Arm (i) reproduces the stored held rates within ±4 (ALOHA 15/40, GR1 19/30).

**Refutation.** If a continued arm from f0 reaches held − 1 or better on a robot, pausing is not
what the hold scheme buys there; the stored gain came from initialisation, and the paper's
repairability reading (the correction's within-episode variation must fit the task margin) loses
its isolated support on that robot. That is reported as the primary result for that robot.

---

## Outcome, GR1 half (appended 2026-09-12 after the three GR1 arms; nothing above was edited)

All three arms start every episode from f0 = the stored three-episode window estimate
(0.104, 0.097, 0.098, 0.099, 0.100, 0.093, 0.096 on joints 7–13); frozen-faulted 0/30.

| arm | successes / 30 | vs held (paired by seed) |
|---|---|---|
| (i) held, updates frozen | **22** | — |
| (ii) continued, innovation law (the cell's own) | 17 | held-only 8, continued-only 3, p = 0.23 |
| (iii) continued, legacy law | 1 | held-only 21, continued-only 0, p = 9.5e-7 |

- **Prediction 4 (held reproduces the stored 19/30 within 4): confirmed**, 22/30.
- **Prediction 3 (held ≥ continued + 4): confirmed** on the registered count margin (22 vs 17, a
  gap of 5); the paired test at n = 30 is not significant (p = 0.23), which the prediction did not
  require and is reported anyway. With initialisation matched, pausing the updates is worth about
  five episodes in thirty on the humanoid under its own law.
- **The legacy-law arm is the sharper result, and it is not the attenuation bias.** From a correct
  estimate the legacy law's right-arm estimate falls from 0.092 (step 0) to 0.029 by step 20 and
  0.013 by step 200 — to roughly the deadzone, not to the 50 % attenuated fixed point the paper
  cites for the humanoid. Mechanism: once the correction cancels the fault the residual drops
  below the deadzone, and the legacy law's leakage convention (zero the observation, decay f̂ by
  1 − γ) forgets the estimate; the fault then re-emerges, the residual re-crosses the deadzone, and
  the estimate settles where residual ≈ deadzone. Under that convention a correct estimate is not
  a fixed point. This is why the innovation form (hold below the deadzone) was required on the
  humanoid, stated now as a measured mechanism rather than as the bias argument. The commit
  message for this run (4298223) gave the attenuation reading; this paragraph corrects it.
- The continued-innovation arm's estimate stays at the fault (final median 0.094–0.103 per joint),
  so its five-episode loss is within-episode movement of the correction, the mechanism of
  Proposition 2, not drift.

## Outcome, ALOHA half (appended 2026-09-12 after the three ALOHA arms; nothing above was edited)

All three arms start every episode from f0 = (0.0193, 0.0202, 0.0190, 0.0191, 0.0192, 0.0192) on
joints 0–5; frozen-faulted 0/40.

| arm | successes / 40 | vs held (paired by seed) | within-episode range of the correction (median) | estimate at step 20 |
|---|---|---|---|---|
| (i) held | **15** | — | 0 | 0.0193 |
| (ii) continued, legacy law | 0 | held-only 15, continued-only 0, p = 6.1e-5 | 0.0037 rad (0.40 cm) | 0.0158 |
| (iii) continued, innovation law | **14** | held-only 4, continued-only 3, p = 1.0 | 0.0029 rad (0.30 cm) | ≈ 0.020 |

- **Prediction 4 (held reproduces 15/40 within 4): confirmed**, 15/40 exactly.
- **Prediction 1 (held ≥ legacy-continued + 5, p < 0.05): confirmed**, margin 15, p = 6.1e-5.
- **Prediction 2 (held ≥ innovation-continued + 3): refuted.** 15 against 14, four and three
  discordant, p = 1.0. Updating throughout with the innovation law from a correct estimate
  repairs the task as well as holding it.

**Reading, with the refutation as the primary result.** The record and the paper attribute the
ALOHA failure of continuous adaptation to the correction's within-episode movement (0.43 cm
range against a sub-centimetre margin). These three arms separate two things that were
confounded: movement and bias. The innovation arm moves 0.30 cm within an episode and repairs
14/40; the legacy arm moves 0.40 cm and repairs 0/40. What differs is not the range but where the
estimate sits during the grasp: the legacy law pulls it from 0.019 to 0.016 by step 20 (an
attenuated fixed point plus deadzone leakage once the correction cancels the fault), leaving a
residual of 0.004 rad ≈ 0.42 cm at the gripper, the size of the task margin, exactly when the cube
is grasped; the innovation law's estimate stays at the fault (0.020) and its movement is
zero-mean. So on ALOHA, holding is sufficient but not necessary; what is necessary is that the
correct estimate be a fixed point of the update, which the innovation form satisfies and the
legacy form does not. Proposition 2 survives as stated (σ < μ is necessary, not sufficient; the
legacy arm has σ = 0.40 < 0.5 and fails), but its ALOHA example, "continuous adaptation, no
repair", is a statement about the legacy law and is now labelled as such in the paper. The same
mechanism produced the GR1 legacy arm's collapse (1/30, previous section), at larger scale
because the humanoid's deadzone is larger relative to its residual.
