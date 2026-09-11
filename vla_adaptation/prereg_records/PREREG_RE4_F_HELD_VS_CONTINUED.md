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
