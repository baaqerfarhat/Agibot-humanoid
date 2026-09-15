# Preregistration: E2 probe qualification and the constrained predictor C (2026-09-14, before the run)

Per `iclr2027/EXPERIMENT_PLAN.md` E2 preparation, on the partitions of `PREREG_UNIFIED_PARTITIONS.md`.

**Checkpoints (declared by rule, outcomes unseen).** Six: three on the fit partition — (task 0,
state 40) at step 30, (task 4, state 41) at step 70, (task 8, state 42) at step 30 — and three on
the qualification partition — (task 1, state 43) at step 30, (task 5, state 43) at step 70,
(task 9, state 43) at step 30. A checkpoint whose source has fewer than 40 commands left is
missing and stays missing. Tool `openpi/re4_theory/e2_probe.py`: baseline plus 6 axes × 2 signs ×
2 amplitudes (0.01, 0.02) constant command offsets over H = 40 steps, 25 branches per checkpoint,
150 total; responses in the FIR's units (increment / OUT per unit command).

**Qualification rule (in the tool, fixed).** Per axis, a response is a DC response iff (i) the
last-window mean (steps 30–40) is within 20 % of the previous window's (20–30), (ii) the 0.02
response is within 25 % of the 0.01 response per sign, (iii) the two signs agree in magnitude
within 25 % at 0.02. Otherwise it is a finite-horizon response at 40 steps and is named so.

**Predictor C (declared).** C differs from U (the deployed per-axis FIR, K = 6, ridge 0.01,
intercept, fitted on the 30 fit episodes) only by a linear constraint on the rotation-y tap sum.
The constraint value is the mean own-axis r_y last-window gain over the three FIT checkpoints.
It is used only if (a) r_y qualifies as DC on at least two of the three fit checkpoints and (b)
the qualification checkpoints' mean r_y gain is within 25 % of the fit value; if (a) fails but the
fit and qualification finite-horizon values agree within 25 %, C uses the finite-horizon value
and every claim is worded as a matched finite-horizon consistency constraint; if (b) fails, the
constraint cannot be qualified, the E2 core is not launched, and the diagnostic is reported.

**Predictions.** The r_y gain qualifies (settled and linear) at 0.20–0.35 per unit (the historical
probe gave 0.276; the onset transient 0.30); cross-axis entries below 30 % of the own-axis one;
translation own-axis gains 0.20–0.35 on x, y and 0.10–0.30 on z (the historical z entry, 0.126,
was a single-state artefact — if z qualifies near 0.24 on these six checkpoints, that reading is
confirmed). Refutation: r_y not settling within 40 steps on the majority of checkpoints, or a
fit/qualification disagreement above 25 %.

---

## Outcome (2026-09-14, 22:14; `results/iclr_unified_v1/E2_probe/*.json`)

All six declared checkpoints ran (no missing, no domain exits; baseline contact on 16–40 of 40
steps). Own-axis r_y gains per unit command, last window: fit 0.254 / 0.257 / 0.252, qualification
0.285 / 0.251 / 0.221; means **0.254 (fit) and 0.252 (qualification), 0.8 % apart**.

- **Qualification rule, applied.** r_y is a DC response on **1 of 3 fit** checkpoints (the other
  two fail the 20 % settling test or the sign symmetry at 0.02) and on 3 of 3 qualification
  checkpoints. Clause (a) therefore fails; the fit and qualification finite-horizon values agree
  within 25 % (0.8 %), so **predictor C takes the finite-horizon value 0.254**, and every E2
  claim is worded as a *matched finite-horizon (40-step) consistency constraint*, not a DC
  constraint. Clause (b) holds; **the E2 core goes ahead.**
- **Predictions.** r_y gain in 0.20–0.35: yes (0.22–0.29 on every checkpoint; historical probe
  0.276; the deployed FIR's tap sum 0.103 is refuted by fresh probes on six states of three
  tasks). Cross-axis entries below 30 % of the own-axis one: see the table in the JSON (reported,
  not all below on every checkpoint; contact checkpoints couple x with r_y). Translation: x 0.26–
  0.29, y 0.26–0.27, **z 0.29 (fit) and 0.22 (qualification)** — the shipped z entry of 0.126 was a
  single-state artefact, as record 45 read it; r_x 0.26, r_z 0.25, both DC on 6/6.

The constraint value for C is fixed here: **rotation-y tap sum = 0.254**, other channels
unconstrained, everything else identical to U.
