# Pre-registration — the innovation-form law on ALOHA

**Written 2026-09-07, BEFORE the port and BEFORE any run.** `openpi/aloha_adapt.py` currently
has no `--law` flag at all; only the legacy form is implemented there. Nothing below has been
executed. Written after the norm-channel hypothesis was measured and refuted
(`PREREG_ALOHA_NORM_CHANNELS.md`), so this is a second, independent attempt with a different
mechanism.

---

## 1. The gap, established from the record and from Arm A

The law ships in two forms, and only one has ever run on ALOHA.

| form | update | fixed point |
|---|---|---|
| **legacy** (all ALOHA runs, and the LIBERO default) | `est = M⁻¹r`, EMA toward it | `f/(1+‖Mf‖²/ρ²)` — **biased low** |
| **innov** (`--law innov`, LIBERO only) | drives `e = r − M f̂` to zero | `f` — unbiased |

- The bias was identified by the external review (`report/REVIEW.md`), sized by the authors at
  ~4.9%, and `--law innov` was shipped as the fix (`report/RESPONSE.md` §1).
- It was tested **only on LIBERO**, where §15 of the record reports it "does not measurably
  help." LIBERO is where the legacy law already succeeds, so that test had no headroom.
- `report/RESPONSE.md` still lists "`--law innov` head-to-head against legacy" as owed.
- **`aloha_adapt.py` has no `--law` flag**, so the fix has never been available on the robot
  where the legacy law fails.

## 2. Why the bias should matter here specifically

Two measurements, both already in hand, that make this a sized prediction rather than a hope:

1. **Arm A (this session, fresh run):** the legacy estimate converges to **0.019** against a
   true **0.020** on all 20 episodes — 95%, exactly the predicted ~5% low bias.
2. **The authors' own static sweep (§27):** a **0.019** correction held for the episode gives
   **5/20**; the exact **0.020** gives **8/20**.

So on this task a 5% estimation bias is worth **3/20 of task success**, measured. And the law
that removes precisely that bias has never been run here.

## 3. Hypothesis

**H1.** On ALOHA, the legacy law's normalisation bias is the difference between matching the
healthy rate and exceeding it. The innovation form removes the bias, so its held estimate
lands at ~0.020 rather than ~0.019, and identify-then-hold success rises accordingly.

**H0 (refutation).** The innovation form's held estimate does **not** land closer to 0.020 than
the legacy form's, or it does and success does not move.

## 4. Arms

`gym_aloha` transfer-cube, `pi0_aloha_sim`, +0.02 rad on left-arm joints 0–5,
`--corr-joints 0,1,2,3,4,5 --norm-r 0.4 --dead 0.002 --gamma 0.08 --clip 0.08`,
**identify-then-hold** (`--identify-episodes 1`), n = 20 paired, seeds fixed, matched
`(task, init)`.

| arm | law | role |
|---|---|---|
| **E** | `legacy` (default) | reproduce the published identify-then-hold cell, ~5/20 |
| **F** | `innov` | the treatment |

Reference points already in the record and not rerun: 0.019 static → 5/20; 0.020 exact → 8/20;
healthy frozen → 5/20 at n=20, 17/40 at n=40.

## 5. Predictions, registered in advance

1. **Arm E reproduces 5/20** (tolerance 3–8/20 given the documented ±10 points at n=20). If E
   does not reproduce, the comparison is void.
2. **Primary endpoint — the held estimate.** Arm F's held `f̂` on joints 0–5 lands at
   **≥ 0.0195 rad** (within 2.5% of truth), against Arm E's ~0.019. This is the mechanism check
   and it is nearly deterministic if the algebra is right; it does not depend on task noise.
3. **Secondary — success.** Arm F reaches **≥ 7/20**, against Arm E's ~5/20. Stated with low
   confidence: n = 20 cannot resolve 5/20 vs 8/20 (exact McNemar on the discordant pairs would
   need a lopsided split), so this is directional evidence, not a test.
4. **If prediction 2 holds and 3 fails**, the honest conclusion is that the bias is real,
   removable, and *not* what bounds this task — which is itself a result and directly supports
   the margin story of §27.

## 6. Analysis, fixed in advance

- Primary: mean held `f̂` over joints 0–5, Arm F vs Arm E, reported with the per-episode spread.
- Secondary: paired success via `openpi/mcnemar_crosscheck.py` **and** `openpi/mcnemar.py`.
- Telemetry recorded for both arms.
- **The port must not change legacy behaviour.** Arm E must be bit-identical in arithmetic to
  the pre-port code under default flags, verified against git HEAD before either arm runs.
- No constant is retuned. No arm is dropped after seeing its result.

## 7. Stated limits, before the fact

- One robot, one task, n = 20. This can establish a mechanism, not generality.
- Pairing is on `(task, init)` only; `pin_rng=False`, so policy sampling noise is not matched
  (`report/DUAL_TRACK_AUDIT.md` §4.10).
- §27 showed continuous adaptation fails even warm-started at the correct value, so any benefit
  here is expected **only** in the hold scheme, not in continuous adaptation.
- The prior attempt in this line (`PREREG_ALOHA_NORM_CHANNELS.md`) was refuted by measurement.
  The base rate for these hypotheses in this project is therefore poor, and prediction 3 in
  particular should be read as a long shot with a measured effect size behind it, not a
  forecast.

---

# RESULT — the bias is removable, and removing it does not repair the task

**Added 2026-09-07, after Arms E and F.** Artifacts: `results/norm_channels/armE_legacy.json`,
`armF_innov.json`.

## Scoring, as registered

| prediction | registered | measured | verdict |
|---|---|---|---|
| 1. E reproduces 5/20 (3–8 tolerated) | 5/20 | **6/20** (0→6, p=0.031) | **satisfied** |
| 2. F held `f̂` ≥ 0.0195 rad | ≥0.0195 | **0.02058** | **satisfied** |
| 3. F ≥ 7/20 | ≥7/20 | **2/20** | **refuted** |
| 4. if 2 holds and 3 fails → the bias is not what bounds this task | — | **this is the outcome** | — |

## What is established

**The bias is real, and the innovation form removes it.** The legacy law holds 0.01870 rad
against a true 0.020 (93.5%); the innovation law holds 0.02058 (102.9%). Mean absolute
estimation error falls **34%**, from 0.00130 to 0.00085 rad. This was predicted from theory
(the selftest's synthetic fixed points are 0.019000 legacy, 0.019591 innov at `dead=0.002`),
it reproduces in `gym_aloha` simulation (NOT on a real robot — corrected 2026-09-07).

**And repair does not improve.** Success went 6/20 → 2/20. **This difference is NOT
statistically resolved**: paired on matched `(task, init)`, the discordance is 5 legacy-only
against 1 innov-only, exact McNemar **p = 0.219**. The direction is worse; the magnitude is
not established at n = 20. **No claim is made that the innovation law harms the task.**

What *is* established is the conjunction registered as prediction 4: **a 34% reduction in mean
estimation error produced no improvement in task success.**

## The candidate mechanism, measured but not causally established

| | E legacy | F innov | change |
|---|---|---|---|
| mean \|error\| vs truth | 0.00130 rad | 0.00085 rad | **−34%** |
| per-joint spread (max−min of means) | 0.00169 rad | 0.00397 rad | **2.4×** |
| worst-joint deviation | 0.00261 rad (0.28 cm) | 0.00334 rad (0.35 cm) | +28% |

The innovation form is more accurate *in aggregate* and more dispersed *across joints*. A
uniform joint offset partly cancels at the end-effector; a dispersed one does not, and the task
margin is under 0.5 cm. That is a plausible account of why better mean accuracy did not help —
**and it is not established**, because the success difference it would explain is itself
p = 0.219. Testing it needs n ≥ 40 and a direct manipulation of dispersion at fixed mean.

## Why this is worth reporting anyway

It is the **second independent demonstration** that estimator accuracy is not what bounds repair
on this task, by a different route from §27's static sweep, with a clean single-flag
intervention: same law, same constants, same seeds, one argument changed. It supports
Proposition 2 rather than undermining it, and it closes the head-to-head that
`report/RESPONSE.md` has listed as owed since 2026-09-01 — on the robot where it was actually
informative, rather than on LIBERO where the legacy law already succeeded and §15 could only
find a null.

## What this does not do

It does not rescue the paper. Prediction 3 was the one that would have produced a positive
result and it failed. Two hypotheses in this line have now been registered and measured; both
were refuted on their headline prediction. The base rate stated in §7 of the pre-registration
was accurate.

---

# CORRECTIONS to this record, 2026-09-07

Applied after an independent review of the result section above. Recorded here rather than
silently edited, because three of them narrow claims this document made.

1. **"reproduces on the real robot" was wrong** and is fixed above. Every ALOHA result in this
   project is `gym_aloha` simulation. No real robot is involved anywhere in this work.

2. **The accuracy improvement is metric-dependent, and only the flattering metric was quoted.**

   | metric | E legacy | F innov | |
   |---|---|---|---|
   | mean absolute error | 0.001300 | 0.000854 | −34.3% **better** |
   | RMSE | 0.001429 | 0.001437 | +0.6% **worse** |
   | worst-joint error | 0.002615 | 0.003340 | +27.7% **worse** |

   So "a 34% accuracy improvement bought no repair" overstates it. The estimator did not become
   uniformly more accurate; its error moved out of a uniform offset and into per-joint spread
   (range across the six coordinates widened 2.35×). Less L1 error with unchanged L2 error and
   worse worst-coordinate error is entirely compatible with task-relevant accuracy still being
   what bounds repair.

3. **Each arm contributes ONE held vector, not twenty.** Under identify-then-hold the estimate
   is learned in episode 0 and frozen for the remaining nineteen. Verified: 20 rows, exactly 1
   distinct vector per arm. The comparison is one identification realisation per law against
   repeated recipient evaluations. Calling prediction 2 "nearly deterministic" was wrong; it is
   a single draw.

4. **E and F cannot speak to Proposition 2.** Both hold a constant correction, so both have zero
   within-episode temporal variation. Their differing outcomes cannot establish a
   temporal-variation threshold. Eliminating one candidate cause does not establish the margin
   explanation; the earlier claim that this "supports Prop 2" is withdrawn.

5. **Arms B and C were never run.** The norm-channel rejection is structural and
   telemetry-based, not a measured A-versus-C null on task outcomes. Stated as such.

6. **"The deadzone cannot fire in any configuration" was too broad.** It holds for the legacy
   law, whose residual `r ≈ Mf` does not shrink as the estimate converges. The innovation law
   gates on `r − M f̂`, which can shrink. The conclusion is specific to this fault magnitude,
   this threshold and that law.

7. **Endpoint accuracy does not exclude loss during convergence.** Arm A's applied correction
   has a median peak-to-peak of **0.0194 rad across the whole episode** against **0.00014 rad
   over the final fifty steps** — a 139× ratio. The estimate is excellent at the end and swings
   by nearly the full fault magnitude on the way there.

The safe statement this study supports:

> In one simulated ALOHA identify-and-hold comparison with a single identification episode per
> law, replacing the legacy update with an innovation update reduced mean absolute joint
> estimation error by 34% while worst-coordinate error rose 28%; observed task success fell
> from 6/20 to 2/20 and that difference was not resolved (p = 0.219).
