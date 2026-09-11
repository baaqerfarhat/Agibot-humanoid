# G.3 — Predictor diagnostics for the joint-space robots (ALOHA, GR1)

CPU only, stored logs only, numpy only. Script: `predictor_diagnostics.py` (this directory);
full numbers: `predictor_diagnostics.json`.

## What was run

- **Predictor.** The shipped per-joint FIR `q[t] = sum_{k=0..6} h_k u[t-k] + c`, fitted by the
  runner's own `fit_plant`. ALOHA: `openpi/aloha_adapt.py` imported directly (its module
  level is stdlib + numpy + constants). GR1: `openpi/gr1_adapt.py` imports gymnasium and
  robocasa at module level, so it was **not imported**; its `fit_plant` was extracted
  verbatim with `ast` and executed with the module's own `NJ=29, K_FIR=6`. The body is
  token-identical to ALOHA's apart from the docstring. The script asserts that the R²
  recomputed from `W` equals `fit_plant`'s returned R² to 1e-9 on every joint.
- **Log convention** (`episode()` in both runners). `u[t]` is the command applied at step t,
  and `q[t]` is the position measured after that step. Rows are `t >= 6` of every episode,
  exactly as in `fit_plant`.
- **Persistence.** `q_hat[t] = q[t-1]`.
- **Increment.** `dy[t] = q[t] - q[t-1]`. The FIR increment is `q_hat_FIR[t] - q[t-1]`, and
  persistence predicts 0. The **skill vs persistence** is `1 - SSE_FIR/SSE_persistence` on
  position. It equals the uncentred incremental R², because the FIR's position error *is*
  its increment error.
- **Extra command-free baseline.** Constant-velocity extrapolation,
  `dy_hat[t] = q[t-1] - q[t-2]`.
- **Pooled** means `1 - sum_j SSE_j / sum_j SST_j`, each joint centred on its own mean.
- **Held-out** means leave-one-episode-out (LOEO). For each episode, `W` is refit by
  `fit_plant` on the remaining episodes and evaluated on the held-out one. The R² is taken
  over all held-out predictions concatenated. The in-sample numbers use the shipped `W`
  (fit on all episodes); those are what the paper reports.

| dataset (log, `args.log` of the cells) | robot, joints | episodes (lengths) |
|---|---|---|
| `results/aloha/healthy_log.json` (all `results/aloha/off*` cells) | ALOHA left arm j0–5, 50 Hz | 8 (225–300) |
| `results/gr1/screen_PosttrainPnPNovelFromPlateToPlateSplitA.json` (`p2p_*`) | GR1 right arm j7–13, 20 Hz | 6 (162–720) |
| `results/gr1/t2p_healthy_log.json` (`t2p_*`) | GR1 right arm j7–13, 20 Hz | 10 (168–720, four 720-step timeouts) |
| supplementary: `results/gr1/healthy_log.json` (`left015_*`, record §32.1–32.5, not a paper cell) | GR1 left arm j0–6 | 5 |

The md5 of each log matches the file at the `args.log` path (`Agibot-humanoid/vla_adaptation/...`).

## Pooled results (corrected joints)

| dataset | split | abs R² FIR | abs R² persistence | incr. R² FIR | incr. R² persistence | incr. R² const-velocity | skill vs persistence |
|---|---|---|---|---|---|---|---|
| ALOHA | in-sample | 0.99980 | 0.99944 | 0.571 | −0.216 | 0.827 | 0.647 |
| ALOHA | **LOEO** | 0.99978 | 0.99944 | **0.531** | −0.216 | 0.827 | **0.614** |
| GR1 plate-to-plate | in-sample | 0.99878 | 0.99777 | 0.449 | −0.006 | 0.942 | 0.452 |
| GR1 plate-to-plate | **LOEO** | 0.99863 | 0.99777 | **0.382** | −0.006 | 0.942 | **0.385** |
| GR1 tray-to-plate | in-sample | 0.99791 | 0.99904 | −1.190 | −0.005 | 0.911 | −1.179 |
| GR1 tray-to-plate | **LOEO** | 0.99736 | **0.99904** | **−1.769** | −0.005 | 0.911 | **−1.756** |
| GR1 can-drawer, left (suppl.) | LOEO | 0.98786 | 0.99933 | −17.2 | −0.004 | 0.864 | −17.1 |

Persistence's incremental R² is not exactly 0. Predicting a zero increment gives
`R² = -mean(dy)²/var(dy)`, which is −0.22 on ALOHA because the arm drifts in one direction
over the rows.

## Per joint, held-out (LOEO) incremental R²

| ALOHA | j0 | j1 | j2 | j3 | j4 | j5 |
|---|---|---|---|---|---|---|
| FIR | −15.1 | 0.02 | **0.997** | 0.25 | **0.971** | **0.993** |
| const-velocity | 0.89 | 0.77 | 0.97 | 0.87 | 0.88 | 0.73 |
| abs R² FIR / persistence | 0.9950 / 0.9997 | 0.9997 / 0.9996 | 1.0000 / 0.9995 | 0.9985 / 0.9980 | 1.0000 / 0.9994 | 1.0000 / 0.9990 |
| rms dy per step (rad) | 0.00007 | 0.0049 | 0.0030 | 0.0016 | 0.0031 | 0.0026 |

| GR1 plate-to-plate | j7 | j8 | j9 | j10 | j11 | j12 | j13 |
|---|---|---|---|---|---|---|---|
| FIR | 0.67 | 0.45 | 0.58 | 0.59 | 0.54 | −0.23 | 0.13 |
| const-velocity | 0.97 | 0.85 | 0.92 | 0.98 | 0.89 | 0.95 | 0.96 |

| GR1 tray-to-plate | j7 | j8 | j9 | j10 | j11 | j12 | j13 |
|---|---|---|---|---|---|---|---|
| FIR | −0.02 | −0.07 | 0.18 | −0.11 | 0.47 | −6.34 | −4.58 |
| const-velocity | 0.93 | 0.79 | 0.87 | 0.95 | 0.87 | 0.93 | 0.95 |

Per held-out episode, pooled skill vs persistence:

| dataset | range |
|---|---|
| ALOHA | 0.40–0.77, positive in 8/8 |
| GR1 plate-to-plate | 0.22–0.65, positive in 6/6 |
| GR1 tray-to-plate | negative in 7/10 |

On tray-to-plate, the four timeout episodes score −0.45, −6.6, −5.5 and −8.3. The six
successful episodes range from −1.6 to +0.28.

FIR DC gain (sum of taps) on the corrected joints:

| dataset | DC gain |
|---|---|
| ALOHA | 0.98–1.015 |
| GR1 plate-to-plate | 0.97–0.99 |
| GR1 tray-to-plate | 0.99–1.00, except wrist j12 at 0.91 and j13 at 0.94 |

## Plain reading

**Absolute-position R² says almost nothing here.** Persistence, which uses no command at all,
scores 0.998–0.9994 on the same rows. The FIR's reported 0.989–1.000 sits within 0.001 of
it, and on tray-to-plate the FIR is *below* persistence (0.9974 vs 0.9990 held-out).

On increments, which is what the command should explain, the picture depends on the robot:
- **ALOHA:** the FIR predicts command-induced motion well beyond persistence. Held-out
  incremental R² is 0.53 pooled, and it removes 61% of persistence's squared one-step error.
  Almost all of that comes from three joints: j2, j4 and j5 score 0.97–0.997 and beat the
  command-free constant-velocity extrapolation. j1 and j3 are weak (0.02, 0.25). j0 is worse
  than persistence (−15); it barely moves (rms 7×10⁻⁵ rad/step), so the FIR's small offset
  error swamps it.
- **GR1 plate-to-plate:** a moderate effect. Held-out incremental R² is 0.38 pooled, and
  0.45–0.67 on shoulder/elbow j7–j11. The FIR fails on wrist j12 (−0.23) and is near zero
  on j13 (0.13). Skill is positive on every held-out episode.
- **GR1 tray-to-plate:** the FIR does **not** predict command-induced motion better than
  persistence (held-out incremental R² −1.77; skill −1.76). Only j9 and j11 are positive.
  The wrist (j12, j13: −6.3, −4.6) and the four timeout episodes dominate. Skill is also
  negative on three of six successful episodes.

In every dataset, the command-free constant-velocity extrapolation beats the FIR on pooled
increments (0.83 / 0.94 / 0.91). That is expected: the FIR is an output-error model that
never sees the measured state. So its high position R² reflects a servo that tracks its
target (DC gain ≈ 1), not a one-step dynamics model.

**What this does not test.** The estimator relies on the steady-state map from a command
offset to a position offset. That is the FIR DC gain (≈ 1 on every corrected joint except
the tray-to-plate wrist), and `M` was measured separately by replay or step response.
Incremental R² bounds how much of the healthy within-episode motion the FIR leaves in the
residual (the phantom and noise floor). It says nothing about whether the static offset is
mapped correctly.

## Corrections this implies for the paper

1. `paper/iclr_draft.tex:522`, ALOHA: *"R² = 0.989–1.000 on all joints"* is wrong as written.
   The shipped fit gives j13 (right gripper) **0.768** and j6 (left gripper) 0.9915. The
   0.989–1.000 range holds for the 12 arm joints (min 0.9891, j11). Suggested wording:
   "R² 0.989–1.000 on the twelve arm joints (grippers 0.99, 0.77); on the corrected left arm,
   held-out incremental-motion R² 0.53 against −0.22 for persistence, driven by j2, j4, j5".
2. `paper/iclr_draft.tex:564`, GR1: *"Plant from six healthy episodes (R² ≥ 0.998 on the arm)"*.
   - This describes the plate-to-plate plant: right-arm min 0.99796, which rounds to 0.998.
   - The tray-to-plate cells use a different plant, from 10 episodes, whose right arm is
     0.9943–0.9997 (both arms, min 0.950). On increments that plant does not beat persistence.
   - Suggested wording: state both plants, and report held-out incremental R² (0.38 on
     plate-to-plate; below persistence on tray-to-plate) next to, or instead of, the position R².
