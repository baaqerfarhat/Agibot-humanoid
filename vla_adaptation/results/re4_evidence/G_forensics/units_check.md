# G.4 — Units check: "0.019 cm per step", "0.43 cm within-episode variation", and the ALOHA 0.019 rad figures

Grep over `paper/`, `report/`, `docs/` for `0.019`, `.019`, `per-step correction`,
`within-episode variation`, `0.43`. Every relevant hit is below; unrelated hits are listed at
the end. All recomputation is from stored result files, CPU only.

Two conversions are used:
- **LIBERO:** OSC_POSE, `output_max = 0.05 m` per translation action unit (commanded).
  The measured achieved static gain is 0.21–0.25, from record §2.2 and `openpi/error_signal.py:5`.
- **ALOHA:** record §27.3's measured *5.3 cm at the gripper per 0.05 rad applied uniformly
  to joints 0–5*, which is 106 cm/rad.

---

## 1. LIBERO's "0.019 cm per step"

**Where it appears**

| file:line | text |
|---|---|
| `paper/iclr_draft.tex:286–287` | Prop. 2 discussion: "It predicts LIBERO (... σ of 0.019 cm per step; repair)" |
| `report/vla_adaptation_report.tex:1104` | "the law's 0.019 cm/step of motion cost nothing" |
| `report/vla_adaptation_report.tex:1119` | criterion table, "correction wander" column: `0.019 cm/step` |
| `docs/ADAPTIVE_CONTROL_VLA.md:1935` | §27.5: ALOHA's per-step change is "a fifth of LIBERO's 0.019 cm per step" |
| `docs/ADAPTIVE_CONTROL_VLA.md:2186` | §27.12: "§18 measured 0.019 cm per step of jitter" |
| `docs/ADAPTIVE_CONTROL_VLA.md:2255` | §27.14 criterion table (same as the report's) |
| `report/DUAL_TRACK_AUDIT.md:534` | "LIBERO (σ = 0.019 cm, ...)" |
| `prereg_records/PREREG_GR1_MARGIN.md:22` (outside the searched dirs) | "`σ` measured? yes (0.019 cm/step)" |

**Provenance.** No script, result file or record section derives the number. The source
cited in §27.12, §18, is the integral-baseline section and contains no per-step jitter figure.

**Recomputed.** The source is `results/phase05/tmag_015.json`, the translation-0.15 cell,
4/20 → 19/20, the row the figure is attached to. Its `traj` holds per-step `f̂` in
normalised action units; channels 0–2 are translation. The mean over episodes, steps and
the three translation channels of `|f̂_t − f̂_{t−1}|` is:

| steps | action units | × 5 cm/unit (commanded) |
|---|---|---|
| all | 0.00416 | 0.0208 cm |
| ≥ 15 | 0.00383 | **0.0192 cm** |
| ≥ 30 | 0.00393 | 0.0197 cm |

So the figure reproduces as **the mean per-step change of the translation correction, per
channel, after the transient, converted at the commanded 5 cm per action unit.**
`tmag_010.json` gives the same to within 10%. Other readings give different numbers:

| reading | value |
|---|---|
| median instead of mean | 0.013–0.015 cm |
| xyz Euclidean norm | 0.032–0.041 cm |
| achieved displacement (gain 0.23) | **0.0044 cm** |

**What it is, and its units.**
- `f̂` is in normalised action units of a *delta*-pose command. A correction of c units
  commands c × 5 cm of end-effector translation **per 50 ms step**, so `f̂` is a velocity
  offset.
- `Δf̂` per step is the step-to-step change of that commanded per-step displacement. Its
  dimension is cm/step per step; the "0.019 cm per step" label compresses this.
- It is neither cm of position nor rad.
- **There is no unit mix-up with ALOHA's 0.019 rad.** The LIBERO number is independently
  recomputable from LIBERO data, and ALOHA's 0.019 rad is a different, verified quantity
  (items 3–4). The two agree only by coincidence.

**Is the text right? No, in kind, although the number reproduces.**
1. The number is a **commanded**, not achieved, quantity. The report's "0.019 cm/step of
   *motion*" is wrong: achieved motion is about 0.004 cm/step.
2. It is a **per-step increment**. Proposition 2 defines σ as the *within-episode variation*
   of the applied correction, and the paper sets it against ALOHA's σ = 0.43 cm, which is a
   within-episode **range** of a **position** offset (item 2). The two are different
   statistics of physically different quantities.
   - **Like-for-like per-step change.** LIBERO 0.019 cm commanded / 0.0044 cm achieved,
     against ALOHA 0.0046 cm (mean per-step `|Δf̂|` on j0–5, warm updating cell
     `off002_histfix_warm.json`, 4.3×10⁻⁵ rad; median 0.0016 cm). On this statistic ALOHA's
     correction moves no more than LIBERO's, which §27.5 itself says. The Prop. 2 contrast
     therefore does **not** come from per-step motion.
   - **Like-for-like within-episode range.** LIBERO translation, after step 15, median
     0.116 units: 0.58 cm/step commanded, 0.13 cm/step achieved. That is the range of a
     *velocity* offset, which integrates over time, against a *position* margin. It has no
     unit-consistent comparison to ALOHA's 0.43 cm without a closed-loop model.

**Suggested correction.**
- `paper/iclr_draft.tex:286–287`: replace "σ of 0.019 cm per step" with something like:
  "on LIBERO the correction is a velocity offset on a delta-pose command (within-episode
  range ≈ 0.12 action units, i.e. ≈ 0.6 cm of commanded and ≈ 0.13 cm of achieved motion per
  50 ms step), which a reach absorbs; σ and μ are commensurable, and the proposition is
  tested quantitatively, only on the position-controlled robots". Alternatively, drop the
  LIBERO number from the proposition.
- Report and record criterion tables: relabel the entry as "mean per-step change of `f̂`,
  0.004 action units (0.019 cm commanded, 0.004 cm achieved, per step) — not a
  within-episode variation".
- Record §27.12 and the report sentence: remove "§18 measured" and "of motion".
- `DUAL_TRACK_AUDIT.md:534` and `PREREG_GR1_MARGIN.md:22`: same relabel.

---

## 2. ALOHA's "0.43 cm" within-episode variation

**Where it appears:**
- `paper/iclr_draft.tex:288` (σ = 0.43 cm), `:539` (table), `:550` (text).
- `paper/iclr_draft_audit_framing.tex:1031` (table).
- `report/vla_adaptation_report.tex:1085`, `:1120`, `:1430`.
- `docs/ADAPTIVE_CONTROL_VLA.md:2148`, `:2178`, `:2221`, `:2256`, `:3741`.
- `report/DUAL_TRACK_AUDIT.md:534`, `:909`, `:952`.

**Recomputed.** The source is `results/aloha/off002_histfix_warm.json`: adaptive arm, 0.02 rad
on j0–5, warm start, history fixed, 0/20. It uses the warm episodes 1–19 and joints 0–5. The
applied correction is `−f̂·mask`, from `applied_correction` in `aloha_adapt.py`, so `f̂`
*is* the applied correction.

| quantity (record §27.11 text) | stored-data value | status |
|---|---|---|
| within-episode range, median | per-(episode, joint) `max−min` of `f̂`: **0.00405 rad × 106 cm/rad = 0.429 cm**; range of the joint-averaged trajectory 0.00410 rad = 0.434 cm | reproduces |
| `f̂` at step 0, median 0.0181 | 0.0181 (joint mean); 0.0177 over all (episode, joint) | reproduces |
| worst-case min 0.0146 rad → 0.57 cm | 0.01459 on the joint-mean trajectory (0.57 cm); per-joint worst case 0.0117 rad (0.88 cm) | reproduces for the joint mean |
| mean residual 0.17 cm; > 0.5 cm 2%; > 0.25 cm 28% | 0.168 cm; 1.6%; 28.1% | reproduces |

**What it is and its units.** A within-episode **range** (max − min, not a standard
deviation), in rad of joint offset. It is converted to cm at the gripper with the
uniform-offset factor of 106 cm/rad. The SD over the same data is 0.0012 rad, 0.13 cm.

**Is the text right? The number is right; two labels overstate it.**
1. The symbol σ and the phrase "variation" suggest an SD; the quantity is a median range.
2. The cm figure assumes a uniform offset on all six joints. The per-joint wander is not
   uniform, so 0.43 cm is an approximate gripper displacement, not a measured one.

**Suggested correction.** "within-episode range of the applied correction, median over
episodes and joints: 0.0041 rad (≈ 0.43 cm at the gripper, uniform-offset conversion)".

---

## 3. ALOHA's "0.019 rad (95%) applied frozen" — the static correction

**Where it appears:**
- `paper/iclr_draft.tex:540`
- `paper/iclr_draft_audit_framing.tex:1032`
- `docs/ADAPTIVE_CONTROL_VLA.md:2157`, `:2169`, `:2198`

**Recomputed.** The source is `results/aloha/off002_static019.json`:
- `args.static_corr = 0.019` on j0–5, against a fault of 0.02 on j0–5, so 95%.
- `applied_correction` returns `−static_corr` whenever it is set, so the correction applied
  during the episode has zero variation. The estimator still runs and its `f̂` in `traj`
  moves, but that `f̂` is never applied.
- Residual: (0.020 − 0.019) × 106 = **0.106 cm** (quoted 0.11 cm). Success 5/20.
- The value 0.019 is the converged estimate of the cold, history-fixed cell (95, 104, 94, 95,
  95, 95%, record §27.10).

**Units:** rad per joint. **Correct.**

A third 0.019 is record-only, not in the paper: §27.6's addendum, "swings by a median 0.019
rad — 2.0 cm". On `results/aloha/off005_warm.json` (0.05 rad fault, zero-history era), the
range of the joint-mean trajectory has median **0.0191 rad** (2.0 cm), which is correct as a
joint-mean range. The per-(episode, joint) median is 0.0139 rad (1.47 cm).

---

## 4. ALOHA held estimate (identify, then hold)

| where | quoted | recomputed (final `f̂` of identification episode 0, j0–5) | status |
|---|---|---|---|
| `paper/iclr_draft.tex:302` (Fig. convergence caption) | 92–96% | `off002_identify1_hold.json` (n=20): 0.01835–0.01907 rad = **92–95%** | 95.4% rounds to 95; the 96 comes from rounding 0.0191 first. Cosmetic |
| `paper/iclr_draft_audit_framing.tex:752`, `paper/FIGURE_AUDIT.md:54` | 0.0183–0.0191 / 0.01835–0.01907 rad | same | correct |
| `paper/iclr_draft.tex:552`, `audit_framing:1044`, record `:2378` | 95–101%, 0.019–0.020 rad, 0.0190–0.0202 | `off002_identify1_hold_n40.json` (n=40): 0.01898–0.02020 rad = 95–101% | correct |
| held episodes | σ = 0 | max within-episode range of `f̂` over all held episodes: exactly 0.0 (both files) | correct |
| `paper/iclr_draft.tex:541`, `audit_framing:1033`: n=40 hold row, "mean residual 0.12 cm" | 0.12 cm | the n=40 estimate (mean 0.01934) gives **(0.02 − f̂)·106 = 0.070 cm**; 0.118 cm is the n=20 run's (mean 0.0189) | **wrong for its row** |

Suggested fix for the n=40 row: write 0.07 cm, or show "—" as the report's table does. The
figure caption's middle panel is the n=20 identification episode while the table row is the
n=40 run; the caption could say so.

---

## 5. Record-only per-step figure (§27.5)

"0.00004–0.00006 rad per step — 0.004–0.006 cm". This is post-transient mean `|Δf̂|` on
j0–5 in the pre-fix 0.02 rad cell (`off002_nr04.json`, steps ≥ 30).
- The pooled mean is 4.1×10⁻⁵ rad (0.0043 cm), which matches the lower end.
- The per-joint means span 1.9–10.6×10⁻⁵ rad; j1 is the outlier at 0.011 cm.
- The history-fixed cells give the same.
- It is not in the paper.

---

## Unrelated hits (checked, not this figure)

| file:line | what it is |
|---|---|
| `paper/iclr_draft.tex:785`, `paper/iclr_draft_audit_framing.tex:1239`, `docs/ADAPTIVE_CONTROL_VLA.md:2649` | "0.044, 0.019, 0.044": LIBERO γ-ablation identification values on rotation channels, action units |
| `paper/authority_appendix.tex:230` | 0.0196 < 0.02, a bound |
| `docs/ADAPTIVE_CONTROL_VLA.md:532` | −0.019, a LIBERO translation reading |
| `docs/ADAPTIVE_CONTROL_VLA.md:1926` | ALOHA identification values 0.019–0.021 rad |
| `docs/LESSONS_VLA.md:298` | 0.43, an ACE score |
| `report/COMPOSITE_TUNING.md:91` | "per-step correction energy", a different metric |
| `docs/independent_adaptation/*` | "within-episode variation" as a phrase (residual cosine) |
| `paper/iclr_draft.tex:83` | contribution list, no number |

## Summary of figures that are wrong

1. **LIBERO "σ of 0.019 cm per step"** (paper :286–287; report :1104, :1119; record; audit;
   prereg). The number reproduces only as a *commanded* mean per-step change of the
   translation correction; achieved motion is ≈ 0.004 cm/step. Its "§18" source does not
   contain it. It is a per-step increment of a velocity offset, used as though it were a
   within-episode position variation comparable to ALOHA's 0.43 cm range. The
   cross-robot σ comparison in Prop. 2 is not unit-consistent, and on a like-for-like
   per-step basis ALOHA's correction moves *less* than LIBERO's.
2. **"0.12 cm" mean residual in the n=40 identify-then-hold row** (paper :541; audit framing
   :1033). It is the n=20 run's value; the n=40 held estimate gives 0.07 cm.
3. **Label-level only.** ALOHA's 0.43 cm is correct but is a median range converted by a
   uniform-offset factor, not an SD. The caption's "92–96%" should read 92–95%.
