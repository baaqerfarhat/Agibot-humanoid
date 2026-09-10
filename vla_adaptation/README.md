# Online adaptive repair of a frozen vision–language–action policy

A frozen π0.5 (3.35 B parameters) is repaired **within a single episode** by adapting six
numbers at its action interface, from the robot's own motion. No fine-tuning, no reward, no
gradient through the policy, no reference trajectory. The update is a matrix–vector product
per control step and runs on CPU.

## The result

Uniform action-interface offset, correction restricted to identifiable channels, four LIBERO
suites, **paired** episodes (same tasks, initial states and seeds in both arms):

| suite | frozen | corrected | exact McNemar |
|---|---|---|---|
| `libero_spatial` | 8/20 = 40% | 18/20 = 90% | 0.0020 |
| `libero_goal` (n=40) | 15/40 = 38% | 29/40 = 72% | 0.00052 |
| `libero_object` | 5/20 = 25% | 16/20 = 80% | 0.00098 |
| `libero_10` (n=40) | 0/40 = 0% | 15/40 = 38% | 6.1×10⁻⁵ |
| **pooled** | **28/120 = 23%** | **78/120 = 65%** | **6.1×10⁻¹³** |

Across **690 paired episodes** spanning two backbones, four fault families, four suites, three
severities each and four time profiles: **275 episodes fixed, 7 broken** — a 1.0% regression rate.
Every suite in the table above is individually significant.

The three headline cells replicate on a second suite (`libero_object`) **without
re-identifying the plant model or `M`** — the offline calibration transfers across task
suites: rotation 0/20 → 10/20, translation 0/20 → 15/20, gain 0/20 → 19/20.

A translation fault large enough to destroy the policy is recovered almost completely:
**4/20 = 20% → 19/20 = 95%**, `p = 6.1×10⁻⁵`.

### Recoverability map

Fault family × severity, `libero_spatial`, n = 20 paired per cell. Every regression in the map
occurs in a ceiling cell: the translation-0.05 cell is 1 fixed / 1 broken behind its equal totals
(a correction from the dual-track audit, `report/FINDINGS.md`):

| fault | 0.05 | 0.10 | 0.15 |
|---|---|---|---|
| translation | 18/20 → 18/20 (ceiling) | 13/20 → **19/20** | 4/20 → **19/20** |
| rotation | 13/20 → 18/20 | 0/20 → **17/20** | 0/20 → **9/20** |
| uniform 6-axis | 8/20 → 18/20 | 2/20 → 11/20 | 0/20 → 2/20 |

Rotation is where this policy breaks: at 0.10 and 0.15 a rotation fault takes it to *exactly
zero*, and the correction recovers 17/20 from that floor. Single-family faults degrade
gracefully; the uniform fault collapses, and §19.3 shows why — adding an *uncorrected*
translation component to an identically-corrected rotation fault drops recovery from 45% to
10%, so the limit is the plant leaving the regime where its own motion is informative, not
the estimator.

### Does the machinery earn its place? (revised after the audit)

A plain integral controller on the raw motion error — no plant model, no `M` — swept across
five gains and reported at its best: **35%**, against this method's **95%** on the same fault.
The measured plant DC gain is 0.23, so the raw error is dominated by a command-proportional
phantom 2.5× the size of the fault and opposite in sign (§18). That comparison shows that
**having the plant model** helps. The dual-track audit then built matched estimators that all
use the same residual, `M` and projection — DOB, RLS, Kalman, a calibrated integral — and every
one of them repairs (19, 17, 17, 18, 15 of 20 against 20/20 for an oracle; `report/FINDINGS.md`
§3). The contribution is the calibrated interface, not the particular update law; the paper
now says so.

### Dual-track audit and the follow-up studies (branch `review/dual-track-audit`, merged 2026-09-10)

Two independent tracks (Claude and Codex) audited the paper against the stored data:
25 of 26 headline numbers reproduce; six summary claims contradicted by the data were corrected;
the McNemar instrument's absolute tolerance was replaced by a relative one (the pooled p moved
from 6.1×10⁻¹³ to 2.4×10⁻¹⁴, conservative either way); `openpi/mcnemar_crosscheck.py` re-derives
every printed p-value independently. Start at `report/FINDINGS.md`; the long form is
`report/DUAL_TRACK_AUDIT.md`; the branch comparison and integration plan are in
`docs/branch_comparison_20260909/`.

| follow-up study | result |
|---|---|
| **all-joint physical faults, Panda** (1,440 rollouts, three suites × seven +5 N·m joint torques + healthy × off/legacy/weighted) | the legacy translation-only correction harms joint 5 (12/20 → 3/20 in the original map); a weighted full-channel allocator repairs it (15/60 → 41/60, 29 fixed / 3 broken, p = 2.6×10⁻⁶) but has a confirmed counterexample (Object joint 3: 16/20 → 2/20). Experimental, not a default (`report/FOLLOWUP_STATUS.md`) |
| **matched estimators, ALOHA** (5,424 rollouts: fit, search, validation, confirmation on 30 fresh seeds) | original law 50.0%, RLS 43.3%, DOB 42.1%, Kalman 36.9%, composite 36.0%, integral 31.9% balanced score; composite − Kalman = −0.95 pp, 95% CI [−9.5, +7.6]; no estimator beats another (`report/COMPOSITE_TUNING.md`) |
| **descriptor-form controller with ARX state identification** (`learned_adaptation/`, an attributed experimental module by the audit's author) | seven Spatial joint-torque cells 112/140 → 135/140 and Object joint 3 7/20 → 18/20 against their own frozen controls, constant basis Φ = E, no trained features; not a matched comparison with the estimators above (`SWEEP_RESULT.md`) |

## How it works

Per control step, at 20 Hz:

1. The frozen policy emits a Cartesian action `a`.
2. A plant model — an FIR fit of action → end-effector motion, identified **offline on
   healthy data** — predicts the motion `a` should produce.
3. The residual `r = y − ŷ` is the part of the motion the model cannot explain.
4. `M⁻¹r` maps that motion-space error into action units. `M = d(motion)/d(fault)` is a 6×6
   sensitivity matrix measured once, offline, by open-loop replay.
5. `f̂ ← f̂ + γ(M⁻¹r − f̂)`, with deadzone, normalisation and projection.
6. The next action is sent as `a − f̂`.

`f̂` **resets to zero at the start of every episode** — there is no learning across episodes.
Each episode detects and cancels the fault from scratch, reaching ~70% of truth within 15
control steps.

## What is solved and what is not

| fault | status |
|---|---|
| constant additive, uniform 6-axis | solved — four suites, `p = 9.3×10⁻¹⁰` |
| constant additive, single-family, 0.05–0.15 | solved — up to 20% → 95% |
| structured mixed-sign | solved — +66 points |
| mid-episode onset | solved |
| ramp / gradual degradation | solved — +30, `p = 0.031` |
| oscillatory (non-zero-mean) | solved — 13/20 → 20/20, `p` = 0.016 |
| integral-control baseline | 35% at best vs 95% (§18) |
| **second backbone (OpenVLA-OFT)** | repaired on all three families with the π0.5 calibration, unchanged — 0/20 → 11/20, 14/20, 17/20 (§25) |
| **second manipulator (ALOHA, 14-DOF joint space)** | identified to 94%; repaired 0/40 → 15/40 (p = 6.1×10⁻⁵, zero regressions) once the correction is held after one sacrificial episode — a continuously updating correction fails on a 0.5 cm margin (§27) |
| **third backbone (NVIDIA GR00T N1.7)** | official LIBERO finetune served behind the same protocol; rotation 0/20 → 14/20, translation 2/20 → 15/20, gain 0/20 → 19/20 with the π0.5 calibration, 46 repaired / 0 broken; healthy control a null (§30) |
| **a humanoid (Fourier GR1 under GR00T N1.5, joint space)** | right-arm +0.10 rad zeroes the frozen policy on two tasks; identify over three episodes then hold: plate-to-plate 1/30 → 19/30 (18 fixed / 0 broken, p = 7.6×10⁻⁶), tray-to-plate 1/30 → 13/30 (12 / 0, p = 4.9×10⁻⁴), both at their healthy rates, paired on identical scenes (§32) |
| **a second simulator (WidowX in SimplerEnv / SAPIEN, GR00T N1.7 Bridge)** | a +0.005 pose-increment offset on x,y,z zeroes the policy (0/20); identify over three episodes then hold: 0/20 → 14/20 on the next 20 paired episodes, healthy 14/20 (14 fixed / 0 broken, p = 1.2×10⁻⁴); continuous adaptation 6/20 at γ = 0.08 and 12/20 at γ = 0.2 because this controller keeps the transient drift; +0.003 held: 2/20 → 13/20 (11 / 0, p = 9.8×10⁻⁴); null 16/20 vs 15/20 (§34) |
| **channel mask from healthy data alone (answers audit §4.5)** | a per-channel gate at 3 sd of the healthy phantom, measured on 20 healthy episodes, reproduces the rotation-only headline on the same scenarios: uniform +0.05, 9/20 → **19/20**, 10 fixed / 0 broken, p = 0.002; nothing in the rule has seen a fault (§37, prereg `PREREG_HEALTHY_GATE.md`) |
| **not a result: base OpenVLA on the SimplerEnv Google robot** | healthy 2/10 (published 46%), plant R² 0.57 on a planner-interpolated 3 Hz delta controller, a +0.02 offset reaches the residual at 20–60% of its linear signature (1–2 sd): both stated conditions fail, no cell run, recorded as a boundary (§35) |
| **faults below the controller (joint-level, in the MuJoCo model)** | elbow torque bias 20/40 → 32/40 (p = 0.0075, 3 broken); heavy friction 0/20 → 8/20 (p = 0.0078, 0 broken); a joint lock is identified and not repairable, a rank change rather than an input (§29) |
| intermittent | solved — 12/20 → 19/20, `p` = 0.039 |
| multiplicative (loss of effectiveness) | solved — 0/20 → 17/20 at 80% authority loss, `p` = 1.5×10⁻⁵ |
| sensor bias, camera misalignment | structurally invisible to this residual |

**Scope.** The fault is injected at the Cartesian action interface, upstream of the OSC
controller — a miscalibrated tool frame, a wrist-mount offset, a stale hand–eye calibration.
It is *not* a joint-level actuator fault, which enters below the controller where the
fault-to-motion map is state-dependent.

## Layout

```
openpi/           experiment code (named for the openpi stack it drives, not vendored upstream)
  adaptive_law.py   the additive law: estimator, correction, fault injection, profiles
  adaptive_gain.py  the multiplicative law (unsolved; see docs §16)
  mcnemar.py        paired statistics: exact McNemar (relative tolerance), permutation, Wilson intervals
  mcnemar_crosscheck.py  independent re-derivation of every printed p-value (audit instrument)
  joint_fault.py, libero_reset.py  joint-level faults with guaranteed restoration; reset fingerprints
  weighted_dob.py, composite_observer.py  the audit's weighted allocator and composite estimator (experimental)
learned_adaptation/  the audit author's descriptor-form method (ARX identification, constant basis); experimental, attributed
scripts/          lambda setup scripts for LIBERO and ALOHA (five install steps SETUP.md did not have)
  openloop_id.py    measures M by open-loop replay
  error_signal.py   identifies the FIR plant on healthy rollouts
  so3.py            rotation increments on the group, not by chart subtraction
hardware/         the same law applied to real X2 humanoid logs (see note below)
docs/             ADAPTIVE_CONTROL_VLA.md is the primary record, §1–§28
report/           LaTeX report (the full record in paper form), the external review, the response,
                  and the dual-track audit (FINDINGS.md, DUAL_TRACK_AUDIT.md, FOLLOWUP_STATUS.md, COMPOSITE_TUNING.md)
paper/            the ICLR draft: 8 pages, built from the record, every number from stored outcomes
results/          every run behind every number above; results/phase05/*.mp4 are the
                  comparison videos (spatial, object, goal, libero_10, OpenVLA-OFT, GR00T rotation and translation, ALOHA, GR1 humanoid on two tasks and at 0.20 rad, WidowX in SimplerEnv); results/groot/ is the third backbone, results/gr1/ the humanoid, results/widowx/ the SimplerEnv WidowX
prereg_records/   predictions registered before their experiments ran
```

`docs/ADAPTIVE_CONTROL_VLA.md` is the document to read. It is written as a running record and
**includes the claims that were refuted**, each marked where it was superseded rather than
quietly deleted — several headline explanations died during the work and the corrections are
part of the evidence.

**Note on `hardware/`:** these scripts detect a real actuator-authority fault on an X2
humanoid from logs recorded during box-pickup deployments. The task itself is not part of this
work and none of its code is here; only the fault-detection analysis, which applies the same
law to a different policy and to hardware. Delete the directory if that connection is
unwanted — nothing else depends on it.

## Reproducing

**Cloning this repo is not enough to run the experiments.** They drive a π0.5 policy served by
[openpi](https://github.com/Physical-Intelligence/openpi) at pinned commit `15a9616`, which is
not vendored here, plus a 12 GB checkpoint and a GPU. openpi also uses **two separate
virtualenvs** — a python 3.11 server with jax, and a python 3.8 simulation client with
robosuite and no jax — and mixing them fails in ways that look like bugs in this code.

**[SETUP.md](SETUP.md) has the exact steps**, including the two-environment split, the server
launch, and how to re-identify the plant model instead of reusing the shipped calibration.

**The analysis runs with nothing installed.** `openpi/mcnemar.py` is pure standard library, so
every p-value in the table above can be recomputed from the stored per-episode outcomes:

```bash
python3 openpi/mcnemar.py results/suites/*_rotonly_paired.json
```

Every result carries per-episode outcomes, so any claim here can be re-tested as a paired
comparison rather than a difference of totals.
