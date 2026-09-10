# Online adaptive repair of a frozen vision–language–action policy

A frozen π0.5 (3.35 B parameters) is repaired **within a single episode** by adapting six
numbers at its action interface, from the robot's own motion. No fine-tuning, no reward, no
gradient through the policy, no reference trajectory. The update is a matrix–vector product
per control step and runs on CPU.

The review branch adds composite adaptation, bounded weighted correction, and
all-joint torque evaluations on Panda and ALOHA. See the
[follow-up evidence and limitations](report/FOLLOWUP_STATUS.md) for those results;
the action-interface results below do not establish repair of every joint fault.
The completed confirmation favors the new candidate over legacy for Panda joint 5,
but exposes harm elsewhere; both new variants remain experimental.

## The result

Uniform action-interface offset, correction restricted to identifiable channels, four LIBERO
suites, **paired** episodes (same tasks and initial states in both arms; the policy's
sampling noise is not pinned, so pairing is on the episode, not the action sequence):

| suite | frozen | corrected | exact McNemar |
|---|---|---|---|
| `libero_spatial` | 8/20 = 40% | 18/20 = 90% | 0.0020 |
| `libero_goal` (n=40) | 15/40 = 38% | 29/40 = 72% | 0.00052 |
| `libero_object` | 5/20 = 25% | 16/20 = 80% | 0.00098 |
| `libero_10` (n=40) | 0/40 = 0% | 15/40 = 38% | 6.1×10⁻⁵ |
| **pooled** | **28/120 = 23%** | **78/120 = 65%** | **2.4×10⁻¹⁴** |

Across **690 paired episodes** spanning two backbones, four fault families, four suites, three
severities each and four time profiles: **275 episodes fixed, 7 broken** — a 1.0% regression rate.
Every suite in the table above is individually significant.

The three headline cells replicate on a second suite (`libero_object`) **without
re-identifying the plant model or `M`** — the offline calibration transfers across task
suites: rotation 0/20 → 10/20, translation 0/20 → 15/20, gain 0/20 → 19/20.

A translation fault large enough to destroy the policy is recovered almost completely:
**4/20 = 20% → 19/20 = 95%**, `p = 6.1×10⁻⁵`.

### Recoverability map

Fault family × severity, `libero_spatial`, n = 20 paired per cell. Every regression falls
in a ceiling cell where the frozen policy already scores 18–19/20 (translation 0.05 is
18/20 → 18/20 with one fixed and one broken):

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

### Does the machinery earn its place?

A plain integral controller on the raw motion error — no plant model, no `M` — swept across
five gains and reported at its best: **35%**, against this method's **95%** on the same fault.
Above `kᵢ = 0.005` it is worse than doing nothing. The measured plant DC gain is 0.23, so the
raw error is dominated by a command-proportional phantom 2.5× the size of the fault and
opposite in sign. Predicting `y ≈ 0.23a` instead of `y ≈ a` is the whole difference (§18).

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
| mid-episode onset | improves, not resolved — 12/20 → 18/20 at step 15 (p = 0.065), 11/15 → 12/15 at step 45; pairing not recoverable |
| ramp / gradual degradation | solved — +30, `p = 0.031` |
| oscillatory (non-zero-mean) | solved — 13/20 → 20/20, `p` = 0.016 |
| integral-control baseline | 35% at best vs 95% (§18) |
| **second backbone (OpenVLA-OFT)** | repaired on all three families with the π0.5 calibration, unchanged — 0/20 → 11/20, 14/20, 17/20 (§25) |
| **second manipulator (ALOHA, 14-DOF joint space)** | identified to 94%; repaired 0/40 → 15/40 (p = 6.1×10⁻⁵, zero regressions) once the correction is held after one sacrificial episode — a continuously updating correction fails on a 0.5 cm margin (§27) |
| **third backbone (NVIDIA GR00T N1.7)** | official LIBERO finetune served behind the same protocol; rotation 0/20 → 14/20, translation 2/20 → 15/20, gain 0/20 → 19/20 with the π0.5 calibration, 46 repaired / 0 broken; healthy control a null on totals, 18/20 both arms with one fixed and one broken (§30) |
| **a humanoid (Fourier GR1 under GR00T N1.5, joint space)** | right-arm +0.10 rad: pooled frozen 4/90, healthy 64/90; identify then hold 21/30 (Fisher p = 8.9×10⁻¹³ vs pooled frozen), continuous 8/30 (p = 0.0016 vs pooled frozen, 2.8×10⁻⁵ vs pooled healthy). Original same-process continuous comparison: 8/30 vs 1/30, p = 0.026. Expanded controls are a subsequent unpaired analysis; healthy null 19/30 vs 18/30 remains unresolved (§32). |
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
  mcnemar.py        paired statistics: exact McNemar, permutation, Wilson intervals
  openloop_id.py    measures M by open-loop replay
  error_signal.py   identifies the FIR plant on healthy rollouts
  so3.py            rotation increments on the group, not by chart subtraction
hardware/         the same law applied to real X2 humanoid logs (see note below)
docs/             ADAPTIVE_CONTROL_VLA.md is the primary record, §1–§28
report/           LaTeX report (the full record in paper form), the external review, and the response
paper/            the ICLR draft: 8 pages, built from the record, every number from stored outcomes
results/          every run behind every number above; results/phase05/*.mp4 are the
                  comparison videos (spatial, object, goal, libero_10, OpenVLA-OFT, GR00T rotation and translation, ALOHA, GR1 humanoid); results/groot/ is the third backbone, results/gr1/ the humanoid
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
