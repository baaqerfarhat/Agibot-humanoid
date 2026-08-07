# Causality-Guided Single-Layer Online Adaptation — test package

Self-contained package for testing the ACC-2026 layer-adaptation law on the AgiBot X2 31-DoF
box-pickup policy. Everything needed to run it is in this folder; nothing external is required
beyond `numpy` and `mujoco`.

Method: M. Taheri, S.-J. Chung, F. Y. Hadaegh, *"Closing the Loop Inside Neural Networks:
Causality-Guided Layer Adaptation for Fault Recovery Control"*, ACC 2026.

---

## 1. What this does — and what it does not

**It does:** substantially improve how well the frozen policy follows its retargeted reference
motion. Leg tracking error drops **~37%** (14.7° → 9.2°). This is confirmed out-of-sample on
three independent seed pools and survives a displacement-matched random control (0/20 random
weight perturbations of the same magnitude come close).

**It does not:** stop the robot falling. The policy falls in **every** run, adapted or not
(0/32 in both conditions). Adaptation delays the fall by ~10–14% (1.94 s → 2.22 s). No
configuration tested across engagement time, layer, and gain prevents the fall.

Read the headline as: **a reference-tracking result, not a fall-prevention result.**

Do not use a combined "steps before falling OR losing tracking" metric as the headline — it
lets the large tracking gain read as a survival gain. Report survival and tracking separately.

The fall is not the motion's fault: the reference is statically feasible throughout (centre of
mass inside the foot support polygon at all 434 frames, minimum margin **1.7 cm**). It is a
knife-edge balance task, which is consistent with the real robot needing physical support during
the recorded runs.

---

## 2. Quick start

```bash
pip install -r requirements.txt

python run_mujoco_demo.py                     # frozen policy, 1 seed
python run_mujoco_demo.py --adapt             # with adaptation
python run_mujoco_demo.py --adapt --view      # interactive MuJoCo viewer
python run_mujoco_demo.py --seeds 32          # batch: frozen summary
python run_mujoco_demo.py --adapt --seeds 32  # batch: adapted summary
python evaluate.py                            # paired frozen-vs-adapted with statistics
```

Expected from `--seeds 32` (seeds 600–631; exact numbers depend on your MuJoCo version):

| | frozen | adapted |
|---|---|---|
| survival | ~97 steps (1.94 s) | ~111 steps (2.22 s) |
| leg tracking error | ~14.7° | ~9.2° |
| never fell | 0/32 | 0/32 |

---

## 3. Contents

| file | what it is |
|---|---|
| `ace_adapt.py` | **The method.** Environment-agnostic, numpy only, no simulator dependency. This is the file to port. |
| `run_mujoco_demo.py` | Reference integration in MuJoCo. Sections marked `### CONTRACT ###` are what your environment must provide. |
| `evaluate.py` | Paired frozen-vs-adapted evaluation with an exact sign test and a matched random null. |
| `INTEGRATION.md` | How to wire the adapter into your own environment. |
| `RESULTS.md` | All measured numbers, the configuration sweep, and the controls. |
| `assets/` | Robot model (MJCF + 35 meshes) and the exported policy `x2_box_policy_v31.npz`. |

---

## 4. The method in one page

Adapt ONE layer of the frozen policy online, driven by the tracking error:

```
delta_L   = g(x)^T P e                            error signal in ACTION space
delta_l   = Psi_l(a_l) . W_{l+1}^T delta_{l+1}    backprop through activation Jacobians
Wdot_{l*} = Gamma . delta_{l*} z_{l*-1}^T - gamma . W_{l*}
```

`-gamma W` is the leakage (sigma-modification) term that keeps the weights bounded. There is no
counterfactual model, no probing, and no learned objective — one backward pass per control step
through the layers above the adapted one. That is why it is cheap and why it works here: an
earlier approach on this same testbed that had to *estimate* a counterfactual model failed
completely (p = 1.00), because the estimate was ill-posed.

**The input mapping `g(x)` matters more than any hyperparameter.** The policy emits a joint
target driven by a PD servo, so the action→torque map is exact and diagonal,
`d(tau)/d(a) = Kp diag(action_scale)`, giving `g(x) = M^-1 S^T Kp diag(s)`. Using
`action_scale * Kp` (i.e. dropping `M^-1`) roughly **doubles** the effect versus `action_scale`
alone. Adding `M^-1` makes it **worse**: inverse-inertia weighting points the correction at the
lightest joints (wrists, head), whose large tracking errors are irrelevant to the task. Physical
fidelity in `g(x)` is not monotonically better — it has to be fidelity about the quantity you are
regulating.

---

## 5. Confirmed configuration

```python
AdaptConfig(
    layer=2,                                       # 0-based; 3rd of 4 weight matrices
    gain=3e-4,                                     # Gamma
    leak=1e-2,                                     # gamma
    gx_level=1,                                    # action_scale * Kp
    error_joints=("hip", "knee", "ankle", "waist"),
    engage_step=0,                                 # adapt from the very first control step
)
```

Layer 2 was selected empirically and independently agreed with by the paper's offline ACE
attribution at adequate sampling (960 Monte-Carlo draws per layer).

---

## 6. Things that will bite you

**Divergence scores as perfection.** If the weights blow up to NaN, `height < threshold` and
`error > limit` are *both* False, so nothing trips and the episode is credited with the full
horizon — a diverged run looks like a perfect one. `ace_adapt.py` guards on `np.isfinite`; keep
that guard in any port. This silently inverted several results before it was caught.

**Gain is the sensitive knob.** Γ = 3e-4 is stable on every layer tested. Γ ≥ 1e-2 diverges
broadly. Layer 3 (the output layer) is unusable at any useful gain — 20–32 divergences out of 32.

**Restoring simulator state is not the same as running continuously.** Saving/restoring
`qpos`/`qvel` (even plus `qacc_warmstart`, `ctrl`, `act`, `time`) does **not** reproduce a
continuous rollout in a contact-rich scene — measured divergence of −31…+57 steps on identical
seeds. If you evaluate arms from restored snapshots, they are comparable to each other but not to
continuously-run arms. Score every arm the same way.

**`model_109500.pt` is not this policy.** That checkpoint (in the vendor repo) is from run
`x2_box_v27`. The deployed policy is v31, shipped only as the `.npz` export included here.

**Seed hygiene.** Configurations were selected on development seeds and confirmed once on
untouched pools. If you tune on seeds 600–631, use different seeds to confirm.

---

## 7. Open questions worth testing

1. **Does a balance-targeted error signal prevent the fall?** The adapter currently regulates
   *joint* error and hopes balance follows. Defining `delta_L` on a CoM/ZMP error via the
   centroidal mapping is the natural next step and directly targets what actually fails.
2. **Multi-layer adaptation.** Everything here adapts exactly one layer.
3. **Does it transfer to other failure modes?** Only this box-pickup motion has been tested.
