# Cross-Embodiment Layer Adaptation — derivation, plan, and kill criteria

**18 Aug 2026.** A VLA trained on robot A (e.g. Unitree G1) deployed on robot B (AgiBot X2).
Claim: online adaptation of the model's interface layers closes the embodiment gap.

This document exists because the previous direction failed for a *reason that is now understood*,
and the same analysis says this direction should not fail the same way. The derivation below is
what distinguishes them; everything after it is the sequence of tests, each with a stated
condition for abandoning the direction.

---

## 1. Derivation

### 1.1 Setup

Robot `X ∈ {A, B}` has constrained whole-body dynamics

```
M_X(q) v̇ + h_X(q, v) = S_Xᵀ τ_X + J_{c,X}ᵀ λ
```

with `S_X` selecting actuated joints. The task is defined on an output `y = Φ_X(q)` — end-effector
pose, base twist, whatever the VLA was trained against — with task Jacobian `J_X = ∂Φ_X/∂q`.

The VLA emits an action `a` which becomes a joint-position reference through that robot's own
convention, `q^des = D_X a + q^0_X`, tracked by an inner servo `τ = K_p^X(q^des − q) − K_d^X v`.

Composing, the instantaneous map from action to task acceleration is

```
G_X(q, v) = J_X · M_{c,X}⁻¹ · S_Xᵀ · K_p^X · D_X                                (1)
```

with `M_c⁻¹ = M⁻¹ − M⁻¹J_cᵀ(J_c M⁻¹ J_cᵀ)⁺ J_c M⁻¹` the contact-consistent inverse mass.

**Equation (1) is exactly the matrix this project already computes and has certified**
(`ACCGain.matrix()`; 8/8 pre-registered gates, action-gradient cosine +0.954 against
contact-stratified finite differences). Nothing new is required to evaluate it.

### 1.2 The embodiment gap is multiplicative on the action

The VLA was trained so that action `a` produces the intended task motion **on A**:
`ẏ_intended = G_A a`. Executing the same `a` on B produces `ẏ_realized = G_B a`. The discrepancy is

```
d(a) = ẏ_realized − ẏ_intended = (G_B − G_A) a                                   (2)
```

This is the central structural fact, and it differs from every disturbance studied so far in
three ways that all matter:

1. **It is multiplicative in the action, not additive in the state.** A payload or a friction
   change enters as a generalised force; this enters as a *wrong gain on the command*.
2. **It is a REFERENCE-level error.** The inner loop tracks `q^des` faithfully — it is doing its
   job correctly while the target itself is wrong. Feedback does not attenuate it.
3. **It is persistent and structured**, not stochastic: `(G_B − G_A)` is a fixed matrix at a
   given configuration.

### 1.3 Proposition 1 (matching) — the correction exists

> **If `G_B` has full row rank (rank = dim y), the cross-embodiment discrepancy is exactly
> matched, and a residual cancelling it in task space exists.**

`G_B : ℝⁿ → ℝᵖ` with `n` actions and `p` task dimensions, `n ≫ p`. If `rank(G_B) = p` then
`range(G_B) = ℝᵖ`, so *every* vector in task space — including `d(a)` from (2) — lies in
`range(G_B)`. Setting

```
r*(a) = −G_B⁺ (G_B − G_A) a  =  (G_B⁺ G_A − I) a                                 (3)
```

gives `G_B(a + r*) = G_A a`, i.e. the realized task motion on B equals the intended motion on A,
exactly.

**The rank premise is already measured on X2: rank 6/6 in every state tested**
(`fault_authority_oracle.py`, three conditions, 45 states). This is not an assumption to be
checked later; it is the one premise that was verified before the direction was chosen.

**Contrast with the failed direction.** For a dynamics disturbance, the analogous question was
whether the *recoverable correction* lay in `range(Gᵀ)` — a 6-dimensional subspace of the 20-D
action space — and it did not (36% on `tq05`). Matching in (3) is a statement about
`range(G_B) = ℝᵖ`, the *output* side, which is full by rank. **The two directions fail and
succeed for structurally different reasons, and the difference is checkable in advance.**

### 1.4 Proposition 2 (representability) — the correction is a final-layer update

Equation (3) says `r*` is a **linear map of the action**, `r* = C a` with `C = G_B⁺G_A − I`.

Let the VLA's action head be affine in its last hidden feature, `a = W h + b`. Then

```
r* = C a = C(W h + b) = (CW) h + (Cb)
```

so the exact correction is realised by the final-layer update

```
ΔW = C W ,      Δb = C b                                                          (4)
```

> **The exact cross-embodiment correction lies inside the reachable set of a single affine
> update to the action head.** No deeper layer is required, and no nonlinearity is needed.

This is the reason to expect success where the walker failed. There, the repair was a constant in
a 20-D action space that a 6-D objective could not reach (ceiling 0.359). Here the repair is
*by construction* in the span of what the adapted layer can express.

Two riders, stated now rather than discovered later:

- (3) cancels the discrepancy in **task space** (`p` dims). Null-space motion is unconstrained —
  expected and harmless for task performance, but it means "exact" is exact in `y`, not in `q`.
- `C` depends on configuration through `G_A(q), G_B(q)`. A single constant `C` is exact only if
  the ratio is configuration-independent; otherwise `C` is the best constant approximation and the
  residual error is second-order. **Measurable** — see Test 3.4.

### 1.5 Proposition 3 (why feedback does not reject it)

For an inner loop with plant `P` and controller `K`, sensitivity `S = (I+PK)⁻¹` and complementary
sensitivity `T = (I+PK)⁻¹PK` satisfy `S + T = I` identically.

- A **dynamics disturbance** enters the plant and is attenuated by `S`. Inside the inner loop's
  bandwidth `S ≈ 0`: already rejected, nothing left for an outer layer to fix. Outside it, the
  outer layer's own commands are attenuated by `T ≈ 0`. **The band where the disturbance survives
  is exactly the band where the VLA has no authority.**
- The **embodiment gap** is an error in the reference itself. The inner loop tracks it with `T ≈ I`
  inside its bandwidth. `S` never acts on it.

**This is the formal statement of why the previous direction was doomed and this one is not.** It
also predicts, at the walker level, that a residual regulating what the base policy already tracks
gets rejected — which was measured three times (velocity row sign-flipped under torque limit,
downhill, and uphill).

### 1.6 The error signal — available online and unlabelled

Adaptation needs an error computable at deployment without supervision. The derivation supplies
one. VLAs emit an **action chunk** — `H` future actions — which is an implicit prediction of the
resulting motion. On robot A, realized ≈ predicted. On B,

```
e_k = y_realized(k) − y_predicted(k)                                              (5)
```

and by (2) this is `(G_B − G_A)a` integrated over the chunk. **The embodiment gap is directly
observable as the mismatch between what the model expected its action to do and what it did.**

This satisfies the constraint the walker work established the hard way: it does **not** regulate a
quantity the inner controller already tracks. The inner loop tracks `q^des`; (5) measures task-space
intent versus outcome, which nothing else is regulating.

Fallbacks if (5) proves too noisy: action-token entropy (TENT-style, available free for
discretised heads), or forward-model consistency.

---

## 2. Predictions this makes, before any experiment

| # | prediction | falsified if |
|---|---|---|
| **P1** | `rank(G_B) = p` at essentially all states on the target robot | rank deficiency is common |
| **P2** | The cross-embodiment correction is ≥ 80% expressible by a final-layer affine update | the reachable fraction is low, as it was for the walker (36%) |
| **P3** | A **constant** `C` recovers most of the gap; configuration dependence is second-order | only a state-dependent `C` works |
| **P4** | Adapting the **interface/projection layers** beats adapting the trunk at matched ‖ΔW‖ | the trunk wins, i.e. the shift is not where the theory says |
| **P5** | The prediction-vs-realization error (5) has positive alignment with the correction | alignment ≈ chance, as `task_error` did on the walker |
| **P6** | Online adaptation beats the frozen model **and** a displacement-matched random control | it does not — the direction dies here |

**P6 is the claim. P1–P5 are cheap and each kills the direction early if it fails.**

---

## 3. Test plan

Ordered so the cheapest disqualifying test runs first. Each phase has a gate; failing it stops
the work rather than triggering a search for a better setting.

### Phase 0 — Feasibility (≈ 1 day)

- **0.1** Stand up [GR00T N1](https://github.com/Nvidia/Isaac-GR00T) with released weights; reproduce
  a reported result on a supported embodiment (GR1 or Unitree G1). *Gate: reproduces.*
- **0.2** Confirm the architecture exposes per-embodiment **state/action projection layers** with a
  shared trunk, and that they are differentiable in PyTorch.
- **0.3** Choose the task and success metric. Manipulation needs scenes and objects; this is a
  different harness from the walking one and must be built or borrowed.
- **0.4** Decide the target: X2 in sim, or a second supported embodiment. **A second supported
  embodiment is the better first target** — it removes "we built the interface wrong" as a
  confound and lets the method be validated before the harder transfer.

**Kill criterion:** no runnable VLA + target pair inside a week → the direction is not testable
with available resources, and that is worth knowing on day 5 rather than week 5.

### Phase 1 — Is there a gap, and is it the right size? (≈ 1 day)

- **1.1** Run the source-embodiment model on the target, unmodified. Measure task success,
  end-effector error, and failure mode.
- **1.2** Run the model on its **own** embodiment as the upper reference.

**Gate (rule 3, learned expensively on the walker):** the gap must be large enough to recover and
small enough to leave a usable signal. If the model fails *completely* on the target there is no
gradient to adapt from; if it barely fails there is nothing to recover. **Report the headroom
explicitly and pick the transfer pair by it** — do not assume a pair is usable.

### Phase 2 — Verify the premises (≈ 2 days, no adaptation yet)

- **2.1 (P1)** Compute `G_B` via (1) across states; report rank, singular values, condition number.
  *Machinery exists* (`ACCGain`, `gradient_protocol.py`).
- **2.2** Certify `G_B` by the finite-difference protocol already built: replay gate, epsilon
  plateau, 32-direction holdout, contact stratification. *Gate: the same 8/8 that the walker's map
  passes.* An uncertified `G` invalidates everything downstream — this cost the project weeks once.
- **2.3 (P2)** Compute the reachable fraction of the correction under a final-layer affine update,
  the direct analogue of `reachable_by_any_objective.py`. *Gate: ≥ 80%.* **This is the single most
  diagnostic cheap test in the plan** — it is the number that was 36% for the walker and predicted
  its failure.

### Phase 3 — Oracle: does a bounded correction exist? (≈ 2 days)

- **3.1** Solve for `C` offline by regression against collected rollouts, or directly from (3) if
  both `G_A` and `G_B` are computable.
- **3.2** Apply `ΔW = CW, Δb = Cb` and measure task success. *Gate: recovers ≥ 50% of the headroom
  from Phase 1.*
- **3.3** Controls: a norm-matched random `ΔW`, and the zero update. **Report both** — "beats
  frozen" without a matched control is not a result, which cost this project a retracted claim.
- **3.4 (P3)** Compare a single constant `C` against a state-dependent one. Quantifies the rider
  in §1.4.

**Kill criterion:** if the offline oracle cannot recover the gap, no online method will. Stop.

### Phase 4 — The error signal (≈ 2 days)

- **4.1 (P5)** Compute (5) online and measure per-component sign agreement and cosine against the
  oracle `C` from Phase 3. *Gate: sign agreement ≥ 65% (chance = 50%).* This is the test the walker's
  objective failed at exactly 50%, and it is available before any adaptation run.
- **4.2** If (5) fails, test the fallbacks (action entropy; forward-model consistency) on the same
  criterion before writing any update rule.

### Phase 5 — Online adaptation (≈ 1 week)

- **5.1** Implement the update on the projection layers: gradient of (5) through the action head,
  trust-region projection `‖ΔW‖_F ≤ b_W`, anchor tether to `W₀`.
- **5.2** Calibrate the step size by the ‖ΔW‖ diagnostic on the **first** seed — the units lesson:
  a scale correction silently retunes every constant downstream of it, and this was nearly missed.
- **5.3 (P4)** Layer ablation: projection layers vs trunk vs both, at **matched ‖ΔW‖**.
- **5.4** Report `cond(J_{r,W} J_{r,W}ᵀ)` per candidate site and precondition where it is large —
  the conditioning result transfers directly and predicts which sites have a trustworthy gradient.

### Phase 6 — Confirmation (≈ 3 days)

- **6.1** Development split: adaptation vs frozen vs **displacement-matched random control**.
- **6.2** **Pre-register** the claim, threshold, statistic and seed pool *before* touching held-out
  data. Paired test chosen from the expected effect distribution, not from habit.
- **6.3** Run the untouched pool **once**. Report it whichever way it comes out.
- **6.4** Generalisation: a second transfer pair, and a task not used during development.

---

## 4. What makes this different from the direction that failed

| | walker / dynamics faults | cross-embodiment |
|---|---|---|
| disturbance type | additive, state-entering | **multiplicative on the action** |
| rejected by inner loop? | **yes** (`S ≈ 0` in band) | **no** — it is a reference error |
| correction reachable by the adapted layer? | 36% (measured, `tq05`) | **≥ 80% predicted by Prop. 2**, testable in Phase 2.3 |
| error signal available online? | yes, but sign-flipped (measured ×3) | (5), derived rather than assumed |
| headroom | had to be hunted; most cells inert | large by construction |

**The honest summary: the previous direction failed because the correction was not expressible
and the objective pointed away from it. Both are structural properties that were measured, and
both are predicted to be favourable here — with cheap tests (2.3, 4.1) that check them before any
adaptation code is written.**

---

## 5. Positioning

- **GR00T N1** already fine-tunes per-embodiment projection layers **offline, with a collected
  dataset.** So "adapt the projection layers" is not the novelty. The contribution must be
  (i) **online / test-time**, no dataset collection; (ii) the **derivation** of which layers, from
  where the shift enters relative to the control loops; (iii) **bounded and non-forgetting**.
- Read before committing: *Modality-Augmented Fine-Tuning of Foundation Robot Policies for
  Cross-Embodiment Manipulation on GR1 and G1* (arXiv 2512.01358) — closest prior work.
- **CLAE** (arXiv 2606.11489) edits *activations* of a frozen policy for behaviour **steering**, with
  an RL-trained steering module. Different problem, different edit site, and it motivates itself by
  catastrophic forgetting — a citation, not a competitor.
- The walker results become the **low-level half of one story**: a single principle — *adapt where
  the shift enters, relative to the loops* — that correctly predicts failure at the low level
  (measured, three conditions) and success at the interface level. That is a more coherent paper
  than either half alone, and it makes the negative results contributions rather than omissions.

---

## 6. Risks, stated up front

| risk | mitigation |
|---|---|
| No runnable VLA + target pair | Phase 0 kill criterion at one week |
| `C` is strongly configuration-dependent | Test 3.4 measures it; if so, adapt a state-conditioned head instead of a constant affine map |
| The chunk-prediction error (5) is too noisy | Fallbacks pre-listed in 4.2, screened by the same criterion |
| Manipulation harness is a large build | Prefer a supported second embodiment first; borrow an existing benchmark rather than building scenes |
| Result is positive but only in sim | Stated as a sim result; hardware is a separate claim |
| Reviewer: "this is just fine-tuning" | The distinction is online, bounded, no dataset — and must be demonstrated against an offline fine-tuning baseline, not asserted |

---

## Phase-0 vehicle amendment — 19 Aug 12:15, after hardware feasibility checks

- **GR00T N1 is blocked locally**: inference floor 16 GB+ VRAM (laptop: 8 GB), eval leans on
  Isaac (documented blocker on this machine).
- **openpi (Physical Intelligence, π0/π0-FAST/π0.5) is the preferred Phase-0 vehicle**:
  open torch weights; MuJoCo-based sim benchmarks (LIBERO / ALOHA-sim) — no Isaac; built-in
  websocket policy-server for remote inference (model on the CUDA box, sim client local);
  action CHUNKS make the §1.6 prediction-vs-realization signal native; continuous
  flow-matching action head = clean final-projection edit site (no FSQ).
- The validated walker method (gradient-free episodic search on the realised metric,
  layer-bias edit site, headroom/reachability gates, matched controls, pre-registration)
  transfers unchanged; LIBERO success rate is the realised metric.
- Phase 0 restated: stand up openpi, reproduce its reported LIBERO numbers via the policy
  server, then the fault suite (perception-side: camera/proprio bias; actuation-side:
  gain/offset; dynamics-side: object mass/friction) + headroom gate before any adaptation.
  GR00T remains the later cross-embodiment variant if hardware allows.
