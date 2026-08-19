# When does online adaptation work? Six conditions, and how to choose the action, the objective, and the sensor

17 Aug 2026. Every condition below is stated so it can be **measured before building anything**,
and each is tied to a measurement already in this project. Written after
`docs/GRADIENT_MAPPING_AND_MATCHING_CONDITION.md` retracted the categorical
position-vs-momentum argument — so nothing here argues from the *name* of a channel or a
disturbance. Where a claim needs a measurement, it says which one.

---

## 0. Setup

Deployment presents conditions `c ~ C` (terrain, payload, fault, wear). The adapted quantity is
`θ ∈ Θ`, bounded by the deployed limits. `M(θ; c)` is the **true task metric** — distance,
completion, restricted survival — higher is better.

Three objects, and almost every confusion in this project is a confusion between them:

| | definition | meaning |
|---|---|---|
| `θ_static` | `argmax_θ E_c[M(θ;c)]` | the best single controller |
| `θ*(c)` | `argmax_θ M(θ;c)` | the oracle schedule |
| **`V_adapt`** | `E_c[M(θ*(c);c)] − E_c[M(θ_static;c)]` | **everything adaptation can ever win** |

`V_adapt` is the **regret of the best static policy on the deployment distribution**. It is a
property of the *task*, not of your algorithm. No law, gain, layer, certificate or sensor can
exceed it.

---

## 1. The six conditions

All are necessary. The project has failed at every one of them at least once, and — this is the
useful part — at *different* ones for different directions.

### A. Value: `V_adapt > noise`

If one static setting is near-optimal everywhere, adaptation is pointless however well it works.

- `gait_period_s`: conflict is **real** (optima split 0.70/0.80 within grade, payload *and* push)
  yet `V_adapt = 0.0 steps` — the two contenders perform identically (§4q).
- impedance `kp`: `V_adapt = +17.6 steps (7%)`, and the best static value is `1.00` = nominal (§4o).

**Conflict is not sufficient.** A two-sided conflict with a zero ceiling is still a retune. Both
gates must be measured, and gate 2 is the one that kills.

> **Why `V_adapt` is usually small, and this is the deepest point in the file.** RL with domain
> randomisation over `C` *returns approximately `θ_static`* — that is its objective. So on any
> condition set the policy was trained over, `V_adapt` is small **by construction**, and the best
> static value landing on nominal (§4o) is training working as designed, not a coincidence.
> `V_adapt` therefore measures **train/deploy distribution shift**, not controller quality.
> Adaptation is worth doing exactly where training could not have averaged: hardware faults,
> damage, wear, sim-to-real, payloads outside the randomisation range.
>
> **CORRECTED 17 Aug — this is right in kind and wrong in application.** Measured on
> single-axis OOD faults, `V_adapt` is **5.3%** against the trained family's 7% (LESSONS
> §4aa): no *individual* off-distribution axis creates room, and `payload` is explicitly OOD
> with no headroom at all. The room is in the **conjunction**: LESSONS rule 22 finds all 16
> single-axis sim-to-real cells inert while seven individually-harmless mismatches applied
> **together** take the frozen policy from 250 steps to **92**. Domain randomisation buys
> marginal robustness, not robustness to the conjunction — so condition A must be screened on
> INTERACTIONS, not marginals. A marginal screen (which is what I ran) returns a confident and
> wrong negative.

**Test.** Sweep `θ` over a grid × the condition set, passive. Report `E[M(θ*(c))]` against
`max_θ E[M(θ)]`. Costs one sweep, no adaptation, and it has killed two directions here.

### B. Identifiability: `θ*(c)` must be a function of what you can observe

If two conditions with the same observable signature want *opposite* corrections, no causal
controller can serve both — regardless of the law.

- Proprioception: `θ*` correct **17/27** within family, **3/27** across families (§4o).
- The disturbance observer identifies the *family* 25/27 (§4s) — but `θ*` across families is still
  3/27, because that is extrapolation to an unseen disturbance type, a property of the lookup
  table rather than the sensor.

**Test.** Leave-one-**family**-out, scored on **realised control performance** (look up what the
schedule would actually have got), not classification accuracy. Rule 13.

### C. Reachability over the horizon, under the faulted plant, with the real bounds

Not one-step, not unbounded, not on the nominal plant.

- `friction_ood`: **80%** of one-step task-error drift removable within budget → recovered distance
  **−0.07 m**, 2/8, pre-registered (§4y). One-step reachability does not establish horizon payoff.
- Contrapositive, and it matters: a constant `hip_pitch = −0.10` recovered **+28.7 steps** uphill,
  so the channel *was* reachable in a cell where the adaptive update found nothing. **Failure to
  find ≠ failure to reach.**

**Test.** Box-constrained oracle over horizons `{h₀, 2, 5, 10, 27, 54, …}` on the *faulted* plant,
scored on the true metric, against norm- and saturation-matched random sequences. Report recovered
headroom `R = (M_oracle − M_off)/(M_nominal − M_off)`. Not a clipped pseudoinverse.

### D. Objective alignment: `⟨∇_θ L_surrogate, ∇_θ M_true⟩ > 0`

You descend a surrogate; you are judged on the metric. If they disagree, descending harder hurts.

- `V = sᵀQs` decreases at every authority level across 50× while survival degrades monotonically
  (§4b).
- Early `V` carries **no** information about falling 2.6 s ahead (+0.10), while spiking 28× in the
  last half-second (§4k).

> **The cheapest test of D, and it generalises beyond robotics.** Vary the *fidelity* of your
> gradient and look at the **sign** of the performance change. If a more accurate gradient makes
> the controller **worse**, your objective is misaligned — you are now descending the wrong thing
> more efficiently. Measured here: restoring `diag(s)` improved gradient cosine (L1 0.446 → 0.513,
> L3 0.752 → 0.865) and moved payoff `+16.2 → +1.5` steps (§4w/§4x). Three fidelity points,
> monotone in the wrong direction.
>
> You usually have two versions of a gradient lying around. This costs one paired run and tests
> the objective, not the gradient.

### E. Sample complexity: `dim(θ) ≲ n_eff`

Online data is *relevant* but not *plentiful*. A 400-step episode with a 27-step gait and one
disturbance realisation gives

    n_eff ≈ (T / τ_corr) × n_conditions ≈ (400/27) × 1 ≈ 15 effective samples

and for identifying the *fault* it is `n = 1`, whatever the step count.

| method | `dim(θ)` | outcome |
|---|---|---|
| `gait_period_s` | 1 | +104 steps |
| `kp` | 1–6 | uphill 250/250, no adaptation |
| oracle canceller | 6 | +54.2 |
| privileged CEM | 12 | +33.8 |
| ES on `mlp.6`, true objective | **2,580** | −14% train, **≈0 held-out** |
| ACC on `mlp.6` | **2,580** | +56.8 → vanishes at a good default (§4u) |

Everything durable adapted **≤ 12** parameters. Layer-scale fits produce train-set gains that do
not transfer — and this is not the objective's fault: ES optimised the **true simulator cost** and
still overfitted.

### F. Timescale: `τ_gait < τ_adapt < τ_fail`

Adaptation must be slower than the gait (or it fights the limit cycle) and faster than the failure
(or it arrives late). Here `τ_gait = 27` steps and `τ_fail = 53–66` steps — **less than a factor of
two**. A 50-step gated confirmation cannot fire in time, which is why `--continuous` was required
(§1).

---

## 2. How to choose the ACTION (channel)

Enumerate candidates by their input map, then **measure**, in this order — cheapest and most
lethal first:

| channel | input map | `dim(θ)` |
|---|---|---|
| position residual | `∂τ/∂a = K_p D_s` (constant) | 1 … 10⁵ |
| stiffness | `∂τ/∂σ = K_p(q_des − q)` (state-dependent) | 1–6 |
| damping | `∂τ/∂σ_d = −K_d q̇` | 1–6 |
| timing / phase / foot placement | changes the future **contact sequence** | 1–3 |
| centroidal wrench | acts on the momentum model directly; needs a feasible allocator | 6 |

1. **`V_adapt` (condition A)** — oracle schedule vs best static, on the true metric. Reject if
   small. *Killed `gait_period_s` (0%) and capped `kp` (7%) before either was built.*
2. **Horizon oracle `R` (condition C)** — box-constrained, faulted plant. Reject below ~50%.
3. **`dim(θ)` (condition E)** — among survivors, prefer the smallest.

Two things worth knowing while choosing:

- **State-dependent input maps are not automatically better, but they are differently bounded.**
  Stiffness authority scales with servo error, so it is largest exactly when tracking is worst —
  and near-zero on a well-tracked nominal orbit, which makes it excitation-starved. Position
  authority is constant and available always. Neither dominates; measure.
- **Timing channels are qualitatively different: they change the reachable set itself** by changing
  the contact sequence, rather than moving within a fixed one. That is why `gait_period_s` was the
  largest single lever measured (+104 steps). It still failed condition A — which is the whole
  point of checking A first.

---

## 3. How to choose the OBJECTIVE (error / cost)

Five requirements. The first three are free and run on data you already have.

1. **Sign-consistency across conditions.** A quantity whose relationship to benefit reverses
   between conditions cannot be a certificate however reachable it is. *Killed: roll, `vx`,
   `load_excess`, `serr_asym_LR`.*
2. **Not a pure magnitude** (rule 15). Any quantity monotone in the adapted parameter has
   constant-sign sensitivity **by construction** — it can say "more" or "less", never "which way".
   Check `∂h/∂θ` analytically first: one line of algebra retired six candidate rows.
3. **Predictive at usable lead**, on passive rollouts (rule 12). Higher early value must forecast
   earlier failure at a lead beyond `τ_fail`, sign holding in *every* condition.
4. **Moving it improves the true metric** (rule 19). Needs an intervention; it is the one that
   caught `friction_ood`.
5. **Passes the fidelity-sign test** (condition D above).

**If nothing passes, drop the surrogate and optimise the true metric directly** (ES/CEM on
realised return). That removes requirements 1–5 entirely at the cost of sample efficiency — which
is exactly why it then needs many episodes and runs straight into condition E.

---

## 4. How to choose the SENSOR

1. **Measure before the loop rejects** (rule 16). A good regulator destroys the signal that
   identifies what it is rejecting — the dual of persistency of excitation. Tracking error, pitch
   rate, leg power and servo error are all *post*-rejection, and every one of them flipped sign by
   condition. The dynamics residual `y = v̇ − f(v,u,t)` is *pre*-rejection and identified the
   family **25/27** where all proprioception failed (§4s).
2. **Carry a direction, not a magnitude** (rule 15) — same check as objective requirement 2.
3. **Discriminate within a family** (rule 13). Across families a sensor can score well by
   reproducing the experiment's design — which disturbance types you happened to include.
4. **Score on realised control performance**, not classification accuracy (§4o). A sensor that
   identifies the family perfectly and still leaves `θ*` at 3/27 has not helped.
5. **Then check the actuator can use it** (rule 17). Identification and cancellation are separate
   budgets; this project paid for the first without the second twice.

---

## 5. The order, and why it is this order

    A  value        does anything beat the best static setting?          sweep, no adaptation
    B  identifiab.  is θ*(c) recoverable from observables?               leave-one-family-out
    C  reachability is it in the bounded horizon reachable set?          box-constrained oracle
    E  dimension    dim(θ) ≲ n_eff?                                      count
    D  objective    does the surrogate agree with the metric?            fidelity-sign test
    F  timescale    τ_gait < τ_adapt < τ_fail?                           two numbers

A, B, C and E need **no adaptation law at all**. D needs one paired run. Only F needs the deployed
loop. This project ran the order backwards — building the law first, then the gradient, then the
objective, and never measuring A until §4o.

**Each failure has a distinct fix, which is why distinguishing them matters:**

| fails | meaning | fix |
|---|---|---|
| A | nothing to win | ship the static value; change the deployment distribution |
| B | can't tell conditions apart | new sensor, pre-rejection |
| C | can't get there | new channel, or wider bounds |
| D | descending the wrong thing | new objective, or optimise the metric directly |
| E | too many parameters | shrink the class |
| F | too slow or too fast | change trigger/gain, not the objective |

---

## 6. What this predicts for this project

Mapping every major direction onto its first failing condition:

| direction | first failure | evidence |
|---|---|---|
| layer adaptation on `sᵀQs` | **D**, then **E** | §4b, §4k; ES held-out ≈ 0 |
| gait-period scheduling | **A** | `V_adapt = 0.0` (§4q) |
| impedance scheduling | **A** (7%) then **B** | §4o: 3/27 across families |
| feedforward cancellation | **A** (+11 to +24 steps) | §4v corrected |
| `friction_ood` residual | **C** at horizon | §4y, pre-registered |

**No direction has failed only one condition, and no two have failed the same one first.** That is
the paper: not "adaptation does not work here", but *six independent requirements, each measurable
in advance, and this system violates a different one on every axis*.

**The constructive corollary.** To get a positive, choose a cell where A is large **by
construction** — genuinely outside the training distribution — then verify B, C, E in that order
before writing any law. On this robot the honest candidates are hardware faults and the
sim-to-real gap, not resampled terrain.
