# The geometric addition, in the residual-policy-for-a-frozen-VLA setting

Written 2026-09-08. This is the Claude track of a two-track pass. It specifies the piece
`manifold_review.md` recommends but does not write down, and it states what that piece can and
cannot do **in the setting that actually applies here**: an online residual correction wrapped
around a frozen vision-language-action policy.

Implementation: `learned_adaptation/symmetry.py`, tests in `test_symmetry.py` (8 exact-identity
tests). Nothing is trained and no robot result is claimed.

---

## 1. The setting fixes what the symmetry is a statement about

A frozen VLA emits a nominal action `a_nom`. The residual layer sends

  `a_sent = a_nom - Phi(xi) z_hat`   (additive), or   `E_hat a_sent = E_0 a_nom - Phi_d z_hat`

and the plant is the robot plus its low-level controller. The estimator models **the plant's**
response, not the policy's.

That distinction decides everything below. A morphological symmetry is a property of the robot —
replicated kinematic branches and compatible mass distribution — and it holds regardless of which
policy is driving. So the symmetry is available to the estimator even though:

> **The frozen VLA is not equivariant.** No released VLA is trained with a morphological
> equivariance constraint, so `pi(mirrored observation)` is not the mirror of `pi(observation)`.
> The closed loop is therefore **not** equivariant, and no claim here should say otherwise.

**This point was already made, and better, in `manifold_review.md`** (Section 4.2 discussion):
"Symmetric dynamics alone do not make an arbitrary frozen VLA equivariant, and a task instruction
can distinguish left/right, directions, or particular objects." The **task-instruction** mechanism
is one I missed entirely and it is the sharper one: a VLA is conditioned on language, and
"pick up the left cup" is not invariant under a left-right reflection, so the symmetry can be broken
by the instruction even if the policy weights happened to be equivariant. It also notes that
equivariance of an optimal *action set* does not make an arbitrary deterministic tie-breaker
equivariant.

The correct scope is: **the plant-side estimator and basis are equivariant; the policy is not;
the coefficient is free.** Any claim that reaches past that is unsupported.

**A qualification I got wrong, found by the second track.** "The coefficient is free, so asymmetric
faults are fine" is true only at contexts with trivial stabilizer. At a configuration **fixed by a
subgroup** — a bimanual robot at a mirror-symmetric pose, for instance — the Reynolds projection
averages over that stabilizer and the equivariant basis **loses rank exactly in the directions the
stabilizer acts on**. No coefficient can encode a one-limb fault there, because the required
direction has left the column space of `Phi`.

Measured with a context-dependent seed, so this is not an artifact of a constant one:

| context | stabilizer | rank Phi | residual on a left-only fault |
|---|---|---|---|
| `[0, 0]` | `G` | 1 | **0.707** |
| `[0.4, 0.4]` | `G` | 1 | **0.707** |
| `[0.4, -0.7]` | trivial | 2 | 0.000 |
| generic | trivial | 2 | 0.000 |

`0.707 = 1/sqrt(2)` is exactly the antisymmetric component of `[1, 0]`. This is not fatal — rank
returns as soon as the robot leaves the symmetric set — but it is a real blind spot: asymmetric
fault identifiability degrades near symmetric configurations, which is precisely where a bimanual
arm often rests. It compounds with the excitation limitation rather than being independent of it.

## 2. The equivariance conditions — ALREADY PRESENT, re-derived as verification

**Correction, added after the second track reported.** I originally presented the table below as
this track's derivation. It is not. `FORMULATION.md` Eq. (21) already contains the complete
transport table, in a more general form than mine: it separates state coordinates `T` from
descriptor-row coordinates `S`, which I collapsed into a single `rho_x`, and it gives
`Lambda' = V Lambda V^{-1}` as a **conjugation**, which I obtained as a congruence only because
orthogonality makes the two coincide. It also already covers the allocation metric
`M_a' = U^{-T} M_a U^{-1}`, and `test_core.py` already verifies it under nonorthogonal changes.

So this section is a **re-derivation that agrees with the existing one on its restricted domain**,
not a new result. Measured: my single-`rho` rule matches Eq. (21) at `1.1e-13` for orthogonal
signed permutations, and breaks at `1.8e+04` when applied to a general gauge with `T != S`, where
Eq. (21) remains exact at `5.9e-12`. Independent agreement on the special case is worth recording
as verification; presenting it as a contribution was wrong.

For morphological symmetries the representations are signed permutations and therefore orthogonal,
so the special case below is the one that applies in practice:

| object | transforms as | status |
|---|---|---|
| basis `Phi` | `rho_x Phi rho_z^T` | **modelling requirement** — this is the equivariance condition |
| descriptor `B`, metrics `L`, `R` | `rho_x (.) rho_x^T` | forced |
| residual `y`, tracking error `e` | `rho_x (.)` | follows |
| process weight `Q`, forgetting `Lambda` | `rho_z (.) rho_z^T` | forced |
| coefficient `z`, covariance `P` | `rho_z z`, `rho_z P rho_z^T` | **induced, not chosen** |
| allocation metric `W` | `rho_a W rho_a^T` | forced, on the controller side |

Verified over 256 randomized signed permutations: `max |zdot(g.x) - rho_z zdot| = 3.5e-12` and the
matching bound for `Pdot`. `matched_action` is equivariant on the controller side under an
invariant allocation metric. **No change to `core.py` was needed** — the law as written is already
covariant; what was missing was the statement of under what it is covariant.

## 3. The constraint that is easy to violate by accident

The table above is the *covariance* statement: transform everything, and the update transforms.
The statement one actually wants is stronger — that **one** estimator serves the whole orbit,
rather than a separate estimator per group element. That holds only if `L`, `R`, `Q` and `Lambda`
are **invariant**, i.e. lie in the commutant of the representation.

For a limb-swap symmetry the commutant ties symmetrically related coordinates together. So:

> **Independent per-joint gain tuning destroys equivariance.** Measured on a `C_2` limb swap with a
> diagonal `L = diag(1.0, 2.0, 5.0, 0.3)`: defect **8.3e+01**. With the same matrices projected onto
> the commutant: **1.1e-13**.

That is a real failure, not a rounding artifact, and it is silent — the estimator still runs and
still appears to converge; it is simply no longer the object the symmetry prior assumed.
`require_invariant` raises rather than warning, because a warning in this position is a defect that
ships. Per-joint tuning is the normal thing to do, which is exactly why it needs a guard.

## 4. What the symmetry buys here, concretely

**A weight-sharing prior on the learned basis.** `Phi` is meta-learned offline. Equivariance means
data recorded on one limb constrains the basis used on the other, which is a factor-of-`|G|`
effective multiplier on basis-training data and a consistency guarantee rather than a soft prior.

**It addresses a problem this setting actually has.** A frozen VLA drives limbs *asymmetrically* —
in the predecessor study's bimanual cell, one gripper does the grasping work and the other is
largely idle, and the two channels' command variance differed by 5-20x. Asymmetric excitation is
the normal case, not a corner case. An equivariant basis lets the well-excited limb's data inform
the representation used on the under-excited one.

**What it does not fix.** Equivariance shares the *basis*, not the *excitation*. The coefficient on
an under-driven limb remains poorly identified and its covariance grows accordingly. The symmetry
does not manufacture information, and no claim should suggest it does.

## 5. Novelty, stated narrowly and adversarially

| already occupied | by |
|---|---|
| meta-learned basis + composite Kalman-Bucy coefficient adaptation | Neural-Fly |
| + context features, + input-effectiveness adaptation | MAGIC-VFM |
| morphological symmetry groups, equivariant supervised models on robot morphology | MorphoSymm (Ordonez-Apraez et al., IJRR 2024) |
| equivariant policies / graph policies over a kinematic tree | MS-HGNN, MS-PPO |
| adaptive augmentation of a *pretrained frozen* policy, with hardware | Cheng et al., RA-L 2022 |

Two further corrections after the second track's source check. **Carrying the group action through
the estimator's objects is also occupied**: Gada et al., *Equivariant Filters are Equivariant*
(arXiv 2210.13728), abstract, explicitly treats consistent coordinate, origin and noise choices for
equivariant filters. And within this repository the transport itself is Eq. (21), which predates
both tracks.

What survives as genuinely additional, after removing everything already present: **the constraint
that a fixed hyperparameter set defines a symmetric algorithm only if `L, R, Q, Lambda` are
invariant** — Eq. (21) gives covariance under transforming *everything*, and is silent on the case
where the same numbers are reused across the orbit, which is what any implementation actually does.
Both tracks reached this independently. This track adds the measured demonstration and an enforcing
check that raises.

**But state its size honestly.** This is a *correctness condition*, not a result. It says how to
combine two existing things without breaking either. On its own it is a component of a paper, not a
paper. To carry weight it has to buy something measurable — basis sample-efficiency at fixed data,
or transfer to a limb held out of basis training — and that experiment has not been run.

## 6. What remains unbuilt

**The wrench basis cannot run against the current core.** `Phi_tau = J_c(q)^T Phi_F` carries
joint-torque units and is admissible only in the torque row of `FORMULATION.md` Eq. (5), whose
descriptor `B = diag(I, M(q))` is state dependent with a rectangular input map. `core.NominalModel`
is a constant-matrix model (Eq. 3), and `integral_observation` additionally assumes constant
`A, B, E`. So the geometric feature the review recommends most strongly needs a model class that
does not yet exist in the reference core. `wrench_basis_joint_rows` provides the shape and frame
contract and says this in its docstring rather than implying it is usable.

Also unbuilt: any verification that a *particular robot's* claimed symmetry is a reachable
morphological transformation rather than a mirrored URDF. `MorphologicalAction` validates
orthogonality only and says so.
