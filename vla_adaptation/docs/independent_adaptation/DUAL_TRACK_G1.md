# Dual-track convergence — the geometric addition

2026-09-08. Two tracks worked the same task from the same artifacts without seeing each other's
answer: Claude (`learned_adaptation/symmetry.py`, `SYMMETRY_FOR_FROZEN_VLA.md`) and Codex
(`learned_adaptation/geometry.py`, `GEOMETRIC_METHOD.md`, `codex_prompt/CODEX_G1.md`). 32 tests
pass across the combined suites. This records what converged, what diverged, and — the part that
decides how much the agreement is worth — what the two tracks **shared**.

## 1. Common-mode audit, first

Agreement between the tracks is only evidence where they did not inherit the answer. Both read
`FORMULATION.md`, `manifold_review.md` and `core.py`, so:

| apparent agreement | actually independent? |
|---|---|
| the wrench basis is dimensionally inadmissible in the first-order servo model | **partly.** §2's Eq. (3) vs Eq. (5) already distinguishes them and already says `T` is a servo time scale, not inertia. Claude derived it before reading §2 and Codex read the source; either way the answer was available in shared text. |
| the estimator transport table | **no.** This is Eq. (21) verbatim, already in the repository and already numerically tested. Both tracks agreeing with a document both had read is not corroboration. |
| the frozen VLA is not equivariant | **no.** Already in `manifold_review.md`, and stated better there — it names the *task instruction* as a symmetry breaker, which neither track's fresh reasoning recovered. |
| **fixed hyperparameters must be invariant, not merely transported** | **yes.** Not present in any shared document. Both tracks reached it separately. |

Both tracks also verified against the **same** `core.composite_rhs`. A defect in that function would
be inherited by both. The mitigation is that its Lyapunov identity test is independent of anything
geometric and was re-run.

## 2. The one genuinely converged finding

Eq. (21) establishes **covariance**: transform every object and the update transforms. It is silent
on the case every implementation actually runs — one fixed set of hyperparameters reused across the
whole orbit. That holds only if `L`, `R`, `Q` and `Lambda` are **invariant**, i.e. lie in the
commutant of the representation, which for a limb swap ties symmetrically related coordinates
together.

Codex states it in a docstring and **tests it per object** —
`test_untransported_hyperparameters_break_update` transports everything except one of `P`, `L`, `R`,
`Q`, `Lambda` in turn and asserts each omission breaks the update, which is per-object attribution
this track did not produce. Claude measured the complementary case, one fixed hyperparameter set
reused across the orbit: on a `C_2` limb swap with `L = diag(1.0, 2.0, 5.0, 0.3)` the defect
is **8.3e+01**; projected onto the commutant, **1.1e-13**. The failure is silent — the estimator
still runs and still appears to converge — and per-joint gain tuning is the normal thing to do, so
`require_invariant` raises rather than warns.

## 3. Divergences, and how each resolved

**Generality of the transport (Codex correct).** Claude used a single orthogonal `rho_x` for both
state and descriptor-row coordinates and a congruence for `Lambda`. Eq. (21) and Codex's
`LinearAction` separate `T` from `S` and use conjugation `V Lambda V^{-1}`. Measured: the two agree
at `1.1e-13` on orthogonal signed permutations, the general form stays exact at `5.9e-12` under a
nonorthogonal gauge with `T != S`, and the single-`rho` rule breaks there at `1.8e+04`. Since
morphological symmetries act by signed permutations, the special case is the one that applies in
practice, but the general form is the correct object and `geometry.py` is the reference for it.

**Novelty (Codex narrower, and right).** Claude scoped the novelty to "carrying the group action
through the estimator's objects." Codex's source check found that occupied too — Gada et al.,
*Equivariant Filters are Equivariant* (arXiv 2210.13728), abstract. After removing everything
already present in the repository or the literature, what remains is a correctness condition, not a
result.

## 4. Uniques worth keeping from each track

**Codex only:** the compliance path `servo_rows` with an explicitly required, never-defaulted
force-to-command map; `EffectivenessFeatures` for the MAGIC-style input-map branch; full group-table
validation (associativity, inverses, representation law over all pairs, five representations);
`SpatialFrameChange` with declared twist/wrench origin conventions; `project_basis`, which
*constructs* an equivariant basis by Reynolds-averaging an arbitrary seed; and the Gada precedent.

**Codex only, and it qualifies a claim this track made confidently:** at a context fixed by a
subgroup, an equivariantly-projected basis loses rank in the stabilizer's non-trivial directions, so
no coefficient can encode a one-limb fault there
(`test_invariant_latent_cannot_encode_left_only_at_symmetric_context`). Verified independently with
a context-dependent seed: rank 1 and residual `0.707` at symmetric contexts, rank 2 and residual
`0.000` at generic ones. This track had asserted "the coefficient is free, so asymmetric faults are
fine" without noticing it holds only where the stabilizer is trivial.

**Claude only:** the measured commutant demonstration and the enforcing check; and the observation
that this setting has asymmetric excitation by construction — a frozen VLA drives limbs unevenly,
and in the predecessor bimanual cell the two gripper channels differed 5-20x in command variance —
so an equivariant basis lets the well-excited limb inform the under-excited one. That is a
motivation for the component, not evidence for it.

## 5. Converged status

The geometric method is **specified and implemented**; nothing is trained and no robot result is
claimed. Both tracks independently reach the same judgement on what it is worth: the composite law,
wrench pullback, group averaging and covariance transport are occupied or standard, and applying
them to a manipulator or a frozen VLA is not by itself a contribution. What would make it one is
evidence that the equivariance constraint **buys** something — basis sample-efficiency at fixed
data, or transfer to a limb held out of basis training. Neither track ran that, and it is the next
thing worth doing.

The wrench basis remains unusable against the current reference core, which implements only the
constant-`A, B, E` model. Closing that requires the state-dependent descriptor of Eq. (5).
