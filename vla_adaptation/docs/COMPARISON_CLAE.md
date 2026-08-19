# Ours vs CLAE (arXiv 2606.11489) — where the two approaches actually differ

18 Aug 2026. *Steering Multirobot Behavior via Closed-Loop Affine Activation Editing*, Das, Chiu,
Hegde, Sukhatme. Read from the abstract/summary; details marked **[verify]** need the PDF.

---

## 1. CLAE in our vocabulary

| stage | CLAE |
|---|---|
| base | frozen policy, weights never touched |
| basis | **sparse autoencoder** on frozen-policy activations; behaviour-relevant latents chosen by **post-hoc probing** |
| edit | **state-dependent affine** (scale + shift) on the *selected latents* |
| who sets it online | a **lightweight RL steering policy**, trained offline |
| objective | the RL reward for the **target behaviour** |
| platform | multi-quadrotor navigation; sim + hardware |

This is **exactly the architecture derived in `LORA_ONLINE_DESIGN.md` §2** — freeze the factors,
adapt the coefficients — with two of the three open choices answered differently from what I
proposed, and one answered the same way.

## 2. Architectural mapping

| | ours (ACC layer adaptation) | CLAE |
|---|---|---|
| **what is edited** | layer **weights** `ΔW` | **activations** (selected latents) |
| **dim(θ) online** | 2,580 – 82,432 | small, sparse **[verify count]** |
| **basis** | none — the whole layer | SAE latents, selected by probing |
| **who decides online** | Lyapunov gradient on `sᵀQs` | **RL policy trained offline** |
| **objective** | hand-designed surrogate | **task reward** |
| **goal** | restore performance under disturbance | **produce behaviour outside the base envelope** |

Against our six conditions, CLAE satisfies by construction the two that killed us:

- **E (dimension).** A sparse selected latent set is orders below a layer. Ours failed this — the
  ES learning curve gave held-out ≈ 0 at every training-set size for 2,580 params.
- **D (objective).** The steering policy is trained on the **task reward**, so no surrogate is
  descended online. Ours failed this — `sᵀQs` agrees with the known fix on 50% of components,
  exactly chance.
- **B (identifiability)** is *learned*: the steering policy is state-conditioned and trained
  offline, so the observation→edit map is fitted rather than assumed. Same structure as RMA.

## 3. THE decisive difference, and it is about the task, not the method

`V_adapt` — the regret of the best static policy — is the quantity that caps everything
(`WHEN_ADAPTATION_WORKS.md` §0).

- **Ours: restore performance on a fixed task under a disturbance.** A static retune is a
  legitimate competitor, and it has now won **five times** (§4aa): gait period, and `kp` on trained
  conditions, hardware faults, nominal, and lag. On our best cell the shipped stiffness alone is
  worth +15–22%.
- **CLAE: produce a behaviour the base policy does not have** — surveillance avoidance, commanded
  velocity profiles, formation preservation. **There is no static setting that emits a novel
  behaviour on demand.** `V_adapt` is effectively unbounded because the comparator does not exist.

> **Adaptation competes with a retune when the goal is to RESTORE. It has no static competitor when
> the goal is to PRODUCE a behaviour the base policy lacks.**

That single sentence explains most of this project's negatives without appealing to any defect in
the law, and it is the most useful thing to take from the comparison. Our task class was chosen so
that a retune is always available; theirs was not.

## 4. The second difference: where the online decision comes from

CLAE moves **all** the hard learning offline — SAE, probing, steering policy — leaving a forward
pass at inference. RMA does the same. We put estimation, objective design and optimisation *online*,
which is where conditions B, D and E each fail.

Our own confirmed positive (`tq05`, +1.70 m held-out, ≈68% of headroom, beating a norm-matched
random control) works precisely because it abandoned the online gradient: CEM on the **realised
metric** over a **20-parameter constant**. It is the same lesson arrived at from the other side.

## 5. Where I disagree, or would want evidence

- **The basis is built from activations, not corrections.** I argued in `LORA_ONLINE_DESIGN.md` §3
  that the subspace a policy's *features* occupy is not the subspace its *corrections* occupy.
  Probing for behaviour-relevant latents partly answers that — selection is by relevance, not
  variance — but it is still a basis over activations. For *disturbance recovery* specifically I
  would still build the basis from oracle fixes; the multi-fault SVD now running is that test.
- **[verify] Is there a static/retuned baseline?** Our whole finding is that this comparison is
  where adaptation papers die. If CLAE does not include one, that is the first question a reviewer
  should ask — and if it does and CLAE wins, that is a strong result worth citing as such.
- **[verify] Does base-task performance degrade?** They claim no catastrophic forgetting; the
  quantity to check is the frozen policy's own metric with edits active but no steering demanded.
- **[verify] `dim(θ)` and the number of selected latents** — the number that decides whether their
  condition E margin is comfortable or marginal.

## 6. What we have that they do not

The methodology, and it is not a small thing: pre-registration with decision rules fixed in
advance, displacement-matched controls with the explicit finding that **a random control is not a
floor** (`acc − frand` = +251.9 against a `frand − off` = −267.5), ceiling-first screening, and 25
rules derived from measured failures. Applied to CLAE's own claims those would be exactly the right
scrutiny.

## 7. What to do with this

1. **Reframe the paper's scope**: our negatives are about *restoration* tasks, where a static
   competitor always exists. Say so, and cite CLAE as the contrasting regime where it does not.
   That converts "adaptation didn't work" into "here is precisely when it cannot beat a retune".
2. **Adopt the offline-heavy structure.** Our one positive already did.
3. **Keep the basis question open and settle it empirically** — the multi-fault SVD decides whether
   corrections are low-rank at all, which is the piece neither CLAE nor RMA answers for
   *disturbance recovery*.
