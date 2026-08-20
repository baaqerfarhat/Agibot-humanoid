# Running a VLA on the AgiBot X2 Ultra by adaptation instead of fine-tuning

**20 Aug 2026.** The objective, stated plainly: put a vision-language-action model on the
X2 Ultra **without collecting a dataset and fine-tuning it**. This document replaces
`PHASE0_OPENPI.md` as the road to that objective. openpi/LIBERO is retained, demoted to
what it actually is — a methodology testbed with a computable ceiling.

Everything in §1 was measured on this machine before this document was written.

---

## 1. Measured ground

| fact | evidence |
|---|---|
| GPU 1 is a 48 GB RTX 8000; GPU 0 is dead and breaks CUDA enumeration | address GPU 1 by UUID; `nvidia-smi --id=0000:C1:00.0` for queries |
| Turing (sm_75) is **not** a blocker: torch 2.9+cu128 still ships sm_75 | `arch_list` includes sm_70/75/80/86/90/100/120 |
| GR00T N1.7 **loads and infers correctly here** | `nvidia/GR00T-N1.7-3B`, DROID sample, MSE ~1e-3 vs ground-truth actions |
| ...at **0.377 s/inference step**, after two mandatory settings | 13.81 s/step without them — a 37x difference |
| `REAL_G1` — a pretrained **humanoid** head — is in the base checkpoint | listed by the checkpoint's own supported-tag error |
| The per-embodiment interface is explicit: **32 slots** over a shared trunk | tensor shapes, §3 |
| A flow-matching head **attenuates** a constant bias edit ~3x once trained | pi0.5 measured 0.340 +/- 0.026 vs 0.942 at random init |

**The two mandatory settings, because both fail silently:**

1. `sys.modules["flash_attn"] = None` before importing gr00t. Having flash-attn installed
   is *worse* than not: the import succeeds on Turing, the code selects
   `flash_attention_2`, and the kernel dies at runtime. Blocking the import triggers
   GR00T's own documented sdpa fallback.
2. `policy.model.half()` after load. The config defaults to bf16, which has **no**
   memory-efficient SDPA kernel on sm_75, so PyTorch quietly drops to MATH attention.
   Nothing errors — the model is just 37x slower. fp16 was also marginally *more*
   accurate (MSE 0.001769 vs 0.001798); the overflow fear did not materialise.

---

## 2. Architecture — and why the embodiment gap lands where the theory wants it

A VLA must not emit 31 joint targets for X2. It emits **task-space actions**, tracked by a
whole-body controller — and the bottom half already exists in this project:

```
GR00T N1.7  --(relative end-effector actions)-->  X2 whole-body controller  -->  31 joints
   frozen trunk                                    holosoma box_pickup WBC
   + adapted X2 embodiment slot
```

This placement is the whole point. `PLAN_CROSS_EMBODIMENT.md` §1.1 defines the task output
`y = Phi_X(q)` and the action-to-task map `G_X` at exactly this level, and **`ACCGain.matrix()`
already computes `G` for X2 and is certified 8/8** (action-gradient cosine +0.954 against
contact-stratified finite differences). So Prop. 1's rank premise and Prop. 2's
reachability test can be evaluated on the *real target* rather than on a tabletop analogue.

GR00T supplies the other half of the premise: its action space is a **relative
end-effector space shared across robot and human embodiments**. Prop. 1 needs a common
task output `y` on both sides of the transfer. LIBERO->X2 has none — a 7-DoF tabletop arm
and a 31-DoF humanoid share no task space, which is why that pairing was never the road
here. G1->X2 does.

---

## 3. The edit site, with shapes

From `model.safetensors.index.json` of the base checkpoint. Every tensor with a leading
axis of 32 is **per-embodiment**; everything else is the shared trunk.

| tensor | shape | role | class |
|---|---|---|---|
| `action_head.action_decoder.layer2.b` | (32, **132**) | final action bias | **b6** |
| `action_head.action_decoder.layer2.W` | (32, 1024, 132) | final action projection | **w6** |
| `action_head.action_decoder.layer1.{W,b}` | (32, 1024, 1024) | decoder hidden | — |
| `action_head.state_encoder.layer1.{W,b}` | (32, 132, 1024) | proprioception in | b0 analogue |
| `action_head.state_encoder.layer2.{W,b}` | (32, 1024, 1536) | proprioception in | — |
| `action_head.action_encoder.W1/W2/W3` | (32, ...) | noisy-action in | — |
| `action_head.model.transformer_blocks.*` | no 32 axis | 32-layer DiT | **shared** |
| `action_head.vl_self_attention.*`, `vlln` | no 32 axis | VL fusion | **shared** |
| VLM backbone (Cosmos-Reason2-2B) | no 32 axis | perception + language | **shared** |

Three consequences, and they are the reason this vehicle fits the theory rather than
merely being available:

- **Prop. 2 is satisfied by construction.** The exact cross-embodiment correction is an
  affine update to the action head. `action_decoder.layer2` *is* that head, and it is
  already factored out per embodiment. No basis has to be invented.
- **Condition E is satisfied.** A bias edit is 132 parameters, and only X2's active
  task dims are live — of order 6-7 for an end-effector action space, against the
  walker's 20 and openpi's 32. Every durable result in this project adapted <= 12.
- **Non-forgetting is architectural, not a claim.** Editing slot `k` cannot touch any
  other embodiment's behaviour, because no other embodiment reads those weights.
  `PLAN_CROSS_EMBODIMENT.md` §5 lists "bounded and non-forgetting" as one of three
  novelty criteria; here it is free, and provable by inspection rather than by measurement.

---

## 4. The baseline, stated in NVIDIA's own code

GR00T enumerates its embodiment tags. There is **no AgiBot entry**. The supported path for
a new robot is a tag literally classified **finetuning-only**:

    NEW_EMBODIMENT   "Any new embodiment."   -> collect a dataset, run launch_finetune.py

That is the comparison. Their answer to a new robot is *collect data and fine-tune the
projection head*. Ours is *adapt that same head online, no dataset*. **Same layer, same
architecture, opposite method** — which makes the baseline concrete and runnable instead of
rhetorical. There are 32 slots against ~16 named tags, so free slots exist for both arms.

Any claim must be reported against a fine-tuned `NEW_EMBODIMENT` arm, not against frozen
alone. `PLAN_CROSS_EMBODIMENT.md` §6 already lists "reviewer: this is just fine-tuning" as
a risk whose mitigation is "must be demonstrated against an offline fine-tuning baseline,
not asserted".

---

## 5. Phases, each with a gate that stops the work rather than triggering tuning

### Phase A — the evaluation harness  *(gate: an X2 manipulation task runs closed-loop)*

The honest blocker, and it is a build rather than a question: GR00T needs RGB + language +
proprioception from an X2 scene, and something must score task success. Candidates, in
preference order: the existing holosoma/Isaac `box_pickup` scene with cameras added; mjlab.

**Gate:** a scripted (non-VLA) policy completes the task and the harness reports a success
rate. **Kill:** no runnable X2 manipulation scene within two weeks -> the sim target is not
available and the direction waits on hardware, which is worth knowing early.

### Phase B — is there a gap, and is it the right size?  *(the condition-A gate)*

Run the frozen `REAL_G1` head on X2 through the harness. Report success rate, end-effector
error, and failure mode against an X2-appropriate reference.

**Gate (rule 3):** the gap must be large enough to recover and small enough to leave signal
— frozen success in (5%, 70%). Floor-dead gives no search signal (the tq04 lesson);
near-ceiling gives nothing to win. **Report the headroom explicitly.**

### Phase C — verify the premises, no adaptation yet  *(the cheap disqualifiers)*

- **C1 (P1).** `rank(G_B) = p` across states on X2 via `ACCGain.matrix()`. *Already measured
  6/6 in 45 states — re-confirm in the manipulation configurations, which are new.*
- **C2.** Certify `G_B` by the existing finite-difference protocol. *Gate: the same 8/8.*
- **C3 (P2).** Reachable fraction of the correction under an `action_decoder.layer2` affine
  update. **Gate: >= 80%.** This is the number that was 36% for the walker and predicted its
  failure, and it is the single most diagnostic cheap test in the plan.

### Phase D — oracle  *(gate: >= 50% of Phase B headroom)*

Solve `C` offline, apply `dW = CW, db = Cb` to X2's slot, measure. Controls: norm-matched
random `dW`, and the zero update. **Kill: if the offline oracle cannot recover the gap, no
online method will.** Also measure the trained-flow attenuation on GR00T's head as §1's
pi0.5 measurement did — a flow-matching decoder may attenuate the edit, and the factor
sets every downstream step size.

### Phase E — adapt, gradient-free first  *(the validated method)*

`core/episodic_search.py` over X2's `action_decoder.layer2` bias, scored on realised task
success. `seeds_per_gen >= 2` (non-negotiable), fresh seeds, settled = elite-mean of the
final generation, arms = search / refit-off control / frozen / **fine-tuned
NEW_EMBODIMENT**. Pre-registered before it runs.

### Phase F — ACC and ACE, which is the destination and not an afterthought

- **ACE as a layer ranker.** GR00T offers the first genuine menu: per-embodiment
  decoder/encoder layers against a shared trunk. The screen design is already written in
  `prereg_records/PREREG_OPENPI_ACE_SCREEN.md` and ports directly — the same primary
  (does `ACE_hat` separate sites beyond draw noise?), the same per-layer rho scaling.
- **The ACC law.** GR00T's torch path is white-box and differentiable, so eq. 4 is
  runnable here in a way it never was through a websocket-served JAX model. **Gate it on
  `PLAN_CROSS_EMBODIMENT.md` Phase 4.1 first:** sign agreement of the chunk-prediction
  error against the oracle `C` from Phase D, **threshold 65%**, the test `s^T Q s` failed
  at exactly 50%. GR00T's 40-step action chunk is a native carrier for that signal.

---

## 6. Risks

| risk | mitigation |
|---|---|
| No X2 manipulation harness | Phase A gate at two weeks; prefer extending `box_pickup` over building scenes |
| The G1->X2 gap is too large to be an "embodiment gap" at all | Phase B measures it; if frozen is floor-dead, fall back to a task-space-only comparison |
| `C` is strongly configuration-dependent | measured in Phase D; if so, adapt a state-conditioned head |
| Turing throughput | measured: 0.377 s/step, ~14 s per 300-step episode, ~1.6 h per 400-episode screen — workable, but a fine-tuned baseline arm wants 40 GB+ and may need rented hardware |
| Reviewer: "this is just fine-tuning" | §4 — the fine-tuned arm is run, not asserted |
| Result is positive but only in sim | stated as a sim result; hardware is a separate claim |

---

## 7. What changes relative to the openpi plan

- **openpi/LIBERO is a testbed, not the road.** It has a computable ceiling and runs
  natively here, which makes it the right place to validate search machinery. It is not a
  path to X2: no humanoid embodiment, no shared task space, no per-embodiment head.
- **The GR00T rejection in `PHASE0_OPENPI.md` is void.** It rested on a 16 GB VRAM floor
  (we have 48) and an Isaac-leaning eval (we *want* Isaac — the X2 assets are there).
- **The vehicle is chosen on structure, not availability.** GR00T is the only surveyed
  model with per-embodiment projection layers over a shared trunk, a shared cross-embodiment
  task space, humanoid source embodiments, and open weights. WholeBodyVLA has no code
  release and no timeline; GO-1 is AgiBot's own and closest to X2's data distribution, but
  its licence is CC BY-NC-SA on the repo and unverified for the weights — worth revisiting
  only if Phase A or B fails on GR00T.
