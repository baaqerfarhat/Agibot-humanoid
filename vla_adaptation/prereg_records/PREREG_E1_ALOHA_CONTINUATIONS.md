# Preregistration: E1 physical continuations on ALOHA (2026-09-16, before the runs)

The second interface of E1 (`iclr2027/EXPERIMENT_PLAN.md`): absolute joint targets on position
servos at 50 Hz, whose joints certify as first-order servos where the Panda under OSC_POSE does not
contract (record 48: ALOHA refuted the first-order route at 50 Hz on most joints by its increment
fit, but its command–position gap is servo-like; this study measures the physical memory directly).
Tool `openpi/re4_theory/e1_aloha_continuations.py` (smoke-tested: duplicate gap 0, replay index
asserted, exact cancellation identical); scorer `e1_aloha_score.py`.

**Sources and partition (declared).** The eight recorded healthy command streams of
`results/aloha/healthy_log.json` (225–300 steps), replayed from declared scene seeds 300–307
(the log records none; seeds 200–239 and 100–129 are the used ones): **fit = episodes 0–2, qualification
= 3–4, locked test = 5–7**. Fresh sources (twenty, per the plan) are collected when a card is free
and added under a separate note; this pass is the registered first cut at n = 3 test sources.
Checkpoints at steps 60 and 120 (fixed rule; every source carries both with H = 50, i.e. 1 s).
Fault +0.02 rad on the six left-arm joints (the plan's value); correction mask joints 0–5; the
deployed ALOHA constants (γ 0.08, δ 0.002, ρ 0.4 over all fourteen joints, κ 0.08); the deployed
position FIR (K = 6, fit on the same healthy log, R² ≥ 0.996 on the left arm) and the shipped M ≈ I.
Hold branch: the median over qualification checkpoints of the innovation branch's last-20-step
estimate, frozen before the locked pass.

**Predictions (locked sources).**
1. Snapshot fidelity: duplicate joint gap ≤ 1e-9 rad on every checkpoint.
2. **Saturation, not growth (the contrast with the Panda):** the faulted/off joint deviation at
   the endpoint is within 1.2× its value at step 20 (Panda v2: 1.8×) — a position servo reaches
   the offset and stays; exact same-mask cancellation is identical to healthy.
3. Adaptive branches from zero below faulted/off with source-level intervals excluding zero, and
   their endpoint deviation tracks the *current* remaining disturbance: the endpoint joint gap
   divided by the remaining disturbance norm is within 30 % of the faulted branch's ratio (no
   persistent memory of the transient), unlike the Panda where it accumulates.
4. A geometric-decay response model (λ fitted on the fit sources) beats pure accumulation on the
   paired integrated-error contrast with the source-level interval excluding zero, with λ in
   0.55–0.90 (certified poles 0.72–0.77); the fitted memory model's advantage over the geometric
   one is not required. (Scored with `e1_memory_model.py` adapted to joint deviation, registered
   in a note before the locked scoring.)
**Refutations** are the failures of 2, 3 or 4 as written; each is reported as primary.

---

## Outcome, locked test (2026-09-16, 01:07; `results/iclr_unified_v1/E1_aloha/`; three sources, six checkpoints, seven branches, 50 steps)

Fit/qualification pass: five sources, ten checkpoints, duplicate gap 0, replay index asserted; hold
estimate on the left arm 0.0190–0.0206 rad per joint (96–103 % of the 0.02 fault).

| branch | integrated joint deviation (rad·step, median) | endpoint (rad) | remaining disturbance (rad, norm over 6 joints) |
|---|---|---|---|
| duplicate healthy / exact cancellation | 0 / 0 | 0 / 0 | 0 / 0 |
| faulted / off | 2.36 | **0.0492** (= the fault's norm 0.02·√6) | 0.049 |
| legacy from zero | 0.79 | 0.0021 | 0.0018 |
| innovation from zero | 0.75 | 0.0015 | 0.0013 |
| hold (0.019–0.021 supplied) | 0.10 | 0.0020 | 0.0020 |

1. **Fidelity: holds** (duplicate gap 0 on 6/6; replay index verified).
2. **Saturation, not growth: holds.** The faulted deviation at the endpoint is **1.01×** its value at
   step 20 (registered ≤ 1.2; the Panda v2 grew 1.8×): the servo reaches the commanded offset within
   20 steps and stays there. Exact cancellation is identical to healthy.
3. **No memory: holds.** Both laws are far below faulted (intervals exclude zero by an order of
   magnitude) and cancel 97 % of the disturbance within 50 steps (the Panda cancelled 55 %). The
   endpoint gap divided by the remaining disturbance is 1.00 for faulted/off and 1.19–1.20 for the
   adaptive branches (registered: within 30 % of the faulted ratio): the deviation tracks the
   *current* remaining disturbance and retains nothing of the transient.
4. **Geometric decay wins: holds.** The geometric response model (λ fitted **0.66**, inside the
   registered 0.55–0.90; certified poles 0.72–0.77) beats pure accumulation on the paired
   integrated-error contrast (memory − neutral −1.03 rad·step [−1.06, −1.00]; memory − geometric
   −0.11 [−0.12, −0.11]; hence geometric − neutral ≈ −0.92 with the interval far from zero). Not
   registered, reported: the qualification-selected envelope covers 60 % of locked continuations
   (two qualification sources), bound/error ratio 1.16.

**Reading.** The two interfaces behave as the theory's memory distinction says they should: the
Panda under OSC_POSE integrates an un-cancelled offset into pose error that persists after the
estimator converges (λ̂ = 1.00, memory model no better than accumulation, 45 % of the fault still
un-cancelled at 50 steps), and ALOHA's position servos track the offset and forget it as soon as
it is cancelled (λ = 0.66, 97 % cancelled, deviation ∝ current remaining disturbance). n = 3 test
sources; the twenty fresh sources of the plan extend this when a card is free.

---

## Extension to the plan's source count (2026-09-16, 07:00; declared before collection)

Twenty fresh healthy source episodes are collected on ALOHA (`aloha_adapt.py log`, π0, scene seeds
**400–419**, unused; recorded in the log as `init`), on the ALOHA policy server run on GPU 0 shared
with another user's training. Partition by episode index: **fit 0–5, qualification 6–9, locked test
10–19** (the plan's 6 / 4 / 10). Checkpoints 60 and 120, H = 50, fault, mask, constants, hold rule and
the four predictions exactly as registered above; the eight-episode pass stands as the first cut and
is not pooled with this one. Outputs `results/iclr_unified_v1/E1_aloha_v2/`.

## Outcome, twenty-source extension, locked test (2026-09-16, 07:13; `results/iclr_unified_v1/E1_aloha_v2/`; ten test sources, twenty checkpoints)

Collection: 20 healthy episodes at seeds 400–419 (7 successes; the replay needs commands, not
success), FIR R² ≥ 0.99 on the arm joints. Hold estimate 0.018–0.020 rad per left-arm joint.

| branch | integrated joint deviation (rad·step, median) | endpoint (rad) [IQR] | remaining (rad) |
|---|---|---|---|
| duplicate / exact | 0 / 0 | 0 / 0 | 0 / 0 |
| faulted / off | 2.36 | **0.0492** [0.0492, 0.0492] | 0.049 |
| legacy from zero | 0.80 | 0.0031 [0.0030, 0.0041] | 0.0028 |
| innovation from zero | 0.77 | 0.0020 [0.0018, 0.0021] | 0.0014 |
| hold | 0.11 | 0.0024 | 0.0024 |

1. **Fidelity: holds** (duplicate gap 0 on 20/20; replay index asserted).
2. **Saturation, not growth: holds** (endpoint/step-20 ratio **1.012**; exact cancellation identical).
3. **No memory: holds for the legacy law, fails its letter for the innovation law.** Both laws are
   below faulted with intervals excluding zero (−1.55 and −1.59 rad·step) and cancel 94–97 % of
   the disturbance within 50 steps. The endpoint-gap / remaining-disturbance ratio is 1.00
   (faulted), **1.24** (legacy, within the registered 30 %) and **1.43** (innovation, outside it).
   The innovation branch's remaining disturbance at the endpoint is 0.0014 rad, so its ratio is a
   quotient of two small numbers dominated by one servo time constant of lag (its absolute
   endpoint gap is 0.0020 rad, 4 % of the fault); reported as the registered failure it is, with
   that reading offered. The first-cut ratio (1.20 on three sources) did not show it.
4. **Geometric decay wins: holds** (λ = **0.663**, inside 0.55–0.90; memory − neutral −0.98
   [−1.09, −0.90], memory − geometric −0.10 [−0.11, −0.09]; geometric − neutral ≈ −0.88). The
   qualification envelope (four sources) covers 72 % of locked continuations (descriptive; not a
   registered clause here).

**Reading, unchanged from the first cut and now on ten sources:** ALOHA's servos reach the offset
and forget it as it is cancelled; the Panda accumulates it. Three predictions hold in full and
the fourth's second clause fails for one law by a ratio that has no physical content at a
0.0014-rad remaining disturbance.
