# PREREG: independent coupled-source replication of the healthy crossed matrix, v1 (Panda / LIBERO-Spatial, pi0.5)

Registered 2026-09-16 (committed ≈ 19:25 machine time), before any episode of this campaign was run. Implements
`papers/frozen_yet_adaptive_closed_loop/EXPERIMENT_PLAN.md` §5 Priority A, first route: the manuscript claims that
substituting the stream generated under adaptation can substantially increase deviation on healthy executions;
that rests on 16 archived keys of one sampler seed (`PREREG_FYA_CROSSED_REPLAY_V1.md` §10, R1 = −3.6e-4
[−5.6e-4, −1.7e-4] m² s). This campaign replicates it on fresh, coupled sources with **R1 as the registered primary
endpoint**; I is a registered secondary and the archived unresolved-I outcome stands. Root
`results/fya_healthy_coupled_v1/`.

## 1. Sources (fresh keys, both cards)

40 keys: tasks 0–9 × free Spatial states **23, 24, 29, 31**, sampler seed **86001** (`manifest.json`). Three
source arms, adapter otherwise as in the Stage 2 header (NT, cap .05, mask {3,4,5}, γ .08, δ .008, ρ .15,
all-channel normalisation, historical W and M, `--scenario-reset`, no fault): **healthy_off** (frozen), **healthy_nt**
(`--adapt-from 30`), and **healthy_off_dup**, an identical repeat of healthy_off to quantify source
reproducibility. 120 policy episodes: healthy_off and healthy_off_dup on GPU 1 (port 8000), healthy_nt on GPU 0
(port 8001), servers at memory fraction .42.

Coupling: the policy server's sampler schedule is fold_in(key(seed), episode, call), so the k-th policy call of a key
draws identical random keys in every arm (identical draws indexed by replan opportunity); the control acknowledgement
with the schedule is logged per episode. Limitation, stated in advance: per-call RNG traces are not exported by the
server; coupling is established by the schedule and by the duplicate arm (expected exact equality of raw actions and
positions on every key), not by a per-call trace.

## 2. Matrix (as the archived healthy control)

Common prefix policy steps 0–29 (identical across arms by construction; verified exactly at extraction), snapshot at
env step 40, window 50 steps. M0 = healthy_off raw stream, M1 = healthy_nt raw stream, ref = healthy_off path. Cells
J00 (≡ ref, cost 0), J10 (M0 + NT causal), J01 (M1 + off), J11 (M1 + NT causal, must reproduce the archived
healthy_nt path), plus J00_fresh and J11_fresh; cell order rotated by key index. Fidelity and tolerances exactly as
in `PREREG_FYA_CROSSED_REPLAY_V1.md` §3 (1e-8 rad / m; corrections ≤ 1e-8); a failing key is excluded and listed.
Continuation semantics as registered there.

## 3. Endpoints, precision, decision

Primary **R1 = J10 − J11** (stream-change benefit with NT active; negative means the adaptation-generated stream
increases deviation from the healthy reference). Equal-task estimand (10 tasks × 4 states), task-clustered
percentile bootstrap, 10,000 draws, seed 20260916; source-weighted sensitivity; per-task effects;
leave-one-task-out. Margin δ = 1e-5 m² s (as before). Decision: R1 interval below −δ → replicated (the
manuscript's healthy-stream statement stands on fresh keys); interval within ±δ → small at this margin; interval
containing both signs beyond δ → unresolved. Secondary: I (same rule), D0 (expected < 0), T.

Precision, from the archived matrix: per-key R1 sd 4.1e-4 with 16 keys / 9 tasks gave a half-width ≈ 1.9e-4; with
40 keys / 10 tasks the expected half-width is ≈ 1.2e-4, so an effect of the archived size (−3.6e-4) would be
resolved and an effect below ≈ 1.5e-4 in magnitude would not. Maximum sample size 40 keys; no extension, no
change of law, cap, mask, window or margin after any matrix cell is read.

Expectations (refutation reported as primary): H1 R1 < −δ (replication); H2 D0 < 0 on most keys (direct cost of
false updates); H3 the duplicate arm equals healthy_off exactly on every key (if not, coupling is reported as
approximate and the key-level disagreement is listed). No task-success claim (healthy arms are expected at or
near 40/40; losses listed).

## 4. Provenance

Runner `adaptive_law.py`, driver `fya_crossed_replay.py --matrix healthy`, extractor `fya_crossed_extract.py --arms
healthy_off,healthy_off,healthy_nt,healthy_off_dup`, scorer `fya_crossed_score.py --primary R1`; chain
`scripts/re4/fya_hc_chain.sh` (waits for the delayed-NT arm to release the servers). Outcome in §5.

## 5. Execution note and amendment (20:15, before any matrix cell existed)

Sources completed (healthy_off 40/40, healthy_off_dup 40/40, healthy_nt 39/40; 120 episodes). Extraction found
**0 of 40 keys eligible**: the healthy_nt prefix (policy steps 0–29, no correction in either arm) differs from the
healthy_off prefix on every key (raw-action gaps 3e-3 to 9e-2). Duplicate check on the same card and server type:
37/40 keys identical (raw actions and lengths); max gap 1.27e-01; length mismatches [(2, 31, 98, 99), (8, 23, 96, 113)]. Reading: the sampler schedule couples draws across arms only within the same server process/device; healthy_nt
ran on the GPU 0 server and healthy_off on the GPU 1 server, and the two devices do not produce bit-identical policy
actions (the Stage 2 arms, which did couple, all ran on one server). No matrix cell was collected (the driver found
no eligible key), so nothing was inspected beyond source success counts and prefix gaps.

**Amendment:** healthy_nt is re-run on the **GPU 1 server** (same device as healthy_off; `sources/healthy_nt.json`),
the GPU 0 run is kept as `sources/healthy_nt_gpu0.*` and reported as a failed-coupling attempt (its 39/40 stands
as a healthy count). Everything else (keys, seed, adapter, matrix, endpoints, precision statement, tolerances) is
unchanged. Coupling limitation, now sharper: same server process and device are required; cross-device coupling
does not hold. Chain `scripts/re4/fya_hc2_chain.sh`.

## 6. Second execution note and amendment 2 (20:45, before any matrix cell existed)

The same-device re-run (`sources/healthy_nt.*`, GPU 1 server, 39/40, lost task 7 state 29 again) is **bit-identical
to the GPU 0 run** (same prefix gaps on every key, e.g. task 1/23 4.421e-3, task 5/31 9.039e-2) and again matches
healthy_off on **0/40** prefixes. Device is therefore not the cause. Diagnosis from saved data
(`fya_source_coupling_report.py`): Stage 2 arms coupled 20/20; the stronger-fault evaluation arms coupled 39/40
(fault off vs NT: one key, task 8 state 19, diverging at step 0; healthy off vs NT: one key, task 5 state 19, at
step 15); the healthy_off process and its duplicate agreed on 37/40; the two NT processes agreed with each other on
40/40 but not with the off process. Reading: the policy server's numerics can differ systematically between
processes (a plausible mechanism is timing-dependent GPU kernel autotuning at start-up on a shared card); within one
process they are deterministic up to rare divergences (≈ 3/40 keys). Coupling is therefore guaranteed only when
all arms of a comparison are served by **one server process**.

**Amendment 2 (registered before running):** all three source arms are re-collected on one server process on
GPU 1 in the order healthy_off, healthy_nt, healthy_off_dup (120 episodes, `sources_v3/`), everything else
unchanged; extraction, matrix and R1 scoring then run on `sources_v3/`. All earlier sources are kept and reported
as failed-coupling attempts (they remain valid healthy counts: off 40/40 ×2, NT 39/40 ×2). Chain
`scripts/re4/fya_hc3_chain.sh`. If the in-process arms still fail the prefix rule on most keys, the replication is
reported as infeasible under this protocol.

## 7. Outcome, amendment 2 (sources 21:15–22:22 on one GPU 1 server process; replay and score 22:23–22:47; `sources_v3/`, `v3/`, `runs_v3/`, `analysis_v3/`)

**Sources and coupling.** healthy_off 40/40, healthy_nt 39/40 (lost task 7 state 29, the same key as both earlier NT
attempts), healthy_off_dup 40/40. Prefix coupling off vs NT: **39/40** exact (task 7 state 29 diverges at policy step 0;
it is also the lost key). Full-episode duplicate: **36/40** exact (divergences at steps 20, 60, 30, 0 on tasks 0/31,
2/23, 7/24, 7/29). **H3 fails as registered**: same-process reproduction is not exact on every key.

**Eligibility and fidelity.** 35 keys eligible (the four task-0 keys are shorter than 80 steps, task 7/29 fails the
prefix rule); one more (task 7 state 24) fails the replay fidelity checks (ref, J00, J11, fresh route) and is
excluded: **34 keys across 9 tasks** scored.

**Primary R1 = J10 − J11: −1.71e-4 m² s, task-clustered 95 % interval [−2.5e-4, −9.8e-5], negative on 34/34
keys, source-weighted −1.71e-4, leave-one-task-out −1.5e-4 to −1.9e-4, per-task −2.4e-5 to −3.7e-4. Decision:
resolved negative (interval entirely below −δ = −1e-5). H1 holds: replicated on fresh, coupled keys.** Secondary:
D0 = −1.9e-5 [−3.1e-5, −8.2e-6], 0/34 positive (H2 holds); T = −1.9e-4 [−2.8e-4, −1.1e-4]; I = −2.5e-5
[−5.9e-5, −7.4e-7], unresolved at the margin. Cell medians: J10 5.7e-6 (2.7 mm endpoint), J01 8.8e-5 (12.4 mm),
J11 9.5e-5 (12.3 mm).

**Reading.** The effect of the archived healthy control replicates in direction and structure with about half the
magnitude (archived R1 −3.6e-4 on 16 keys; here −1.7e-4 on 34 keys): the direct cost of false updates on the
off-generated stream is small (2.7 mm) and the stream the policy generates while feeling those corrections lies
an order of magnitude further from the healthy reference (12.4 mm) on every key. Proximity to one healthy path is
not task success (healthy NT 39/40, off 40/40); the statement is about physical deviation from the reacting
policy's own healthy path. Figures `figures/crossed_*_coupled_v3.pdf`, `figures/healthy_R1_two_cohorts.pdf`.
