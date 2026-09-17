# PREREG: healthy GR00T N1.7 / Panda crossed replication, v1: technical pilot (part A)

Registered 2026-09-17 (committed before any GR00T episode of this campaign). Implements
`papers/frozen_yet_adaptive_recovery_study/EXPERIMENT_PLAN.md` §§2–3. Root `results/fya_groot_healthy_crossed_v1/`.
Part B (the confirmatory forty-key allocation) is registered separately below only if the pilot passes.

## A. Server change (implemented before the pilot)

`openpi/groot_server.py` now applies control requests: `{sampler_seed, episode, reset}` seeds the global torch/CUDA
RNGs before every inference with sha256("groot|seed|episode|call") (GR00T's only inference-time randomness is the
flow-matching initial noise, `torch.randn` in `gr00t_n1d7.py`), resets the call counter at each control write,
calls the policy's `reset()` (a no-op: the policy holds no episode state; the eight-step replan buffer lives in the
client), and acknowledges with the applied seed/episode, server pid, server-source and checkpoint-config hashes,
torch version and cuDNN determinism flags. `--calllog` writes one line per control application and per inference
(episode, call, derived seed, first action). The runner (`adaptive_law.py`) now stores each episode's control
acknowledgement in its result JSON. A server self-test (`--selftest-seed`) must show identical actions for identical
observations and seed, and different actions for the next call index and for another seed, before the pilot.

## B. Pilot design (inspected keys; technical only, never confirmatory evidence)

Keys: tasks 0, 3, 6, 9 at Spatial state 13 (Stage 2 keys, inspected with pi0.5), sampler seed 88001
(`pilot/manifest.json`). Server: GR00T N1.7 LIBERO-Spatial checkpoint (`Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO/
libero_spatial`, config sha recorded in the ack), GPU 1, port 8003, `GROOT_HF_LOCAL_FIRST=1`, cuDNN deterministic.
Runner flags as the historical GR00T cells plus the fixed adapter: `--suite libero_spatial --replan-steps 8
--scenario-reset --log results/phase05/error_signal_so3.json --openloop results/phase05/openloop_so3.json --gamma 0.08
--dead 0.008 --norm-r 0.15 --clip 0.05 --corr-dims 3,4,5 --norm-channels all --law legacy --baseline none
--fault-vec 0,0,0,0,0,0 --onset 40 --adapt-from 30`. Arms on one server process, in order: healthy_off, healthy_nt,
healthy_off_dup (12 episodes). Replan phase: with eight-step replans at policy steps 0, 8, 16, 24, 32 the activation
at step 30 falls inside the chunk that began at 24; logged and reported.

Then the diagonal replay only (ref, J00 ≡ ref, J11, fresh routes) via `fya_crossed_extract.py --arms
healthy_off,healthy_off,healthy_nt,healthy_off_dup --config-arm healthy_nt` and `fya_crossed_replay.py --matrix healthy
--cells diagonal` (20 continuations).

Pass criteria (all required): (1) self-test passed and every episode's ack shows the requested seed/episode applied;
(2) off/NT raw actions, positions and joints agree exactly over the thirty-step prefix on all four keys and off/dup
agree over the whole episode (first divergence logged otherwise); (3) J11 reproduces the archived NT path (positions,
joints, orientation, corrections, estimates) within 1e-8 and ref reproduces healthy_off; (4) fresh and restored
routes agree within 1e-8; (5) every source has ≥ 80 policy commands; runtimes recorded. A failure stops the
confirmatory campaign; fixes are made on pilot keys only and re-registered.

## C. Confirmatory design (frozen now; run only if the pilot passes; registered as part B when launched)

Forty keys, tasks 0–9 × Spatial states **23, 24, 29, 31** (already inspected with pi0.5 sources; the ledger has one
free Spatial state, so this is a new-backbone replication on inspected physical states, stated as such), sampler seed
**89001**. Same server process, arms healthy_off, healthy_nt, healthy_off_dup (120 episodes). Eligibility: ≥ 80
commands, exact off/NT prefix, diagonal and fresh-route fidelity at 1e-8; duplicate agreement reported and used as a
registered sensitivity. Primary R1 = J10 − J11 (equal-task, task-clustered bootstrap 10,000 draws, seed 20260920,
margin 1e-5 m² s); registered secondary S = D0 − R1 (J00 = 0 verified); D0, T, I, angular effects, endpoints, all
task outcomes with every NT regression. Decisions exactly as the plan's §3.3. Reporting-completeness threshold:
≥ 30 fidelity-valid keys covering ≥ 8 tasks; otherwise an incomplete replication is reported as such.

## D. Pilot outcome (13:11–13:28) and part B registration (committed before any confirmatory episode)

**Self-test passed:** identical observation and derived seed → identical actions (max difference 0.0); next call
index and another seed → different actions. **Sources:** 12 episodes on one server process in 834 s (≈ 70 s per
episode, eight-step replans at 3.8 s per chunk); every episode's acknowledgement records the requested seed 88001,
its episode ordinal and an applied reset. **Coupling:** off/NT prefixes exact on 4/4 keys; off/duplicate exact over
whole episodes on 4/4 (GR00T's seeded sampling reproduces exactly, unlike pi0.5's ≈ 3/40 divergences). **Diagonal
replay:** all four keys valid, every gap 0.0 (prefix, ref, J00, J11 with corrections and estimates, fresh routes),
64 s. All sources ≥ 80 commands (healthy 4/4 in every arm). All five pass criteria hold; the pilot keys (state 13)
are excluded from part B. Replan phase: with `--replan-steps 8` the client replans at policy steps 0, 8, 16, 24, 32,
…; activation at step 30 falls at position 6 of the chunk begun at 24; the confirmatory runs log replan boundaries
(`--timing`) so this is verified per episode.

**Part B, frozen now.** Manifest `manifest.json` (sha dae3153da751159a): tasks 0–9 × states 23, 24, 29, 31, seed 89001; server
source sha c27a8d0d199f9dba, checkpoint config sha in each ack. Arms healthy_off, healthy_nt, healthy_off_dup on one process
(120 episodes, ≈ 2.5 h), then extraction, the healthy matrix (`--matrix healthy`, tolerances 1e-8) and scoring
with **R1 primary, margin 1e-5 m² s, task-clustered bootstrap 10,000 draws, seed 20260920**, secondary S (J00 = 0
verified), D0, T, I, angular effects, endpoints, duplicate-exact sensitivity, all task outcomes. Decisions as in
§C. Chain `scripts/re4/fya_groot_confirm_chain.sh`. Outcome appended as §E.

**Part B addendum (13:33, registered before it ran): cross-device reproducibility control.** A second GR00T server
process on GPU 0 (port 8004, same checkpoint and source, rendering on GPU 1) re-runs healthy_off on the same 40 keys
and seed 89001 (`xdevice_control/`). It is compared whole-episode to the GPU 1 healthy_off arm with
`fya_source_coupling_report.py`. Registered expectation: exact agreement on every key (as the pilot's same-process
duplicate). Outcome: exact → future GR00T arms may run on different cards; otherwise arms stay on one process. This
control adds no independent task sample and does not enter the primary analysis.

**Cross-device control outcome (13:32–14:18).** healthy_off from the GPU 0 process reproduces the GPU 1 arm exactly on
**35/40** keys (whole episodes, raw actions and lengths); five keys diverge mid-episode (first differing policy step 40,
184, 88, 80, 200 on tasks 1/31, 2/29, 2/31, 6/29, 7/31; the largest gap is a gripper sign flip at step 184 of a
220-step failed episode), with the prefix identical on all 40. Task outcomes are **38/40 in both processes**, with
the same failures at task/state 2/29 and 7/31 (corrected from the initial outcome prose using both source summaries
and per-step termination records). The registered
expectation of exact agreement fails; cross-device reproduction is close but not exact, so **GR00T arms stay on one
server process** for paired comparisons, as pi0.5's do. The control does not enter the primary analysis.

## E. Outcome, part B (sources 13:29–15:51 on one GPU 1 server process; replay and score 15:51–16:02; `sources/`, `runs/`, `analysis/`)

**Sources (120 episodes).** healthy_off 38/40 (lost 2/29, 7/31), healthy_nt 39/40 (lost 7/31; rescued 2/29, no
regression), healthy_off_dup 38/40 (same losses as off). Every acknowledgement carries seed 89001 and an applied reset;
replan boundaries logged at policy steps 0, 8, 16, 24, 32, … on every episode, none at 30: activation at step 30 sits
at position 6 of the chunk begun at 24, as registered. Coupling: off/NT prefixes exact on **38/40** (1/31 diverges at
step 24, 7/29 at 16); duplicate off exact over whole episodes on **35/40** (divergences at steps 32–184).

**Eligibility and fidelity.** 36 keys eligible (0/23 and 0/24 have 79 commands; 1/31 and 7/29 fail the prefix rule);
**all 36 reproduce exactly** (ref, J00, J11 with corrections and estimates, fresh routes; gaps 0.0). 36 keys across
all ten tasks scored; J00 = 0 on every key. Completeness threshold (≥ 30 keys, ≥ 8 tasks) met.

| Contrast | equal-task | 95 % CI | keys +/− | decision |
|---|---:|---|---|---|
| **R1 = J10 − J11 (primary)** | **−1.81e-4** | **[−4.2e-4, −3.6e-5]** | 0 / 36 | **upper endpoint below −δ: transfer supported** |
| S = D0 − R1 (registered secondary) | 1.64e-4 | [2.8e-5, 4.0e-4] | 26 / 10 | positive, interval above δ |
| D0 = J00 − J10 | −1.7e-5 | [−3.3e-5, −6.7e-6] | 0 / 36 | negative on every key |
| T = J00 − J11 | −2.0e-4 | [−4.6e-4, −4.3e-5] | 0 / 36 | negative |
| I = D1 − D0 | −1.1e-5 | [−3.0e-5, 3.4e-6] | 19 / 17 | unresolved |
| angular D0_r / R1_r / T_r (rad² s) | −2.5e-3 / −1.2e-3 / −3.7e-3 | all below zero | 0/36, 11/25, 0/36 | descriptive |

Per-task R1 from −1.7e-5 (task 1) to −1.2e-3 (task 9); leave-one-task-out −6.7e-5 (without task 9) to −2.0e-4,
every one below −δ; source-weighted −1.81e-4. Cells (medians): J10 7.8e-6 (2.5 mm endpoint), J01 5.0e-5
(9.4 mm), J11 6.2e-5 (9.8 mm).

**Reading.** With the Spatial calibration, rotation-only NT, cap .05 and activation at step 30 held fixed, the healthy
stream effect measured on pi0.5 transfers to GR00T N1.7: direct false-update deviation 2.5 mm, the stream the
policy generates under those corrections 9.4 mm from its own healthy path, R1 negative on every key and S clearing
its margin prospectively. Four cohorts, two suites, two backbones, four seeds now agree in sign and structure. Task
outcomes are essentially unchanged (off 38/40, NT 39/40 with one rescue and no regression), so this stays a
physical-deviation statement on inspected physical states with a new backbone and new sampling. Campaign closed
16:05; both cards released.
