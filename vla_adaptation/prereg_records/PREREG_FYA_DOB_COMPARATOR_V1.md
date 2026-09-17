# PREREG: matched calibrated disturbance-observer (DOB) comparator, v1 (stronger-fault cohort)

Registered 2026-09-16 (committed ≈ 20:30 machine time), before any DOB episode was run. Implements the closed-loop
EXPERIMENT_PLAN.md §5, route A (eighty-episode extension on the inspected stronger-fault evaluation cohort) with the
optional development gain selection. Route B (fresh keys) is not feasible: the Spatial used-state ledger has one free
state left. Root `results/fya_dob_comparator_v1/`.

**Comparator.** `adaptive_law.py --arms adaptive --law legacy --baseline dob --gamma α`: the same calibrated residual
(historical W, full M), the same mask {3,4,5}, cap .05 (projection), all-channel residual, activation at policy step
30, seven-sample command history; the update is exponential smoothing `f̂ ← clip((1−α) f̂ + α (M⁻¹ r − bias))`,
i.e. NT's target without its deadzone gate and its normalisation/attenuation. NT stays the previously fixed
controller (γ .08, δ .008, ρ .15). This is a gain-matched or selected-gain comparison on an inspected cohort, not a
test against an optimally tuned DOB and not an implementation of any named published method.

**Development gain selection (states 15, 17, seed 84001, the stronger-fault development keys; GPU 0).** DOB with
α ∈ {.02, .08, .20} × {healthy, F4 faulted}: 120 episodes. Selection rule, frozen: the α with the largest mean of
(healthy successes + faulted successes)/40; tie-breaks: fewer healthy losses, then the lower α. All candidates are
reported; the choice is written to `development/SELECTED_GAIN.txt` before the evaluation starts.

**Evaluation (route A).** DOB with the selected α, healthy and F4-faulted, on the original 40-key evaluation manifest
(`results/fya_stronger_fault_v1/eval_manifest.json`, seed 85001, complete ordered manifest per arm; arms, not key
subsets, are parallelised across the cards): 80 episodes. Archived Off and immediate-NT arms are reused after the
full-key, seed and configuration checks of `fya_comparator_score.py`.

**Primary:** faulted NT − faulted DOB success, paired on the 40 keys: wins/losses, every discordant key, per-task
differences, leave-one-task-out, task-clustered bootstrap 95 % interval (10,000 draws, seed 20260916), exact McNemar
descriptive. A positive lower limit supports a setting-specific NT advantage; a negative upper limit a DOB advantage;
otherwise unresolved (nonsignificance is not equivalence). Secondary, exploratory: DOB − off (faulted), healthy DOB −
healthy off with every individual loss listed, NT − off. Expectations: none registered on the sign; a tie is
informative (the calibrated interface, not the NT law, would carry the benefit). Outcome appended below.

## Amendment 1 (20:45, before any DOB evaluation episode; development selection was running)

Because coupling across server processes is not guaranteed (see `PREREG_FYA_HEALTHY_COUPLED_V1.md` §6), the
evaluation no longer relies on reusing the archived Off and NT arms. **Primary comparison: in-process trios.** On the
40-key evaluation manifest, one server process on GPU 0 runs faulted off, faulted NT, faulted DOB (selected α) back
to back; one server process on GPU 1 runs healthy off, healthy NT, healthy DOB back to back (240 episodes,
`evaluation_v2/`). Primary endpoint unchanged (faulted NT − faulted DOB, paired). The archived stronger-fault arms
and any DOB arm from the first evaluation route are secondary reproducibility checks, reported with their coupling
status; they do not enter the primary. Chain `scripts/re4/fya_dob_eval2_chain.sh` (starts after the development
selection has been written; the first evaluation route is cancelled before it starts).

## Outcome (development 20:29–21:40 on GPU 0; in-process trios 21:53–23:41, faulted on GPU 0, healthy on GPU 1; `analysis/score_v2.json`, `analysis/coupling_*.json`)

**Development selection (20 keys, seed 84001).** DOB α = .02: healthy 20/20, faulted 6/20; α = .08: 20/20, 8/20;
α = .20: 19/20, 9/20. Means .650, .700, .700; the tie is broken by fewer healthy losses → **α = .08 selected**
(`development/SELECTED_GAIN.txt`), which is also NT's γ, so the evaluation is gain-matched as well as selected.

**Evaluation, in-process trios on the 40 evaluation keys (seed 85001).** Coupling within each process: faulted
off/NT 36/40 and off/DOB 36/40 exact prefixes (the same four keys diverge, at steps 0–25); healthy off/NT 37/40,
off/DOB 37/40. The route-A reuse of archived arms was cancelled before it started, as amended.

| Arm | faulted | healthy |
|---|---:|---:|
| off | 6/40 | 39/40 |
| NT (γ .08) | 14/40 | 40/40 |
| DOB (α .08) | 14/40 | 40/40 |

**Primary, faulted NT − faulted DOB: 1 win / 1 loss, net 0, 0 points, task-clustered 95 % interval [−7.5, +7.5],
exact McNemar p = 1.0 → unresolved / tie** (win key (5,18), loss key (2,19)). Secondary: DOB − off 8 wins / 0
losses (+20 points [10, 32.5], p = .008); NT − off 9 wins / 1 loss (+20 points [10, 30], p = .021), reproducing
the archived 6/40 → 14/40 exactly in count on a fresh server process; healthy DOB − off and NT − off each +1
(the fresh healthy off arm lost one key, task 9 state 19, that both adapters completed), no healthy losses for
either adapter.

**Reading.** With the same calibrated residual, sensitivity, mask, cap and timing, plain exponential smoothing
recovers the same keys as the gated, attenuated NT law; NT's gate and normaliser add nothing measurable at this
size on this fault. The task benefit belongs to the calibrated execution interface, not to the NT update rule.
Nonsignificance is not equivalence: the interval spans ±3 keys. Both cards released 23:42.
