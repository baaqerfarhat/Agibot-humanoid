# Executing `iclr2027/EXPERIMENT_PLAN.md` — status, mapping, budgets (2026-09-14, evening)

Deadlines: abstract 2026-09-18, full paper 2026-09-25 (AoE). One GPU, one job at a time.

## What the plan asks for and where it stands

| package | plan's requirement | state on this side |
|---|---|---|
| E0 evidence interface | recounts, labels, law-faithful reconstruction, claim table | done by the review (`iclr2027/evidence/`); record 52 fixed the recorder and the Part 6 R² |
| shared protocol 1 (paired key) | robot, suite, task, init, sim seed, sampler seed + schedule, fault, horizon, arm | `re4_record.py` now writes the sampler schedule and manifest hash; `episodes.csv` carries `schedule:<seed>` per episode |
| shared protocol 3 (unused states, no wrapping) | enumerate used `(suite, task, init)` and pick unused | `openpi/re4_theory/scenario_manifest.py`; manifests in `results/iclr_unified_v1/manifests/` — spatial inits 49 and 44, libero_10 41 and 40 are unused on every task; 60 keys per suite × 3 sampler seeds |
| shared protocol 4 (real sampler replication) | explicit seed, deterministic per-call schedule shared by arms, reset per episode | `ace_server.py`: `sampler_seed` + `episode` in the control file → `fold_in(fold_in(key(seed), episode), call)`; runner `--sampler-seed`, `--manifest`; old `pin_rng` kept for replay |
| single-arm runner (840 not 1,200 rollouts) | run the frozen arm once per condition | runner `--arms {both,frozen,adaptive}` |
| E2 constrained predictor C | FIR with the r_y tap sum constrained to a **separately probed** local response | `--dc-constrain corrected` exists (constraint from the historical M probe); the plan's probe qualification (150 branches, settling and linearity checks) is not built |
| E1 physical continuations | 7-branch, 100-step replays from full-state snapshots with physical traces and adapter-state branching | `paired_rollout.py` does state/command perturbations and short continuations with q/v/ee-position only; needs orientation, object pose, contact flags, controller target, adapter branches, and a scorer |
| E3 / E4 | optional | not started |

## Running now (a pilot of E2's core hypothesis, registered as `PREREG_DC_CONSTRAINED_PLANT.md`)

Legacy reference with only the plant changed (`--dc-constrain corrected`, constraint from the
historical probe), spatial n = 20, libero_10 n = 40, healthy spatial n = 20; unpinned, inits
from 45. It answers the plan's first E2 question cheaply (does enforcing low-frequency
consistency move the r_y estimate?) before the 840-rollout confirmation is spent. It does not
satisfy E2's protocol (no probe qualification, no fresh partition, no sampler schedule, legacy
law) and will be reported as the pilot it is. Decision rule for E2 from it, stated now:
r_y ≥ 70 % → build E2's probe qualification and run the core; r_y < 60 % → E2's constraint
cannot be qualified on this mechanism, report the pilot and do not launch the core.

## Budgets on this machine (measured episode times: spatial ≈ 1.3 min, libero_10 ≈ 2.2 min)

| package | rollouts | GPU time | earliest start |
|---|---|---|---|
| E2 preparation: 40 healthy spatial episodes + 150 probe branches (sim only) | 40 | ≈ 1 h + minutes | after the DC pilot, tooling for probe qualification needed |
| E2 core: 120 keys × 7 arms with `--arms` | 840 | ≈ 24 h | 15 Sep if the pilot passes |
| E1: 20 spatial + 20 ALOHA sources, 560 branches (no policy calls in the branches) | 40 + sim | ≈ 1 h + sim time | after the E1 tooling (½–1 day of work) |
| E2 optional r_y-only condition | 480 | ≈ 14 h | only if declared before the core |

E2 core and E1 together fit before the full-paper deadline if E2 starts by 16 Sep. They do not
both fit before the abstract deadline; the plan's own order puts the abstract on completed
evidence.

## Next actions, in order
1. Score the DC pilot tonight (prereg outcome, record §54); apply the E2 decision rule above.
2. If E2 goes ahead: healthy fit/qualification partition (40 spatial episodes, fresh inits from
   the manifest builder's ledger), probe qualification tool (extend `paired_rollout.py`'s command
   perturbations with the settling/linearity checks), then the constrained predictor from the
   qualified probe, then the 840-rollout core under one prereg with the primary contrasts fixed
   (observation-bias reduction ≥ 25 % with a clustered interval; libero_10 C − U with a
   clustered interval; healthy change > −5 points).
3. E1 tooling in parallel on CPU (the plan's first new simulator work), then its locked runs.
4. Abstract on 18 Sep from completed evidence; results land in the paper by 25 Sep.
