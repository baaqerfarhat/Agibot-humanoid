# PREREG: repair of the delayed-NT task comparison (stronger-fault campaign), v1

Registered 2026-09-16 (committed ≈ 20:30 machine time), before any repaired episode was run.

**Defect (found by the writing session's identity check).** The delayed-NT increment (`PREREG_FYA_STRONGER_FAULT_V1.md`
§7–8) split the 40 evaluation keys into two 20-key manifests run on the two cards. The runner sends its local
episode index to the sampler, so the second shard (states 21, 22) used sampler ordinals 0–19 instead of the
original 20–39: those twenty keys were not sampled like the immediate-NT and off arms, and their prefixes differ
before adaptation could matter. The increment's 14/40 is therefore descriptive, not a controlled task-delay
comparison. The original increment files and §8 are preserved unchanged.

**Route (frozen): forty-episode repair with the original complete ordered manifest**
(`results/fya_stronger_fault_v1/eval_manifest.json`, seed 85001, original ordinals 0–39), one arm on one server
(GPU 1, port 8000), no runner change. Settings as registered: F4 uniform +.10 from policy step 30 (`--onset 40`),
delayed NT enabled at policy step 40 (`--adapt-from 40`), cap .05, mask {3,4,5}, γ .08, δ .008, ρ .15,
all-channel normalisation, historical W and M, `--scenario-reset`. Output `delay_repair/eval_delay_nt_full.json`.

**Checks before any effect is read:** all 40 keys present once; sampler seed per key equals the manifest; the
pre-adaptation prefix (policy steps 0–29) and the faulted pre-activation segment (30–39) are compared step by step
against immediate NT and off (`fya_source_coupling_report.py`, exact expected on the prefix except keys already known
to be non-reproducible at the ~3/40 level; every mismatch listed). No key is removed because of its outcome.

**Analysis:** repaired delayed NT vs immediate NT (primary; paired, exact McNemar descriptive, task-clustered
bootstrap 10,000 draws seed 20260916, per-task and leave-one-task-out) and vs off (secondary), all 40 keys.
Registered expectation D1 (delayed not better than immediate: net ≤ 0). Equal totals or an interval containing zero
do not establish equivalence. Outcome appended below.

## Outcome (run 20:38–21:14 on the GPU 1 server; `delay_repair/eval_delay_nt_full.json`, `coupling_report.json`, `score.json`)

**Coupling verified.** All 40 keys present once with seed 85001; the pre-adaptation prefix matches immediate NT on
**40/40** keys and off on 39/40 (the known key task 8 state 19, which already differed between the archived off and
NT arms); the faulted pre-activation segment (steps 30–39) matches off on 39/40. First update at policy step 40 and
first correction at step 41 on every key, as specified.

**Result.** Repaired delayed NT **13/40**; immediate NT 14/40; off 6/40.
- Delayed − immediate (primary): 2 wins / 3 losses, net −1, −2.5 points, task-clustered 95 % interval
  [−10, +5] points, exact McNemar p = 1.0.
- Delayed − off (secondary): net +7, +17.5 points [7.5, 30].

**D1 holds** (delayed not better than immediate). At this n a ten-step later start is not measurably worse either;
the interval spans −10 to +5 points and does not establish equivalence or tolerance to delay. The earlier
increment's 14/40 (second shard with changed sampler ordinals) is superseded for controlled comparisons and kept as
recorded in `PREREG_FYA_STRONGER_FAULT_V1.md` §8.

## Outcome (run 20:38–21:14 on one GPU 1 server process; `delay_repair/`, `coupling_report.json`, `score.json`)

**Coupling checks passed.** All 40 keys present once with seed 85001; the pre-adaptation prefix matches immediate NT
exactly on **40/40** keys and off on 39/40 (task 8 state 19, the key already known to differ from step 0 in the
archived off arm); the faulted pre-activation segment (policy steps 30–39) matches off on 39/40 (same key). First
estimator update at policy step 40 and first nonzero correction at step 41 on every key, as registered.

| Arm (40 keys) | success |
|---|---:|
| faulted off (archived) | 6/40 |
| immediate NT (archived) | 14/40 |
| **delayed NT, repaired** | **13/40** |

Primary, delayed − immediate NT: **2 wins / 3 losses, net −1, −2.5 points, task-clustered 95 % interval [−10, +5],
exact McNemar p = 1.0**. Win keys (2,19), (9,21); loss keys (0,18), (2,18), (5,18). Leave-one-task-out means from
−5.6 to 0.0 points. Secondary, delayed − off: 7 wins / 0 losses, +17.5 points [7.5, 30], p = .016.

**D1 holds** (delayed not better than immediate). Reading: with a half-second later start of correction the task
outcome is indistinguishable from immediate correction at this size (five discordant keys), while both beat off.
Equal totals do not establish equivalence; the physical delay cost (fixed-command study, 18/19 sources) and task
success remain different estimands. This repaired increment replaces §8 of the stronger-fault registration for any
controlled delay statement; the uncoupled 14/40 there stays as the record of the defect.
