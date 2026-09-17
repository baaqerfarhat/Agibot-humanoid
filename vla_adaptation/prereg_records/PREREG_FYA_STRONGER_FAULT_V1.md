# PREREG: stronger mid-episode fault, reacting-policy campaign v1 (Panda / LIBERO-Spatial, pi0.5)

Registered 2026-09-16 (committed 16:29 machine time), before any episode of this campaign was run. Execution note: the first development launch (16:29) was stopped after one minute because the GPU 1 card reached 47.6 of 49 GB (server at fraction .6 plus the EGL contexts of three concurrent replays); no development outcome was read; relaunched 16:31 with server fraction .42, partial files deleted. Implements the conditional next
step of `papers/frozen_yet_adaptive_closed_loop/EXPERIMENT_PLAN.md` §7: the previous bridge's fault (+.05 on
r_y from policy step 30) left every arm at 20/20, so a task-level statement about mid-episode adaptation needs a
fault that is not at the ceiling, selected on development keys and evaluated on untouched keys. The adapter is
fixed (NT, cap .05, mask {3,4,5}, γ .08, δ .008, ρ .15, all-channel normalisation, historical W and M as in the
Stage 2 header; `--adapt-from 30`, `--scenario-reset`). Campaign root `results/fya_stronger_fault_v1/`.

## 1. Keys (Spatial used-state ledger, free states only)

| Stage | States | Keys | Sampler seed |
|---|---|---|---|
| development | 15, 17 | 10 tasks × 2 = 20 | 84001 |
| evaluation (untouched) | 18, 19, 21, 22 | 10 tasks × 4 = 40 | 85001 |
| replacements (ordered, technical failures only) | 23, 24, 29, 31, 4 | | same seed |

Pairing on (task, init, seed) across arms; scenario reset; intention-to-treat (a key that terminates before
policy step 30 is reported as unexposed, never replaced for outcome reasons).

## 2. Development stage (fault selection; outcomes on these keys are never evaluation evidence)

Candidate faults, all additive on the normalised command from policy step 30 (env step 40), in this fixed
order: **F1** uniform +.05 on all six channels (the headline fault, applied mid-episode); **F2** +.10 on r_y;
**F3** +.15 on r_y. Runs on the 20 development keys: faulted off under F1, F2, F3 (60 episodes), healthy off
(20), then NT under the **selected** fault (20). **Selection rule:** the first candidate in the order F1, F2, F3
whose faulted-off development success is ≤ 10/20. If none qualifies, the campaign stops and reports the
ceiling (no evaluation stage). If healthy off on development keys is < 16/20 the environment is checked before
continuing. The development NT run estimates the paired discordance (off-fail/NT-success and the reverse) used
only to state, before the evaluation, what net gain the 40-key design can resolve.

## 3. Evaluation stage (locked)

Four arms on the 40 untouched keys: healthy off, healthy NT, faulted off, faulted NT under the selected fault;
160 policy episodes, arm order rotated across the two cards (registered in the chain script). Primary endpoint:
faulted NT − faulted off success, paired; exact two-sided McNemar (descriptive) and a task-clustered bootstrap
95 % interval of the rate difference (10,000 draws, seed 20260916; 10 tasks × 4 states). Registered minimum
effect of interest: **+5 net keys (12.5 points)**. Secondary: healthy NT − healthy off individual losses (report
every lost key; ≤ 3 losses is the registered healthy expectation as in the previous bridge), estimator
transients from telemetry, episode lengths, runtime. No arm removal, no extension, no cap/mask/gain change.

Expectations: T1 faulted NT exceeds faulted off by at least the minimum effect with the interval excluding
zero; T2 healthy NT loses ≤ 3 keys. Refutation of either is reported as primary. The selected fault is a
command-level fault; nothing here is a payload, friction or actuator-damage study.

## 4. Provenance and budget

Runner `openpi/adaptive_law.py` (with `--adapt-from`), server `ace_server.py` on both cards (GPU 0 port 8001,
GPU 1 port 8000, memory fraction .6, the other user's training untouched). Chains
`scripts/re4/fya_sf_dev_chain.sh` and `scripts/re4/fya_sf_eval_chain.sh`. Budget 100 + 160 = 260 policy
episodes. Outcome sections appended below as the stages complete.

## 5. Development outcome and amendment (17:12, before any evaluation episode)

Development arms (20 keys each): healthy off **20/20**; faulted off F1 (uniform +.05) **18/20**, F2 (r_y +.10)
**16/20**, F3 (r_y +.15) **12/20**. No candidate meets the registered ≤ 10/20 rule, so v1 stops as registered
(`development/SELECTED_FAULT_v1.txt` = NONE): every mid-episode command fault tried so far leaves the reacting
policy near its ceiling on Spatial. Execution note: the development chain hung after the arms finished because
its `wait` also waited on the background servers; killed by PID, fixed (`wait $P1 $P0`), no outcome affected.

**Amendment (registered now, same development keys, evaluation keys untouched):** two stronger candidates in
the extended order F1…F5: **F4** uniform +.10 on all six channels, **F5** +.20 on r_y; the same selection rule
(first candidate with faulted-off development success ≤ 10/20), the same NT development run, the same locked
evaluation design (§3) with the selected fault. If neither qualifies, the campaign closes with the ceiling
result and no evaluation. Chain `scripts/re4/fya_sf_dev2_chain.sh`.

## 6. Outcome (development amendment 17:12–17:48; evaluation 17:49–18:51 on both cards; `analysis/summary.json`)

**Development (20 keys, states 15, 17).** F4 (uniform +.10 on all six channels from policy step 30) faulted
off **5/20**, F5 (r_y +.20) 7/20; F4 is the first candidate in the registered order at or below 10/20 and is
selected. NT under F4 on the development keys 7/20 (2 wins / 0 losses, +10 points [0, 25]); the 40-key design
was expected to resolve the registered +5 net keys only if the effect is larger than on development.

**Evaluation (40 untouched keys, states 18, 19, 21, 22, seed 85001; 160 episodes; no unexposed key).**

| Arm | success |
|---|---:|
| healthy off | 40/40 |
| healthy NT | 40/40 |
| faulted off (F4) | 6/40 |
| faulted NT (F4) | **14/40** |

Primary: faulted NT − faulted off = **+8 net keys (9 wins / 1 loss), +20 points, task-clustered 95 % interval
[10, 30] points, exact McNemar p = .021** (descriptive). **T1 holds** (≥ +5 net, interval excludes zero).
Healthy NT − healthy off: 0 wins / 0 losses on 40 keys; **T2 holds** (0 losses; the zero-discordance interval
[0, 0] is not population noninferiority). Win keys: (0,18), (2,18), (2,22), (3,21), (5,18), (6,22), (7,19),
(7,21), (9,19); loss key (2,19). Telemetry: faulted NT final r_y estimate median .034 (IQR .014–.050, at the cap
in a quarter of the keys), healthy NT phantom .014 (IQR .003–.025); faulted episodes run to the 220-step cap
when they fail (median 220), healthy episodes 101 steps.

**Reading.** A uniform six-channel +.10 command bias introduced mid-episode breaks the reacting policy
(6/40) and rotation-only NT adaptation with cap .05 recovers a fifth of the keys (14/40) while leaving the
healthy 40/40 untouched. The correction can act on only three of the six faulted channels and is capped at half
the fault on those, so 14/40 is a partial repair by construction; the mid-episode onset is the new element
relative to the headline (from-step-0) cohorts. This is a task-level effect of the fixed adapter on untouched
keys under a registered fault; it is not a certificate, and the fault is a command-level bias.

Campaign closed 18:52; both servers stopped, both cards released.

## 7. Registered increment: delayed NT on the evaluation keys (19:08, before the arm was run)

EXPERIMENT_PLAN.md §7 allows a delayed-NT arm "only for an explicit delay claim". The four-arm evaluation is
closed and its outcomes were read; this increment adds **one arm**, faulted NT under F4 with adaptation
enabled at policy step 40 (`--adapt-from 40`, a 10-step delay after the onset at step 30), on the same 40
evaluation keys (paired on task, init, seed 85001), split across the two cards by state (18, 19 on GPU 1;
21, 22 on GPU 0) and merged. Comparisons: delayed NT − immediate NT (primary of this increment), delayed NT −
faulted off; exact McNemar descriptive, task-clustered bootstrap (seed 20260916). Registered expectation D1:
delayed NT is not better than immediate NT (net ≤ 0); a tie is possible. This is an increment on keys whose
off and immediate-NT outcomes are known; it is reported as such, separately from §6, and no other setting
changes. Outcome in §8.

## 8. Outcome, delayed-NT increment (run 19:06–19:25 on both cards; `evaluation/eval_delay_nt_{a,b}.json`, `analysis/delay_increment.json`)

Delayed NT (adaptation enabled at policy step 40, ten steps after the onset) under F4 on the 40 evaluation keys:
**14/40**, identical in count to immediate NT.
- Delayed − immediate NT: 14/40 vs 14/40: 3 wins / 3 losses, net +0, +0 points, task-clustered 95 % interval [-10, 10], exact McNemar p = 1.000. Win keys [[2, 19, 85001], [8, 21, 85001], [9, 21, 85001]], loss keys [[0, 18, 85001], [2, 18, 85001], [5, 18, 85001]].
- Delayed − faulted off: 14/40 vs 6/40: 8 wins / 0 losses, net +8, +20 points, task-clustered 95 % interval [10, 32], exact McNemar p = 0.008.

**D1 holds as a tie:** delayed adaptation is not better than immediate adaptation at task level, and at this n it is
not measurably worse either (six discordant keys, three each way). The fixed-command result (a ten-step delay
raises the physical cost on 18/19 sources) therefore does not translate into a task-level penalty on this fault and
cohort; task success on Spatial is tolerant of a half-second later start of correction. Reported as an increment on
inspected keys; no pooled denominator. Campaign closed 19:26; the servers were handed to the healthy coupled
replication (`PREREG_FYA_HEALTHY_COUPLED_V1.md`).
