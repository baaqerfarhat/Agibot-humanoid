# PREREG: independent healthy crossed matrix on LIBERO-10 (recovery study P2), v1

Registered 2026-09-17 (committed before any episode). Implements
`papers/frozen_yet_adaptive_recovery_study/EXPERIMENT_PLAN.md` §4. Root `results/fya_recovery_study_v1/healthy_libero10/`.

**Keys.** LIBERO-10 used-state audit (full-text scan of `results/`, this morning): states 25 and 39–49 used;
0–24 and 26–38 free. Allocation, frozen: **tasks 0–9 × states 30, 31, 32, 33, sampler seed 87001** (`manifest.json`,
40 keys). No inspected LIBERO-10 state is reused. Calibration provenance: the shipped W/M are Spatial-based (documented
suite transfer, as in the E2 and six-channel studies).

**Sources.** One server process on GPU 1 (port 8000, memory fraction .42), arms run sequentially with the full ordered
manifest: healthy_off (`--arms frozen`), healthy_nt (`--arms adaptive --law legacy --baseline none`), healthy_off_dup
(`--arms frozen`); `--suite libero_10 --scenario-reset --replan-steps 5 --gamma 0.08 --dead 0.008 --norm-r 0.15
--clip 0.05 --corr-dims 3,4,5 --norm-channels all --fault-vec 0,0,0,0,0,0 --onset 40 --adapt-from 30`; horizon 520
policy steps. 120 episodes; healthy failures are retained as sources. Per-episode control acknowledgements and the
server identity are logged by the runner.

**Eligibility and fidelity.** ≥ 80 policy steps in off and NT; exact off/NT prefix agreement (raw actions,
positions, joints) is a primary eligibility criterion; duplicate-off full-episode agreement is reported for every key
and used only as a separately labelled duplicate-exact sensitivity (continuity with the Spatial study), decided now.
Replay: `fya_crossed_replay.py --matrix healthy --suite libero_10` from the extracted bundle
(`--arms healthy_off,healthy_off,healthy_nt,healthy_off_dup --config-arm healthy_nt`), fault off in every branch,
tolerances and continuation semantics as registered before.

**Endpoints.** **Primary R1 = J10 − J11**, equal-task mean over the eligible keys, task-clustered percentile
bootstrap, 10,000 draws, **seed 20260918**, margin **1e-5 m² s**. Expectation (refutation primary): R1 < −1e-5, as on
Spatial (archived −3.6e-4 on 16 keys; independent −1.7e-4 on 34 keys). **Secondary, registered now: S = D0 − R1**
(= J11 − 2·J10 when J00 = 0, to be verified), same bootstrap; positive S means the stream-associated cost exceeds the
direct false-correction cost. Also D0, T, I, cell costs, endpoints, source-weighted and leave-one-task-out
sensitivities, all healthy task outcomes with every NT regression. Suites are not pooled. Chain
`scripts/re4/fya_hl10_chain.sh`. Outcome appended below.

## Outcome (sources 02:49–05:25 on one GPU 1 server process; replay and score 05:25–05:33; `sources/`, `runs/`, `analysis/`)

**Sources (120 episodes, LIBERO-10, states 30–33, seed 87001).** healthy_off 38/40 (lost 4/31, 8/32), healthy_nt
37/40 (lost 8/30, 8/32, 6/33), healthy_off_dup 37/40 (lost 8/30, 4/31, 8/32). Every assigned key ran; healthy
failures were retained as sources. Coupling: off/NT prefixes exact on **36/40** (2/30 and 8/31 diverge at step 0,
8/30 and 8/33 at step 5); duplicate off exact over whole episodes on **31/40** (divergences at steps 5–405, mostly
tasks 2 and 8). NT healthy regressions relative to off: 8/30 and 6/33 lost, 4/31 gained.

**Eligibility and fidelity.** 36 keys eligible (the four prefix mismatches excluded; every key has ≥ 80 steps);
**all 36 reproduce the archived paths exactly** (ref, J00, J11 incl. corrections, fresh routes, gaps 0.0).
36 keys across all ten tasks scored; J00 = 0 on every key, so S = J11 − 2·J10 holds exactly.

| Contrast | equal-task | 95 % CI | keys +/− | decision |
|---|---:|---|---|---|
| **R1 = J10 − J11 (primary)** | **−8.5e-5** | **[−1.6e-4, −2.4e-5]** | 0 / 36 | **resolved negative: replicated** |
| S = D0 − R1 (registered secondary) | 7.2e-5 | [1.1e-5, 1.5e-4] | 27 / 9 | positive |
| D0 = J00 − J10 | −1.3e-5 | [−2.3e-5, −4.2e-6] | 0 / 36 | negative on every key |
| T = J00 − J11 | −9.8e-5 | [−1.7e-4, −3.8e-5] | 0 / 36 | negative |
| I = D1 − D0 | 1.1e-5 | [−3.0e-6, 3.0e-5] | 18 / 18 | unresolved |

Per-task R1 from −1e-6 (task 2) to −3.9e-4 (task 5); leave-one-task-out −5.1e-5 to −9.4e-5; source-weighted
−8.5e-5. Cells (medians): J10 2.0e-6 (endpoint 1.0 mm), J01 1.9e-5 (5.5 mm), J11 2.0e-5 (5.3 mm).

**Reading.** The healthy stream effect transfers to a second suite with the same structure and about half the
Spatial magnitude (Spatial archived −3.6e-4 on 16 keys, fresh −1.7e-4 on 34 keys; LIBERO-10 −8.5e-5 on 36 keys):
direct false-update deviation is 1 mm at the endpoint, the stream the policy generates under those corrections
lands 5.5 mm from the healthy reference, and the registered secondary S confirms prospectively that the
stream-associated cost exceeds the direct cost on 27 of 36 keys. Physical deviation only; healthy task success on
LIBERO-10 is imperfect in every arm (38/40, 37/40, 37/40), which the paper reports alongside.
