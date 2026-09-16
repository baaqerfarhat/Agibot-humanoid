# FrozenYet Adaptive recovery campaign, deadline v1: status

Registration: `prereg_records/PREREG_FYA_RECOVERY_DEADLINE_V1.md`. Chain: `scripts/re4/fya_chain.sh`.

| Item | State | Notes |
|---|---|---|
| Stage 0 synthetic checks | complete | `stage0/synthetic_check.json` (timing assertions, scalar floor vs LP: 2.7e-11) |
| Stage 0 configuration bundle | frozen | `configuration.json` sha256 7ee10b1f… |
| Stage 0 development replays (old states 39, 33) | complete | `stage0/dev_state{39,33}.json` |
| Stage 0 forecast feasibility (leave-one-state-out) | complete | prereg §9: interval certificate infeasible (0 decisive intervals on both folds); narrowed claims registered |
| Stage 1 sources (states 9,10 / 11,12) | complete | 40/40 successful; qualification 18/20 eligible by length (task 0 at both states short), test 20/20 |
| Stage 1 qualification replay + calibration | complete | 17 valid sources (one more, task 3 state 10, failed the fresh-prefix fidelity tolerance: gap 2.3e-6 rad); ε max .0257, affine a .010 / ρ 3.83; 0 decisive intervals; point-forecast sign agreement nt .90 / innovation .88 / reference .99 |
| Stage 1 locked physical test | complete | 19 valid sources (task 7 state 12 failed the fresh-prefix tolerance); 0 decisive intervals (registered); point-forecast sign agreement nt .88 / innovation .87; E1 holds, E2 cost holds / fraction clause fails, E3 fails, E4 holds (prereg §10) |
| Stage 2 reacting-policy bridge (states 13,14) | complete | 160 episodes; all arms 20/20 except healthy NT 19/20: the registered mid-episode r_y fault does not degrade the reacting policy (ceiling); S1 fails, S2 trivial, S3 holds; healthy phantom r_y estimate .015 (prereg §11) | 160 policy episodes (faulted off aliased for the delayed condition) |
| Scoring / figures | complete | `analysis/`, `figures/fya_{benefits,delay_cap,forecast,healthy,stage2}.pdf` |

Counts and missing cells are filled in as each block completes. A registration or a partial log is not a completed result.

Campaign closed 2026-09-16 15:20. Missing cells: none planned; excluded sources: qualification 3 (two by length, one by fidelity), test 1 (fidelity). Registration or partial logs are not results; every outcome above is scored.
