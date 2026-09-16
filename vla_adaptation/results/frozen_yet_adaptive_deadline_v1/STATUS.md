# FrozenYet Adaptive recovery campaign, deadline v1: status

Registration: `prereg_records/PREREG_FYA_RECOVERY_DEADLINE_V1.md`. Chain: `scripts/re4/fya_chain.sh`.

| Item | State | Notes |
|---|---|---|
| Stage 0 synthetic checks | complete | `stage0/synthetic_check.json` (timing assertions, scalar floor vs LP: 2.7e-11) |
| Stage 0 configuration bundle | frozen | `configuration.json` sha256 7ee10b1f… |
| Stage 0 development replays (old states 39, 33) | complete | `stage0/dev_state{39,33}.json` |
| Stage 0 forecast feasibility (leave-one-state-out) | complete | prereg §9: interval certificate infeasible (0 decisive intervals on both folds); narrowed claims registered |
| Stage 1 sources (states 9,10 / 11,12) | running (chain launched 2026-09-16 13:42) | 40 policy episodes |
| Stage 1 qualification replay + calibration | planned | 20 sources × 20 branches |
| Stage 1 locked physical test | planned | predictions frozen first |
| Stage 2 reacting-policy bridge (states 13,14) | planned | 160 policy episodes (faulted off aliased for the delayed condition) |
| Scoring / figures | planned | `analysis/` |

Counts and missing cells are filled in as each block completes. A registration or a partial log is not a completed result.
