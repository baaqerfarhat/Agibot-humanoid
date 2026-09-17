# Stronger mid-episode fault campaign v1: status

Registration `prereg_records/PREREG_FYA_STRONGER_FAULT_V1.md` (§§5–6 outcomes). Closed 2026-09-16 18:52.

| Item | State |
|---|---|
| Development (states 15, 17): F1 18, F2 16, F3 12, F4 5, F5 7 of 20; healthy 20/20; NT under F4 7/20 | complete |
| Selection | F4 = uniform +.10 on all six channels from policy step 30 (registered rule) |
| Evaluation (40 untouched keys): healthy 40/40 → 40/40; faulted 6/40 → 14/40 (+8 net, [10, 30] points, p = .021) | complete |
| Analysis | `analysis/summary.json`, `figures/sf_summary.pdf` |

| Delayed-NT repair (`PREREG_FYA_DELAY_REPAIR_V1.md`): 13/40, coupled 40/40, vs immediate net −1 [−10, +5] pts | complete |
| Delayed-NT increment (prereg §7–8, superseded for controlled comparisons): 14/40, tie with immediate NT (3/3 discordant), `analysis/delay_increment.json` | complete |
| Delayed-NT repair (`PREREG_FYA_DELAY_REPAIR_V1.md`): full manifest, one process, coupled 40/40; delayed 13/40 vs immediate 14/40 (net −1, [−10, +5]); the sharded increment above is superseded for delay claims | complete |

No missing cells; no unexposed keys; telemetry gzipped.
