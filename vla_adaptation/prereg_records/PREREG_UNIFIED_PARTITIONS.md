# Declaration: healthy source partitions for E1 and E2 (2026-09-14, before collection)

Shared protocol items 2–3 of `iclr2027/EXPERIMENT_PLAN.md`. LIBERO stores 50 initial states per
task; the scenario-manifest ledger (`results/iclr_unified_v1/manifests/`) lists the states every
stored result or calibration has used on `libero_spatial`: 20, 30, 32, 35–38, 45–48, and the E2
confirmation manifest reserves 49 and 44. The partitions below use none of those. Outcomes of the
collections are not inspected before the partitions are fixed; the rule is the state index.

| partition | suite | states (all ten tasks) | episodes | use |
|---|---|---|---|---|
| E2 fit | libero_spatial | 40, 41, 42 | 30 | fit predictors U and C (same episodes, regressor, K = 6, ridge) |
| E2 qualification | libero_spatial | 43 | 10 | probe qualification checkpoints, envelope selection |
| E1 fit | libero_spatial | 39 (tasks 0–5) | 6 | signed-memory response model |
| E1 qualification | libero_spatial | 39 (tasks 6–9) | 4 | freeze model order/state/conversion; remainder envelope |
| E1 locked test | libero_spatial | 34 | 10 | prospective coverage; never refit |
| E2 confirmation | libero_spatial, libero_10 | 49, 44 / 41, 40 | 120 keys × 3 seeds | the 840-rollout core (separate prereg) |

Collection: `error_signal.py --healthy-only` under the shipped protocol (unpinned sampler,
recorded as such; policy pi0.5, replan 5), 20 Hz, suite cap. Checkpoints for E1/E2 branching:
steps 30 and 70 of each source (declared; a source shorter than checkpoint + horizon leaves that
checkpoint missing). Horizons: E1 100 steps, E2 probes 40 steps. Faults: E1 +0.05 on normalised
rotation-y; E2 core uniform +0.05 on six channels, rotation mask {3, 4, 5}.
