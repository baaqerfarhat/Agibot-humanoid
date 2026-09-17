# PREREG: crossed command-stream replay under the stronger fault (recovery study P1), v1

Registered 2026-09-17 (committed before any counterfactual cell was replayed). Implements
`papers/frozen_yet_adaptive_recovery_study/EXPERIMENT_PLAN.md` §3. Root `results/fya_recovery_study_v1/stronger_nt/`.

**Sources (archived, inspected; no new policy episodes).** `results/fya_stronger_fault_v1/evaluation/`: reference
`eval_healthy_off` (sha 985ccb885357…), M0 `eval_fault_off` (8dd98d318d26…), M1 `eval_fault_nt` (011c1194f171…),
manifest `eval_manifest.json` (ebe2fbceb477…), tasks 0–9, states 18/19/21/22, seed 85001. Extraction with
`fya_crossed_extract.py` (sha 3fd7137c0b44a4ae; `--arms eval_healthy_off,eval_fault_off,eval_fault_nt --config-arm eval_fault_nt`):
configuration from the NT telemetry header (sha 7e40a9754cfc2380: deployed W with r_y tap sum .10281, full M, γ .08,
δ .008, ρ .15, cap .05, mask {3,4,5}, all-channel normalisation, fault +.10 on all six coordinates, activation at
policy step 30). **32 eligible keys across all ten tasks** (`source_keys.csv` sha 9701599cc4b7eeff, audit sha
0d70bdde65335419): excluded before replay are 5/19 and 8/19 (healthy prefix mismatch) and 0/19, 0/21, 0/22, 3/19, 3/21,
3/22 (fewer than 80 policy steps). Task outcomes of the 32 keys (reference 32/32, off 4/32, NT 11/32) are a subset;
**the task-success claim stays 6/40 → 14/40 on all assigned keys**.

**Protocol.** Prefix 30 policy steps (uncorrected, unfaulted), snapshot at env step 40, window 50 steps; fault
+.10 on all six command coordinates throughout the window; cells ref, J00, J10, J01, J11 plus J00_fresh, J11_fresh;
NT rerun causally from a zero estimate with the FIR history of the prefix; cell order rotated by key index. Driver
`fya_crossed_replay.py` (sha 25e25fc3b730c71f; `--matrix nt --ref-arm eval_healthy_off --m0-arm eval_fault_off --m1-arm
eval_fault_nt`). Fidelity as registered for the archived matrices: prefix, ref, J00, J11 (incl. corrections) and fresh
routes within 1e-8 (rad, m, SO(3)); a failing key is excluded and listed; continuation after a wrapper done follows
the registered semantics; input clipping (|command| > 1: 35/1,600 off steps and 67/1,600 NT steps on these windows,
all translation) is a range diagnostic, reported.

**Endpoints.** Cost J = .05 Σ‖p − p_ref‖² (m² s) against the healthy reference of the same key. **Primary
R1 = J10 − J11**, equal-task estimand (average keys within task, then tasks), task-clustered percentile bootstrap,
10,000 draws, **seed 20260918**, practical margin **1e-5 m² s**; positive = the NT-generated stream reduces cost
under correction, negative = increases; registered without a sign expectation. Secondary: D0, T, I (same rule),
all four cell costs, rotation energy, endpoints, done and clipping flags; source-weighted and leave-one-task-out
sensitivities. Scorer `fya_crossed_score.py` (sha 7a5353c2b62f0474; `--primary R1 --delta-I 1e-5 --seed 20260918`; the
`--delta-I` option is the primary's margin). No change of primary, margin, window, tolerance or key set after any
cell is read. Outcome appended below.

## Outcome (replayed 02:49–03:04, scored 03:04; `runs/crossed_strong_nt.json.gz`, `analysis/`)

**Provenance note.** During the replay the driver and scorer files were edited on disk to add DOB support and the
S column for P3 (driver sha now f88ab2229e8c1da3, scorer 8aaa407b195872cb); the process that executed this matrix had loaded the registered
driver (d27a7c8f, commit 81ee1f9) before the edit, and the scorer edit inserts S after T so the primary and the I/D0/T
draws are unchanged; the rotation-energy draws shift. Both registered versions are in commit 81ee1f9.

**Fidelity.** 30 of 32 keys reproduce the archived paths exactly (all gaps 0.0). Two keys fail the registered
tolerances and are excluded: task 7 state 18 (live-vs-replay 3.1e-5 rad on the reference, 6.7e-5 on J00, 4.4e-4 on
J11) and task 8 state 22 (J00 3.6e-5, J11 1.6e-3 rad); on both the fresh and restored replay routes agree exactly with
each other, so these are live-process divergences of the kind seen before. **30 keys across all ten tasks** scored.

**Cells (medians, m² s; endpoint mm):** J00 7.33e-3 (98.7), J10 6.44e-3 (100.2), J01 6.07e-3 (74.7), J11 6.24e-3
(81.5). Costs are ten to forty times the healthy and benign-fault matrices: under +.10 on all six coordinates every
faulted path leaves the healthy reference by about 10 cm within 2.5 s, and 35–67 of 1,500 window steps request
clipped translation inputs. No window terminated except one J11 done flag.

| Contrast | equal-task | 95 % CI | keys +/− | decision |
|---|---:|---|---|---|
| **R1 = J10 − J11 (primary)** | **1.20e-3** | **[−2.7e-3, 4.5e-3]** | 19 / 11 | **unresolved** |
| D0 = J00 − J10 | 5.2e-5 | [−1.7e-4, 3.0e-4] | 14 / 16 | unresolved |
| T = J00 − J11 | 1.26e-3 | [−2.8e-3, 4.8e-3] | 18 / 12 | unresolved |
| I = D1 − D0 | −1.7e-4 | [−4.7e-4, 1.4e-4] | 12 / 18 | unresolved |
| S = D0 − R1 | −1.15e-3 | [−4.3e-3, 2.8e-3] | 12 / 18 | unresolved |

Per-task R1 ranges from −1.4e-2 (task 7) to +1.1e-2 (task 9); leave-one-task-out means 1.2e-4 to 2.9e-3.

**Reading.** Under the fault that actually changes task outcomes (6/40 → 14/40), the fifty-step fixed-command
decomposition is uninformative: the direct correction term D0 is not distinguishable from zero (the rotation-only
correction, capped at half the fault, barely moves a path that is already 10 cm off within the window), and the
stream and total terms are dominated by task-specific divergence with heavy tails. The task repair measured over
full episodes is therefore not accounted for by physical proximity to the healthy path in the first 2.5 s after
onset; the mechanism operates on a longer horizon or on a different quantity than this cost. This is the registered
negative result of P1; the 6/40 → 14/40 task comparison stands unchanged.
