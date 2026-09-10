# Preserve current main when integrating review findings

Audit date: 2026-09-09. This is a read-only comparison and prospective integration
plan; no merge, checkout, simulation, or result-file modification was performed.

- Destination main: `7ca2f26fa919c8b4cb8e4532b64c8072e36368db`.
- Reviewed branch: `cda6c92357e884ef6e2b1195abda673fc18853ac`.
- Merge base: `96555f3`.
- Source repository: `https://github.com/mtaheriee/vla-adaptation`.

The branches have diverged. Review supplies substantial scientific audits and
engineering changes, but current main has eight later commits with new GR1 and
WidowX work that review does not contain. Replacing main's tree with review's tree
would discard that work. A merge request also needs an explicit decision about
project identity: review is now an independent learned-adaptation project, while
main still develops the inherited calibrated FIR method.

## 1. Main-only work to retain

| Main commit | Work absent from review at the compared snapshot |
|---|---|
| `ac510d3` | Paired GR1 experiments on two tasks; identify over three episodes and hold a reach-window median; `median` and `window` CLI statistics. |
| `0a4a5d6` | Tray-to-plate demonstration video, paired scenes, three-episode held estimate. |
| `eccf015` | GR1 right-arm offset magnitudes 0.05, 0.10, 0.20 rad with the held scheme. |
| `6787265` | Expanded experiment report including newer backbones, physical faults, ablations and GR1. |
| `ce9ee93` | GR1 plate-to-plate demonstration at +0.20 rad. |
| `1502189` | GR1 ramp, biased sinusoid and intermittent faults, including a held-ramp comparison. |
| `96a9275` | Paper wording recognizes identification across episodes and revises aggregate experiment counts. |
| `7ca2f26` | WidowX/SimplerEnv/SAPIEN, GR00T N1.7 Bridge adapter, identification, continuous and held experiments. |

Main's later GR1 results **supersede the old unpaired results as the newest GR1
evidence**. The older files retain their original interpretation; a reset fix does
not retroactively pair them. Review's predecessor paper still reports those older
results and must not overwrite main's newer table during an integration.

### Recomputed GR1 cells

Counts below were recomputed from each committed JSON's `arms[*].per_ep`, keyed by
`(task, init)`. The p-values are the ordinary exact two-sided binomial McNemar
values for the stored corrected/frozen discordances, without a new multiplicity
adjustment. This arithmetic check does not independently establish simulator reset
integrity or a prespecified analysis plan.

| File under `results/gr1/` | Healthy | Frozen faulted | Corrected | Fixed / broken | Exact p |
|---|---:|---:|---:|---:|---:|
| `p2p_right010_hold3w_paired.json` | 22/30 | 1/30 | 19/30 | 18 / 0 | 0.0000076294 |
| `t2p_right010_hold3w_paired.json` | 17/30 | 1/30 | 13/30 | 12 / 0 | 0.00048828 |
| `p2p_right005_hold3w_paired.json` | 19/30 | 7/30 | 24/30 | 17 / 0 | 0.000015259 |
| `p2p_right020_hold3w_paired.json` | 25/30 | 0/30 | 21/30 | 21 / 0 | 0.00000095367 |
| `p2p_tv_ramp.json` | 16/20 | 1/20 | 9/20 | 9 / 1 | 0.021484 |
| `p2p_tv_ramp_hold.json` | 13/20 | 1/20 | 11/20 | 10 / 0 | 0.0019531 |
| `p2p_tv_sine_bias.json` | 11/20 | 1/20 | 6/20 | 6 / 1 | 0.125 |
| `p2p_tv_intermittent.json` | 14/20 | 0/20 | 4/20 | 4 / 0 | 0.125 |

The headline two-task +0.10 rad total is healthy 39/60, frozen 2/60, corrected
32/60, with 30 fixed and zero broken. The three plate-to-plate magnitudes have
56 fixed and zero broken across 90 comparisons. **These aggregates overlap in the
0.10 rad plate-to-plate cell and must not be added as independent evidence.**

The held GR1 rows include their first three identification episodes. The method is
`law=innov`, `hold_stat=window`: a median within steps 50–200 of each identification
episode, then a median over the identification windows. The nominal fault is an
offset on all seven right-arm joint targets. It is not a torque fault, a trained
feature-basis method, or an evaluation of review's composite observer. The ramp
comparison has different realized healthy successes across its two runs; 11/20
versus 9/20 is not by itself a controlled demonstration that holding beats online
adaptation.

Source: [main GR1 runner](https://github.com/mtaheriee/vla-adaptation/blob/7ca2f26fa919c8b4cb8e4532b64c8072e36368db/openpi/gr1_adapt.py),
[main experimental record, §§32.16–32.23](https://github.com/mtaheriee/vla-adaptation/blob/7ca2f26fa919c8b4cb8e4532b64c8072e36368db/docs/ADAPTIVE_CONTROL_VLA.md).

### Recomputed WidowX cells

The new task is `simpler_env_widowx/widowx_spoon_on_towel`, using SAPIEN and the
GR00T N1.7 Bridge finetune. The perturbation is +0.005 in each xyz **command**
coordinate. The benchmark controller accumulates its end-effector target
(`use_delta=True`, `use_target=True`); this is a different control interface from
absolute joint targets and must remain visible in the experiment description.

| File / scoring window under `results/widowx/` | Healthy | Frozen | Corrected | Fixed / broken | Exact p |
|---|---:|---:|---:|---:|---:|
| `cell_tra005.json`, continuous, all 20 | 13/20 | 0/20 | 6/20 | 6 / 0 | 0.03125 |
| `cell_tra005_hold3w.json`, all 23 | 17/23 | 0/23 | 16/23 | 16 / 0 | 0.000030518 |
| Same file, held episodes 3–22 only | 14/20 | 0/20 | 14/20 | 14 / 0 | 0.00012207 |

These two windows of the held run are not additional independent experiments. The
three identification episodes cost three rollouts per arm in the stored evaluation;
the held-only headline must explicitly exclude them. On held episodes the corrected
and healthy outcomes disagree in five cases each way: equal totals do not mean
identical behavior or proven equivalence.

The recorded settings use `gamma=0.08`, `dead=0.001`, `norm_r=0.009`, `clip=0.03`,
`law=innov`, xyz corrections, a 150-step episode limit, and the diagonal FIR DC-gain
matrix in `openloop_dc.json`. The held statistic uses **steps 30–150**, not GR1's
50–200 window. The held-pose sensitivity probe produced a poor map and an aborted
run; preserve `cell_tra005_probeM_aborted.json` as failed/partial evidence.

**`cell_tra005_g02.json` is incomplete at this main commit:** it contains only eight
frozen episodes, no corrected arm, and no completed faster-gain comparison. Do not
turn the queued `gamma=0.2` trial into a reported result.

The current record acknowledges non-bit-repeatable resets across SAPIEN processes.
Same seed and same process do not alone prove that every hidden controller or
simulator state matches across arms. The saved result schema has outcome keys and
trajectories, but no full reset-state fingerprints. Retain these observations while
adding a simulator-specific reset audit before stronger counterfactual claims.

Source: [main WidowX runner](https://github.com/mtaheriee/vla-adaptation/blob/7ca2f26fa919c8b4cb8e4532b64c8072e36368db/openpi/widowx_adapt.py),
[main WidowX result directory](https://github.com/mtaheriee/vla-adaptation/tree/7ca2f26fa919c8b4cb8e4532b64c8072e36368db/results/widowx),
[main setup](https://github.com/mtaheriee/vla-adaptation/blob/7ca2f26fa919c8b4cb8e4532b64c8072e36368db/SETUP.md).

## 2. Corrections still appropriate for latest main

- Replace “at the healthy rate” or “indistinguishable” when intended to assert
  equivalence with “the difference from healthy remains unresolved in this sample.”
  Most GR1 held rates are numerically below healthy. Strong repair versus faulted
  control and equality to healthy are different hypotheses.
- Label the GR1 reach-window selection as a development decision informed by
  failed identification episodes. Confirm that rule and its parameters on fresh
  tasks/seeds and multiple independently learned held vectors. Many evaluation
  episodes sharing one held estimate do not establish robustness of identification
  across new draws.
- Preserve statistical scope: old unpaired GR1 comparisons use an independent-arm
  analysis; newer matched-scene results may use the documented paired protocol.
  Independent policy randomness does not invalidate pairing on a shared initial
  condition, but “fixed” and “broken” remain observed outcome differences.
- Keep multiplicity and repeated-use denominators explicit for severity maps,
  profile maps, task sums and report aggregates. A sum across cells, controls and
  ablations is not a count of independent unique physical conditions.
- Retain WidowX as evidence that a small final parameter error can coexist with
  task failure because earlier target error persists. “The estimator is exact”
  overstates finite noisy measurements and does not make the transient harmless.
- Correct the last sentence of main record §34.2: ALOHA's adapter sends absolute
  joint targets. Calling both LIBERO and ALOHA “delta controllers” is inaccurate;
  the interfaces have different memory and contact behavior.
- Keep command-offset successes separate from torque, friction, damping, gain and
  lock faults below the controller. GR1/WidowX additions do not resolve review's
  demonstrated all-joint torque failures or validate the new learned-basis theory.

## 3. Conflict and portability map

Both branches changed these six paths since the merge base. A read-only classic
three-way `git merge-tree 96555f3 7ca2f26 cda6c92` inspection found text conflict
markers in README, the experimental record, and the paper source, plus a divergent
binary paper. A text merge that succeeds still requires semantic review.

| Path | Integration decision |
|---|---|
| `README.md` | Keep main's project identity and newer robots; incorporate measured limitations and links to audits. Review's replacement README declares a different independent project. |
| `docs/ADAPTIVE_CONTROL_VLA.md` | Preserve main §§32.16–34.2 and newer results, while keeping review's historical corrections and convergence provenance. Do not choose one entire file. |
| `openpi/gr1_video.py` | Retain main's `--scheme-label` support and review's corrected same-scene comment. These edits occur at different locations and should combine. |
| `paper/iclr_draft.tex` | Perform a scientific rewrite, not an ours/theirs choice. Retain current main GR1/WidowX evidence, integrate review's negative results/theory/statistics, and keep the new untrained learned-basis project separate. |
| `paper/iclr_draft.pdf` | Regenerate from the final merged source; never resolve by selecting an unrelated prebuilt PDF. |
| `paper/refs.bib` | Merge verified entries, retain main's SimplerEnv reference, deduplicate keys, and verify every new citation against proceedings/source metadata. |

Main-only changed paths include `openpi/gr1_adapt.py`, the three WidowX Python files,
`SETUP.md`, report source/PDF, and new GR1/WidowX JSONs and videos. An ordinary merge
can retain them; a wholesale checkout/copy of review's tree would lose them.

Other paths can merge without textual conflict while carrying scientific changes:
review's `adaptive_law.py`, ALOHA runner, reset/fault lifecycle, calibration tools,
statistical tooling, tests, paper layout system and experimental artifacts. Review
and test them as behavior changes rather than assuming absence of conflict means
compatibility.

**Do not import as main-wide instructions or configuration:** review's `AGENTS.md`,
canonical-repository policy, independent-project migration declarations, local
dual-push remote arrangement, machine/account paths, launch schedules, and remote
service assumptions. Git remote push URLs are local configuration and do not travel
with a source merge. Preserve inherited authorship. The learned-adaptation library
may be an attributed experimental module if desired; its constant-basis robot
results do not validate the proposed learned-feature replacement.

## 4. Recommended integration order for an agent working on main

1. **Pin and inventory.** Start from the actual current main in a separate clean
   integration branch/worktree; save both full SHAs, merge base and result manifest.
   Recompute the diff if either upstream moved. Obtain merge authorization before
   publishing to main; this comparison request itself does not authorize that merge.
2. **Import integrity fixes and their tests first.** Bring statistical checks,
   duplicate episode-key rejection, reproducible LIBERO reset handling, fault
   lifecycle restoration and calibration provenance controls with dependencies.
   Existing frozen results remain historical; changed reset/calibration behavior
   defines a new evaluation protocol.
3. **Import optional controller variants explicitly.** Bring channel-restricted and
   weighted reconstruction, matched estimators, composite-reference preparation and
   artifact validation, with matching CLI/output-schema changes. Preserve legacy
   defaults for reproducibility and name new settings in configuration artifacts.
   Do not enable a weighted or composite variant everywhere based on its name.
4. **Retain main's new robot paths.** Keep GR1 reset handling, hold statistics and
   scheme labels; keep WidowX's accumulation semantics, scale normalization,
   dimensions, DC-gain calibration and distinct hold window. Generalizing the shared
   estimator to these runners requires a separate interface-level test; copying the
   Panda matrix or ALOHA gain is not a port.
5. **Import evidence with provenance.** Preserve the original JSONs and hash/seed
   manifests. Import review's all-joint confirmations and estimator tuning with
   development/validation/confirmation labels. Keep partial and failed runs labeled;
   do not pool them into completed comparisons or recount overlapping cells.
6. **Keep prospective research separate.** If importing `learned_adaptation/`, run its
   mathematical tests and expose it as experimental. It has no trained feature model;
   `libero_adapter.py` already provides a runnable Panda integration with a second-order
   ARX model and constant basis `Phi = E`. Its seven Spatial joint cells improve from
   112/140 to 135/140, with an additional Object joint-3 cell improving from 7/20 to
   18/20. These are constant-basis results, not evaluations of learned features,
   effectiveness adaptation or geometric robot control. Main's inherited FIR results cannot become
   learned-basis evaluations through a wording change.
7. **Rewrite claims and rebuild artifacts last.** Update manuscript/report/README
   from a single verified result inventory. Preserve newer main results and review's
   negative findings. Rebuild figures from cited data, regenerate PDFs, check the
   nine-page main-body constraint separately from references/appendices, and review
   all figure/table labels and significance families.

Useful read-only planning commands (replace hashes if the branches move):

```sh
git log --left-right --oneline 7ca2f26...cda6c92
git diff --stat 96555f3..7ca2f26
git diff --stat 96555f3..cda6c92
git diff 96555f3..7ca2f26 -- openpi/gr1_adapt.py openpi/gr1_video.py
```

Avoid printing an unrestricted `git merge-tree` across this repository: its binary
figures/videos and large telemetry produce hundreds of megabytes. Inspect relevant
text paths or summarize its output programmatically.

## 5. Validation required after integration

No robot simulations were run for this merge audit. These are acceptance checks
for the future integration, not a claim that an unperformed merge passes them.

- **Statistics/data:** independently recompute per-cell outcomes, unique episode
  keys, exact McNemar/Fisher distinctions, multiplicity adjustments and confidence
  intervals. Confirm incomplete WidowX `g02` is excluded. Verify GR1 includes the
  identification cost, and WidowX reports both all-23 and held-20 windows clearly.
- **Reset/fault lifecycle:** run `test_joint_fault_lifecycle.py`, the ALOHA fault
  tests and the reset auditor on the actual supported simulator. Demonstrate matching
  hidden controller targets and model parameters before paired arms; success, timeout
  and exception exits must restore injected faults. A seed-only check is insufficient.
- **Controller math:** run weighted DOB/ALOHA weighted tests, composite-observer
  tests, calibration-pipeline tests and composite-reference validation tests. Include
  rejected stale hashes and incompatible state/action ordering, not just happy paths.
- **GR1 protocol:** test the three-episode window statistic, short trajectories,
  held-state immutability after identification, identical scene regeneration and
  video scheme labels. Confirm legacy `last`/`mean50` behavior remains reproducible.
- **WidowX protocol:** test normalized pose increments, angle wrapping, input units,
  known corrected-command history, selected-axis masks, target accumulation and the
  30–150 identification window. Add reset-state evidence before stronger paired
  claims. Do not infer these properties from the LIBERO reset helper.
- **Learned module, if imported:** run
  `python -m unittest learned_adaptation.test_core learned_adaptation.test_geometry -v`.
  Passing these mathematical identity tests is not evidence of robot task repair.
- **Paper:** run the layout checker on the rebuilt source/PDF; verify citations,
  figure provenance and that every table's scheme/settings match the underlying
  files. Never transfer review's old unpaired GR1 table over the newer paired table.

The practical objective is to transfer reliable fixes, diagnoses and reusable tools
while retaining main's newer experiments. The branch is not a universally superior
controller that can be enabled by merging it.
