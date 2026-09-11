# ALOHA weighted-observer transfer confirmation

Prospective registration, 2026-09-08. This is a new evaluation of the fixed
weighted disturbance-allocation controller on ALOHA. It does not retrospectively
preregister the completed ALOHA development comparisons. No weighted-ALOHA outcome
is used to select its parameters in this plan. The input hashes, exact commands,
and selection record are written before the confirmation runs start.

## Fixed allocation

Task: `gym_aloha/AlohaTransferCube-v0`, using the existing frozen policy server.
There are thirteen cells: one healthy cell, then one constant positive physical
torque fault on each of the twelve arm joints. Physical indices 0–5 are the left
arm and 6–11 the right arm; their command coordinates are 0–5 and 7–12. Both
grippers are excluded from adaptive correction.

Every cell uses twenty complete initial scenes, reset seeds **3000–3019**, and
three arms in the input order `off,legacy_allchannels,weighted`: **780 scored
rollouts**. `off` is one shared cohort within a cell. It is not rerun separately
for each comparator. The healthy cell is run once. All thirteen cell results are
reported; cells are not removed because the frozen policy has little headroom or
the candidate causes regressions.

Joint torque magnitude equals the live nominal position-actuator
command-to-torque gain times **+0.02 rad**. For each arm the gains are
`[800,1600,800,10,50,20]`, yielding torques `[16,32,16,0.2,1,0.4]` N·m.
The runner rejects a different live gain or command mapping. This scale describes
an unsaturated nominal servo offset; it does not guarantee a 20-mrad displacement
under load or saturation, equal dynamic difficulty across joints, or equivalence
to the Panda's 5-N·m conditions. Position/control/force limits and saturation
diagnostics remain part of the result.

## Controller and calibration lock

All arms use the same fourteen-channel healthy FIR `W` and forward sensitivity
`M` carried by `aloha_reference.json`. Its healthy residual covariance `R` supplies
the weighted allocation. The weighted arm uses `baseline=weighted_dob`, fixed
regularization scale **sigma=0.1 in numerical ALOHA command units**, update gain
**gamma=0.08**, and correction/estimate bound **0.08**. The twelve active command
coordinates are arm angles in radians. Transferring the numerical value 0.1 does
not make its physical meaning identical to normalized Cartesian action units in
LIBERO. There is no tuning on the reserved ALOHA outcomes.

The legacy comparator uses its original all-channel residual normalization,
deadzone `0.002`, normalization radius `0.4`, gain `0.08`, and bound `0.08`.
Both candidates correct all twelve arm coordinates. Bias correction is absent,
estimates and estimator covariance reset every episode, and no estimate is
carried between scenes. The action horizon remains ten commands, control duration
0.02 seconds, and episode cap 300 policy actions.

The observer artifact's healthy log is linked by SHA-256 to the seed-2600
collection: fit episodes 0–5 and model-validation episodes 6–9. The existing
sensitivity artifact is reused and linked by hash; its historical file does not
provide complete reset-seed provenance. This limitation is disclosed rather than
describing the whole identification history as newly held out. No new calibration
is fitted on seeds 3000–3019.

The artifact qualified a left-six position reference only. The attempted
all-twelve reference failed its held-out position-error threshold; that rejected
artifact remains preserved. This study uses observer matrices only, **no
composite reference feedback**. It does not promote the left-six qualification to
an all-twelve claim.

Physical torque is injected below the servo through the environment wrapper. Its
truth is never supplied to the estimator. Any underlying `f_true` telemetry field
denotes the action-interface fault, which is zero in these cells, not the physical
torque or a directly comparable disturbance target.

## Pairing and execution

For episode counter `i`, cyclically rotate the input arm order by `i modulo 3`
and reverse that order on odd `i`. Complete all arms of a scene before advancing.
The order is independent of outcomes. The policy sampling RNG remains unpinned.
Identical scenes therefore do not imply identical policy action samples.

After reset and before inference or torque injection, record complete simulator
`qpos` and `qvel`, including object state, and their hashes. Compare every later
arm with the first arm of that scene; discrepancies above `1e-9` abort and mark
the cell failed. The collector verifies all twenty local episode indices, their
actual seeds, pairing-check counts, complete three-arm allocation, and common
off-cohort identity. Partial or failed cells are retained but cannot be silently
scored as the complete study. No optional stopping or outcome-dependent extension
is authorized by this plan.

Source scripts, calibration, this registration, and the transfer selection record
are hashed before launch in a separate frozen ALOHA source directory. A preflight
checks those bytes and rejects existing confirmation outcome directories. The
generator and preflight do not launch simulations. Compact result views preserve
the authoritative study and telemetry hashes; full source artifacts are retained.

## Prespecified reporting

There are three separate **global families of thirteen paired comparisons**:

1. `legacy_allchannels` versus its cell's shared `off` cohort.
2. `weighted` versus its cell's shared `off` cohort.
3. `weighted` versus `legacy_allchannels` within the cell.

Use exact two-sided McNemar tests and Bonferroni adjustment by thirteen within
each family. No benchmark/cell subset is used to reduce these denominators. Report
successes/20, absolute rate changes, gains and losses on paired scenes, losses/all
scenes, losses/comparator-success opportunities, and one-sided 95% upper bounds on
conditional loss rates. Zero opportunities give an undefined conditional rate.
An unresolved contrast is not equivalence or demonstrated noninferiority.

Report actuator-limit and estimate-clipping counts alongside outcomes. Across-cell
totals may be descriptive, but gains on one joint must not hide harm on another;
there is no registered pooled-all-joints primary claim. A positive finding would
support this fixed ALOHA transfer configuration under the tested conditions, not
general physical-fault safety or novelty of an observer update.
