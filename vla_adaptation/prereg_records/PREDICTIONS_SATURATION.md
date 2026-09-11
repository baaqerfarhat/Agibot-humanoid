# Two registered predictions, locked before the result

**2026-09-08.** Both tracks committed before the cells finished. The second track did not see the
first's prediction and did not control the run. Recorded here so the scoring cannot be revised.

| | outcome category | joint 5 corrected | joint 3 corrected |
|---|---|---|---|
| **Claude track** | harm gets **worse** | ≤ 3/20 | (gate only, ≥ 14/20) |
| **Codex track** | harm **unchanged** | **1–5/20**, centre 3 | **15–18/20**, centre 18 |

Baselines on disk at `clip 0.30`: joint 5 is 12/20 → 3/20; joint 3 is 11/20 → 18/20.

## The tracks disagree on category, and the reasoning differs in quality

**Claude:** the estimator adapted continuously and still produced zero fixes in twelve
opportunities, which reads as tracking the wrong quantity rather than failing to reach the right
magnitude; more headroom should therefore make it worse. Held weakly, and registered against the
7.9%-vs-1.3% saturation asymmetry that motivated the cell.

**Codex:** derived a per-row reachability bound from the stored calibration,
`|q_i| <= (rho/2)||row_i(M+)||_2 = [0.337, 0.289, 0.569, 0.298, 0.276, 0.309]`, so **only the z
channel can exceed the old 0.30 cap** — raising the clip changes one channel and nothing else. It
then replayed each probe residual stream through the estimator at both caps, validating the replay
by reproducing the saved estimates to `5.6e-17`, and found a mean absolute z difference of 0.008
(joint 5 frozen probe, max 0.082). Modest change, so harm persists.

**Independent check of the reachability claim.** Codex called `clip 0.60` "nonbinding." Recomputing
from `openloop_so3.json`: `||M^-1||_2 = 8.287`, giving an isotropic worst case of `0.6215`, and
sampling gives a maximum reachable translation-dim estimate of **0.5575** — consistent with its
per-row `0.569` for z. So 0.60 is *just* above the ceiling. The intervention is therefore not inert:
it moves the cell from clipped-at-0.30 to effectively unclipped, roughly doubling reachable z
authority. That is a cleaner test than a 2x cap increase would have been.

## What each outcome would mean

- **J5 >= 7/20 with the gate passing** falsifies both tracks' substantive claims and identifies
  projection as the dominant bottleneck. The successor's representational motivation would then be
  aimed at the wrong target.
- **J5 in 1-5/20** confirms Codex and refutes Claude's category. Saturation is not dominant, and the
  cause remains unidentified among at least three candidates.
- **J5 <= 1/20** would confirm Claude's direction, but the ranges overlap at 1, so a value of 1
  scores for both and discriminates nothing.
- **J3 < 15/20** fails the validity gate and neither arm is interpretable.

## Limits both tracks agree on

Success counts alone cannot identify a physical mechanism. Codex states this explicitly: the
experiment can falsify the practical claim about removing projection, but not uniquely establish a
feedback, contact or reachability mechanism. The frozen arm is rerun, so a shifted baseline must be
reported rather than silently compared against the stored 12/20.

Codex also notes a confound in the saturation statistic that motivated this cell: failure episodes
run 220 steps while successes run 75-117, so whole-episode bound occupancy is inflated by duration.
Restricted to the first 60 applied steps, joint-5 occupancy is **10.6% in successes** against
**2.75% in failures** — the opposite ordering. Reaching the bound does not select failures. That
weakens the premise of my own gate, and it was found by the track that did not design it.
