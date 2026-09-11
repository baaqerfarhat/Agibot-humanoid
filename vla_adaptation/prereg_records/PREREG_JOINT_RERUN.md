# Preregistration: joint-level cells under the corrected fault protocol (2026-09-10)

**Written before the reruns.** Dual-track audit, "what must be rerun": the historical
friction and lock cells (§29.3–29.4 of the record) ran with a runner that mutated the MuJoCo
model (`dof_frictionloss`, `jnt_range`) on a cached environment without guaranteed restoration
on every exit, and with `reset(); set_init_state(...)`, which does not prove the two arms
start from the same physical scene.

## What changed in the runner
- `joint_fault.JointFault` restores every touched field in a `finally`, on success, timeout and
  exception (`openpi/test_joint_fault_lifecycle.py`, passing).
- `--scenario-reset` (exposed today; the merged branch had the helper but left the flag off)
  resets through `libero_reset.reset_libero`: applied and external forces cleared, the cached
  env seeded per scenario, model/state fingerprint recorded.

## Cells, unchanged in every other respect
π0.5, `libero_spatial`, n = 20 paired, elbow (joint 3), translation corrected (`--corr-dims 0,1,2`),
healthy phantom subtracted (`--bias 0.007,-0.008,0.016`), γ 0.08, dead 0.008, ρ 0.15, the shipped
plant and M, `--scenario-reset`:

| cell | historical | clip |
|---|---|---|
| friction +20 | 0/20 → 8/20, 8 fixed / 0 broken, p = 0.0078 | 0.30 |
| lock ±0.05 rad | 0/20 → 0/20, 0 / 0 | 0.30 |
| healthy control (no joint fault, law running) | not previously run for this class | 0.30 |

## Predictions
1. **friction +20** stays a repair: corrected ≥ 5/20 from a frozen ≤ 3/20, with ≤ 2 broken.
   If the frozen arm is no longer near zero, the historical cell's damage was partly the
   uncleared force state and the claim is withdrawn.
2. **lock** stays not-repaired: |corrected − frozen| ≤ 2 with no significant McNemar. A repair
   appearing here would mean the historical non-repair was an artefact of the old reset, which
   would overturn the paper's rank argument.
3. **healthy control**: the law on a healthy arm with this clip changes the count by ≤ 2.
4. Fault restoration: the recorded fingerprints show the friction and range fields back at
   their nominal values after every episode.

Refutation is any prediction failing; the historical rows are then relabelled as run under
the old protocol and the new rows replace them in the paper.

---

## Amendment, written before the n=40 run (2026-09-11)

The friction +20 rerun at n=20 came out `1/20 → 4/20`, 3 fixed / 0 broken, p = 0.25, against
the historical `0/20 → 8/20`, 8 fixed / 0 broken, p = 0.0078. Prediction 1 required at least
5 corrected, so **it is refuted as registered**. But 8/20 against 4/20 is a 20-point gap on a
benchmark whose measurement noise at n = 20 is ±11 points, and the estimate is nearly
unchanged between the two runs (final median x, z: −0.23, −0.23 historical against −0.19,
−0.20 rerun), so identification is not what differs. The cell is underpowered to decide
whether the corrected protocol removed the effect or merely halved a noisy one.

**Extension, registered now:** the same cell at n = 40 (initial states 45–48), same protocol.
- If corrected ≥ 10/40 with ≤ 2 broken and p < 0.05, the repair survives the corrected protocol
  and the paper keeps the row with the n = 40 numbers.
- If corrected < 10/40 or the test does not resolve, the friction row is reported as **not
  reproduced under fault restoration**: the historical significant cell is relabelled as run
  under the old protocol, and the claim is withdrawn from the paper's joint-fault table.
- Either way the n = 20 rerun above is reported, not discarded.
