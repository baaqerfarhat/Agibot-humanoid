# What we need to run v18 in sim

**Ask: `model_81499.pt`** — the training checkpoint behind
`x2_box_policy_ankle_scale_v18_iter81499.npz`. Its own metadata says it lives at

    /home/baaqer/baaqer_ws/holosoma/logs/WholeBodyTracking/
      20260827_235155-x2_box_ankle_scale_v18-locomotion/model_81499.pt

and the matching `holosoma_config.yaml` from that run directory.

## Why the exported npz is not enough

The npz is the deploy artifact: normalizer, actor weights, reference arrays, and the
joint metadata. It has no critic and no training config, and the sim evaluation path
(`eval_record_driver.py <checkpoint.pt> <out.npz>`) takes a `.pt`. So with the npz
alone the policy can be run on the robot but not in the simulator it was trained in.

We could hand-write an npz-driven sim harness, but the 164-dim observation has to be
assembled in exactly holosoma's order and normalisation. If any of that is subtly
wrong the rollout still runs and still looks plausible, and every number that comes
out of it is wrong. Not worth the risk when the `.pt` exists.

## The question it answers

The robot completes the clip on hardware in 2 of 9 engaged runs (`20260831_174433`,
`174513`) and stops early in the other 7. In simulation it completes. Nobody has
compared the two on the same channels, so "why does sim work" is still unanswered.

With the checkpoint we would run v18 in sim, log the same signals as the hardware
logs, and compare through the four infeasible single-support spells:

- ankle_roll / ankle_pitch duty cycle at the effort limit
- foot loading — does the policy actually enter single support in sim, or does it keep
  both feet down and eat the tracking error?
- base pitch and lateral CoM offset from the stance foot's roll axis

The specific hypothesis to test: **on hardware the runs that survive are the ones that
refuse to follow the reference into single support.** Measured over frames 179-250,
foot height asymmetry is 17.5 mm mean in the run that completed and 29.7 / 43.3 mm in
two that fell — the more the robot lifted a foot, the sooner it went down. If sim
also refuses, then sim and hardware differ only in how reliably that improvisation
holds, and the gap is disturbance rejection rather than behaviour. If sim follows the
reference into single support and survives anyway, then sim is producing restoring
moment the real foot cannot, and the contact model is the gap.

## Also useful, lower priority

- `model_*.pt` + config for **v17 iter49000**, for the same comparison on the policy
  that produced the first complete hardware run
- Whatever sim rollout logs already exist behind the substep duty-cycle numbers in
  2cd5b81 (ankle_roll 64%/72% v17, 64%/70% v18) — if those are already on disk, they
  may answer the question without re-running anything

## Two negative results, so nobody repeats them

Both were checked against the hardware runs and both fail:

- **ankle_pitch torque entering the first infeasible window.** Separates completed
  from stopped on v18 (7.2-14.7 vs 28.9-34.6 N-m) and **overlaps on v17** (completed
  9.4, stopped 5.6-20.9). It was reading the difference between two policies' ankle
  usage, not a precursor of failure.
- **Lateral CoM offset against the 57 mm bound.** Separates in one window (frames
  160-179) on both policies, and **fails in the next window** (205-217: completed 80,
  80, 12 mm against stopped 32-257) and over the whole run (completed average 65-129
  mm, one failed run the lowest of all at 43 mm).

So there is no simple scalar threshold that predicts the failure across windows. A
gated supervisor or a triggered adapter has nothing reliable to gate on; it would
need to run continuously or use a learned multivariate signal.

## What is not in question

The reference asks for up to 127.1 N-m of ankle_roll restoring moment in single
support. The ground reaction acts inside the sole, so the moment arm cannot exceed
the 50 mm half width whatever the actuator does — 21.1 N-m. Verified independently
by running your `check_ankle_roll_feasible.py` on the shipped clip. Rebuilding the
four spells is the fix; everything above is about understanding the failure while
that happens.
