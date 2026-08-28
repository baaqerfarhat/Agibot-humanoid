# What diagnose_clip.py says about the shipped reference

Output of your own `box_pickup/diagnose_clip.py` on `sub3_largebox_003_walk.npz`
(raw retarget) and `sub3_largebox_003_walk_feasible.npz` (the clip v17 trained on).
Reproduce with:

    cd box_pickup
    python diagnose_clip.py \
      raw=<motions>/sub3_largebox_003_walk.npz \
      refined=<motions>/sub3_largebox_003_walk_feasible.npz

One portability fix was needed to run it at all: `rebuild_reference_motion.py` had
`WS = Path("/home/baaqer/baaqer_ws")` hardcoded, so `diagnose_clip.py` and
everything else importing it dies immediately on another machine. It now takes
`X2_WS`, then the checkout it lives in, then your original path.

## Clean

| check | refined |
|---|---|
| joint position violations | **0** |
| joint velocity violations | **0** |
| palms below 15 cm | 0 of 591 |
| foot penetration below lowest contact | 0 mm |
| foot creep per stance | 5 mm left, 7 mm right (worst 13) |
| box lift | z 0.183 -> 0.606 m |

Worth stating plainly because an earlier note of mine claimed otherwise: **nothing in
either clip exceeds a joint limit.** That claim came from a bug in my own limit table
(I applied the left hip_roll range to the right hip_roll, which is mirrored). The
clip is feasible, which is what it looks like in sim and in the reference video.

## Flagged

| finding | left | right |
|---|---|---|
| ankle roll hard against its limit | 40 frames | 64 frames |
| sole tilt while loaded | 4.0 mean / 11.6 peak deg | **6.6 mean / 24.5 peak deg** |
| leg reach (straight leg 0.618 m) | max 0.629 m, 111 frames over 0.615 | max **0.644 m**, 140 frames over |

Plus, whole-body:

- **ZMP margin min -250 mm, outside the support polygon on 194 of 591 frames (33%)**
- CoM margin min -101 mm, outside on 25 of 591
- waist_pitch at **86%** of its effort limit from gravity alone at 0 kg payload;
  **106%** at 3 kg
- planar CoM accel peak 5.18 m/s^2, peak joint jerk 1215 rad/s^3

## The refinement passes did help

Every one of these is better than the raw retarget:

| | raw | refined |
|---|---|---|
| CoM outside support | 111 / 325 frames | 25 / 591 |
| ZMP margin min | -608 mm | -250 mm |
| CoM accel peak | 9.76 m/s^2 | 5.18 |
| peak joint jerk | 5011 rad/s^3 | 1215 |
| median abs(ankle_roll) | 14.5 / 13.3 deg | 8.1 / 7.7 |
| frames at the ankle_roll stop | 49.5% / 40.6% | 6.8% / 5.4% |

In the raw retarget each ankle is rolled to one side and never crosses zero -- left
+2.4..+15.0, right -15.0..+1.4, both clamped at 0.262 rad. That reads as a static
splay from mapping human ankle geometry onto the X2 foot rather than gait. The
refinement removes most of it.

## How much of this matters is an open question

Ordering them by how much we think they matter, with the reasoning, so it can be
argued with:

**Probably minor: leg reach.** 26 mm over a 0.618 m straight leg, 4%, on 140 frames.
RL does not break on an unreachable reference -- the policy converges to the closest
achievable pose and carries a small permanent tracking penalty. Consistent with the
hardware run, which tracked the legs to 8-11 deg mean error. What is NOT checked is
which substitution the policy makes: fully extending the knee (locked leg, no
compliance at contact) would matter, shifting the pelvis would not.

**Probably the ones that matter: ZMP and sole tilt.** 33% of frames with the ZMP
outside the support polygon, and a foot asked to carry load at 24.5 deg of tilt, are
about whether the robot can stay balanced on a real floor rather than whether it can
hit a pose exactly.

**And the clip is demonstrably trainable.** v17 trained on it and produced
`20260827_115642`, the first hardware run of any box policy to play a motion end to
end. So none of the above prevents learning. Whether any of it is why the robot
still needs catching by hand is untested.

## Separately, from the hardware logs

On that complete run, measured torque against actuator limits:

| joint | limit | peak | |
|---|---|---|---|
| hip_pitch | 120 | 73.7 | 61% |
| knee | 120 | 106.8 | 89% |
| waist_pitch | 48 | 38.9 | 81% |
| **right ankle_pitch** | **36** | **40.5** | **112%**, over 90% for 3.5% of ticks |
| left ankle_roll | 24 | 24.1 | 100% |

ankle_pitch is the only joint over its torque limit, in 4 of 7 engaged v17 runs (up
to 46.0 N-m, 128%, at gain 1.1). The refined clip takes ankle_pitch to -46.0 deg,
exactly its lower stop, against -31.1 in the raw retarget -- so a refinement pass
moved it there, and that is a plausible cause worth checking.
