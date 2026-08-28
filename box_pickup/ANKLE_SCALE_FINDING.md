# The reference commands the ankles into their joint stops

**Do not retrain against this clip, and do not raise the ankle_roll scale.** The
earlier version of this document recommended `action_scale_overrides={"ankle_roll":
0.06}` so the policy could follow the reference. That was wrong: the reference is
asking for angles the robot physically cannot reach, so no scale makes it trackable.

`ankle_roll`'s joint limit is +-15.0 deg. Frames sitting at a stop:

| clip | joint | limit | at the stop | span |
|---|---|---|---|---|
| raw retarget | left_ankle_roll | +-15 | **49.5%** at +15 | +2.4 .. +15.0 |
| raw retarget | right_ankle_roll | +-15 | **40.6%** at -15 | -15.0 .. +1.4 |
| refined | left_ankle_roll | +-15 | 6.8% at +15 | -8.7 .. +15.0 |
| refined | right_ankle_roll | +-15 | 5.4% / 5.4% both | -15.0 .. +15.0 |
| refined | left_ankle_pitch | -46..+26 | 1.7% at **-46** | -46.0 .. -2.3 |
| refined | right_hip_roll | -13.5..+166 | **16.9% past -13.5** | **-21.9** .. +6.7 |

In the raw retarget both ankles are rolled to one side and held there — left never
negative, right never positive, median |roll| 14.5 and 13.3 deg. That is a static
splay from mapping human ankle geometry onto the X2 foot, not gait. A squat-lift
does not need 30 deg of ankle roll; the question that exposed this was simply "why
would it?".

The refinement passes reduce it (median 14.5 -> 8.1 deg, time at the stop 49.5% ->
6.8%) but do not remove it, and they make ankle_pitch worse: raw span -31.1 deg,
refined -46.0, which is exactly its lower stop. That is the likely cause of the one
joint that exceeds its torque limit on hardware -- ankle_pitch at 112% in the
complete run -- since the PD is pushing against a mechanical stop.

**The fix belongs in the retargeting**, not in the action scale and not in the
reward. Until the clip keeps the ankles inside their travel, a policy trained on it
is being asked to track the impossible.

The rest of this document records how the shortfall was found and is still accurate
as measurement; the recommendation in "Two candidate fixes" is superseded by the
above.

---


**Finding:** `ankle_roll`'s action scale of 0.02 rad/unit was derived from the
in-place reference. The walking reference needs three times that range, so on
hardware the policy cannot roll its ankles far enough to shift its weight — which
is why the feet do not step and the robot has to be held up.

Measured on `20260827_115642` (v17 iter49000, gain 0.9), the run that played to the
end, over the 591 frames inside the clip.

## The arithmetic

`commanded angle = action x scale`, and `scale = 0.02` for ankle_roll.

| | in-place clip (what 0.02 was sized on) | walking clip (what ships now) |
|---|---|---|
| ankle_roll reference span | −5.2 .. +4.2 deg | **−15.0 .. +15.0 deg** |
| action needed for full span | ~4.5 | **13.1** |

The policy's action does not reach 13. It pushed |a| to 12.4 (mean 5.0) and still
fell short.

## What that does on the robot

Range of motion over the clip, measured / reference:

| joint | ref range | measured | ratio |
|---|---|---|---|
| left_hip_pitch | 172.5 deg | 146.9 | 0.85 |
| right_hip_pitch | 148.6 | 134.6 | 0.91 |
| left_knee | 119.9 | 116.5 | 0.97 |
| right_knee | 116.3 | 113.6 | 0.98 |
| left_ankle_pitch | 43.7 | 34.4 | 0.79 |
| **left_ankle_roll** | **23.7** | **17.6** | **0.74** |
| **right_ankle_roll** | **30.0** | **19.6** | **0.65** |

The ankles are the only joints that cannot follow. Commanded range is short too
(20.1 / 19.3 deg against 23.7 / 30.0 asked), so this is the scale binding, not the
policy choosing not to move.

`ankle_roll` is the lateral centre-of-pressure joint: it is how the robot rolls onto
one foot to unload the other. Capped, it cannot make that shift, so it cannot step
to catch itself. Reported from the robot: the feet were not moving to keep it
stable, and it fell whenever it was not held.

## It is the command that is short, not the torque

Ruled out on the same run. Over the ticks where the robot is more than 5 deg off the
reference (64% of the clip left, 69% right):

| | left | right |
|---|---|---|
| the COMMAND was also off-reference | 93.9% | 92.9% |
| torque >= 90% of the 24 N-m limit | 0.5% | 0.2% |
| mean torque during the shortfall | 5.5 N-m | 5.4 N-m (of 24) |

The ankle was using 23% of its torque while failing to reach the reference. It was
not saturated — the policy was not asking for the angle. Reaching the reference
needs |a| = 13.1; the policy output up to 12.4 right and 10.6 left, far out in a
distribution that normally sits near +-3-4, and still fell short.

## Two candidate fixes, and the choice is not settled

**1. Raise the scale to 0.06.** Puts +-15 deg back within |a| ~ 4.4. That is also
what `cfg * effort / kp` gives, so the override becomes documentation: the number is
derived from the clip and must be re-derived whenever the reference changes.

**REQUIRES RETRAINING.** The scale is part of the interface the policy learned
against: a policy trained at 0.02 that is deployed at 0.06 commands 3x the ankle
angle it intends. This cannot be applied to v17 or any existing checkpoint — it only
takes effect for a policy trained with the new value. A warm start may not absorb it
either; changing kd by 3x (a comparable interface change) left a warm-started policy
unable to recover the task in 3000 iterations.

**2. Question the reference instead.** The clip asks for a median |ankle_roll| of
8.1/7.7 deg, with 34-36% of frames above 10 deg and 25% above 12, in four blocks
spread through the motion. For a squat-lift-carry that is high — a human uses a few
degrees, and walking is typically 5-10. The recent passes manipulate the feet
directly (`seat_opening_frames`, the foot-box clearance passes, `level the feet
individually`, `balance_opening_stance`), and ankle_roll is exactly the joint they
move. The 30 deg span may be an artifact of levelling a retargeted human foot onto
robot foot geometry rather than motion the task needs.

If it is an artifact, fixing the reference is the real fix and raising the scale
just trains a policy to chase it. The check is what the source motion asks versus
what the refinement passes added.

## Why 0.02 was there, and what to watch

It stopped a real exploit: the policy commanded a sustained +35 deg ankle offset the
foot could not follow, pinning the 24 N-m actuator for 99% of the episode. Three
reward penalties failed to price it out (-0.05 gated, -0.01 ungated, -0.10
position-only); removing the authority ended it at once.

But that was measured on the in-place clip, where holding a constant offset cost no
tracking. A walking clip should remove the incentive, since the ankle genuinely has
to move. The check is the command gap (mean |commanded − achieved| on ankle_roll):

    exploited   33.3 deg
    capped       4.6 deg

If it grows after this change, tighten the scale only as far as the reference needs
— not back to a value from a different motion.

## Still open, same family

`ankle_pitch` exceeded its 36 N-m limit on hardware in 4 of 7 engaged v17 runs
(38.4, 40.5, 38.8, and 46.0 N-m at gain 1.1 = 128%), and is the only joint over
limit in the complete run. It has no scale override. Hips and knees stayed at
42-89% throughout, so the legs remain the joints with margin.
