# The ankle_roll scale is sized to the wrong clip

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

## Fix

`action_scale_overrides={"ankle_roll": 0.06}` — puts +-15 deg back within |a| ~ 4.4.
That is also what `cfg * effort / kp` gives, so the override becomes documentation:
this number is derived from the clip and must be re-derived whenever the reference
changes.

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
