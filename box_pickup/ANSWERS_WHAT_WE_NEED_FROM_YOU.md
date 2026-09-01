# Answers to WHAT_WE_NEED_FROM_YOU.md

Everything asked for is in this commit, and the question it was wanted for is already
answered by the rollouts that were on disk. **Your hypothesis is right**, and the sim
numbers sit on the same monotonic line as the hardware ones rather than off to one
side.

## 1. The artifacts

| what | where |
|---|---|
| v18 checkpoint | `box_pickup/policy/x2_box_ankle_scale_v18_model_81499.pt` |
| v18 run config | `box_pickup/policy/x2_box_ankle_scale_v18_holosoma_config.yaml` |
| v17 checkpoint | `box_pickup/policy/x2_box_walk_feasible_v17_model_49000.pt` |
| v17 run config | `box_pickup/policy/x2_box_walk_feasible_v17_holosoma_config.yaml` |
| v18 sim rollout | `box_pickup/sim_rollouts/x2_box_ankle_scale_v18_iter81499_rollout.npz` |
| v17 sim rollout | `box_pickup/sim_rollouts/x2_box_walk_feasible_v17_iter49000_rollout.npz` |

`.pt` is gitignored except under `box_pickup/policy/`, which is why they are named
rather than foldered. Both are 6.97 MB. Run them the way you described:

    python eval_record_driver.py box_pickup/policy/x2_box_ankle_scale_v18_model_81499.pt \
        out.npz 590 demo <motions>/sub3_largebox_003_walk_feasible.npz

The two rollout npzs are the ones behind the duty-cycle numbers in 2cd5b81, recorded
in `demo` mode (clean, from t=0, no noise, no early termination). They carry
`dof_pos_target`, `dof_pos`, `dof_vel`, `torques`, `torques_substep`, `actions`,
`body_pos_w`, `body_quat_xyzw`, `object_pos` and the joint metadata including effort
limits, which is every channel your comparison list needs.

## 2. Does sim follow the reference into single support?

**No. It refuses harder than the hardware run that completed.**

Support state over the clip, sole below 20 mm counting as loaded:

| | double | single | airborne | foot asymmetry, frames 179-250 |
|---|---|---|---|---|
| reference | 386 | 199 | 6 | 18.6 mm |
| **v18 sim** | **537** | **53** | **0** | **9.7 mm** |
| **v17 sim** | **545** | **45** | **0** | **9.5 mm** |
| hardware, completed | | | | 17.5 mm |
| hardware, fell | | | | 29.7 and 43.3 mm |

Put in order, the relationship you found on hardware extends cleanly onto sim:

    sim            9.5 - 9.7 mm    completes
    hw completed      17.5 mm      completes
    hw fell        29.7 - 43.3 mm  falls

The more of the reference's single support the robot actually performs, the sooner it
goes down, and sim survives because it performs the least of it. Against the
reference's own schedule, sim keeps **both feet down on 77% (v18) and 80% (v17) of the
frames where the reference is in single support**, following it in on only 23% and 20%.

So the answer to "why does sim work" is that **sim is not doing the motion.** It has
learned a both-feet-down shuffle that eats the tracking error rather than enter a
phase it cannot survive. That is the same glued-foot behaviour the `com_support_margin`
docstring warns about as the thing not to pay the policy for; it turns out the policy
found it anyway, through the tracking reward, without being paid.

This resolves your fork in favour of the first branch, with a caveat. Sim and hardware
are doing the same thing, and the difference is how reliably the improvisation holds
— hardware has to improvise against disturbance, sensor noise and a real contact patch,
and 7 of 9 runs it did not hold. The gap is **not** the contact model: sim is not
producing restoring moment the real foot cannot, because sim is not putting itself in
the position of needing it.

The caveat is that where sim *does* enter single support, it is in the same trouble the
reference is:

| | single-support frames | lateral CoM offset from stance ankle | implied moment | over 24 N-m |
|---|---|---|---|---|
| v18 sim | 53 | mean 115 mm, max 288 mm | mean 48, max 121 N-m | 68% of them |
| v17 sim | 45 | mean 132 mm, max 299 mm | mean 56, max 126 N-m | 78% of them |

Same 300 mm and same ~127 N-m as the reference. Sim has not found a feasible way to do
single support; it has found a way to mostly not do it.

## 3. Ankle duty cycle, one source for the numbers

Substeps at or above 99% of the effort limit, from the two rollouts above:

| | ankle_roll L/R | ankle_pitch L/R |
|---|---|---|
| v17 i49000 | 64% / 71% | 34% / 26% |
| v18 i81499 | 64% / 70% | 51% / 31% |

Reproduce with `box_pickup/compare_sim_single_support.py`, which prints the support
table, the schedule agreement, the CoM offsets and this table from the committed
rollouts. Hips and knees are at 0% and peak at 47-62% of a 120 N-m limit.

## 4. On the two negative results

Agreed, and the sim numbers say why no scalar threshold works. Both of the candidates
are measured **during** single support, but the thing that separates the runs is *how
much single support happens at all*. A run that mostly stays on two feet never
generates the values either statistic is looking for, so both end up reading the
consequence of the choice rather than the choice.

If a triggered signal is still wanted, foot height asymmetry is the one that ordered
all five cases here correctly, and it is available before the ankle is in trouble
rather than during. It is a proxy for "am I about to commit to a phase I cannot
survive", which is the actual decision. It is one window on five samples, so it is a
hypothesis and not a result.

## 5. Agreements and one correction

Agreed and independently confirmed:

- the 127.1 N-m against a 21.1 N-m foot bound, which you reproduced with
  `check_ankle_roll_feasible.py`. Rebuilding the four spells is the fix.
- your retraction in 09b31cb. `diagnose_clip` reports zero joint position and velocity
  violations on the refined clip, and I get the same: nothing exceeds a limit. The
  clip is kinematically feasible and dynamically not, which are different claims and
  worth keeping apart.
- `X2_WS` over the hardcoded `WS` path. That was mine and it should never have been
  written that way.

One correction, on `fb9c333`, "the reference commands the ankles into their stops --
do not raise the scale". The premise is right and I recorded the same thing in
`128244e`: +-15.0 is the mechanical stop, and the refined clip sits within 0.5 deg of
it on 6.8% and 10.8% of frames. But the conclusion does not follow, for two reasons.

First, deploy already clamps the position target to the joint limit and passes the
remainder as a feedforward torque capped at the effort limit
(`deploy_x2_box_pickup.py`, `build_area_cmd`, line 311). An out-of-limit command becomes a
saturated torque, not a joint driven into a hard stop, so the hardware risk the
sentence is guarding against is already handled and is handled identically at 0.02.

Second, and the reason the scale is not the lever either way: raising it 3x changed the
ankle duty cycle by nothing, 64%/71% to 64%/70%. Scale sets what can be commanded; the
duty cycle is set by what the foot can hold once it is loaded. Both values leave the
ankle saturated for most of the run, so this is not a decision that has to be got right
before the reference is fixed — 0.06 tracks more of the reference range (0.41/0.53
against 0.35/0.46) and is what shipped, but reverting it would cost little.

## 6. What is left

The four spells to rebuild, with their peak demand:

    t 3.58 - 4.28 s   127.1 N-m   5.3x
    t 4.34 - 5.02 s    85.5 N-m   3.6x
    t 5.12 - 6.06 s    97.6 N-m   4.1x
    t 9.34 - 9.80 s   102.8 N-m   4.3x

Two levers, and they multiply rather than add. The stance is 42-48 cm wide while the
hips are 29 cm apart, so each leg is splayed and the CoM starts far from either foot;
narrowing it shortens the distance the weight has to travel. Then the lateral weight
shift over each stance foot, which the retarget dropped, has to be put back — that is
what makes the CoM arrive over the foot before the other one leaves the ground.

Worth deciding before that work starts: it changes the walk enough that warm-starting
may not be worth it, and the in-place clip is a feasible fallback that already ran
clean on hardware if a demo is needed before the walk is ready.
