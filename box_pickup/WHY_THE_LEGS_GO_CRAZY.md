# Why the legs go crazy after the robot comes back up

Question asked: after picking the box and standing back up, the robot starts wild leg
and foot movement. Is it trying to recover its CoM, and is that why it falls?

**Answer: no.** It is not recovering anything. Measured across the Sep 2 v19 runs
(7 engaged, 0 completed) with the Aug 31 v18 runs (2 completed) as the contrast, the
"crazy legs" are two distinct mechanisms, and neither is CoM recovery. The ankle_roll
actuator -- the joint that regulates lateral CoM -- is at **6-10 N-m of its 24 N-m
limit during every CoM excursion in every run**, completing or not. The policy never
tries. Everything below is from the hardware logs, his committed sim rollouts, and two
new 741-step sim rollouts recorded on this machine.

## Mechanism 1 -- runs that die in the walk right after the lift (174124, 174523)

The reference's walk spell starts at f313. In 174124:

    frames   ref CoM off   hw CoM off   foot asym   roll    ankle_roll used
    313-320      67->41        8-9         7-11      -3..-5      13
    321-336      11-25        44->128     12-15      -5..-3      8-11     <- CoM leaves the polygon, feet still down
    337-348      55-75        111->64     22->81      4->15      3-5      <- THEN the scheduled foot lift
    353-368       9-40        69-88       8-28       20-22       8-15     <- leaning 20 deg, foot back down
    369-380      44           52-37      116->430    27->33               <- topple

The CoM leaves the support polygon at 321-336 **while the reference's own CoM is
inside** (11-25 mm) and **both feet are still down**, with ankle_roll at 8-11 N-m.
Then the foot lifts on schedule onto that excursion. This is not tracking the
reference into a fall -- the reference is in. It is a hardware lateral sway that lags
and overshoots the reference, unregulated.

Sim does the same commitment and survives. First foot lift in the spell, CoM offset at
that moment: sim v19 81 mm, sim v18 56 mm, hardware 174124 77-128 mm. What differs
is the recovery afterwards:

    case                  CoM peak   back under 58 mm   |roll| peak
    sim v19               106 mm     never (stays out)    6.9 deg
    sim v18               224 mm     never (stays out)   13.5 deg
    hw 174701 (survived)  162 mm     24 frames, 0.48 s    9.9 deg
    hw 174124 (fell)      137 mm     35 frames, 0.70 s   32.8 deg

Same commitment, 3x the roll. The sim keeps the CoM past the ankle-only bound for most
of the walk and is fine (the bound is for the ankle alone; hips carry it). The
hardware run that fell rolled to 33 deg on the same excursion. That is the residual
dynamic: an under-damped lateral body mode that sim does not have. Sim CoM vs
reference over the walk: gain 1.20x, lag 0.22 s, corr +0.89. Hardware: gain up to
3.65x, lag up to 0.78 s.

Being past the ankle bound is normal, not fatal -- the survivor 174513 spent 84% of the
walk beyond it (mean 102 mm); 174124 died at 38% (mean 44 mm). And the CoM crosses from
40 mm to past the bound in **1 frame (0.02 s)**, so a robot-state-triggered ankle law
has no lead time to act on.

## Mechanism 2 -- the run that dies after standing back up (174701, f653)

174701 survived the walk (CoM back in 0.48 s, roll 10 deg), survived the set-down
(CoM 45 mm, pitch range 7 deg -- the calmest phase of any run), stood back up, and
died in the **end hold** -- after the clip has finished at f591, where the deploy
freezes the final frame for 150 ticks and the reference is standing still.

    frames    |dtgt| mrad   reversal   worst joint p-p    gyro
    572-583     53-168        --                          0.17-0.45
    584-595     285-251                                   1.1-1.8      <- onset at the boundary
    592-595     527           right_hip_roll +11.3 deg/tick   2.0
    596-607     503-683       left_knee +10.7, hip_yaw +9.5   3.3-3.5   <- falling
    620-639                   foot asym 380-411 mm

Ten-degree **single-tick** target jumps, reversal rate only 8-21% -- not chatter, a
large discrete policy output at the moment the reference freezes. The CoM was
**inside the bound** (36-46 mm) the whole time. Nothing to recover from.

Why: the frozen hold is a regime the policy has never seen.

- Training: `max_episode_length_s = 12.0` = 600 ticks against a 591-frame clip, and
  at the clip end the motion command calls `self.reset(ended_env_ids)` -- the robot
  is **teleported to a fresh clip start** (BeyondMimic resample). Never a hold.
- Sim evaluation: same. The two 741-step rollouts recorded here show a root position
  jump of **1.46 m (v19) and 2.32 m (v18) at f589->590**. Sim never holds either.
- Deploy: `frame = min(frame, T-1)` and feeds `ref_joint_vel[590]` (0.47 deg/s) for
  150 ticks. This regime exists **only on the robot**.

The v18 survivors hit the same boundary: 174433 produced a +48.7 deg single-tick knee
jump at 560-563 and +11.8 deg hip_roll at 592-595, gyro 2.4 rad/s -- and settled by
~620. Both policies jump; v18 damped out, v19 escalated. It is not entry state --
v19 arrived *calmer* (leg vel 50 vs 73 deg/s, gyro 1.8 vs 2.6, tracking error 32 vs
59 deg). And it is not a missing hip strategy -- v19 used the *most* torque in the
recovery window: hip_roll 114 N-m (95%), waist_pitch 47 (98%), knee 110 (92%),
against 27-58% for the survivors. It thrashed harder and toppled.

The squat policy (4/4 complete) shows nothing at its own freeze -- 19 mrad, 0.2 deg
jumps -- because it is a phase-conditioned cycle whose hold pose is the stand it spends
the last second of every episode in. The walk policy's hold pose is one it was
teleported away from every episode.

## What this says about fixing it

- **Mechanism 2 is a deploy/training mismatch, not a control problem.** Either train
  the hold (`enable_default_pose_append` or a longer episode with the clip clamped
  instead of resampled) or don't hold on deploy (blend into the policy's standing
  behaviour or end the run at f591). Sim cannot test any fix for it as things stand,
  because sim teleports.
- **Mechanism 1 is a hardware lateral mode the sim does not reproduce.** A CoM /
  capture-point law has the authority (13-16 N-m of unused ankle_roll, hips at ~50%)
  but no sim-based test can show benefit, because sim never produces the excursion.
  That is a specific argument for hardware-in-the-loop adaptation -- and it is where
  the ComSlidingAdapter (a capture-point regulator; lambda=3 vs omega=sqrt(g/z)=3.9)
  would have to be evaluated.
- **No scalar trigger has survived validation.** ankle_pitch at window entry, lateral
  CoM margin, and lateral sway gain each separated completed from failed runs on
  n<=6 in one window and failed on a second window or a second policy generation.
  The one-frame crossing time above is why: there is nothing to trigger on.

## Sim-to-real, restated

The robot does not do this task in sim either -- sim keeps both feet down on 77-80% of
the frames where the reference asks for single support. Where it does commit, sim and
hardware commit alike; hardware recovers worse (roll 33 vs 7 deg) and, uniquely, is
asked to hold a frozen reference it was never trained on.
