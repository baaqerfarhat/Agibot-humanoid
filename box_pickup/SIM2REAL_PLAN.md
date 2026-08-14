# Closing the box-pickup sim-to-real gap

Written after the 2026-08-12 hardware runs, where the torque feed-forward fix worked
(the motors now deliver what training intended, confirmed at 0.98-1.00 correlation) and
the robot still fell while standing back up with the box.

## 1. What the gap actually is

**The policy spends saturated torque on contacts that do not push back on the real robot.**

`action_scale = 0.25*effort_limit/kp`, so an action of 4 already commands the full effort
limit. Training never bounded this (`action_clip_value: 100.0`), so the policy learned to
command 10-40 on the joints that press into something. That only works if the contact
absorbs the excess. Two interfaces, both rigid in sim, neither rigid on the robot:

| interface | sim | hardware |
|---|---|---|
| foot / ground, via `ankle_roll` | joint holds at **+0.03 rad** under 22 Nm | travels to **+0.34 rad**, past its +0.263 URDF stop, and stays there 96% of the motion |
| hand / box, via the wrists | rigid half-sphere hand grips a rigid box | soft hand deforms; `left_wrist_roll` pinned at **-1.56 rad**, its mechanical stop, 78% of the run |

Consequences, measured:

- **No lateral ankle authority during the stand-up.** `right_ankle_roll` reads +0.34 on
  every frame of the rise (frames 120-165). Pelvis roll then grows -0.22 -> -0.40 -> -0.62
  and the run aborts at 2.9 s. That is both unsupported runs.
- **Half the arm load is missing.** Over the rise the arms carry 8.83 Nm in Isaac against
  3.8-5.2 Nm measured on hardware, while the legs carry the same 13-15 Nm in both. The
  force is going missing at the hand, not in the legs.

Ruled out with data, so do not spend time here: the joint-limit table (matches the URDF
exactly on all 31 joints), gyro bias (0.007-0.05 rad/s at rest), and the hands-push-off-
the-ground theory (forward kinematics puts the wrists at box height, 0.17 m, at the
deepest bend and never below 0.387 m during the rise -- they are on the box, never the
floor).

## 2. Where the control authority still is

This decides what any online method can fix. `|action|` during the rise, against the
limit of 4:

| joint | sim | hardware | saturated? |
|---|---|---|---|
| `right_ankle_roll` | 21.05 | 10-30 | **yes, 526% of limit** |
| wrists | 16-24 | 10-40 | **yes** |
| `waist_pitch` | 0.54 | 1.5-2.5 | no, 0% of ticks |
| `waist_roll` / `waist_yaw` | 0.4-0.5 | 0.2-0.8 | no |
| `left_knee` | 0.71 | 1.85 | no |
| `left_hip_pitch` | 2.44 | 2.5-4.8 | partly, 23-65% |

The waist is clean in both, which is what makes adaptation viable at all. Also note the
allocation problem: during the rise the policy puts **526% of the effort limit into one
ankle** and only **16-25% into the hips**, which have 5x the torque (120 vs 24 Nm) and
10x the travel (2.9 vs 0.26 rad).

## 3. Hardware experiments, in order

Each one answers a question that changes what you do next. Run them in this order; later
ones are wasted if the earlier ones fail.

**E1. Ankle-roll torque-vs-angle curve.** Standing, both feet down, ramp commanded
ankle-roll torque 0 -> 24 Nm over 2-3 s, log angle against torque. Repeat suspended. Run
the identical ramp in Isaac.
*Gives:* the foot-ground compliance curve, directly comparable to sim. Loaded vs suspended
separates "weak ground reaction" from "compliant joint / backlash". This is the residual,
measured.

**E2. Static pose torque ID.** Hold frames 0, 80, 120, 160, 420 for 5 s each, log
`eff_meas` on all 31 joints, compare against Isaac in the identical pose.
*Gives:* mass, CoM and inertia error. A systematic leg/waist offset here is a residual
adaptation **can** absorb -- unlike a jammed joint.

**E3. Hand-box coupling.** Hold the box in the mid-pose for 30 s. Log wrist and shoulder
torque and whether the box sags or slips.
*Gives:* how much commanded grip becomes real grip. Confirms the 8.83 -> 4 Nm arm deficit
and sizes the compliance for E6.

**E4. Weigh the box.** Training randomized 0.3-2.0 kg. Outside that range is a large
residual with a trivial fix.

**E5. Action-clip A/B.** `--action-clip 4` against `--action-clip 0`, 3+ trials each, same
box and stance. Check whether `right_ankle_roll` still pins at +0.34 through the rise.
*This is the next run.* Everything downstream depends on the ankle being unjammed.

## 4. Retrain changes

**Bound the actions.** `action_clip_value: 100.0 -> 4.0` in the training config. One line,
and it is the root cause: it stops the policy learning any strategy that needs torque the
actuator cannot make. Isaac, 7 seeds, clipping the ankle rolls and wrists at inference:
survival unchanged from baseline on 7/7. Clipping every joint fails on 0/9, so if you
instead bound it during training, watch that the legs keep their range.

**Model the soft hand.** Training uses `x2_31dof_w_object_halfspherehand.urdf`, a rigid
half-sphere, and the object randomization covers mass, friction and restitution but
**nothing randomizes contact compliance**. Every episode saw a perfectly rigid grip. Add
contact stiffness/damping on the hand geoms, sized from E3, and randomize it. This is the
hand-box half of the gap.

**Fix the lateral allocation.** Penalize `ankle_roll` action magnitude and reward hip-roll
lateral correction, so the stand-up does not depend on the weakest, shortest-range joint
in the chain.

**Keep the margin honest.** Even in sim this motion peaks at 22 deg lateral lean against a
40 deg abort. Train with lateral pushes during the bend-and-rise specifically, not just
the generic push randomizer.

## 5. Where adaptation fits

Grounded in `../adaptation/README.md` plus the run with the clip enabled (5 seeds:
frozen 10.97 s, waist mask 10.60 s, paper default 2.53 s).

- **It cannot fix a jammed joint.** The adapter's only output is a change to the action,
  and on a joint already past its effort limit that changes the delivered torque by zero.
  Do E5 first or the ankle and wrist channels are dead to it.
- **It can plausibly fix a load residual.** If E2 shows a systematic leg/waist torque
  offset, that is exactly the case where adaptation won under an actuator fault (+53%
  survival, p = 0.0011). The waist is unsaturated in both sim and hardware, so `--mask
  waist` has a real control channel.
- **It is not a lateral balance mechanism.** The ankles have to stay out of the mask -- in
  sim, including them puts the robot down in 2.2 s -- and the ankle is what failed.
- **On hardware:** `--gain 0` first to confirm the adapted path reproduces the frozen
  deploy, then `--mask waist --gain 1e-5`, 3+ trials each, compare with
  `compare_adapt_runs.py`. Read the drift log, not just the outcome: climbing into the
  `||W-W0|| < 5` bound means it is fighting something it cannot fix; converging and
  plateauing means it found a real compensation.

## 6. Order of work

1. E5, the action-clip A/B. Cheapest, and it unblocks everything.
2. E1 and E2 while the robot is set up. Static, safe, and they tell you which residual you
   have.
3. Retrain with the bounded actions and the compliant hand. E3 sizes the compliance.
4. Adaptation A/B last, on a policy that already completes the task unaided.
