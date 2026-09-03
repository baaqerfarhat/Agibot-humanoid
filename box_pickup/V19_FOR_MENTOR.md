# v19: what changed, and what it did and did not fix

Follow-up to `WHAT_WE_NEED_FROM_YOU.md` / `ANSWERS_WHAT_WE_NEED_FROM_YOU.md`.

Short version: the reference's infeasible single-support spells were the target, the
clip now asks for meaningfully less of them, and every sim metric improved — ankle
saturation roughly halved. **On hardware it made no difference: v19 completed 0 of 7
runs against v18's 2 of 9.** Section 5 has the numbers and our reading of why the sim
improvement did not transfer, which is that chatter, not ankle saturation, appears to
govern survival, and the two policies are identical on it.

## 1. The artifacts

| what | where |
|---|---|
| v19 checkpoint (shipped) | `box_pickup/policy/x2_box_walk_retimed_v19_model_85500.pt` |
| v19 run config | `box_pickup/policy/x2_box_walk_retimed_v19_holosoma_config.yaml` |
| v19 deployable npz | `box_pickup/policy/x2_box_policy_walk_retimed_v19_iter85500.npz` |
| v19 sim rollout, iter 85500 | `box_pickup/sim_rollouts/x2_box_walk_retimed_v19_iter85500_rollout.npz` |
| v19 sim rollout, iter 89000 | `box_pickup/sim_rollouts/x2_box_walk_retimed_v19_iter89000_rollout.npz` |
| retimed reference clip | `box_pickup/holosoma_overlay/src/holosoma/holosoma/data/motions/x2_31dof/whole_body_tracking/sub3_largebox_003_walk_feasible.npz` |
| the retiming pass | `box_pickup/retime_gait_for_feasibility.py` |
| what we ruled out, and why | `box_pickup/add_lateral_weight_shift.py` |
| on-robot runner | `agibot_control_functions/run_x2_box_v19.sh` |

v17 and v18 checkpoints, configs and rollouts are where they were. Run v19 the same
way, but **against the retimed clip** — it is not the same reference v18 trained on:

    python eval_record_driver.py box_pickup/policy/x2_box_walk_retimed_v19_model_85500.pt \
        out.npz 590 demo <motions>/sub3_largebox_003_walk_feasible.npz

The pre-retiming clip, if you want the v17/v18 reference for comparison, is the same
path at commit `d0f6241^` (`d0f6241` is the retiming).

## 2. What was actually wrong with the clip

Not the stance width, which is a reasonable 184–240 mm through the walk. The sway.

Over all 199 single-support frames the stance ankle sits **99 mm** off the ankle
midline and the CoM moves **7 mm** towards it. The lateral motion is not missing —
it is 78 mm, roughly the right size — it is *out of phase*. Correlation between the
CoM's side and the stance foot's side is **+0.29**, and the CoM is on the **wrong
side entirely for 98 of the 199 frames**. The robot is asked to stand on one leg with
its weight over the other one. That is where the 127 N·m came from.

Our reading is that this is a retargeting artifact: the human's sway survived the
transfer, the foot timing did not, and the two desynchronised.

## 3. What v19 does about it

Since the sway already exists at roughly the right amplitude, v19 does not try to
move the body. Each of the six swings is **shortened** and **slid onto the sub-window
where the CoM is genuinely nearest the foot about to carry**.

| swing | was | now | mean \|CoM − stance\| |
|---|---|---|---|
| L | f179–214 (36f) | f167–186 (20f) | 190.3 → 60.6 mm |
| R | f217–255 (39f) | f221–241 (21f) | 99.4 → 50.7 mm |
| L | f252–303 (52f) | f256–284 (29f) | 108.1 → 63.9 mm |
| R | f311–334 (24f) | f316–329 (14f) | 44.5 → 25.9 mm |
| L | f333–368 (36f) | f348–367 (20f) | 37.6 → 29.6 mm |
| L | f467–490 (24f) | f460–473 (14f) | 159.3 → 109.4 mm |

Clip-level: frames over the 24 N·m ankle_roll limit **134 → 94**, mean moment
**44.4 → 31.2 N·m**, peak **127 → 88**, single support **199 → 168 frames**.

Footfalls, root and box are untouched — each foot still lands exactly where it landed,
so the grasp and the pickup are unchanged. Feet hold to 1.7 mm and peak joint jerk is
unchanged at 5451 rad/s³.

## 4. What it did to the policy

Same rollout protocol as before (`demo`, clean, from t=0):

| | v17 iter49000 | v18 iter81499 | v19 iter85500 |
|---|---|---|---|
| ankle_roll at effort limit | — | 64% / 70% | **53% / 37%** |
| ankle_pitch at effort limit | — | 51% / 31% | **35% / 21%** |
| box lift | — | 608 mm | 621 mm |
| waist error over pickup | — | 1.6° | 1.5° |
| leg target chatter, RMS | — | 33.1 mrad | 34.7 mrad |

Total time on the ankle stops, summed over all four joints: **v18 216 points, v19 146**.
We shipped iter 85500 rather than the newest because reward plateaued at ~82 from
81.5k on and 85500 is the only checkpoint better than v18 on all four ankle joints
(85500 / 87500 / 89000 score 146 / 197 / 172).

## 5. The uncomfortable part: it refuses harder

Your hypothesis still holds, and v19 sits further along it rather than off it.

| | double | single | airborne |
|---|---|---|---|
| reference (v17/v18) | 386 | 199 | 6 |
| reference (retimed, v19) | 418 | 168 | 5 |
| v17 sim | 545 | 45 | 0 |
| v18 sim | 537 | 53 | 0 |
| **v19 sim** | **561** | **29** | **0** |

Against each policy's own reference schedule:

| | reference single | sim follows | sim refuses |
|---|---|---|---|
| v17 | 199 | 40 (20%) | 159 (80%) |
| v18 | 199 | 45 (23%) | 154 (77%) |
| **v19** | **168** | **27 (16%)** | **141 (84%)** |

And the foot-asymmetry metric that ordered your hardware runs, over frames 179–250:

    v19 sim              8.3 mm
    v18 sim              9.6 mm
    v17 sim              9.5 mm
    reference (retimed) 13.5 mm
    reference (old)     18.4 mm
    hw completed        17.5 mm
    hw fell        29.7 - 43.3 mm

So v19 is the most conservative policy of the three by every one of these measures.
On the monotonic line you found — less single support, more survival — that predicts
it should survive hardware better than v18.

**It did not, and the hardware runs in `96c1448` settle it: v19 completed 0 of 7,
v18 completed 2 of 9.** Best v19 was 655 of the 741 ticks a full clip needs, at gain
0.95. So the prediction above is falsified, and the survival line does not extend from
sim onto hardware the way the sim-internal ordering suggested it would.

The chatter numbers say why the sim comparison was not predictive. Same metric, same
twelve leg joints, hardware against sim:

| | sim | hardware, mean over runs | ratio |
|---|---|---|---|
| v18 | 33.1 mrad | 49.7 mrad (9 runs) | 1.50× |
| v19 | 34.7 mrad | 52.3 mrad (6 runs) | 1.51× |

Reversal rate is 22.0% on hardware for **both**. The two policies are
indistinguishable on the axis that appears to govern whether a run survives, and v19
is very slightly worse on it — which tracks the sim ordering (34.7 against 33.1) far
better than the ankle duty cycle does. Halving ankle saturation bought nothing
measurable on hardware; the ~1.5× chatter amplification did not change either.

**But it is still not doing the motion**, and it is doing less of it than v18 was. We
are not claiming the walk is fixed. The reference now demands less of what cannot be
done, the policy has taken the slack as more shuffle, and on hardware that was not
worth anything.

## 6. What we ruled out, with numbers

Moving the CoM instead of retiming the feet **cannot work**, and we would rather you
not spend time on it. With the sole held flat the ±15° ankle caps the pelvis at about
**161 mm** to the side of the foot it stands on, and a wrong-side CoM has further than
that to travel. Three attempts, all in `add_lateral_weight_shift.py`:

- uncapped — fixed the moment properly (67% → 31% of frames over limit) but dragged
  the feet **112 mm**, put in **15× the jerk** and pushed the soles 21 mm through the
  floor;
- capped at the reachable 115 mm — feet held, jerk fine, but bought only 11 of the 67
  points and introduced **4.6 g** of lateral acceleration;
- stacked on top of the retimed clip — **58% → 58%**, nothing at all.

Foot placement and CoM trajectory are coupled through the ankle range. You cannot fix
one with the other held fixed.

## 7. What we think is actually left

The remaining 94 frames need the **footsteps replanned**, not another correction pass
over the existing ones. The binding constraint is step length: the clip steps **746
and 812 mm on a 600 mm leg**, and two feet that far apart cannot both be planted, so
double support cannot be extended across those strides no matter how the swings are
timed. That is what limited the retiming — 82 of its frames had to take a partial
target for exactly this reason.

Shorter, more numerous steps over the same 1.53 m would let the CoM stay inside a
support polygon that actually exists, and would let us plan the sway and the footfalls
together instead of fitting one to the other. Before we commit to that, two questions:

1. Does a shorter-stride walk still serve the demo, or does the stride length matter
   to what you want to show?
2. Is it worth keeping the reference's original footfalls at all at this point, or
   should we plan the walk from scratch and keep only the pickup and set-down from
   the mocap?

## 8. Also fixed, and worth knowing

`penalty_joint_torque_saturation` had **never applied**. Its ramp counter restarts on
a warm start, so across all of v18 it sat at **0.0026 of its full 8.85 value** while
saturated ankles were the failure mode — it was worth −0.0178 of a ~65 point reward.
It is now weight −5.0 with a 4k ramp. If you warm start anything with a ramped term,
this is worth checking on your side too.

One caveat on tooling: `export_and_verify_waist.py` prints `FAIL — do not deploy` for
v19, as it did for v17 and v18. It is a false alarm. The gate is open-loop with
perfect observations; in the closed-loop rollout v19 tracks waist_pitch to **1.5°
mean error with 0 of 304 frames inverted**, marginally better than v18's 1.6°. The
gate wants recalibrating for the walking clip and we have not done it.
