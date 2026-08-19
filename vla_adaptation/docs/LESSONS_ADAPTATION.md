# Lessons learned — online adaptation of a learned locomotion policy

Working notes, 11–14 August 2026. AgiBot X2, 20 actuated DoF, whole-body tracking policy,
MuJoCo/mjlab. Written to stop the same mistakes being repeated; every claim below is tied to a
measurement, and the retracted ones are kept with their retractions.

**Start at §8** if you want the frame the rest of this file is evidence for; §1–§7 are chronological.

---

## 1. What actually works

**Downhill (+0.262 rad, 15°), position channel.** Bounded Lyapunov adaptation of one layer
(`mlp.6`, 2,580 of 249,236 parameters = 1.0%) extends reference tracking by **+56.8 control
steps** over a displacement-matched random direction, on a disjoint set of trials never used for
tuning, pre-registered, exact sign test **p = 0.0156** (7 positive, 1 tie, 0 negative).

Three settings were load-bearing, each measured rather than assumed:

| change | from → to | effect |
|---|---|---|
| adaptation gain | 1e-3 → **3e-5** | +23.2 → +51.6 |
| function class | bias-only → full weight matrix | 2.2× the constant-offset ceiling |
| trigger | gated → continuous | the 50-step confirmation cannot act before failure at ~140 |

It is **not** reproducible by retuning a scalar: the best static servo-stiffness change reaches
216.0 steps against adaptation's ~231.

---

## 2. What does not work, and why

**Uphill, ~18 interventions, all null or negative.** Gradient fidelity (L1/L3), adaptation gain
(4 values), function class (bias-only), trigger timing, shift magnitude (8 values), commanded
speed (3), objective family (2), regulated quantity (7 variants), impedance channel, command
channel.

**The mechanism.** Uphill is a **servo-stiffness mismatch**, not an adaptation failure:

```
kp     flat    uphill -0.13   downhill +0.262
1.00   250.0      173.7           185.2      nominal
0.85   250.0      215.8           216.0
0.70   250.0    * 250.0 *         160.3      fixes uphill, HURTS downhill
0.53    54.0    * 250.0 *          34.8      destroys flat AND downhill
```

The frozen policy recovers the entire uphill loss at `kp = 0.70`, 6/6 trials, **with no
adaptation at all**. No bounded offset on joint POSITION targets can repair a stiffness
mismatch — which retires all the nulls under one cause rather than eighteen separate stories.
No single stiffness dominates across terrain, which is the real argument for adapting impedance
online.

---

## 3. The measurement traps (each one cost a result)

**mjlab auto-resets inside `env.step()`.** A state-based fall check inspects a *respawned* robot
at height 0.77 and reports it healthy. This faked a +107-step survival gain and a clean
treated-vs-control separation on day one. Fix: take the environment's own termination flag.
Note it returns the OLD 4-tuple `(obs, reward, done, info)` — slot 3 is the info **dict**, and a
non-empty dict is truthy, so reading it as `truncated` terminates every episode at step 1.

**The same bug, twice more.** It corrupted `fault_screen.py` (why the policy looked implausibly
robust — "0/4 fell, just stops advancing" was falling and respawning), and later the *distance*
metric added specifically to guard against gaming: reading position after termination returns
`respawn − x0`, giving exactly −2.00 m at every tilt. Identical values across conditions is the
tell.

**`rand` is not a valid control.** A direction re-drawn each step random-walks to ~half the
treatment's displacement, so any difference is confounded with magnitude. Use `frand` — one
direction per trial, held, rescaled to the same `‖ΔW‖`. Every result here is against that.

**Survival credits slowing down.** Terminations are dominated by `ee_body_pos` (reference drift),
not falls, so a controller that lowers its commanded speed tracks an easier reference and
"survives" longer while covering less ground. Always report FORWARD DISTANCE alongside.

---

## 4. Diagnostics that turned out to be worthless

**`cos(g, ∂e/∂a)` does not predict performance.** Built as a premise gate, and it correctly
rejected the useless identified model (cos 0.001, held-out R² 0.02). But among gradients that do
carry signal it has no predictive power, and sometimes the wrong sign:

| configuration | cos | outcome |
|---|---|---|
| corrupted `g(x)`, downhill | 0.208 * | **+69.3** |
| repaired `g(x)`, downhill | 0.789 | +29.5 |
| repaired `g(x)`, uphill | 0.807 | null |

\* scored against the wrong error — see §5.

**The privileged CEM search is a lower bound, not a ceiling.** It found +33.8 downhill where the
adapter achieves +56.8, i.e. it explores a *weaker* class (12 structured parameters vs a
state-dependent function of a 160-d observation). A claim that "uphill offers almost nothing to
win (+17.8 ceiling)" was retracted on that basis.

**Alignment with the oracle's best constant direction has the wrong sign.** Downhill −0.054
(works), uphill +0.107 (fails).

---

## 4b. THE RESULT: the certificate is on the wrong quantity (15 Aug)

**Authority sweep, −0.13 uphill, development pool, 4 seeds, eta 3e-5, continuous.** Every terrain
run before this used `u_max=0.12 / b_W=0.4` — inherited defaults, never derived from this plant.
Against actions of mean |a| = 8.27 (peaks 86) that is ~1.5% authority. The fault scenarios had
been run at `u_max` up to 1.2 and `b_W` 4.0; when the testbed moved to slopes, those never moved
with it.

| `u_max` | `b_W` | ΔV (`delta_post`) | Δsurvival | V improved |
|---|---|---|---|---|
| 0.12 | 0.4 | −0.31 | −16.5 | 4/4 |
| 0.5 | 1.5 | −0.94 | −36.5 | 4/4 |
| 1.2 | 4.0 | −1.43 | −44.8 | 4/4 |
| 3.0 | 10 | −2.05 | −59.5 | 4/4 |
| 6.0 | 20 | −2.04 | −60.2 | 4/4 |

> **Raw data for four of these five rows no longer exists** (found 15 Aug). The output filename
> tag carried no token for `--dcm` / `--q-keep`, and `config` did not record them either, so the
> DCM sweep of §4d — same shift, same arms, same eta, same authority levels — wrote to the same
> five paths and overwrote 0.12 / 0.5 / 1.2 / 3.0. Only `u_max = 6.0` survives as task_error data;
> the four survivors at those levels carry `n = 8`, the DCM signature, against this table's 4.
> The table above stands as the record but cannot be re-analysed per-seed. Fixed: the tag and
> `config` now carry `dcm`, `q_keep`, `q_yaw`, `u_max`, `b_W`, `continuous`, `shift`, `objective`.
> This is rule 5's cousin — **a file that cannot be attributed to its experiment from its own
> contents is not a measurement**, and the umpteenth instance of a silent overwrite in this
> project (the tag comment two lines above records the previous one).

**The bound was never the limiter.** Across a 50x range of authority the Lyapunov decrease
condition holds at EVERY level (4/4 seeds, always) and `V` falls monotonically — while survival
degrades in lockstep. The harder the law minimises its own certificate, the sooner the robot
falls.

This rules out, in one sweep: the bound, the gain, the layer, the function class, the authority.
**`V = eᵀPe` on `task_error` satisfies its decrease condition and is anti-correlated with
survival. UUB in `e` is a true theorem about the wrong quantity.**

Two riders:
- **The direction carries real information.** At high authority `acc_frand` collapses to
  −118/−122 while `acc` holds at −60 — the law consistently does LESS damage than a
  displacement-matched random direction. `g(x)ᵀPe` is a real gradient aimed at the wrong target.
- **It saturates past `u_max ≈ 3`** (−2.05 → −2.04). The residual stops using extra headroom, so
  the reachability conclusion holds in effect even with the leash off.

**The requirement on any replacement `V` is now precise and testable:** its decrease must IMPLY
staying upright, i.e. this same sweep must show ΔV and Δsurvival moving in OPPOSITE directions.
Candidates with a claim to that property: capture point / DCM (`ξ = x_com + v_com/ω`), orbital
energy, angular momentum about the contact point. **This sweep is now the screen** — five
authority levels, does minimising it harder help or hurt. Every one of the ~20 uphill
interventions lacked exactly this test.

### 4c. P is not the fix — `e` itself is inadequate (15 Aug)

One-hot `P`: regulate ONE component at a time, −0.13 uphill, 4 seeds. **Every component has
NEGATIVE Δsurvival**, so no re-weighting yields a valid certificate:

| component | Δsurvival | ΔV |
|---|---|---|
| vx | −8.8 | −0.280 |
| vy | −2.5 | −0.016 |
| wz | −8.8 | −0.025 |
| roll | −3.5 | −0.004 |
| pitch | −5.8 | +0.017 |
| height | −6.5 | +0.002 |
| **full P** | **−16.5** | −0.309 |

Full `P` is roughly the accumulation of the individual harms: the more of `e` you regulate, the
more damage, in proportion. This closes the cheap repair and converts "the objective is wrong"
into the specific claim that **the survival-relevant quantity is not in `e` at all**.

### 4d. The divergent mode: promising, NOT established (15 Aug)

Rows 3/4 of `e` are `wx + λ·roll`, `wy + λ·pitch` with a shipped `λ = 0.3` — an arbitrary blend.
The divergent eigenmode of an inverted pendulum of height `h` is `θ̇ + ω·θ` with
`ω = √(g/h) = 3.885` at `h_ref = 0.65` — the same `ω` that defines the capture point. **The
shipped coupling is ~13x too small.** Added as `--dcm` in `online_adapt.py`.

Screen (4 seeds): DCM divergent-mode-only **+4.2**, DCM full P −12.8, and the decisive control —
same two rows at `λ = 0.3` — **−2.5**. So the swing is attributable to the COUPLING, not the row
selection. Authority screen (8 seeds):

| `u_max` | acc Δsurv | frand Δsurv | acc ΔV | acc−frand |
|---|---|---|---|---|
| 0.12 | +13.5 | +4.6 | +0.057 | +8.9 |
| 0.5 | +16.0 | +18.6 | +0.035 | −2.6 |
| 1.2 | +16.4 | −5.1 | +0.151 | +21.5 |
| 3.0 | +15.1 | −39.1 | +0.100 | +54.2 |

**First consistent positive in the project** — ~+15 steps replicated at four authority levels.
**But it FAILS the stated pass condition:** Δsurvival is FLAT across a 25x authority range, not
rising. A real certificate should buy more as you grant more. `frand` also ties it at `u_max=0.5`
(+18.6 vs +16.0), so the direction only separates at high authority — where the honest
description is that `acc` RESISTS damage (+15 while frand collapses to −39) rather than produces
benefit.

**Confound:** ΔV is POSITIVE here (`improved 2/8`) — the law raises its own cost while improving
survival. Partly mechanical (surviving 15 steps longer averages cost over more of the hard
regime), but it means ΔV and Δsurvival are entangled through episode length and the certificate
cannot be claimed to be descending at all. — **RESOLVED, and it was entirely mechanical: see
§4e.** The pairs were in the JSONs all along; neither gate needed the simulator re-run.

### 4e. Both gates run (15 Aug) — the certificate IS descending, and survival rises with it

`scripts/dcm_validate.py`, off the stored per-seed records. Output: `outputs/dcm_validate.txt`.

**Gate 1 — per-seed paired sign test, `acc − off`, one-sided, ties dropped:**

| `u_max` | Δsurvival | pos/neg | p |
|---|---|---|---|
| **0.12** (pre-reg) | +13.5 | 6/2 | **0.1445** ✗ |
| 0.5 | +16.0 | 7/1 | 0.0352 |
| **1.2** (pre-reg) | +16.4 | 7/1 | **0.0352** ✓ |
| 3.0 | +15.1 | 7/1 | 0.0352 |

One of the two pre-registered levels clears 0.05. 0.12 is the weakest-authority cell and misses.
Across all four, 27 of 32 pairs are positive.

**Gate 2 — length-normalised cost. The reported ΔV sign was an artefact, and it reverses.**
Comparing each pair only over the steps BOTH arms ran (`L = min` of the two `v_trace` lengths),
which closes the episode-length channel. Decomposed so the reversal is attributable:

| `u_max` | ΔV as reported | ΔV pure `V`, own length | ΔV pure `V`, common prefix | improved |
|---|---|---|---|---|
| 0.12 | +0.0574 | +0.0567 | **−0.1623** | 6/8 |
| 0.5 | +0.0345 | +0.0295 | **−0.1608** | 7/8 |
| 1.2 | +0.1511 | +0.1395 | **−0.1852** | 7/8 |
| 3.0 | +0.0995 | +0.0845 | **−0.1934** | 7/8 |

The effort term `λ_u‖r‖²` accounts for +0.0007 → +0.0151 of the gap; the length channel accounts
for +0.19 → +0.32. So the positive ΔV was **entirely** the adapter surviving further into the
hard late regime and averaging over it.

**This is the first quantity in the project to pass the §4b screen.** ΔV negative and Δsurvival
positive, simultaneously, at every authority level — the two moving in OPPOSITE directions, which
is exactly what §4b demanded of any replacement `V` and what `task_error` failed.

**Four things that must travel with the number:**
- **The four levels share ONE `off` arm.** Per-seed baseline survival is byte-identical across all
  four files (204/191/158/180/174/181/159/158) — as it must be, `u_max` cannot touch the frozen
  arm. They are four treatments against one control, **not** four independent replications, and
  must not be counted as 4× evidence.
- **Seed 3000 is negative at all four levels** (−19, −12, −28, −28). The "1 neg" is structural —
  one seed reliably harmed — not sampling noise. Expect it again on the test pool.
- **`acc_frand` is not a floor here either.** It swings −2.6 / +21.5 / +54.2 as authority rises,
  because `frand` collapses (down to 110 at `u_max` 3.0) while `acc` holds. `acc − off` is the
  honest read; unlike the gait case, it is genuinely positive.
- **No censoring:** longest episode is 239 against `n_post = 400`, so the cap is not binding and
  no pair is a censored tie.

**What still fails:** Δsurvival is FLAT (+13.5 / +16.0 / +16.4 / +15.1) across 25× authority,
while ΔV strengthens mildly and monotonically (−0.162 → −0.193). More authority buys more descent
but no more survival — saturation, not the anti-correlation of §4b, but still not a certificate
that pays more as you grant more.

**Verdict:** gate 2 passes outright, gate 1 passes at 1.2 and misses at 0.12. Next is the reserved
test pool at `u_max = 1.2`, pre-registering the prediction (+16 steps, one-sided sign test) before
running it.

### 4f. Test pool: NOT CONFIRMED, and the control reverses (15 Aug, same day)

Pre-registered in `docs/PREREG_DCM_TESTPOOL.md` before the run; seeds **4000–4007**, disjoint from
the 3000s every DCM number above was obtained on. Same command as the development 1.2 cell, only
`--pool` changed. Analysis `python scripts/dcm_validate.py --testpool`.

| | development (3000s) | **test (4000s)** |
|---|---|---|
| `acc − off` | +16.4, 7/1, p = 0.0352 | **+8.6, 6/2, p = 0.1445** ✗ |
| `acc − frand` | +21.5 | **−31.0, 4/4, p = 0.6367** |
| ΔV common prefix | −0.185, 7/8 | −0.061, 5/8 |
| forward distance `acc − off` | not logged | **−0.34 m, 2/6** |

**Three independent failures, and the second is the one that matters.**

1. **Primary missed.** +8.6 against a predicted +16; p = 0.1445.
2. **The displacement-matched random direction now BEATS the law** (+39.6 vs +8.6), reversing a
   +21.5 on development. `‖ΔW‖` rules out a handicap — acc 0.729, frand **0.835**, frand moved 14%
   further — and only one episode capped (frand, 4002), which understates frand. So the direction
   carries no information on seeds it was not tuned on. This is the third time in this project a
   positive has died on its displacement-matched control (§5 twice); the difference is that this
   time the control ran in the same job, under a rule written first.
3. **Rule 4.** The survival gain is bought by covering LESS ground, and in absolute terms every
   arm nets BACKWARDS: off −1.67 m, acc −2.01 m, frand −1.13 m. At `vx = 1.0` on −0.13 the robot
   cannot deliver the commanded speed at all — the ENERGETIC regime `shift_conditions` already
   documents (ceiling **+11.0 of 250** at 1.0 m/s, nearly doubling at 0.6 m/s).

**The warning was on the page before the run.** Development's +16.4 sat ABOVE the +11.0 ceiling
measured for that operating point. Rule 3 says screen by ceiling; an effect exceeding a documented
ceiling should be read as a measurement artefact to explain, not a result to confirm. **New
corollary: an effect LARGER than the known ceiling is evidence against itself.**

**What survives.** §4e's finding that the reported ΔV sign was an episode-length artefact is
untouched — that is an analysis correction, not a claim about the plant, and it replicates in
direction here (−0.061, 5/8). What does not survive is that the divergent mode is a valid
certificate. It passes the §4b screen on development seeds and fails on held-out ones.

**Not retried**, per the pre-registration: no other `u_max`, `--q-keep`, `eta`, or larger
`n_seeds`. If the DCM line is picked up again the honest next step is the 0.6 m/s operating point,
where a regulatory failure exists to repair — but that is a NEW experiment on a different cell,
and it must be pre-registered as one rather than presented as this one rescued.
— **Run as a development screen the same day, and the cell is dead too: see §4g.**

### 4g. `vx = 0.6` screen: the premise is false, and uphill is energetic at BOTH speeds

`docs/SCREEN_DCM_VX06.md`, development seeds, `n_post = 900`, conditions written before the run.

| arm | survival | **distance** | `‖ΔW‖` |
|---|---|---|---|
| off | 198.5 | **−2.60 m** | 0.000 |
| acc | 203.8 | **−2.82 m** | 0.663 |
| acc_frand | 165.2 | **−1.73 m** | 1.218 |

`acc − off` = +5.2 steps (6/8) and **−0.22 m**. The screen required ≥7/8 and non-negative
distance; both fail.

**The cell's premise is falsified.** Dropping the command to 0.6 m/s was supposed to make the
reference deliverable. The robot nets BACKWARDS on 8 of 8 seeds and does so *further* than at 1.0
(−2.60 vs −1.67 m). `vx` verified applied (`twist_ranges=[[0.6, 0.6], …]`). **So −0.13 uphill is an
ENERGETIC failure at both speeds, and no bounded joint-space residual addresses it at either.**
This is the same conclusion as §2's stiffness result arriving by a second road, and together they
say the uphill axis is simply the wrong place to demonstrate adaptation with this policy.

**Correction to a comment that has been steering operating-point choices.** `shift_conditions`
claims 0.6 m/s makes the reference deliverable with the ceiling "nearly doubling to +18.8". Direct
measurement contradicts it. **Do not reuse +18.8 until it is re-derived.**

**A methodological finding that reaches every `acc − frand` number in this file.** `frand` ran 165
steps against `acc`'s 204 and still finished with **1.84× the accumulated `‖ΔW‖`** (1.218 vs
0.663). Per-step rescaling does NOT deliver matched *cumulative* displacement once arms survive
different lengths — which is exactly the case the control was built for. Rule 6 said state
`‖ΔW‖` for both arms; the reason is now concrete. §4f's pair was close (0.729/0.835) so that
verdict stands, but **a large `acc − frand` accompanied by a large `‖ΔW‖` gap is not a result.**

---

## 4h. The screening battery (15 Aug): aim the next experiment instead of guessing it

`scripts/screen_battery.py`. The mechanisms above, turned into an up-front test applied to every
condition at once. **Five of six untested faults eliminated in 11 minutes**, against the ~2 days
per null this project had been paying.

**Stage A — is there a loss, and is it slow enough to act on?** (4 seeds, n_post 250)

| condition | steps | dist | median fall | verdict |
|---|---|---|---|---|
| nominal | 250.0 | +4.90 | — | — |
| `kp_left` | 124.2 | +3.41 | 124 | **VIABLE** |
| `slope15` | 181.0 | +9.84 | 176 | **VIABLE** (control: the confirmed positive) |
| `slope13_up` | 172.8 | **−1.70** | 173 | VIABLE by survival, but see the distance |
| `kp_legs` | 41.5 | +0.68 | 42 | TOO FAST — no detection |
| `tqlimit_legs` | 40.8 | +0.67 | 42 | TOO FAST |
| `payload_elbow` | 39.2 | +0.53 | 39 | TOO FAST |
| `payload25` | 250.0 | +4.30 | — | NO LOSS |
| `friction_ood` | 250.0 | +3.63 | — | NO LOSS |

The controls rank correctly, which is what makes the ranking trustworthy. Note `kp_left`
(asymmetric) is viable while `kp_legs` (symmetric) is not — the asymmetry is what topples the
robot slowly enough to act on.

**Stage B — static dominance and CONFLICT.** `payload25` wants `kp = 1.00` and collapses to 33.3
at 0.70; `slope13_up` wants 0.70 and degrades to 174.7-going-backwards at 1.00. Opposite
directions, ~217-step swing. **This is the positive criterion for online adaptation, measured
cleanly for the first time.** Also: `slope13_up` at kp=0.70 gives 250.0 steps **and +0.61 m**
(+0.94 m on the wider sweep) — the first POSITIVE forward distance uphill in the project,
supplying the distance axis the original `kp=0.70` finding lacked.

*Bug found by rule 5, kept here because it will recur:* `kp_legs`/`kp_left` first printed
byte-identical rows. `motor_kp_scale` is computed from the pristine model and `fault_joints` is
read from the same config dict, so a retune passed alongside a kp fault REPLACES it — the fault
silently vanished and the cell measured a healthy robot. There is no honest kp sweep for a kp
fault through this API; those cells are now skipped as NOT SEPARABLE. Ties are no longer resolved
by argmax over list order either.

**Stage C — reachability, and does the gradient point at the fix?** Best constant residual over
four structured leg directions, then `cos` against the ACC descent direction.

| condition | base | best | gain | direction | cos | distance check |
|---|---|---|---|---|---|---|
| `kp_left` | 125.0 | 151.5 | +26.5 | hip_pitch −0.12 | **+0.574** | +3.35 → **+4.20 m** OK |
| `slope15` | 186.5 | 202.5 | +16.0 | knee −0.06 | **+0.639** | +10.20 → **+11.32 m** OK |
| `slope13_up` | 172.5 | 199.0 | +26.5 | ankle_pitch −0.06 | **−0.000** | −1.66 → **−1.95 m FAILS** |

**`kp_left` is the best cell in the project right now**: largest reachable gain, gains on BOTH
axes, and the objective already points at it. `slope15` reproduces the known positive, which is
the screen validating itself.

**Uphill, finally unambiguous.** Stage C's +26.5 is pure rule-4 gaming — it survives longer while
covering LESS ground (−1.66 → −1.95 m). So there is no partial position-channel fix uphill either,
and `cos = −0.000` is consistent with there being nothing to point at. Stage C selected on
survival alone, which is the same trap; `scripts/verify_stage_c_dirs.py` now re-checks winners on
distance and must be run on any future stage-C result.

### 4i. The objective screen, and a positive retracted by its own null (15 Aug)

`g_sᵀPe = Σᵢ pᵢ (g_{s,i} · eᵢ)` is **linear in the weights**, so "does ANY non-negative diagonal
weighting steer stiffness correctly?" is a linear feasibility problem, not a search — one LP
decides the whole family. Ground truth is two-sided for the first time (grades want softer,
payloads want stiffer), which a one-sided truth could never test.

It returned **FEASIBLE**, margin +0.356, weight on height-sag. Then the null (`objective_battery
_null.py`, pure arithmetic, no simulator):

- **16 of 16 sign patterns are feasible.** The family fits any answer, so feasibility carried no
  information about the plant.
- **Leave-one-out predicts 1/4**, below the 2/4 chance rate.

Retracted the same hour. The cause is structural — **4 constraints against 7 free weights** — and
no solver fixes it. `ground_truth_kp.py` sweeps kp over 14 parametric conditions to raise the
constraint count, at which point feasibility can fail and leave-one-out means something.

**Rule 10: a screen that cannot fail has not been passed.** Before reporting any fitted object,
run the exhaustive null over the target patterns and a leave-one-out. Both are free.

### 4j. THE OBJECTIVE FAMILY IS CLOSED: the feedback does not carry the stiffness direction

Re-run against **21 measured constraints vs 7 free weights (3.0x)** — the ratio at which the LP
can actually fail. It fails. And this time the null certifies the test: **0 of 2001 sampled sign
patterns are feasible**, against 16/16 before.

Infeasibility alone is all-or-nothing, so the useful question is how well ANY weighting can do
against the trivial predictor that ignores the state and always says the commoner answer
(13 of the 21 cells want *stiffen*, so the majority baseline is **13/21**):

| row | sign across the 21 cells | score |
|---|---|---|
| `wz` | **CONSTANT** | 13/21 |
| `roll_mode` | **CONSTANT** | 13/21 |
| `vz+height` | **CONSTANT** | 13/21 |
| `height_sag` | 4 pos / 17 neg | 11/21 |
| `vx` | 11 pos / 10 neg | 10/21 |
| `vy` | 3 pos / 18 neg | 10/21 |
| `pitch_mode` | 9 pos / 12 neg | 8/21 |

**Three of the seven rows have a CONSTANT sign across conditions that demand OPPOSITE stiffness
responses** — they cannot discriminate by construction. No single row beats 13/21, and the three
that reach it do so by always saying "stiffen", i.e. by being the trivial predictor.

Best over 200k random non-negative weightings: 15/21. **That is not a signal** — the same search
on SHUFFLED targets reaches ≥15 in **70% of trials** (scores 13–16, mean 14.8). Seven free
weights maximised over 200k draws manufactures +2 from noise.

**Conclusion, and it is the most decisive thing measured today:** the observed error components do
not carry the stiffness direction *at all*. This is not a gain problem, a trust-region problem, a
layer problem, or a parameter-set problem — **widening `theta` cannot help, because the FEEDBACK
is uninformative for the decision the law has to make.** It closes the diagonal objective family
the way §4b closed the authority axis: by ruling out the family rather than a point.

**Where to go instead.** The physical discriminator between *sagging under load* (stiffen) and
*fighting a grade* (soften) is not in base velocity/attitude. Candidates, all requiring sensing
not currently plumbed through `observe()`: vertical GRF against body weight, leg mechanical work
per stride, servo tracking error resolved along gravity vs along travel, stance duty factor.
Note `height_sag` *should* have been the discriminator and is 17/21 negative — worth checking
whether gait oscillation is swamping it before abandoning it.

**Rule 11: compare against the majority-class baseline, and null the SEARCH, not just the
result.** A maximum over many draws is a fitted object and needs its own shuffle null.

**Strengthened: it is not the PSD restriction** (`scripts/objective_signed_lp.py`). The LP above
forces `p ≥ 0`, which is required for `V = eᵀPe` to be a Lyapunov candidate but is not required
for a direction *chooser*. Re-solved with `p` free in sign (`‖p‖₁ = 1`, a strictly larger family):
**still INFEASIBLE**, and the null confirms the test can fail — only **31/600 = 5%** of random
sign patterns are feasible under the signed family. So the closure is about the SIGNALS, not the
certificate shape: **no linear read-out of these observables, PSD or otherwise, carries the
stiffness direction.**

**Servo-error bug in this section's inputs — found, measured, verdict unaffected.**
`MODEL_AND_DERIVATIONS.md` §7.1 asked for a unit test on `ẽ = q_des − q` "given this project's
history with frame and quantity mix-ups". It was right to. Both this screen and
`objective_sign_screen.py` built `g_s = G(x)·trk` from `trk = nominal_action() − q`, but
`nominal_action()` returns the **raw 20-d policy output**, while mjlab's `JointPositionAction`
forms the target as

    q_des = scale ⊙ a + offset,   scale ∈ [0.225, 1.0],  offset = default pose (‖offset‖ = 1.49)

so `a − q` is an unscaled action minus an angle with the whole default pose omitted.
`scripts/servo_error_check.py` measures the damage: cos(wrong, right) **0.978–0.984** on the servo
error and **0.992–0.996** on `g_s`, with `‖ẽ_wrong‖ ≈ 93` against `‖ẽ_right‖ ≈ 32` — close to a
uniform 3× rescale. **No `z` row changes sign in any condition**, and since the screen reads only
signs and the LP is invariant to uniform positive scaling, **§4j's INFEASIBLE verdict stands.**
Fixed in `objective_battery.py`; `objective_sign_screen.py` annotated in place as the historical
record. Any *magnitude* previously read off either script is wrong.

### 4k. `V` is a LAGGING indicator, and the channel cannot reach the leading one (15 Aug)

Offline analysis session, **no simulator**; everything below is computed from JSON already in
`outputs/`. Scripts and full write-up: `theory_2026-08-15/` (`FINDINGS.md` + 6 scripts). This is
the actuation-side counterpart to §4j's feedback-side closure, and the two arrive together.

**The instrument.** `baseline_pre_shift` is the mean pre-shift cost at `rho = 0`, so it is exactly
`mean(sᵀQs)` on the *unperturbed* orbit for that seed — `V*`, free, inside every episode. And
`v_trace` is pure `sᵀQs` (`online_adapt.py:492`). Same quantity, same units. Every result here is
on the **`off` arm alone** — frozen policy, no updates — which removes the law, gain, layer,
bound, gradient and authority from the question. **That makes it a stronger form of §4b:** §4b
infers the certificate is wrong from adaptation outcomes; this measures it with no adaptation.

**`V` predicts nothing until the fall is underway.** Spearman(signal over a leading window,
`steps_survived`), within shift, n = 69–75 passive episodes that fell (capped ones excluded):

| window | median lead to fall | `V` (shipped) | `\|pitchdot\|` |
|---|---|---|---|
| 50 steps | 130 (**2.6 s**) | **+0.102** | **−0.610** |
| 75 steps | 106 (2.1 s) | −0.047 | −0.505 |
| 100 steps | 81 (1.6 s) | −0.254 | −0.594 |

`V` acquires a usable sign only as the lead shrinks, and spikes **28×** over the final 25 steps.
That is a lagging indicator. (A within-episode AUC version of this test looked supportive and is
**degenerate** — the label is a function of elapsed time and a pure clock scores 0.959, above every
candidate. Discarded.)

**But the channel cannot reach `|pitchdot|` either.** Arms are paired by seed under byte-identical
conditions and the sim is deterministic, so every arm-vs-arm difference *is* causal authority — no
null model needed. Induced shift in `|pitchdot|` over its seed-to-seed sd: **0.14** (steps 0–50),
**0.21** (steps 100–200, the fair window once `‖dW‖` has matured). Payoff
Spearman(Δ`|pitchdot|`, Δsurvival) = **+0.012** over n = 494 pairs (`V`: −0.067). Specificity: `acc`
beats displacement-matched `frand` on **269/478 = 56%** of pairs (chance 50%), with `frand` carrying
the LARGER `‖dW‖`. **So do not swap `Q` to pitch rate — it would be another null.**

**Rule 12: a candidate quantity must be BOTH predictive and reachable, and both are free.**
Predictive: on PASSIVE rollouts, higher early value must forecast earlier failure at a lead beyond
the fall timescale (53–66 steps), with the sign holding in EVERY condition. Reachable: the induced
shift must be an appreciable fraction of the quantity's seed-to-seed spread. `sᵀQs` fails the
first, `|pitchdot|` fails the second. This screen needs no adaptation runs at all, where §4b's
needs 5 authority levels × 3 arms × n seeds.

**Correction to §4d: the coupling does nothing.** Sweeping `|thetadot + lam*theta|`: −0.594 at
`lam = 0` (pure rate), −0.622 at 2, **−0.537 at `lam = w = 3.885`**, +0.052 at pure angle — CIs
fully overlapping from 0 to 3. The **convergent** mode scores −0.456 against the divergent −0.537,
a gap of **−0.019** on the n=39 group, so the eigenmode structure is not what is being detected.
Re-coupling 0.3 → 3.885 moved the correlation the WRONG way. §4d's "the swing is attributable to
the COUPLING, not the row selection" does not survive.

**Explanation for §4f's dev-pass/test-fail.** Roll's sign *reverses with terrain* — roll RATE is
+0.466 uphill and −0.578 downhill (pooled +0.041) — while roll ANGLE is consistent (−0.415) and
pitch ANGLE is not (+0.052). **Sagittal is rate-dominated, lateral is angle-dominated.** The
pre-registered cell used `--q-keep roll,pitch`, bundling a sign-consistent sagittal predictor with
a terrain-dependent lateral one. That is exactly what passes on tuning seeds and dies on held-out.

**The projection ball is binding, which explains §1's gain result.** `‖dW‖` from `trace`:

| cell | eta | final `‖dW‖` | saturates at | still rising |
|---|---|---|---|---|
| slope13_up | 1e-3 / 3e-4 / 1e-4 | **0.4000 = `b_W`** | 31 / 56 / 50 steps | 0.0% |
| slope13_up | 3e-5 | 0.2195 | 162 steps | +37.7% |
| **slope15 (the +56.8 cell)** | **3e-5** | **0.4000 = `b_W`** | **42 steps** | +3.9% |

Every gain ≥ 1e-4 pins at the boundary within ~1 s and never moves again. **§1's unexplained
"1e-3 → 3e-5: +23.2 → +51.6" is that the large gains were saturating, not adapting** — both end at
0.4 downhill, so eta changes only the direction reached. It also qualifies the headline: the
confirmed +56.8 runs at its authority ceiling from step 42 onward.
(`b_W` is absent from `config` in pre-§4b files, so the `umax1.2`/`umax6.0` rows fall back to the
0.4 default and are excluded here.)

**One repair closed for free.** Hypothesis: `V` has a limit-cycle floor `V*` (the gait's own ripple)
and harm comes from descending past it, fixable with a dead zone. **False.** The adapter penetrates
below `V*` no more than the frozen policy (13.0 vs 12.5%, 2.7 vs 2.7%, 1.8 vs 1.8%) and post-shift
`V` runs 2–9× ABOVE `V*` throughout. A dead zone at `V*` would do nothing.

**What it means together with §4j.** §4j closed the *feedback* side — the observed error components
do not carry the stiffness direction. This closes the *actuation* side: the residual acts on joint
POSITION targets, a configuration-level input, while falling is a MOMENTUM-level failure forecast
by a rate. One statement covers ~20 uphill nulls, why `kp = 0.70` works with no adaptation (§2),
why `gait_period_s` works (§7, momentum reset at impact), and why `‖dW‖` pins at `b_W` in 42 steps.
**Change the CHANNEL, not the cost** — which is where the impedance-conflict result points, and it
agrees with §4j's shortlist (vertical GRF, leg work per stride, stance duty factor) being
momentum/energy-level quantities rather than configuration-level ones.

**Caveats.** Correlational; 7 candidates screened across 3–4 conditions with sign-consistency as
the only multiplicity guard; **not pre-registered**; rates are finite differences of `att_trace` at
50 Hz, a proxy for the true `w_y`; the pooled payoff figure mixes experiment families. This is a
screen, not a confirmation.

---

### 4m. Momentum sensing, a phase-aliasing bug in every earlier screen, and a candidate that died

**Momentum/energy sensing does not rescue it.** `scripts/objective_battery_sensing.py` adds
`leg_power` (`Σ_legs τᵢq̇ᵢ`, `∂P/∂σ = Σ K_p,ᵢ ẽᵢ q̇ᵢ` exactly), `leg_power_LR` (left/right work
asymmetry) and `load_excess` (`(F_z − mg)/mg` from summed foot contacts,
`∂F_z/∂σ = m·G[vz]·ẽ`), against the same 21 constraints. **Still INFEASIBLE.** `duty_factor` was
deliberately excluded: contact count is piecewise constant, so its derivative is zero almost
everywhere and a gradient law cannot regulate it.

**A phase-aliasing bug that affects §4j and §4i as well.** The realised gait period is **27
control steps** — measured, not assumed: the pitch autocorrelation has clean maxima at
**27 / 54 / 81 / 109** (`scripts/gait_period_estimate.py`). `gait_period_s = 0.9` at 50 Hz would
imply 45. Every earlier screen averaged `z` over **6 states × stride 3 = 18 steps = 67% of a
stride, after a FIXED settle** — so every sample sat at the same gait phase with the same leg in
stance. Two independent tells, both caught by sanity checks rather than inferred: summed vertical
GRF read **1.33×** body weight (a stride average must be ~1.0×), and `leg_power_LR` came out
**20 positive / 1 negative**. Re-run over exactly one stride, `leg_power_LR` becomes sign-CONSTANT
and GRF falls to 1.23×. **Averaging must span a whole number of strides**; the note in §4k that
`height_sag` "should have discriminated" may have the same cause.

**The verdict is unchanged but the numbers moved a lot,** which is the point: `vx` went from 10/21
aliased to **17/21** stride-averaged, the first row in this project to beat the 13/21 baseline.

**And it died on a test worth keeping.** `scripts/vx_candidate_test.py`:

| test | result |
|---|---|
| exact hypergeometric | p = 0.0117, **×10 rows = 0.117, fails** |
| condition-level permutation (12 blocks, 20k) | p = 0.0019, ×10 = **0.019, survives** |
| leave-one-condition-out | 17/21 vs baseline 13/21 |
| **within-family discrimination** | **grade 7/11 vs baseline 8/11 — WORSE than trivial** |

Grades are the only family containing both target signs (8 soften / 3 stiffen); payload (6/6) and
push (4/4) are all-stiffen, so `vx` scores there by saying "stiffen" always and discriminates
nothing. **The entire 17/21 is between-family structure — `vx` learned that payload cells want
stiffening, which is the experiment's design, not physics.**

The first version of that script checked only the permutation and LOO tests and printed *"worth
testing as a controller"*. It was wrong, and it is the exact failure mode this file exists to
prevent.

### 4s. THE DISTURBANCE OBSERVER WORKS — the composite law's other term has a live signal

MAGIC-VFM (Lupu, Xie, Preiss, Alindogan, Anderson, Chung, arXiv 2407.12304) uses a **composite**
law, eq. 19:

    θ̂̇ᵢ = −λθ̂ᵢ − γᵢ uᵀΦᵢᵀR⁻¹(Σⱼ Φⱼu θ̂ⱼ − y) + γᵢ sᵀΦᵢu
            forgetting        PREDICTION error           TRACKING error

**This project has only ever implemented the third term.** Every screen — 42 constraints, base
state, momentum/energy, joint encoders — asked whether the TRACKING-error direction is
informative. It is not, and rule 15 explains why structurally.

The prediction term is driven by `y = v̇ − f(v,u,t)`: a **disturbance observer**, measured here as
the actual base acceleration minus a pristine-model `mj_forward` prediction at the same
`(q, q̇, ctrl)`, rotated into the body frame. `scripts/residual_observer.py`, 9 conditions × 3
seeds, one stride each.

**It separates the disturbance families almost perfectly, and physically rather than by fitting:**

| family | `a_x` (travel) | `a_y` (lateral) | `a_z` (vertical) |
|---|---|---|---|
| grade | **1.560** | 0.105 | 0.132 |
| payload | 0.121 | 0.120 | **4.611** |
| push | 0.268 | **1.219** | 0.155 |

A grade is a longitudinal force (12× its own vertical), a payload vertical (38× its longitudinal),
a push lateral. **Family classification 25/27 against a 12/27 majority baseline.** This is the
first signal in the project that identifies the disturbance at all — every proprioceptive quantity
failed at it, and the reason it works is that `y` is the disturbance force *before* the loop has
rejected it, whereas tracking error, pitch rate, leg power and servo error are all what remains
*after*. That is also why their signs flip with operating point and this does not.

**What it does NOT do: `kp*` across families is still 3/27, identical to proprioception.** Knowing
you are in a payload does not tell you what a payload wants if you have never seen one.
Leave-one-family-out demands extrapolation to an unseen disturbance TYPE, which is a property of
the lookup table, not the sensor. Within-family (unseen severity — the realistic deployment case)
proprioception already reached 17/27.

**And it does not move the ceilings.** kp is +17.6 steps (7%, §4o), stride is 0% (§4q). Those are
properties of the KNOB, not the law, so no observer rescues gain scheduling.

**Where the value would be, if anywhere:** composite adaptation does not schedule a scalar — it
fits `Φθ̂` to match the measured disturbance force, i.e. feedforward cancellation. `y` is exactly
that term's input, so the second term of the composite law has a working signal on this robot
where the third provably does not. **The open question is that channel's ceiling: how much of `y`
can a bounded residual actually cancel?** The existing compensability probe answers a related
question for the position channel — min-norm cancelling residual needs 6–54× the budget, oracle
leaves 57–95% standing — and that number, not the observer, is what decides this.

### 4u. RE-BASELINE: the confirmed gain does not survive a good default (16 Aug)

Pre-registered in `docs/PREREG_REBASELINE_GAIT08.md` before running, with the prediction
`acc − off ≤ 0` on record and **Wilcoxon** chosen in advance per rule 14. Same cell as the
confirmed +56.8 (`slope15`, `mlp.6`, eta 3e-5), same 10 development seeds as the T = 0.90 run,
`n_post = 900` so the good arm cannot be censored.

**Paired within-seed period contrast — the headline:**

    acc − off at T = 0.90 : +16.2 steps
    acc − off at T = 0.80 : −15.6 steps        change of −31.8 steps

**Primary endpoint: prediction confirmed in sign, NOT significant** — `acc − off` = −15.6 steps,
−2.85 m, Wilcoxon one-sided **p = 0.5273**. The honest claim is therefore *not* "adaptation harms
at a good default" but **"the benefit disappears into noise"**. Per-seed swings run −543 to +557,
so §7's "per seed it is a gamble, not a wash" replicates at n = 10 — and that variance argues for
treating the +16.2 at T = 0.90 with more caution than its mean suggests. 2/10 pairs censored.

**The trap fired exactly as named in advance.** `acc − frand` = **+251.9**, which alone reads as a
large win; but `frand − off` = **−267.5**, so both arms damage the good baseline and the gap only
means *acc harms less than a random direction harms*. Reporting `acc − frand` here would have
produced a spectacular and entirely false result. This is the single most dangerous statistic in
the project and it has now been caught by pre-registration rather than after the fact.

**What this does and does not do to the headline.** The +56.8 stands as a measurement — properly
controlled, pre-registered, held-out seeds. Its *meaning* changes: adaptation was recovering
ground that one line of `rom_params.yaml` gives for free, and gives more of (+63.5 in the same
cell). Combined with §4b, §4o and §4t, four independent objectives now show the same signature.

**Ship `gait_period_s = 0.80` regardless** (+11.9 steps, +30% distance across all 14 conditions).

### 4x. A MORE ACCURATE GRADIENT MAKES THE CONTROLLER WORSE (16 Aug)

The §4w fix is unambiguously a better gradient (L1 cos 0.446 → 0.513, L3 0.752 → 0.865, better at
6/6 states). Running the confirmed cell with it, same 10 development seeds, `off` arm verified
**byte-identical** so the contrast is paired on the gradient alone (`gx_fidelity_compare.py`):

| | `acc − off` | distance |
|---|---|---|
| old `g(x)`, no `diag(s)` | **+16.2** | +0.76 m |
| corrected `g(x)` | **+1.5** | −0.21 m |
| change | **−14.7** (7/10 seeds worse) | −0.98 m |

Wilcoxon one-sided p = 0.0596 — marginal, not significant, and reported as such.

**The fidelity ladder now has three points and is monotone in the WRONG direction:**

| `g(x)` | L1 cos vs measured `∂e/∂a` | result |
|---|---|---|
| corrupted (§5) | 0.208 | +69.3 *(different pool)* |
| no `diag(s)` | 0.446 | +16.2 |
| with `diag(s)` | 0.513 | **+1.5** |

**If the law worked by descending the true error gradient, a better gradient would give a better
result.** It gives a worse one. This is the sharpest available statement of what §4b, §4o, §4t and
§4u have each shown from their own angle: **the method's benefit was never gradient descent on the
error.** Because the error is the wrong quantity (§4b: its certificate decreases while survival
degrades), descending it *more faithfully* is actively harmful — which is exactly the sign this
ladder shows.

It also retires "fix the gradient" as a repair. §4 already listed `cos(g, ∂e/∂a)` as a worthless
diagnostic; this makes the stronger claim that fidelity is **anti**-predictive on a properly paired
contrast, not merely uninformative.

**Caveat kept:** only the last two rows are strictly paired (the +69.3 was a different pool), and
p = 0.0596 at n = 10. The direction is consistent with §4's independent pair (0.208 → +69.3 vs
0.789 → +29.5) but neither contrast alone is decisive.

### 4w. `ACCGain` omitted the action scale — the law's gradient was rotated (16 Aug)

Found while reconciling §4v against §4t. `acc_gain.py:147` built `G = A[:, dofadr] * kp`, but the
policy output is not the joint target: mjlab's `JointPositionAction` forms
`q_des = scale ⊙ a + offset`, so `∂τ/∂a = K_p ⊙ scale` and `MODEL_AND_DERIVATIONS.md` §3 already
specified `g(x) = M⁻¹SᵀK_p diag(s)`. **`scale` runs 0.225–1.000 across joints**, so the omission
ROTATED the gradient rather than merely rescaling it.

**A/B against the finite-difference truth** (`validate_gx.py --no-action-scale`, +0.262, 6 states):

| fidelity | without `diag(s)` | with `diag(s)` |
|---|---|---|
| L1 | +0.446 (sd 0.141) | **+0.513** (sd 0.145) |
| L3 | +0.752 (sd 0.085) | **+0.865** (sd 0.106) |

Better at **6/6 states for both fidelities**; random floor −0.025. The omission was real and the
fix is a measurable improvement in gradient accuracy.

**Do not assume that improves the controller.** §4 of this file records that gradient fidelity does
NOT predict performance here, and has had the wrong sign: corrupted `g(x)` at cos 0.208 gave
**+69.3**, repaired at cos 0.789 gave **+29.5**. So the fix is correct and its effect on the law is
an open question. `use_action_scale=False` reproduces the old behaviour for checking prior results.

**Blast radius.** Every `g(x)`-derived MAGNITUDE before 16 Aug is understated (the ceiling figures
in §4v most of all). DIRECTIONS are rotated but keep their sign structure, since `scale > 0`
elementwise — which is why the sign-based screens (§4j, §4m, §4r) are unaffected, as
`servo_error_check.py` separately confirmed (no `z` row changes sign). The +56.8 stands: it was
validated empirically against a displacement-matched control, whatever direction it descended.

**Also: a misleading line in `validate_gx.py`'s output.** It prints `random control : mean cos =`
followed by the **L3** value, while the per-state `random` column and the `random floor` line both
correctly report ≈ −0.025. Read alone, that line says the gradient is no better than random.

### 4v. The composite channel, measured on `y` — **SUPERSEDED, see §4t**

> **This section is wrong and is kept only as the retraction.** Its `G` omitted the action
> scale: `q_des = scale⊙a + offset`, so `∂τ/∂a = K_p⊙scale`, and
> `MODEL_AND_DERIVATIONS.md` §3 specifies `M⁻¹SᵀK_p diag(s)`. Leg `scale` runs 0.225–1.000
> (mean 0.567), so `G` was overstated ~1.76× and the required budget understated by the
> same factor — the quoted 2.2× should be ~3.9×. §4t, computed independently from the
> 29 July compensability probe, gives 5.2–8.6× and 0–14% cancellable, and is the number to
> use. `scripts/oracle_cancel.py` inherited the same `G`, so its +11.0 steps is a valid
> measurement of THAT feedforward controller but is NOT the oracle it was labelled.
> Retained because the payoff pattern it found still stands on its own terms: cancellation
> helped downhill (+30.5 / +1.27 m vs a displacement-matched control) and HURT lateral push
> (−6.8), i.e. non-uniform in sign — and was beaten 2:1 by shipping `gait_period_s = 0.80`.
>
> **RE-RUN WITH THE CORRECTED `G` — the payoff MORE THAN DOUBLES, and the "modest" verdict above
> was an artefact of the bug:**
>
> | condition | buggy `G` | corrected `G` | vs frand |
> |---|---|---|---|
> | tilt −0.13 | +20.2 | **+46.5** | +45.5 |
> | tilt +0.26 | +30.5 / +1.27 m | **+54.2 / +2.88 m** | +45.8 |
> | payload25 | +0.0 | +0.0 (capped) | +0.0 |
> | push60 | −6.8 | −3.2 | −5.5 |
> | **mean** | +11.0 / +0.32 m | **+24.4 / +0.75 m** | |
>
> At +0.262 the oracle canceller now reaches **+54.2 steps and +2.88 m** — comparable to the
> confirmed adaptation gain (+56.8) and close to the gait retune (+63.5), where the buggy version
> read half that. **The composite/feedforward direction is materially more promising than §4v
> originally concluded.**
>
> This also runs against §4t's expectation: the oracle applies exactly the CLIPPED min-norm
> solution §4t analysed, and §4t's `left = 1.52` for grade implied clipping would make grades
> WORSE. Empirically grade gets +46.5. The reconciliation is that §4t targets task-error drift
> while this targets `y`, but the payoff here is measured rather than derived.
>
> Unchanged limits: this is an ORACLE (perfect `y`, no estimator — a real `θ̂` does worse), 4 seeds,
> still non-uniform in sign, and still below a one-line static retune in its best cell.

> **Re-run with `diag(s)` restored — the two independent routes CONVERGE.** Required budget by
> family, mine (fresh rollouts cancelling `y`) against §4t (July compensability data on task-error
> drift):
>
> | family | mine, buggy | mine, fixed | §4t |
> |---|---|---|---|
> | grade | 2.8× | **6.5×** | 8.6× |
> | payload | 2.1× | **5.5×** | 5.4× |
> | push | 1.7× | **4.0×** | 5.2× |
> | ALL | 2.2× | **5.4×** | 5.2–8.6× |
>
> Payload agrees to 2%. **The required budget is ~4–9× the cap by two unrelated measurements**,
> which is a much firmer number than either alone.
>
> The remaining difference in *cancellable fraction* (my 66% at `u_max` vs §4t's 0/5/14%) is
> methodological, not an error in either: **§4t CLIPS the unconstrained min-norm solution while I
> SOLVE the box-constrained problem.** Clipping can be worse than doing nothing — §4t's `left`
> = 1.52 for grade is exactly that — whereas constrained least-squares never is. **§4t's number is
> the one that describes deployed behaviour**, because the ACC law saturates rather than solving a
> QP. Mine is the best achievable within budget, i.e. a strict upper bound on any saturating
> implementation.

#### Original text (do not cite the budget figures)


**Reachability (`scripts/residual_ceiling.py`, 486 states, 9 conditions).** Full cancellation of
`y` needs **2.2× the current `u_max`** (1.7–2.8 by family) and **88% is achieved at the existing
0.12**, 100% by 0.5. Against the position channel's 6–54× with 57–95% left standing, a different
regime. *(The "in-range 100%" column is arithmetic, not evidence: `G` is 6×20 and generically full
row rank, so every 6-vector is in range by dimension count. And the comparison is not
apples-to-apples — the compensability probe targeted task error through closed-loop dynamics,
this targets an instantaneous acceleration residual through an algebraic map.)*

**Payoff — the oracle** (`scripts/oracle_cancel.py`): `r = clip(−G⁺y, u_max)` as pure feedforward,
no learning. This is the CEILING of the composite prediction term — `θ̂` perfectly converged, no
estimator error, no excitation cost — so nothing built on that law can beat it.

| condition | cancel − off | cancel − frand | distance |
|---|---|---|---|
| tilt +0.26 | **+30.5** | **+38.0** | **+1.27 m** |
| tilt −0.13 | +20.2 | +19.0 | −0.10 m |
| payload25 | +0.0 (both capped) | +0.0 | +0.32 m |
| push60 | **−6.8** | −16.2 | −0.22 m |
| **mean** | **+11.0** | | **+0.32 m** |

Downhill is a real win on BOTH axes against a displacement-matched control (‖r‖ 0.395 vs 0.338) —
and it is the same cell as the confirmed +56.8, which is coherent. But it HURTS lateral push, does
nothing for payload, and is distance-negative uphill.

**The deciding comparison: at `tilt+0.26` the oracle canceller buys +30.5 where shipping
`gait_period_s = 0.80` buys +63.5** — twice as much, from one scalar, with no observer, no
estimator, no law.

**So the third objective fails the same way as the first two.** Every stage of the mechanism is
sound — the disturbance is identified (§4s), 88% of it is reachable (above), and removing it helps
in 2 of 4 cells — but *cancelling the disturbance is not the same as walking better*, and where it
helps, a static retune helps more. Combined with §4b (certificate decreases, survival degrades)
and §4o (tracking improves, survival falls), that is three independent objectives with the same
signature. It is now the most robust finding in the project and belongs in the paper as such.

**Rule 16: prefer signals measured BEFORE the loop rejects them.** A disturbance observer carries
the disturbance; a tracking error carries whatever the controller failed to remove, which is a
function of the operating point and therefore sign-unstable across conditions. Every candidate
that flipped sign by condition (roll F4, `vx` §4m, `load_excess` §4n, `serr_asym_LR` §4r) was a
post-rejection quantity; the one that did not flip was pre-rejection.

### 4q. STRIDE: conflict is real, the ceiling is ZERO — and 0.80 is the default you should ship

`scripts/gait_conflict.py`. `gait_period_s` 0.9→0.8 was the largest lever ever measured here
(+104 steps) and was set aside because it dominated on all five TERRAINS — no conflict, so retune.
That reasoning was suspect for the same reason it was wrong for `kp`: payloads, pushes and
frictions had never been tested. Both gates, full 14-condition family:

**Gate 1 — CONFLICT: passes.** Optima split 0.70/0.80 *within* grade, payload AND push families.

**Gate 2 — CEILING: fails outright.**

| policy | steps | distance |
|---|---|---|
| static T=0.70 | 232.2 | +5.22 m |
| **static T=0.80** | **242.1** | **+5.83 m** |
| static T=0.90 (shipped) | 230.2 | +4.49 m |
| static T=1.00 | 226.0 | +4.03 m |
| **oracle scheduling** | **242.1** | +5.99 m |

**Perfect per-condition scheduling buys +0.0 steps (0.0%) and +0.16 m (2.7%).** The conflict is
between 0.70 and 0.80, which perform almost identically, so there is nothing to win. Gate 2 was
run BEFORE building the runtime knob — the §4o lesson applied in advance for once, and it saved
the build. (`gait_period_s` is baked in at env construction; adapting it online would have meant
driving `RomOnlineMotionCommand.rom_phase_lf/rf` directly.)

**What to ship instead: T = 0.80.** +11.9 steps and +1.34 m (**+30% distance**) over the shipped
0.90, averaged across all 14 conditions — terrains, payloads, pushes and frictions alike. At
`tilt+0.26` it is **250.0 / +15.72 m against 186.5 / +10.20**, i.e. **+63.5 steps in the very cell
where the confirmed adaptation result is +56.8.** Third independent measurement of that pattern.

### 4r. JOINT ENCODERS: the magnitude result was PREDICTED; the ratios flip sign anyway

The IMU-derived base state was screened and closed, but the encoders themselves never were —
`ẽ = q_des − q` and `q̇` had only ever been used to FORM `g_s`, never regulated.

**Stated before running, from the derivation:** for `h = ½‖ẽ‖²`,
`∂h/∂σ ≈ −½dt²·ẽᵀM_c⁻¹Sᵀk_p ẽ < 0`, so stiffening always reduces tracking error whether or not
stiffening is correct — aggregate servo error must be CONSTANT-sign and cannot discriminate.
**Confirmed:** `serr_mag` constant, `qd_mag` 1/41. Magnitudes are structurally useless here, and
that now covers every magnitude screened: three of seven base rows, both leg-power rows, and both
encoder magnitudes.

The **ratios** were the reason to run it — normalised differences cancel the common monotone
factor and leave the SHAPE of the load (hip/ankle for a grade, knee for a payload, one-sided for a
push). They do vary in sign (`qd_asym_LR` 26/16, `serr_asym_LR` 8/34), so the reasoning held — but
they do not align with the targets. `serr_asym_LR` scores **7/7 on payload and 8/23 on grade
against a 20 baseline**: the same condition-dependent reversal as roll (F4), `vx` (§4m) and
`load_excess` (§4n). LP infeasible; best raw p = 0.0242, ×6 rows = 0.145; no row beats the
within-family baseline.

**Rule 15: a magnitude cannot carry a direction.** Any quantity monotone in the adapted parameter
has constant-sign sensitivity by construction, so it can only ever say "more" or "less", never
"which way". Check monotonicity analytically before spending a sweep on it — it costs one line of
algebra and has now retired six candidate rows.

### 4p. LAYER CHOICE may matter, and `mlp.6` is the worst — prediction FAILED at 10 seeds (16 Aug)

CAS selects which layer is plastic, which presupposes the choice changes the outcome. That was
never tested: `mlp.6` was picked by hand (§7). Same law, same cell as the confirmed positive
(`slope15`, eta 3e-5, continuous), all four layers, `acc − off`:

| layer | params | Δsurvival | Δdistance | ‖ΔW‖ |
|---|---|---|---|---|
| **mlp.0** | 82,432 | **+40.0** | **+2.28 m** | 0.400 |
| mlp.2 | 131,328 | +33.8 | +1.99 m | 0.400 |
| mlp.4 | 32,896 | +17.3 | +0.68 m | 0.390 |
| mlp.6 | 2,580 | +15.5 | +0.67 m | 0.380 |

Displacement matched to 5% despite a 51× parameter spread, so this is not the layer-SIZE confound
`ProjectedLayerUpdater`'s docstring warns of; the ordering is monotone; distance tracks survival.
Harness validated by reproduction: `mlp.6` gives `acc − frand` = **+14.3**, exactly the figure §7
records independently for that cell.

**Higher-powered follow-up, prediction stated first (≥8/10 positive): FAILED.**

| | 6 seeds | 10 seeds |
|---|---|---|
| mean `mlp.0 − mlp.6` | +24.5 | **+26.2** |
| sign test | 5/6, p = 0.109 | **7/10, p = 0.1719** |

The mean rose; the sign consistency fell. Per-seed diffs `+86, +56, +52, +46, +28, +14, +4, −3,
−7, −14` — every negative small, four positives very large. Post-hoc and **NOT pre-specified**:
Wilcoxon p = 0.0254, paired t p = 0.0164. By §4f's rule ("if it misses, that is the result") this
is **NOT established**, and the reserved pool is not earned by a failed prediction.

**The methodological lesson is about the test, not the effect.** Every pre-registration in this
project has used a sign test, which discards magnitude. With effects spanning +4 to +86 that
throws away most of the signal, and a sign test is underpowered by construction for heavy-tailed
paired differences. **Rule 14: choose the paired test from the expected effect DISTRIBUTION, not
from habit — and state it before running.** The next run should pre-register Wilcoxon on the
selection pool (2000s), disjoint from both development and test and unused on this axis.

### 4o. INDIRECT adaptive control is closed too — and the prize was never large (15 Aug)

The direct form (descend a gradient on an objective) is closed by §4j/§4m/§4n. The obvious
alternative is INDIRECT: identify which disturbance you are in, then look up the optimal `kp` from
the measured 14×5 map. None of the objective screens bear on that — they tested instantaneous
read-outs of the DIRECTION, not identification of the CONDITION.

`scripts/identifiability.py`. Features are one full 27-step stride of proprioception at
`kp = 1.00` — the state a deployed robot is actually in before it decides. Prediction is 1-NN in
standardised feature space, so there is nothing to overfit. **Crucially the scoring is realised
control performance, not classification accuracy:** the sweep already recorded survival and
distance at every `(condition, kp)`, so a scheduling policy is evaluated offline by looking up what
it would actually have got.

| split | `kp*` correct | scheduled | best static | nominal | oracle |
|---|---|---|---|---|---|
| leave-one-CONDITION-out | 17/27 | 218.1 / 4.30 m | 219.3 / 3.84 m | 219.3 / 3.84 | 236.9 / 4.61 |
| leave-one-FAMILY-out | **3/27** | **139.9 / 2.82 m** | 219.3 / 3.84 m | 219.3 / 3.84 | 236.9 / 4.61 |

**Identification works WITHIN a family and fails ACROSS families** — 17/27 having seen other
payloads, 3/27 having not. The robot can tell "heavier than the payload I saw"; it cannot tell
"payload rather than grade". Leave-one-family-out is **−79.4 steps**, far worse than doing nothing.

**THE CEILING, and it corrects the framing of §4n.** The oracle — perfect identification, the most
any scheduler could achieve — beats the best static `kp` by only **+17.6 steps (7%)** and +0.77 m
(20%). And **the best static `kp` is 1.00, i.e. nominal**: averaged over the nine conditions with a
measured optimum, the shipped stiffness is already the best single choice.

So the impedance conflict is real — no single `kp` is optimal everywhere, and the optima span
0.70 to ≥1.30 — but **the value of resolving it on this condition set is 7% of survival.** By
rule 3 that is a cell where adaptation cannot produce a result, and the ceiling should have been
measured before the conflict was called "the case for the method". The same omission that §4f
punished (an effect above the documented ceiling) in a new place: **measure the ceiling of the
COMPARISON you intend to win, not just of the failure you intend to fix.**

Both routes are now closed on measurement rather than argument:
- **direct** — no read-out of base state or momentum/energy sensing, linear PSD or signed or
  nonlinear, phase-corrected, 42 constraints, three two-sided families, recovers the direction;
- **indirect** — identification does not generalise across disturbance types, and the ceiling is
  +17.6 steps even with perfect identification.

What survives is the theory session's statement arrived at from the other side: **the impedance
channel has genuine authority (~40×) over the quantity that forecasts falling, and nothing
observable tells you which way to use it.**

### 4n. The stiffness grid was the binding constraint on the ground truth (15 Aug)

Rule 13 immediately indicted the ground truth it came from. `within_family_screen.py` on the
3-level grid: of four families, **only `grade` contained both target signs.** Payload, friction
and push were single-class, so any constant rule scores perfectly there and nothing is tested —
**§4j's "21 constraints" was really 11 usable ones**, and inside those no row beat the 8/11
baseline (`leg_power` and `leg_power_LR` tie it exactly).

The cause was the grid, not the physics: capping `kp` at **1.00** meant no cell was ever evaluated
*above* its optimum, so any family wanting 0.85–1.00 could only ever generate "stiffen" targets.
Extending to `{0.70, 0.85, 1.00, 1.15, 1.30}`:

| condition | 0.70 | 0.85 | 1.00 | 1.15 | 1.30 | best |
|---|---|---|---|---|---|---|
| tilt −0.13 | **250.0 / +0.94** | 222.5 | 172.5 | 159.0 | 140.0 | 0.70 * |
| tilt +0.26 | 162.5 | **199.0 / +10.97** | 186.5 | 157.5 | 132.5 | 0.85 |
| payload25 | 33.0 | 94.0 | **250.0 / +4.29** | 250.0 / +3.39 | 114.0 | 1.00 |
| **push60** | 63.5 / +1.12 | 99.0 | 115.5 | 155.0 | **183.5 / +3.70** | **1.30 \*** |

\* boundary — flagged automatically now, since a best at the edge may not be the optimum.

**Two things fall out.** `payload25` has a genuine INTERIOR optimum (it collapses to 114.0 at
1.30), so 1.00 was a real optimum and not a grid artefact. And **`push60` wants `kp ≥ 1.30` and is
still improving at the edge** — survival rises monotonically 63.5 → 183.5 and distance +1.12 →
+3.70 m. **That widens the measured conflict from 0.70–1.00 to 0.70–≥1.30**, nearly 2×: grades want
compliance, lateral pushes want stiffness, and no static value can serve both. It strengthens
[[the impedance conflict]] rather than qualifying it.

Ground truth is now **22 survival-derived + 20 distance-derived = 42 constraints, 26 soften / 16
stiffen** (was 8/13), ratio 6.0× against 7 weights, and both `grade` and `payload` are two-sided.

**Rule 13: a candidate must discriminate WITHIN a condition family, not across families.** Across
families it can pass by reproducing the design — which disturbance types you happened to include
and what they happen to want. Within a family the disturbance type is fixed and only its severity
varies, which is the case a deployed controller actually faces. Report the within-family score
against the within-family baseline, and treat single-class families as uninformative.

### 4l. The IMPEDANCE channel has ~40× the authority — and `|pitchdot|` still fails as its target

`scripts/impedance_authority.py`, 6 seeds, 4 conditions, frozen policy, `kp ∈ {0.70, 0.85}` paired
against `kp = 1.00`. Identical statistic, windows, pitch definition and pairing as
`theory_2026-08-15/controllability.py`, so the numbers sit beside F6's directly.

**T1 — reachability: PASSES, and not marginally.**

| cell | kp | `\|pitchdot\|` ref | sd | mean Δ | **\|Δ\|/sd** |
|---|---|---|---|---|---|
| slope13_up | 0.70 | 0.191 | 0.008 | −0.059 | **7.26** |
| slope13_up | 0.85 | 0.191 | 0.008 | −0.027 | **3.27** |
| slope15 | 0.70 | 0.210 | 0.034 | +0.240 | **6.99** |
| slope15 | 0.85 | 0.210 | 0.034 | +0.039 | 1.13 |
| payload25 | 0.85 | 0.284 | 0.050 | +0.364 | **7.28** |
| payload15 | 0.85 | 0.139 | 0.033 | +0.139 | **4.22** |

**Impedance median 5.60, 6/6 cells above 0.5. Position channel: 0.14, 12/60 above 0.5.** About a
40× difference, and it is structural, not tuning — `∂τ/∂a = K_p diag(s)` is constant while
`∂τ/∂σ = K_p ẽ` scales with the servo error (`MODEL_AND_DERIVATIONS.md` §7.1). **This is the first
channel in the project with real authority over the quantity that forecasts falling.**

**T2 — payoff: FAILS, and my first pass overstated it.** The raw pooled Spearman is −0.896, but
that is *not* F6's statistic: F6 pools **within-cell standardised** ranks, which removes
between-cell structure. Recomputed correctly (`scripts/impedance_payoff_fix.py`):

    rho(Δ|pitchdot|, Δsurvival) = −0.067      [position channel, F6: +0.012]
    rho(Δ|pitchdot|, Δdistance) = −0.162

Per cell the sign **flips with condition class** — payload15 +0.486, payload25 +0.543,
slope13_up@0.85 −0.486, slope15@0.85 −0.714. Softening *lowers* pitch rate uphill (−0.059) and
*raises* it downhill (+0.240) and under payload (+0.364). **That is F4's roll pathology again:** a
quantity whose relationship to benefit reverses between condition classes cannot be a certificate,
however reachable it is.

**Net:** the channel problem is solved and the objective problem is not. Rule 12's two halves now
have different owners — `sᵀQs` fails predictive, `|pitchdot|` fails reachable *on the position
channel* and fails **sign-consistency of payoff** on the impedance channel. Combined with §4j (no
diagonal weighting of the base state carries the stiffness direction), the single remaining
blocker is **sensing**: a signal that distinguishes sagging-under-load from fighting-a-grade.
That is exactly the GRF / leg-work / duty-factor shortlist, and it is now the only thing in the way.

**Caveats.** The matured (100–200) window exists for only 1 of 4 cells — at `kp` 0.70/0.85 most
episodes end before step 200, so the early window carries the T1 claim. Both payload cells at
`kp = 0.70` have **zero** usable pairs (the robot falls before step 50, consistent with the 33.0
steps in the ground-truth sweep). n = 6 seeds per cell, so per-cell rho rests on 6 points. Mean
Δsurvival −5.7 steps against mean Δdistance +0.44 m — the pooled treatment is not uniformly good,
which is expected since it applies grade-appropriate softening to payload cells that want the
opposite.


### 4t. THE FEEDFORWARD CEILING: the observer's families are the ones the residual cannot cancel

§4s named the open question — composite adaptation fits `Φθ̂` to CANCEL `y`, so what fraction can a
bounded residual remove? **It was already measured on 29 July** by `scripts/probe_compensability.py`,
which happens to cover all three families the observer discriminates.
`theory_2026-08-15/feedforward_ceiling.py`, no simulator.

**First, a normalisation correction to that probe.** It stored `need = ‖r*‖₂ / u_max`. The residual
is saturated PER COMPONENT (`sat_{u_max}`), so the admissible set is the box `‖r‖_∞ ≤ u_max` whose
2-norm budget is `√n_a·u_max = 0.537`, not `u_max`. **The stored `need` overstates by √20 = 4.47×**
and the widely-quoted "6–54× the budget" should be **1.3–12×**. The `left` column was always correct
(it clips per component). The correction matters because only under it does `need < 1` line up with
`left ≈ 0`, i.e. only then does the column mean what its name says.

**The answer, on the observer's own three families:**

| family | observer `\|a\|` | axis | need (corrected) | `left` | **cancellable** |
|---|---|---|---|---|---|
| grade | 1.560 | longitudinal | 8.6× | **1.52** | **0%** |
| payload | 4.611 | vertical | 5.4× | **0.95** | **5%** |
| push | 1.219 | lateral | 5.2× | **0.86** | **14%** |

`left > 1` for grade means the bounded best-case correction makes the drift WORSE — the clip
destroys the direction. **Perfect identification changes none of this.** §4s solved the sensing
problem for a channel that cannot act on the answer.

**Positive control, so this is not a probe that says "nothing works":** `joint_offset 0.05` needs
0.26× and leaves **0.000** — 100% removable; `ground_friction 0.08` needs 0.80×, leaves 0.201.
Matched, in-budget faults resolve cleanly. The observer's families are neither.

**Cross-check — the budget excuse is closed independently.** Grade needs 8.6× the 2-norm budget.
§4b swept `u_max` 0.12 → 6.0, i.e. **50×, which brackets that requirement**, and survival still
degraded monotonically (−16.5 → −60.2). So budget and direction fail SEPARATELY: authority was
raised past what compensability demands and it did not help, because the target was wrong.

**Scope, and the first item cuts against the number being generous.**
- The probe cancels one-step task-error DRIFT, not `y`. Drift is post-rejection, `y` is
  pre-rejection (rule 16), so `‖y‖ > ‖d‖` and these figures are **optimistic** for feedforward:
  cancelling `y` needs strictly more.
- `PROBE_STATES=4` at `STATE_STRIDE=12` after a fixed settle samples phases {0,12,24,9} mod the
  realised 27-step stride — a fair spread, not a uniform stride average. §4m's caveat applies
  weakly; order of magnitude survives. n = 2 seeds × 4 states.
- This is the POSITION channel. Impedance compensability has never been measured — but its payoff
  ceilings are already 7% (§4o) and 0% (§4q).

**Verdict: the composite law's prediction term has a working signal and no channel to spend it on.**
Both terms are now closed for the position residual — the tracking term by §4j/§4m/§4r (no
observable carries the direction) and the prediction term here (the direction is now observable and
the residual cannot realise it). Note also that `probe_compensability.py`'s own docstring made the
matched-uncertainty argument on 29 July; §7.0 of `MODEL_AND_DERIVATIONS.md` restated it rather than
discovering it, and the project had the answer to §4s's question three weeks before asking it.

**Rule 17: before building a sensor, check the actuator's ceiling for what the sensor would tell
you.** Identification and cancellation are separate budgets and this project has now paid for the
first without the second twice (§4o's scheduler, §4s's observer).

### 4y. A SCREENED cell, pre-registered, and the residual still recovers nothing (16 Aug)

Run in an isolated workspace (`~/theory_ws/x2_ttcl`, see `theory_2026-08-15/WORKSPACE.md`) so
nothing here could touch the primary session's tree. **The workspace was copied at 19:16, after
§4w's `use_action_scale` fix landed at 15:56, and `acc_gain.py` is byte-identical to the primary
tree's — so this is the first pre-registered run to use the CORRECTED `g(x)`.** Pre-registration written before the run:
`theory_2026-08-15/PREREG_FRICTION_WEEK2.md`.

**Week 1 — a three-gate screen over 7 fault cells, passive arm only.** Gates fixed in advance:
real headroom, repeatable at n = 6, and not already known hopeless on compensability.

| cell | survival | capped | distance | cv | dist loss | compens. | verdict |
|---|---|---|---|---|---|---|---|
| nominal | 400.0 | 6/6 | 7.90 | 0.006 | — | — | reference |
| sensor10 | 400.0 | 6/6 | 7.65 | 0.014 | −3.2% | 100%* | flat |
| sensor15 | 360.2 | 5/6 | 6.26 | **0.427** | −20.8% | 100%* | noisy |
| **friction_ood** | **400.0** | **6/6** | **5.35** | **0.039** | **−32.3%** | **80%** | **PASSES** |
| payload25 | 400.0 | 6/6 | 6.94 | 0.020 | −12.2% | **5%** | hopeless |
| actuator | 109.0 | 0/6 | 2.82 | **0.224** | −64.3% | 27% | noisy |
| delay_legs2 | 186.5 | 0/6 | 2.28 | **0.406** | −71.1% | untested | noisy |

\* analogue join (observation-side bias vs a command-side offset), not the same fault.

**The screen's own outcome variable was wrong, and it nearly discarded the winner.** Gated on
SURVIVAL, `friction_ood` scores a flat 0.0 loss — 400/400 on 6 of 6, indistinguishable from no
fault. Survival is **censored at the `n_post` cap in 4 of 7 cells** and has no resolving power
there. Distance is uncensored and extremely tight here: nominal `cv = 0.006`, `friction_ood`
`cv = 0.039` against a 32% effect. Note also that the two cells with the LARGEST headroom
(`actuator` −64%, `delay_legs2` −71%) are the two that fail repeatability — headroom without
repeatability is worthless, which is the trap `shift_conditions` documents.

**Week 2 — pre-registered on the one surviving cell. FALSIFIED.** `friction_ood`, `mlp.6`,
established configuration (only the fault differs), `selection` pool 2000–2007 (disjoint from the
Week-1 screen and from the reserved test pool), primary endpoint forward distance, one-sided
Wilcoxon, predicted **+0.5 m** and ≥6/8.

    acc − off = −0.070 m,  2/8 positive,  p = 0.9258        survival +0.0 (8/8 capped both arms)

**What is falsified is precise: 80% one-step compensability does NOT convert into forward
progress.** The residual demonstrably *can* act on this fault — measured on this exact fault,
`left = 0.201`, corrected `need = 0.80×`, inside budget — and it recovers **none** of the 2.55 m
the fault costs.

This is the strongest negative in the file because **the cell selection cannot be blamed**:
`friction_ood` was the only cell of seven to clear a pre-declared screen on the project's own best
criteria, not a cell picked by intuition.

**The `acc − frand` trap fired a third time.** `acc − frand` = +0.44 m reads as a win and is
manufactured entirely by seed 2005, where `frand` collapsed to 2.03 m against `off`'s 4.92 m
(+2.81 from that seed alone); `frand − off` = −0.33 m overall. Excluded from the decision by the
pre-registration, per §4u.

**Exploratory `mlp.0`: +0.140 m, 5/8, p = 0.0742.** Below the predicted threshold, not
significant, and pre-declared non-confirmatory because §4p's layer prediction failed at 10 seeds.
**Not a result.** It recovers 5.5% of the fault's cost, so the ceiling argument applies regardless.
No rescue runs were made at another `u_max`, `eta`, `b_W`, layer or larger `n`.

**How this must be read against §4x, which was written independently the same evening.** §4x
measures the corrected `g(x)` as the WEAKER configuration on the confirmed cell: `acc − off` falls
+16.2 → **+1.5** when `diag(s)` is restored. This run used the corrected gradient. So the null here
is *consistent with* §4x rather than independent of it, and two honest consequences follow:

- **The +0.5 m prediction was badly calibrated.** It was written without knowledge of §4x; had the
  corrected law's effect on the confirmed cell (+1.5 steps) been known, predicting 20% recovery on
  a new cell would have been unreasonable. The falsification of the *compensability* claim stands —
  80% removable drift, −0.07 m recovered — but the prediction should not be cited as a calibrated
  forecast that missed.
- **It corroborates §4x from a second, independently screened cell.** §4x has one cell (`slope15`,
  p = 0.0596, marginal); this adds `friction_ood`, selected by a pre-declared three-gate screen,
  where the corrected law also delivers ~nothing. Fidelity being anti-predictive now has two cells
  rather than one.

Whether the pre-fix gradient would have scored positively here is **untested and must not be
assumed**; §4x's direction suggests it might, which is a reason to run it as its own pre-registered
contrast, not a reason to discount this one.


---

## 4z. The gradient protocol (17 Aug): six bugs fixed, and the map is now CERTIFIED in direction

Working through `docs/GRADIENT_MAPPING_AND_MATCHING_CONDITION.md` §15's required order. Six
defects found and repaired, one earlier claim corrected, and — for the first time — the analytic
map passes a real acceptance protocol rather than a single-epsilon cosine.

### The six defects

| # | Defect | Where | Consequence |
|---|---|---|---|
| 1 | Shared ROM YAML mutated in place, no lock | `rom_command.py` | Concurrent runs raced: A could restore the file while B was still constructing, so B walked at the vendor default with its `task_error` chasing something else |
| 2 | `set_command(1.0, 0, 0)` hardcoded while `--vx` set the reference | `online_adapt.py:416` | At vx=0.6, row 0 of every task error was `v_x − 1.0` — a permanent phantom deficit no residual could remove |
| 3 | `motor_kd_scale` restored `actuator_biasprm`, wiping the `−kp` term | `mjlab_backend.set_condition` | Every requested stiffness fault became a torque offset. Fault alone: 400/400 steps. Fault + `motor_kd_scale: None`: [115, 72, 99, 100] |
| 4 | `pop_matured(k)` called after `be.step`, so the realized gap was `h+1` | `online_adapt.py` | Every horizon study was off by one; at h=1 that is 100% error in the varied quantity |
| 5 | `acc.grad()` evaluated at the MATURITY state, multiplied by a residual Jacobian from `o_{k−h}` | `online_adapt.py` | Two different states in one chain rule |
| 6 | `D_fault` and `D_ctrl` both assumed to be `I` | `acc_gain.py` | Under a delay the true one-step `D_fault` is **zero**; under a gain fault it is that gain; a clipped or slew-limited component is zero. The update pushed hardest exactly where the fault had removed its authority |

Plus, within `acc_gain` itself: the world-vertical row 5 used bare `self.kp` while rows 0–4 used
`kp·scale`, so the 6-vector was not a single linear map; `K_p` was cached before faults were
applied, describing the healthy robot; and both ankle bodies were pinned at all times, so during
single support (**90% of this gait**) the SWING foot was constrained.

### What the protocol measures (`scripts/gradient_protocol.py`)

Replay gate → epsilon plateau → coordinate finite differences → 32-direction holdout →
preregistered gates, contact-stratified. Nominal, 16 states, 2 seeds:

| gate | result |
|---|---|
| replay gate | **bit-exact, 0.00e+00** — snapshot/restore is sound |
| epsilon plateau | found at 0.005 in 100% of states |
| holdout NRMSE / R² | 0.001 / **1.000** — the FD is a genuinely accurate local model |
| calibration slope | 1.000 |
| action-gradient cos | **+0.954** |
| descent sign | 100% |
| q norm ratio | **67.65** ← the one failure |

Old configuration (both feet pinned, no channel Jacobians) on the identical states: **+0.919**.
So the contact/vertical-row fixes are worth **+0.035** in the consumed-gradient cosine.

### The magnitude failure is a UNITS error, and it is the largest miscalibration found so far

`G` is an instantaneous **acceleration** map. The consumed quantity is `∂e_{k+1}/∂r_k`, and `e`'s
dominant rows are velocities — so the two differ by one policy step. `1/dt = 50`; measured **68**.
Direction is unaffected (that is why every past *direction* result stands), but **`eta` was
silently absorbing a factor of ~68**, and every bound ever stated in gradient magnitude — the
authority sweep, the `b_W` projection radius, the σ_max figures — was in the wrong units.

### Correction to this morning's read of defect 2

I first called the phantom command "severe" on the grounds that row 0 flipped sign (+0.046 →
−0.421) and carries the largest weight. Measured, that is **wrong on the metric that matters**:
`cos(q_true, q_phantom)` is **+0.982** with 100% descent-sign agreement and magnitude within 6%
(`scripts/phantom_command_impact.py`, 24 states). The phantom tripled row 0's share of `|Qe|`
(14% → 43%) without redirecting what the controller consumed. **The vx=0.6 results are not
invalidated.** It bites only where the true error is smallest — one state at cos 0.43, true
e₀ = +0.019.

### The replay gate holds under every tested fault, and the map is certified under three of four

`scripts/fault_replay_gates.sh`, 4 states each, contact-stratified. The point of running this
per fault is that a fault adds STATE the snapshot may not carry — `action_lag` keeps a full
action vector of filter memory, `action_delay_steps` keeps a queue — and neither was in
`snapshot()` until today.

| fault | channel Jacobian it exercises | gates | cos_q | norm ratio |
|---|---|---|---|---|
| `lag08` | `D_fault = (1−λ)I`, attenuates every channel | **8/8** | +0.969 | 1.45 |
| `actuator` | `D_fault = diag(gain)`, asymmetric | **8/8** | +0.920 | 1.51 |
| `kp_left` | changes `K_p` itself (tests `live_gains`) | 7/8 | +0.945 | **2.75** |
| `delay_legs2` | `D_fault = 0` on delayed joints at h=1 | 6/8 | **+0.545** | 2.96 |

**Replay noise is 0.00e+00 under all four** — the snapshot fix works and every finite
difference below it is signal.

Two things this says that were not known before:

- **A stiffness fault makes the map OVER-CONFIDENT.** `kp_left`'s direction is fine (0.945) but
  the magnitude is 2.75× too large, so an adaptive law believes it has ~3× the authority it
  has — precisely where the fault has taken authority away.
- **A delay fault has no one-step gradient at all.** The delayed joints have exactly zero
  one-step sensitivity, so `B[:, legs] = 0` and the gradient can only see the arms; the cosine
  collapses to 0.545 for that reason alone. Any authority or adaptation measurement on a delay
  fault must use `h > delay` or it is a statement about the horizon, not the fault.
  `fault_authority_oracle.py` now refuses the short-horizon case rather than reporting it.

### THE ACTOR JACOBIAN DECIDES WHICH LAYER CAN BE ADAPTED

`grad_W = J_(r,W)^T q` is **linear** in `q`, so the parameter-space angle between two gradients
is the `M`-weighted action-space angle, with `M = J_(r,W) J_(r,W)^T` (20×20):

```
cos(J^T q1, J^T q2) = q1^T M q2 / sqrt( q1^T M q1 . q2^T M q2 )
```

Measured (`scripts/actor_jacobian_conditioning.py`):

| layer | n_param | cond(M) | worst-case param cos at 0.954 action agreement |
|---|---|---|---|
| mlp.0 | 82,432 | 3.30e+04 | **−0.997** |
| mlp.2 | 131,328 | 9.20e+03 | −0.991 |
| mlp.4 | 32,896 | 1.45e+03 | −0.943 |
| **mlp.6** | 2,580 | **1.00** | **+0.954** |

`mlp.6` is the output layer: `r_w = (W−W_0)h + (b−b_0)`, so `∂r_j/∂W = e_j h^T` and
`M = (‖h‖²+1) I` — **isotropic by construction, not by numerical accident**. It is the only
layer at which a certified action gradient is guaranteed to remain a descent direction after
the chain rule.

This is already visible end to end: on `mlp.0`, with a valid ε_W plateau at H = 1 and 2, the
FD gradient descends the realized cost while the analytic gradient **climbs**, at
`cos_action = 0.954` and `cos_param = −0.12`. And it is state-dependent exactly as an
ill-conditioned `M` predicts — on the state where `cos_param` came out **+0.849**, both
directions descend.

**Consequence for the paper's own question.** The recorded reading of the layer contrast was
"`mlp.0` +42.4 beats `mlp.6` +16.2, so the hand-picked `mlp.6` is the worst of four." That is
incomplete: `mlp.6` is the only layer whose gradient is *trustworthy*, and `mlp.0`'s advantage
— if it survives a properly powered test — cannot be attributed to following the gradient
better, because on `mlp.0` the deployed gradient is not reliably a descent direction at all.

It also yields a method change rather than a caveat: **precondition by `M`** (a natural-gradient
step, `grad_W = J^T M^(-1) q`) and every layer inherits the property that makes `mlp.6` safe.
Pre-registered with a falsifier in `docs/PREREG_NATURAL_GRADIENT.md` before testing.

### The weight-update path WORKS — given the right action gradient (doc §10)

`scripts/parameter_gradient_validation.py`, `mlp.0`, 12 contact-stratified states, 2 seeds.
From each snapshot: form `grad_W = J_(r,W)^T (2 B^T Q e_(k+h) + 2 λ_u r)`, perturb the layer by
`± ε_W d_W`, roll the REAL loop forward H steps through the real controller and backend, and
require a NEGATIVE directional derivative. ε_W swept over {5e-4, 1e-3, 2e-3, 4e-3} because a
directional derivative with no plateau is a secant, not a derivative.

| H | analytic descends | **fd descends** | ε_W plateau (fd) |
|---|---|---|---|
| 1 | 58% | **100%** | 58% |
| 2 | 50% | **100%** | 58% |
| 5 | 33% | 75% | 50% |
| 10 | 75% | 67% | 17% |
| 27 | 58% | 50% | **0%** |

Three separate findings:

1. **The machinery is correct.** The finite-difference action gradient, pushed through the
   actor Jacobian, residual controller, fault channel and projection, descends the realized
   cost in **100% of states at H = 1 and H = 2**, against 33% for matched random. Nothing in
   the deployment path is miswired — a positive result, and a new one.
2. **The analytic gradient fails exactly where the actor Jacobian inverts it.** Where
   `cos_param > 0`, analytic descends **100%** (n=7); where `cos_param < 0`, **0%** (n=5).
   Twelve out of twelve, on a quantity computed from `M` alone. This is almost certainly the
   mechanism behind §4x — a *more accurate* `g(x)` making the controller *worse* (+16.2 → +1.5).
   Action-space accuracy does not transfer through an ill-conditioned `J`.
3. **Beyond H ≈ 10 there is no ε_W plateau in any state** (0% at H = 27). The realized cost is
   not differentiable in the weights at gait-cycle horizons, so *no* descent claim there is
   measurable, in either direction. This is the mechanism behind "the +55% was chaos": it was
   not that the effect was small, it is that the quantity has no derivative at that horizon.

Cost-accounting note: the first version of this scored the error BEFORE each step, which puts a
W-independent term in the sum and makes H = 1 vacuous (the only W-dependence left is the effort
term, whose optimum is `r = 0`). It also used the CURRENT error where the gradient of `ℓ_{k+h}`
needs the FUTURE one. Both were fixed before the numbers above; the first pass's table is void.

### The fault bands are ONE severity level wide

Re-screened on the fixed backend (30 cells, 4 seeds, nominal 250.0 steps / +4.90 m). Exactly 3
usable, and in every case the neighbouring severities are inert and lethal respectively:

| axis | inert | **usable** | too fast |
|---|---|---|---|
| `motor_kp_scale` | 0.7 → 250.0 steps | **0.5 → 93.8, +2.50 m** | 0.3 → 32.0 |
| `torque_limit_scale` | 0.7 → 250.0 | **0.5 → 165.5, +3.52 m** | 0.3 → 27.2 |
| `action_delay_steps` | 1 → 250.0 | **2 → 61.2, +1.01 m** | 3 → 24.8 |

**The policy is either untouched or dead, with one severity level in between.** That is worth
saying in the paper: it is why fault-cell selection has been so expensive, and it means any
fault result is sensitive to a severity choice that has almost no valid range.

### The sim-to-real axes are INERT on this policy

`scripts/sim2real_screen.py`, 4 seeds, 250 post-shift steps. Nominal: 250.0 steps, +4.90 m.

`armature_scale` 0.5/2.0/4.0, `inertia_scale` 0.8/1.25/1.5, `dof_friction` 0.05/0.15/0.30,
`dof_damping_scale` 0.5/2.0/5.0, `gear_scale` 0.8/1.2 — **every one gives 250/250 steps with
distance within 2% of nominal.** Thirteen cells, zero usable.

These are the canonical sim-to-real mismatch axes and the ones named as the paper's proxy for
the gap. The frozen policy was trained with domain randomisation over exactly these quantities,
so each one alone is IN-distribution for it.

### …but the CONJUNCTION breaks it, and that is the realistic sim-to-real cell

`--family combo`, same seeds and horizon:

| cell | steps | dist | full | verdict |
|---|---|---|---|---|
| `s2r_mild` | 250.0 | +4.77 m | 4/4 | no loss |
| **`s2r_moderate`** | **92.2** | **+1.75 m** | **0/4** | **USABLE** (median fall 86) |

`s2r_moderate` = armature 3.0, inertia 1.35, dof_friction 0.15, dof_damping 3.0, gear 0.85,
action_lag 0.8, action_delay 1. **Every one of those severities is individually inert** — each
sits inside a range measured at 250/250 steps on its own axis. Applied together they take the
frozen policy from 250 steps / +4.90 m to **92 steps / +1.75 m**.

**This is the correct reading, and it reverses the single-axis conclusion.** Domain
randomisation buys robustness to each axis marginally, not to the conjunction — and a real
hardware transfer differs on all seven at once, never on one. So the sim-to-real story CAN be
told on this walker; it just cannot be told one axis at a time. `s2r_moderate` is the test cell,
with the fall at ~86 steps leaving enough time for a monitor to act.

Methodological point worth keeping: **screening axes marginally and concluding "no usable
cells" was wrong**, and would have cost the paper its headline experiment. When each factor is
individually in-distribution, the interaction is exactly where the out-of-distribution behaviour
lives.

### The channel runs out of AUTHORITY — and it is not a matching failure (doc §11.2)

`scripts/fault_authority_oracle.py`. From each snapshot on the faulted rollout: `d_e` = the
drift the fault actually causes (same snapshot replayed with the fault cleared), `B_fault` by
central differences **on the faulted plant**, then the true box-constrained optimum
`argmin_{l≤r≤u} ‖d_e + B r‖²_P + λ‖r‖²` under the real deployment bounds.

| cell | ρ_lin | ρ_sim (realized) | gap | pinned | rank |
|---|---|---|---|---|---|
| `torque_limit 0.5` | +20.1% | **+19.9%** | +0.2% | 10.9/20 | 6/6 |
| `motor_kp 0.5` | +13.8% | **+13.9%** | −0.1% | 14.7/20 | 6/6 |
| `s2r_moderate` (h=2) | +2.0% | **+2.1%** | −0.1% | 13.0/20 | 6/6 |

Three things make this a measurement rather than a null. The **linear model is exact**
(gap ±0.2%), so the optimum is the real optimum. The **rank is 6/6 in every state**, so the
6-D task error is fully spanned and this is *not* a subspace-matching failure — it cleanly
separates the doc's two competing hypotheses and the answer is **bounds**. And the optimum is
**already saturated**, 55–75% of components on the boundary, still removing only a fifth of the
drift. A matched-norm gradient step scores NEGATIVE (−9.7%, −8.6%) while the constrained
optimum helps, which is what to expect when most of the solution sits on the boundary.

#### …but the limit is the BOUND, not the channel — and this corrects the sentence above

The obvious reading of those three rows is "the position channel cannot recover these faults."
That reading is **wrong**, and the sweep that separates the two claims is nearly free, because
`B` is already measured and only the nonlinear replay costs anything. On `torque_limit 0.5`,
scaling the whole bound envelope (cap *and* slew together):

| `u_max` | ρ_lin | ρ_sim | ‖r*‖ | pinned |
|---|---|---|---|---|
| 0.05 | +12.1% | +12.1% | 0.133 | 11.1/20 |
| **0.10 (deployed)** | +20.1% | **+19.9%** | 0.265 | 10.9/20 |
| 0.20 | +39.7% | +39.3% | 0.531 | 10.4/20 |
| **0.40** | +77.8% | **+76.9%** | 1.018 | 9.4/20 |
| 0.80 | +91.6% | +76.9% | 1.954 | 6.7/20 |

**Recovery scales almost exactly linearly with the bound** — doubling `u_max` roughly doubles
recovery all the way to 77%, comfortably past the §11.4 authority gate of 50%. So the actuation
direction exists and is simply **capped**. "There is nothing for any bounded position residual
to recover" is true at `u_max = 0.10` and false as a statement about the channel; stating it the
second way would have retired a live direction.

The linear model and the plant agree exactly up to `u_max = 0.40` (77.8% vs 76.9%) and diverge
at 0.80 (91.6% vs 76.9%) — so 0.40 is roughly the largest bound at which the one-step
linearisation is still trustworthy, and the 0.80 row is the linearisation breaking, not a
structural plateau.

**This is not a licence to raise the cap.** §4b measured survival degrading *monotonically* as
deployed authority rose (−16.5 → −60.2 over a 50× sweep) while the Lyapunov decrease condition
held at every level. Put together, the two results say something sharper than either alone:
**the cap is doing stability work, not actuation work.** The channel has the authority; the
objective does not know how to spend it without falling over. That is the same conclusion the
four independent lines in §4x reached, now with a number on how much authority is being left
on the table.

**And it is still one-step.** §11.1: pointwise cancellation is sufficient for the classical
argument and NOT necessary for task recovery — a controller can win over many steps by changing
foot placement and contact timing. `scripts/horizon_oracle.py` is what settles that.

Trap found while building the sweep, worth remembering: at a snapshot `prev = 0`, so the
effective bound is `min(u_max, rate_limit)`. Raising `u_max` alone above `rate_limit = 0.08`
changes nothing and manufactures a **fake plateau attributable to the other constraint** — which
would have produced exactly the "structural limit" conclusion the real sweep refutes. Scale the
whole envelope together.

### THE RESULT: on a cell where recovery is PROVEN available, the objective carries none of it

This is the strongest negative the project has produced, because for the first time every
competing explanation was eliminated by measurement rather than argued away.

**The dissociation between the two fault cells** (`scripts/horizon_oracle.py`, 3 seeds, H=250,
harmonic search warm-started from the constant so it can only win by beating it):

| cell | best constant | best harmonic | matched random | authority gate |
|---|---|---|---|---|
| **tq05** (`torque_limit 0.5`) | **+100%**, 250/250 steps, +6.45 m | +100%, +6.64 m | +16% | **PASS** |
| `kp05` (`motor_kp 0.5`) | +16% | +21% | −0% | FAIL |

So `kp05` genuinely has nothing to recover — a null there is uninformative — while on `tq05` a
bounded **constant** residual fully recovers the fault and walks *further than nominal*.

**Adaptation on tq05, eta swept over 1600×** (`scripts/eta_recalibrate.sh`, selection pool):

| eta | ‖dW‖ | survival vs frozen |
|---|---|---|
| 3e-5 | 0.005–0.010 | −22.2 |
| 3e-4 | 0.025–0.037 | −23.0 |
| 2e-3 | **0.400 (pinned at b_W)** | −56.0 |
| 1e-2 | **0.400 (pinned)** | −9.2 |
| 5e-2 | **0.400 (pinned)** | −1.8 |

Never positive, including the entire regime where the projection binds — i.e. the regime the
method was originally characterised in.

**And the objective does not point at the fix** (`scripts/objective_vs_oracle.py`, 16
contact-stratified states, 2 seeds):

| quantity | value |
|---|---|
| `cos(−GᵀPe, r* − r)` all joints | **−0.099** median, range [−0.252, +0.207] |
| same, legs only | **−0.191** |
| per-component sign agreement | **50%** — exactly chance |
| states where the gradient points the right way | 38% |

The cosine is *more* negative on the legs, which are the joints that matter for locomotion.

#### The mechanism: the one-step optimum and the horizon fix are different objects

| | ‖r*‖ | linf | components pinned |
|---|---|---|---|
| one-step box-QP optimum | 0.265 | at bound | **11 / 20** |
| horizon constant (+100%) | **0.149** | **0.068** | **0 — strictly interior** |

The one-step objective demands a **large, saturated** correction to cancel instantaneous error.
The actual repair is a **small, interior** constant that reshapes the gait. These are not the
same solution at two scales; descending the former moves away from the latter. That is why an
accurate gradient of the one-step cost is worthless here, and it reconciles the two authority
numbers (20% saturated instantaneously vs 100% interior over a horizon) that otherwise look
contradictory.

#### What is eliminated

| candidate explanation | why it is dead |
|---|---|
| the gradient is inaccurate | certified 8/8, cos 0.954 vs contact-stratified FD |
| the channel lacks authority | horizon oracle **+100%**, 3/3 seeds |
| the bound is binding | the fix is **interior** (linf 0.068 < u_max 0.10) |
| the step size is wrong | 5 etas across 1600×, incl. the projection-bound regime |
| the layer's gradient is mis-conditioned | measured (cond M) and fixed (natural gradient) |
| **the objective** | **sign agreement with the fix = chance** |

The paper's honest framing follows from this: the contribution is a **protocol that localises
why test-time adaptation fails**, plus the localisation itself for locomotion — the tracking
objective is not aligned with recovery, and no amount of gradient accuracy repairs that.

### Contact modes, measured rather than assumed

`scripts/contact_mode_census.py`: **90% single support, 10% double**, double-support onsets every
13.5 steps → gait period **27 steps**, independently confirming the recorded value. The first
protocol run drew 16/16 single support *by chance*, not by a detection bug. A map validated only
in single support is not validated for walking, so `--stratify` now walks to the target mode
instead of sampling blind.

### 4aa. OOD MARGINALS are not the answer — five retune wins, and the shipped stiffness is wrong (17 Aug)

Isolated workspace (`theory_2026-08-15/WORKSPACE.md`). Three screens, passive arm throughout.
**Note: `z` was the last free section letter under this scheme.**

**Premise being tested.** `theory_2026-08-15/WHEN_ADAPTATION_WORKS.md` argued `V_adapt` is small
*by construction* wherever the policy was trained (RL with domain randomisation returns
approximately `θ_static`), and therefore predicted it would be LARGER off-distribution. The
backend's own comments say training randomises **foot friction 0.3–1.2, base-CoM offset and
encoder bias, and nothing else** — so payload, joint friction, damping, inertia, lag and delay are
all documented OOD.

**Result: the prediction fails.**

| family | `V_adapt` (distance) |
|---|---|
| trained conditions (grade/payload/push/friction, §4o) | **7%** |
| hardware faults, extended grid, uncensored | **+0.47 m (5.3%)** |

Off-distribution created no extra room **on SINGLE-AXIS faults** — and rule 22 (§4z) is the reason that framing is incomplete: every one of 16 single-axis sim-to-real cells is inert, while `s2r_moderate`, seven individually-harmless mismatches applied together, takes the frozen policy from 250 steps to **92**. So this section measures the MARGINALS, and the conjunction is where the out-of-distribution behaviour actually lives. Read the 5.3% as "no single OOD axis creates room", never as "OOD creates no room". `V_adapt` on survival was +0.00 — but survival was capped
6/6 in most cells even at `n_post = 500`, so only the distance number is admissible (rule 18).
**OOD is necessary and not sufficient**: `payload` is explicitly OOD and has no headroom at all.

**The lag cell, and why it looked like the answer.** `ood_screen.py` over 10 documented-OOD cells
found exactly one clearing both gates: `action_lag 0.8` — −11% distance, `cv 0.014`, no falls.
Uniquely, it has **no fixed-weight repair**: the compensation for a first-order lag is a *lead*,
which no offset and no stiffness can produce but `r = ΔW z(o)` can. `rescue_search` had
independently flagged `lag08` as the one fault of 21 worth rescuing, and the compensability screen
had excluded it on a technicality. It was the best-screened cell in the project.

**It died to a static retune, and by a wide margin:**

| lag | loss vs nominal | best static kp | recovers | % of loss | LEFT |
|---|---|---|---|---|---|
| 0.60 | 0.60 m | 1.30 → 11.40 | +2.13 | **357%** | −1.53 |
| 0.70 | 1.05 m | 1.30 → 10.78 | +1.96 | **187%** | −0.91 |
| 0.80 | 1.12 m | 0.85 → 10.50 | +1.75 | **157%** | **−0.64** |

A retuned lagged robot walks FURTHER than nominal. The retune does not invert the lag — `kp = 1.30`
is simply better than `1.00` lagged or not, and the "loss" was inflated by a bad reference.

**THE FINDING: the shipped stiffness is wrong, and it is the largest lever in this project.**

    nominal, no fault:   kp 1.00 -> 9.86 m     kp 1.30 -> 11.34 m    (+15%)
    hardware faults:     kp 1.00 -> 7.29 m     kp 1.45 ->  8.87 m    (+22%)

That is the **fifth** independent retune win — gait period (§4q), `kp` on trained conditions (§4o),
`kp` on hardware faults, `kp` on nominal, `kp` on lag. Every axis, same answer. And 4 of 8 fault
cells still optimise at the grid edge (1.60), so **+22% is a lower bound** — the third time a grid
has been too narrow here.

**What survives as genuinely new.** `lag0.8` optimises at **0.85 (softer)** while nominal, 0.6 and
0.7 optimise at **1.30 (stiffer)**, with an interior optimum and a collapse at 0.70 (101 steps).
That is the first clean two-sided conflict measured in this project, and scheduling is worth
**+1.03 m (11%)** on that one condition. But across the lag family `V_adapt = +0.343 m (3.2%)`, and
it is a lookup-table gain that §4o already showed does not identify across families.

**A dead knob.** `dof_damping_scale` is inert: `dof_damping` is **identically zero (0 of 26 DOFs)**,
so scaling it does nothing. Both damping cells returned numbers byte-identical to nominal, which is
rule 5 catching it. Damping lives in the controller's `K_d`, not the MJCF. Remove it from the fault
vocabulary.

**Rule 20: baseline every condition at its OWN best static setting before calling anything a loss.**
A deficit measured against a suboptimal default is the default's deficit, not the condition's. Here
it inflated `lag0.8`'s apparent headroom from **0.84 m to 1.12 m** and made a dead cell look like
the best candidate in the project. This is §4u's trap in screening form: §4u caught it in a
*result*, rule 20 catches it in the *screen* that selects what to test.


---

## 4aa. The velocity row is ANTI-informative -- replicated three times (18 Aug)

Solving for the error vector that WOULD point at the recoverable fix
(`scripts/reachable_by_any_objective.py`, `e_ideal = pinv(G^T) r*`) gives the same answer in
conditions that share nothing else:

| row | tq05 ideal | tq05 shipped | slope15 ideal | slope15 shipped |
|---|---|---|---|---|
| **vx** | **+0.820** | **-0.103** | **-0.411** | **+0.921** |
| vy | +0.460 | -0.067 | +0.485 | -0.011 |
| wz | +0.011 | +0.155 | +0.004 | -0.212 |
| height | -0.129 | -0.054 | +0.245 | -0.002 |

Under `tq05` the robot is torque-limited and too SLOW, so the objective pushes it to speed up
and the repair wants the opposite sign. Downhill on `slope15` gravity makes it too FAST, the
objective pushes it to slow down, and the repair *again* wants the opposite sign. Uphill, the
third instance, measured cos **-0.96** between the descended direction and the correction that
works. **Three conditions, opposite velocity-error signs, same failure.**

### The explanation, and it generalises beyond this robot

The base policy is a TRACKING policy. It is already spending its whole 160-input network on
holding the commanded velocity. An adaptive residual whose objective is dominated by the same
velocity error is therefore asking for MORE of what the base policy is already doing -- through
a far weaker channel (a linear map from a 6-vector). That is redundant when it is harmless and
destructive when the base policy is already saturated, which under a fault it is.

**Design rule: an adaptive residual layered on a trained policy must not regulate the quantity
that policy already tracks.** The base policy owns that term; the residual's job is what the
base policy is NOT handling -- here posture and height, which the shipped objective weights at
essentially zero (`height` shipped -0.002 against an ideal +0.245).

Directly testable and being tested: zeroing the velocity rows (`novel`) should raise alignment
above the shipped objective. If it does not, this explanation is wrong.

---

## 4ab. The loop-gain spectrum: the loop rejects the tracked rows, and chaos owns the long horizon (18 Aug)

`scripts/loop_gain_spectrum.py` (main tree), central-difference `B_h = de_{k+h}/dr_k` on the
FAULTED plant, 8 contact-stratified states, 2 seeds, run on `tq05` AND `slope15`.
Persistence `|B_h[row]| / (h |B_1[row]|)`: ~1 = the effect accumulates (the residual owns the
row); <<1 = the base policy's own loop is cancelling it.

| row | h=5 tq05 | h=5 slope15 | base policy tracks? |
|---|---|---|---|
| **height** | **0.84** | **0.93** | **NOT tracked** |
| roll | 0.67 | 0.19 | implicit (balance) |
| wz | 0.66 | 0.50 | commanded |
| vy | 0.48 | 0.38 | commanded |
| **vx** | **0.33** | **0.38** | commanded |
| pitch | 0.21 | 0.24 | implicit (balance) |

- **Within the Lyapunov time (h = 2–10) the only row that keeps its authority is `height` —
  the one row the policy does not track.** `vx`, the shipped objective's dominant row, is
  attenuated ~3× by h = 5. Replicated on two conditions that share nothing else. This is the
  closed-loop sensitivity mechanism (`de/dr = (I+GK)^{-1}G`) measured directly, and it is WHY
  the velocity row is anti-informative (§4aa) and why 48-cell objective reweighting sat at
  chance: the shipped objective spends its weight exactly where the loop takes it back.
- **Do not design from the h ≥ 27 columns.** Ratios there grow to 4.4–9.1 — superlinear.
  From roughly the Lyapunov time (~25 steps) onward the FD response measures chaotic
  amplification, not steerable authority: the state moves a lot, but not predictably. The
  script's own printed suggestion ("build the objective from the h = 54 top rows") is
  over-claiming by its own data and should be read as h = 2–10 only.
- **What this does NOT license: a height-error objective.** The alignment screen already
  measured `height` at chance on both conditions, and the tq05 fix moves height by z = +0.14
  (§ what_the_fix_does). The ownable row carries no error signal, because the repair does not
  express itself there at one-step scale. Authority and information are different axes; a
  usable surrogate needs both, and no row has both.
- **Consequence — the within-episode chapter closes with a three-part measured mechanism:**
  (i) tracked rows: rejected by the loop (this section); (ii) the untracked row: nothing to
  regulate (max |z| = 0.37 across all 20 joints + 11 observables); (iii) beyond the Lyapunov
  time: no differentiable structure (§4z ε-plateau; chaos here). The only signal at the
  repair's own timescale is the realised outcome itself — hence intervention search on the
  realised metric (ACE carried online; `theory_2026-08-15/ACE_CONNECTION.md`) is not one
  option among several, it is the remaining channel.

## 4ab. CLOSED: the walker objective cannot be fixed by reweighting -- two distinct reasons

120 objective x horizon cells across two conditions, every one at chance (45-55% sign agreement
against the oracle correction). Including four variants built specifically from the ideal-error
solve. The direction is closed, and the two cells fail for structurally DIFFERENT reasons.

### The construction error, first

`ideal_ish` and friends were built by setting **Q to look like `e_ideal`**. Wrong: `Q` multiplies
`e`, so producing `e_ideal` needs `Q = e_ideal / e_raw`. Those four variants did not test the
hypothesis they were built for. Solving it correctly is what produced the real finding.

### slope15 -- an OBSERVABILITY failure

| row | e_ideal | raw e | **Q needed** |
|---|---|---|---|
| vx | -0.411 | +0.921 | -0.4 |
| **vy** | **+0.485** | **-0.0107** | **-45.4** |
| roll | -0.174 | +0.0194 | -8.9 |
| **height** | **+0.245** | **-0.0033** | **-75.0** |

The ideal direction needs `vy` and `height` components that the measured error barely contains.
Producing them takes 45-75x gain on signals of order 0.003-0.01 -- amplifying numerical noise.
**`Q` can only rescale what is present; it cannot manufacture a component that is absent.**

### tq05 -- a REACHABILITY failure

Different mechanism entirely: only **36%** of the correction lies in `range(G^T)`, so the ceiling
on every objective of the form `-G^T(Qe)` is 0.359 regardless of weighting. The fix is not in the
span at all.

| cell | in range(G^T) | failure mode |
|---|---|---|
| tq05 | 36% | **reachability** -- cannot reach the fix |
| slope15 | 58.8% | **observability** -- can reach it, cannot see it |

### The lesson that generalises

Both failures come from INHERITING an error definition and hoping it happens to contain what the
repair needs. The screen that would have caught either on day one -- sign agreement against a
known oracle correction, cost seconds -- was built on day three, after 120 cells. **Screen the
objective against the oracle BEFORE searching over objectives.**

---

## 4ac. The ceiling is ARCHITECTURAL (~55%), not a property of the disturbance

The reachability sweep across five disturbance families settles what the earlier two-cell
comparison could not.

| condition | ceiling | shipped achieves | ideal sign agreement | in range(G^T) |
|---|---|---|---|---|
| slope15 (downhill) | 0.588 | **+0.229** | 67.5% | 58.8% |
| kp05 (stiffness) | 0.559 | +0.011 | 70.0% | 55.9% |
| actuator (gain) | 0.533 | -0.005 | 65.0% | 53.3% |
| slope15_up (uphill) | 0.505 | -0.073 | 72.5% | 50.5% |
| tq05 (torque limit) | 0.359 | -0.099 | -- | 35.9% |

**Five very different disturbances, ceilings all in 0.50-0.59.** The limit is therefore NOT a
property of the disturbance, and the hope that motivated the sweep -- "find the condition where
the fix happens to be reachable" -- was misconceived. **A 6-dimensional task error driving a
20-dimensional action space caps the expressible correction at ~55% regardless of what went
wrong.** That is the architecture, not the cell.

Two riders. The IDEAL error reaches 65-72% sign agreement in every cell, so real room exists
above the 50% floor -- but sec 4ab showed claiming it needs 45-75x gain on near-zero signals.
And **uphill scores the lowest ceiling with a negative shipped alignment**, consistent with its
15 recorded nulls: the measure does not contradict known outcomes even where it cannot predict
success.

### And the correction is not summarisable by ANY observable

`scripts/what_the_fix_does.py`, tq05: the largest shift in any regulated quantity is
`pitch_angle` at **z = -0.37**, and **0 of 20 joints move by more than 1 sd**. The repair is a
small distributed change; it does not work by moving height, posture, or any single quantity.

This kills the "regulate the untracked quantity" hypothesis that the loop-gain result suggested.
There is no low-dimensional observable to build an objective from -- exactly what a ~55% ceiling
implies, since most of the correction lives outside anything a 6-vector can express.

**Caveat on the measurement's own design:** both arms were scored over the FROZEN arm's survival
window, so late-emerging differences are invisible -- and the fix's whole benefit is surviving to
250 where the frozen arm dies at 103. That window is the right control (it stops the treated arm
being credited for the extra steps it bought) but it makes these z-scores a LOWER BOUND.

### Why this is the cleanest argument for the cross-embodiment direction

The walker's ~55% ceiling exists because a disturbance correction has no reason to lie in the
span of a 6-D error map. Proposition 2 of `docs/PLAN_CROSS_EMBODIMENT.md` says the
cross-embodiment correction is `r* = C a` -- a LINEAR MAP OF THE ACTION -- which an affine action
head expresses **exactly**, by construction, not at 55%. That contrast is now measured across
five conditions rather than argued. Test 2.3 of the plan computes the same number for the VLA
case; if it returns ~55%, the new direction has the same disease and should be dropped there.

---

## 4ad. THE POSITIVE: a MULTIPLICATIVE fault needs a WEIGHT, not a bias -- and the
cross-embodiment derivation is confirmed in miniature (19 Aug, parallel session `theory_ws`)

`outputs/param_route_g0.5.json`. Fault `joint_gain 0.5` on the legs, train 2000-2001, held-out
3000-3005, parameter route (UNCLIPPED).

| arm | steps | distance | recovery |
|---|---|---|---|
| nominal | 300.0 | +5.91 m | -- |
| frozen | **44.3** | +0.79 m | -- |
| **w6** -- mlp.6 **weight** | **231.3** | **+6.31 m** | **107.7%** |
| b6 -- mlp.6 **bias** | 46.0 | +0.84 m | **0.9%** |
| b0 -- obs-channel | 210.3 | +4.44 m | 71.1% |

44 steps -> 231, full recovery, distance exceeding nominal. The ACE control (random draws at the
same scale) gives **-0.089 m** against w6's ~+5.5 m, so this is not a large-perturbation artefact.

### Why this is the derivation confirmed, not merely a win

A `joint_gain` fault is exactly a **diagonal embodiment mismatch**: `a_env = g (*) a`, i.e.
`G_B = G_A diag(g)`. That is the structure of Proposition 2 in
`docs/PLAN_CROSS_EMBODIMENT.md`, written the day before this result existed, which states:

> `r* = C a` is a LINEAR MAP OF THE ACTION, realised by `dW = CW, db = Cb`.

A multiplicative mismatch therefore needs a **weight** change; a bias cannot express it.
**Measured: weight 107.7%, bias 0.9% -- a hundredfold dissociation in the predicted direction.**
And `cos(w6_found, analytic) = 0.604`: the analytic `C` from equation (3) matches the update the
search found.

### It retro-explains every earlier failure in this project

All prior walker adaptation injected an **additive bounded residual**. Against a multiplicative
fault that channel structurally cannot express the repair -- which is why `b6` scores 0.9% here
yet 68% on `tq05` (torque SATURATION, not a pure gain). **The fault's structure determines which
parameter expresses the fix**, and that is mechanistic rather than a tuning observation:

| fault structure | example | the fix lives in |
|---|---|---|
| additive | sensor bias | the **bias** |
| multiplicative | joint gain, embodiment mismatch | the **weight** |
| saturation | torque limit | partially both (`b6` 68%) |

### Caveats that must travel with the number

- **UNCLIPPED.** `contract: parameter route, UNCLIPPED`, `mean_dr = 17.5` for w6. This is NOT
  authority-matched to the +/-0.10 residual envelope every earlier result used. Defensible -- a
  weight change is a different contract from an injected residual -- but not comparable to those
  numbers without saying so.
- **Held-out 3000-3005 again**, the same set as the `online_const` v2 run. The test pool
  (4000-4009) is still clean and should be spent on the headline.
- Train on 2000-2001 (selection pool) is correct.

**Consequence for the paper: Proposition 2 now has empirical support on a real robot with a real
multiplicative mismatch, BEFORE any VLA work.** The cross-embodiment plan is no longer
speculative -- its central claim is validated in miniature.

---

## 5. Retractions

**The criterion "separates" slope15 from tq05 (18 Aug, same day).** I reported that the
structural ceiling separated the known positive from the known negative (0.588 vs 0.359,
shipped alignment +0.229 vs -0.099) and called it a validated predictor. It is not. The
criterion separates the two cells on a METRIC; it does not separate them on OUTCOME, because
**adaptation fails on slope15 too**. Re-run on the corrected pipeline (mlp.6, n_post=400,
10 development seeds, three learning rates spanning the pinned-`dW` regime), `acc` LOSES to the
displacement-matched control at two of three etas (+5.1 vs +10.9; +3.3 vs +23.0; +13.0 vs +0.7)
with 4-6/10 seeds improved -- chance. So the criterion currently has one confirmed negative and
one FAILED positive prediction, which is not a predictor. Letting a metric separation stand in
for an outcome separation is the same error class as reporting a treatment without its control.

**+56.8 held-out downhill (11 Aug) does not reproduce on the corrected pipeline (18 Aug).**
It predates the dt units fix (which moved `eta` by 68x), the contact/vertical-row fixes, and
`D_fault`/`D_ctrl`. Re-earned rather than cited, it is not there: no eta beats the matched
random control consistently. It was already known to be fragile (it vanished under the
`gait_period_s = 0.80` rebaseline, -15.6, p = 0.53). **There is currently NO cell in which
adaptation beats a displacement-matched control on the corrected pipeline.**

**+72.5 held-out (11 Aug).** Produced by a `g(x)` miswired three ways: **torso** Jacobian against
a **pelvis**-referenced error, **world** frame against a **body**-frame error, and **rows 2 and 5
transposed** (vertical velocity paired with yaw-rate error). It scored cos 0.23 against the error
it was actually descending, because the premise gate validated against `reference_error` while
every run used `task_error`. The effect replicated across two pools — it was real — but could not
be attributed to the stated mechanism. Repaired, the law still confirms at **+56.8**.

**"Roll is the fix."** A 2-trial CEM looked `hip_roll`-dominated; at 4 trials `hip_roll` alone is
worth ≤ +6.2 and the best constant is still sagittal.

**"The objective is anti-correlated with survival."** The *residual* is (cos −0.96); the
*gradient* is roughly orthogonal (+0.056). The harm enters downstream.

**Command adaptation "+16.5/+24.5/+19.0".** Survival only. On forward distance the adapter is
worse wherever climbing is possible (+3.20 vs +5.65 m at −0.05).

---

## 6. Rules adopted

1. **Run the control before reporting the treatment.** Both retracted positives had an obvious
   control identified in advance and run afterwards.
2. **Re-measure anything from ≤ 2 trials before building on it.** Two claims died this way.
3. **Screen by ceiling, not by trying adaptation.** Testing variants where the achievable margin
   is small cannot produce a result — that is what most of the uphill work did.
4. **Distance and survival together.** Either alone is gameable.
5. **Identical numbers across conditions mean a bug**, not a robust effect.
6. **Match the control's displacement**, and state `‖ΔW‖` for both arms when reporting.
7. **Check the termination flag BEFORE reading any state.** This trap has now cost four results
   (`fault_screen`, the grade metric, `online_adapt`, `gait_screen`) because the natural way to
   write the loop — step, read, test — is the wrong order. Report how many trials ran to full
   length per cell: a distance is only honest when nothing terminated, and printing that count
   makes a contaminated cell impossible to misread as a measurement.
8. **Every setting that changes WHAT WAS MEASURED goes in the filename AND in `config`.** Two
   runs that regulate different quantities are different experiments and must not share a path.
   This has now silently destroyed one sweep's raw data (§4b) after already having overwritten one
   arm set with another. A result you cannot re-open per-seed is a table, not a measurement.
9. **A confound you can name, you can measure.** Both §4d blockers — the missing pairs and the
   length-entangled ΔV — were answered from data already on disk, in one script, no simulator.
   Check what the logs already contain before scheduling a re-run.
10. **A screen that cannot fail has not been passed** (§4i). Before reporting any fitted object,
    run the exhaustive null over the target patterns and a leave-one-out. Both are free. The
    objective-family LP returned FEASIBLE with margin +0.356 — and so did 16 of 16 sign patterns.
11. **Compare against the majority-class baseline, and null the SEARCH, not just the result**
    (§4j). A maximum over many draws is itself a fitted object: 200k weightings reached 15/21, and
    the same search on SHUFFLED targets reached ≥15 in 70% of trials.
12. **A candidate quantity must be BOTH predictive and reachable** (§4k), and both are free.
    *Predictive*: on PASSIVE rollouts, a higher early value must forecast earlier failure at a lead
    beyond the fall timescale (53–66 steps), sign holding in EVERY condition. *Reachable*: the
    residual-induced shift must be an appreciable fraction of the quantity's seed-to-seed spread.
    `sᵀQs` fails the first (+0.10 at 2.6 s lead); `|pitchdot|` fails the second (0.14–0.21 sd).
    Neither failure needs a single adaptation run to detect.
13. **A candidate must discriminate WITHIN a condition family, not across families** (§4n).
    Across families a rule can pass by reproducing the experiment's design — which disturbance
    types were included and what they happen to want. Treat single-class families as
    uninformative and report the within-family score against the within-family baseline.
14. **Choose the paired test from the expected effect DISTRIBUTION, not from habit — and state it
    before running** (§4p). Every pre-registration here used a sign test, which discards magnitude;
    with effects spanning +4 to +86 that throws away most of the signal.
15. **A magnitude cannot carry a direction** (§4r). Any quantity monotone in the adapted parameter
    has constant-sign sensitivity by construction, so it can only say "more" or "less", never
    "which way". One line of algebra has now retired six candidate rows.
16. **Prefer signals measured BEFORE the loop rejects them** (§4s). A disturbance observer carries
    the disturbance; a tracking error carries what the controller failed to remove, which is a
    function of the operating point and therefore sign-unstable across conditions. Every candidate
    that flipped sign by condition was post-rejection; the one that did not was pre-rejection.
17. **Check the actuator's ceiling for what a sensor would tell you, BEFORE building the sensor**
    (§4t). Identification and cancellation are separate budgets, and this project has paid for the
    first without the second twice (§4o's scheduler, §4s's observer).
18. **State the outcome variable's censoring before screening on it** (§4y). If a variable is at
    its cap in the reference cell it cannot measure degradation and must not be the gate. Survival
    was capped in 4 of 7 cells and made the only viable cell look dead — the mirror image of the
    auto-reset trap, which made dead cells look alive. Rule 4 says report both; this says the
    CHOICE of outcome variable decides which cells appear alive at all.
19. **Reachability is necessary and not sufficient** (§4y). A channel that can move a quantity may
    still not move it in a direction the task rewards. Compensability answers "can the residual
    cancel this fault's one-step DRIFT" — a question about the drift, not the task. Measured at
    80% removable, the recovered distance was −0.07 m. Rule 12 therefore has a third clause:
    predictive, reachable, **and moving it must help the task**.
20. **A gradient certified in ACTION space is not certified in PARAMETER space** (§4z). The
    parameter-space angle is the `M`-weighted action angle with `M = J_(r,W) J_(r,W)^T`, and
    `cond(M)` runs from **1.00** at `mlp.6` to **3.3e4** at `mlp.0`. At the measured
    `cos_action = 0.954`, the parameter-space cosine can be anywhere from +0.954 to **−0.997**
    depending on the layer. `M` is 20×20 and costs 20 backward passes, so there is no excuse
    for not checking it before attributing anything to gradient quality.
21. **A directional derivative with no ε plateau is a SECANT, and its sign carries no
    information** (§4z, extending doc §8.5 from action space to parameter space). Beyond
    H ≈ 5 the realized cost has no ε_W plateau at all on this walker — so "the update
    descends/climbs" is unanswerable there, and any long-horizon descent claim (in either
    direction) is reading noise. This is the mechanism behind the July "+55% was chaos" result.
22. **Screen INTERACTIONS, not just marginals** (§4z). All 16 single-axis sim-to-real cells are
    inert on this policy, and the honest-looking conclusion — "no usable cells, wrong vehicle" —
    was wrong. `s2r_moderate`, seven individually-harmless mismatches applied together, takes
    the frozen policy from 250 steps to 92. Domain randomisation buys marginal robustness, not
    robustness to the conjunction; when every factor is individually in-distribution, the
    interaction is exactly where the out-of-distribution behaviour lives. A marginal screen
    would have cost the paper its headline experiment.
23. **A units fix silently re-tunes every hyperparameter multiplied by that quantity** (§4z).
    The dt correction divided `|g|` by 68 — correct, and it made the map pass its magnitude
    gate — but `eta` multiplies that gradient, so every `eta` tuned before the fix now takes a
    step 68× smaller. Caught only because the first seed showed `dW = 0.007` where
    pre-correction runs had `‖dW‖` **pinned at the projection radius** for every `eta ≥ 1e-4`.
    Running the comparison there would have produced a confident null caused entirely by the
    fix. **After correcting a scale, re-derive every constant downstream of it before
    interpreting anything** — and check a magnitude diagnostic (here `‖dW‖`) on the first seed
    rather than at the end of the sweep.
24. **A fault changes the gradient, not just the plant** (§4z). `D_fault = ∂a_env/∂r` was
    assumed `I`. Under a delay it is **zero** on the delayed joints at one step — so the
    one-step gradient literally cannot see them, and `delay_legs2`'s cosine collapse to 0.545
    is a statement about the horizon rather than the fault. Under a stiffness fault the map
    stays accurate in direction (0.945) but overstates authority **2.75×**, i.e. it is most
    over-confident exactly where authority has been removed.
25. **Baseline every condition at its own best static setting before calling anything a loss**
    (§4aa). A deficit measured against a suboptimal default is the default's deficit. It inflated
    a lag cell's headroom from 0.84 m to 1.12 m and promoted a dead cell to best-candidate. §4u
    catches this in a result; this catches it in the screen that chooses what to test.

---

## 7. Open

- **Is the confirmed +56.8 a retune?** `‖dW‖` pins at `b_W` from step 42 downhill (§4k), and a
  constant norm cannot distinguish "parked at a constant offset" from "rotating on the sphere".
  One logging line settles it: record `cos(dW_t, dW_{t-25})`. Near 1 ⇒ a retune found online, and
  the `gait_period_s` comparison in this section becomes fatal rather than awkward.
- **Run rule 12's reachability screen on the IMPEDANCE channel** before building an adaptive-kp
  law. The paired-arm method costs nothing beyond runs that already exist: does moving `kp` shift
  `|pitchdot|` by an appreciable fraction of its seed-to-seed spread? The position channel scored
  0.14–0.21 and was dead; if impedance scores high, the conflict result has somewhere to land.
- Adaptive **impedance** with a regulated quantity that has the right sign: driven by
  `task_error` it moved stiffness the wrong way (s → 1.16 where 0.70 is correct).
- **ACE layer attribution was never used** to pick the layer — `mlp.6` was chosen by hand. First
  run of it selects `mlp.0` uphill and `mlp.2` downhill, with all layers *positive* (no causal
  purchase) downhill, which does not yet line up with where the update works.
- **Forward lean: NULL** (14 Aug). Distance change −0.01 / −0.02 / −0.15 m out of 5–6 m at
  −0.03 / −0.05 / −0.07, with `lam` pinned at the +0.12 authority cap — so even a maximal lean
  does not change climbing. It also drove the OPPOSITE sign to the oracle's beneficial direction
  (`hip_pitch +0.12` vs the oracle's −0.10), the same sign inversion seen in the impedance
  channel.
- **Gait period: the first lever that actually CLIMBS** (14 Aug, 6 seeds). Shortening the ROM
  stride period `gait_period_s` 0.9 → 0.8 turns the −0.11 uphill from a fall at step 237 with net
  *negative* progress into **6/6 full episodes and +3.26 m of real forward distance**. Replicated
  a 3-seed screen (+2.45). Ordering 0.8 > 0.7 > 0.9, so nominal is a genuine local minimum.
  Everything previously tried could only *prevent* the failure by reducing demand; this is the
  first change that adds propulsion.
  **But it is a static retune and it dominates nominal EVERYWHERE**, which is the opposite of the
  stiffness result and removes the argument for adapting it:

  | terrain | 0.80 | 0.90 nominal |
  |---|---|---|
  | flat | +7.40 m, 6/6 | +6.91 m, 6/6 |
  | −0.11 uphill | **350.0, 6/6, +3.26 m** | 237.5, 0/6, falls |
  | −0.13 uphill | 305.7, 1/6 | 173.7, 0/6 |
  | +0.150 downhill | +16.12 m, 6/6 | +13.01 m, 6/6 |
  | +0.262 downhill | **289.3** | 185.2 |

  There is no terrain conflict anywhere, so 0.9 is simply a bad default. 0.7 is not the answer
  either — it collapses at +0.262 (134.7) and is worse at both uphill grades.

  **This is the serious problem for the headline result.** At +0.262 the retune is worth ~+104
  steps (185.2 → 289.3) where the confirmed adaptation gain is **+56.8** — roughly twice, from
  one scalar, with no adaptation at all. It does not invalidate the +56.8 (properly paired
  against a displacement-matched control, p=0.0156) but it changes its meaning: the adapter may
  be recovering ground a good default gives for free.

- **Does ACC still pay once the default is good? First answer: NO** (14 Aug, development pool,
  6 seeds, +0.262, eta 3e-5, continuous). Reading `acc − off`, which is the "does adaptation
  help" comparison:

  | period | off | acc | acc_frand | **acc − off** |
  |---|---|---|---|---|
  | 0.9 nominal | 187.7 | 203.2 | 188.8 | **+15.5** |
  | 0.8 good | 373.2 | 368.0 | 322.8 | **−5.2** |

  The gain tracks how bad the default is. Per seed it is a gamble, not a wash: 3004 goes
  239 → 400 (+161) and 3000 goes 400 → 208 (**−192**).

  **Do not read `acc − frand` here.** It is +45.2 at 0.8 and looks good, but `frand` *damages* the
  good default (373.2 → 322.8), so the gap means "acc harms less than a random direction harms".
  Displacement-matching fixes the magnitude confound; it does NOT make the control a floor when
  both arms are net-negative against `off`. Always report `acc − off` alongside.

  **Ceiling lifted (n_post=900) — the cap was hiding HARM, and the verdict is clearly negative:**

  ```
  seed    off    acc    frand      acc-off
  3000    751    208      370        -543
  3001    900    900      900           0
  3002    748    432      365        -316
  3003    900    717      445        -183
  3004    239    568      199        +329
  3005    900    900      203           0
  mean  739.7  620.8    413.7      -118.8     3 harmed / 1 helped / 2 tied
  ```

  `acc_frand` is worse still (−326.0), which is why `acc − frand` looked positive throughout.
  **Both arms damage a good baseline.**

  **The mechanism, and it is the important part:** the harness reports `acc` as *improved 5/6* on
  `delta_post` — the adapter drove its own cost DOWN by −0.55 while survival fell 119 steps. The
  law is working; it is descending the wrong quantity. At a bad default, reducing `task_error`
  correlates with staying upright, so adaptation looks beneficial; fix the default and the
  correlation breaks. This is the tracking-vs-survival anti-correlation returning, and it means
  the objective — not the update rule, the gain, the layer, or the gradient — is what is broken.

  Also: `acc` vs `frand` at 0.9 on the development pool is only +14.3 (5/6, one-sided sign
  p=0.109), much weaker than the +56.8 confirmed on the test pool.
- **Swing height, step-period multiplier and step width are INERT.** Identical to nominal in
  survival *and* distance across 7 cells. Three of the four gait knobs are dead ends.
- **Why the corrupted gradient outperformed the repaired one** (+69.3 vs +29.5). It descends a
  *permuted* error and replicated across two disjoint pools, so it is not noise.

---

## 8. The frame: channel, objective, mapping — and the order they must be checked in

Written 15 Aug, after §4j and §4k closed the two halves. This section adds no measurement; it is
the shortest statement of what the measurements in §1–§4k collectively mean, and it is placed last
because it only became sayable once both halves were in.

An adaptive law needs **three** things to be right, not two:

| | requirement | status on this system |
|---|---|---|
| **Channel** — what the action physically *is* | must have AUTHORITY over some quantity that predicts failure | ✗ position targets: induced shift 0.14–0.21 sd, specificity 56% vs 50% chance (§4k) |
| **Objective** — what is measured | must PREDICT failure at usable lead | ✗ `sᵀQs`: +0.10 at 2.6 s, and 0 of 2001 sign patterns feasible for the stiffness decision (§4j, §4k) |
| **Mapping** — `g(x)`, action → objective | must be ACCURATE | ✓ repaired, and it made no difference |

**The mapping is not the lever, and this is measured rather than argued.** If fidelity were what
mattered, a better `g(x)` would monotonically help. §4's table: corrupted `g(x)` at cos 0.208 gives
**+69.3**; repaired at cos 0.789 gives **+29.5**. Four-fold better alignment, worse outcome. The
L0→L3 ladder in `MODEL_AND_DERIVATIONS.md` §3 is correctly derived and was never the binding
constraint — see that file's §7.

**Saturation was the mapping working, not failing.** §4k found `‖dW‖` pinned at `b_W` within ~1 s
and held there. Read correctly, that is the gradient computing the right answer and reporting
*"push as hard as permitted; there is nothing here."* A perfect gradient of the wrong objective
through a channel with no reach is a perfectly accurate zero.

**~~The classical name is the MATCHING CONDITION.~~ — RETRACTED 17 Aug.** This paragraph claimed
payload is matched and a grade is unmatched because "a bounded `q_des` offset supplies no power".
**That is false**: a persistent target offset creates servo error and therefore sustained torque
and power, and `Range(SᵀK_pD_s) = Range(Sᵀ)`, so a position residual and a joint-torque
perturbation share the same instantaneous actuated span. Matching is state-, mode-, bound- and
horizon-dependent and must be MEASURED, not inferred from the words "position" and "momentum".
See `docs/GRADIENT_MAPPING_AND_MATCHING_CONDITION.md`, which supersedes it, and its §14.5: a
constant `hip_pitch = −0.10` recovered **+28.7 steps** uphill, so the channel was usable there and
the *update* failed to find the correction. What survives is narrower and still useful — the
robot nets backwards 8/8 seeds at both speeds on −0.13 (§4g) and the 50× sweep did not convert
(§4b) — but those are facts about those cells under the deployed law, not about the channel.

**The order, and this project ran it backwards.** The three are strictly dependent:

1. **Channel.** Is `∂(anything that predicts failure)/∂(action) ≠ 0`? — *never asked until §4k*
2. **Objective.** Among what the channel CAN reach, what predicts failure? — asked repeatedly
   (7 regulated quantities, the one-hot sweep of §4c, the full diagonal LP of §4i/§4j)
3. **Mapping.** Now compute the derivative accurately. — done first, and done well

You cannot search over objectives to fix a channel problem, nor over mappings to fix an objective
problem. **Every uphill null was step-2 or step-3 work on a step-1 failure.** Both step-1 and
step-2 checks are free and run on rollouts that already exist (rule 12); neither needs an
adaptation run.

**Where this points.** Not "adaptation cannot help" — the claim is narrower and it is falsifiable:
*a bounded position-target residual minimising tracking error cannot help on this robot.* The
channel that survives the argument is **impedance**: `∂τ/∂σ = K_p ẽ` grows with the tracking error
instead of being constant (`MODEL_AND_DERIVATIONS.md` §7.1), and unlike `gait_period_s` no single
static value serves both grades and payloads — so there is genuinely something to adapt. Screen it
with rule 12 before building it.
