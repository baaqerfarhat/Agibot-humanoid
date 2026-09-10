# The input map is identifiable on translation and not on rotation — measured, from healthy data

2026-09-08, Claude track. The second track is answering the same question independently.

## 1. What was asked

G5 concluded the method is not viable as specified, and named a fixed calibrated `E` with a bounded
envelope as the first defensible restriction — while noting the existing averaged `M` files do not
establish such bounds. `authority_margin(E_true, E_hat) = ||I - E_true pinv(E_hat)||_2` makes the
bound computable: it is exactly the worst-case amplification, so **below 1 the correction helps for
every disturbance, above 1 some disturbance is amplified by that factor.**

The decisive test needs no fault and no ground truth: **do two honest identifications of the SAME
healthy system agree to within a margin of 1?** If they do not, no fixed calibration can be safe,
because the map is not identifiable to the precision the method requires.

## 2. The measurement

The stored LIBERO calibration contains **three healthy episodes** (`ep_len = [75, 116, 94]`), and
`fit_plant` accepts an episode subset, so seven independent fits can be formed. Implied input map
(diagonal steady-state gain) per fit of the same healthy robot:

| fit | x | y | z | rx | **ry** | rz |
|---|---|---|---|---|---|---|
| {0} | .237 | .263 | .232 | .110 | **.029** | .244 |
| {1} | .241 | .260 | .240 | .302 | **.155** | .253 |
| {2} | .251 | .245 | .247 | .051 | **.074** | .233 |
| all | .242 | .258 | .240 | .228 | .103 | .248 |

Translation is tight. Rotation is not: episode 1's `ry` gain is **5.3 times** episode 0's, on the
same healthy robot with no fault present.

Pairwise authority margins over all 42 ordered pairs of the seven fits:

| axis set | max margin | median | fraction above 1 | verdict |
|---|---|---|---|---|
| **translation (0,1,2)** | **0.071** | 0.034 | **0%** | a fixed map is **safe** |
| rotation (3,4,5) | **4.974** | 0.755 | 36% | **unsafe** |
| all six | 4.974 | 0.755 | 36% | **unsafe** |

## 3. What this explains

The margin is computed from **healthy calibration data alone** — no fault, no rollout, no outcome —
and it separates the configuration that works from the one that catastrophically fails:

| configuration | margin | observed at LIBERO Object joint 3 |
|---|---|---|
| legacy, `corr-dims 0,1,2`, translation only | **0.071** | **16/20**, double the 8/20 OFF baseline |
| weighted full, all six inputs | **4.974** | **2/20**, a quarter of OFF |

The counterexample family this project has been circling — six scenarios where the richer correction
destroys what both OFF and legacy achieved — is consistent with adding authority on axes whose input
map is unidentified to within a factor of five.

It also reconciles several things already in the record: the paper's own note that rotation is where
this policy breaks; the `r_y` fault being "only 23% identified"; and `fit_plant`'s held-out R^2 of
~0.97 on translation against 0.107-0.376 on rotation. All of these are the same fact.

## 4. Honesty about what this is

**Post-hoc.** I computed the margin knowing the outcomes. It is an explanation that fits, not a
prediction that was made.

**The candidate changed several things at once.** Weighted-full moved from three to six inputs *and*
altered residual filtering, regularisation, bound placement, and bypassed the deadzone and
normaliser. The margin explains the input-dimension change only; it does not isolate it.

**One cell.** A single Object joint-3 comparison, n = 20.

**The map used is the FIR's implicit diagonal gain**, which is itself one of the two mutually
inconsistent input models in the predecessor (`||I - M pinv(FIR_gain)|| = 1.896`). A margin computed
against the other model would differ.

## 5. Why it is nonetheless the strongest thing here

The quantity needs **no fault data, no rollout and no outcome**. It is therefore preregisterable:
compute the margin on a candidate correction subspace from healthy episodes, predict safe or unsafe
before running, then run. That is exactly the shape both tracks independently named as the strongest
available contribution — *a prespecified measurement that predicts an outcome on conditions it was
not fit to* — and unlike every earlier attempt in this line, the instrument is cheap, the prediction
is binary, and the data to compute it already exists for every cell.

**The concrete prospective test:** the margin says translation is safe and rotation is not. Take a
correction subspace that has never been run — translation plus a single rotation axis, or rotation
alone — compute its margin from healthy episodes, register the prediction, and run the cell.

## 6. Consequence for the method

The fixed-`E_cal` restriction G5 named is **viable, and only on the subspace where the map is
identifiable.** On this robot and calibration that subspace is translation. The successor should
therefore not be scoped as "a learned state-dependent correction at the action interface" but as
**"a correction restricted to the subspace where the input map is identifiable, with the margin as
the admission criterion"** — which is a smaller claim, measurable in advance, and the one the
evidence supports.
