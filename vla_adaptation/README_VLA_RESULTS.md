# VLA adaptation — what was established, August 2026

One page, for someone picking this up cold. Detail lives in `docs/LESSONS_ADAPTATION.md` §9;
every claim below traces to a pre-registration in `prereg_records/` and raw data in `results/`.

Subject: **π0.5-LIBERO**, 3.35 B parameters, frozen — no fine-tuning, no gradients through
the task. Benchmark: `libero_spatial`, nominal 99.0% over 500 episodes.

## 1. The headline: a frozen VLA is repairable online, and the ceiling is total

A fault that costs half the task — a constant +0.05 offset on the arm action, 99% → 46.7% —
is **fully repaired by adding a fixed 6-vector to `action_out_proj/bias`**: 15/15 = **100%**.

It is a targeted inverse, not a tonic: the same edit on the *healthy* policy gives 6.7%, and
sign-flipped it gives 0%. The basin is narrow — 1.5× the correct edit scores 20%, worse than
no repair at all. The edit was **computed, not searched**, from the checkpoint's quantile norm
stats plus a measured attenuation, and landed 15/15 in closed loop.

**The transferable part is the units.** π0.5 emits *normalised* actions and `Unnormalize` runs
after, so a uniform env-space fault is anisotropic in the model's own units: +0.05 is 3.0% of
the action range on translation and **19.5% on wrist rotation**. The repair is therefore a
per-dim vector spanning 7.3×, not a scalar. Parameterise edits in the units the task is
measured in, or a correct method reads as "this class is not reachable".

## 2. The selection stage: ACE works as a ranker, and is not a usable tool

The published ACC selection stage (ACE: perturb a layer randomly, see how much the task metric
moves) was pre-registered and **failed**: p = 0.974, η² = 0.036.

That null was a **measurement artifact**. Four faults, each measured: unpaired scoring (96% of
a 5-episode score is sampler noise); a perturbation scale below the threshold of behaviour
(0 of 6 outcome flips); matching by relative parameter norm (30× spread in actual effect); and
78% of each draw spent on the 25 action dims LIBERO discards as padding.

Fix those and a **pre-registered re-test passes decisively: F(8,63) = 35.8, p = 1.3e-20,
η² = 0.820**, with the tier ordering interface > action expert > VLM trunk.

**But it does not do the job the pipeline needs.** The site with the verified 100% repair
ranks **5th of 9**. And **54% of the between-site variance is explained by how hard each site
was actually perturbed** — matching cannot be fixed, because the within-site response varies
by CV 0.29–0.98 across observations, so no scalar per layer equalises it.

Two modifications help, both cheap: scoring the **max** draw rather than the mean (#5 → #2,
stable across every upper-tail statistic), and scoring a **scale-free ratio of measured
effects** rather than a difference of task metrics (#1; confound 54% → 26%).

## 3. What is not established

- **Ground truth is n = 1.** Only one site is known repairable, so "criterion X ranks it
  first" is one data point. Five attempts to measure per-site repairability all failed, for
  diagnosed reasons (§9k) — the last and most informative being that **a random probe basis
  needs ~4× the parameter norm of the analytic repair for the same effect**, which saturates
  the policy. Random directions detect that a layer matters; they cannot build the edit.
- **The fault is the easy geometry.** A constant action offset repaired by an output bias:
  fault class and edit class coincide. The gate never supplied a state-dependent one.
- **Nothing here shows a search recovers the ceiling.** The 100% is a computed edit. Whether
  CEM finds it unaided is Phase 0.5 and was not run.
- **ACE is not shown to predict searchability.** That claim was falsified five times earlier
  in this project and is untouched by any of the above.

## 4. Practical notes

Serving π0.5 for weight-level experiments: openpi's `module_jit` **freezes module state at
wrap time**, so mutating the model in place is a silent no-op through the serving path — pass
state as a jit argument instead (`openpi/ace_server.py`). Pin the sampler RNG and episodes
become exactly deterministic (`max|ΔAction| = 0.0`), which turns a noisy 5-episode score into
a paired measurement. And a LIBERO cell at n = 20 is ±11 points: the same faulted policy
scored 40.0 / 46.7 / 52.0 / 55.0% on four disjoint initial-state sets.
