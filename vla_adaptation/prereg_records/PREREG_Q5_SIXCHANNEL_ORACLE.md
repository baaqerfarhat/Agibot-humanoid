# Preregistration: six-channel oracle on libero_10 (queue item Q5)

**Written 2026-09-15, before any run. GPU runs are on hold until lambda is free.**

**Question.** The rotation-only oracle (`D3a_oracle_rot_libero_10`: the true fault cancelled exactly on
r_x, r_y, r_z) reaches 22/40 against a healthy rate of about 36–38/40. Is the remaining gap the
rotation-only correction mask, or something no additive correction can fix?

**Run.** `D3b_oracle_all_libero_10`: identical to `D3a_oracle_rot_libero_10` (unpinned, 40 episodes,
`--eval-init 45`, `--scenario-reset`, legacy law and constants, `--static-corr -0.05` on all six
channels) except `--corr-dims 0,1,2,3,4,5`. Same scenario keys as D3a; its frozen arm is re-run and
reported. 80 rollouts.

**Predictions.**
1. If the mask caps libero_10, the six-channel oracle reaches ≥ 30/40. **Refutation: ≤ 26/40**, which
   would place the gap outside additive command correction (policy behaviour under a fault it can see,
   contact, or a non-additive effect of the fault).
2. The frozen arm stays ≤ 3/40 (D3a's frozen arm: 1/40).
Between 27 and 29 is reported as inconclusive at this n. The comparison with D3a is between runs on
the same keys with unpinned sampling (the measured noise floor is up to 5 episodes per 20).

**Why it matters.** It decides whether the paper's libero_10 shortfall is a channel-selection problem,
which tracking cannot reach (translation is unobservable in the pose channel, DUAL_TRACK_C2.md), or
something else.

---

## Outcome (2026-09-15, 17:16; run by Mahdi's session on GPU 1; `results/collab_q/q5_D3b_oracle_all_libero_10/`)

D3a's protocol with all six channels corrected (`--corr-dims 0,1,2,3,4,5`, `--static-corr −0.05 ×6`,
unpinned, `--eval-init 45`, 40 episodes, legacy law and constants, shipped calibration):
frozen **0/40**, six-channel oracle **38/40**.

- **Prediction 1: holds, decisively.** 38/40 against the registered ≥ 30/40; the rotation-only
  oracle on the same 40 keys (D3a, 22/40) loses to it 17 to 1. With the whole fault cancelled
  the suite returns to its healthy rate (healthy controls 36–38/40). **The libero_10 shortfall
  is the rotation-only correction mask**, not something outside additive command correction.
- **Prediction 2: holds** (frozen 0/40 ≤ 3/40).

Consequences, stated for the paper: every adaptive libero_10 result in the record (15–25/40 or
/60) sits under a 22/40 rotation-only ceiling that the six-channel oracle lifts to 38/40; the
gap the estimator cannot close on this suite is translation, which the rotation mask never
touches and which tracking cannot reach either. The masked comparison remains the registered
one; a translation-including correction on libero_10 is a new registration.
