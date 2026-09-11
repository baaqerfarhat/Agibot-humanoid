# Preregistration: realisation of a native decoder-bias edit (re4 Part G.1, 2026-09-11)

Written before the measurement. The ACE experiments edit π0.5's `action_out_proj` bias in place
rather than subtracting a correction from the executed action; re4 states that "the final-action
displacement induced by the decoder edit was not measured". This measures it.

**Procedure** (`openpi/g1_decoder.py`, run only when no rollout is running, because the edit lands
on the shared server). Thirty fixed observations: the first policy step of the 20 headline
`libero_spatial` scenarios (tasks 0–9 × inits 45, 46) and 10 mid-episode states (tasks 0–9, init
45, 40 healthy steps in). Flow-sampler noise pinned (`pin_rng`). Edits to the 7 LIBERO bias dims
at ±0.02, ±0.05, ±0.1 (normalised model units), one dim at a time; response = the first executed
LIBERO action after unnormalisation.

**Quantities.** J: central-difference Jacobian at ±0.02, averaged over observations. D:
least-squares linear map over all magnitudes. Q: even part, fitted as Q m². ζ: for intended
corrections c = ±0.05 e_i on the six arm dims, apply b = J⁻¹c and measure ‖Δa − c‖ / ‖c‖.
The re4 appendix symbols are not on this machine; these operational definitions are what the
writing side maps onto its equation.

**Predictions.**
1. Validity: repeated queries at a fixed observation with the pinned sampler agree to 1e-5.
   Otherwise the measurement is invalid and reported as such.
2. J is diagonal-dominant: every off-diagonal ≤ 20 % of its row's diagonal on the arm dims.
3. Near-linear: ‖D − J‖_F / ‖J‖_F ≤ 0.1, and the quadratic part at ±0.1 is ≤ 10 % of the linear part.
4. ζ median ≤ 0.1 across observations and axes.
5. State dependence: the across-observation coefficient of variation of J's diagonal ≤ 20 %.

**Refutation.** Any of 2–5 failing means a native bias edit does not realise the intended final
action to that precision; the paper's decoder bound then carries the measured ζ and J spread.
