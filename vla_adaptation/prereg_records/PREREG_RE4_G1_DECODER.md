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

---

## Outcome (appended 2026-09-12; nothing above was edited)

30 observations, 1,656 s of server time, every bias edit acknowledged at the requested norm.

1. Validity: repeated queries with the pinned sampler agree to **0.0** (bit-identical). Confirmed.
2. Diagonal dominance: max off-diagonal/diagonal per arm row **0.04–0.15** ≤ 0.2. Confirmed.
3. Near-linear: ‖D − J‖_F/‖J‖_F = **0.035** ≤ 0.1; quadratic part at ±0.1 is **2–8 %** ≤ 10 %. Confirmed.
4. ζ median ≤ 0.1: **refuted.** ζ median **0.43**, p95 1.92, max 4.50.
5. State dependence CV ≤ 20 %: **refuted.** CV of J's diagonal across observations **0.27–0.41**.

Reading. At a fixed observation the decoder's response to a bias edit is linear and nearly
diagonal, so an edit computed from that observation's own Jacobian would land. But the Jacobian's
gain varies 30–40 % from observation to observation (J diag −0.26, −0.22, −0.18, −0.03, −0.05,
−0.05 on the arm dims, in unnormalised action per normalised bias unit), so a single bias vector
intended to realise a correction c misses it by a median 43 % and up to 4.5× across the
observations a rollout visits. A native bias edit is not a reliable way to apply a computed
correction; the external subtraction the runners use is exact by construction. For re4's decoder
bound: ζ = 0.43 median, J diag CV 0.27–0.41, quadratic term ≤ 8 % at ±0.1. Chunk-mean response is
1.0–1.6× the first-action response, so later actions in a chunk respond more than the first.
