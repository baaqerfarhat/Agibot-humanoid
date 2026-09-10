# Dual-track convergence — the descriptor-form implementation

2026-09-08. Both tracks implemented FORMULATION.md Eq. (14) independently into separate files.
**57 tests pass.** The two implementations agree to machine precision, and the comparison found a
real defect in mine.

## 1. The convergence that matters

Two separately written implementations, cross-checked on identical inputs over 128 randomized cases:

| quantity | max disagreement |
|---|---|
| `F_z = expm(-Lambda h)` | **2.2e-19** |
| `Q_d` (Van Loan) | **5.6e-17** |
| `z_hat` after one composite step | **8.9e-16** |
| `P` after one composite step | **8.9e-16** |

Machine precision. Neither track saw the other's code. That is the strongest evidence available
that Eq. (14) has been implemented as written rather than as either track imagined it.

## 2. The defect the comparison found, in my implementation

Eq. (14) writes the observation with `Psi_k` — the basis **integrated over the window** — and the
tracking injection with `H_{k+1}` — the basis at the window's **endpoint**:

```
z^pred  = z^- + K (Y_k - Psi_k z^-)
z_{k+1} = z^pred + h P^+ H_{k+1}^T L e_{k+1}
```

The Codex track exposed this in its API by taking `integrated_basis` and `tracking_basis` as
**separate arguments**. My implementation used the same `Phi` for both.

For a **constant** basis the two coincide, which is why all ten of my original tests passed: none
of them varied the basis within a window. For a state-dependent `Phi(xi)` — which is the entire
point of the successor direction — they differ, and conflating them is simply wrong.

Fixed, and the fix is cross-verified: `observe` now takes an explicit `endpoint_basis`, my updated
result matches the other implementation to `1e-10`, and a new test confirms the two bases give
genuinely different updates in more than 60 of 64 randomized cases.

**This is the clearest example in the session of why the second track earns its cost.** My own tests
could not detect the defect, because the assumption that produced it also shaped the tests.

## 3. Where the Codex implementation is more complete

- **The effectiveness branch.** Its `_regressor` combines effectiveness and additive disturbance
  features into one regressor, so it adapts `E(xi, z_E)`. Mine implements the disturbance branch
  only. Given G4 and G5 established that `E` is the **binding** component, I implemented the part
  that was shown not to bind and skipped the part that does.
- **Exact constant-forcing descriptor propagation** where I used explicit Euler for the reference
  and labelled it as not accurate.
- Rectangular input maps, and validated-failure transactionality tested explicitly rather than
  merely holding by construction.

## 4. What my track adds

`authority_margin(E_true, E_hat) = ||I - E_true pinv(E_hat)||_2`. With a perfect disturbance
estimate and no nominal command the plant sees `(I - E_true pinv(E_hat)) Phi z` where OFF would see
`Phi z`, so the margin is exactly the worst-case amplification factor:

- `< 1` the correction strictly helps for **every** disturbance;
- `> 1` some disturbance is amplified by that factor;
- for a scalar map at `s` times the believed value the margin is `|1 - s|`, so a correction becomes
  harmful once the true map exceeds **twice** the believed one.

Applied to the predecessor's own two input models — the separately identified `M` and the FIR's
implicit steady-state gain, which *should be the same object* — the margin is **1.896**, and 1.177
the other way. Both above one.

This is the computable form of the `||E - E_cal|| <= delta` bound the identifiability track said the
method would need, and it is what makes that restriction checkable rather than aspirational.

## 5. Status

Both implementations are retained. The Codex loop is the more complete reference; mine is kept as
the independent cross-check that made the endpoint defect visible, and the cross-check is now a
test rather than a one-off script. Neither is certified for a robot: Eq. (14) is a candidate
splitting, no basis is trained, and no robot adapter exists.
