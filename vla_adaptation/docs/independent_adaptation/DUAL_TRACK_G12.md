# Dual-track convergence — the geometric addition, integrated

2026-09-08. Both tracks integrated the three geometric pieces into the adaptation loop
independently, into separate files. **98 tests pass** across the package.

## Convergence on the substance

| claim | Claude `geometric_loop.py` | Codex `geometric_loop_codex.py` |
|---|---|---|
| basis equivariance enforced by Reynolds projection | verified `< 1e-10` | **8.9e-16** |
| whole update commutes with the group action | via composed loop | **7.1e-15** |
| non-invariant `L, R, Q, Lambda` raise at construction | yes | yes |
| wrench features refused in a first-order row without a compliance map | yes | yes |
| correct map cancels exactly | yes | yes |
| **wrong map is worse than doing nothing** | ratio `= |1−s|` to 8 places | `−3×` error, `9×` squared, to `3e-14` |

**The counterexample converged numerically by two independent routes.** Codex's test sets the true
gain to 1 and the estimate to 1/4 and asserts the amplification is exactly 3. My `authority_margin`
predicts `|1 − E_true/E_hat| = |1 − 4| = 3`. Neither track saw the other's derivation.

## Where the Codex implementation is better

- **It has the effectiveness branch.** `z_hat` is ordered `[z_E; z_d]` and allocation uses
  `E_hat = E0 + sum_j z_E[j] E_j`. Mine is disturbance-only. Given G4 and G5 established `E` is the
  binding component, mine again omits the part that binds.
- **It projects the wrench construction over the orbit**, taking `jacobians(xi)` and
  `wrench_seed(xi)` as callbacks evaluated at every transformed context, then Reynolds-projecting in
  joint-force rows *before* row placement. Mine projects a generic seed and places fixed matrices,
  which cannot express the group action on the contact geometry itself.
- **It tests against a nonabelian group.** My equivariance tests use `C_2`, which is abelian, so
  they cannot catch errors that appear only for noncommuting elements.
- **It enforces the structural constraints per row mode** — `mechanical` requires `n = 2d`,
  `A[:d] = [0, I]`, `E0[:d] = E_j[:d] = 0` — rather than only checking a `descriptor_order` flag.
- It states the design restriction explicitly: `T = S` restricted to `rho` or `diag(rho, rho)`,
  orthogonal `U, W`, block-preserving `W = diag(W_E, W_d)`, with nonorthogonal gauge transport
  deferred to the more general family-of-designs API.

## One point both tracks make, and it is the right one

Codex: *"z_hat ... may be asymmetric across limbs and are NEVER projected onto an invariant
subspace."* The symmetry constrains the basis and the metrics; the coefficient stays free. That is
the equivariance-not-invariance distinction, enforced in code rather than stated in prose.

## A defect in my own prompt, not in the work

`CODEX_G12.md` was never written because my prompt contained a contradiction: it said "write ONLY to
`geometric_loop_codex.py` and `test_geometric_loop_codex.py`" and also "report to
`codex_prompt/CODEX_G12.md`". The track flagged the conflict and declined to violate the file
restriction rather than silently picking one. That is the correct behaviour and the fault is mine.

## Status

Both implementations retained: Codex's as the more complete reference, mine as the independent
cross-check that confirmed the counterexample by a different derivation. The geometric addition is
now part of the method rather than a set of utilities beside it.

Still true: nothing is trained, and nothing has run on a robot.
