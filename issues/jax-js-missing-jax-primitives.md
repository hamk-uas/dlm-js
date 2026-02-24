# Feature request: `np.where` with AD support

**Status**: ✅ Resolved — `np.where` AD support fixed in `297f93a`; matmul broadcasting in `c99db9a`; einsum fast path + analytical inv + auto checkpoint in `70dea65`

**Package**: `@hamk-uas/jax-js-nonconsuming` v0.7.9  
**Filed**: 2026-02-24  
**Severity**: Enhancement — significant ergonomic and performance improvements  
**Context**: Kalman filter / state-space model library ([dlm-js](https://github.com/hamk-uas/dlm-js)) using jax-js for autodiff + GPU

## Summary

All requested features are now resolved:

1. ~~**`np.where(condition, x, y)` with JVP/VJP** — replace float-mask multiply pattern~~ ✅ Resolved: `np.where` now has full JVP/VJP support (commit `297f93a`). Replaced all high-value blend patterns (5 ops → 1 op) in `src/mle.ts` (makeKalmanLossAssoc) and `src/index.ts` (assoc backward terminal blend, forward NaN handling). Remaining `multiply(mask, x)` sites are 1-for-1 substitutions with no op-count savings.
2. ~~**Broadcasting in `matmul` / `einsum`** — eliminate manual `np.tile`~~ ✅ Resolved: `np.matmul` now supports NumPy-style batch broadcasting (commit `c99db9a`). Workaround `np.tile` calls reduced from 27 to 20 in `src/`; remaining tiles are for non-matmul patterns (mask tiling, NaN handling, element construction).

### Note on array slicing

`np.split` already works under trace with full VJP support (confirmed in v0.7.8), covering all 40 array-partitioning sites in our codebase. The remaining 7 scalar-extraction sites use `np.dot(vec, one_hot_mask)` because `np.split(vec, [i, i+1], 0)[1]` gives shape `[1]` not `[]` — requiring an extra reshape. A `lax.slice` or NumPy-style `arr[i]` returning a scalar would be a nice ergonomic improvement but isn't blocking.

The bigger architectural friction (flat parameter vectors instead of pytrees) comes from the scalar extraction pattern in `buildDiagW` / `buildG`, where extracting `theta[i]` inside `grad()` requires building a one-hot mask per element. With a scalar-returning slice primitive, these 50+ lines of loop code collapse to `theta[i]`. But this is a convenience issue — the code works correctly today.

## 1. ~~`np.where(condition, x, y)` with AD support~~ ✅ RESOLVED

**Fixed in**: commit `297f93a` — `np.where(condition, x, y)` now has full JVP/VJP support. Condition must be boolean dtype (float masks throw "Wrong dtype in where: expected bool, got float32").

**Workaround cleanup performed**: Replaced blend patterns (5 ops → 1 op) in `src/mle.ts` (makeKalmanLossAssoc: A_all, C_all, K_obs, eta_obs, J_obs, b_all, eta_all, J_all) and `src/index.ts` (sqrt-assoc backward terminal: E_all, g_all, D_all; standard assoc backward terminal: E_all; forward NaN handling in both assoc variants: b_all, eta_all). All 200 tests pass.

### Remaining `multiply(mask, x)` sites (not converted)

These are 1-for-1 substitutions (`multiply(float_mask, x)` → `np.where(bool_mask, x, zeros)`) with no op-count savings:
- **Diagnostic reductions** (5 sites): `np.multiply(mask_flat, expr)` before `np.sum()` — standard zeroing-before-reduction idiom
- **Per-element v_arr/K_arr masking** (4 sites): `multiply(mask_arr, v_raw/K_raw)` in assoc forward paths
- **First-element K0/K1 masking** (2 sites): `multiply(mask1, K_obs)` — single-element masks

## 2. ~~Broadcasting in `matmul` / `einsum`~~ ✅ RESOLVED

**Fixed in**: commit `c99db9a` — `np.matmul` now supports NumPy-style batch dimension broadcasting with `broadcastShapes` + `broadcastTo`. 1D vector support added.

**Workaround cleanup performed**: Replaced 7 `np.tile` + `einsum` sites in `src/mle.ts` and `src/index.ts` with `np.matmul` (batch broadcast) and broadcasting `einsum` patterns. All 200 tests pass.

### Remaining `np.tile` sites (20)

The remaining `np.tile` calls are for non-matmul patterns that don't benefit from matmul broadcasting:
- **Mask tiling** (5 sites): `np.tile(is_nan, [1, m, m])` for `np.where` NaN handling
- **Identity expansion** (4 sites): `np.tile(I_eye, [n, 1, 1])` for element-wise `subtract(I_exp, KF)`
- **Element construction** (7 sites): G_exp, W_exp, F_exp, Ft_exp in `makeKalmanLossAssoc` — tiled for multiple subsequent `einsum` calls (changing all would require rewriting the entire forward-element construction; einsum 2-op broadcasting already works but 6+ consumers each would need individual changes)
- **Broadcast scalars/additions** (4 sites): V2 scalar, FF_scan, G_scan/W_scan for time-varying state-space matrices

## Priority

All items resolved. This issue file is kept for reference — remaining `np.tile` and `multiply(mask, x)` sites are documented above but are not worth converting (marginal benefit).

## References

- NumPy broadcasting rules: [numpy.broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- JAX `jnp.where`: [jax.numpy.where](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.where.html)
- JAX `lax.slice` (static indexing, for context): [jax.lax.slice](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.slice.html)
