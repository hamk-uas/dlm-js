# Feature request: `np.where` with AD support

**Status**: 🟡 Partially resolved — matmul broadcasting fixed in commit `c99db9a` (installed 2026-02-24 via `09ddadb`)

**Package**: `@hamk-uas/jax-js-nonconsuming` v0.7.8  
**Filed**: 2026-02-24  
**Severity**: Enhancement — significant ergonomic and performance improvements  
**Context**: Kalman filter / state-space model library ([dlm-js](https://github.com/hamk-uas/dlm-js)) using jax-js for autodiff + GPU

## Summary

One remaining JAX feature that would significantly improve code quality:

1. **`np.where(condition, x, y)` with JVP/VJP** — replace float-mask multiply pattern
2. ~~**Broadcasting in `matmul` / `einsum`** — eliminate manual `np.tile`~~ ✅ Resolved: `np.matmul` now supports NumPy-style batch broadcasting (commit `c99db9a`). Workaround `np.tile` calls reduced from 27 to 20 in `src/`; remaining tiles are for non-matmul patterns (mask tiling, NaN handling, element construction).

### Note on array slicing

`np.split` already works under trace with full VJP support (confirmed in v0.7.8), covering all 40 array-partitioning sites in our codebase. The remaining 7 scalar-extraction sites use `np.dot(vec, one_hot_mask)` because `np.split(vec, [i, i+1], 0)[1]` gives shape `[1]` not `[]` — requiring an extra reshape. A `lax.slice` or NumPy-style `arr[i]` returning a scalar would be a nice ergonomic improvement but isn't blocking.

The bigger architectural friction (flat parameter vectors instead of pytrees) comes from the scalar extraction pattern in `buildDiagW` / `buildG`, where extracting `theta[i]` inside `grad()` requires building a one-hot mask per element. With a scalar-returning slice primitive, these 50+ lines of loop code collapse to `theta[i]`. But this is a convenience issue — the code works correctly today.

## 1. `np.where(condition, x, y)` with AD support

### Current workaround

To conditionally blend two traced arrays (e.g., for NaN masking in Kalman filters), we use float-mask multiplication:

```typescript
// mask is [n, 1, 1]: 1.0 for observed, 0.0 for NaN
using inv_mask = np.subtract(np.ones_like(mask), mask);
using result = np.add(
  np.multiply(mask, observed_value),
  np.multiply(inv_mask, default_value),
);
// 5 ops (sub, ones, mul, mul, add), 4 intermediate arrays
```

### What JAX provides

```python
result = jnp.where(mask, observed_value, default_value)
# 1 op, 0 intermediate arrays
```

JAX's `jnp.where` has full JVP/VJP: the gradient flows through the selected branch only. This is both cleaner and generates fewer intermediate tensors.

### Impact

- **dlm-js**: The NaN-masking pattern appears throughout the assocScan forward filter (`makeKalmanLossAssoc`), the smoother (`dlmSmo`), and the forecast function — about 10+ sites with ~20 individual float-blend operations
- **General**: Any model with missing data, masking, or conditional computation benefits. This is fundamental to time series, NLP (attention masks), and scientific computing.

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

1. **`np.where` with AD** — the only remaining item; highest impact, universally useful, eliminates the most boilerplate and intermediate tensors

## References

- NumPy broadcasting rules: [numpy.broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- JAX `jnp.where`: [jax.numpy.where](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.where.html)
- JAX `lax.slice` (static indexing, for context): [jax.lax.slice](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.slice.html)
