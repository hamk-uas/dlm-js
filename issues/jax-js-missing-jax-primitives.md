# Feature request: broadcasting and `where` with AD support

**Package**: `@hamk-uas/jax-js-nonconsuming` v0.7.8  
**Filed**: 2026-02-24  
**Severity**: Enhancement — significant ergonomic and performance improvements  
**Context**: Kalman filter / state-space model library ([dlm-js](https://github.com/hamk-uas/dlm-js)) using jax-js for autodiff + GPU

## Summary

Two JAX features that would significantly improve code quality and on-device memory efficiency for any project doing traced array computation (not just ours):

1. **`np.where(condition, x, y)` with JVP/VJP** — replace float-mask multiply pattern
2. **Broadcasting in `matmul` / `einsum`** — eliminate manual `np.tile`

Each of these would reduce the number of intermediate arrays, improve code readability, and keep data on-device better by avoiding materialisation of expanded/tiled tensors.

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

## 2. Broadcasting in `matmul` / `einsum`

### Current workaround

To multiply a constant `[m, m]` matrix by a batch of `[n, m, 1]` vectors, we must first tile the matrix:

```typescript
using G_exp = np.tile(np.reshape(G, [1, m, m]), [n, 1, 1]);  // [n, m, m]
using result = np.einsum('nij,njk->nik', G_exp, x_batch);     // [n, m, 1]
```

This creates an intermediate `[n, m, m]` tensor that is a pure waste of memory — every slice is identical.

### What JAX provides

JAX's `jnp.matmul` and `jnp.einsum` support NumPy-style broadcasting:

```python
# G is [m, m], x_batch is [n, m, 1]
result = jnp.matmul(G, x_batch)  # broadcasts G to [n, m, m] without materializing
# or
result = jnp.einsum('ij,njk->nik', G, x_batch)  # implicit broadcast
```

The `einsum` form `'ij,njk->nik'` (contraction of a 2D matrix with a 3D batch) is a standard NumPy broadcasting pattern. Currently jax-js-nonconsuming already supports this specific einsum pattern — but `np.matmul` doesn't broadcast, and explicit `np.tile` is needed for some `einsum` patterns with more than 2 operands.

### Impact

- **dlm-js**: 17 `np.tile` calls across `src/` broadcast a constant `[m, m]` or `[1, m]` matrix to `[n, ...]`
- **General**: Any batched linear algebra (Kalman filters, RNNs, attention layers, batched MLPs) would benefit. Broadcasting avoids O(n·m²) memory for what is logically a O(m²) constant.

## Priority suggestion

1. **`np.where` with AD** — highest impact, universally useful, eliminates the most boilerplate and intermediate tensors
2. **`matmul` broadcasting** — nice-to-have; the `einsum` broadcast path already works for 2-operand cases

## References

- NumPy broadcasting rules: [numpy.broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- JAX `jnp.where`: [jax.numpy.where](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.where.html)
- JAX `lax.slice` (static indexing, for context): [jax.lax.slice](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.slice.html)
