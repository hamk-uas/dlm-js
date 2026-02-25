# Feature request: missing JAX primitives — `np.broadcast_to`

**Status**: 🟢 Enhancement request  
**Package**: `@hamk-uas/jax-js-nonconsuming`  
**Filed**: 2026-02-24 (revised; original `np.where`/matmul items resolved)  
**Context**: Kalman filter / state-space model library ([dlm-js](https://github.com/hamk-uas/dlm-js)) using jax-js for autodiff + GPU

## Resolved items (for reference)

- ✅ `np.where(condition, x, y)` with JVP/VJP — fixed in `297f93a`
- ✅ `np.matmul` batch broadcasting — fixed in `c99db9a`
- ✅ Scalar array indexing (`lax.dynamicIndexInDim`, `lax.sliceInDim`) — fixed in `6e6d4fe`

---

## 1. `np.broadcast_to(arr, shape)` — eliminate `np.tile` boilerplate

**JAX equivalent**: [`jax.numpy.broadcast_to`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.broadcast_to.html)

### Problem

We have **20 `np.tile` call sites** across `src/index.ts` (14) and `src/mle.ts` (6) that expand a `[1, m, m]` or `[1, 1, m]` array to `[n, m, m]` or `[n, 1, m]`. Every one follows the same pattern:

```ts
// Current: reshape + tile (2 ops, allocates n copies)
using G_exp = np.tile(np.reshape(G, [1, m, m]), [n, 1, 1]);

// With broadcast_to: 1 op, zero-copy view (no physical allocation)
using G_exp = np.broadcast_to(np.reshape(G, [1, m, m]), [n, m, m]);
```

### Impact

| Category | Sites | Example |
|----------|-------|---------|
| Constant matrix expansion (G, W, I, F, Ft) | 11 | `np.tile(np.reshape(G, [1,m,m]), [n,1,1])` |
| Boolean mask tiling for `np.where` | 5 | `np.tile(is_nan, [1, m, m])` |
| Scalar broadcast (V2) | 1 | `np.tile(np.reshape(V2, [1,1,1]), [n,1,1])` |
| Time-varying matrix construction | 3 | `G_scan = np.tile(...)` |

`broadcast_to` is a view (no memory allocation in NumPy/JAX), so this would also reduce peak memory in `makeKalmanLossAssoc` where G, W, F, Ft, I are all tiled to `[n, m, m]` simultaneously (5 × n × m² floats → 5 × m² floats).

### AD requirements

`np.broadcast_to` needs JVP/VJP rules. JAX implements VJP as a reduce-sum over the broadcast axes — straightforward.

### Affected files

- `src/index.ts`: 14 sites (lines 161, 536, 622, 923, 986, 1009, 1042, 1044, 1047, 1295, 1612, 1613 + 2 more)
- `src/mle.ts`: 6 sites (lines 403, 409, 411, 412, 413, 415, 416)

## References

- NumPy `broadcast_to`: [numpy.broadcast_to](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html)
