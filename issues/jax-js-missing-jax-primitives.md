# Feature request: missing JAX primitives — `np.broadcast_to` and scalar array indexing

**Status**: 🟢 Enhancement request  
**Package**: `@hamk-uas/jax-js-nonconsuming`  
**Filed**: 2026-02-24 (revised; original `np.where`/matmul items resolved)  
**Context**: Kalman filter / state-space model library ([dlm-js](https://github.com/hamk-uas/dlm-js)) using jax-js for autodiff + GPU

## Resolved items (for reference)

- ✅ `np.where(condition, x, y)` with JVP/VJP — fixed in `297f93a`
- ✅ `np.matmul` batch broadcasting — fixed in `c99db9a`

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

## 2. Scalar array indexing with AD — `lax.dynamic_index_in_dim` or `arr[i]`

**JAX equivalents**: [`lax.dynamic_index_in_dim`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dynamic_index_in_dim.html), or simply `arr[i]` returning a scalar `[]`-shaped array under trace.

### Problem

Extracting a single element from a parameter vector inside `grad()` requires building a one-hot mask and using `np.dot`:

```ts
// Current: 3 lines, allocates a full-length mask array per extraction
const maskData = new Array(nTheta).fill(0);
maskData[nSwParams + i] = 1;
using mask = np.array(maskData, { dtype });
using phi_i = np.dot(theta, mask);  // scalar ← [nTheta] · [nTheta]

// With scalar indexing: 1 line, no allocation
using phi_i = theta.at(nSwParams + i);  // or lax.dynamic_index_in_dim(theta, idx, 0)
```

### Impact

**`buildG`** (AR coefficient injection, `src/mle.ts:118–152`): The entire function is a loop that:
1. Extracts `phi_i = theta[nSwParams + i]` via one-hot mask + `np.dot` (3 lines)
2. Builds rank-1 outer product `e_i · e_j'` (4 lines)
3. Accumulates `G += phi_i * outer` (3 lines + disposal)

With scalar indexing, the extraction collapses from 3 lines to 1 line per iteration, and the mask array allocation disappears.

**`buildDiagW`** (diagonal process-noise matrix, `src/mle.ts:88–108`): Currently uses a `[m, nTheta]` mask matrix + matmul to extract `m` elements simultaneously. With scalar indexing, the mask matrix is unnecessary — just index `expTheta[wOffset + i]` in a loop and call `np.diag`.

**`makeKalmanLoss` / `makeKalmanLossAssoc`** (observation std extraction, `src/mle.ts:246, 388`): `np.dot(expTheta, mask_s)` extracts `s = expTheta[0]`. With indexing: `expTheta.at(0)`.

| Function | Current pattern | Lines saved | Sites |
|----------|----------------|-------------|-------|
| `buildG` | mask + `np.dot` loop | ~20 lines → ~8 lines | 1 per AR coeff (up to 5) |
| `buildDiagW` | mask matrix + matmul | ~10 lines → ~4 lines | 1 |
| `makeKalmanLoss` | `np.dot(expTheta, mask_s)` | 3 lines → 1 line | 1 |
| `makeKalmanLossAssoc` | `np.dot(expTheta, mask_s)` | 3 lines → 1 line | 1 |

### AD requirements

VJP of `arr[i] → scalar` is: grad flows into a zeros array with `grad` placed at index `i`. This is `lax.dynamic_index_in_dim`'s VJP in JAX — standard, well-defined.

### Note on `np.split`

`np.split` already works under trace with VJP (confirmed v0.7.8), covering 40+ partitioning sites. But `np.split(vec, [i, i+1], 0)[1]` returns shape `[1]` not `[]`, requiring an extra `np.squeeze` — less ergonomic than scalar indexing.

## References

- NumPy `broadcast_to`: [numpy.broadcast_to](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html)
- JAX `lax.dynamic_index_in_dim`: [jax.lax.dynamic_index_in_dim](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dynamic_index_in_dim.html)
- JAX `lax.slice`: [jax.lax.slice](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.slice.html)
