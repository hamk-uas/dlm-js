# jax-js-nonconsuming: `einsum` (3+ operands) and `tensordot` crash under `grad`

**Package**: `@jax-js-nonconsuming/jax` v0.2.2  
**Discovered**: 2026-02-17  
**Severity**: Medium — workaround exists (decompose to matmul chains)  
**Updated**: 2026-02-17 — root cause partially re-attributed to `np.eye` (see
[jax-js-np-eye-ad.md](jax-js-np-eye-ad.md)); tensordot and einsum 2-op
failures were caused by `np.eye` contamination, not the ops themselves.

## Summary

Two ops fail when used inside `grad()`:

1. **`np.einsum` with 3+ operands** — crashes with `Inconsistent size for index N in einsum: [object Object] vs <number>`. The JVP tracer produces abstract shape objects, but `computeSizeMap` compares them by value against concrete integers. **This is a genuine einsum bug** — it reproduces even without `np.eye`.

2. **~~`np.tensordot`~~** — the original reproducer used `np.eye` as an operand. After replacing `np.eye` with `np.diag(np.ones([n]))`, tensordot may work. **Needs re-testing** now that the `np.eye` root cause is known (see [jax-js-np-eye-ad.md](jax-js-np-eye-ad.md)).

Both ops work fine in eager and JIT contexts — the bug is specifically in the AD (JVP) tracing path.

## Reproduction

```typescript
import { numpy as np, grad, DType } from "@jax-js-nonconsuming/jax";

// ✗ einsum 3-op under grad
const f1 = (W) => {
  using A = np.array([[1, 2], [3, 4]], { dtype: DType.Float64 });
  return np.sum(np.einsum("ij,jk,lk->il", A, W, A));
};
using W = np.eye(2, { dtype: DType.Float64 });
using g1 = await grad(f1)(W); // Error: Inconsistent size for index 2 in einsum

// ✗ tensordot under grad
const f2 = (A) => {
  using B = np.eye(2, { dtype: DType.Float64 });
  return np.sum(np.tensordot(A, B, [[1], [0]]));
};
using A = np.array([[1, 2], [3, 4]], { dtype: DType.Float64 });
using g2 = await grad(f2)(A); // Error: Invalid reshape
```

## Root cause

During JVP tracing, array `.shape` elements are tracer objects (`{ dtype: "float64" }`) rather than concrete integers. The einsum path callback `computeSizeMap` and the tensordot/dot path `prod(shape)` both assume concrete numbers.

### einsum 3-op

In `computeSizeMap` (dist/index.js:7940):
```js
else if (existing !== dim && dim !== 1) throw new Error(...)
```
`existing` is a tracer object, `dim` is 2 — the `!==` comparison always fails because object ≠ number.

### tensordot

In `dot` → `reshape` (dist/index.js:976):
```js
if (prod(originalShape) !== prod(shape$1)) throw new Error(...)
```
`prod([tracerObj, 2])` produces `NaN` or `[object Object]` instead of the expected integer.

## What works under grad (verified)

| Op | Status | Notes |
|---|---|---|
| `np.dot` (1D) | ✓ | |
| `np.matmul` (any 2D shapes) | ✓ | **All shape combos work** when np.eye is avoided |
| `np.einsum` (2-op) | ✓ | Previously thought broken — was np.eye contamination |
| `np.einsum` (3-op) | ✗ | Genuine bug in `computeSizeMap` |
| `np.tensordot` | ? | Needs re-test without np.eye |
| `np.transpose`, `np.flip`, `np.squeeze` | ✓ | |
| `lax.scan` (including matrix carries) | ✓ | |
| `np.divide`, `np.log`, `np.sqrt`, `np.abs` | ✓ | |
| `np.zeros`, `np.ones`, `np.full`, `np.diag` | ✓ | All produce concrete shapes |
| **`np.eye`** | **✗** | **See [jax-js-np-eye-ad.md](jax-js-np-eye-ad.md)** |

## Workaround

Decompose 3-operand einsum `A @ W @ A'` into chained `np.matmul`:
```typescript
// Instead of: np.einsum('ij,jk,lk->il', G, C, F)
// Use:        np.matmul(np.matmul(G, C), np.transpose(F))
```

This is what we do in the `dlmMLE` AD-compatible Kalman filter.

## Impact on dlm-js

The existing `dlmSmo` Kalman filter uses 3-operand einsum extensively. For MLE via autodiff, we wrote a separate AD-compatible filter core that uses only `np.matmul` + `np.transpose`. The existing `dlmFit` / `dlmSmo` remain unchanged.

## Additional note: optax not found

The `optax` module (Adam, SGD, etc.) is not exported in v0.2.2. We implemented a simple gradient descent loop as a workaround. ~~If optax is available in a newer version or the upstream fork, that would be the preferred optimizer.~~ **Resolved**: as of v0.4.0, optax is available via `@hamk-uas/jax-js-nonconsuming/optax` and is fully jittable. dlm-js now uses optax Adam.
