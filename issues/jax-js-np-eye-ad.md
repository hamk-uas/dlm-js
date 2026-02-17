# `np.eye()` produces tracer-contaminated shapes under `grad`/`valueAndGrad`

**Package:** `@jax-js-nonconsuming/jax` v0.2.2

## Summary

`np.eye(n)` inside a function being differentiated via `grad()` or
`valueAndGrad()` produces an array whose `.shape` contains abstract tracer
objects instead of concrete numbers. Any subsequent operation that inspects
`.shape` (reshape, matmul, broadcast) then fails with misleading errors like:

- `"Invalid reshape: [tracer,2] -> [1,tracer,2]"`
- `"Invalid reshape: [2,tracer] -> [-1]"`

Other constructors (`np.zeros`, `np.ones`, `np.full`, `np.diag`, `np.array`)
all produce concrete shapes under the same tracing context and work correctly.

## Reproducer

```ts
import { numpy as np, grad, DType } from "@jax-js-nonconsuming/jax";

// FAILS — np.eye leaks tracer shapes
const f_eye = (x: np.Array) => {
  const I = np.eye(2, { dtype: DType.Float64 });
  const A = np.matmul(I, np.reshape(x, [2, 1]));
  return np.sum(A);
};
const g_eye = grad(f_eye);
const result = await g_eye(np.array([1, 2], { dtype: DType.Float64 }));
// Error: "Invalid reshape: [tracer,2] -> [1,tracer,2]"

// WORKS — np.diag(np.ones([n])) as workaround
const f_diag = (x: np.Array) => {
  const I = np.diag(np.ones([2], { dtype: DType.Float64 }));
  const A = np.matmul(I, np.reshape(x, [2, 1]));
  return np.sum(A);
};
const g_diag = grad(f_diag);
const result2 = await g_diag(np.array([1, 2], { dtype: DType.Float64 }));
// ✓ returns [1, 1]
```

## Systematic testing of constructors under grad

| Constructor                      | AD-safe? | Notes                                       |
|----------------------------------|----------|---------------------------------------------|
| `np.array([[1,0],[0,1]])`       | ✓        | Literal — always concrete                   |
| `np.zeros([m, m])`              | ✓        |                                             |
| `np.ones([m, m])`               | ✓        |                                             |
| `np.full([m, m], val)`          | ✓        |                                             |
| `np.diag(np.ones([m]))`         | ✓        | **Workaround for np.eye**                   |
| `np.reshape(arr, shape)`        | ✓        |                                             |
| `np.transpose(arr)`             | ✓        |                                             |
| **`np.eye(m)`**                 | **✗**    | `.shape` contains tracer objects            |
| `np.multiply(np.eye(m), x)`     | ✗        | Inherits poisoned shape from np.eye         |
| `np.add(np.eye(m), x)`          | ✗        | Same — downstream ops all fail              |

## Impact

This is the **sole** blocker for running a full Kalman filter + RTS smoother
under `grad()` / `valueAndGrad()`. All other operations (matmul of any shape
combination, lax.scan, np.sum, np.dot, np.square, np.log, np.exp, etc.) work
correctly once `np.eye` is replaced with `np.diag(np.ones([n]))`.

Previously reported matmul failures (`[m,m]@[m,m]` failing under grad) were
all caused by np.eye contamination in the identity matrix used to initialize
covariance. With `np.diag(np.ones([m]))`, matmul works for all shape
combinations under AD.

## Workaround

Replace every `np.eye(n, opts)` with `np.diag(np.ones([n], opts))`:

```ts
// Before (breaks under grad)
const I = np.eye(m, { dtype });

// After (works under grad)
const I = np.diag(np.ones([m], { dtype }));
```

## Context

Discovered while implementing MLE parameter estimation for `dlm-js` via
`valueAndGrad` + Adam optimizer. The full Kalman filter (100 timesteps, m=2
state dimension, using `lax.scan`) differentiates correctly with this single
substitution applied.
