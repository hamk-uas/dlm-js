# jax-js: `jit(valueAndGrad)` over `lax.associativeScan` fails on WebGPU — buffer limit (v0.7.1–3, fixed in v0.7.4) / tracer disposal after result (v0.7.3–4)

## Summary

`jit(valueAndGrad(lossFn))` (and `valueAndGrad` without `jit`) crashes on the
WebGPU backend when `lossFn` contains `lax.associativeScan` with a 3-tuple
compose body.  The error changed between versions:

| Version | `jit(valueAndGrad)` error | `valueAndGrad` (no jit) error |
|---------|--------------------------|--------------------------------|
| v0.7.1  | `Too many buffers (12) for WebGPU pipeline (max: 8)` | `Referenced tracer Array:float32[m,m] has been disposed` |
| v0.7.2  | `Nonlinear operation in backward pass for concatenate` | `Nonlinear operation in backward pass for concatenate` |
| v0.7.3  | `Too many buffers (9) for WebGPU pipeline (max: 8)` | **computes correctly**, then `Referenced tracer Array:float32[1] has been disposed` |
| v0.7.4  | **computes correctly**, then `Referenced tracer Array:float32[1] has been disposed` | **computes correctly**, then `Referenced tracer Array:float32[1] has been disposed` |

`dlmFit` (forward smoother, no AD) works correctly on WebGPU in all versions.
The WASM/CPU path for `valueAndGrad` over `lax.associativeScan` works fine in
all versions (112/112 dlm-js tests pass on WASM with v0.7.4).

This blocks `dlmMLE` from running on the WebGPU backend.

## Context: where this arises in dlm-js

`dlmMLE` on WebGPU + Float32 dispatches to `makeKalmanLossAssoc`, which uses
`lax.associativeScan` to compute the Kalman filter log-likelihood in O(log n)
depth.  Each scan element carries a `(A: [m,m], b: [m,1], S: [m,m])` struct and
the compose function performs 5 tensor ops:

```typescript
// compose((A_a, b_a, S_a), (A_b, b_b, S_b)) → (A_c, b_c, S_c)
A_c = A_b @ A_a                    // einsum nij,njk→nik
b_c = A_b @ b_a + b_b              // einsum + add
S_c = A_b @ S_a @ A_b' + S_b      // einsum (3-arg) + add
```

The AD backward pass through this compose function must hold inputs and
intermediate activations alive across the backward sweep.  When jax-js fuses the
associativeScan backward into a single WebGPU shader, the bind group needs
references to all inputs + outputs — counting inputs, outputs, and intermediates
of the compose body simultaneously gives > 8 storage buffers.

## v0.7.1 failure details

### Failure 1: `jit(valueAndGrad(lossFn))` — buffer limit (12 > 8)

```typescript
const lossFn = makeKalmanLossAssoc(...); // uses lax.associativeScan internally
const optimStep = jit((theta, optState) => {
  const [lik, grad] = valueAndGrad(lossFn)(theta);
  const [newTheta, newState] = optimizer.update(grad, optState, theta);
  // ...
});
// → Error: Too many buffers (12) for WebGPU pipeline (max: 8)
```

This is the natural implementation.  `jit` traces `valueAndGrad(lossFn)`, fuses
the backward kernel, and produces a bind group with 12 buffers.

### Failure 2: `valueAndGrad(lossFn)` without `jit` — tracer disposal conflict

Without `jit`, the `using` keyword (TC39 explicit resource management) in the
scan body disposes forward-pass tensor results as each traced scope exits,
before the AD backward pass can read them:

```typescript
// Inside lossFn — forward pass under valueAndGrad (no jit):
function compose(a, b) {
  using A_c = np.einsum('ij,jk->ik', b.A, a.A);  // disposed when scope exits!
  using b_c = ...;
  using S_c = ...;
  return [A_c, b_c, S_c];  // returned refs are already disposed
}
// → Error: Referenced tracer Array:float32[2,2] has been disposed
```

`using` is correct inside `jit()`-traced bodies — the JIT tracer intercepts
disposal and manages tensor lifetimes.  But without `jit`, disposal is immediate
and the backward pass loses its saved activations.

## v0.7.2 failure details

In v0.7.2 both `jit(valueAndGrad)` and bare `valueAndGrad` fail with:

```
Nonlinear operation in backward pass for concatenate
```

This indicates the v0.7.2 `lax.associativeScan` backward-pass implementation
emits an internal `concatenate` op whose VJP is not implemented (or is
considered "nonlinear") on the WebGPU backend.  The WASM backend does not hit
this error — the associativeScan VJP works correctly on WASM in v0.7.2.

## v0.7.3 status

v0.7.3 fixes the `concatenate` VJP regression from v0.7.2.  Two residual issues
remained:

- `jit(valueAndGrad)`: buffer count reduced 12 → 9, still 1 over limit.
- `valueAndGrad` (no jit): computes correctly, then tracer disposal after result.

## v0.7.4 status

v0.7.4 fixes the buffer limit.  **Both** `jit(valueAndGrad(lossFn))` and bare
`valueAndGrad(lossFn)` now compute the correct result:

```
lik=284451.6875  grad=4813868.5000   ← correct
```

One residual issue remains: a tracer disposal error thrown *after* the result
is returned, in both cases:

```
Referenced tracer Array:float32[1] has been disposed
```

The computation completes successfully before the error — the loss and gradient
values are available.  The error occurs during backward-pass cleanup, suggesting
an intermediate activation tensor reference is retained beyond its scope and
then accessed after disposal.  The fix likely involves adjusting the lifetime
or access order of saved activations in the WebGPU backward-pass finalisation.

Note: the repro compose function uses `const` (not `using`) for returned values
`a_new`/`b_new`/`c_new`, and the outer function uses `const lik` (not `using`)
for the returned scalar.  Removing `using` from returned values does not affect
the error — inside a traced body, jax-js intercepts `using` disposal and manages
tensor lifetimes symbolically, so `using` on returned values is handled
correctly.  The disposal error originates inside jax-js's own backward-pass
finalisation, not in the user code.

## Reproduction

Minimal self-contained repro: [`issues/repro-webgpu-mle-buffer-limit.ts`](repro-webgpu-mle-buffer-limit.ts)

```
deno run --no-check --unstable-webgpu --allow-read --allow-env \
  issues/repro-webgpu-mle-buffer-limit.ts
```

The repro uses a single differentiable parameter `theta=[1]` (no split needed)
and a 3-tuple compose `(a,b,c) ⊕ (d,e,f) → (d·a, d·b+e, d·c+f)` — 5 ops, no
matmul, pure element-wise.  This is the minimum structure needed to reproduce
both failure modes.

Also reproducible via dlm-js directly (full Kalman loss):

```
deno run --no-check --unstable-webgpu --allow-read --allow-env \
  tmp/test-mle-webgpu.ts
```

Tested on: v0.7.1 (buffer limit, 12), v0.7.2 (concatenate VJP error),
v0.7.3 (buffer limit 9; bare `valueAndGrad` computes correctly then disposal
error), v0.7.4 (buffer limit fixed; both paths compute correctly then disposal
error).  `dlmFit` (no AD) works correctly on WebGPU with all versions.

## Expected behaviour

`jit(valueAndGrad(fn))` (and bare `valueAndGrad(fn)`) where `fn` uses
`lax.associativeScan` with a multi-output compose body should complete without
error on WebGPU.  As of v0.7.4, computation is correct — only the post-result
cleaning needs fixing:

**Remaining: tracer disposal after result (v0.7.4)**
- Fix the backward-pass cleanup order so that saved activation tensors are not
  accessed after their scope has disposed them.
- Alternatively, extend the lifetime of the relevant intermediate(s) until the
  full backward finalisation is complete.

## Impact

- `dlmMLE` on WebGPU is almost unblocked as of v0.7.4 — the residual tracer
  disposal error is thrown after the correct result is returned, so it only
  prevents clean integration (exception propagation breaks the optimizer loop).
- The issue affects any use of `jit(valueAndGrad)` or `valueAndGrad` over a
  loss from `lax.associativeScan` with tuple-valued compose bodies.
- Workaround in dlm-js: currently uses `makeKalmanLoss` (sequential scan,
  CPU/WASM only) for MLE and generates a static placeholder SVG for the
  WebGPU MLE animation.  Will enable the full WebGPU MLE path once the
  disposal error is resolved.

## Related issues

- [`jax-js-webgpu-jit-einsum.md`](jax-js-webgpu-jit-einsum.md) — shape-inference
  failures in `jit(scan)` (resolved in v0.7.1 / `fix/jit-scan-einsum-maxargs`)
- [`jax-js-webgpu-laxscan-sequential-dispatch.md`](jax-js-webgpu-laxscan-sequential-dispatch.md)
  — O(n) backward RTS smoother dispatch (architectural, not a bug)
