# jax-js: `jit(valueAndGrad)` over `lax.associativeScan` exceeds WebGPU 8-buffer bind-group limit

## Summary

`jit(valueAndGrad(lossFn))` crashes on the WebGPU backend when `lossFn`
contains `lax.associativeScan` with a compose body that outputs a tuple of
matrices (e.g. `(A, b, S)`).  The backward-pass kernel fusion produces fused
shaders that reference **12 storage buffers** in a single bind group, exceeding
WebGPU's hard architectural limit of **8 storage buffers per bind group** (per
the WebGPU spec and all current implementations).

This blocks `dlmMLE` from running on the WebGPU backend: `dlmFit` (forward
smoother, no gradients) works fine on WebGPU, but MLE requires differentiating
through the scan loss, which triggers the buffer overflow.

**Error observed:**
```
Error: Too many buffers (12) for WebGPU pipeline (max: 8)
```

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

## Two independent failure modes

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
and the backward pass loses its saved activations.  This means eager-mode AD
through scan bodies with `using` is not viable as a workaround.

## Reproduction

Minimal repro: [`issues/repro-webgpu-mle-buffer-limit.ts`](repro-webgpu-mle-buffer-limit.ts)

```
deno run --no-check --unstable-webgpu --allow-read --allow-env \
  issues/repro-webgpu-mle-buffer-limit.ts
```

Also reproducible via dlm-js directly:

```
# In tmp/test-mle-webgpu.ts:
deno run --no-check --unstable-webgpu --allow-read --allow-env \
  tmp/test-mle-webgpu.ts
```

Tested on: jax-js-nonconsuming v0.7.1, `fix/jit-scan-einsum-maxargs` branch
(commit f09121d / v0.7.1).  `dlmFit` (no AD) works correctly on WebGPU with
this version.

## Expected behaviour

`jit(valueAndGrad(fn))` where `fn` uses `lax.associativeScan` with a multi-output
compose body should succeed on WebGPU.  Possible approaches:

1. **Kernel splitting**: when the fused backward shader would exceed 8 buffers,
   split it into multiple sequential dispatches each under the limit.  Each
   dispatch reads from the previous one's output buffers.

2. **Bind group tiling / multiple bind groups**: use multiple bind groups per
   dispatch (WebGPU allows up to 4 bind groups with separate slots).  Buffers
   > 8 would be distributed across bind group slots 0–3.

3. **Checkpoint + recompute strategy**: instead of saving all activations for
   the backward pass, recompute intermediate results from checkpointed inputs
   (trading memory/buffer count for recomputation).  This is the standard
   rematerialization approach used in JAX's `jax.checkpoint`.

## Impact

- `dlmMLE` on WebGPU is completely blocked.  `dlmFit` (smoother, forward-only,
  no AD) works fine on WebGPU today.
- The issue likely affects any use of `jit(valueAndGrad)` over a loss built
  from `lax.associativeScan` with tuple-valued compose bodies of state dimension
  m ≥ 2, i.e. any non-trivial differentiable parallel scan.
- Workaround in dlm-js: WebGPU + Float32 falls back to `makeKalmanLossAssoc`
  without `jit`, but this hits Failure 2 (tracer disposal).  Currently dlm-js
  uses `makeKalmanLoss` (sequential scan, CPU/WASM only) for MLE and generates
  a static placeholder SVG for the WebGPU MLE animation.

## Related issues

- [`jax-js-webgpu-jit-einsum.md`](jax-js-webgpu-jit-einsum.md) — shape-inference
  failures in `jit(scan)` (resolved in v0.7.1 / `fix/jit-scan-einsum-maxargs`)
- [`jax-js-webgpu-laxscan-sequential-dispatch.md`](jax-js-webgpu-laxscan-sequential-dispatch.md)
  — O(n) backward RTS smoother dispatch (architectural, not a bug)
