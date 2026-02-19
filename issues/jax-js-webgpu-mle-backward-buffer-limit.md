# jax-js: `jit(valueAndGrad)` over `lax.associativeScan` fails on WebGPU — buffer limit (v0.7.1) / concatenate VJP (v0.7.2)

## Summary

`jit(valueAndGrad(lossFn))` (and `valueAndGrad` without `jit`) crashes on the
WebGPU backend when `lossFn` contains `lax.associativeScan` with a 3-tuple
compose body.  The error changed between versions:

| Version | `jit(valueAndGrad)` error | `valueAndGrad` (no jit) error |
|---------|--------------------------|--------------------------------|
| v0.7.1  | `Too many buffers (12) for WebGPU pipeline (max: 8)` | `Referenced tracer Array:float32[m,m] has been disposed` |
| v0.7.2  | `Nonlinear operation in backward pass for concatenate` | `Nonlinear operation in backward pass for concatenate` |

`dlmFit` (forward smoother, no AD) works correctly on WebGPU in both versions.
The WASM/CPU path for `valueAndGrad` over `lax.associativeScan` works fine in
both versions (112/112 dlm-js tests pass on WASM with v0.7.2).

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

Tested on: jax-js-nonconsuming v0.7.1 (buffer limit error) and v0.7.2
(concatenate VJP error).  `dlmFit` (no AD) works correctly on WebGPU with
both versions.

## Expected behaviour

`jit(valueAndGrad(fn))` (and bare `valueAndGrad(fn)`) where `fn` uses
`lax.associativeScan` with a multi-output compose body should succeed on WebGPU.
Possible approaches:

**For the v0.7.2 `concatenate` VJP issue:**
1. Implement/fix the `concatenate` VJP on the WebGPU backend so it is
   recognised as differentiable.
2. Replace the internal `concatenate` in the `lax.associativeScan` backward
   implementation with a stack/split approach whose VJP is implemented.

**For the v0.7.1 buffer limit (may still apply after fixing concatenate):**
1. **Kernel splitting**: when the fused backward shader would exceed 8 buffers,
   split it into multiple sequential dispatches each under the limit.
2. **Bind group tiling / multiple bind groups**: distribute buffers across
   multiple bind group slots (WebGPU allows up to 4 bind group slots).
3. **Checkpoint + recompute strategy**: rematerialise intermediate results from
   checkpointed inputs instead of saving all activations.

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
