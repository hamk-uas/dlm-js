# jax-js: lax.scan on WebGPU is O(n) regardless of jit — compiled-loop is WASM-only or unavailable for reduction ops

## Summary

`lax.scan` on WebGPU dispatches O(n) GPU kernels even when wrapped in `jit()`,
while `lax.associativeScan` inside `jit` achieves the architecturally-optimal
⌈log₂N⌉+1 dispatches.  The jax-js team stated that `lax.scan` inside `jit`
should produce 1 dispatch (compiled in-shader loop), but empirical testing shows
this does **not** hold on WebGPU when the step body contains reduction ops
(einsum/matmul).  The compiled-loop may be WASM-only, or may only apply to
purely elementwise scan bodies.

The impact in dlm-js is that `dlmFit` on WebGPU scales O(n) with the series
length.  The forward filter uses `lax.associativeScan` (⌈log₂N⌉+1 dispatches,
fast), but the **backward RTS smoother** uses `lax.scan` inside `jit(core)`,
which still runs O(n) dispatches on WebGPU.

## Reproduction

Minimal repro script: [`issues/repro-associativescan-dispatch.ts`](repro-associativescan-dispatch.ts)

```
deno run --no-check --unstable-webgpu --allow-read --allow-env \
  issues/repro-associativescan-dispatch.ts
```

The script benchmarks four cases with m=2, N=100…1600 on WebGPU/Float32:

| Case | Description |
|------|-------------|
| A | `lax.associativeScan(compose, elems)` — bare, no jit |
| B | `jit(fn)(elems)` where `fn` calls `lax.associativeScan` |
| C | `lax.scan(backwardStep, carry, elems, {reverse:true})` — bare, no jit |
| D | `jit(fn)(carry, elems)` where `fn` calls `lax.scan` — how dlm-js backward smoother actually runs |

## Measured results — commit `3112787c` on `fix/jit-scan-einsum-maxargs`

```
       N   A:assocScan  ratio  B:jit(aScan)  ratio      C:scan  ratio  D:jit(scan)  ratio
     100          15.5      -          15.4      -        48.0      -         50.7      -
     200          17.5  1.12x          16.7  1.08x       104.8  2.18x        122.2  2.41x
     400          21.6  1.24x          21.2  1.27x       240.5  2.30x        253.2  2.07x
     800          24.2  1.12x          24.1  1.14x       526.0  2.19x        551.6  2.18x
    1600          25.3  1.05x          25.5  1.06x      1046.2  1.99x       1045.5  1.90x
```

- **Cases A and B**: ~15–25ms across N=100–1600, ratio ~1× — `lax.associativeScan` dispatches ⌈log₂N⌉+1 kernels, architecturally optimal for WebGPU. ✅
- **Cases C and D**: ~48–1046ms, ratio ~2× per doubling — O(n) regardless of `jit()`. C (bare) and D (jit) show nearly identical timing, meaning `jit` provides no compiled-loop benefit for `lax.scan` on WebGPU in this case. ❌

## jax-js team clarification vs empirical findings

The jax-js team provided this dispatch table:

| | lax.scan | lax.associativeScan |
|-|----------|---------------------|
| WebGPU dispatches | 1 (in-shader for-loop) | ⌈log₂N⌉+1 (architecturally optimal) |
| WASM invocations | 1 (entire loop in WASM) | ⌈log₂N⌉ (JS→WASM per round, fixable) |

Empirical Case D contradicts the WebGPU row for `lax.scan`: `jit(lax.scan)` is
O(n) dispatches, not 1.  Possible explanations:

1. **Compiled-loop is WASM-only** (`codegenNativeScanGeneral`): the "1 dispatch"
   figure may apply to WASM and not yet to WebGPU.
2. **Reduction ops prevent compilation**: `backwardStep` uses einsums with inner
   sums (matmul-style contractions).  The compiled-loop codepath may only handle
   elementwise bodies; `reductionEndpointEqns` in the body may bypass it.
3. **Silent jit fallback**: shape variation (different N per call) may cause jit
   to fall back to eager if the backend doesn't cache per-shape compiled shaders.

## Root cause in dlm-js

`dlmFit` on WebGPU calls `dlmSmo` twice, each inside `jit(core)`.  `core` has:

1. **Forward filter**: `lax.associativeScan` — ⌈log₂N⌉+1 dispatches — fast ✅
2. **Backward RTS smoother**: `lax.scan(backwardStep, ..., {reverse:true})` — O(n) dispatches ❌
3. Vectorised diagnostics (lik, yhat, ystd) — O(1) data-parallel ✅

`backwardStep` contains 5 ops (2 einsum transposes + 2 einsum matmuls + 1 add).
At N=100 with ~0.5 ms/dispatch, the backward pass costs ~250 ms alone (2 dlmSmo
calls × ~5 ops × N=100 steps), consistent with the observed ~870 ms total.

## Architectural context (from jax-js team)

`lax.associativeScan` on WebGPU at ⌈log₂N⌉+1 dispatches is **already the
architectural optimum**.  True single-dispatch Kogge-Stone is impossible on
WebGPU: `workgroupBarrier()` only synchronises within a workgroup; there is no
cross-workgroup sync primitive.  Every round's results must be globally visible
before the next round begins, requiring a separate `dispatchWorkgroups()` call.

For `lax.scan`, a single-dispatch compiled loop *is* theoretically possible on
WebGPU (the sequential dependency fits in one shader with a time-step loop), but
it isn't working for bodies with reduction ops in the current implementation.

## Expected fix (upstream in jax-js)

1. **Priority fix — WebGPU `lax.scan` compiled loop for reduction bodies**: Extend
   `codegenNativeScanGeneral` (or equivalent) to handle scan bodies containing
   reductions/einsums on WebGPU.  This would reduce dlm-js backward smoother
   from O(n) to 1 dispatch, making the full `dlmFit` forward+backward cost
   ⌈log₂N⌉+2 dispatches regardless of N.

2. **Lower priority — WASM `lax.associativeScan` compiled module**: The current
   ⌈log₂N⌉ JS→WASM crossings per round could be eliminated by compiling the
   Kogge-Stone ladder into a single WASM module (analogous to
   `codegenNativeScanGeneral` for `lax.scan`).

## Workaround in dlm-js

**WASM is preferred for performance** (`defaultDevice('wasm')`): `lax.scan`
inside `jit` compiles to a single native WASM call; ~22 ms flat for
N=100–102400.  WebGPU is only useful when the GPU compute benefit outweighs the
dispatch overhead, which does not occur for dlm-js state sizes (m ≤ 8).

## References

- Repro script: [`issues/repro-associativescan-dispatch.ts`](repro-associativescan-dispatch.ts)
- Related upstream fix (assocScan correctness): [`issues/jax-js-webgpu-jit-einsum.md`](jax-js-webgpu-jit-einsum.md)
- Scaling benchmark: `pnpm run bench:scaling` → `assets/timings/bench-scaling.json`
- dlm-js backward smoother: `src/index.ts` `backwardStep` function
