# jax-js: lax.scan backward (RTS smoother) is O(n) on WebGPU — intra-step buffer deps trigger executeScanFallback

## Summary

`lax.scan` on WebGPU dispatches O(n) GPU kernels even under `jit()` when the
step body has internal buffer dependencies between ops within a single
iteration.  The backward RTS smoother body has exactly this shape:
matmul→matmul→add where each op's output feeds the next *within the same step*.
WebGPU has no cross-workgroup barrier, so fusing N sequential steps into one
shader is architecturally impossible.  jax-js `planScan` correctly selects
`executeScanFallback()` — a JS for-loop calling `bodyProgram.execute()` once
per iteration — giving O(N) JS→GPU roundtrips.  This is the correct behaviour,
not a missing optimisation.

For comparison, `lax.associativeScan` inside `jit` achieves ⌈log₂N⌉+1
dispatches (Kogge-Stone, also architecturally optimal — each round requires a
separate dispatch due to the same lack of cross-workgroup barrier).  The forward
Kalman filter compose body has *no* intra-step deps (each op reads only its
own inputs), so it qualifies for the fused path.

The impact in dlm-js is that `dlmFit` on WebGPU scales O(n) with series length
despite the forward filter being O(log n).  The **backward RTS smoother** runs
inside `jit(core)` but cannot escape O(n) dispatches on WebGPU.

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

## Root cause — `executeScanFallback` (confirmed by jax-js team)

The jax-js `planScan` function applies a WebGPU compiled-loop eligibility test:

> **No internal buffer dependencies between steps** — if op outputs within one
> iteration feed other ops in the same iteration, the path falls back.

The backward RTS step body violates this:
```
At  = A'            (transpose  — reads elem.A)
Atr = At @ carry.r  (matmul     — reads At output)
r_new = Atr + elem.b             — reads Atr output
AtN  = At @ carry.N (matmul     — reads At output)
AtNA = AtN @ elem.A (matmul     — reads AtN output)
N_new = AtNA + elem.S            — reads AtNA output
```
Ops 2→3 and 4→5→6 form sequential chains of buffer dependencies within a
single iteration.  WebGPU has no `workgroupBarrier()` across workgroups and no
cross-dispatch barrier inside a single shader, so these chains cannot be
executed in one shader dispatch.  `planScan` selects `executeScanFallback()`:
a JS for-loop that calls `bodyProgram.execute()` once per step, each call
crossing the JS→GPU boundary and submitting a command buffer.  O(N) roundtrips.

Full dispatch-count breakdown:

| Path | Dispatch count | Why |
|------|---------------|-----|
| `associativeScan` forward filter | ⌈log₂N⌉+1 | Kogge-Stone rounds, JIT-fused, no intra-round deps |
| `lax.scan` backward RTS (eager *or* jit) | O(N) | Intra-step matmul→matmul deps → WebGPU fallback |
| `lax.scan` on WASM | O(1) JS→WASM | Entire loop runs inside one WASM module call |

## Impact on dlm-js

`dlmFit` on WebGPU calls `dlmSmo` twice, each inside `jit(core)`.  `core` has:

1. **Forward filter**: `lax.associativeScan` — ⌈log₂N⌉+1 dispatches — fast ✅
2. **Backward RTS smoother**: `lax.scan(backwardStep, ..., {reverse:true})` — O(N) via `executeScanFallback` ❌
3. Vectorised diagnostics (lik, yhat, ystd) — O(1) data-parallel ✅

`backwardStep` has sequential matmul→matmul→add chains within each step
(see dispatch table above).  At N=100 with ~0.5 ms/dispatch:
2 dlmSmo calls × N steps × ~P ops/step ≈ 870 ms observed at N=100.

## Fix options (confirmed by jax-js team)

Three options exist, in order of impact for dlm-js:

**Option 1 — Use WASM backend for backward scan** (drop-in, effective):
WASM's compiled-loop handles intra-step buffer dependencies by allocating
temporaries inside the module.  The entire N-iteration backward pass runs in
one JS→WASM call.  This is exactly what the current WASM path already does:
`dlmFit` on WASM is ~22 ms flat for N=100–102400.  No code change needed;
users should prefer WASM for time-series work.

**Option 2 — Reformulate backward smoother as `associativeScan`** (O(log N) on WebGPU):
The RTS recursion can be expressed as a parallel prefix over associative affine
maps (Solin 2021, parallel Kalman smoother).  Each element would encode an
affine map `r_prev = A·r_next + c`; the compose function is affine composition,
which has no intra-step deps and qualifies for the Kogge-Stone fused path.
This is the **only path to O(log N) dispatches on WebGPU** for the backward
smoother.  Requires non-trivial mathematical reformulation of the RTS recursion.

**Option 3 — Wait for WASM compiled-loop for `associativeScan`** (future jax-js work):
The current ⌈log₂N⌉ JS→WASM crossings for `associativeScan` could be
eliminated by compiling the Kogge-Stone ladder into a single WASM module.  This
would make the forward filter faster on WASM but does **not** help the backward
scan problem at all.

## Status in dlm-js

- **Option 1** is the current recommendation: use WASM (`defaultDevice('wasm')`).
  ~22 ms flat for N=100–102400.  WebGPU is slower for all practical dlm-js
  state sizes (m ≤ 8) because dispatch latency dominates GPU compute benefit.
- **Option 2** is deferred pending need; the mathematical reformulation is
  non-trivial and moot while WASM meets performance requirements.

## References

- Repro script: [`issues/repro-associativescan-dispatch.ts`](repro-associativescan-dispatch.ts)
- Related upstream fix (assocScan correctness): [`issues/jax-js-webgpu-jit-einsum.md`](jax-js-webgpu-jit-einsum.md)
- Scaling benchmark: `pnpm run bench:scaling` → `assets/timings/bench-scaling.json`
- dlm-js backward smoother: `src/index.ts` `backwardStep` function
