# jax-js: lax.scan dispatches each step from JS on WebGPU — O(n) overhead

## Summary

`lax.scan` on WebGPU dispatches each of its *n* steps individually from
JavaScript.  This means a scan body with *P* ops incurs **P·n GPU kernel
launches** regardless of the scan direction.  For comparison,
`lax.associativeScan` with the same body now dispatches **P·log₂(n) kernels**
(fusion confirmed working in branch `fix/jit-scan-einsum-maxargs` commit
`3112787c`).

The impact in dlm-js is that `dlmFit` on WebGPU scales O(n) with the time
series length, despite the forward filter using the O(log n) `associativeScan`
path.  The **backward RTS smoother** uses a sequential `lax.scan` with ~5 ops
per step, making it the dominant O(n) bottleneck.

## Reproduction

Minimal repro script: [`issues/repro-associativescan-dispatch.ts`](repro-associativescan-dispatch.ts)

```
deno run --no-check --unstable-webgpu --allow-read --allow-env \
  issues/repro-associativescan-dispatch.ts
```

The script benchmarks three cases with m=2, N=100…1600 on WebGPU/Float32:

| Case | Description |
|------|-------------|
| A | `lax.associativeScan(compose, elems)` — parallel prefix |
| B | `jit(fn)(elems)` where `fn` calls `lax.associativeScan` |
| C | `lax.scan(backwardStep, carry, elems, { reverse: true })` — sequential |

## Measured results — commit `3112787c` on `fix/jit-scan-einsum-maxargs`

```
       N   A:assocScan  ratio   B:jit(scan)  ratio   C:lax.scan  ratio
     100          17.5      -          15.8      -         47.6      -
     200          16.8  0.96x          16.5  1.04x        102.4  2.15x
     400          20.3  1.21x          19.0  1.15x        201.1  1.96x
     800          20.8  1.02x          20.6  1.09x        380.4  1.89x
    1600          22.6  1.08x          23.0  1.11x       1053.5  2.77x
```

- **Cases A and B**: ~17–23ms constant across N=100–1600 (ratio ≈ 1×) — O(log n) kernel fusion is **working**.
- **Case C**: 47ms → 1054ms, ratio ≈ 2× per doubling — O(n) per-step JS dispatch — **not fixed**.

## Root cause

`lax.associativeScan` fuses its compose calls into batched GPU kernels at each
round (one dispatch per op per round).  `lax.scan` does **not** do the same: it
iterates the step function from JS one step at a time, synchronously issuing
the constituent GPU kernels for each step.

For a scan of length n with P ops in the body, the total GPU kernel launches are:
- `associativeScan`: P × log₂(n)  (fused, batched per round)
- `lax.scan`:        P × n         (one-by-one from JS)

## Impact on dlm-js

`dlmFit` on WebGPU calls `dlmSmo` twice (diffuse prior + final pass), each
of which runs inside `jit(core)`.  The `core` function contains:

1. **Forward filter**: `lax.associativeScan` — O(log n) — now fast ✅
2. **Backward RTS smoother**: `lax.scan(backwardStep, ..., { reverse: true })` — O(n) ❌
3. Vectorised diagnostics (lik, yhat, ystd) — O(1) data-parallel ✅

The backward smoother has ~5 ops per step (2 einsums + 2 more einsums + add).
At N=100, Case C already costs 47ms vs 17ms for assocScan; at N=1600, 1054ms
vs 23ms.  Matching the `bench:scaling` output where full `dlmFit` WebGPU
took 869ms at N=100 and 9214ms at N=1600.

## Expected fix (upstream in jax-js)

`lax.scan` should emit a native GPU loop / unrolled batched dispatch instead
of iterating from JS.  Conceptually, option A (native loop) is preferred:
the GPU driver can pipeline stages; JS overhead per step is eliminated.

Option A — **native GPU loop**: compile the scan body to WGSL, emit a shader
that loops `n` times internally.  One launch per op (P total launches regardless
of n).  `O(1)` JS overhead.

Option B — **batched dispatch from JS once at startup**: at trace/JIT time,
pre-schedule all n dispatches into a `GPUCommandEncoder`; submit as a single
`commandBuffer`.  This removes JS-to-GPU round-trip latency per step but still
does P·n total GPU kernel executions.

Option C — **keep as-is** (not recommended for large n): per-step JS dispatch
at ~0.5ms/kernel means a 1000-step scan with 5 ops = 2.5s on WebGPU,
regardless of what the GPU is actually computing.

## Workaround in dlm-js

No perfect workaround exists without changing jax-js.  Current mitigations:

1. **WASM is preferred for performance** (`defaultDevice('wasm')`): WASM runs
   the sequential scan as a tight native loop; no JS-per-step overhead.
   ~22ms flat for N=100–102400 (see `bench:scaling` output).

2. **Associative-scan backward smoother** (research): the backward RTS smoother
   can be reformulated as a parallel associative scan (Solin 2021), enabling
   O(log n) GPU depth.  This requires non-trivial algebra and is deferred until
   `lax.scan` dispatch is fixed upstream, as it would be moot if upstream
   emits a native loop.

## References

- Repro script: [`issues/repro-associativescan-dispatch.ts`](repro-associativescan-dispatch.ts)
- Related upstream fix (assocScan fusion): [`issues/jax-js-webgpu-jit-einsum.md`](jax-js-webgpu-jit-einsum.md)
- Scaling benchmark: `pnpm run bench:scaling` → `assets/timings/bench-scaling.json`
- dlm-js backward smoother: `src/index.ts` `backwardStep` function
