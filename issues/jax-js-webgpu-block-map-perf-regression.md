# WebGPU `associativeScan` dispatch count — performance target for block-map fusion

🔴 **Open**

## Summary

dlm-js uses `lax.associativeScan` for both the forward Kalman filter and backward RTS smoother on WebGPU (`algorithm: 'assoc'`). The compose functions contain matmul, einsum, and inv operations on small matrices (m×m, typically m=2–5).

**Current state:** WebGPU `dlmFit` is 20–200× slower than WASM, scaling with N. This has always been the case — the old `ultimate-architecture-plan` branch was 10–18× slower, and `block-map` is currently 20–200×. Neither is acceptable for production use.

**Root cause:** Too many GPU dispatches. The Kogge-Stone prefix scan dispatches ⌈log₂N⌉+1 rounds, and each round dispatches multiple kernels for the matmul/einsum/inv operations in the compose function. At N=25,600, that's ~15 rounds × multiple kernels per round = hundreds of dispatches for a single `associativeScan` call, and `dlmFit` runs two of them (forward + backward).

**Target:** With jaxpr-based block-map fusion, the entire `associativeScan` compose function should fuse into a single kernel per round, and ideally the entire scan into 1–2 dispatches total. This would make WebGPU competitive with (or faster than) WASM for large N, which is the whole point of GPU acceleration.

## What dlm-js needs from `associativeScan`

The `dlmFit` forward filter composes 5-tuple elements $(A, b, C, \eta, J)$ where each component is `[n, m, m]` or `[n, m, 1]`. The compose function (`composeForward`) does:
- Matrix multiplications (`np.matmul`)
- Matrix-vector products (`np.einsum`)
- Matrix inversions (`np.linalg.inv`) with regularization
- Element-wise additions

The backward smoother has a similar compose function (`composeBackward`).

**Ideal dispatch budget for `dlmFit`:**
- Forward scan: 1–2 dispatches (fused Kogge-Stone with all compose ops in one kernel)
- Backward scan: 1–2 dispatches
- Total: ≤5 dispatches for the entire filter+smoother

For comparison, WASM sequential scan at N=25,600 takes ~46ms. Even matching that on WebGPU would be a 200× improvement over current block-map performance (~10,600ms).

## Current measurements (block-map branch, 0e5a982)

### Cold single-call timings (Nile order=1, m=2, includes JIT)

| N | WASM/f64 (warm) | WebGPU/f32 (cold) | ratio |
|---|-----------------|-------------------|-------|
| 100 | 32 ms | 649 ms | 20× |
| 3200 | 31 ms | 1,839 ms | 59× |
| 12800 | 37 ms | 5,436 ms | 147× |
| 25600 | 46 ms | 10,676 ms | 232× |

### Warm-run timings (small N, varying m)

| Model | m | WASM/f64 (warm) | WebGPU/f32 (warm) | ratio |
|-------|---|-----------------|-------------------|-------|
| Nile order=0 | 1 | 24 ms | 395 ms | 16× |
| Nile order=1 | 2 | 36 ms | 399 ms | 11× |
| Kaisaniemi trig | 4 | 26 ms | 740 ms | 28× |
| Energy trig+AR | 5 | 25 ms | 799 ms | 32× |

The warm-run table shows that even with JIT already compiled, WebGPU is 11–32× slower at small N (100–120 data points). The scaling table shows this gets dramatically worse with N.

## Reproduction

Run with Deno (requires WebGPU):

```bash
deno run --allow-read --allow-env --allow-net --allow-ffi --allow-sys --unstable-webgpu issues/repro-webgpu-block-map-perf.ts
```

## What fusion should target

The compose functions in dlm-js are representative of real `associativeScan` workloads — small-matrix algebra on batched elements. If block-map fusion can express:

1. **Fused Kogge-Stone round:** All matmul/einsum/inv ops in `composeForward` become a single kernel operating on the N-length batch
2. **Fused scan:** Multiple rounds expressed as a single GPU dispatch (or minimal dispatches with GPU-side synchronization)

...then WebGPU `associativeScan` would go from hundreds of dispatches to single-digits, and the 20–200× overhead should collapse to near-parity with WASM or better.

## Hardware

- GPU: NVIDIA RTX 4070 (eGPU, Thunderbolt 4)
- OS: Linux (Ubuntu)
- Deno: v2.7.1
