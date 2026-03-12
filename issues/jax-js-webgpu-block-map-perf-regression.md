# WebGPU `associativeScan` dispatch count — performance target for block-map fusion

🟡 **Mitigated** — v0.8.4 (`cc53907`): command tape O8a + bind group cache O9b + analytical Cholesky n≤4 + subgroupShuffleUp/InclusiveAdd P8 + Dot vmap batch fix. Warm median **~35ms** at N=100, GPU/WASM ratio **~6×** (was ~71ms / ~13× in v0.8.2). Still above ≤2× target — remaining overhead is per-round dispatch latency.

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

## Failure observation: `3c37a3d` (2026-03-09)

Commits `39c9328` (stable jaxpr hash + unrolled polyfills) and `3c37a3d` (JIT execute loop optimization) claim WebGPU N=100 m=2 warm goes from 323ms → 7ms. On our hardware, **no improvement observed**:

| N | WASM/f64 (warm) | WebGPU/f32 (cold) | WebGPU/f32 (warm) | cold/WASM | warm/WASM |
|---|-----------------|-------------------|-------------------|-----------|-----------|
| 100 | 6.7 ms | 548 ms | 318 ms | 82× | 48× |
| 3200 | 8.2 ms | 1761 ms | 1561 ms | 216× | 191× |

Previous measurement at `0e5a982`: N=100 cold 649ms (20×), N=3200 cold 1839ms (59×). Warm runs are essentially unchanged from before the optimization commits.

Tests: 317/318 pass (correctness is fine — this is purely a performance issue).

## Observation: `3e580f4` — parallel readback (2026-03-09)

Commit `3e580f4` adds `tree.data()` / `tree.consumeData()` for parallel GPU readback (all `mapAsync` calls dispatched via `Promise.all`).

**Correction:** The `3c37a3d` claim of "323ms → 7ms warm" was wrong — the upstream benchmark used `dtype: "float32"` which `parseDtype` didn't recognise, silently falling back to Float64/WASM. Real WebGPU warm was still ~320ms.

### dlm-js adoption

Applied three changes to `src/index.ts`:
1. **Pass 2 (17 readbacks):** Replaced 17 sequential `consumeData()` calls with single `tree.consumeData(out2)` — parallel GPU readback.
2. **Pass 1 (2 readbacks):** Replaced sequential `out1.x.data()` / `out1.C.data()` with `Promise.all([...])`.
3. **dlmForecast (4 readbacks):** Same `tree.consumeData(out)` pattern.

### Results (Nile order=1, m=2, N=100, RTX 4070 eGPU)

| Metric | Before (`3c37a3d`) | After (`3e580f4` + parallel readback) |
|--------|-------------------|--------------------------------------|
| WebGPU warm | 318 ms | 126 ms |
| WebGPU cold | 548 ms | ~500 ms |
| WASM/f64 warm | 6.7 ms | 6.7 ms |
| warm ratio | 48× | 19× |

Warm improved 2.5× from parallel readback alone. Dispatch overhead (the kernel launch count) is unchanged — this optimization only reduces the readback tail.

### Remaining bottleneck

The dispatch count is still the dominant cost. At N=100, the forward+backward `associativeScan` generates hundreds of GPU kernel dispatches. Until block-map fusion collapses the compose function into a single kernel per round (or fewer rounds), WebGPU will remain 19–200× slower than WASM for typical dataset sizes.

**Status: 🔴 Open** — parallel readback is adopted, but the core dispatch-count issue persists.

## Observation: `689d8a5` — WASM compiled-loop + binding limit guard (2026-03-09)

Commit `689d8a5` (single commit since `3e580f4`) adds three features:

1. **Multi-output kernel slot mapping** in `planAssociativeScan` — fixes WASM "unmapped slot" crash for multi-output compose functions like DLM's 5-tuple.
2. **Adaptive block size** with shmem pre-estimation (tries B=256, 128, 64, 32).
3. **Storage buffer binding limit check** — rejects block-map when bindings > `maxStorageBuffersPerShaderStage`. For the 5-tuple forward compose: `numConsts + 2×numLeaves = 1 + 2×5 = 11 bindings > limit 10` → falls back to Kogge-Stone on WebGPU.

### WASM compiled-loop activation

The WASM compiled-loop-blocked path now activates for the forward 5-tuple compose:
```
[assoc-scan] SUCCESS! Using WASM compiled-loop-blocked (B=256) with 4 step(s)
```

Confirmed via `setDebug(2)`. The jaxpr traces per-element shapes `float64[2,2]` — both `np.einsum('nij,njk->nik', ...)` and `np.matmul(...)` lower to the same `dot_general` primitive and both activate the compiled-loop equally. **No einsum→matmul conversion needed.**

### A/B test: einsum vs matmul in compose functions

Tested both variants at `dlmFit` level (Nile N=100, m=2, 30 warm runs each):

| Compose variant | scan p50 | assoc p50 | assoc/scan |
|-----------------|----------|-----------|------------|
| einsum (original) | 3.29 ms | 12.98 ms | 3.94× |
| matmul (converted) | 3.25 ms | 12.58 ms | 3.87× |

Identical within noise. The jaxpr tracing normalises both to `dot_general` — the compiled-loop doesn't care which JS-level API produced it.

### WebGPU unchanged

WebGPU warm: ~122 ms (Nile order=1, m=2). The 5-tuple binding limit (11 > 10) means WebGPU still uses Kogge-Stone fallback. The 3-tuple backward compose (0 constants + 6 leaves = 6 bindings ≤ 10) could potentially use block-map, but this isn't observed yet.

### Scaling (bench-scaling.ts, Nile order=1, m=2)

| N | WASM/f64 (warm) | WebGPU/f32 (cold) | ratio |
|---|-----------------|-------------------|-------|
| 100 | 9.5 ms | 366 ms | 38× |
| 800 | 7.1 ms | 736 ms | 104× |
| 3200 | 9.0 ms | 1,692 ms | 188× |
| 12800 | 16.3 ms | 5,300 ms | 325× |
| 51200 | 42.9 ms | 19,363 ms | 451× |
| 102400 | 100.1 ms | timeout | — |

Ratios are worse than `3e580f4` because WebGPU cold includes JIT compile time while WASM is warm (polymorphic JIT).

**Status: 🔴 Open** — WASM compiled-loop activates (scan itself is fast), but full `dlmFit(algorithm:'assoc')` is still ~4× slower than `scan` due to element construction and smoother overhead. WebGPU dispatch count unchanged.

## Observation: `cd8d47a` — constants → uniform buffers (2026-03-10)

Commits since v0.8.0:
- `10ae2aa`: Moves small constants (≤4 elements) from group(0) storage bindings to group(1) uniform buffers in block-map fused shaders. This directly addresses the 5-tuple binding limit blocker.
- `cd8d47a`: Einsum cleanup (renamed `_compose5_einsum` to active, fixed misleading comment).

### 5-tuple block-map now activates on WebGPU

**Before (`689d8a5` / v0.8.0):** 5-tuple forward compose needed `1 const + 5 in + 5 out = 11 storage bindings > maxStorageBuffersPerShaderStage (10)` → fell back to Kogge-Stone with per-op dispatches per round.

**After (`cd8d47a`):** Constants ≤4 elements use group(1) uniform buffers, so `5 in + 5 out = 10 storage bindings ≤ 10` → block-map fused path activates:

```
[assoc-scan] SUCCESS! Using WebGPU block-map path (B=256)
block_map: using fused WebGPU shader path
```

### Warm timings (Nile order=1, m=2, RTX 4070 eGPU)

| N | WASM/f64 (warm) | GPU/f32 assoc (warm) | GPU/f32 scan (warm) | assoc/WASM | scan/WASM |
|---|-----------------|---------------------|---------------------|------------|-----------|
| 100 | 8 ms | 123 ms | 136 ms | 15× | 17× |
| 800 | 7 ms | 428 ms | 979 ms | 63× | 144× |

Compared to v0.8.0 warm (N=100): assoc 158ms → 123ms (**~21% faster**), scan 200ms → 136ms (**~32% faster**).

### Assessment

The specific binding-limit blocker is resolved — the 5-tuple forward compose now uses the block-map fused path on WebGPU, reducing per-round dispatch count for the compose function. However:

1. **GPU/WASM ratio still 13–63× at N=100–800.** The remaining overhead is in the full `dlmFit` pipeline: element construction, backward smoother scan, diagnostic recovery, readback. These still generate many dispatches.
2. **Ratio grows with N**, indicating the total dispatch count still scales with data size.
3. **Block-map B=256** means the local scan phase handles up to 256 elements in one dispatch, but global merge rounds still add dispatches proportional to ⌈log₂(N/B)⌉.

**Status: 🟡 Mitigated** — 5-tuple binding limit resolved, ~21% warm speedup at N=100. The core dispatch-count architecture issue persists for the full pipeline.

## Hardware

- GPU: NVIDIA RTX 4070 (eGPU, Thunderbolt 4)
- OS: Linux (Ubuntu)
- Deno: v2.7.1
