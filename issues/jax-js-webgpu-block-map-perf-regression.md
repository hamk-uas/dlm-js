# WebGPU performance regression on `block-map` branch

🔴 **Open**

## Summary

After upgrading `@hamk-uas/jax-js-nonconsuming` from `ultimate-architecture-plan` (commit `d08dd54`) to `block-map` (commit `17d2257`), WebGPU `associativeScan` performance regressed dramatically. The regression scales with both N (dataset size) and m (state dimension):

- **Small N (100), small m (2):** ~12% slower (cold 541ms → 606ms)
- **Small N (100–120), medium m (4–5):** 45–55% slower (warm 510ms → 740ms for m=4; 515ms → 799ms for m=5)
- **Large N (≥6400):** Catastrophic slowdown. At N=25,600: **11.3× slower** (899ms → 10,193ms). At N=51,200: **16.6× slower** (1,203ms → 19,986ms). N≥102,400 times out at 20s (was 1,827ms before).

All 317 tests pass with correct numerical results — this is a pure performance regression, not a correctness issue.

## Affected version

- **Regressed:** `block-map` branch, commit `17d22577b02172423832e6c5fa2add71199a8604`
- **Last known good:** `ultimate-architecture-plan` branch, commit `d08dd54`

## Root cause hypothesis

The `block-map` branch likely changed how WebGPU shader dispatch, buffer mapping, or kernel fusion works for `associativeScan`. The regression pattern — moderate at small N, catastrophic at large N — suggests:

1. **Per-round overhead increased** (buffer map/unmap, shader dispatch, or synchronization costs per Kogge-Stone round), which compounds across ⌈log₂N⌉+1 rounds.
2. **Kernel fusion degradation** — previously fused matmul/einsum operations inside `associativeScan` compose functions may now dispatch as separate kernels.
3. **Block-map memory access pattern** may be less cache-friendly for the strided access patterns in Kogge-Stone prefix scan.

The regression is worse for larger m (state dimension), consistent with per-dispatch overhead increases — larger m means more computation per round but the same number of dispatches.

## Reproduction

See `issues/repro-webgpu-block-map-perf.ts`. Run with:

```bash
deno run --allow-read --allow-write --allow-env --allow-run --allow-net --allow-ffi --allow-sys --unstable-webgpu issues/repro-webgpu-block-map-perf.ts
```

The script measures WebGPU `dlmFit` (algorithm: 'assoc') at N=100, 3200, 12800, and 25600. Expected output on the current `block-map` version will show significantly higher timings compared to the baseline values printed alongside.

## Data

### Scaling benchmark (Nile order=1, m=2, cold single-call including JIT)

| N | Before (d08dd54) | After (17d2257) | Slowdown |
|---|-------------------|------------------|----------|
| 100 | 541 ms | 606 ms | 1.12× |
| 800 | 543 ms | 842 ms | 1.55× |
| 3200 | 615 ms | 1,864 ms | 3.03× |
| 6400 | 652 ms | 3,043 ms | 4.67× |
| 12800 | 738 ms | 5,715 ms | 7.75× |
| 25600 | 899 ms | 10,193 ms | 11.3× |
| 51200 | 1,203 ms | >20,000 ms | >16.6× |
| 102400 | 1,827 ms | >20,000 ms | >10.9× |
| 1638400 | 17,978 ms | >20,000 ms | — |

### Warm-run benchmark (small N, varying m)

| Model | m | Before (d08dd54) warm | After (17d2257) warm | Slowdown |
|-------|---|----------------------|---------------------|----------|
| Nile order=0 | 1 | 339 ms | 395 ms | 1.16× |
| Nile order=1 | 2 | 349 ms | 399 ms | 1.14× |
| Kaisaniemi trig | 4 | 510 ms | 740 ms | 1.45× |
| Energy trig+AR | 5 | 515 ms | 799 ms | 1.55× |

## Impact on dlm-js

- **Scaling benchmark** (`pnpm run bench:scaling`): WebGPU rows at N≥102,400 now time out at 20s. The benchmark completes but reports `+Infinity` / `null` for those entries. Previously the full table completed in ~3 min; now only N≤51,200 finishes within timeout.
- **bench:full** (120 combinations): Completes — individual dlmFit calls at small N (100–120) are slow but under the 5s timeout.
- **MLE frame collection** (`collect-*-webgpu.ts`): Still works but slower (Energy: 35s vs ~20s before).
- **README scaling table**: Rows N≥102,400 show stale pre-regression values because the sidecar has `null` for timed-out entries and `update:timings` skips null slots.

## Suggested fix

1. Profile the WebGPU dispatch pipeline on the `block-map` branch vs `ultimate-architecture-plan` for a simple `associativeScan` with matmul-containing compose function at N=12,800.
2. Compare the number of `queue.submit()` calls and buffer operations per Kogge-Stone round between the two branches.
3. If the regression is in buffer mapping (the "block-map" in the branch name suggests a new buffer mapping strategy), consider whether the old mapping strategy can be preserved for `associativeScan` workloads where the same buffers are reused across rounds.

## Observations

### 2026-03-08: Commit a6f588f — regression NOT fixed

Updated to `a6f588f1cc20a287ab7b28fdeb373d1543c3e5dc` (block-map branch). All 317 tests pass (correctness OK). Repro output:

```
  N       | baseline (d08dd54) | observed | slowdown
      100 |     541 ms       |     645 ms | 1.19×
     3200 |     615 ms       |    1785 ms | 2.90× <<<
    12800 |     738 ms       |    5333 ms | 7.23× <<<
    25600 |     899 ms       |   10612 ms | 11.80× <<<
```

Numbers essentially identical to 17d2257. The fix in a6f588f does not address this regression.

### 2026-03-08: Commit 9b5bf7d — regression NOT fixed

Updated to `9b5bf7d79a0716129b8c29a34e5058f74d3d9500` (block-map branch). All 317 tests pass (correctness OK). Repro output:

```
  N       | baseline (d08dd54) | observed | slowdown
      100 |     541 ms       |     654 ms | 1.21×
     3200 |     615 ms       |    1786 ms | 2.90× <<<
    12800 |     738 ms       |    5372 ms | 7.28× <<<
    25600 |     899 ms       |   10523 ms | 11.70× <<<
```

Numbers identical to a6f588f and 17d2257. Three consecutive block-map commits tested — none address this regression.

## Hardware

- GPU: NVIDIA RTX 4070 (eGPU, Thunderbolt 4)
- OS: Linux (Ubuntu)
- Deno: v2.7.1
