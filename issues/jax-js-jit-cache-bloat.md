# JIT closure cache bloat: inline `jit(fn)()` accumulates GPU constants indefinitely

**Status**: 🟡 Internal + upstream ergonomics  
**Package**: `@hamk-uas/jax-js-nonconsuming`  
**Filed**: 2026-02-25  
**Upstream commit**: `d08dd54`  
**Context**: dlm-js Kalman filter library — repeated `dlmFit` calls accumulate ~30 GPU buffer slots per call from JIT-cached constants

## Summary

When `jit(fn)(args)` is called inline (no persistent reference to the jit wrapper), each invocation creates a new `JitProgram` with its own cached constant buffers on the GPU. These constants are only freed when `clearCaches()` is called or the closure is GC'd — but V8 GC timing for GPU resources is unreliable.

In dlm-js, `dlmSmo` calls `jit(core)(...)` inline every time. Each call caches ~15 constant tensors (stabilization matrices, identity, epsilon constants). Since `dlmFit` calls `dlmSmo` twice (Pass 1 + Pass 2), each `dlmFit` invocation leaks ~30 GPU slots.

## Reproduction

```ts
// Profile script (Deno --unstable-webgpu):
import { init, defaultDevice, getBackend, clearCaches, jit, numpy as np } from "@hamk-uas/jax-js-nonconsuming";
import { dlmFit } from "../src/index.ts";

await init("webgpu");
defaultDevice("webgpu");
const backend = getBackend("webgpu");

const y = Float32Array.from({ length: 100 }, () => Math.random() * 100);

console.log("Before:", backend.slotCount(), "slots");

for (let i = 0; i < 100; i++) {
  await dlmFit(y, { obsStd: 10, processStd: [5, 1], dtype: 'f32', order: 1 });
}

console.log("After 100 dlmFit:", backend.slotCount(), "slots");
// Observed: ~3000+ slots (30 per call × 100 calls)

clearCaches();
console.log("After clearCaches:", backend.slotCount(), "slots");
// Observed: ~1100 slots freed
```

## Measured impact (Nile model: n=100, m=2, Float32)

| Metric | Before fix¹ | After fix¹ | After clearCaches |
|--------|------------|-----------|-------------------|
| Slots after init | 0 | 0 | — |
| Slots after MLE (108 iters) | 85 | 82 | — |
| Slots after 109 dlmFit calls | 3715 | 3352 | 1121 |
| GPU bytes (jax-js tracked) | 1.6 MB | 1.5 MB | 862 KB |
| nvidia-smi VRAM | 236 MiB | 236 MiB | 236 MiB |

¹ "fix" = disposing `np.reshape` intermediates in dlm-js (3 sites). Remaining slots are JIT cache constants.

## Root cause analysis

### dlm-js side (our responsibility)

`dlmSmo` creates the `core` closure inline every call. The closure captures ~15 `using`-declared constant tensors (`stab_I_eye`, `stab_off_I`, `stab_nLeak_fact`, etc.) plus data arrays (`y_arr`, `V2_arr`, `x0`, `C0`). When `jit(core)` traces, these become **captured constants** in the `JitProgram` — each gets a GPU buffer that persists until the JIT closure is GC'd.

**dlm-js fix path**: Refactor `dlmSmo` to accept a pre-compiled `JitProgram` (or use a module-level cache keyed by `(stateSize, dtype, algorithm, stabilization)`) so the same compiled program is reused across calls. Constants that change per-call (y, V2, x0, C0) would become inputs instead of captured constants.

### Upstream ergonomics concern

The inline `jit(fn)(args)` pattern is natural but hazardous for GPU memory. Options the upstream could consider:

1. **`jit` with WeakRef caching**: If `jit(fn)` is called with the same `fn` reference, reuse the existing `JitProgram`. This is what JAX does (Python `jit` caches on the function identity).

2. **Warn on high slot count**: When `slotCount()` exceeds a threshold, log a warning suggesting `clearCaches()` or persistent `jit` references.

3. **Auto-evict by LRU**: JIT programs that haven't been called for N iterations could auto-evict their cached constants.

## Impact on dlm-js

- **Current severity**: Low. At 1.5 MB across 109 calls on a 16 GB GPU, this is negligible. nvidia-smi shows 236 MiB total (230 MiB is Dawn/Vulkan init overhead).
- **Scaling concern**: For long-running MLE optimization (300+ iterations, each calling `dlmFit` for frame collection), or for larger models (m=13, n=1000+), the linear growth could become problematic.
- **Workaround**: Call `clearCaches()` periodically in long-running loops.

## Suggested fix (dlm-js internal)

Refactor to hoist `jit(core)` with shape-based caching. The core function's behavior is determined by:
- `stateSize` (m)
- `dtype` (f32/f64)
- `algorithm` (scan/assoc/sqrt-assoc)
- `stabilization` flags

A `Map<string, JitFunction>` keyed by `${m}:${dtype}:${algo}:${stabFlags}` would eliminate all per-call JIT overhead. This is a medium-effort refactor — not urgent given the low memory impact.

## Workaround locations

- `src/index.ts:1459` — `jit(core)(...)` inline call in `dlmSmo`
- `src/mle.ts` — `jit(valueAndGrad(lossFn))` (already persistent per MLE run, OK)
