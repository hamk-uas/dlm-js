# jax-js: WASM allocator / JIT cache memory grows unboundedly across `jit()` invocations

## Summary

When calling `jit(fn)(tensor)` repeatedly (hundreds of times) in a single Node.js process on the WASM backend, heap consumption grows monotonically and eventually triggers an OOM crash, even when:

- All input/output `np.Array` objects are properly disposed via `using` / `.dispose()`
- The JIT'd function's shape signature never changes (same dtypes, same shapes every call)
- No references to intermediate tensors are held by user code

The growth appears to come from WASM-side allocator fragmentation and/or JIT cache entries that are never evicted.

## Reproduction

A minimal reproducer using dlm-js's `dlmMLE` (which internally calls `jit(valueAndGrad(lossFn))`):

```typescript
import { defaultDevice } from '@hamk-uas/jax-js-nonconsuming';
import { dlmMLE } from './src/index';

defaultDevice('wasm');

// Same data, same options, same shapes every call
const y = Array.from({ length: 200 }, (_, i) => Math.sin(i * 0.1) + Math.random());

for (let i = 0; i < 400; i++) {
  // Each call creates a fresh jit(valueAndGrad(lossFn)) internally,
  // runs ~10 iterations, and returns plain JS scalars + TypedArrays.
  const result = await dlmMLE(y, {
    order: 0,
    init: { obsStd: 1, processStd: [0.5] },
    dtype: 'f64',
    optimizer: 'natural',
    maxIter: 50,
    tol: 1e-6,
  });
  // result.fit contains only TypedArray wrappers (StateMatrix, CovMatrix) — no np.Arrays
  if (i % 50 === 0) {
    const mem = process.memoryUsage();
    console.log(`i=${i}  rss=${(mem.rss / 1e6).toFixed(0)}MB  heap=${(mem.heapUsed / 1e6).toFixed(0)}MB`);
  }
}
```

**Expected**: RSS stays roughly constant after the first few JIT compilations.

**Observed**: RSS grows ~100MB per 20 calls. At ~100 calls the process hits the default 4GB V8 heap limit. With `--max-old-space-size=8192` it survives ~200 calls before crashing at 8GB.

## Where the memory is retained

Two likely sources (both internal to jax-js):

1. **JIT cache**: Each `dlmMLE` call creates a new `jit(valueAndGrad(lossFn))` closure. Even though the shape signature is identical, the JIT cache is keyed by closure identity, so every call adds a new compiled entry that is never evicted. This includes the WASM function pointers and the tracing artifacts.

2. **WASM linear memory**: The WASM allocator grows its linear memory (`WebAssembly.Memory`) to accommodate peak usage during JIT tracing + execution, but never shrinks it. After disposal, the WASM-side free list holds the blocks, but V8 still sees the full ArrayBuffer backing the linear memory. Over hundreds of calls, fragmentation prevents reuse and forces further growth.

## Impact

This prevents long-running or batch workloads from using jax-js in a single process. Current workaround in dlm-js: fork a subprocess per batch item to get a fresh WASM instance.

## Suggested fixes

1. **JIT cache eviction**: Add an LRU eviction policy to the JIT cache, or allow users to explicitly clear it (`jax.clearCaches()`). Even a simple "max 100 entries" policy would prevent unbounded growth.

2. **JIT cache deduplication**: Key the cache by the traced computation graph (or its hash) rather than by closure identity. Multiple closures over the same function structure would share a single compiled entry.

3. **WASM memory pool reset**: Expose a `resetAllocator()` or `compactMemory()` API that defragments the WASM linear memory free list, or returns unused pages to the OS (where supported by the WASM runtime).

4. **Diagnostic API**: Expose `jax.memoryStats()` → `{ wasmPages, jitCacheEntries, liveArrays, … }` so users can monitor and debug memory growth.

## Workaround

Fork a subprocess per batch item:

```typescript
// Parent: search-natural-hyperparams.ts
for (const combo of grid) {
  const out = execFileSync('npx', ['tsx', 'worker.ts', ...combo.args], {
    timeout: 120_000, encoding: 'utf-8',
  });
  results.push(JSON.parse(out.trim().split('\n').pop()!));
}
```

Each subprocess gets a fresh WASM instance and JIT cache, so memory stays bounded.

## Environment

- `@hamk-uas/jax-js-nonconsuming` v0.7.8
- Node.js v24.13.0 (V8 12.x)
- WASM backend, Float64
- OS: Linux x86_64

## Severity

**Medium** — does not affect correctness, but blocks common batch/tuning patterns (grid search, cross-validation, Monte Carlo calibration) from running in a single process.
