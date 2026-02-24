# jax-js: WASM allocator exhausts 4 GB across jit() calls with varying input shapes

## Status: 🔴 Open

## Summary

The WASM bump allocator accumulates memory across repeated `jit()` invocations with different input shapes (different `lax.scan` sequence lengths). Even when all output `np.Array` objects are properly disposed via `Symbol.dispose`, the underlying WASM memory is never reclaimed. After ~33 calls with ascending N values (100 → 204,800), the allocator exhausts the WebAssembly 4 GB limit and crashes with `WebAssembly.Memory.grow(): Maximum memory size exceeded`.

The same workload (N=819,200 with m=2, f64) succeeds when run as the **first** call in a fresh process. The issue is purely about cross-call memory accumulation.

## Affected version

- `@hamk-uas/jax-js-nonconsuming` commit `062eb9e` on `ultimate-architecture-plan`
- Also affects prior commit `a1037b4` (the accumulation pattern is the same; the threshold N just shifts slightly between versions)

## Root cause hypothesis

`WasmAllocator.#bumpAlloc` only grows the WASM memory; it never compacts or reuses freed regions. When `jit()` is called with a new input shape:

1. A new `JitProgram` is compiled (or an existing one is reused from cache).
2. The JIT program allocates workspace buffers via `WasmBackend.malloc` → `WasmAllocator.malloc` → `#bumpAlloc`.
3. After execution, the output tensors are copied to JS-side `TypedArray`s and the `np.Array` wrappers are disposed.
4. But the WASM-side bump allocator **does not reclaim** the freed regions — the bump pointer never moves backward.

Each call with a different N effectively leaves a "high water mark" in the WASM linear memory. After enough distinct shapes, the accumulated high water marks exhaust the 4 GB limit.

## Reproduction

```bash
npx tsx issues/repro-wasm-oom-large-n.ts
```

**Script:** [`issues/repro-wasm-oom-large-n.ts`](repro-wasm-oom-large-n.ts)

### Results

```
=== WASM OOM reproducer ===
N=819,200, m=2, dtype=f64
Expected output memory: ~92 MB
WASM limit: 4096 MB

Test 1: Single cold dlmFit call at N=819200...
  ✓ Success (yhat length: 819200)

Test 2: N=409600 (warmup+timed) then N=819200...
  N=409600: 3 calls completed.
  ✓ N=819200 success after N=409600 warmup (yhat length: 819200)

Test 3: Full ascending sequence 100 → 819200...
  N=    100: ✓
  N=    200: ✓
  ...
  N=102,400: ✓
  N=204,800: ✗ OOM — WebAssembly.Memory.grow(): Maximum memory size exceeded
  → Failed after 11 prior N values × 3 calls = 33 prior dlmFit calls
```

**Key observations:**
- **Test 1** (cold N=819,200): ✅ succeeds — memory is sufficient for a single call
- **Test 2** (3× N=409,600 then N=819,200): ✅ succeeds — same-shape calls don't accumulate
- **Test 3** (ascending N sequence, 3 calls each): ❌ fails at N=204,800 — 33 prior calls with 11 distinct shapes exhaust WASM memory

This shows the problem is **cross-shape accumulation**, not the absolute size of any single call.

## Memory analysis

For a single `dlmFit(y, { order: 1, dtype: 'f64' })` with N=819,200, m=2:
- Output arrays (yhat, ystd, x, C, x_pred, C_pred): ~92 MB
- `lax.scan` carry storage (forward + backward): ~79 MB
- Total live data: ~171 MB

The 4 GB limit should accommodate many such calls. The problem is that freed WASM memory from prior calls is never reused.

## Impact on dlm-js

The scaling benchmark (`scripts/bench-scaling.ts`) measures `dlmFit` at 14 ascending N values (100 → 819,200), with 2 warmup + 4 timed runs per N. After ~33 calls, the WASM allocator exhausts 4 GB. Workaround: reduced max N to 409,600 and added error handling.

## Suggested fix

1. **Arena/pool allocator**: Replace the bump allocator with one that can reuse freed regions. A simple free-list or buddy allocator would allow reclaiming workspace memory between `jit()` calls.

2. **JIT workspace reset**: After a `JitProgram.execute()` completes, explicitly reset the bump pointer for the workspace region (keeping the JIT cache and persistent allocations intact). This is safe because workspace buffers are only live during execution.

3. **Explicit `backend.reset()` API**: Expose a method for users to reset the WASM allocator state when they know no tensors are live. This would allow benchmark scripts to call `reset()` between iterations with different shapes.

## Related

- [`jax-js-wasm-memory-growth.md`](jax-js-wasm-memory-growth.md) — same root cause observed with repeated `dlmMLE` calls (hundreds of optimizer iterations across multiple `jit()` invocations)
