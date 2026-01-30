# Scan Primitive Implementation

This document describes the implementation of `lax.scan` in jax-js, a JAX-style scan operation for efficiently iterating over arrays while threading state.

## Overview

`lax.scan` is a higher-order function that applies a function repeatedly over the leading axis of arrays, threading an accumulator (carry) through the computation. It's similar to a functional fold/reduce but also collects outputs at each step.

```typescript
const [finalCarry, stackedOutputs] = await lax.scan(f, initCarry, xs);
```

Where:
- `f: (carry, x) => [newCarry, y]` - the step function
- `initCarry` - initial carry state
- `xs` - input arrays to iterate over (first axis is the iteration axis)
- `finalCarry` - carry state after all iterations
- `stackedOutputs` - all `y` outputs stacked along a new leading axis

## Current Status (January 2026)

### Completed Milestones

| Milestone | Status | Impact |
|-----------|--------|--------|
| **Primitive.Scan implementation** | ✅ | Full JAX-compatible scan semantics |
| **JIT + scan integration** | ✅ | Compiled body execution via `bodyProgram.execute()` |
| **WASM instance caching** | ✅ | ~2x speedup for jit(scan): 4.4s → 2.3s |
| **Browser pending ops fix** | ✅ | Fixed const Array realization in jit+scan |
| **PyTree carry/outputs** | ✅ | Nested structures supported |

### Test Environment

**Important**: All tests run on **CPU backend** in a headless Chromium browser.

- WebGPU is not available in the CI/test environment (`requestAdapter()` returns null)
- The "chromium" in vitest config refers to the browser runtime, not the compute backend
- jit+scan has been tested on: **CPU** ✅, **WASM** ✅
- WebGPU backend: **untested** (requires GPU hardware or Dawn/SwiftShader emulator)

### Performance Summary (CPU backend)

| Mode | Time (niledemo, n=100) | Overhead vs for-loop |
|------|------------------------|---------------------|
| Manual for-loop | ~1.6s | baseline |
| jit(scan) | ~2.3s | +50% |
| lax.scan (eager) | ~2.9s | +93% |

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ lax.scan(f, init, xs)                                       │
│   ↓                                                         │
│ Trace f → bodyJaxpr (once)                                  │
│   ↓                                                         │
│ Primitive.Scan(jaxpr, numCarry, numConsts, length)          │
│   ↓                                                         │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ scanRunner (JS loop)                                    │ │
│ │   for i in 0..length:                                   │ │
│ │     xSlice = xs[i] via ShapeTracker (zero-copy)        │ │
│ │     [carry, y] = bodyProgram.execute(carry, xSlice)    │ │
│ │     ys.push(y)                                         │ │
│ │   return [carry, stack(ys)]                            │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### Zero-Copy Views via `ref.#reshape`

Creating slices doesn't copy data. An Array is a lightweight wrapper (~100 bytes) containing:
- A reference to the underlying Slot (buffer handle)
- A ShapeTracker (shape, strides, offset metadata)

```
Memory layout:
┌─────────────────────────────────┐
│  Slot 123: Float32 buffer       │  ← Actual data (shared)
└─────────────────────────────────┘
         ▲           ▲
         │           │
┌────────┴──┐   ┌────┴──────┐
│ Array x   │   │ Array view│   ← Different ShapeTrackers
│ slot: 123 │   │ slot: 123 │     Same underlying buffer
└───────────┘   └───────────┘
```

The pattern `x.ref.#reshape(newSt)` creates a non-consuming view:
1. `x.ref` increments refcount, returns `x`
2. `#reshape(newSt)` creates new Array wrapper with different ShapeTracker
3. Original survives; both share the same buffer

### Carry vs Output Memory Strategy

| Component | Memory Pattern | Size | Purpose |
|-----------|---------------|------|----------|
| **Carry** | Reused per iteration | O(carry_size) | Loop state |
| **Outputs** | Accumulated linearly | O(N × output_size) | For stacking & gradients |

This separation enables efficient memory reuse for carry while streaming outputs.

### ShapeTracker-Based Execution

Operations read through the ShapeTracker without copying:
- Element-wise ops use `AluExp.globalView(dtype, gid, shapeTracker, indices)`
- Realization (copy) only occurs when data is read to JS or passed to routines requiring contiguity

---

## Usage Examples

### Basic Cumulative Sum

```typescript
import { lax, numpy as np, DType } from '@jax-js/jax';

const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
  const sum = np.add(carry, x);
  return [sum, sum.ref];
};

const init = np.zeros([1], { dtype: DType.Float32 });
const xs = np.array([1, 2, 3, 4, 5], { dtype: DType.Float32 });

const [total, cumsum] = await lax.scan(step, init, xs);
// total = [15], cumsum = [1, 3, 6, 10, 15]
```

### PyTree Carry

```typescript
const step = (carry, x) => {
  const { count, sum } = carry;
  const newSum = np.add(sum, x);
  const newCount = np.add(count, np.ones_like(count));
  return [{ count: newCount, sum: newSum }, newSum.ref];
};

const [finalCarry, outputs] = await lax.scan(step, { count, sum }, xs);
```

### Reference Counting in Scan

When values are used in both carry and output, use `.ref`:

```typescript
// CORRECT - use .ref for values used multiple times
const [newCarry, y] = lax.scan((carry, x) => {
  const newSum = carry.sum.ref.add(x.ref);
  return [{ sum: newSum.ref }, newSum];  // .ref keeps it alive
}, init, xs);
```

---

## Implementation Details

### Files Changed (in jax-js)

- `src/frontend/core.ts` - `Primitive.Scan` enum and params type
- `src/frontend/jaxpr.ts` - Abstract eval rule for type checking
- `src/frontend/array.ts` - Scan impl rule with scanRunner callback
- `src/frontend/jit.ts` - Scan JitStep type with compiled bodyProgram
- `src/library/lax-scan.ts` - Public `lax.scan()` API

### Primitive Parameters

```typescript
[Primitive.Scan]: {
  jaxpr: Jaxpr;      // Traced body function
  numCarry: number;  // Number of carry values
  numConsts: number; // Number of closed-over constants
  length: number;    // Number of iterations
}
```

### Argument Layout

```
Primitive args: [...consts, ...initCarry, ...xs]
Body jaxpr expects: [...consts, ...carry, ...x_slice]
```

---

## Key Lessons Learned

### 1. Pending Operations Must Be Submitted Before Synchronous Reads

When JIT-compiled code creates const Arrays (e.g., `np.array([1.0])` inside a jitted function), their pending write operations must be submitted before any synchronous slot reads. The fix:

1. Call `_realizeSource()` on args to create pending operations
2. Submit all pending BEFORE calling `jp.execute()`
3. Submit body's pending after each iteration so next iteration sees updated values

### 2. Reference Counting at Slot/Array Boundaries

When extracting slots from Arrays for return:
- Increment slot refcount (`backend.incRef(slot)`) before disposing the Array
- Increment pending refcount (`pending.updateRc(+1)`) for each Array referencing it

### 3. WASM Instance Caching is Critical

WebAssembly instance creation is expensive (~334x slower than reuse). Cache instances by module:

```typescript
#instanceCache: WeakMap<WebAssembly.Module, WebAssembly.Instance>;
```

### 4. Sequential vs Associative Scan

- `lax.scan`: Strictly sequential, O(N) depth. Cannot be parallelized.
- `lax.associative_scan`: Parallel, O(log N) depth. Only for associative operators.

Do NOT attempt to parallelize `lax.scan`. For cumsum-like operations, use a separate `Primitive.AssociativeScan`.

### 5. Loop Unrolling Pitfall

Python/JS loops get unrolled during tracing → O(N) graph size. For 1000+ iterations, this causes memory exhaustion. `lax.scan` maintains O(1) graph size by keeping the loop structure as a primitive.

### 6. Gradient Tape = stackedOutputs

The AD system handles gradients by modifying the scan body to output intermediates. The backward pass is just another scan with reversed inputs. No special backend support needed—if forward scan handles `stackedOutputs` efficiently, backward works automatically.

---

## Future Work: Phase 3 Backend Loop

### The Remaining Overhead

Current overhead (~50% vs manual for-loop) comes from JavaScript orchestration:
1. JS → Backend boundary crossing per iteration
2. ShapeTracker view creation per iteration
3. Array wrapper overhead
4. Refcount management

### Target Architecture

Move the loop into the backend so JavaScript only crosses the boundary twice (start, get results):

```
┌─────────────────────────────────────────────────────────────┐
│ Current: JS loop calling backend per iteration              │
│   for (i < n) { bodyProgram.execute() }  // N crossings     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Target: Backend-native loop                                 │
│   wasmScanLoop(slots, length)  // 2 crossings total        │
└─────────────────────────────────────────────────────────────┘
```

### Backend-Specific Strategies

| Backend | Strategy | Rationale |
|---------|----------|-----------|
| **CPU** | Keep JS loop | No boundary crossing overhead |
| **WASM** | WASM loop function | Supports `call` for body kernel |
| **WebGPU** | Shader inlining | WGSL has no function pointers; inline body code |

### Design Decisions

1. **Force contiguous xs**: Copy non-contiguous inputs before the backend loop. Simple pointer arithmetic in tight loops beats complex stride calculations.

2. **Ring buffer for carry**: Size 2 (current + previous). Reused across iterations.

3. **Linear memory for outputs**: Stream to pre-allocated buffer. This IS the gradient tape.

---

## Tests

### jax-js (`test/lax-scan.test.ts`)

**Backend**: CPU (in headless Chromium browser)

- Basic cumulative operations
- Multiple carry/output values
- Empty sequences
- PyTree inputs/outputs
- Reference counting correctness
- JIT+scan integration

### dlm-js (`tests/niledemo.test.ts`)

**Backend**: WASM

- Real-world DLM (Dynamic Linear Model) computation
- Validates for-loop, lax.scan, and jit(scan) modes
- Compares against MATLAB reference output

### WebGPU Testing Status

WebGPU backend remains **untested** for jit+scan due to environment limitations:
- Headless Chromium in WSL2/Linux lacks GPU adapter
- `navigator.gpu.requestAdapter()` returns `null`
- Requires hardware GPU or Dawn/SwiftShader emulator for testing
