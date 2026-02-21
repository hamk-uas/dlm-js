# WebGPU: `np.isnan` returns wrong results for batched tensors — NaN not detected in `[n,1]` shape

**Status**: open  
**Affects**: jax-js-nonconsuming v0.7.8, WebGPU backend, Float32  
**Impact**: `dlmFit` with missing (NaN) observations produces all-NaN output on WebGPU  

---

## Summary

`np.isnan` correctly detects NaN in a scalar-shaped `[1,1]` tensor on WebGPU, but
silently returns **false for every element** when the input has shape `[n,1]` (n > 1)
and some elements are NaN.  As a result, `np.where(is_nan, zero, y)` does not
replace NaN values with zero — the original NaN propagates through all downstream
arithmetic, giving NaN in every output.

The failure affects all WebGPU execution paths (sequential scan, associative scan,
with or without covariance stabilisation).

---

## Reproduction

```
deno run --no-check --unstable-webgpu --allow-read --allow-env \
  issues/repro-webgpu-nan-masking.ts
```

See [`issues/repro-webgpu-nan-masking.ts`](repro-webgpu-nan-masking.ts) for the
complete reproducer.

---

## Observed output

```
=== CPU baseline ===
  PASS  where(isnan(NaN),  0, NaN)  → 0                  expected: 0  got: 0
  PASS  where(isnan(1.0),  0, 1.0)  → 1                  expected: 1  got: 1
  PASS  where(isnan(NaN),  0, 1)   → 0                   expected: 0  got: 0
  PASS  where(isnan(1.0),  0, 1)   → 1                   expected: 1  got: 1
  PASS  batched mask [NaN,1,NaN,2] → [0,1,0,1]  got: [0,1,0,1]
  PASS  batched safe [NaN,1,NaN,2] → [0,1,0,2]  got: [0,1,0,2]

=== WebGPU: np.isnan + np.where ===
  PASS  isnan(NaN) → 1 (via float where)                 expected: 1  got: 1
  PASS  isnan(1.0) → 0 (via float where)                 expected: 0  got: 0
  PASS  where(isnan(NaN),  0, NaN)  → 0                  expected: 0  got: 0
  PASS  where(isnan(1.0),  0, 1.0)  → 1                  expected: 1  got: 1
  PASS  where(isnan(NaN),  0, 1)   → 0                   expected: 0  got: 0
  PASS  where(isnan(1.0),  0, 1)   → 1                   expected: 1  got: 1
  FAIL  batched mask [NaN,1,NaN,2] → [0,1,0,1]  got: [1,1,1,1]   ← isnan missed NaN
  FAIL  batched safe [NaN,1,NaN,2] → [0,1,0,2]  got: [NaN,1,NaN,2] ← NaN not replaced

=== dlmFit end-to-end (WebGPU, gapped) ===
  FAIL  dlmFit WebGPU/f32/scan   gapped: yhat[3]=NaN  anyNaN=true
  FAIL  dlmFit WebGPU/f32/assoc  gapped: yhat[3]=NaN  anyNaN=true

────────────────────────────────────────────────────────────
Results: 12 passed, 4 failed
```

---

## Key observations

| Input shape | `np.isnan(NaN_element)` on WebGPU | Correct? |
|-------------|-----------------------------------|----------|
| `[1,1]`     | true (1.0 via `where→float`)      | ✅       |
| `[4,1]`     | false for ALL elements            | ❌       |

The scalar case is tested via `np.where(np.isnan(x), one, zero)` since boolean
arrays do not expose `.tolist()`.  The batched case is tested the same way — the
mask `np.where(is_n4, zeros, ones)` returns `[1,1,1,1]` for input
`[NaN, 1.0, NaN, 2.0]`, which means `is_n4` was `[false,false,false,false]`.

CPU / WASM both handle the batched case correctly.

---

## How dlm-js is affected

dlm-js handles gapped (missing) observations by masking NaN timesteps:

```typescript
// src/index.ts — sequential scan path (same pattern in assocScan path)
using is_nan = np.isnan(yi);           // [1,1] bool — scalar, works on WebGPU
using y_safe = np.where(is_nan, zero, yi);  // should be 0 at NaN steps
using mask   = np.where(is_nan, zero, one); // should be 0 at NaN steps
// innovation = (y_safe − F·x) · mask  → should be 0 at NaN steps
// K          = mask · raw_K           → should be 0 at NaN steps
```

In the sequential scan path each `yi` is `[1,1]`, so `np.isnan` always sees a
scalar tensor and works.  But in the associative scan path the entire observation
array is processed at once:

```typescript
// src/index.ts — associative scan path
using is_nan  = np.isnan(y_arr);       // [n,1,1] bool — BATCHED, broken on WebGPU
using mask_arr = np.where(is_nan, zero_n11, one_n11);
using y_safe  = np.where(is_nan, zero_n11, y_arr);
```

In this path `y_arr` has shape `[n,1,1]` with n observations.  `np.isnan` fails
for n > 1 on WebGPU, so `is_nan` is all-false, `y_safe` retains the NaN values,
and the innovation `(y_safe − F·x)` propagates NaN to every subsequent state
update.

Even the sequential scan path fails end-to-end on WebGPU (both scan and assoc
rows show ⚠️ NaN in `bench-full.ts`), suggesting the bug may also affect the
sequential path under JIT/shader fusion once the loop is compiled as a kernel.

---

## Root cause hypothesis

`np.isnan` on a batched tensor is lowered to a WGSL `isnan()` call dispatched
over a multi-element buffer.  WGSL `isnan()` is **implementation-defined** for
IEEE 754 NaN operands:

> "Note: GPUs do not always support IEEE 754 NaN values, and some implementations
> may flush NaN to zero before the shader executes." — WebGPU spec / WGSL spec

The most likely scenario: the WebGPU buffer upload path for Float32 **flushes
NaN bits to 0.0** (or another finite value) before the compute shader reads them,
so `isnan()` sees a finite value and returns false.  The flush happens only when
the tensor has multiple elements (perhaps a different upload code path is used for
size > 1 vs. size == 1).

A secondary hypothesis: WGSL `select(false_val, true_val, cond)` (used by
`np.where`) evaluates both branches unconditionally, and the NaN branch poisons
the result regardless of the condition.  This is ruled out by the scalar test
passing — if `where` evaluated both branches, `where(isnan(NaN), 0, NaN)` would
still return NaN even for the scalar case.

---

## Suggested investigation

1. **In the WebGPU upload path**: verify that NaN bit patterns (`0x7FC00000` for
   quiet NaN in Float32) survive the CPU→GPU buffer copy for tensors with n > 1
   elements.
2. **In the WGSL kernel for `isnan`**: print the raw bit pattern of the input with
   `bitcast<u32>(x)` before calling `isnan(x)` to determine whether NaN arrives
   in the shader or was flushed earlier.
3. **Workaround**: instead of relying on `isnan()` in WGSL, encode NaN positions
   as a separate boolean/integer buffer on the CPU side before dispatch, then pass
   that mask to the kernel.  For dlm-js this would mean computing `isnan` on CPU
   and uploading a precomputed `[n]` int32 mask alongside the observation array.

---

## Environment

- jax-js-nonconsuming: v0.7.8  
- Deno: 2.6.7  
- OS: Linux  
- GPU: (run `issues/repro-webgpu-nan-masking.ts` to obtain)  
