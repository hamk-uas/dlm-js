# Reverse `lax.associativeScan` stride-boundary step-change on WebGPU/Float32

**Status:** 🔴 Open  
**Affected version:** `cd8d47a` (also present in earlier versions — likely since initial WebGPU associativeScan implementation)  
**Severity:** High — produces visually obvious artifacts in smoothed covariance output

## Summary

`lax.associativeScan(..., { reverse: true })` on WebGPU/Float32 produces a **step-change discontinuity** in the composed output at exactly index `N − 32`, where N is the sequence length. The same code on WASM (both f32 and f64) produces smooth, correct output.

The bug manifests as a sudden jump in the backward smoother's covariance matrix at the Kogge-Stone stride-32 boundary in the parallel prefix tree. The smoothed state means (`b` component) are approximately correct — only the covariance (`S` component) is visibly affected.

## Root cause hypothesis

The Kogge-Stone parallel prefix scan on WebGPU dispatches one kernel per round (⌈log₂N⌉ + 1 rounds). For `reverse: true`, the implementation reverses the input, runs a forward scan, and reverses the output (or equivalent). At the stride-32 boundary (round 5), there appears to be a **floating-point non-associativity artifact** that is amplified by the 2×2 matrix–matrix products in the compose function.

Specifically, the compose function for the backward smoother is:
```
compose(a, b) = {
  A: b.A @ a.A,           // [n, m, m] batched matmul
  b: b.A @ a.b + b.b,     // [n, m, 1] matmul + add
  S: b.A @ a.S @ b.A' + b.S  // [n, m, m] triple product + add
}
```

The `S` (covariance) component involves `b.A @ a.S @ b.A'` — a triple matrix product where floating-point rounding in the intermediate product differs depending on which pair of elements were composed first. At stride boundaries, the composition tree switches from one sub-tree to another, and the accumulated rounding error differs discontinuously.

Key observation: **the discontinuity always appears at index `N − 32`**, regardless of:
- The noise parameters (tested ratios from 1:1 to 1000:1)
- The sequence length (tested N = 16, 32, 50, 64, 70, 80, 100)
- The data values

This is consistent with a stride-32 (round 5 = 2^5) boundary in the Kogge-Stone prefix tree.

## Reproduction

```bash
deno run --unstable-webgpu --allow-read --allow-env \
  issues/repro-webgpu-reverse-assocscan-boundary.ts
```

### Expected output
```
wasm/f32:
  ystd[67] = 131.0347
  ystd[68] = 131.0347
  jump = 0.0000
  PASS

webgpu/f32:
  ystd[67] ≈ 131
  ystd[68] ≈ 131
  jump ≈ 0.0
  PASS
```

### Actual output
```
wasm/f32:
  ystd[67] = 131.0347
  ystd[68] = 131.0347
  jump = 0.0000
  PASS

webgpu/f32:
  ystd[67] = 131.1224
  ystd[68] = 139.7128
  jump = 8.5905
  FAIL
BUG CONFIRMED: reverse associativeScan has stride-32 boundary artifact on WebGPU/f32
```

### Pattern across sequence lengths

| N   | Jump index | Jump index = N−32? | Jump magnitude |
|-----|------------|---------------------|----------------|
| 100 | 68         | ✓                   | ~8.6           |
| 80  | 48         | ✓                   | ~10.7          |
| 70  | 38         | ✓                   | ~10.3          |
| 64  | 32         | ✓                   | ~10.8          |
| 50  | 18         | ✓                   | ~10.8          |

## Impact on dlm-js

- **1 file affected**: `src/index.ts` — the backward RTS smoother's reverse `lax.associativeScan` call.
- **Visual artifacts**: The Nile MLE WebGPU animation shows confidence intervals that jump at year 1939 (= index 68 = N−32 for N=100). The animation is unusable for demonstration purposes.
- **Smoothed covariance**: C_smooth[0,0] steps from ~2522 to ~4848 at the boundary (93% jump). Smoothed state means are approximately correct — the covariance/uncertainty estimate is the primary casualty.
- **Forward scan unaffected**: The forward `lax.associativeScan` (5-tuple forward filter) does not exhibit this issue — filtered covariance is smooth. Only the reverse scan is affected.
- **Workaround tested**: Padding backward elements to the next power of 2 ≥ 2N with identity elements `{A=I, b=0, S=0}` pushes all stride boundaries outside the data region and eliminates the artifact. This workaround was verified but not committed — the fix should be in the scan implementation itself.

## Suggested fix

The root cause is likely in how the reverse scan handles stride boundaries in the Kogge-Stone tree. Two possible approaches:

1. **Padding in the scan implementation**: Internally pad the input to the next power of 2 before scanning, then slice the result. This ensures stride boundaries only fall on padding elements. The identity element can be inferred from the first compose call or explicitly provided.

2. **Double-precision accumulation for intermediate products**: Use f64 intermediate values for the matmul/einsum operations within the compose function, even when inputs are f32. This reduces the rounding discontinuity at stride boundaries (though it may not eliminate it entirely).

3. **Kahan summation or compensated composition**: Track rounding error in the `S` component across composition steps. This is more complex but would fix the root cause for all data types.

Option 1 is the simplest and was verified to work as a dlm-js-side workaround.
