# WebGPU reverse associativeScan corruption with multi-field compose (m ≥ 5)

**Status:** 🔴 Open  
**Affected version:** `4fd9f8d55100296ea66272af567b59b0ebbf1d4d`  
**Severity:** High — produces silently wrong Kalman smoother output on WebGPU  

## Summary

`lax.associativeScan(fn, elems, { reverse: true })` on WebGPU produces incorrect
results when the compose function (`fn`) has **3 output fields** and the state
dimension is **m ≥ 5**. The corruption is systematic: large-magnitude errors
(30–90 absolute) in the computed prefix/suffix scan output.

The bug renders the backward RTS smoother of `dlmSmo` unusable on WebGPU for
models with m ≥ 5 (e.g., Energy model: order=1 + harmonics=1 + AR(1) → m=5).

## Root cause hypothesis

The Phase 4 block_map fusion (commit `4fd9f8d`) replaces per-block
`mapOverBlocks` iterations with a single fused kernel dispatch. When the compose
body produces **multiple output fields** and involves **reduction operations**
(einsum with inner sums), the fused shader may:

1. Miscalculate `gridOffset` for multi-field writes, causing cross-field data
   corruption within a single Kogge-Stone round, OR
2. Have incorrect `pointInputs[]` indexing for the 3rd+ field, reading stale
   input data from a prior round, OR
3. Produce a race condition on shared intermediates when the per-element work
   (3 einsums + 2 adds) exceeds the per-block dispatch assumptions.

Evidence supporting Phase 4 as the culprit:
- **2-field compose** (1 matmul + 1 add) works for n ≤ 128; only breaks at n=129
  (1 element, the "overflow" past 2^7).
- **3-field compose** (3 einsum + 2 add, matching the backward smoother) breaks
  for **all n** when m=5, including power-of-2 n. This is qualitatively different.
- m=2 with 3-field compose still passes (smaller per-element work / fewer GPU ops).
- WASM backend (same algorithm, same compose body) is correct for all cases.

## Reproduction

### Prerequisites
- Deno with `--unstable-webgpu`
- A WebGPU-capable GPU (tested: NVIDIA RTX 4070)
- `DISPLAY=:1` if running headless

### Repro 1: Real-world (Energy model via dlmFit)

```bash
DISPLAY=:1 deno run --unstable-webgpu --allow-read --allow-write --allow-env --allow-run \
  issues/repro-webgpu-energy-backward-smoother.ts
```

Runs `dlmFit` with the Energy model (n=120, m=5) on both WASM and WebGPU.
Compares the combined smoothed signal F·x_smooth.

**Expected output:**
```
WebGPU assoc max error: 89.45 at index 44
WASM assoc max error:   0.00000000
WASM f32 max error:     0.0001

Corruption boundary: index 56 = n(120) - 64 = n - 2^⌊log₂n⌋

❌ FAIL — WebGPU backward smoother produces corrupted smoothed states
   for indices 0..55 (first n - 2^⌊log₂n⌋ elements).
   WASM assoc is fine → bug is in WebGPU reverse associativeScan.
```

### Repro 2: Isolated reverse assocScan (no dlm-js dependency)

```bash
DISPLAY=:1 deno run --unstable-webgpu --allow-read --allow-write --allow-env --allow-run \
  issues/repro-webgpu-reverse-assocscan-stride.ts
```

Tests 2-field and 3-field compose functions at various (n, m) combinations.

**Key results:**

| Compose body | n | m | Result | Corrupted |
|---|---|---|---|---|
| 2-field (matmul+add) | 120 | 5 | ✅ | 0/600 |
| 2-field | 128 | 5 | ✅ | 0/640 |
| 2-field | 129 | 5 | ❌ | 4/645 (t=0 only) |
| 3-field (3 einsum + 2 add) | 120 | 5 | ❌ | 280/600 |
| 3-field | 128 | 5 | ❌ | 319/640 |
| 3-field | 65 | 5 | ❌ | 5/325 (t=0 only) |
| 3-field | 200 | 5 | ❌ | 680/1000 |
| 3-field | 100 | 2 | ✅ | 0/200 |

The 3-field compose fails even at n=128 (power of 2), ruling out a simple
stride-boundary origin. The common denominator is **m=5 + 3 output fields**.

## Impact on dlm-js

- **Backward smoother on WebGPU** (`algorithm: 'assoc'`) is broken for all models
  with m ≥ 5 (Energy/Kaisaniemi/TriGAR: m=5; FullSeasonal: m=13).
- The **forward filter** may also be affected in the sqrt-assoc path (5-field
  compose with QR/triangular-solve), though this isn't exercised on WebGPU+f32.
- **Nile model (m=2)** is NOT affected — WebGPU smoothed states are correct.
- MLE on WebGPU with the Energy model produces visually wrong animation frames
  (first ~half of the time series has random-looking smoothed states).

### Workaround locations

- `src/index.ts` line ~1700: `lax.associativeScan(composeBackward, {A, b, S}, {reverse: true})`
- `src/index.ts` line ~1138: `lax.associativeScan(composeSqrtForward, {A, b, U, eta, Z})`
- `src/mle.ts`: `makeKalmanLossAssoc` uses 5-tuple forward filter via assocScan

No workaround applied — the correct fix is in the upstream fused block_map shader.

## Suggested fix

The bug is in the Phase 4 block_map fusion (`4fd9f8d`). Possible approaches:

1. **Validate `pointInputs[]` + `gridOffset` indexing** for compose bodies with
   3+ output fields. If the fused shader assumes a fixed number of outputs per
   block invocation, a 3-field compose would overflow that assumption.

2. **Check buffer aliasing** in the fused kernel: with 3 output fields (A, b, S),
   each composed from 2 inputs, there are 6 input reads + 3 output writes per
   element per round. If the input/output buffers overlap (scan in-place update),
   earlier writes may corrupt later reads within the same dispatch.

3. **Regression test**: add a test for reverse assocScan with a 3-field compose
   body containing reduction ops (einsum), at m=5, n=120. The 2-field matmul+add
   test that presumably exists does not catch this because it has too few fields.
