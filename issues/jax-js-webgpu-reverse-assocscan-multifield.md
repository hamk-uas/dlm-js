# WebGPU reverse associativeScan corruption with multi-field compose — regression from 66670f4

**Status:** 🔴 Open
**Affected version:** `66670f46157e7815bdd4e3e5f26f3aaefa3fd6d6` (post-v0.8.1)
**Severity:** High — produces silently wrong Kalman smoother output on WebGPU
**Previously fixed in:** `8501de9` (bodyHasReductions guard → mapOverBlocks fallback)
**Regression introduced by:** `66670f4` (per-element reduction codegen, removed bodyHasReductions guard)

## Summary

`lax.associativeScan(fn, elems, { reverse: true })` on WebGPU produces corrupted
results when the compose function has **3 output fields** containing **reduction
operations** (einsum with inner sums) and the state dimension is **m ≥ 5**.

This is a regression of the same bug class fixed in `8501de9`. That fix added a
`bodyHasReductions` guard that fell back to `mapOverBlocks` when the compose body
contained reduction kernels. Commit `66670f4` removed that guard, adding "proper
per-element reduction & multi-element codegen in fused block_map shader" with three
codegen paths (workgroup tree / per-element reduction / elementwise multi-element).
The per-element reduction codegen path still produces wrong results for multi-field
reverse scans with m ≥ 5.

## Root cause hypothesis

The `66670f4` per-element reduction codegen path handles bodies with reduction
kernels (matmul, einsum with inner sum) by switching from the workgroup-tree
approach to a per-element reduction strategy. However, with **3 output fields**
each involving einsums over m×m matrices (m ≥ 5), the fused shader likely:

1. Has incorrect buffer aliasing between output fields — with 3 fields ({A, b, S}),
   earlier writes in the compose may corrupt reads for later fields within the
   same Kogge-Stone round, OR
2. The per-element reduction unrolling miscalculates offsets for the 3rd output
   field's einsum operands when kernel.size > 1, OR
3. The `var<private>` body intermediates (fix from `2554290` for the 256-thread
   `var<workgroup>` race) interact incorrectly with the per-element reduction
   codegen, causing stale intermediate values to leak across elements.

Evidence:
- **2-field compose** (1 matmul + 1 add) works perfectly at all (n, m) — the
  per-element reduction path works for 2 fields.
- **3-field compose** (3 einsum + 2 add) fails for all n when m=5, including
  power-of-2 n (128). The common denominator is **m ≥ 5 + 3 output fields +
  reduction ops**.
- m=2 with 3-field compose passes (smaller per-element work / fewer GPU ops per
  reduction).
- WASM backend (same algorithm, same compose body) is correct for all cases.

## Corruption pattern

For reverse assocScan with the 3-field backward compose (n, m=5):

| n   | Corrupted elements | Corrupted timesteps | Pattern |
|-----|-------------------|--------------------|-|
| 120 | 278/600           | 0..55 (56 ts)       | boundary = n − 2^⌊log₂n⌋ = 56 |
| 128 | 319/640           | 0..~63             | entire sequence (2^7 = 128) |
| 65  | 4/325             | t=0 only (1 ts)     | boundary = 1 |
| 200 | 677/1000          | 0..~135            | extends past boundary |

Error magnitudes: 26–53 absolute (random-looking but deterministic). Clean
indices have ε < 0.00001 (exact to float32 precision except for associativity
rounding).

## Reproduction

### Prerequisites
- Deno with `--unstable-webgpu`
- A WebGPU-capable GPU (tested: NVIDIA RTX 4070)
- `DISPLAY=:1` if running headless

### Run

```bash
DISPLAY=:1 deno run --unstable-webgpu --allow-read --allow-write --allow-env --allow-run \
  issues/repro-webgpu-reverse-assocscan-multifield.ts
```

Tests 2-field compose (control, passes) and 3-field compose at various (n, m).

**Expected output (current, broken):**
```
── Part 1: 2-field compose (control) ──
n=120, m=5: ✅ PASS
n=100, m=2: ✅ PASS

── Part 2: 3-field backward compose (failing cases) ──
n=120, m=5: ❌ FAIL  (278/600 corrupted, max |Δ|=45.86)
n=100, m=2: ✅ PASS
n=128, m=5: ❌ FAIL  (319/640 corrupted, max |Δ|=45.86)
n=65,  m=5: ❌ FAIL  (4/325 corrupted, max |Δ|=38.75)
n=200, m=5: ❌ FAIL  (677/1000 corrupted, max |Δ|=52.79)
```

## Impact on dlm-js

- **Backward smoother on WebGPU** (`algorithm: 'assoc'`) is broken for all models
  with m ≥ 5 (Energy/Kaisaniemi/TriGAR: m=5; FullSeasonal: m=13).
- **Nile model (m=2)** is NOT affected.
- MLE on WebGPU with the Energy model produces visually wrong animation frames
  (first ~half of the time series has random-looking smoothed states).
- Energy MLE on WebGPU fails to converge (300 iterations, oscillating around
  lik=443.14 vs scan converging to 443.13 in 19 iterations).

### Workaround locations

- `src/index.ts` line ~1700: `lax.associativeScan(composeBackward, {A, b, S}, {reverse: true})`
- `src/mle.ts`: `makeKalmanLossAssoc` uses 5-tuple forward filter via assocScan

No workaround applied — the correct fix is in the upstream fused block_map shader.
The `bodyHasReductions` guard from `8501de9` was the correct safety net and should
be restored until the per-element reduction codegen is fixed.

## Suggested fix

1. **Restore the `bodyHasReductions` guard** from `8501de9` as an immediate fix.
   Bodies containing reduction kernels should fall back to `mapOverBlocks` until
   the per-element reduction codegen is proven correct for 3+ output fields.

2. **Debug the per-element reduction codegen** for the 3-field case. The 2-field
   compose works, so the bug is specific to the 3rd+ output field's reduction
   handling. A minimal test: `associativeScan(fn, {A,b,S}, {reverse:true})` where
   fn has 3 einsums, at n=120, m=5.

3. **Regression test**: add a test for reverse assocScan with a 3-field compose
   body containing reduction ops (einsum), at m=5, n=120.
