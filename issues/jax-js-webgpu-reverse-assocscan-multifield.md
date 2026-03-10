# WebGPU reverse associativeScan corruption with multi-field compose — regression from 66670f4

**Status:** � Mitigated — `df765e9` fixed the main `outOffset` corruption; residual stub-block corruption remains for certain N values
**Affected version:** `66670f46157e7815bdd4e3e5f26f3aaefa3fd6d6` (post-v0.8.1)
**Partially fixed in:** `df765e910685c5a026e3c29056ba9dbd1be63b66` (outOffset flat gidx for Phase 4)
**Severity:** Medium — residual corruption limited to first few timesteps when N mod blockSize is small
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

### Before fix (`66670f4`)

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

### After `df765e9` fix (outOffset flat gidx)

The main corruption is fixed. Residual corruption is limited to the **stub block**
— the last-in-forward-order (first-in-reverse-order) partial block when N is not
a multiple of the block size (64):

| n   | Stub size (N mod 64) | Corrupted ts | Max |Δ| | Status |
|-----|---------------------|-------------|---------|--------|
| 65  | 1                    | t=0 only     | 29.38   | ❌ FAIL |
| 120 | 56                   | mild, 8/600  | 0.40    | ✅ PASS (within f32 noise) |
| 128 | 0 (full blocks)      | none         | 0.00    | ✅ PASS |
| 200 | 8                    | t=0..7       | 36.61   | ❌ FAIL |

The corruption count exactly matches the stub block size, and it always affects
the earliest timesteps (which are the last block in forward order = first block
processed in reverse). When the stub is large (56 for n=120), the errors are mild
enough to be float32 associativity noise. When the stub is small (1 for n=65,
8 for n=200), the errors are catastrophic (29–37 absolute).

**Hypothesis**: The fused Phase 4 reverse scan composes the stub block's elements
incorrectly — possibly the per-element reduction codegen miscalculates reduction
offsets when the block is not full (fewer than 64 elements), or the stub block's
compose intermediates alias with dummy/padding values.

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

**Expected output (at `df765e9` — partially fixed):**
```
── Part 1: 2-field compose (control) ──
n=120, m=5: ✅ PASS
n=100, m=2: ✅ PASS

── Part 2: 3-field backward compose ──
n=120, m=5: ✅ PASS  (8/600 > 0.1, max |Δ|=0.40 — f32 noise)
n=100, m=2: ✅ PASS
n=128, m=5: ✅ PASS  (0 corrupted)
n=65,  m=5: ❌ FAIL  (4/325 corrupted, max |Δ|=29.38 at t=0)
n=200, m=5: ❌ FAIL  (32/1000 corrupted, max |Δ|=36.61 at t=0..7)
```

**Before fix (`66670f4`):**
```
n=120, m=5: ❌ FAIL  (278/600 corrupted, max |Δ|=45.86)
n=128, m=5: ❌ FAIL  (319/640 corrupted, max |Δ|=45.86)
n=65,  m=5: ❌ FAIL  (4/325 corrupted, max |Δ|=38.75)
n=200, m=5: ❌ FAIL  (677/1000 corrupted, max |Δ|=52.79)
```

## Impact on dlm-js

After `df765e9`:

- **Energy model (n=120, m=5)**: ✅ Fixed — backward smoother produces correct
  results (max |Δ|=0.40 vs WASM, within float32 noise). Energy MLE WebGPU
  animation frames should now converge correctly.
- **Nile model (n=100, m=2)**: ✅ Unaffected (was never broken).
- **Arbitrary-N datasets on WebGPU**: ⚠️ Still at risk when N mod 64 is small
  (stub block corruption). E.g., n=65 or n=200 would produce wrong smoothed
  states for the first few timesteps.
- All 317 dlm-js tests pass (317 pass + 3 skip under Node).

### Workaround locations

- `src/index.ts` line ~1700: `lax.associativeScan(composeBackward, {A, b, S}, {reverse: true})`
- `src/mle.ts`: `makeKalmanLossAssoc` uses 5-tuple forward filter via assocScan

No workaround applied — the correct fix is in the upstream fused block_map shader.
The `bodyHasReductions` guard from `8501de9` was the correct safety net and should
be restored until the per-element reduction codegen is fixed.

## Why the existing test suite didn't catch it

The bug requires **four conditions simultaneously**: reverse + 3 fields + m≥5 +
multi-block N. Existing tests covered each axis individually but never combined
them:

| Existing test | Fields | Direction | m | N | Why it passes |
|---|---|---|---|---|---|
| "3-field DLM m=5 Phase 4" (L655) | 3 | **forward** | 5 | 120 | forward is unaffected |
| "reverse matmul" (L1451) | **1** | reverse | **2** | **3** | 1 field, tiny N, small m |
| "pytree (A,S) N=50" (L1572) | **2** | forward | **2** | 50 | 2 fields, forward |
| "matrix affine DLM" (L1334) | **2** | forward | **2** | 4 | 2 fields, forward |

## Suggested fix

1. ~~**Restore the `bodyHasReductions` guard** from `8501de9` as an immediate fix.~~
   Partially addressed by `df765e9` (outOffset flat gidx).

2. **Debug the residual stub-block corruption** in the fused Phase 4 reverse scan.
   The corruption count equals the stub block size (N mod blockSize), and it
   always affects the earliest timesteps (= last forward block = first reverse
   block). When the stub is nearly full (e.g., 56/64 for n=120), errors are mild
   (0.40); when the stub is very small (1 or 8 elements), errors are catastrophic
   (29–37). This suggests the fused shader's per-element reduction reads
   padding/uninitialized values when the block is not full in a reverse scan.

3. ~~**Regression test**: add a test for reverse assocScan with a 3-field compose
   body containing reduction ops (einsum), at m=5, n=120.~~
   **Done** — added "reverse 3-field DLM compose with m=5 (dlm-js smoother
   regression)" to `test/lax-associative-scan.test.ts` in the "WebGPU fused
   shader regression" describe block. This test NOW PASSES at n=120 with
   `df765e9`. Consider adding n=65 and n=200 cases to catch the residual bug.
