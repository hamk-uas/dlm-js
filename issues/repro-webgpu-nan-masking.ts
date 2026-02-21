/**
 * Minimal reproduction: np.isnan / np.where do not correctly mask NaN values
 * on the WebGPU backend — jax-js-nonconsuming v0.7.8.
 *
 * ─── BACKGROUND ──────────────────────────────────────────────────────────────
 *
 * dlm-js handles gapped (missing) observations by masking NaN positions:
 *
 *   using is_nan  = np.isnan(yi);                        // [1,1] bool
 *   using y_safe  = np.where(is_nan, zero, yi);          // [1,1]: 0 if NaN
 *   using mask    = np.where(is_nan, zero, one);         // [1,1]: 0 if NaN
 *   using innov   = (y_safe - F·x) * mask                // should be 0 at NaN steps
 *   using K       = mask * raw_K                         // should be 0 at NaN steps
 *
 * On CPU / WASM this masking works correctly: NaN observations are treated as
 * missing (prediction-only steps, K=0, innovation=0).
 *
 * On WebGPU the masking silently fails: `np.isnan` returns the wrong boolean
 * (or `np.where` does not honour it), so `y_safe` retains the NaN value.  The
 * NaN then propagates through the innovation, state update, and all downstream
 * computations, giving NaN in every output.
 *
 * The failure affects ALL WebGPU paths (scan, assoc, with or without
 * stabilisation) — the root cause is a single broken primitive.
 *
 * ─── HOW TO RUN ──────────────────────────────────────────────────────────────
 *
 *   deno run --no-check --unstable-webgpu --allow-read --allow-env \
 *     issues/repro-webgpu-nan-masking.ts
 *
 * ─── EXPECTED OUTPUT ─────────────────────────────────────────────────────────
 *
 *   Test 1  np.isnan(NaN)    expected: 1  got: 1  PASS
 *   Test 2  np.isnan(1.0)    expected: 0  got: 0  PASS
 *   Test 3  np.where(nan→0)  expected: 0  got: 0  PASS
 *   Test 4  np.where(ok→y)   expected: 3  got: 3  PASS
 *   Test 5  mask scalar      expected: 0  got: 0  PASS
 *   Test 6  batched isnan [NaN,1,NaN,2] → [1,0,1,0]  PASS
 *   Test 7  dlmFit/WebGPU gapped (order=1, n=10, 3 NaN)  PASS (no NaN in output)
 *
 * ─── ACTUAL OUTPUT (jax-js-nonconsuming v0.7.8 on WebGPU) ───────────────────
 *
 *   === CPU baseline ===
 *     PASS  where(isnan(NaN), 0, NaN) → 0   ...  (all 6 scalar/batch PASS)
 *
 *   === WebGPU: np.isnan + np.where ===
 *     PASS  isnan(NaN) → 1 (via float where)           scalar [1,1] ✓
 *     PASS  where(isnan(NaN), 0, NaN) → 0              scalar [1,1] ✓
 *     ...
 *     FAIL  batched mask [NaN,1,NaN,2] → [0,1,0,1]  got: [1,1,1,1]
 *     FAIL  batched safe [NaN,1,NaN,2] → [0,1,0,2]  got: [NaN,1,NaN,2]
 *
 *   === dlmFit end-to-end (WebGPU, gapped) ===
 *     FAIL  dlmFit WebGPU/f32/scan   gapped: yhat[3]=NaN  anyNaN=true
 *     FAIL  dlmFit WebGPU/f32/assoc  gapped: yhat[3]=NaN  anyNaN=true
 *
 *   Results: 12 passed, 4 failed
 *
 * ─── FINDING ─────────────────────────────────────────────────────────────────
 *
 *   np.isnan works for scalar [1,1] tensors on WebGPU but silently returns
 *   false for ALL elements of a batched [n,1] tensor (n > 1), even when NaN
 *   values are present.  The NaN bit pattern is likely flushed to zero during
 *   the CPU→GPU buffer upload for multi-element tensors.
 *
 * ─── ROOT CAUSE HYPOTHESIS ───────────────────────────────────────────────────
 *
 * WGSL `isnan(x)` behaviour is implementation-defined for non-nan operands, but
 * the handling of actual IEEE 754 NaN payloads may vary across GPU drivers.
 * Some drivers flush NaN to a finite value before the shader executes
 * (e.g. flush-to-zero mode), making `isnan()` return false for what was NaN
 * in CPU memory.  Alternatively, the `select()` intrinsic used to implement
 * `np.where` may evaluate both branches before selecting, propagating NaN even
 * when the mask says "zero".
 *
 * The fix should live in the jax-js-nonconsuming WebGPU backend: either
 *   (a) encode NaN positions as a separate boolean buffer passed alongside y, or
 *   (b) ensure the WGSL isnan/select path round-trips IEEE 754 NaN correctly.
 */

import { numpy as np, defaultDevice, init, DType } from "@hamk-uas/jax-js-nonconsuming";
import { dlmFit } from "../src/index.ts";

const NAN = Number.NaN;
let pass = 0, fail = 0;

function checkSync(label: string, got: number, expected: number) {
  const ok = isNaN(expected) ? isNaN(got) : Math.abs(got - expected) < 1e-5;
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label.padEnd(48)} expected: ${expected}  got: ${got}`);
  ok ? pass++ : fail++;
}
async function checkAsync(label: string, tensor: np.Array, expected: number) {
  // WebGPU tensors need jsAsync() — .js() needs OffscreenCanvas (not in Deno).
  const arr = await tensor.jsAsync() as number[][] | boolean[][];
  const raw = arr[0][0];
  const got = typeof raw === "boolean" ? (raw ? 1 : 0) : (raw as number);
  checkSync(label, got, expected);
}

// ── CPU baseline ──────────────────────────────────────────────────────────

console.log("\n=== CPU baseline ===");
defaultDevice("cpu");

{
  using a    = np.array([[NAN]], { dtype: DType.Float32 });
  using b    = np.array([[1.0]], { dtype: DType.Float32 });
  using zero = np.zeros([1, 1], { dtype: DType.Float32 });
  using one  = np.ones ([1, 1], { dtype: DType.Float32 });

  using is_nan_a = np.isnan(a);
  using is_nan_b = np.isnan(b);
  // Note: np.isnan returns a boolean array; .js() works on it but not .tolist().
  // We test isnan correctness indirectly through np.where outputs.
  using a_safe   = np.where(is_nan_a, zero, a);   // NaN → 0
  using b_safe   = np.where(is_nan_b, zero, b);   // 1.0 → 1.0
  using mask_a   = np.where(is_nan_a, zero, one); // NaN → 0
  using mask_b   = np.where(is_nan_b, zero, one); // 1.0 → 1

  checkSync("where(isnan(NaN),  0, NaN)  → 0",  (a_safe.js() as number[][])[0][0], 0);
  checkSync("where(isnan(1.0),  0, 1.0)  → 1",  (b_safe.js() as number[][])[0][0], 1);
  checkSync("where(isnan(NaN),  0, 1)   → 0",   (mask_a.js() as number[][])[0][0], 0);
  checkSync("where(isnan(1.0),  0, 1)   → 1",   (mask_b.js() as number[][])[0][0], 1);
}

// Batched [NaN, 1.0, NaN, 2.0]
{
  using y4    = np.array([[NAN], [1.0], [NAN], [2.0]], { dtype: DType.Float32 });
  using z4    = np.zeros([4, 1], { dtype: DType.Float32 });
  using o4    = np.ones ([4, 1], { dtype: DType.Float32 });
  using is_n4 = np.isnan(y4);
  using mask4 = np.where(is_n4, z4, o4);
  using safe4 = np.where(is_n4, z4, y4);
  const m4 = (mask4.js() as number[][]).map(r => r[0]);
  const s4 = (safe4.js() as number[][]).map(r => r[0]);
  const mOk = [0,1,0,1].every((e, i) => Math.abs(m4[i] - e) < 1e-5);
  const sOk = [0,1,0,2].every((e, i) => Math.abs(s4[i] - e) < 1e-5);
  console.log(`  ${mOk ? "PASS" : "FAIL"}  batched mask [NaN,1,NaN,2] → [0,1,0,1]  got: [${m4}]`);
  console.log(`  ${sOk ? "PASS" : "FAIL"}  batched safe [NaN,1,NaN,2] → [0,1,0,2]  got: [${s4}]`);
  mOk ? pass++ : fail++;
  sOk ? pass++ : fail++;
}

// ── WebGPU: np.isnan + np.where primitives ───────────────────────────────

console.log("\n=== WebGPU: np.isnan + np.where ===");
await init("webgpu");
defaultDevice("webgpu");

{
  using a    = np.array([[NAN]], { dtype: DType.Float32 });
  using b    = np.array([[1.0]], { dtype: DType.Float32 });
  using zero = np.zeros([1, 1], { dtype: DType.Float32 });
  using one  = np.ones ([1, 1], { dtype: DType.Float32 });

  using is_nan_a      = np.isnan(a);
  using is_nan_b      = np.isnan(b);
  using isnan_float_a = np.where(is_nan_a, one, zero);  // 1 if isnan detected NaN
  using isnan_float_b = np.where(is_nan_b, one, zero);  // 0 if isnan detected non-NaN
  using a_safe        = np.where(is_nan_a, zero, a);
  using b_safe        = np.where(is_nan_b, zero, b);
  using mask_a        = np.where(is_nan_a, zero, one);
  using mask_b        = np.where(is_nan_b, zero, one);

  await checkAsync("isnan(NaN) → 1 (via float where)",   isnan_float_a, 1);
  await checkAsync("isnan(1.0) → 0 (via float where)",   isnan_float_b, 0);
  await checkAsync("where(isnan(NaN),  0, NaN)  → 0",    a_safe,        0);
  await checkAsync("where(isnan(1.0),  0, 1.0)  → 1",    b_safe,        1);
  await checkAsync("where(isnan(NaN),  0, 1)   → 0",     mask_a,        0);
  await checkAsync("where(isnan(1.0),  0, 1)   → 1",     mask_b,        1);
}

// Batched [NaN, 1.0, NaN, 2.0]
{
  using y4    = np.array([[NAN], [1.0], [NAN], [2.0]], { dtype: DType.Float32 });
  using z4    = np.zeros([4, 1], { dtype: DType.Float32 });
  using o4    = np.ones ([4, 1], { dtype: DType.Float32 });
  using is_n4 = np.isnan(y4);
  using mask4 = np.where(is_n4, z4, o4);
  using safe4 = np.where(is_n4, z4, y4);
  const m4 = (await mask4.jsAsync() as number[][]).map(r => r[0]);
  const s4 = (await safe4.jsAsync() as number[][]).map(r => r[0]);
  const mOk = [0,1,0,1].every((e, i) => Math.abs(m4[i] - e) < 1e-5);
  const sOk = [0,1,0,2].every((e, i) => Math.abs(s4[i] - e) < 1e-5);
  console.log(`  ${mOk ? "PASS" : "FAIL"}  batched mask [NaN,1,NaN,2] → [0,1,0,1]  got: [${m4}]`);
  console.log(`  ${sOk ? "PASS" : "FAIL"}  batched safe [NaN,1,NaN,2] → [0,1,0,2]  got: [${s4}]`);
  mOk ? pass++ : fail++;
  sOk ? pass++ : fail++;
}

// ── End-to-end: dlmFit with gapped data on WebGPU ────────────────────────

console.log("\n=== dlmFit end-to-end (WebGPU, gapped) ===");
{
  const y = [1.1, 1.3, 1.2, NAN, 1.4, 1.5, NAN, 1.6, 1.5, NAN];
  for (const algo of ["scan", "assoc"] as const) {
    defaultDevice("webgpu");
    const r = await dlmFit(y, { obsStd: 0.5, processStd: [0.1, 0.1], order: 1, dtype: "f32", algorithm: algo });
    const yhat = r.yhat as number[];
    const anyNaN = yhat.some(v => !isFinite(v));
    console.log(`  ${anyNaN ? "FAIL" : "PASS"}  dlmFit WebGPU/f32/${algo}  gapped: yhat[3]=${yhat[3]?.toFixed(4)}  anyNaN=${anyNaN}`);
    anyNaN ? fail++ : pass++;
    r[Symbol.dispose]?.();
  }
}

console.log(`\n${"─".repeat(60)}`);
console.log(`Results: ${pass} passed, ${fail} failed`);
if (fail > 0) Deno.exit(1);
