/**
 * Reproducer: WASM allocator accumulates memory across jit() calls with
 * varying input shapes — eventually hits 4 GB limit
 *
 * Key finding: N=819200 works fine in isolation (Test 1, Test 2).
 * But after ~33 prior dlmFit calls at ascending N values (each with proper
 * dispose()), the WASM allocator cannot grow to accommodate even N=204800.
 *
 * This demonstrates that dispose() on np.Array outputs does NOT free the
 * underlying WASM allocator memory. Each new input shape likely creates a
 * new JIT program entry, and the compiled workspace allocations accumulate.
 *
 * Expected memory per call (N=819200, m=2, f64): ~92 MB
 * WASM maximum: 4 GB (65536 × 64 KB pages)
 * Observed: OOM after ~33 calls totalling ~120 MB of live output data
 *
 * Run with: npx tsx issues/repro-wasm-oom-large-n.ts
 */

import { defaultDevice, init } from "../node_modules/@hamk-uas/jax-js-nonconsuming/dist/index.js";
import { dlmFit } from "../src/index.ts";

await init("wasm");
defaultDevice("wasm");

// ── Step 1: Single call at N=819200 (no prior allocations) ─────────────────

const N = 819_200;
const m = 2;
const obsStd = 122.9;
const processStd = [38.33, 0]; // Nile order=1

// Tile 100-point Nile data to N
const baseY = Array.from({ length: 100 }, (_, i) => 800 + 200 * Math.sin(i * 0.1));
const y = Array.from({ length: N }, (_, i) => baseY[i % 100]);

console.log(`\n=== WASM OOM reproducer ===`);
console.log(`N=${N.toLocaleString()}, m=${m}, dtype=f64`);
console.log(`Expected output memory: ~${(N * (1 + 1 + m + m * m + m + m * m) * 8 / 1e6).toFixed(0)} MB`);
console.log(`WASM limit: 4096 MB\n`);

// Test 1: Cold start — single dlmFit call, no prior WASM allocations
console.log("Test 1: Single cold dlmFit call at N=819200...");
try {
  const r1 = await dlmFit(y, { obsStd, processStd, dtype: "f64", order: 1 });
  console.log(`  ✓ Success (yhat length: ${r1.yhat.length})`);
  r1[Symbol.dispose]?.();
} catch (e) {
  console.log(`  ✗ FAILED: ${(e as Error).message?.split("\n")[0]}`);
  console.log("  → OOM on first call — not an accumulation issue.");
  process.exit(1);
}

// Test 2: Call at N=409600 first, then N=819200
// (mirrors bench-scaling.ts which benchmarks ascending N values)
console.log("\nTest 2: N=409600 (warmup+timed) then N=819200...");
const y409k = Array.from({ length: 409_600 }, (_, i) => baseY[i % 100]);

for (let i = 0; i < 3; i++) {
  const r = await dlmFit(y409k, { obsStd, processStd, dtype: "f64", order: 1 });
  r[Symbol.dispose]?.();
}
console.log("  N=409600: 3 calls completed.");

try {
  const r2 = await dlmFit(y, { obsStd, processStd, dtype: "f64", order: 1 });
  console.log(`  ✓ N=819200 success after N=409600 warmup (yhat length: ${r2.yhat.length})`);
  r2[Symbol.dispose]?.();
} catch (e) {
  console.log(`  ✗ N=819200 FAILED after N=409600: ${(e as Error).message?.split("\n")[0]}`);
  console.log("  → Accumulated WASM allocator state from prior calls causes OOM.");
  console.log("  → dispose() on results doesn't free WASM allocator memory.");
  process.exit(1);
}

// Test 3: Ascending N sequence (like bench-scaling.ts)
console.log("\nTest 3: Full ascending sequence 100 → 819200...");
const Ns = [100, 200, 400, 800, 1_600, 3_200, 6_400, 12_800, 25_600, 51_200, 102_400, 204_800, 409_600, 819_200];
for (const n of Ns) {
  const yn = Array.from({ length: n }, (_, i) => baseY[i % 100]);
  try {
    // 2 warmup + 1 timed (like bench-scaling)
    for (let i = 0; i < 3; i++) {
      const r = await dlmFit(yn, { obsStd, processStd, dtype: "f64", order: 1 });
      r[Symbol.dispose]?.();
    }
    console.log(`  N=${n.toLocaleString().padStart(7)}: ✓`);
  } catch (e) {
    console.log(`  N=${n.toLocaleString().padStart(7)}: ✗ OOM — ${(e as Error).message?.split("\n")[0]}`);
    console.log(`  → Failed after ${Ns.indexOf(n)} prior N values × 3 calls = ${Ns.indexOf(n) * 3} prior dlmFit calls`);
    break;
  }
}

console.log("\nDone.");
