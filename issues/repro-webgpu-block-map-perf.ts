/**
 * WebGPU associativeScan dispatch overhead benchmark.
 *
 * Measures WebGPU dlmFit (assoc, f32) vs WASM dlmFit (scan, f64) at
 * increasing N. The GPU/WASM ratio reveals how many dispatches are
 * being wasted — the target is ratio ≤ 2× (near-parity with WASM).
 *
 * Run:
 *   deno run --allow-read --allow-env --allow-net --allow-ffi --allow-sys --unstable-webgpu issues/repro-webgpu-block-map-perf.ts
 */

import { defaultDevice, init } from "../node_modules/@hamk-uas/jax-js-nonconsuming/dist/index.js";
import { dlmFit } from "../src/index.ts";
import type { DlmDtype } from "../src/types.ts";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");

// ── Load Nile data ──────────────────────────────────────────────────────────

const nileIn = JSON.parse(readFileSync(resolve(root, "tests/niledemo-in.json"), "utf-8"));
const baseY: number[] = nileIn.y;   // 100 points
const s: number       = nileIn.s;
const w: number[]     = nileIn.w;
const options = { order: 1 };       // m=2

function makeY(n: number): number[] {
  const y: number[] = [];
  while (y.length < n) y.push(...baseY.slice(0, n - y.length));
  return y;
}

const TEST_N = [100, 3200, 12800, 25600];
const WASM_RUNS = 4;

// ── Init both backends ──────────────────────────────────────────────────────

await init("wasm");
await init("webgpu");

// ── WASM JIT warmup ─────────────────────────────────────────────────────────

defaultDevice("wasm");
const rWarmup = await dlmFit(makeY(100), { obsStd: s, processStd: w, dtype: 'f64' as DlmDtype, ...options });
rWarmup[Symbol.dispose]?.();

// ── Benchmark ───────────────────────────────────────────────────────────────

console.log("dlmFit benchmark: WebGPU/f32/assoc (cold) vs WASM/f64/scan (warm)");
console.log("Model: Nile order=1, m=2. Target: GPU/WASM ratio ≤ 2×.\n");
console.log("  N       | WASM/f64 (warm) | WebGPU/f32 (cold) |  ratio | target?");
console.log("  --------|-----------------|-------------------|--------|--------");

let anyFail = false;

for (const n of TEST_N) {
  const y = makeY(n);

  // WASM warm (median of WASM_RUNS)
  defaultDevice("wasm");
  const wasmTimes: number[] = [];
  for (let i = 0; i < WASM_RUNS; i++) {
    const t0 = performance.now();
    const r = await dlmFit(y, { obsStd: s, processStd: w, dtype: 'f64' as DlmDtype, ...options });
    wasmTimes.push(performance.now() - t0);
    r[Symbol.dispose]?.();
  }
  wasmTimes.sort((a, b) => a - b);
  const wasmMs = wasmTimes[Math.floor(wasmTimes.length / 2)];

  // WebGPU cold (single call including JIT)
  defaultDevice("webgpu");
  const t0 = performance.now();
  const rGpu = await dlmFit(y, { obsStd: s, processStd: w, dtype: 'f32' as DlmDtype, ...options });
  const gpuMs = performance.now() - t0;
  rGpu[Symbol.dispose]?.();

  const ratio = gpuMs / wasmMs;
  const ok = ratio <= 2.0;
  if (!ok) anyFail = true;

  console.log(
    `  ${String(n).padStart(7)} | ${wasmMs.toFixed(0).padStart(8)} ms    | ${gpuMs.toFixed(0).padStart(10)} ms    | ${ratio.toFixed(1).padStart(5)}× | ${ok ? "  ✅" : "  ❌"}`
  );
}

console.log();
if (anyFail) {
  console.log("FAIL: WebGPU/WASM ratio exceeds 2× target — too many dispatches.");
  Deno.exit(1);
} else {
  console.log("PASS: WebGPU within 2× of WASM at all sizes.");
  Deno.exit(0);
}
