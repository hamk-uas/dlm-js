/**
 * Repro: WebGPU performance regression on block-map branch.
 *
 * Measures WebGPU dlmFit (assoc, f32) at N=100, 3200, 12800, 25600.
 * Prints observed timings alongside baselines from the last-known-good
 * version (ultimate-architecture-plan, d08dd54).
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

// ── Baselines from ultimate-architecture-plan (d08dd54) ─────────────────────

const baselines: Record<number, number> = {
  100:   541,
  3200:  615,
  12800: 738,
  25600: 899,
};

const TEST_N = [100, 3200, 12800, 25600];

// ── Init WebGPU ─────────────────────────────────────────────────────────────

await init("webgpu");
defaultDevice("webgpu");

console.log("WebGPU dlmFit (Nile order=1, m=2, algorithm:'assoc', f32, cold single-call)\n");
console.log("  N       | baseline (d08dd54) | observed | slowdown");
console.log("  --------|--------------------|---------|---------");

let anyRegression = false;

for (const n of TEST_N) {
  const y = makeY(n);
  const t0 = performance.now();
  const r = await dlmFit(y, { obsStd: s, processStd: w, dtype: 'f32' as DlmDtype, ...options });
  const elapsed = performance.now() - t0;
  r[Symbol.dispose]?.();

  const base = baselines[n];
  const ratio = elapsed / base;
  const flag = ratio > 1.5 ? " <<<" : "";
  console.log(
    `  ${String(n).padStart(7)} | ${String(base).padStart(7)} ms       | ${elapsed.toFixed(0).padStart(7)} ms | ${ratio.toFixed(2)}×${flag}`
  );
  if (ratio > 1.5) anyRegression = true;
}

console.log();
if (anyRegression) {
  console.log("REGRESSION DETECTED: one or more N values are >1.5× slower than baseline.");
  Deno.exit(1);
} else {
  console.log("No significant regression detected.");
  Deno.exit(0);
}
