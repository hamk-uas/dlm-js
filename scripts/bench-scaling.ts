/**
 * Backend scaling benchmark — WASM/f64 at all N, WebGPU/f32 at all N.
 *
 * Measures `dlmFit` (Nile order=1, m=2, data tiled) at exponentially
 * increasing N to characterise:
 *   - The WASM fixed-overhead plateau and its inflection point.
 *   - The WebGPU O(log n) scaling from associativeScan (both forward and
 *     backward passes use associativeScan since the Särkkä & García-Fernández
 *     2020 parallel smoother was implemented).
 *
 * Must be run with Deno (WebGPU requires --unstable-webgpu):
 *   pnpm run bench:scaling
 *
 * Writes timing data to assets/timings/bench-scaling.json, then invokes
 * `pnpm run update:timings` to patch <!-- timing:scale:... --> markers in .md files.
 *
 * Output: assets/timings/bench-scaling.json
 */

import { defaultDevice, init } from "../node_modules/@hamk-uas/jax-js-nonconsuming/dist/index.js";
import { dlmFit } from "../src/index.ts";
import type { DlmDtype } from "../src/types.ts";
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const sidecarDir = resolve(root, "assets/timings");

// ── Config ─────────────────────────────────────────────────────────────────

/** All N values measured for WASM/f64. */
const N_ALL: number[] = [100, 200, 400, 800, 1_600, 3_200, 6_400, 12_800, 25_600, 51_200, 102_400, 204_800, 409_600, 819_200];

/** N values also measured for WebGPU/f32.
 *  Both forward and backward passes use associativeScan (O(log n) depth),
 *  so scaling should be sub-linear. Measured at all N values. */
const N_GPU: number[] = [100, 200, 400, 800, 1_600, 3_200, 6_400, 12_800, 25_600, 51_200, 102_400, 204_800, 409_600, 819_200];

const WARMUP = 2;
const RUNS   = 4;

// ── Load Nile data ──────────────────────────────────────────────────────────

const nileIn  = JSON.parse(readFileSync(resolve(root, "tests/niledemo-in.json"), "utf-8"));
const baseY: number[] = nileIn.y;   // 100 points
const s: number       = nileIn.s;
const w: number[]     = nileIn.w;
const options = { order: 1 };       // m=2

function makeY(n: number): number[] {
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i++) out[i] = baseY[i % baseY.length];
  return out;
}

function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const m = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[m] : (sorted[m - 1] + sorted[m]) / 2;
}

async function timedMedian(n: number, dtype: DlmDtype): Promise<number> {
  const y = makeY(n);

  // First run (JIT compilation)
  const r0 = await dlmFit(y, { obsStd: s, processStd: w, dtype, ...options });
  r0[Symbol.dispose]?.();

  // Warmup runs (discard — let JIT stabilize)
  for (let i = 1; i < WARMUP; i++) {
    const r = await dlmFit(y, { obsStd: s, processStd: w, dtype, ...options });
    r[Symbol.dispose]?.();
  }

  // Timed runs
  const times: number[] = [];
  for (let i = 0; i < RUNS; i++) {
    const t1 = performance.now();
    const r  = await dlmFit(y, { obsStd: s, processStd: w, dtype, ...options });
    times.push(performance.now() - t1);
    r[Symbol.dispose]?.();
  }

  return median(times);
}

// ── Init both backends ─────────────────────────────────────────────────────

await init("wasm");
await init("webgpu");

// ── Run benchmarks ─────────────────────────────────────────────────────────

console.log("\n=== Backend scaling benchmark (WASM/f64 all N · WebGPU/f32 small N) ===\n");
console.log(`Model: Nile order=1, m=2, data tiled. Warmup=${WARMUP}, Runs=${RUNS}, median.\n`);

const colW = [10, 16, 14, 18, 10];
const hdr = [
  "N".padStart(colW[0]),
  "wasm/f64 (ms)".padStart(colW[1]),
  "µs/step".padStart(colW[2]),
  "webgpu/f32 (ms)".padStart(colW[3]),
  "ratio".padStart(colW[4]),
].join("  ");
console.log(hdr);
console.log("─".repeat(hdr.length));

const sidecar: Record<string, number> = {};

for (const n of N_ALL) {
  // ── WASM / f64 ──────────────────────────────────────────────────────────
  defaultDevice("wasm");
  const wasmMs = await timedMedian(n, 'f64');
  sidecar[`wasm_f64_n${n}`] = wasmMs;

  const usPerStep = (wasmMs / n) * 1000;

  // ── WebGPU / f32 (small N only) ─────────────────────────────────────────
  let gpuMs: number | null = null;
  if (N_GPU.includes(n)) {
    defaultDevice("webgpu");
    gpuMs = await timedMedian(n, 'f32');
    sidecar[`webgpu_f32_n${n}`] = gpuMs;
  }

  // ── Print row ────────────────────────────────────────────────────────────
  const gpuCell   = gpuMs !== null ? gpuMs.toFixed(1)                        : "—";
  const ratioCell = gpuMs !== null ? (gpuMs / wasmMs).toFixed(1) + "×"      : "—";
  const cells = [
    n.toLocaleString("en-US").padStart(colW[0]),
    wasmMs.toFixed(1).padStart(colW[1]),
    usPerStep.toFixed(n >= 3_200 ? 1 : 0).padStart(colW[2]),
    gpuCell.padStart(colW[3]),
    ratioCell.padStart(colW[4]),
  ];
  console.log(cells.join("  "));
}

console.log("\nDone.");

// ── Write sidecar ──────────────────────────────────────────────────────────

if (!existsSync(sidecarDir)) mkdirSync(sidecarDir, { recursive: true });
const outPath = resolve(sidecarDir, "bench-scaling.json");
writeFileSync(outPath, JSON.stringify(sidecar, null, 2) + "\n");
console.log(`\nWrote ${outPath}`);

// ── Run update:timings via subprocess ──────────────────────────────────────

console.log("\nRunning pnpm run update:timings...");
const proc = new Deno.Command("pnpm", {
  args: ["run", "update:timings"],
  cwd: root,
  stdout: "inherit",
  stderr: "inherit",
});
const { code } = await proc.output();
if (code !== 0) {
  console.error("update:timings failed with exit code", code);
  Deno.exit(1);
}
