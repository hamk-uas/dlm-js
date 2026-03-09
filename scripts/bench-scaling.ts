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
 * Timing methodology:
 *   - WASM: jit() is polymorphic in N — one JIT compilation covers all array
 *     sizes.  A single warmup at N=100 compiles the JIT, then each N is timed
 *     with RUNS warm calls (median reported).
 *   - WebGPU: jit() is NOT polymorphic in N — each new N triggers
 *     recompilation.  Each N is measured as a cold call (includes JIT).
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

/** All N values measured for WASM/f64.
 *  N=819200 re-added after allocator fix in upstream commit 089b0e5.
 *  N=3276800 OOMs even with the 2 GB limit (297f93a) — WASM allocator
 *  overflows its signed 32-bit page-count calculation when a single
 *  intermediate allocation exceeds 2^31 bytes. See
 *  issues/jax-js-wasm-allocator-size-overflow.md. */
const N_ALL: number[] = [100, 200, 400, 800, 1_600, 3_200, 6_400, 12_800, 25_600, 51_200, 102_400, 204_800, 409_600, 819_200, 1_638_400];

/** N values also measured for WebGPU/f32.
 *  Both forward and backward passes use associativeScan (O(log n) depth),
 *  so scaling should be sub-linear. Measured at the same N values as WASM. */
const N_GPU: number[] = [100, 200, 400, 800, 1_600, 3_200, 6_400, 12_800, 25_600, 51_200, 102_400, 204_800, 409_600, 819_200, 1_638_400];

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

const TIMEOUT_MS = 20_000;

/** Race a promise against a timeout; returns Infinity on timeout. */
async function withTimeout(fn: () => Promise<number>): Promise<number> {
  let timer: ReturnType<typeof setTimeout>;
  const timeout = new Promise<number>(resolve => {
    timer = setTimeout(() => resolve(Infinity), TIMEOUT_MS);
  });
  const result = await Promise.race([fn(), timeout]);
  clearTimeout(timer!);
  return result;
}

/** WASM warm median: JIT is polymorphic in N — already compiled once upfront.
 *  Just run RUNS times and take the median. */
async function timedWarm(n: number): Promise<number> {
  return withTimeout(async () => {
    const y = makeY(n);
    const times: number[] = [];
    for (let i = 0; i < RUNS; i++) {
      const t1 = performance.now();
      const r  = await dlmFit(y, { obsStd: s, processStd: w, dtype: 'f64' as DlmDtype, ...options });
      times.push(performance.now() - t1);
      r[Symbol.dispose]?.();
    }
    return median(times);
  });
}

/** WebGPU cold: JIT is NOT polymorphic in N — each N recompiles.
 *  Measure the single first call (includes JIT). */
async function timedCold(n: number): Promise<number> {
  return withTimeout(async () => {
    const y = makeY(n);
    const t1 = performance.now();
    const r  = await dlmFit(y, { obsStd: s, processStd: w, dtype: 'f32' as DlmDtype, ...options });
    const elapsed = performance.now() - t1;
    r[Symbol.dispose]?.();
    return elapsed;
  });
}

// ── Init both backends ─────────────────────────────────────────────────────

await init("wasm");
await init("webgpu");

// ── WASM JIT warmup (polymorphic in N — one compilation covers all sizes) ──

defaultDevice("wasm");
console.log("Compiling WASM JIT (single warmup at N=100)...");
const rWarmup = await dlmFit(makeY(100), { obsStd: s, processStd: w, dtype: 'f64' as DlmDtype, ...options });
rWarmup[Symbol.dispose]?.();

// ── Run benchmarks ─────────────────────────────────────────────────────────

console.log("\n=== Backend scaling benchmark ===");
console.log("  WASM/f64:   warm (JIT polymorphic in N, compiled once)");
console.log(`  WebGPU/f32: cold (JIT recompiles per N)\n`);
console.log(`Model: Nile order=1, m=2, data tiled. WASM: ${RUNS} warm runs, median. WebGPU: single cold call.\n`);

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
let gpuBailed = false;          // once GPU times out, skip all larger N

for (const n of N_ALL) {
  try {
    // ── WASM / f64 (warm — JIT already compiled) ─────────────────────────────
    defaultDevice("wasm");
    const wasmMs = await timedWarm(n);
    sidecar[`wasm_f64_n${n}`] = wasmMs;

    const usPerStep = (wasmMs / n) * 1000;

    // ── WebGPU / f32 (cold — JIT recompiles for this N) ─────────────────────
    let gpuMs: number | null = null;
    if (N_GPU.includes(n) && !gpuBailed) {
      defaultDevice("webgpu");
      gpuMs = await timedCold(n);
      sidecar[`webgpu_f32_n${n}`] = gpuMs;
      if (gpuMs === Infinity) {
        gpuBailed = true;
        console.log("  ⏱ WebGPU timed out — skipping GPU for remaining sizes.");
      }
    }

    // ── Print row ────────────────────────────────────────────────────────────
    const fmtMs = (v: number) => v === Infinity ? "+Inf" : v.toFixed(1);
    const gpuCell   = gpuMs !== null ? fmtMs(gpuMs)                            : "—";
    const ratioCell = gpuMs !== null && gpuMs !== Infinity && wasmMs !== Infinity
      ? (gpuMs / wasmMs).toFixed(1) + "×" : "—";
    const cells = [
      n.toLocaleString("en-US").padStart(colW[0]),
      fmtMs(wasmMs).padStart(colW[1]),
      (wasmMs === Infinity ? "+Inf" : usPerStep.toFixed(n >= 3_200 ? 1 : 0)).padStart(colW[2]),
      gpuCell.padStart(colW[3]),
      ratioCell.padStart(colW[4]),
    ];
    console.log(cells.join("  "));
  } catch (e) {
    console.error(`\n⚠ N=${n.toLocaleString("en-US")} failed: ${(e as Error).message?.split("\n")[0] ?? e}`);
    console.log("  Stopping benchmark at this N — writing partial results.\n");
    break;
  }
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
