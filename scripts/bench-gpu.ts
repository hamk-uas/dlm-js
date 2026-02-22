/**
 * WebGPU `dlmFit` benchmark — float32 with associativeScan forward filter.
 *
 * Runs the same models as bench-backends.ts but on the WebGPU backend,
 * which triggers the exact 5-tuple parallel forward filter + assoc path.
 *
 * Must be run with Deno (WebGPU requires --unstable-webgpu):
 *   pnpm run bench:gpu
 *
 * Writes timing data to assets/timings/bench-gpu.json, then invokes
 * `pnpm run update:timings` to patch .md markers.
 *
 * Output: assets/timings/bench-gpu.json
 */

import { defaultDevice, init } from "../node_modules/@hamk-uas/jax-js-nonconsuming/dist/index.js";
import { dlmFit } from "../src/index.ts";
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const sidecarDir = resolve(root, "assets/timings");

// ── Init WebGPU ──────────────────────────────────────────────────────────

await init("webgpu");
defaultDevice("webgpu");

// ── Load data ──────────────────────────────────────────────────────────────

const nileIn       = JSON.parse(readFileSync(resolve(root, "tests/niledemo-in.json"), "utf-8"));
const kaisaniemiIn = JSON.parse(readFileSync(resolve(root, "tests/kaisaniemi-in.json"), "utf-8"));
const trigarIn     = JSON.parse(readFileSync(resolve(root, "tests/trigar-in.json"), "utf-8"));
const order0In     = JSON.parse(readFileSync(resolve(root, "tests/order0-in.json"), "utf-8"));

// ── Models (same as bench-backends.ts) ──────────────────────────────────

interface Model {
  label: string;
  key: string;
  y: number[];
  s: number | number[];
  w: number[];
  options: Record<string, unknown>;
  n: number;
  m: number;
}

const models: Model[] = [
  {
    label: "Nile, order=0",
    key: "nile_o0",
    y: order0In.y, s: order0In.s, w: order0In.w,
    options: { order: 0 },
    n: 100, m: 1,
  },
  {
    label: "Nile, order=1",
    key: "nile_o1",
    y: nileIn.y, s: nileIn.s, w: nileIn.w,
    options: { order: 1 },
    n: 100, m: 2,
  },
  {
    label: "Kaisaniemi, trig",
    key: "kaisaniemi",
    y: kaisaniemiIn.y, s: kaisaniemiIn.s, w: kaisaniemiIn.w,
    options: { order: 1, harmonics: 1, seasonLength: 12 },
    n: 117, m: 4,
  },
  {
    label: "Energy, trig+AR",
    key: "trigar",
    y: trigarIn.y, s: trigarIn.s, w: trigarIn.w,
    options: { order: 1, harmonics: 1, seasonLength: 12, arCoefficients: trigarIn.arphi },
    n: 120, m: 5,
  },
];

// ── Timing helper ──────────────────────────────────────────────────────────

async function timedFit(model: Model): Promise<{ firstMs: number; warmMs: number }> {
  const { y, s, w, options } = model;

  // First run (JIT compilation)
  const t0 = performance.now();
  const r1 = await dlmFit(y, { obsStd: s, processStd: w, dtype: 'f32', ...options });
  const t1 = performance.now();
  r1[Symbol.dispose]?.();

  // Warm run (cached)
  const t2 = performance.now();
  const r2 = await dlmFit(y, { obsStd: s, processStd: w, dtype: 'f32', ...options });
  const t3 = performance.now();
  r2[Symbol.dispose]?.();

  return { firstMs: t1 - t0, warmMs: t3 - t2 };
}

// ── Run benchmarks ─────────────────────────────────────────────────────────

console.log("=== dlmFit WebGPU benchmark (float32, assoc) ===\n");

const colW = [32, 8, 6, 14, 14];
const hdr = [
  "Model".padEnd(colW[0]),
  "n".padStart(colW[1]),
  "m".padStart(colW[2]),
  "first (ms)".padStart(colW[3]),
  "warm (ms)".padStart(colW[4]),
].join("  ");
console.log(hdr);
console.log("─".repeat(hdr.length));

const sidecar: Record<string, number> = {};

for (const model of models) {
  const { firstMs, warmMs } = await timedFit(model);

  const cells = [
    model.label.padEnd(colW[0]),
    String(model.n).padStart(colW[1]),
    String(model.m).padStart(colW[2]),
    `${firstMs.toFixed(1)}`.padStart(colW[3]),
    `${warmMs.toFixed(1)}`.padStart(colW[4]),
  ];
  console.log(cells.join("  "));

  sidecar[`${model.key}__webgpu_f32__first`] = firstMs;
  sidecar[`${model.key}__webgpu_f32__warm`]  = warmMs;
}

console.log("\nDone.");

// ── Write sidecar ──────────────────────────────────────────────────────────

if (!existsSync(sidecarDir)) mkdirSync(sidecarDir, { recursive: true });
const outPath = resolve(sidecarDir, "bench-gpu.json");
writeFileSync(outPath, JSON.stringify(sidecar, null, 2) + "\n");
console.log(`Wrote ${outPath}`);

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
