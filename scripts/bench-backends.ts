/**
 * Cross-backend `dlmFit` benchmark — cpu/f32, wasm/f32, wasm/f64 timings.
 *
 * Runs each demo model twice per backend (first run = JIT, warm run = cached).
 * Writes timing data to assets/timings/bench-backends.json and patches .md
 * timing markers via update:timings.
 *
 * WebGPU is benchmarked separately via `pnpm run bench:gpu` (requires Deno
 * with --unstable-webgpu). See scripts/bench-gpu.ts.
 *
 * Usage:  pnpm run bench:backends
 * Output: assets/timings/bench-backends.json
 */

import { DType, defaultDevice, init } from "@hamk-uas/jax-js-nonconsuming";
import { dlmFit } from "../src/index.ts";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { performance } from "node:perf_hooks";
import { writeTimingsSidecar, stampMachineInfo } from "./lib/timing-sidecar.ts";

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");

// ── Load data ──────────────────────────────────────────────────────────────

const nileIn       = JSON.parse(readFileSync(resolve(root, "tests/niledemo-in.json"), "utf-8"));
const kaisaniemiIn = JSON.parse(readFileSync(resolve(root, "tests/kaisaniemi-in.json"), "utf-8"));
const trigarIn     = JSON.parse(readFileSync(resolve(root, "tests/trigar-in.json"), "utf-8"));
const order0In     = JSON.parse(readFileSync(resolve(root, "tests/order0-in.json"), "utf-8"));

// ── Models ─────────────────────────────────────────────────────────────────

interface Model {
  label: string;
  key: string;        // sidecar key prefix
  y: number[];
  s: number | number[];
  w: number[];
  options: Parameters<typeof dlmFit>[4];
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
    options: { order: 1, trig: 1, ns: 12 },
    n: 117, m: 4,
  },
  {
    label: "Energy, trig+AR",
    key: "trigar",
    y: trigarIn.y, s: trigarIn.s, w: trigarIn.w,
    options: { order: 1, trig: 1, ns: 12, arphi: trigarIn.arphi },
    n: 120, m: 5,
  },
];

// ── Backends ───────────────────────────────────────────────────────────────

interface Backend {
  label: string;
  key: string;        // sidecar key suffix
  device: string;
  dtype: DType;
}

const backends: Backend[] = [
  { label: "cpu/f32",  key: "cpu_f32",  device: "cpu",  dtype: DType.Float32 },
  { label: "wasm/f32", key: "wasm_f32", device: "wasm", dtype: DType.Float32 },
  { label: "wasm/f64", key: "wasm_f64", device: "wasm", dtype: DType.Float64 },
];

// ── Timing helper ──────────────────────────────────────────────────────────

async function timedFit(
  model: Model,
  backend: Backend,
): Promise<{ firstMs: number; warmMs: number }> {
  defaultDevice(backend.device as "cpu" | "wasm");
  const { y, s, w, options } = model;
  const dtype = backend.dtype;

  // First run (JIT compilation)
  const t0 = performance.now();
  await dlmFit(y, s, w, dtype, options);
  const t1 = performance.now();

  // Warm run (cached)
  const t2 = performance.now();
  await dlmFit(y, s, w, dtype, options);
  const t3 = performance.now();

  return { firstMs: t1 - t0, warmMs: t3 - t2 };
}

// ── Run benchmarks ─────────────────────────────────────────────────────────

console.log("=== dlmFit cross-backend benchmark ===\n");

const colW = [32, 8, 6, 14, 14, 14];
const hdr = [
  "Model".padEnd(colW[0]),
  "n".padStart(colW[1]),
  "m".padStart(colW[2]),
  ...backends.map(b => b.label.padStart(colW[3])),
].join("  ");
console.log(hdr);
console.log("─".repeat(hdr.length));

const sidecar: Record<string, number> = {};

for (const model of models) {
  const cells = [
    model.label.padEnd(colW[0]),
    String(model.n).padStart(colW[1]),
    String(model.m).padStart(colW[2]),
  ];

  for (const backend of backends) {
    const { firstMs, warmMs } = await timedFit(model, backend);
    cells.push(`${warmMs.toFixed(1)} ms`.padStart(colW[3]));

    sidecar[`${model.key}__${backend.key}__first`] = firstMs;
    sidecar[`${model.key}__${backend.key}__warm`]  = warmMs;
  }

  console.log(cells.join("  "));
}

console.log("\n(warm-run timings shown; first-run includes JIT compilation overhead)");
console.log("Done.");

// ── Write sidecar ──────────────────────────────────────────────────────────

writeTimingsSidecar("bench-backends", sidecar);
stampMachineInfo();
