/**
 * Checkpoint benchmark — lax.scan `checkpoint: false` vs `checkpoint: true`.
 *
 * Produces the values used in the "Gradient checkpointing" table in
 * README.md.  Writes a sidecar to
 * assets/timings/bench-checkpoint.json.
 *
 * Methodology: 1 warm-up + BENCH_RUNS timed runs per strategy; stores the
 * mean.  Fixed maxIter=60 so the budget is the same regardless of convergence.
 *
 * Usage:  pnpm run bench:checkpoint
 * Output: assets/timings/bench-checkpoint.json
 */

import { DType, defaultDevice } from "@hamk-uas/jax-js-nonconsuming";
import { dlmMLE } from "../src/mle.ts";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { writeTimingsSidecar, stampMachineInfo } from "./lib/timing-sidecar.ts";

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");

defaultDevice("wasm");

const WARMUP_RUNS = 1;
const BENCH_RUNS  = 4;
const MAX_ITER    = 60;
const LR          = 0.05;

// ── Data ───────────────────────────────────────────────────────────────────

const nileIn   = JSON.parse(readFileSync(resolve(root, "tests/niledemo-in.json"), "utf-8"));
const energyIn = JSON.parse(readFileSync(resolve(root, "tests/energy-in.json"), "utf-8"));

// ── Helper ─────────────────────────────────────────────────────────────────

async function runBench(
  y: number[],
  options: Parameters<typeof dlmMLE>[1],
  checkpoint: boolean,
): Promise<number> {  // returns mean ms
  const times: number[] = [];
  for (let r = 0; r < WARMUP_RUNS + BENCH_RUNS; r++) {
    const t0 = performance.now();
    await dlmMLE(
      y,
      options,
      undefined,  // init
      MAX_ITER,
      LR,
      1e-6,       // tol
      DType.Float64,
      undefined,  // callbacks
      undefined,  // X
      undefined,  // sFixed
      undefined,  // adamOpts
      checkpoint,
    );
    if (r >= WARMUP_RUNS) times.push(performance.now() - t0);
  }
  return times.reduce((a, b) => a + b, 0) / times.length;
}

// ── Run ────────────────────────────────────────────────────────────────────

console.log("=== Checkpoint benchmark (wasm, Float64) ===");
console.log(`maxIter=${MAX_ITER}  warmup=${WARMUP_RUNS}  timed=${BENCH_RUNS} runs (mean)\n`);

console.log(
  ["Dataset", "checkpoint:false", "checkpoint:true", "speedup"].map(s => s.padEnd(22)).join(" ")
);
console.log("─".repeat(92));

const nile_false  = await runBench(nileIn.y,   { order: 1 },                         false);
const nile_true   = await runBench(nileIn.y,   { order: 1 },                         true);
const nile_speedup = (nile_true / nile_false - 1) * 100;
console.log(
  ["Nile (n=100, m=2)",
    `${Math.round(nile_false)} ms`,
    `${Math.round(nile_true)} ms`,
    `+${Math.round(nile_speedup)}%`].map(s => s.padEnd(22)).join(" ")
);

const energy_false  = await runBench(energyIn.y, { order: 1, trig: 1, arphi: [0.85] }, false);
const energy_true   = await runBench(energyIn.y, { order: 1, trig: 1, arphi: [0.85] }, true);
const energy_speedup = (energy_true / energy_false - 1) * 100;
console.log(
  ["Energy (n=120, m=5)",
    `${Math.round(energy_false)} ms`,
    `${Math.round(energy_true)} ms`,
    `+${Math.round(energy_speedup)}%`].map(s => s.padEnd(22)).join(" ")
);

console.log("\nDone.");

// ── Write sidecar ─────────────────────────────────────────────────────────

writeTimingsSidecar("bench-checkpoint", {
  nile_false_ms:    nile_false,
  nile_true_ms:     nile_true,
  nile_speedup_pct: nile_speedup,
  energy_false_ms:  energy_false,
  energy_true_ms:   energy_true,
  energy_speedup_pct: energy_speedup,
});
stampMachineInfo();
