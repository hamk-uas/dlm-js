/**
 * MLE benchmark — all comparison-table rows for the dlm-js column.
 *
 * Runs dlmMLE on every model in README.md, writes timing data to
 * assets/timings/collect-mle-benchmark.json, and prints a summary table.
 *
 * Models covered (matching README.md benchmark table):
 *   • Nile,       order=1, fit s+w          (n=100, m=2)
 *   • Nile,       order=0, fit s+w          (n=100, m=1)
 *   • Kaisaniemi, order=1, trig=1, ns=12    (n=117, m=4)
 *   • Energy MLE timings come from collect-energy-mle-frames.ts (it writes
 *     its own sidecar), so Energy is not repeated here.
 *
 * Usage:  pnpm run bench:mle
 * Output: assets/timings/collect-mle-benchmark.json
 */

import { defaultDevice } from "@hamk-uas/jax-js-nonconsuming";
import { dlmMLE } from "../src/mle.ts";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { writeTimingsSidecar, stampMachineInfo } from "./lib/timing-sidecar.ts";

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");

defaultDevice("wasm");

// ── Data ───────────────────────────────────────────────────────────────────

const nileIn        = JSON.parse(readFileSync(resolve(root, "tests/niledemo-in.json"), "utf-8"));
const kaisaniemiIn  = JSON.parse(readFileSync(resolve(root, "tests/kaisaniemi-in.json"), "utf-8"));

// ── Helper: median of N timed runs (after 1 warm-up) ──────────────────────

const WARMUP = 1;
const RUNS   = 3;
const MAX_ITER = 300;
const LR       = 0.05;

async function timedMle(
  y: number[],
  options: Record<string, unknown>,
): Promise<{ elapsed: number; iterations: number; lik: number }> {
  // warm-up
  await dlmMLE(y, { ...options, maxIter: MAX_ITER, lr: LR, tol: 1e-6, dtype: 'f64' as const });

  const times: number[] = [];
  let last = { elapsed: 0, iterations: 0, lik: 0 };
  for (let i = 0; i < RUNS; i++) {
    const r = await dlmMLE(y, { ...options, maxIter: MAX_ITER, lr: LR, tol: 1e-6, dtype: 'f64' as const });
    times.push(r.elapsed);
    last = { elapsed: r.elapsed, iterations: r.iterations, lik: r.deviance };
  }
  times.sort((a, b) => a - b);
  return { elapsed: times[Math.floor(RUNS / 2)], iterations: last.iterations, lik: last.lik };
}

// ── Benchmarks ────────────────────────────────────────────────────────────

console.log("=== MLE benchmark (wasm, Float64) ===");
console.log(`maxIter=${MAX_ITER}  warmup=${WARMUP}  runs=${RUNS}  taking median\n`);

const header = ["Model", "n", "m", "median ms", "iters", "-2logL"].map(s => s.padEnd(28)).join(" ");
console.log(header);
console.log("─".repeat(header.length));

const nileOrder1 = await timedMle(nileIn.y, { order: 1 });
console.log(["Nile order=1 (s+w)", "100", "2",
  `${Math.round(nileOrder1.elapsed)} ms`, String(nileOrder1.iterations), nileOrder1.lik.toFixed(1)]
  .map(s => s.padEnd(28)).join(" "));

const nileOrder0 = await timedMle(nileIn.y, { order: 0 });
console.log(["Nile order=0 (s+w)", "100", "1",
  `${Math.round(nileOrder0.elapsed)} ms`, String(nileOrder0.iterations), nileOrder0.lik.toFixed(1)]
  .map(s => s.padEnd(28)).join(" "));

const kaisaniemi = await timedMle(kaisaniemiIn.y, { order: 1, harmonics: 1, seasonLength: 12 });
console.log(["Kaisaniemi trig (s+w)", "117", "4",
  `${Math.round(kaisaniemi.elapsed)} ms`, String(kaisaniemi.iterations), kaisaniemi.lik.toFixed(1)]
  .map(s => s.padEnd(28)).join(" "));

console.log("\nDone.");

// ── Write sidecar ─────────────────────────────────────────────────────────

writeTimingsSidecar("collect-mle-benchmark", {
  nile_order1_elapsed:     nileOrder1.elapsed,
  nile_order1_iterations:  nileOrder1.iterations,
  nile_order1_lik:         nileOrder1.lik,
  nile_order0_elapsed:     nileOrder0.elapsed,
  nile_order0_iterations:  nileOrder0.iterations,
  nile_order0_lik:         nileOrder0.lik,
  kaisaniemi_elapsed:      kaisaniemi.elapsed,
  kaisaniemi_iterations:   kaisaniemi.iterations,
  kaisaniemi_lik:          kaisaniemi.lik,
});
stampMachineInfo();
