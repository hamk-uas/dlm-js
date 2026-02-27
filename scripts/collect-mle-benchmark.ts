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
const energyIn      = JSON.parse(readFileSync(resolve(root, "tests/energy-in.json"), "utf-8"));

// ── Helper: median of N timed runs (after 1 warm-up) ──────────────────────

const WARMUP = 1;
const RUNS   = 3;
const MAX_ITER = 300;
const LR       = 0.05;

/** Hard timeout per dlmMLE run — report >30 s and stop if exceeded. */
const TIMEOUT_MS = 30_000;

async function timedMle(
  y: number[],
  options: Record<string, unknown>,
): Promise<{ elapsed: number; iterations: number; lik: number }> {
  // warm-up
  const warmup = await dlmMLE(y, { ...options, maxIter: MAX_ITER, lr: LR, tol: 1e-6, dtype: 'f64' as const });
  if (warmup.elapsed > TIMEOUT_MS) {
    return { elapsed: Infinity, iterations: warmup.iterations, lik: warmup.deviance };
  }

  const times: number[] = [];
  let last = { elapsed: 0, iterations: 0, lik: 0 };
  for (let i = 0; i < RUNS; i++) {
    const r = await dlmMLE(y, { ...options, maxIter: MAX_ITER, lr: LR, tol: 1e-6, dtype: 'f64' as const });
    times.push(r.elapsed);
    last = { elapsed: r.elapsed, iterations: r.iterations, lik: r.deviance };
    if (r.elapsed > TIMEOUT_MS) {
      return { elapsed: Infinity, iterations: last.iterations, lik: last.lik };
    }
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

// Nile order=1 with observation noise fixed (MATLAB DLM fitv=0).
// obsStdFixed = constant array of initial s from niledemo-in.json.
const sFixed = new Array(nileIn.y.length).fill(nileIn.s);
const nileWonly = await timedMle(nileIn.y, { order: 1, obsStdFixed: sFixed });
console.log(["Nile order=1 (w only)", "100", "2",
  `${Math.round(nileWonly.elapsed)} ms`, String(nileWonly.iterations), nileWonly.lik.toFixed(1)]
  .map(s => s.padEnd(28)).join(" "));

const kaisaniemi = await timedMle(kaisaniemiIn.y, { order: 1, harmonics: 1, seasonLength: 12 });
console.log(["Kaisaniemi trig (s+w)", "117", "4",
  `${Math.round(kaisaniemi.elapsed)} ms`, String(kaisaniemi.iterations), kaisaniemi.lik.toFixed(1)]
  .map(s => s.padEnd(28)).join(" "));

console.log("\nDone.");

// ── Natural gradient benchmarks ───────────────────────────────────────────

console.log("\n=== MLE benchmark — natural gradient (wasm, Float64) ===");
console.log(`maxIter=50  warmup=${WARMUP}  runs=${RUNS}  taking median\n`);

const natHeader = ["Model", "n", "m", "median ms", "iters", "-2logL"].map(s => s.padEnd(28)).join(" ");
console.log(natHeader);
console.log("─".repeat(natHeader.length));

async function timedMleNatural(
  y: number[],
  options: Record<string, unknown>,
): Promise<{ elapsed: number; iterations: number; lik: number }> {
  // warm-up
  const warmup = await dlmMLE(y, { ...options, maxIter: 50, tol: 1e-6, dtype: 'f64' as const, optimizer: 'natural' as const });
  if (warmup.elapsed > TIMEOUT_MS) {
    return { elapsed: Infinity, iterations: warmup.iterations, lik: warmup.deviance };
  }

  const times: number[] = [];
  let last = { elapsed: 0, iterations: 0, lik: 0 };
  for (let i = 0; i < RUNS; i++) {
    const r = await dlmMLE(y, { ...options, maxIter: 50, tol: 1e-6, dtype: 'f64' as const, optimizer: 'natural' as const });
    times.push(r.elapsed);
    last = { elapsed: r.elapsed, iterations: r.iterations, lik: r.deviance };
    if (r.elapsed > TIMEOUT_MS) {
      return { elapsed: Infinity, iterations: last.iterations, lik: last.lik };
    }
  }
  times.sort((a, b) => a - b);
  return { elapsed: times[Math.floor(RUNS / 2)], iterations: last.iterations, lik: last.lik };
}

const natNileOrder1 = await timedMleNatural(nileIn.y, { order: 1 });
console.log(["Nile order=1 (s+w)", "100", "2",
  `${Math.round(natNileOrder1.elapsed)} ms`, String(natNileOrder1.iterations), natNileOrder1.lik.toFixed(1)]
  .map(s => s.padEnd(28)).join(" "));

const natNileOrder0 = await timedMleNatural(nileIn.y, { order: 0 });
console.log(["Nile order=0 (s+w)", "100", "1",
  `${Math.round(natNileOrder0.elapsed)} ms`, String(natNileOrder0.iterations), natNileOrder0.lik.toFixed(1)]
  .map(s => s.padEnd(28)).join(" "));

const natNileWonly = await timedMleNatural(nileIn.y, { order: 1, obsStdFixed: sFixed });
console.log(["Nile order=1 (w only)", "100", "2",
  `${Math.round(natNileWonly.elapsed)} ms`, String(natNileWonly.iterations), natNileWonly.lik.toFixed(1)]
  .map(s => s.padEnd(28)).join(" "));

const natKaisaniemi = await timedMleNatural(kaisaniemiIn.y, { order: 1, harmonics: 1, seasonLength: 12 });
console.log(["Kaisaniemi trig (s+w)", "117", "4",
  `${Math.round(natKaisaniemi.elapsed)} ms`, String(natKaisaniemi.iterations), natKaisaniemi.lik.toFixed(1)]
  .map(s => s.padEnd(28)).join(" "));

const energyOpts = { order: 1, harmonics: 1, seasonLength: 12, arCoefficients: [0.5], fitAr: true };
const natEnergy = await timedMleNatural(energyIn.y, energyOpts);
console.log(["Energy trig+AR (s+w+φ)", "120", "5",
  `${Math.round(natEnergy.elapsed)} ms`, String(natEnergy.iterations), natEnergy.lik.toFixed(1)]
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
  // Natural gradient optimizer
  nat_nile_order1_elapsed:     natNileOrder1.elapsed,
  nat_nile_order1_iterations:  natNileOrder1.iterations,
  nat_nile_order1_lik:         natNileOrder1.lik,
  nat_nile_order0_elapsed:     natNileOrder0.elapsed,
  nat_nile_order0_iterations:  natNileOrder0.iterations,
  nat_nile_order0_lik:         natNileOrder0.lik,
  nile_wonly_elapsed:           nileWonly.elapsed,
  nile_wonly_iterations:        nileWonly.iterations,
  nile_wonly_lik:               nileWonly.lik,
  nat_nile_wonly_elapsed:       natNileWonly.elapsed,
  nat_nile_wonly_iterations:    natNileWonly.iterations,
  nat_nile_wonly_lik:           natNileWonly.lik,
  nat_kaisaniemi_elapsed:      natKaisaniemi.elapsed,
  nat_kaisaniemi_iterations:   natKaisaniemi.iterations,
  nat_kaisaniemi_lik:          natKaisaniemi.lik,
  nat_energy_elapsed:          natEnergy.elapsed,
  nat_energy_iterations:       natEnergy.iterations,
  nat_energy_lik:              natEnergy.lik,
});
stampMachineInfo();
