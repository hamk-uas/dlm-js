/**
 * Repro: WebGPU backward smoother corruption for Energy model (m=5, n=120).
 *
 * The first n - 2^⌈log₂n⌉ = 120 - 64 = 56 indices of the smoothed states
 * are wildly wrong on WebGPU (assoc algorithm). From index 56 onward, values
 * match WASM/scan within normal Float32 tolerance.
 *
 * The corrupted region shows a period-5 oscillation (matching m=5 state dim),
 * suggesting individual state components leak through instead of the correct
 * F-projected combined signal.
 *
 * Run with Deno:
 *   DISPLAY=:1 deno run --unstable-webgpu --allow-read --allow-write --allow-env --allow-run \
 *     issues/repro-webgpu-energy-backward-smoother.ts
 */

import { defaultDevice, init } from "../node_modules/@hamk-uas/jax-js-nonconsuming/dist/index.js";
import { dlmFit } from "../src/index.ts";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const input = JSON.parse(readFileSync(resolve(root, "tests/energy-in.json"), "utf8"));
const y: number[] = input.y;
const n = y.length;

// Fixed parameters (from converged MLE on WASM/f64)
const s = 1.4265;
const w = [2.1494, 0.0016, 0.2783, 0.0123, 1.546];
const arphi = [0.6803];

const modelOpts = { order: 1, harmonics: 1, seasonLength: 12, arCoefficients: arphi };

// F indices where F[j]=1 for this model: [0, 2, 4]
const fInds = [0, 2, 4];

async function runVariant(device: string, algorithm: string | undefined, dtype: string) {
  const label = `${device}/${dtype}/${algorithm ?? 'default'}`;
  const fit = await dlmFit(y, {
    obsStd: s, processStd: w, dtype: dtype as 'f32' | 'f64', ...modelOpts,
    algorithm: algorithm as 'scan' | 'assoc' | undefined,
  });

  const combined = Array.from({ length: n }, (_, i) =>
    fInds.reduce((sum, fi) => sum + fit.smoothed.get(i, fi), 0),
  );

  return { label, combined };
}

async function main() {
  // ── WASM reference (scan, f64) ──
  await init("wasm");
  defaultDevice("wasm");
  const ref = await runVariant("wasm", "scan", "f64");

  // ── WASM assoc (f64) — should match reference ──
  const wasmAssoc = await runVariant("wasm", "assoc", "f64");

  // ── WebGPU (assoc, f32) — THIS IS THE BUG ──
  await init("webgpu");
  defaultDevice("webgpu");
  const webgpu = await runVariant("webgpu", "assoc", "f32");

  // ── WASM (scan, f32) — isolate f32 from WebGPU ──
  defaultDevice("wasm");
  const wasmF32 = await runVariant("wasm", "scan", "f32");

  // ── Compare ──
  console.log("\n═══ Comparison: WebGPU assoc vs WASM scan (reference) ═══");
  console.log(`n=${n}, m=5, boundary = n - 2^⌈log₂n⌉ = ${n} - 64 = 56\n`);

  let maxDiffWebgpu = 0, maxIdxWebgpu = -1;
  let maxDiffWasmAssoc = 0;
  let maxDiffWasmF32 = 0;

  console.log("idx  | ref(wasm/f64) | webgpu/f32 |  Δ(webgpu) | wasm-assoc/f64 | Δ(assoc) | wasm/f32 | Δ(f32)");
  console.log("---- | ------------- | ---------- | ---------- | -------------- | -------- | -------- | ------");

  for (let i = 0; i < n; i++) {
    const dWebgpu = Math.abs(webgpu.combined[i] - ref.combined[i]);
    const dAssoc = Math.abs(wasmAssoc.combined[i] - ref.combined[i]);
    const dF32 = Math.abs(wasmF32.combined[i] - ref.combined[i]);

    if (dWebgpu > maxDiffWebgpu) { maxDiffWebgpu = dWebgpu; maxIdxWebgpu = i; }
    if (dAssoc > maxDiffWasmAssoc) maxDiffWasmAssoc = dAssoc;
    if (dF32 > maxDiffWasmF32) maxDiffWasmF32 = dF32;

    // Print corrupted region + a few good indices for contrast
    if (i < 60 || i % 20 === 0 || dWebgpu > 1) {
      console.log(
        `${String(i).padStart(4)} | ${ref.combined[i].toFixed(2).padStart(13)} | ${webgpu.combined[i].toFixed(2).padStart(10)} | ${dWebgpu.toFixed(2).padStart(10)} | ${wasmAssoc.combined[i].toFixed(2).padStart(14)} | ${dAssoc.toFixed(6).padStart(8)} | ${wasmF32.combined[i].toFixed(2).padStart(8)} | ${dF32.toFixed(4).padStart(6)}`
      );
    }
  }

  console.log("\n═══ Summary ═══");
  console.log(`WebGPU assoc max error: ${maxDiffWebgpu.toFixed(2)} at index ${maxIdxWebgpu}`);
  console.log(`WASM assoc max error:   ${maxDiffWasmAssoc.toFixed(8)}`);
  console.log(`WASM f32 max error:     ${maxDiffWasmF32.toFixed(4)}`);
  console.log(`\nCorruption boundary: index 56 = n(${n}) - 64 = n - 2^⌈log₂n⌉`);

  // Verdict
  const corrupted = maxDiffWebgpu > 5;
  if (corrupted) {
    console.log("\n❌ FAIL — WebGPU backward smoother produces corrupted smoothed states");
    console.log("   for indices 0.." + (n - 64 - 1) + " (first n - 2^⌈log₂n⌉ elements).");
    console.log("   WASM assoc is fine → bug is in WebGPU reverse associativeScan.");
    // @ts-ignore Deno.exit
    if (typeof Deno !== 'undefined') Deno.exit(1);
    else process.exit(1);
  } else {
    console.log("\n✅ PASS — WebGPU backward smoother matches reference within tolerance.");
  }
}

main().catch(console.error);
