/**
 * WebGPU associativeScan dispatch overhead benchmark.
 *
 * Measures WebGPU dlmFit (assoc, f32) vs WASM dlmFit (scan, f64) at
 * increasing N. The GPU/WASM ratio reveals how many dispatches are
 * being wasted — the target is ratio ≤ 2× (near-parity with WASM).
 *
 * Run:
 *   GPU=nvidia bash scripts/gpu-test.sh run issues/repro-webgpu-block-map-perf.ts
 */

import { describe, it, expect } from 'vitest';
import { commands } from 'vitest/browser';
import { defaultDevice, init } from "@hamk-uas/jax-js-nonconsuming";
import { dlmFit } from "../src/index.ts";
import type { DlmDtype } from "../src/types.ts";

function makeY(baseY: number[], n: number): number[] {
  const y: number[] = [];
  while (y.length < n) y.push(...baseY.slice(0, n - y.length));
  return y;
}

const TEST_N = [100, 3200, 12800, 25600];
const WASM_RUNS = 4;

describe('repro-webgpu-block-map-perf', () => {
  it('GPU/WASM ratio within target', async () => {
    const nileIn = JSON.parse(await commands.readFile("tests/niledemo-in.json"));
    const baseY: number[] = nileIn.y;
    const s: number = nileIn.s;
    const w: number[] = nileIn.w;
    const options = { order: 1 };

    await init("wasm");
    await init("webgpu");

    // WASM JIT warmup
    defaultDevice("wasm");
    const rWarmup = await dlmFit(makeY(baseY, 100), { obsStd: s, processStd: w, dtype: 'f64' as DlmDtype, ...options });
    rWarmup[Symbol.dispose]?.();

    console.log("dlmFit benchmark: WebGPU/f32/assoc (cold) vs WASM/f64/scan (warm)");
    console.log("Model: Nile order=1, m=2. Target: GPU/WASM ratio ≤ 2×.\n");
    console.log("  N       | WASM/f64 (warm) | WebGPU/f32 (cold) |  ratio | target?");
    console.log("  --------|-----------------|-------------------|--------|--------");

    let anyFail = false;

    for (const n of TEST_N) {
      const y = makeY(baseY, n);

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

    expect(anyFail, "WebGPU/WASM ratio exceeds 2× target").toBe(false);
  });
});
