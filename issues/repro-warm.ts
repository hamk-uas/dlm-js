/**
 * Reproduce: warm WebGPU vs warm WASM for dlmFit N=100 m=2
 *
 * Run:
 *   GPU=nvidia bash scripts/gpu-test.sh run issues/repro-warm.ts
 */

import { describe, it } from 'vitest';
import { commands } from 'vitest/browser';
import { defaultDevice, init } from "@hamk-uas/jax-js-nonconsuming";
import { dlmFit } from "../src/index.ts";

describe('repro-warm', () => {
  it('warm WebGPU vs warm WASM', async () => {
    const nileIn = JSON.parse(await commands.readFile("tests/niledemo-in.json"));
    const baseY = nileIn.y;
    const s = nileIn.s;
    const w = nileIn.w;

    // ── WebGPU warm ──
    await init("webgpu");
    defaultDevice("webgpu");

    const r0 = await dlmFit(baseY, { obsStd: s, processStd: w, order: 1, dtype: 'f32' });
    r0[Symbol.dispose]?.();
    const r1 = await dlmFit(baseY, { obsStd: s, processStd: w, order: 1, dtype: 'f32' });
    r1[Symbol.dispose]?.();

    const gpuTimes: number[] = [];
    for (let i = 0; i < 5; i++) {
      const t0 = performance.now();
      const r = await dlmFit(baseY, { obsStd: s, processStd: w, order: 1, dtype: 'f32' });
      gpuTimes.push(performance.now() - t0);
      r[Symbol.dispose]?.();
    }
    gpuTimes.sort((a, b) => a - b);
    console.log(`WebGPU warm (5 runs): ${gpuTimes.map(t => t.toFixed(1)).join(', ')} ms`);
    console.log(`WebGPU median: ${gpuTimes[2].toFixed(1)}ms`);

    // ── WASM warm ──
    await init("wasm");
    defaultDevice("wasm");

    const w0 = await dlmFit(baseY, { obsStd: s, processStd: w, order: 1, dtype: 'f64' });
    w0[Symbol.dispose]?.();
    const w1 = await dlmFit(baseY, { obsStd: s, processStd: w, order: 1, dtype: 'f64' });
    w1[Symbol.dispose]?.();

    const wasmTimes: number[] = [];
    for (let i = 0; i < 5; i++) {
      const t0 = performance.now();
      const r = await dlmFit(baseY, { obsStd: s, processStd: w, order: 1, dtype: 'f64' });
      wasmTimes.push(performance.now() - t0);
      r[Symbol.dispose]?.();
    }
    wasmTimes.sort((a, b) => a - b);
    console.log(`\nWASM warm (5 runs): ${wasmTimes.map(t => t.toFixed(1)).join(', ')} ms`);
    console.log(`WASM median: ${wasmTimes[2].toFixed(1)}ms`);
    console.log(`\nRatio: ${(gpuTimes[2] / wasmTimes[2]).toFixed(1)}x`);
  });
});