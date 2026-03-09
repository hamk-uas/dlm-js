// @ts-nocheck
// Reproduce: warm WebGPU vs warm WASM for dlmFit N=100 m=2
import { defaultDevice, init, clearCaches, setDebug } from "../node_modules/@hamk-uas/jax-js-nonconsuming/dist/index.js";
import { dlmFit } from "../src/index.ts";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const nileIn = JSON.parse(readFileSync(resolve(root, "tests/niledemo-in.json"), "utf-8"));
const baseY = nileIn.y;
const s = nileIn.s;
const w = nileIn.w;

// ── WebGPU warm ──
await init("webgpu");
defaultDevice("webgpu");

// Warmup (2 calls: cold + warm cache)
const r0 = await dlmFit(baseY, { obsStd: s, processStd: w, order: 1, dtype: 'f32' });
r0[Symbol.dispose]?.();
const r1 = await dlmFit(baseY, { obsStd: s, processStd: w, order: 1, dtype: 'f32' });
r1[Symbol.dispose]?.();

// Measure warm (5 runs)
const gpuTimes = [];
for (let i = 0; i < 5; i++) {
  const t0 = performance.now();
  const r = await dlmFit(baseY, { obsStd: s, processStd: w, order: 1, dtype: 'f32' });
  const dt = performance.now() - t0;
  gpuTimes.push(dt);
  r[Symbol.dispose]?.();
}
gpuTimes.sort((a, b) => a - b);
console.log(`WebGPU warm (5 runs): ${gpuTimes.map(t => t.toFixed(1)).join(', ')} ms`);
console.log(`WebGPU median: ${gpuTimes[2].toFixed(1)}ms`);

// ── WASM warm ──
await init("wasm");
defaultDevice("wasm");

// Warmup
const w0 = await dlmFit(baseY, { obsStd: s, processStd: w, order: 1, dtype: 'f64' });
w0[Symbol.dispose]?.();
const w1 = await dlmFit(baseY, { obsStd: s, processStd: w, order: 1, dtype: 'f64' });
w1[Symbol.dispose]?.();

const wasmTimes = [];
for (let i = 0; i < 5; i++) {
  const t0 = performance.now();
  const r = await dlmFit(baseY, { obsStd: s, processStd: w, order: 1, dtype: 'f64' });
  const dt = performance.now() - t0;
  wasmTimes.push(dt);
  r[Symbol.dispose]?.();
}
wasmTimes.sort((a, b) => a - b);
console.log(`\nWASM warm (5 runs): ${wasmTimes.map(t => t.toFixed(1)).join(', ')} ms`);
console.log(`WASM median: ${wasmTimes[2].toFixed(1)}ms`);
console.log(`\nRatio: ${(gpuTimes[2] / wasmTimes[2]).toFixed(1)}x`);
