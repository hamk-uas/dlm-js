/**
 * Instrumented repro: measures WHERE time is spent in dlmFit.
 */
import { defaultDevice, init, setDebug, clearCaches, jit, valueAndGrad, numpy as np, lax, DType } from "../node_modules/@hamk-uas/jax-js-nonconsuming/dist/index.js";
import { dlmFit } from "../src/index.ts";
import type { DlmDtype } from "../src/types.ts";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");

const nileIn = JSON.parse(readFileSync(resolve(root, "tests/niledemo-in.json"), "utf-8"));
const baseY: number[] = nileIn.y;
const s: number = nileIn.s;
const w: number[] = nileIn.w;
const options = { order: 1 };

function makeY(n: number): number[] {
  const y: number[] = [];
  while (y.length < n) y.push(...baseY.slice(0, n - y.length));
  return y;
}

await init("webgpu");
defaultDevice("webgpu");

setDebug(1);

// Single run at N=25600 to capture JIT compile details
const n = 25600;
const y = makeY(n);
console.log(`\n=== dlmFit N=${n} ===`);
const t0 = performance.now();
const r = await dlmFit(y, { obsStd: s, processStd: w, dtype: 'f32' as DlmDtype, ...options });
const elapsed = performance.now() - t0;
console.log(`\nTOTAL: ${elapsed.toFixed(0)} ms`);
r[Symbol.dispose]?.();

Deno.exit(0);
