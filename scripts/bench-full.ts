/**
 * Comprehensive dlmFit benchmark — full coverage of all backend/dtype/algorithm
 * option combinations for each demo model.
 *
 * Combinations tested per model:
 *   backend:       cpu, wasm                    → dtype: f64 or f32
 *                  webgpu                       → dtype: f32 only
 *   dtype:         f64, f32
 *   algorithm:     scan, assoc
 *
 * Note: float32 + m > 2 is documented as numerically unstable — those rows
 * are included but marked "⚠️ NaN" when the output is non-finite.
 *
 * Error columns compare dlm-js output (yhat, ystd, x, xstd) against the
 * Octave/MATLAB reference stored in tests/*-out-m.json.
 *
 * Must be run with Deno (WebGPU requires --unstable-webgpu):
 *   pnpm run bench:full
 *
 * Output: assets/timings/bench-full.json
 */

import { DType, defaultDevice, init } from "../node_modules/@hamk-uas/jax-js-nonconsuming/dist/index.js";
import { dlmFit } from "../src/index.ts";
import type { DlmDtype, DlmAlgorithm, DlmFitResult } from "../src/types.ts";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const sidecarDir = resolve(root, "assets/timings");

// ── Init WebGPU (needed before first webgpu call) ─────────────────────────

await init("webgpu");

// ── Load data ──────────────────────────────────────────────────────────────

const nileIn       = JSON.parse(readFileSync(resolve(root, "tests/niledemo-in.json"), "utf-8"));
const kaisaniemiIn = JSON.parse(readFileSync(resolve(root, "tests/kaisaniemi-in.json"), "utf-8"));
const trigarIn     = JSON.parse(readFileSync(resolve(root, "tests/trigar-in.json"), "utf-8"));
const order0In     = JSON.parse(readFileSync(resolve(root, "tests/order0-in.json"), "utf-8"));
const gappedIn    = JSON.parse(readFileSync(resolve(root, "tests/gapped-in.json"), "utf-8"));
const gappedY: number[] = (gappedIn.y as (number | null)[]).map((v: number | null) => v === null ? NaN : v);

// ── Load Octave/MATLAB reference outputs ───────────────────────────────────
// Used for error comparison (max absolute and max percentage error).

type RefJson = Record<string, number[] | number[][]>;
const refMap: Record<string, { ref: RefJson; m: number; n: number }> = {
  "Nile, order=0":   { ref: JSON.parse(readFileSync(resolve(root, "tests/order0-out-m.json"),    "utf-8")), m: 1, n: 100 },
  "Nile, order=1":   { ref: JSON.parse(readFileSync(resolve(root, "tests/niledemo-out-m.json"),  "utf-8")), m: 2, n: 100 },
  "Kaisaniemi, trig":{ ref: JSON.parse(readFileSync(resolve(root, "tests/kaisaniemi-out-m.json"),"utf-8")), m: 4, n: 117 },
  "Energy, trig+AR": { ref: JSON.parse(readFileSync(resolve(root, "tests/trigar-out-m.json"),    "utf-8")), m: 5, n: 120 },
  "Gapped, order=1": { ref: JSON.parse(readFileSync(resolve(root, "tests/gapped-out-m.json"),  "utf-8")), m: 2, n: 100 },
};

// ── Models ─────────────────────────────────────────────────────────────────

interface Model {
  label: string;
  y: number[];
  s: number | number[];
  w: number[];
  options: Record<string, unknown>;
  n: number;
  m: number;
}

const toW = (v: unknown): number[] => Array.isArray(v) ? v as number[] : [v as number];

const models: Model[] = [
  {
    label: "Nile, order=0",
    y: order0In.y, s: order0In.s, w: toW(order0In.w),
    options: { order: 0 },
    n: 100, m: 1,
  },
  {
    label: "Nile, order=1",
    y: nileIn.y, s: nileIn.s, w: toW(nileIn.w),
    options: { order: 1 },
    n: 100, m: 2,
  },
  {
    label: "Kaisaniemi, trig",
    // kaisaniemi-in.json includes the options used to generate the reference
    y: kaisaniemiIn.y, s: kaisaniemiIn.s, w: toW(kaisaniemiIn.w),
    options: kaisaniemiIn.options,      // { order: 1, harmonics: 1 }
    n: 117, m: 4,
  },
  {
    label: "Energy, trig+AR",
    y: trigarIn.y, s: trigarIn.s, w: toW(trigarIn.w),
    options: { order: 1, harmonics: 1, seasonLength: 12, arCoefficients: [0.7] },
    n: 120, m: 5,
  },
  {
    label: "Gapped, order=1",
    y: gappedY, s: gappedIn.s, w: toW(gappedIn.w),
    options: gappedIn.options,      // { order: 1 }
    n: 100, m: 2,
  },
];

// ── Combinations ───────────────────────────────────────────────────────────

interface Combo {
  backend: 'cpu' | 'wasm' | 'webgpu';
  dlmDtype: DlmDtype;
  algorithm: DlmAlgorithm;
}

const combos: Combo[] = [];

for (const backend of ['cpu', 'wasm'] as const) {
  for (const dlmDtype of ['f64', 'f32'] as const) {
    for (const algorithm of ['scan', 'assoc'] as const) {
      combos.push({ backend, dlmDtype, algorithm });
    }
  }
}
// webgpu: float32 only
for (const algorithm of ['scan', 'assoc'] as const) {
  combos.push({ backend: 'webgpu', dlmDtype: 'f32', algorithm });
}

// ── Error computation helpers ──────────────────────────────────────────────

/**
 * Flatten MATLAB/Octave reference values for comparison.
 * Fields compared: yhat[n], ystd[n], x (state means), xstd (state stds).
 * Reference shapes:
 *   m=1: x=[n] flat,  xstd=[n] flat
 *   m>1: x=[m][n],    xstd=[n][m]  (ref stores row-per-time for xstd)
 */
function flattenRef(ref: RefJson, m: number, n: number): number[] {
  const out: number[] = [];
  for (const v of ref['yhat'] as number[]) out.push(v);
  for (const v of ref['ystd'] as number[]) out.push(v);
  if (m === 1) {
    for (const v of ref['x'] as number[]) out.push(v);
    for (const v of ref['xstd'] as number[]) out.push(v);
  } else {
    const x = ref['x'] as number[][];
    for (let i = 0; i < m; i++) for (let t = 0; t < n; t++) out.push(x[i][t]);
    const xstd = ref['xstd'] as number[][];
    for (let t = 0; t < n; t++) for (let i = 0; i < m; i++) out.push(xstd[t][i]);
  }
  return out;
}

/**
 * Flatten dlmFit result values in the same order as flattenRef.
 * result.x    = FloatArray[m_actual], each of length n → x[i][t]
 * result.xstd = FloatArray[n],        each of length m → xstd[t][i]
 * Uses the result's own dimensions; m/n params are from the reference for sizing only.
 */
function flattenResult(r: DlmFitResult, m: number, n: number): number[] {
  const out: number[] = [];
  const yhat = r.yhat;
  const ystd = r.ystd;
  const actualN = yhat.length;
  const useN = Math.min(n, actualN);
  const useM = Math.min(m, r.m);

  for (let t = 0; t < useN; t++) out.push(yhat[t]);
  for (let t = 0; t < useN; t++) out.push(ystd[t]);
  for (let i = 0; i < useM; i++) {
    const xi = r.smoothed.series(i);
    for (let t = 0; t < useN; t++) out.push(xi[t]);
  }
  for (let t = 0; t < useN; t++) {
    for (let i = 0; i < useM; i++) out.push(r.smoothedStd.get(t, i));
  }
  return out;
}

/** Return a prefix-trimmed version of refVals to match length of gotVals. */
function trimRef(refVals: number[], gotLen: number): number[] {
  return gotLen < refVals.length ? refVals.slice(0, gotLen) : refVals;
}

function computeErrors(got: number[], refVals: number[]): { maxAbsErr: number; maxPctErr: number } {
  const threshold = 1e-10;
  let maxAbs = 0, maxPct = 0;
  for (let i = 0; i < refVals.length; i++) {
    if (!isFinite(got[i])) continue;  // skip NaN/Inf rows (already flagged as unstable)
    const abs = Math.abs(got[i] - refVals[i]);
    if (abs > maxAbs) maxAbs = abs;
    if (Math.abs(refVals[i]) > threshold) {
      const pct = abs / Math.abs(refVals[i]) * 100;
      if (pct > maxPct) maxPct = pct;
    }
  }
  return { maxAbsErr: maxAbs, maxPctErr: maxPct };
}

// ── Timing helper ──────────────────────────────────────────────────────────

const dtypeLabel = (d: DlmDtype) => d;

function isAllFinite(arr: number[] | Float32Array | Float64Array): boolean {
  for (let i = 0; i < arr.length; i++) {
    if (!isFinite((arr as number[])[i])) return false;
  }
  return true;
}

interface TimingResult {
  firstMs: number;
  warmMs: number;
  stable: boolean;   // false if output yhat contains NaN/Infinity
  maxAbsErr: number; // max |dlm-js - octave| across yhat,ystd,x,xstd
  maxPctErr: number; // max |(dlm-js - octave)/octave|×100 (%)
}

async function timedFit(model: Model, combo: Combo): Promise<TimingResult> {
  defaultDevice(combo.backend);
  const { y, s, w, options } = model;

  const { ref, m, n } = refMap[model.label];
  const refVals = flattenRef(ref, m, n);

  let stable = true;

  // First run (JIT compilation)
  const t0 = performance.now();
  let r1: DlmFitResult;
  try {
    r1 = await dlmFit(y, { obsStd: s, processStd: w, dtype: combo.dlmDtype, algorithm: combo.algorithm, ...options });
    if (!isAllFinite(r1.yhat as number[])) stable = false;
    r1[Symbol.dispose]?.();
  } catch {
    return { firstMs: NaN, warmMs: NaN, stable: false, maxAbsErr: NaN, maxPctErr: NaN };
  }
  const t1 = performance.now();

  // Warm run (cached) — also used for error computation
  const t2 = performance.now();
  let r2: DlmFitResult;
  try {
    r2 = await dlmFit(y, { obsStd: s, processStd: w, dtype: combo.dlmDtype, algorithm: combo.algorithm, ...options });
  } catch {
    return { firstMs: t1 - t0, warmMs: NaN, stable: false, maxAbsErr: NaN, maxPctErr: NaN };
  }
  const t3 = performance.now();

  // Compute errors from warm run before dispose
  const gotVals = flattenResult(r2, m, n);
  const { maxAbsErr, maxPctErr } = computeErrors(gotVals, trimRef(refVals, gotVals.length));
  r2[Symbol.dispose]?.();

  return { firstMs: t1 - t0, warmMs: t3 - t2, stable, maxAbsErr, maxPctErr };
}

function fmtMs(ms: number): string {
  if (isNaN(ms)) return '  crash';
  return ms.toFixed(0).padStart(7);
}

function fmtErr(v: number, digits: number): string {
  if (isNaN(v)) return '          -';
  if (v === 0) return '          0';
  return v.toExponential(digits).padStart(11);
}

// ── Run all combinations ───────────────────────────────────────────────────

const allResults: Record<string, unknown>[] = [];

for (const model of models) {
  const colW = { be: 7, dt: 4, al: 14, ti: 8, err: 11 };
  const divW = colW.be + colW.dt + colW.al + colW.ti * 2 + colW.err * 2 + 14;

  console.log(`\n${'═'.repeat(divW)}`);
  console.log(`Model: ${model.label}  (n=${model.n}, m=${model.m})`);
  console.log('═'.repeat(divW));

  const header = [
    'backend'.padEnd(colW.be),
    'dtype'.padEnd(colW.dt),
    'algorithm'.padEnd(colW.al),
    'first(ms)'.padStart(colW.ti),
    'warm(ms)'.padStart(colW.ti),
    '  ' + 'max|Δ|'.padStart(colW.err),
    'max|Δ|%'.padStart(colW.err),
    '  status',
  ].join('  ');
  console.log(header);
  console.log('─'.repeat(divW));

  for (const combo of combos) {
    const result = await timedFit(model, combo);
    const isDefaultAlgo =
      (combo.backend === 'webgpu' && combo.dlmDtype === 'f32' && combo.algorithm === 'assoc') ||
      (combo.backend !== 'webgpu' && combo.algorithm === 'scan');
    const defaultMark = isDefaultAlgo ? ' ←def' : '';
    const status = result.stable
      ? (defaultMark ? `✓${defaultMark}` : '✓')
      : (isNaN(result.firstMs) ? '✗ crash' : '⚠️ NaN');

    console.log([
      combo.backend.padEnd(colW.be),
      dtypeLabel(combo.dlmDtype).padEnd(colW.dt),
      combo.algorithm.padEnd(colW.al),
      fmtMs(result.firstMs),
      fmtMs(result.warmMs),
      fmtErr(result.maxAbsErr, 2),
      fmtErr(result.maxPctErr, 2),
      `  ${status}`,
    ].join('  '));

    allResults.push({
      model: model.label, n: model.n, m: model.m,
      backend: combo.backend,
      dtype: dtypeLabel(combo.dlmDtype),
      algorithm: combo.algorithm,
      firstMs: result.firstMs,
      warmMs: result.warmMs,
      stable: result.stable,
      maxAbsErr: result.maxAbsErr,
      maxPctErr: result.maxPctErr,
    });
  }
}

console.log('');

// ── Save sidecar ───────────────────────────────────────────────────────────

mkdirSync(sidecarDir, { recursive: true });
const sidecarPath = resolve(sidecarDir, "bench-full.json");
writeFileSync(sidecarPath, JSON.stringify({ results: allResults }, null, 2));
console.log(`Wrote ${sidecarPath}`);
