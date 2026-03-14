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
 * Runs in Chromium browser mode via @vitest/browser-playwright:
 *   GPU=nvidia bash scripts/gpu-test.sh run scripts/bench-full.ts
 *
 * Output: assets/timings/bench-full.json
 */

import { describe, it } from 'vitest';
import { commands } from 'vitest/browser';
import { DType, defaultDevice, init } from "@hamk-uas/jax-js-nonconsuming";
import { dlmFit } from "../src/index.ts";
import type { DlmDtype, DlmAlgorithm, DlmFitResult, DlmStabilization } from "../src/types.ts";

/** Hard timeout per dlmFit call — skip the combo if it exceeds this. */
const TIMEOUT_MS = 10_000;

async function readJSON(path: string): Promise<any> {
  return JSON.parse(await commands.readFile(path));
}

// ── Types ──────────────────────────────────────────────────────────────────

type RefJson = Record<string, number[] | number[][]>;

interface Model {
  label: string;
  y: number[];
  s: number | number[];
  w: number[];
  options: Record<string, unknown>;
  n: number;
  m: number;
}

interface Combo {
  backend: 'cpu' | 'wasm' | 'webgpu';
  dlmDtype: DlmDtype;
  algorithm: DlmAlgorithm;
  stabilization?: DlmStabilization;
  stabLabel?: string;
}

interface TimingResult {
  firstMs: number;
  warmMs: number;
  stable: boolean;
  maxAbsErr: number;
  maxPctErr: number;
}

// ── Pure helpers (no I/O) ──────────────────────────────────────────────────

const toW = (v: unknown): number[] => Array.isArray(v) ? v as number[] : [v as number];
const dtypeLabel = (d: DlmDtype) => d;

function isAllFinite(arr: number[] | Float32Array | Float64Array): boolean {
  for (let i = 0; i < arr.length; i++) {
    if (!isFinite((arr as number[])[i])) return false;
  }
  return true;
}

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

function trimRef(refVals: number[], gotLen: number): number[] {
  return gotLen < refVals.length ? refVals.slice(0, gotLen) : refVals;
}

function computeErrors(got: number[], refVals: number[]): { maxAbsErr: number; maxPctErr: number } {
  const threshold = 1e-10;
  let maxAbs = 0, maxPct = 0;
  for (let i = 0; i < refVals.length; i++) {
    if (!isFinite(got[i])) continue;
    const abs = Math.abs(got[i] - refVals[i]);
    if (abs > maxAbs) maxAbs = abs;
    if (Math.abs(refVals[i]) > threshold) {
      const pct = abs / Math.abs(refVals[i]) * 100;
      if (pct > maxPct) maxPct = pct;
    }
  }
  return { maxAbsErr: maxAbs, maxPctErr: maxPct };
}

function fmtMs(ms: number): string {
  if (isNaN(ms)) return '  crash';
  if (!isFinite(ms)) return '   >5 s';
  return ms.toFixed(0).padStart(7);
}

function fmtErr(v: number, digits: number): string {
  if (isNaN(v)) return '          -';
  if (v === 0) return '          0';
  return v.toExponential(digits).padStart(11);
}

// ── Combination generation ─────────────────────────────────────────────────

function buildCombos(): Combo[] {
  const combos: Combo[] = [];
  for (const backend of ['cpu', 'wasm'] as const) {
    for (const dlmDtype of ['f64', 'f32'] as const) {
      for (const algorithm of ['scan', 'assoc', 'sqrt-assoc', 'ud'] as const) {
        combos.push({ backend, dlmDtype, algorithm });
        if (dlmDtype === 'f64' && algorithm === 'scan') {
          combos.push({ backend, dlmDtype, algorithm, stabilization: { cTriuSym: false }, stabLabel: 'off' });
        }
        if (dlmDtype === 'f32' && algorithm === 'scan') {
          combos.push({ backend, dlmDtype, algorithm, stabilization: { cTriuSym: true }, stabLabel: 'joseph+triu' });
        }
      }
    }
  }
  for (const algorithm of ['scan', 'assoc', 'sqrt-assoc', 'ud'] as const) {
    combos.push({ backend: 'webgpu', dlmDtype: 'f32', algorithm });
  }
  combos.push({ backend: 'webgpu', dlmDtype: 'f32', algorithm: 'scan', stabilization: { cTriuSym: true }, stabLabel: 'joseph+triu' });
  return combos;
}

// ── Main benchmark (vitest test body) ──────────────────────────────────────

describe('bench-full', () => {
  it('comprehensive dlmFit benchmark', async () => {
    // Init backends
    await init("webgpu");

    // Load data
    const nileIn       = await readJSON("tests/niledemo-in.json");
    const kaisaniemiIn = await readJSON("tests/kaisaniemi-in.json");
    const trigarIn     = await readJSON("tests/trigar-in.json");
    const order0In     = await readJSON("tests/order0-in.json");
    const gappedIn     = await readJSON("tests/gapped-in.json");
    const gappedY: number[] = (gappedIn.y as (number | null)[]).map((v: number | null) => v === null ? NaN : v);

    // Load Octave references
    const refMap: Record<string, { ref: RefJson; m: number; n: number }> = {
      "Nile, order=0":    { ref: await readJSON("tests/order0-out-m.json"),     m: 1, n: 100 },
      "Nile, order=1":    { ref: await readJSON("tests/niledemo-out-m.json"),   m: 2, n: 100 },
      "Kaisaniemi, trig": { ref: await readJSON("tests/kaisaniemi-out-m.json"), m: 4, n: 117 },
      "Energy, trig+AR":  { ref: await readJSON("tests/trigar-out-m.json"),     m: 5, n: 120 },
      "Gapped, order=1":  { ref: await readJSON("tests/gapped-out-m.json"),     m: 2, n: 100 },
    };

    // Build models
    const models: Model[] = [
      { label: "Nile, order=0",   y: order0In.y, s: order0In.s, w: toW(order0In.w), options: { order: 0 }, n: 100, m: 1 },
      { label: "Nile, order=1",   y: nileIn.y,   s: nileIn.s,   w: toW(nileIn.w),   options: { order: 1 }, n: 100, m: 2 },
      { label: "Kaisaniemi, trig", y: kaisaniemiIn.y, s: kaisaniemiIn.s, w: toW(kaisaniemiIn.w), options: { order: 1, harmonics: 1 }, n: 117, m: 4 },
      { label: "Energy, trig+AR", y: trigarIn.y, s: trigarIn.s, w: toW(trigarIn.w), options: { order: 1, harmonics: 1, seasonLength: 12, arCoefficients: [0.7] }, n: 120, m: 5 },
      { label: "Gapped, order=1", y: gappedY, s: gappedIn.s, w: toW(gappedIn.w), options: gappedIn.options, n: 100, m: 2 },
    ];

    // Timing helper
    async function timedFit(model: Model, combo: Combo): Promise<TimingResult> {
      try { defaultDevice(combo.backend); } catch {
        return { firstMs: NaN, warmMs: NaN, stable: false, maxAbsErr: NaN, maxPctErr: NaN };
      }
      const { y, s, w, options } = model;
      const { ref, m, n } = refMap[model.label];
      const refVals = flattenRef(ref, m, n);
      let stable = true;
      const fitOpts = {
        obsStd: s, processStd: w,
        dtype: combo.dlmDtype, algorithm: combo.algorithm,
        ...(combo.stabilization !== undefined ? { stabilization: combo.stabilization } : {}),
        ...options,
      };

      const t0 = performance.now();
      let r1: DlmFitResult;
      try {
        r1 = await dlmFit(y, fitOpts);
        if (!isAllFinite(r1.yhat as number[])) stable = false;
        r1[Symbol.dispose]?.();
      } catch {
        return { firstMs: NaN, warmMs: NaN, stable: false, maxAbsErr: NaN, maxPctErr: NaN };
      }
      const firstMs = performance.now() - t0;
      if (firstMs > TIMEOUT_MS) return { firstMs, warmMs: Infinity, stable, maxAbsErr: NaN, maxPctErr: NaN };

      const t2 = performance.now();
      let r2: DlmFitResult;
      try { r2 = await dlmFit(y, fitOpts); } catch {
        return { firstMs, warmMs: NaN, stable: false, maxAbsErr: NaN, maxPctErr: NaN };
      }
      const warmMs = performance.now() - t2;
      if (warmMs > TIMEOUT_MS) { r2[Symbol.dispose]?.(); return { firstMs, warmMs: Infinity, stable, maxAbsErr: NaN, maxPctErr: NaN }; }

      const gotVals = flattenResult(r2, m, n);
      if (!isAllFinite(gotVals)) stable = false;
      const { maxAbsErr, maxPctErr } = computeErrors(gotVals, trimRef(refVals, gotVals.length));
      r2[Symbol.dispose]?.();
      return { firstMs, warmMs, stable, maxAbsErr, maxPctErr };
    }

    // Run all combinations
    const combos = buildCombos();
    const allResults: Record<string, unknown>[] = [];

    for (const model of models) {
      const colW = { be: 7, dt: 4, al: 14, ti: 8, err: 11 };
      const divW = colW.be + colW.dt + colW.al + colW.ti * 2 + colW.err * 2 + 14;

      console.log(`\n${'═'.repeat(divW)}`);
      console.log(`Model: ${model.label}  (n=${model.n}, m=${model.m})`);
      console.log('═'.repeat(divW));

      const header = [
        'backend'.padEnd(colW.be), 'dtype'.padEnd(colW.dt), 'algorithm'.padEnd(colW.al),
        'first(ms)'.padStart(colW.ti), 'warm(ms)'.padStart(colW.ti),
        '  ' + 'max|Δ|'.padStart(colW.err), 'max|Δ|%'.padStart(colW.err), '  status',
      ].join('  ');
      console.log(header);
      console.log('─'.repeat(divW));

      for (const combo of combos) {
        const result = await timedFit(model, combo);
        const isDefaultCombo = combo.stabilization === undefined &&
          ((combo.backend === 'webgpu' && combo.dlmDtype === 'f32' && combo.algorithm === 'assoc') ||
           (combo.backend !== 'webgpu' && combo.algorithm === 'scan'));
        const defaultMark = isDefaultCombo ? ' ←def' : '';
        const status = result.stable
          ? (defaultMark ? `✓${defaultMark}` : '✓')
          : (isNaN(result.firstMs) ? '✗ crash' : '⚠️ NaN');
        const stabStr = combo.stabLabel ?? (combo.stabilization !== undefined ? 'override' : 'default');

        console.log([
          combo.backend.padEnd(colW.be), dtypeLabel(combo.dlmDtype).padEnd(colW.dt),
          `${combo.algorithm}+${stabStr}`.padEnd(colW.al),
          fmtMs(result.firstMs), fmtMs(result.warmMs),
          fmtErr(result.maxAbsErr, 2), fmtErr(result.maxPctErr, 2),
          `  ${status}`,
        ].join('  '));

        allResults.push({
          model: model.label, n: model.n, m: model.m,
          backend: combo.backend, dtype: dtypeLabel(combo.dlmDtype),
          algorithm: combo.algorithm, stabLabel: combo.stabLabel ?? 'default',
          firstMs: result.firstMs, warmMs: result.warmMs, stable: result.stable,
          maxAbsErr: result.maxAbsErr, maxPctErr: result.maxPctErr,
        });
      }
    }

    // Save sidecar
    await commands.writeFile("assets/timings/bench-full.json", JSON.stringify({ results: allResults }, null, 2));
    console.log('\nWrote assets/timings/bench-full.json');
  });
});
