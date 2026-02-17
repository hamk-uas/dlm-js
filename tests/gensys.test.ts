import { checkLeaks } from '@jax-js-nonconsuming/jax';
import { describe, it, expect } from 'vitest';
import { dlmFit, dlmgensys } from '../src/index';
import { deepAlmostEqual } from './utils';
import { getTestConfigs, applyConfig, getModelTolerances, assertAllFinite, type TestConfig } from './test-matrix';
import type { DlmOptions } from '../src/dlmgensys';
import * as fs from 'fs';
import * as path from 'path';

// ─── dlmgensys unit tests (pure math, no device/dtype needed) ───────────────

describe('dlmgensys', () => {
  it('order=0: local level (m=1)', () => {
    const sys = dlmgensys({ order: 0 });
    expect(sys.m).toBe(1);
    expect(sys.G).toEqual([[1]]);
    expect(sys.F).toEqual([1]);
  });

  it('order=1: local linear trend (m=2)', () => {
    const sys = dlmgensys({ order: 1 });
    expect(sys.m).toBe(2);
    expect(sys.G).toEqual([[1, 1], [0, 1]]);
    expect(sys.F).toEqual([1, 0]);
  });

  it('order=2: quadratic trend (m=3)', () => {
    const sys = dlmgensys({ order: 2 });
    expect(sys.m).toBe(3);
    expect(sys.G).toEqual([[1, 1, 0], [0, 1, 1], [0, 0, 1]]);
    expect(sys.F).toEqual([1, 0, 0]);
  });

  it('default is order=1', () => {
    const sys = dlmgensys();
    expect(sys.m).toBe(2);
    expect(sys.G).toEqual([[1, 1], [0, 1]]);
  });

  it('fullseas=true, ns=12: 11 seasonal states (m=13)', () => {
    const sys = dlmgensys({ order: 1, fullseas: true, ns: 12 });
    expect(sys.m).toBe(13);
    expect(sys.F[0]).toBe(1);
    expect(sys.F[1]).toBe(0);
    expect(sys.F[2]).toBe(1);
    expect(sys.G[2][2]).toBe(-1);
    expect(sys.G[2][3]).toBe(-1);
    expect(sys.G[3][2]).toBe(1);
    expect(sys.G[3][3]).toBe(0);
  });

  it('trig=2, ns=12: 4 harmonic states (m=6)', () => {
    const sys = dlmgensys({ order: 1, trig: 2, ns: 12 });
    expect(sys.m).toBe(6);
    expect(sys.F).toEqual([1, 0, 1, 0, 1, 0]);
    expect(sys.G[2][2]).toBeCloseTo(Math.cos(2 * Math.PI / 12), 10);
    expect(sys.G[2][3]).toBeCloseTo(Math.sin(2 * Math.PI / 12), 10);
    expect(sys.G[3][2]).toBeCloseTo(-Math.sin(2 * Math.PI / 12), 10);
    expect(sys.G[3][3]).toBeCloseTo(Math.cos(2 * Math.PI / 12), 10);
  });

  it('trig > ns/2 throws', () => {
    expect(() => dlmgensys({ trig: 7, ns: 12 })).toThrow();
  });

  it('trig=ns/2 removes redundant last element', () => {
    const sys = dlmgensys({ order: 1, trig: 6, ns: 12 });
    expect(sys.m).toBe(13);
  });

  it('arphi: AR(1) adds 1 state', () => {
    const sys = dlmgensys({ order: 1, arphi: [0.8] });
    expect(sys.m).toBe(3);
    expect(sys.G[2][2]).toBeCloseTo(0.8);
    expect(sys.F[2]).toBe(1);
  });

  it('arphi: AR(2) adds 2 states', () => {
    const sys = dlmgensys({ order: 1, arphi: [0.5, 0.3] });
    expect(sys.m).toBe(4);
    expect(sys.G[2][2]).toBeCloseTo(0.5);
    expect(sys.G[2][3]).toBeCloseTo(1);
    expect(sys.G[3][2]).toBeCloseTo(0.3);
    expect(sys.G[3][3]).toBeCloseTo(0);
  });
});

// ─── Helper: normalize MATLAB output for comparison ─────────────────────────

/**
 * Normalize MATLAB JSON output to match JS output format.
 * MATLAB collapses 1×1 matrices to scalars; JS always uses arrays.
 */
function normalizeMatlabOutput(
  obj: Record<string, unknown>,
  m: number,
): Record<string, unknown> {
  const result = { ...obj };

  if (typeof result.G === 'number') result.G = [[result.G]];
  if (typeof result.F === 'number') result.F = [result.F];
  if (typeof result.W === 'number') result.W = [[result.W]];
  if (typeof result.x0 === 'number') result.x0 = [result.x0];
  if (typeof result.C0 === 'number') result.C0 = [[result.C0]];

  if (m === 1 && Array.isArray(result.xf) && typeof result.xf[0] === 'number') {
    result.xf = [result.xf];
  }
  if (m === 1 && Array.isArray(result.x) && typeof result.x[0] === 'number') {
    result.x = [result.x];
  }
  if (m === 1 && Array.isArray(result.Cf) && typeof result.Cf[0] === 'number') {
    result.Cf = [[result.Cf]];
  }
  if (m === 1 && Array.isArray(result.C) && typeof result.C[0] === 'number') {
    result.C = [[result.C]];
  }
  if (m === 1 && Array.isArray(result.xstd) && typeof result.xstd[0] === 'number') {
    result.xstd = (result.xstd as number[]).map((v: number) => [v]);
  }

  // Remove MATLAB-only fields not in JS output
  for (const k of ['options', 's', 'ss', 'xr', 'xrd', 'xrp', 'yrp']) {
    delete result[k];
  }

  return result;
}

// ─── Integration tests against Octave references ────────────────────────────

interface ModelCase {
  name: string;
  inputFile: string;
  referenceFile: string;
  options: DlmOptions;
}

const modelCases: ModelCase[] = [
  {
    name: 'order=0 (local level on Nile data)',
    inputFile: 'order0-in.json',
    referenceFile: 'order0-out-m.json',
    options: { order: 0 },
  },
  {
    name: 'order=0 minimal (m=1, n=50)',
    inputFile: 'level-in.json',
    referenceFile: 'level-out-m.json',
    options: { order: 0 },
  },
  {
    name: 'order=2 (quadratic trend)',
    inputFile: 'order2-in.json',
    referenceFile: 'order2-out-m.json',
    options: { order: 2 },
  },
  {
    name: 'fullseas=1, ns=12 (full seasonal)',
    inputFile: 'seasonal-in.json',
    referenceFile: 'seasonal-out-m.json',
    options: { order: 1, fullseas: true, ns: 12 },
  },
  {
    name: 'trig=2, ns=12 (trigonometric seasonal, synthetic)',
    inputFile: 'trig-in.json',
    referenceFile: 'trig-out-m.json',
    options: { order: 1, trig: 2, ns: 12 },
  },
  {
    name: 'Kaisaniemi seasonal demo (order=1, trig=1)',
    inputFile: 'kaisaniemi-in.json',
    referenceFile: 'kaisaniemi-out-m.json',
    options: { order: 1, trig: 1 },
  },
];

/** Keys to compare (subset that both JS and MATLAB produce) */
const COMPARE_KEYS = [
  'xf', 'Cf', 'x', 'C', 'xstd',
  'G', 'F', 'W',
  'y', 'V', 'x0', 'C0',
  'yhat', 'ystd', 'resid0', 'resid', 'resid2',
  'ssy', 'v', 'Cp', 's2',
  'nobs', 'lik', 'mse', 'mape',
  'class',
];

describe('dlmgensys integration tests', async () => {
  const configs = await getTestConfigs();

  for (const config of configs) {
    describe(config.label, () => {
      for (const mc of modelCases) {
        it(mc.name, async () => {
          const sys = dlmgensys(mc.options);
          const tol = getModelTolerances(config, sys.m);
          if (!tol) {
            // Float32 + large state space: Kalman filter is numerically
            // unstable in single precision (covariance goes negative → NaN).
            return;
          }

          applyConfig(config);

          const inputPath = path.join(__dirname, mc.inputFile);
          const refPath = path.join(__dirname, mc.referenceFile);
          if (!fs.existsSync(inputPath)) throw new Error(`Input not found: ${inputPath}`);
          if (!fs.existsSync(refPath)) throw new Error(`Reference not found: ${refPath}`);

          const input = JSON.parse(fs.readFileSync(inputPath, 'utf-8'));
          const reference = JSON.parse(fs.readFileSync(refPath, 'utf-8'));

          const w: number[] = Array.isArray(input.w) ? input.w : [input.w];

          checkLeaks.start();
          const result = await dlmFit(input.y, input.s, w, config.dtype, mc.options);
          checkLeaks.stop();

          // Write debug output
          const outputDir = path.join(__dirname, 'out');
          if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });
          fs.writeFileSync(
            path.join(outputDir, mc.referenceFile.replace('-m.json', `-${config.label.replace('/', '-')}.json`)),
            JSON.stringify(result, (_key, value) =>
              ArrayBuffer.isView(value) ? Array.from(value as Float64Array) : value
            , 2)
          );

          const normalizedRef = normalizeMatlabOutput(reference, sys.m);

          const keysToCompare = COMPARE_KEYS;

          const filteredResult: Record<string, unknown> = {};
          const filteredRef: Record<string, unknown> = {};
          for (const k of keysToCompare) {
            if (k in result) filteredResult[k] = (result as Record<string, unknown>)[k];
            if (k in normalizedRef) filteredRef[k] = normalizedRef[k];
          }

          assertAllFinite(filteredResult);

          const cmp = deepAlmostEqual(
            filteredResult,
            filteredRef,
            tol.relativeTolerance,
            '',
            tol.absoluteTolerance,
          );
          if (!cmp.equal) {
            throw new Error(
              `[${config.label} | ${mc.name}] Mismatch at: ${cmp.path}\n` +
              `Result:    ${JSON.stringify(cmp.a)}\n` +
              `Reference: ${JSON.stringify(cmp.b)}`
            );
          }
        });
      }
    });
  }
});
