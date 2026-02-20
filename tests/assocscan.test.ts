/**
 * Associative scan (parallel filter + smoother) validation against Octave references.
 *
 * The exact O(log N) associative scan path (Särkkä & García-Fernández 2020)
 * is normally only active on webgpu+Float32. These tests force it on via
 * `forceAssocScan: true` so we can validate the algorithm against Octave
 * ground truth using wasm/f64 (full precision, no GPU needed).
 *
 * Covers the same model suite as gensys.test.ts: local level, linear trend,
 * quadratic, seasonal, trigonometric, AR, and missing data.
 */
import { describe, it } from 'vitest';
import { dlmFit, dlmGenSys } from '../src/index';
import { deepAlmostEqual, filterKeys, normalizeMatlabOutput, normalizeNulls, withLeakCheck } from './utils';
import { getTestConfigs, applyConfig, getModelTolerances, assertAllFinite, type TestConfig } from './test-matrix';
import type { DlmOptions } from '../src/dlmgensys';
import * as fs from 'fs';
import * as path from 'path';

// ── Model cases (same as gensys.test.ts) ───────────────────────────────────

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
  {
    name: 'trig=1, ns=12, arphi=0.7 (seasonal + AR)',
    inputFile: 'trigar-in.json',
    referenceFile: 'trigar-out-m.json',
    options: { order: 1, trig: 1, ns: 12, arphi: [0.7] },
  },
  {
    name: 'synthetic energy demand (trend + seasonal + strong AR)',
    inputFile: 'energy-in.json',
    referenceFile: 'energy-out-m.json',
    options: { order: 1, trig: 1, ns: 12, arphi: [0.85] },
  },
  {
    name: 'synthetic AR(2) (damped oscillation)',
    inputFile: 'ar2-in.json',
    referenceFile: 'ar2-out-m.json',
    options: { order: 1, arphi: [0.6, -0.3] },
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

// ── Integration tests: assocScan vs Octave ─────────────────────────────────

describe('associativeScan dlmFit vs Octave', async () => {
  const configs = await getTestConfigs();
  // Only test Float64 configs — Float32 has known precision limits for m > 2
  const f64Configs = configs.filter(c => c.label.includes('f64'));

  for (const config of f64Configs) {
    describe(`assocScan / ${config.label}`, () => {
      for (const mc of modelCases) {
        it(mc.name, async () => {
          const sys = dlmGenSys(mc.options);
          const tol = getModelTolerances(config, sys.m);
          if (!tol) return;

          applyConfig(config);

          const inputPath = path.join(__dirname, mc.inputFile);
          const refPath = path.join(__dirname, mc.referenceFile);
          if (!fs.existsSync(inputPath)) throw new Error(`Input not found: ${inputPath}`);
          if (!fs.existsSync(refPath)) throw new Error(`Reference not found: ${refPath}`);

          const input = JSON.parse(fs.readFileSync(inputPath, 'utf-8'));
          const reference = JSON.parse(fs.readFileSync(refPath, 'utf-8'));

          const w: number[] = Array.isArray(input.w) ? input.w : [input.w];

          const result = await withLeakCheck(() =>
            dlmFit(input.y, input.s, w, config.dtype, mc.options, undefined, true)
          );

          // Write debug output
          const outputDir = path.join(__dirname, 'out');
          if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });
          fs.writeFileSync(
            path.join(outputDir, mc.referenceFile.replace('-m.json', `-assoc-${config.label.replace('/', '-')}.json`)),
            JSON.stringify(result, (_key, value) =>
              ArrayBuffer.isView(value) ? Array.from(value as Float64Array) : value
            , 2)
          );

          const normalizedRef = normalizeMatlabOutput(reference, sys.m);

          const filteredResult: Record<string, unknown> = {};
          const filteredRef: Record<string, unknown> = {};
          for (const k of COMPARE_KEYS) {
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
              `[assocScan / ${config.label} | ${mc.name}] Mismatch at: ${cmp.path}\n` +
              `Result:    ${JSON.stringify(cmp.a)}\n` +
              `Reference: ${JSON.stringify(cmp.b)}`
            );
          }
        });
      }
    });
  }
});

// ── Niledemo: assocScan vs Octave ──────────────────────────────────────────

describe('associativeScan niledemo vs Octave', async () => {
  const configs = await getTestConfigs();
  const f64Configs = configs.filter(c => c.label.includes('f64'));

  const inputFile = path.join(__dirname, 'niledemo-in.json');
  const refFile = path.join(__dirname, 'niledemo-out-m.json');
  if (!fs.existsSync(inputFile)) throw new Error(`Input not found — run: pnpm run test:octave`);
  if (!fs.existsSync(refFile)) throw new Error(`Reference not found — run: pnpm run test:octave`);

  const nileInput = JSON.parse(fs.readFileSync(inputFile, 'utf-8'));
  const reference = JSON.parse(fs.readFileSync(refFile, 'utf-8'));

  const keysFile = path.join(__dirname, 'niledemo-keys.json');
  const compareKeys: string[] | null = fs.existsSync(keysFile)
    ? JSON.parse(fs.readFileSync(keysFile, 'utf-8'))
    : null;

  for (const config of f64Configs) {
    it(`should match reference — assocScan (${config.label})`, async () => {
      applyConfig(config);

      const result = await withLeakCheck(() =>
        dlmFit(nileInput.y, nileInput.s, nileInput.w, config.dtype, {}, undefined, true)
      );

      const outputDir = path.join(__dirname, 'out');
      if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });
      fs.writeFileSync(
        path.join(outputDir, `niledemo-out-assoc-${config.label.replace('/', '-')}.json`),
        JSON.stringify(result, (_key, value) =>
          ArrayBuffer.isView(value) ? Array.from(value as Float64Array) : value
        , 2)
      );

      let filteredResult: Record<string, unknown> = result as unknown as Record<string, unknown>;
      let filteredReference: Record<string, unknown> = reference;
      if (compareKeys) {
        filteredResult = filterKeys(result, compareKeys) as Record<string, unknown>;
        filteredReference = filterKeys(reference, compareKeys) as Record<string, unknown>;
      }

      const cmp = deepAlmostEqual(
        filteredResult,
        filteredReference,
        config.relativeTolerance,
        '',
        config.absoluteTolerance,
      );
      if (!cmp.equal) {
        throw new Error(
          `[assocScan / ${config.label}] Niledemo mismatch at: ${cmp.path}\n` +
          `Result:    ${JSON.stringify(cmp.a)}\n` +
          `Reference: ${JSON.stringify(cmp.b)}`
        );
      }
    });
  }
});

// ── Missing data: assocScan vs Octave ──────────────────────────────────────

describe('associativeScan missing data vs Octave', async () => {
  const configs = await getTestConfigs();
  const f64Configs = configs.filter(c => c.label.includes('f64'));

  const inFileA = path.join(__dirname, 'missing-in.json');
  const refFileA = path.join(__dirname, 'missing-out-m.json');
  const inFileB = path.join(__dirname, 'missing-order0-in.json');
  const refFileB = path.join(__dirname, 'missing-order0-out-m.json');

  if (!fs.existsSync(inFileA) || !fs.existsSync(refFileA) ||
      !fs.existsSync(inFileB) || !fs.existsSync(refFileB)) {
    throw new Error('Missing data reference files not found — run: pnpm run test:octave');
  }

  const inpA = JSON.parse(fs.readFileSync(inFileA, 'utf-8'));
  const refA = normalizeNulls(JSON.parse(fs.readFileSync(refFileA, 'utf-8'))) as Record<string, unknown>;
  const inpB = JSON.parse(fs.readFileSync(inFileB, 'utf-8'));
  const refB = normalizeNulls(JSON.parse(fs.readFileSync(refFileB, 'utf-8'))) as Record<string, unknown>;

  const y_missing = inpA['y'] as (number | null)[];
  const s = inpA['s'] as number;
  const w = inpA['w'] as number[];
  const w_level = (inpB['w'] instanceof Array ? inpB['w'][0] : inpB['w']) as number;

  const MISSING_KEYS = ['x', 'xstd', 'yhat', 'ystd', 'nobs'];

  for (const config of f64Configs) {
    it(`order=1 (m=2) should match Octave — assocScan (${config.label})`, async () => {
      applyConfig(config);

      const result = await withLeakCheck(() =>
        dlmFit(
          Float64Array.from(y_missing.map(v => (v === null ? NaN : v))),
          s, w, config.dtype, { order: 1 }, undefined, true
        )
      );

      const normalizedRef = normalizeMatlabOutput(refA, 2);
      const filteredResult = filterKeys(result, MISSING_KEYS) as Record<string, unknown>;
      const filteredRef = filterKeys(normalizedRef, MISSING_KEYS) as Record<string, unknown>;

      const cmp = deepAlmostEqual(
        filteredResult,
        filteredRef,
        config.relativeTolerance,
        '',
        config.absoluteTolerance,
      );
      if (!cmp.equal) {
        throw new Error(
          `[assocScan / ${config.label}] Missing order=1 mismatch at: ${cmp.path}\n` +
          `Result:    ${JSON.stringify(cmp.a)}\n` +
          `Reference: ${JSON.stringify(cmp.b)}`
        );
      }
    });

    it(`order=0 (m=1) should match Octave — assocScan (${config.label})`, async () => {
      applyConfig(config);

      const result = await withLeakCheck(() =>
        dlmFit(
          Float64Array.from(y_missing.map(v => (v === null ? NaN : v))),
          s, [w_level], config.dtype, { order: 0 }, undefined, true
        )
      );

      const normalizedRef = normalizeMatlabOutput(refB, 1);
      const filteredResult = filterKeys(result, MISSING_KEYS) as Record<string, unknown>;
      const filteredRef = filterKeys(normalizedRef, MISSING_KEYS) as Record<string, unknown>;

      const cmp = deepAlmostEqual(
        filteredResult,
        filteredRef,
        config.relativeTolerance,
        '',
        config.absoluteTolerance,
      );
      if (!cmp.equal) {
        throw new Error(
          `[assocScan / ${config.label}] Missing order=0 mismatch at: ${cmp.path}\n` +
          `Result:    ${JSON.stringify(cmp.a)}\n` +
          `Reference: ${JSON.stringify(cmp.b)}`
        );
      }
    });
  }
});
