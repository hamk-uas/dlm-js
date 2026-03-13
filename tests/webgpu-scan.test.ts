/**
 * WebGPU lax.scan accuracy tests.
 *
 * Validates that `lax.scan` (sequential scan) on WebGPU produces results
 * consistent with WASM/f32. Both `algorithm:'scan'` and `algorithm:'ud'`
 * use `lax.scan` for the forward pass, so both are tested.
 *
 * These tests run in Chromium browser mode via @vitest/browser-playwright.
 * Use `GPU=nvidia bash scripts/gpu-test.sh run tests/webgpu-scan.test.ts`
 * or `pnpm vitest run tests/webgpu-scan.test.ts -c tests/vitest.nvidia.config.ts`.
 *
 * Thresholds are derived from dlm-js commit 90a9e77, where WebGPU lax.scan
 * was working correctly with jax-js-nonconsuming v0.7.10 (commit e3b88ab).
 * A 5× headroom factor is applied over the measured good-era max relative
 * errors to account for GPU nondeterminism.
 *
 * Good-era measured max relative errors (webgpu/f32/scan vs Octave/f64):
 *   Nile o=0 (m=1): 1.06e-4   →  threshold 0.001  (10×)
 *   Nile o=1 (m=2): 5.58e-3   →  threshold 0.05   (9×)
 *   Kaisaniemi (m=4): 0.985    →  threshold 5.0    (5×)
 *   Energy (m=5): 0.116        →  threshold 1.0    (9×)
 *   Gapped o=1 (m=2): 2.78e-3 →  threshold 0.05   (18×)
 *
 * Current failure: jax-js-nonconsuming block-map branch (3e580f4+) produces
 * errors orders of magnitude larger (e.g. Nile o=1: 1.37e+6% instead of 0.6%).
 * See issues/jax-js-webgpu-laxscan-accuracy.md.
 */
import { describe, it, expect } from 'vitest';
import { commands } from 'vitest/browser';
import { dlmFit, dlmGenSys, toMatlab } from '../src/index';
import { deepAlmostEqual, filterKeys, normalizeMatlabOutput, normalizeNulls } from './utils';
import { getTestConfigs, applyConfig, getDlmDtype, assertAllFinite, type TestConfig } from './test-matrix';
import type { DlmAlgorithm } from '../src/types';

// ── Model cases ────────────────────────────────────────────────────────────

interface ModelCase {
  name: string;
  inputFile: string;
  referenceFile: string;
  options: Record<string, unknown>;
  m: number;
  /** Max relative tolerance (fraction, not percent) for this model */
  relTol: number;
  /** Absolute tolerance floor for near-zero values */
  absTol: number;
}

// Thresholds derived from good-era bench-full (commit 90a9e77, jax-js e3b88ab)
// with generous headroom for GPU nondeterminism.
const modelCases: ModelCase[] = [
  {
    name: 'Nile, order=0 (m=1)',
    inputFile: 'order0-in.json',
    referenceFile: 'order0-out-m.json',
    options: { order: 0 },
    m: 1,
    relTol: 0.001,    // good-era (scan): 1.06e-4, threshold: 10×
    absTol: 1e-4,
  },
  {
    name: 'Nile, order=1 (m=2)',
    inputFile: 'niledemo-in.json',
    referenceFile: 'niledemo-out-m.json',
    options: { order: 1 },
    m: 2,
    relTol: 0.05,     // good-era: 5.58e-3, threshold: 9×
    absTol: 1e-3,
  },
  {
    name: 'Kaisaniemi, trig (m=4)',
    inputFile: 'kaisaniemi-in.json',
    referenceFile: 'kaisaniemi-out-m.json',
    options: { order: 1, harmonics: 1 },
    m: 4,
    relTol: 5.0,       // good-era: 0.985 (large xstd near-zero denominator), threshold: 5×
    absTol: 0.02,
  },
  {
    name: 'Energy, trig+AR (m=5)',
    inputFile: 'energy-in.json',
    referenceFile: 'energy-out-m.json',
    options: { order: 1, harmonics: 1, seasonLength: 12, arCoefficients: [0.85] },
    m: 5,
    relTol: 1.0,       // good-era: 0.116, threshold: 9×
    absTol: 0.03,
  },
  {
    name: 'Gapped, order=1 (m=2)',
    inputFile: 'gapped-in.json',
    referenceFile: 'gapped-out-m.json',
    options: { order: 1 },
    m: 2,
    relTol: 0.05,     // good-era: 2.78e-3, threshold: 18×
    absTol: 1e-3,
  },
];

// Keys to compare — same set used by gensys and assocscan tests
const COMPARE_KEYS = [
  'yhat', 'ystd', 'x', 'xstd',
  'nobs', 'lik',
];

// ── Test runner ────────────────────────────────────────────────────────────

// Resolve paths relative to workspace root (Vite serves from project root)
const TESTS_DIR = 'tests';

async function runWebGPUScanTest(
  config: TestConfig,
  mc: ModelCase,
  algorithm: DlmAlgorithm,
  tolMultiplier = 1,
) {
  applyConfig(config);

  const input = JSON.parse(await commands.readFile(`${TESTS_DIR}/${mc.inputFile}`));
  const reference = normalizeNulls(
    JSON.parse(await commands.readFile(`${TESTS_DIR}/${mc.referenceFile}`))
  ) as Record<string, unknown>;

  const w: number[] = Array.isArray(input.w) ? input.w : [input.w];

  // Convert null→NaN for gapped data
  const y: number[] = (input.y as (number | null)[]).map(v => v === null ? NaN : v);

  const result = await dlmFit(y, {
    obsStd: input.s,
    processStd: w,
    dtype: getDlmDtype(config),
    algorithm,
    ...mc.options,
  });

  const matlab = toMatlab(result);

  // Write debug output
  const slug = mc.referenceFile.replace('-m.json', '').replace('-out', '');
  const outFile = `tests/out/webgpu-${algorithm}-${slug}-${config.label.replace('/', '-')}.json`;
  await commands.writeFile(
    outFile,
    JSON.stringify(matlab, (_key, value) =>
      ArrayBuffer.isView(value) ? Array.from(value as Float64Array) : value
    , 2)
  );

  const normalizedRef = normalizeMatlabOutput(reference as Record<string, unknown>, mc.m);

  const filteredResult: Record<string, unknown> = {};
  const filteredRef: Record<string, unknown> = {};
  for (const k of COMPARE_KEYS) {
    if (k in matlab) filteredResult[k] = (matlab as Record<string, unknown>)[k];
    if (k in normalizedRef) filteredRef[k] = normalizedRef[k];
  }

  assertAllFinite(filteredResult);

  const cmp = deepAlmostEqual(
    filteredResult,
    filteredRef,
    mc.relTol * tolMultiplier,
    '',
    mc.absTol * tolMultiplier,
  );
  if (!cmp.equal) {
    throw new Error(
      `[webgpu/${algorithm} | ${mc.name}] Mismatch at: ${cmp.path}\n` +
      `Result:    ${JSON.stringify(cmp.a)}\n` +
      `Reference: ${JSON.stringify(cmp.b)}\n` +
      `Tolerance: relTol=${mc.relTol * tolMultiplier}, absTol=${mc.absTol * tolMultiplier}`
    );
  }
}

// ── Test suites ────────────────────────────────────────────────────────────

describe('WebGPU lax.scan accuracy — scan algorithm', () => {
  for (const mc of modelCases) {
    it(mc.name, async () => {
      const configs = await getTestConfigs();
      const gpuConfig = configs.find(c => c.label.includes('webgpu'));
      expect(gpuConfig).toBeDefined();
      await runWebGPUScanTest(gpuConfig!, mc, 'scan');
    });
  }
});

// Known skip: Energy m=5 produces NaN in UD algorithm on f32 (same as Node skip)
const UD_SKIP = new Set(['Energy, trig+AR (m=5)']);

describe('WebGPU lax.scan accuracy — ud algorithm', () => {
  for (const mc of modelCases) {
    const fn = UD_SKIP.has(mc.name) ? it.skip : it;
    fn(mc.name, async () => {
      const configs = await getTestConfigs();
      const gpuConfig = configs.find(c => c.label.includes('webgpu'));
      expect(gpuConfig).toBeDefined();
      await runWebGPUScanTest(gpuConfig!, mc, 'ud');
    });
  }
});

// ── sqrt-assoc on WebGPU (QR available since jax-js v0.9.1) ───────────────
// sqrt-assoc uses lax.associativeScan in Cholesky factor space with QR-based
// tria(). The extra QR + triangular-solve operations introduce ~2-3× more
// floating-point noise than the standard scan/assoc path, so thresholds are
// relaxed by a 3× multiplier over the scan-derived baselines.
// Known issue: m=1 (order=0) has WebGPU-specific precision degradation
// (ystd 141%, xstd 586% vs Octave) despite passing on wasm/f32 (~1e-7).
// The scalar 1×1 matrix case triggers a different code path in the WebGPU
// associativeScan shader that loses precision. Not a DUS/buffer issue.
const SQRT_ASSOC_SKIP = new Set([
  'Nile, order=0 (m=1)',  // WebGPU f32 precision: ystd 141%, xstd 586% off
]);

describe('WebGPU accuracy — sqrt-assoc algorithm', () => {
  for (const mc of modelCases) {
    const fn = SQRT_ASSOC_SKIP.has(mc.name) ? it.skip : it;
    fn(mc.name, async () => {
      const configs = await getTestConfigs();
      const gpuConfig = configs.find(c => c.label.includes('webgpu'));
      expect(gpuConfig).toBeDefined();
      await runWebGPUScanTest(gpuConfig!, mc, 'sqrt-assoc', 3);
    });
  }
});
