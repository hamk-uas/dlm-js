/**
 * MLE parameter estimation tests for dlm-js.
 *
 * Tests dlmMLE with:
 *   1. Basic s/w recovery (Nile-like order=0 model)
 *   2. AR coefficient fitting (order=0 + AR(1), params.arCoefficients.fit=true)
 *   3. Fixed-AR baseline (AR coefficients stay fixed)
 *
 * Uses WASM backend + Float64 for all tests.
 * Synthetic data from a deterministic PRNG guarantees reproducibility.
 */
import { defaultDevice, numpy as np } from '@hamk-uas/jax-js-nonconsuming';
import { describe, it, expect } from 'vitest';
import { dlmMLE, dlmGenSys, findArInds, dlmPrior, stuffIntegerTimestamps } from '../src/index';
import type { DlmLossFn } from '../src/index';

// ─── Deterministic PRNG (same as synthetic.test.ts) ─────────────────────────

function mulberry32(seed: number): () => number {
  let a = seed | 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianRng(uniform: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u1: number;
    do { u1 = uniform(); } while (u1 === 0);
    const u2 = uniform();
    const mag = Math.sqrt(-2 * Math.log(u1));
    spare = mag * Math.sin(2 * Math.PI * u2);
    return mag * Math.cos(2 * Math.PI * u2);
  };
}

/** Generate synthetic DLM data from known parameters. */
function generateData(
  G: number[][], F: number[], s: number, w: number[], n: number, seed: number,
): number[] {
  const m = G.length;
  const randn = gaussianRng(mulberry32(seed));
  const x = new Array(m).fill(0);
  x[0] = randn() * 10;
  for (let k = 1; k < m; k++) x[k] = randn();
  const y: number[] = [];
  for (let t = 0; t < n; t++) {
    const x_new = new Array(m).fill(0);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) x_new[i] += G[i][j] * x[j];
      if (i < w.length) x_new[i] += w[i] * randn();
    }
    let obs = 0;
    for (let k = 0; k < m; k++) obs += F[k] * x_new[k];
    y.push(obs + s * randn());
    for (let k = 0; k < m; k++) x[k] = x_new[k];
  }
  return y;
}

function mleParams(spec: {
  obsStdInit?: number;
  obsStdFixed?: number | ArrayLike<number>;
  processStdInit?: number[];
  processStdGroups?: ArrayLike<string | number>;
  processStdFixed?: number | ArrayLike<number | null | undefined> | Partial<Record<number, number>>;
  fitAr?: boolean;
  arInit?: number[];
  arFixed?: number | ArrayLike<number | null | undefined> | Partial<Record<number, number>>;
} = {}) {
  const params: Record<string, unknown> = {};

  if (spec.obsStdInit !== undefined || spec.obsStdFixed !== undefined) {
    params.obsStd = {
      ...(spec.obsStdInit !== undefined ? { init: spec.obsStdInit } : {}),
      ...(spec.obsStdFixed !== undefined ? { fixed: spec.obsStdFixed } : {}),
    };
  }

  if (spec.processStdInit !== undefined || spec.processStdGroups !== undefined || spec.processStdFixed !== undefined) {
    params.processStd = {
      ...(spec.processStdInit !== undefined ? { init: spec.processStdInit } : {}),
      ...(spec.processStdGroups !== undefined ? { groups: spec.processStdGroups } : {}),
      ...(spec.processStdFixed !== undefined ? { fixed: spec.processStdFixed } : {}),
    };
  }

  if (spec.fitAr !== undefined || spec.arInit !== undefined || spec.arFixed !== undefined) {
    params.arCoefficients = {
      ...(spec.fitAr !== undefined ? { fit: spec.fitAr } : {}),
      ...(spec.arInit !== undefined ? { init: spec.arInit } : {}),
      ...(spec.arFixed !== undefined ? { fixed: spec.arFixed } : {}),
    };
  }

  return Object.keys(params).length > 0 ? { params } : {};
}

// ─── Tests ──────────────────────────────────────────────────────────────────

describe('dlmMLE', async () => {
  // Set backend once for all MLE tests
  defaultDevice('wasm');

  it('recovers s and w for local-level model', async () => {
    const s_true = 10;
    const w_true = [3];
    const options = { order: 0 };
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    const result = await dlmMLE(y, { ...options, ...mleParams({ obsStdInit: s_true, processStdInit: w_true }), maxIter: 200, lr: 0.05, tol: 1e-6, dtype: 'f64' });

    // MLE should converge
    expect(result.iterations).toBeLessThan(200);
    // Parameters should be in the right ballpark (MLE is not exact on finite data)
    expect(result.obsStd).toBeGreaterThan(s_true * 0.3);
    expect(result.obsStd).toBeLessThan(s_true * 3);
    expect(result.processStd[0]).toBeGreaterThan(0);
    // Fit result should be populated
    expect(result.fit.y.length).toBe(200);
    expect(result.fit.smoothed.series(0).length).toBe(200);
    // arCoefficients should be undefined (no AR)
    expect(result.arCoefficients).toBeUndefined();
    // devianceHistory should be non-increasing (mostly)
    expect(result.devianceHistory.length).toBeGreaterThan(0);
  });

  it('recovers AR coefficient with params.arCoefficients.fit=true', async () => {
    const phi_true = 0.8;
    const s_true = 3;
    const w_true = [5, 4]; // level noise, AR noise

    const options = { order: 0, arCoefficients: [phi_true] };
    const sys = dlmGenSys(options);
    expect(sys.m).toBe(2);

    // Verify findArInds
    const arInds = findArInds(options);
    expect(arInds).toEqual([1]); // AR state is at index 1 (after order=0 trend)

    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    const result = await dlmMLE(
      y, { ...options,
      ...mleParams({ obsStdInit: s_true, processStdInit: w_true, fitAr: true, arInit: [0.5] }), // init arCoefficients away from true
      maxIter: 200, lr: 0.02, tol: 1e-6, dtype: 'f64' },
    );

    // arCoefficients should be returned
    expect(result.arCoefficients).toBeDefined();
    expect(result.arCoefficients!.length).toBe(1);

    // arCoefficients should be reasonably close to true value
    // (on 200 datapoints with good init, expect within 0.2 of true)
    const phi_err = Math.abs(result.arCoefficients![0] - phi_true);
    expect(phi_err).toBeLessThan(0.2);

    // Fit result should exist and be valid
    expect(result.fit.y.length).toBe(200);
    expect(result.fit.smoothed.series(0).length).toBe(200);

    // deviance should be finite
    expect(Number.isFinite(result.deviance)).toBe(true);
  });

  it('omitting params.arCoefficients.fit keeps AR coefficients fixed', async () => {
    const phi_fixed = 0.8;
    const s_true = 3;
    const w_true = [5, 4];

    const options = { order: 0, arCoefficients: [phi_fixed] };
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    const result = await dlmMLE(y, { ...options, maxIter: 100, lr: 0.02, tol: 1e-6, dtype: 'f64' });

    // arCoefficients should NOT be returned (not fitted)
    expect(result.arCoefficients).toBeUndefined();

    // obsStd and processStd should still be fitted
    expect(result.obsStd).toBeGreaterThan(0);
    expect(result.processStd.length).toBe(2);
    expect(result.fit.y.length).toBe(200);
    expect(Number.isFinite(result.deviance)).toBe(true);
  });

  it('dlmMLE converges with NaN observations (missing data)', async () => {
    // Generate clean data, then punch 20% holes in it
    const s_true = 8;
    const w_true = [3];
    const options = { order: 0 };
    const sys = dlmGenSys(options);
    const yClean = generateData(sys.G, sys.F, s_true, w_true, 100, 7);

    // Remove every 5th observation (20 out of 100)
    const y: (number | null)[] = yClean.map((v, i) => (i % 5 === 0 ? null : v));
    const yForMle = y.map(v => (v === null ? NaN : v));
    const nobs_expected = y.filter(v => v !== null).length;
    expect(nobs_expected).toBe(80);

    const result = await dlmMLE(yForMle, { ...options, maxIter: 150, lr: 0.05, tol: 1e-6, dtype: 'f64' });

    // deviance must be finite (NaN deviance would indicate a bug in masking)
    expect(Number.isFinite(result.deviance)).toBe(true);

    // obsStd and processStd estimates should be positive
    expect(result.obsStd).toBeGreaterThan(0);
    expect(result.processStd[0]).toBeGreaterThan(0);

    // fit.nobs should reflect only the observed timesteps
    expect(result.fit.nobs).toBe(nobs_expected);

    // fit outputs should be fully interpolated (finite everywhere)
    expect(Array.from(result.fit.yhat).every(Number.isFinite)).toBe(true);
    expect(Array.from(result.fit.smoothed.series(0)).every(Number.isFinite)).toBe(true);
  });

  it('stuffIntegerTimestamps matches explicit NaN-stuffed MLE input', async () => {
    const s_true = 6;
    const w_true = [2.5];
    const options = { order: 0 };
    const sys = dlmGenSys(options);
    const yClean = generateData(sys.G, sys.F, s_true, w_true, 14, 101);

    const missing = new Set([3, 4, 9]);
    const yStuffed = yClean.map((v, i) => (missing.has(i) ? NaN : v));
    const yObs = yClean.filter((_v, i) => !missing.has(i));
    const tsObs = yClean
      .map((_v, i) => i)
      .filter((_t, i) => !missing.has(i));

    const stuffedResult = await dlmMLE(yStuffed, {
      ...options,
      ...mleParams({ obsStdInit: s_true, processStdInit: w_true }),
      maxIter: 30,
      tol: 1e-6,
      dtype: 'f64',
      optimizer: 'natural',
    });
    const stuffed = stuffIntegerTimestamps(yObs, tsObs, undefined, undefined);
    const tsResult = await dlmMLE(stuffed.y, {
      ...options,
      ...mleParams({ obsStdInit: s_true, processStdInit: w_true }),
      maxIter: 30,
      tol: 1e-6,
      dtype: 'f64',
      optimizer: 'natural',
    });

    expect(tsResult.obsStd).toBeCloseTo(stuffedResult.obsStd, 8);
    expect(tsResult.processStd[0]).toBeCloseTo(stuffedResult.processStd[0], 8);
    expect(tsResult.deviance).toBeCloseTo(stuffedResult.deviance, 8);
    expect(Array.from(tsResult.fit.yhat)).toEqual(Array.from(stuffedResult.fit.yhat));
    for (let k = 0; k < stuffed.y.length; k++) {
      expect(tsResult.fit.smoothed.get(k, 0)).toBeCloseTo(stuffedResult.fit.smoothed.get(k, 0), 8);
    }
  });

  it('stuffIntegerTimestamps rejects fractional steps', async () => {
    const options = { order: 0 };
    const yObs = [4, 4.5, 5.1, 6.0];
    const tsObs = [0, 1 / 12, 2 / 12, 5 / 12];

    expect(() => stuffIntegerTimestamps(yObs, tsObs, undefined, undefined)).toThrow(/positive integers relative to the initial timestamp/);
  });

  it('stuffIntegerTimestamps + obsStdFixed match explicit NaN-stuffed MLE input', async () => {
    const options = { order: 0 };
    const yClean = [4, 4.5, 5.2, 5.1, 6.0, 6.3, 6.1, 7.0];
    const missing = new Set([2, 5]);
    const yStuffed = yClean.map((v, i) => (missing.has(i) ? NaN : v));
    const sFixedStuffed = yClean.map((_v, i) => (missing.has(i) ? 1.0 : 0.5 + i * 0.05));
    const yObs = yClean.filter((_v, i) => !missing.has(i));
    const sFixedObs = sFixedStuffed.filter((_v, i) => !missing.has(i));
    const tsObs = yClean
      .map((_v, i) => i)
      .filter((_t, i) => !missing.has(i));

    const stuffedResult = await dlmMLE(yStuffed, {
      ...options,
      ...mleParams({ obsStdFixed: sFixedStuffed, processStdInit: [0.3] }),
      maxIter: 20,
      tol: 1e-6,
      dtype: 'f64',
      optimizer: 'natural',
    });
    const stuffed = stuffIntegerTimestamps(yObs, tsObs, undefined, sFixedObs);
    const tsResult = await dlmMLE(stuffed.y, {
      ...options,
      ...mleParams({ obsStdFixed: stuffed.obsStdFixed, processStdInit: [0.3] }),
      maxIter: 20,
      tol: 1e-6,
      dtype: 'f64',
      optimizer: 'natural',
    });

    expect(tsResult.obsStd).toBeNaN();
    expect(tsResult.processStd[0]).toBeCloseTo(stuffedResult.processStd[0], 8);
    expect(tsResult.deviance).toBeCloseTo(stuffedResult.deviance, 8);
  });

  it('params.obsStd.fixed accepts a scalar', async () => {
    const options = { order: 0 };
    const y = [4, 4.5, 5.2, 5.1, 6.0, 6.3, 6.1, 7.0];

    const result = await dlmMLE(y, {
      ...options,
      ...mleParams({ obsStdFixed: 0.5, processStdInit: [0.3] }),
      maxIter: 20,
      tol: 1e-6,
      dtype: 'f64',
      optimizer: 'natural',
    });

    expect(result.obsStd).toBeNaN();
    expect(Array.from(result.fit.obsNoise)).toEqual(new Array(y.length).fill(0.5));
    expect(result.processStd[0]).toBeGreaterThanOrEqual(0);
  });

  it('params.processStd respects fixed zeros and tied groups', async () => {
    const options = { order: 1, harmonics: 1, seasonLength: 12 };
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, 2, [0, 0.1, 0.4, 0.4], 36, 17);
    const thetaLengths: number[] = [];

    const result = await dlmMLE(y, {
      ...options,
      ...mleParams({
        obsStdInit: 2,
        processStdInit: [0.2, 0.1, 0.4, 0.6],
        processStdFixed: { 0: 0 },
        processStdGroups: [0, 1, 2, 2],
      }),
      maxIter: 1,
      tol: 1e-6,
      dtype: 'f64',
      callbacks: {
        onInit(theta) { thetaLengths.push(theta.length); },
      },
    });

    expect(thetaLengths).toEqual([3]);
    expect(result.processStd[0]).toBe(0);
    expect(result.processStd[2]).toBeCloseTo(result.processStd[3], 12);
  });

  it('params.arCoefficients.fixed overrides the model AR value without exposing fitted coefficients', async () => {
    const phi_model = 0.8;
    const phi_fixed = 0.2;
    const s_true = 3;
    const w_true = [5, 4];
    const options = { order: 0, arCoefficients: [phi_model] };
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, s_true, w_true, 80, 52);

    const result = await dlmMLE(y, {
      ...options,
      ...mleParams({
        obsStdInit: s_true,
        processStdInit: w_true,
        fitAr: true,
        arFixed: [phi_fixed],
      }),
      maxIter: 20,
      lr: 0.02,
      tol: 1e-6,
      dtype: 'f64',
    });

    expect(result.arCoefficients).toBeUndefined();
    expect(result.fit.modelSpec.arCoefficients).toEqual([phi_fixed]);
  });

  it('custom loss sees expanded physical params under grouped and fixed processStd controls', async () => {
    const options = { order: 1, harmonics: 1, seasonLength: 12 };
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, 2, [0, 0.1, 0.4, 0.4], 36, 23);
    let seenMeta: { nObs: number; nProcess: number; nAr: number } | undefined;

    const loss: DlmLossFn = (deviance, params, meta) => {
      seenMeta = { nObs: meta.nObs, nProcess: meta.nProcess, nAr: meta.nAr };
      const target = np.array([1, 0, 0.1, 0.2, 0.2]);
      const diff = np.subtract(params, target);
      const penalty = np.multiply(np.array(0.01), np.sum(np.square(diff)));
      return np.add(deviance, penalty);
    };

    const result = await dlmMLE(y, {
      ...options,
      ...mleParams({
        obsStdInit: 1,
        processStdInit: [0.2, 0.1, 0.4, 0.6],
        processStdFixed: { 0: 0 },
        processStdGroups: [0, 1, 2, 2],
      }),
      maxIter: 1,
      tol: 1e-6,
      dtype: 'f64',
      loss,
    });

    expect(seenMeta).toEqual({ nObs: 1, nProcess: 4, nAr: 0 });
    expect(result.priorPenalty).toBeDefined();
    expect(result.processStd[0]).toBe(0);
    expect(result.processStd[2]).toBeCloseTo(result.processStd[3], 12);
  });

  it('dlmMLE rejects timestamps directly', async () => {
    await expect(dlmMLE([1, 2, 3], {
      order: 0,
      timestamps: [0, 1, 2],
      maxIter: 5,
      dtype: 'f64',
      optimizer: 'natural',
    } as any)).rejects.toThrow(/unknown option 'timestamps'/);
  });

  // ─── Natural gradient (Fisher dualization) optimizer ──────────────────────

  it('natural: recovers s and w for local-level model', async () => {
    const s_true = 10;
    const w_true = [3];
    const options = { order: 0 };
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    const result = await dlmMLE(y, { ...options, ...mleParams({ obsStdInit: s_true, processStdInit: w_true }), tol: 1e-6, dtype: 'f64', optimizer: 'natural' });

    // Natural gradient should converge in very few iterations (quadratic convergence)
    expect(result.iterations).toBeLessThan(20);
    // Parameters should be in the right ballpark
    expect(result.obsStd).toBeGreaterThan(s_true * 0.3);
    expect(result.obsStd).toBeLessThan(s_true * 3);
    expect(result.processStd[0]).toBeGreaterThan(0);
    // Fit result should be populated
    expect(result.fit.y.length).toBe(200);
    expect(result.fit.smoothed.series(0).length).toBe(200);
    expect(result.arCoefficients).toBeUndefined();
    expect(result.devianceHistory.length).toBeGreaterThan(0);
  });

  it('natural: recovers s and w for order=1 model', async () => {
    const s_true = 5;
    const w_true = [2, 1];
    const options = { order: 1 };
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    const result = await dlmMLE(y, { ...options, ...mleParams({ obsStdInit: s_true, processStdInit: w_true }), tol: 1e-6, dtype: 'f64', optimizer: 'natural' });

    expect(result.iterations).toBeLessThan(30);
    expect(result.obsStd).toBeGreaterThan(0);
    expect(result.processStd.length).toBe(2);
    expect(result.processStd.every(v => v > 0)).toBe(true);
    expect(Number.isFinite(result.deviance)).toBe(true);
    expect(result.fit.y.length).toBe(200);
  });

  it('natural: recovers AR coefficient with params.arCoefficients.fit=true', async () => {
    const phi_true = 0.8;
    const s_true = 3;
    const w_true = [5, 4];

    const options = { order: 0, arCoefficients: [phi_true] };
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    const result = await dlmMLE(y, {
      ...options,
      ...mleParams({ obsStdInit: s_true, processStdInit: w_true, fitAr: true, arInit: [0.5] }),
      tol: 1e-6, dtype: 'f64', optimizer: 'natural',
    });

    expect(result.arCoefficients).toBeDefined();
    expect(result.arCoefficients!.length).toBe(1);
    const phi_err = Math.abs(result.arCoefficients![0] - phi_true);
    expect(phi_err).toBeLessThan(0.3);
    expect(Number.isFinite(result.deviance)).toBe(true);
    expect(result.fit.y.length).toBe(200);
  });

  it('natural: converges with NaN observations (missing data)', async () => {
    const s_true = 8;
    const w_true = [3];
    const options = { order: 0 };
    const sys = dlmGenSys(options);
    const yClean = generateData(sys.G, sys.F, s_true, w_true, 100, 7);

    const y = yClean.map((v, i) => (i % 5 === 0 ? NaN : v));

    const result = await dlmMLE(y, { ...options, tol: 1e-6, dtype: 'f64', optimizer: 'natural' });

    expect(Number.isFinite(result.deviance)).toBe(true);
    expect(result.obsStd).toBeGreaterThan(0);
    expect(result.processStd[0]).toBeGreaterThan(0);
    expect(result.fit.nobs).toBe(80);
    expect(Array.from(result.fit.yhat).every(Number.isFinite)).toBe(true);
  });

  // ─── Natural gradient + associative scan loss ─────────────────────────────

  it('natural+assoc: recovers s and w for local-level model', async () => {
    const s_true = 10;
    const w_true = [3];
    const options = { order: 0 };
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    const result = await dlmMLE(y, { ...options, ...mleParams({ obsStdInit: s_true, processStdInit: w_true }), tol: 1e-6, dtype: 'f64', optimizer: 'natural', algorithm: 'assoc' });

    expect(result.iterations).toBeLessThan(20);
    expect(result.obsStd).toBeGreaterThan(s_true * 0.3);
    expect(result.obsStd).toBeLessThan(s_true * 3);
    expect(result.processStd[0]).toBeGreaterThan(0);
    expect(result.fit.y.length).toBe(200);
    expect(result.devianceHistory.length).toBeGreaterThan(0);
  });

  it('natural+assoc: recovers s and w for order=1 model', async () => {
    const s_true = 5;
    const w_true = [2, 1];
    const options = { order: 1 };
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    const result = await dlmMLE(y, { ...options, ...mleParams({ obsStdInit: s_true, processStdInit: w_true }), tol: 1e-6, dtype: 'f64', optimizer: 'natural', algorithm: 'assoc' });

    expect(result.iterations).toBeLessThan(30);
    expect(result.obsStd).toBeGreaterThan(0);
    expect(result.processStd.length).toBe(2);
    expect(result.processStd.every(v => v > 0)).toBe(true);
    expect(Number.isFinite(result.deviance)).toBe(true);
    expect(result.fit.y.length).toBe(200);
  });

  it('natural+assoc: recovers AR coefficient with params.arCoefficients.fit=true', async () => {
    const phi_true = 0.8;
    const s_true = 3;
    const w_true = [5, 4];

    const options = { order: 0, arCoefficients: [phi_true] };
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    const result = await dlmMLE(y, {
      ...options,
      ...mleParams({ obsStdInit: s_true, processStdInit: w_true, fitAr: true, arInit: [0.5] }),
      tol: 1e-6, dtype: 'f64', optimizer: 'natural', algorithm: 'assoc',
    });

    expect(result.arCoefficients).toBeDefined();
    expect(result.arCoefficients!.length).toBe(1);
    const phi_err = Math.abs(result.arCoefficients![0] - phi_true);
    expect(phi_err).toBeLessThan(0.3);
    expect(Number.isFinite(result.deviance)).toBe(true);
    expect(result.fit.y.length).toBe(200);
  });

  it('natural+assoc: converges with NaN observations (missing data)', async () => {
    const s_true = 8;
    const w_true = [3];
    const options = { order: 0 };
    const sys = dlmGenSys(options);
    const yClean = generateData(sys.G, sys.F, s_true, w_true, 100, 7);

    const y = yClean.map((v, i) => (i % 5 === 0 ? NaN : v));

    const result = await dlmMLE(y, { ...options, tol: 1e-6, dtype: 'f64', optimizer: 'natural', algorithm: 'assoc' });

    expect(Number.isFinite(result.deviance)).toBe(true);
    expect(result.obsStd).toBeGreaterThan(0);
    expect(result.processStd[0]).toBeGreaterThan(0);
    expect(result.fit.nobs).toBe(80);
    expect(Array.from(result.fit.yhat).every(Number.isFinite)).toBe(true);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // MAP / custom loss tests
  // ═══════════════════════════════════════════════════════════════════════════

  // Strong L2 prior on natural-scale params that pulls estimates toward a known target.
  // The prior is: penalty = λ · Σ (param_i − μ_i)²
  // With large λ the MAP estimate should be close to μ, different from MLE.
  // Params are natural scale: [s, w₀, ...] (std devs), not log-transformed.
  const makePrior = (priorMean: number[], strength: number): DlmLossFn =>
    (deviance, params, _meta) => {
      const mu = np.array(priorMean);
      const diff = np.subtract(params, mu);
      const penalty = np.multiply(np.array(strength), np.sum(np.square(diff)));
      return np.add(deviance, penalty);
    };

  it('MAP: adam returns priorPenalty and shifted parameters', async () => {
    const s_true = 10;
    const w_true = [3];
    const options = { order: 0 };
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    // MLE baseline
    const mle = await dlmMLE(y, { ...options, ...mleParams({ obsStdInit: s_true, processStdInit: w_true }),
      maxIter: 200, lr: 0.05, tol: 1e-6, dtype: 'f64' });
    expect(mle.priorPenalty).toBeUndefined();

    // MAP: strong prior pulling natural-scale params toward 1 (i.e. s≈1, w≈1)
    // params layout: [s, w0] (natural scale, not log-transformed)
    const prior = makePrior([1, 1], 50);
    const map = await dlmMLE(y, { ...options, ...mleParams({ obsStdInit: s_true, processStdInit: w_true }),
      maxIter: 200, lr: 0.05, tol: 1e-6, dtype: 'f64', loss: prior });

    // priorPenalty should exist and be positive (prior adds a non-negative term)
    expect(map.priorPenalty).toBeDefined();
    expect(map.priorPenalty!).toBeGreaterThan(0);
    // deviance should be pure −2·logL (larger than MLE deviance since MAP is suboptimal for likelihood)
    expect(Number.isFinite(map.deviance)).toBe(true);
    expect(map.deviance).toBeGreaterThanOrEqual(mle.deviance - 1);
    // MAP obsStd should be pulled toward 1, i.e. smaller than MLE
    expect(map.obsStd).toBeLessThan(mle.obsStd);
    // Fit should be valid
    expect(map.fit.y.length).toBe(200);
  });

  it('MAP: natural gradient returns priorPenalty and shifted parameters', async () => {
    const s_true = 10;
    const w_true = [3];
    const options = { order: 0 };
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    // MAP: strong prior pulling natural-scale params toward 1
    const prior = makePrior([1, 1], 50);
    const map = await dlmMLE(y, { ...options, ...mleParams({ obsStdInit: s_true, processStdInit: w_true }),
      tol: 1e-6, dtype: 'f64', optimizer: 'natural', loss: prior });

    expect(map.priorPenalty).toBeDefined();
    expect(map.priorPenalty!).toBeGreaterThan(0);
    expect(Number.isFinite(map.deviance)).toBe(true);
    // MAP should shift obsStd toward 1
    expect(map.obsStd).toBeLessThan(s_true);
    expect(map.fit.y.length).toBe(200);
  });

  it('MAP: loss="ml" behaves identically to no loss option', async () => {
    const s_true = 10;
    const w_true = [3];
    const options = { order: 0 };
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    const mlDefault = await dlmMLE(y, { ...options, ...mleParams({ obsStdInit: s_true, processStdInit: w_true }),
      maxIter: 200, lr: 0.05, tol: 1e-6, dtype: 'f64' });
    const mlExplicit = await dlmMLE(y, { ...options, ...mleParams({ obsStdInit: s_true, processStdInit: w_true }),
      maxIter: 200, lr: 0.05, tol: 1e-6, dtype: 'f64', loss: 'ml' });

    // Same deviance (both pure MLE)
    expect(Math.abs(mlDefault.deviance - mlExplicit.deviance)).toBeLessThan(0.01);
    expect(mlExplicit.priorPenalty).toBeUndefined();
  });

  it('MAP: identity loss (pass-through) gives same result as MLE', async () => {
    const s_true = 10;
    const w_true = [3];
    const options = { order: 0 };
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    // Identity loss: just return the Kalman deviance unchanged
    const identity: DlmLossFn = (deviance, _params, _meta) => deviance;
    const result = await dlmMLE(y, { ...options, ...mleParams({ obsStdInit: s_true, processStdInit: w_true }),
      maxIter: 200, lr: 0.05, tol: 1e-6, dtype: 'f64', loss: identity });

    // priorPenalty should be ~0 (identity adds nothing)
    expect(result.priorPenalty).toBeDefined();
    expect(Math.abs(result.priorPenalty!)).toBeLessThan(0.1);
    // Parameters should be close to pure MLE
    expect(result.obsStd).toBeGreaterThan(s_true * 0.3);
    expect(result.obsStd).toBeLessThan(s_true * 3);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // dlmPrior factory tests
  // ═══════════════════════════════════════════════════════════════════════════

  it('dlmPrior: IG on obsVar shifts obsStd toward prior mode', async () => {
    const s_true = 10;
    const w_true = [3];
    const options = { order: 0 };
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    // MLE baseline
    const mle = await dlmMLE(y, { ...options, ...mleParams({ obsStdInit: s_true, processStdInit: w_true }),
      maxIter: 200, lr: 0.05, tol: 1e-6, dtype: 'f64' });

    // Strong IG prior on obsVar pulling s toward low values
    // IG(shape=2, rate=0.5): mode = β/(α+1) = 0.5/3 ≈ 0.17 for variance
    // This should pull obsStd well below the MLE estimate
    const prior = dlmPrior({ obsVar: { shape: 2, rate: 0.5 } });
    const map = await dlmMLE(y, { ...options, ...mleParams({ obsStdInit: s_true, processStdInit: w_true }),
      maxIter: 300, lr: 0.05, tol: 1e-6, dtype: 'f64', loss: prior });

    expect(map.priorPenalty).toBeDefined();
    expect(map.priorPenalty!).toBeGreaterThan(0);
    expect(map.obsStd).toBeLessThan(mle.obsStd);
    expect(map.fit.y.length).toBe(200);
  });

  it('dlmPrior: IG on processVar shifts processStd', async () => {
    const s_true = 10;
    const w_true = [3];
    const options = { order: 0 };
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    // MLE baseline
    const mle = await dlmMLE(y, { ...options, ...mleParams({ obsStdInit: s_true, processStdInit: w_true }),
      maxIter: 200, lr: 0.05, tol: 1e-6, dtype: 'f64' });

    // Strong IG prior on processVar pulling w toward low values
    const prior = dlmPrior({ processVar: { shape: 2, rate: 0.1 } });
    const map = await dlmMLE(y, { ...options, ...mleParams({ obsStdInit: s_true, processStdInit: w_true }),
      maxIter: 300, lr: 0.05, tol: 1e-6, dtype: 'f64', loss: prior });

    expect(map.priorPenalty).toBeDefined();
    expect(map.priorPenalty!).toBeGreaterThan(0);
    expect(map.processStd[0]).toBeLessThan(mle.processStd[0]);
  });

  it('dlmPrior: IG on both obsVar + processVar (MATLAB DLM style)', async () => {
    const s_true = 10;
    const w_true = [3];
    const options = { order: 0 };
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    // Combined prior — matches MATLAB dlmGibbsDIG style
    const prior = dlmPrior({
      obsVar:     { shape: 2, rate: 1 },
      processVar: { shape: 2, rate: 1 },
    });
    const map = await dlmMLE(y, { ...options, ...mleParams({ obsStdInit: s_true, processStdInit: w_true }),
      maxIter: 300, lr: 0.05, tol: 1e-6, dtype: 'f64', loss: prior });

    expect(map.priorPenalty).toBeDefined();
    expect(map.priorPenalty!).toBeGreaterThan(0);
    expect(Number.isFinite(map.deviance)).toBe(true);
    expect(map.fit.y.length).toBe(200);
  });

  it('dlmPrior: IG with natural gradient optimizer', async () => {
    const s_true = 10;
    const w_true = [3];
    const options = { order: 0 };
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    const prior = dlmPrior({
      obsVar:     { shape: 2, rate: 1 },
      processVar: { shape: 2, rate: 1 },
    });
    const map = await dlmMLE(y, { ...options, ...mleParams({ obsStdInit: s_true, processStdInit: w_true }),
      tol: 1e-6, dtype: 'f64', optimizer: 'natural', loss: prior });

    expect(map.priorPenalty).toBeDefined();
    expect(map.priorPenalty!).toBeGreaterThan(0);
    expect(Number.isFinite(map.deviance)).toBe(true);
  });

  it('dlmPrior: validation rejects invalid specs', () => {
    expect(() => dlmPrior({ obsVar: { shape: -1, rate: 1 } })).toThrow();
    expect(() => dlmPrior({ obsVar: { shape: 1, rate: 0 } })).toThrow();
    expect(() => dlmPrior({ processVar: { shape: 0, rate: 1 } })).toThrow();
    expect(() => dlmPrior({ arCoef: { mean: 0, std: -1 } })).toThrow();
    // Valid specs should not throw
    expect(() => dlmPrior({ obsVar: { shape: 0.001, rate: 0.001 } })).not.toThrow();
  });
});

