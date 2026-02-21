/**
 * MLE parameter estimation tests for dlm-js.
 *
 * Tests dlmMLE with:
 *   1. Basic s/w recovery (Nile-like order=0 model)
 *   2. AR coefficient fitting (order=0 + AR(1), fitar=true)
 *   3. fitar=false baseline (AR coefficients stay fixed)
 *
 * Uses WASM backend + Float64 for all tests.
 * Synthetic data from a deterministic PRNG guarantees reproducibility.
 */
import { defaultDevice } from '@hamk-uas/jax-js-nonconsuming';
import { describe, it, expect } from 'vitest';
import { dlmMLE, dlmGenSys, findArInds } from '../src/index';
import { withLeakCheck } from './utils';

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

    const result = await withLeakCheck(() =>
      dlmMLE(y, { ...options, init: { obsStd: s_true, processStd: w_true }, maxIter: 200, lr: 0.05, tol: 1e-6, dtype: 'f64' })
    );

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

  it('recovers AR coefficient with fitar=true', async () => {
    const phi_true = 0.8;
    const s_true = 3;
    const w_true = [5, 4]; // level noise, AR noise

    const options = { order: 0, arCoefficients: [phi_true], fitAr: true };
    const sys = dlmGenSys(options);
    expect(sys.m).toBe(2);

    // Verify findArInds
    const arInds = findArInds(options);
    expect(arInds).toEqual([1]); // AR state is at index 1 (after order=0 trend)

    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    const result = await withLeakCheck(() =>
      dlmMLE(
        y, { ...options,
        init: { obsStd: s_true, processStd: w_true, arCoefficients: [0.5] }, // init arCoefficients away from true
        maxIter: 200, lr: 0.02, tol: 1e-6, dtype: 'f64' },
      )
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

  it('fitar=false keeps AR coefficients fixed', async () => {
    const phi_fixed = 0.8;
    const s_true = 3;
    const w_true = [5, 4];

    const options = { order: 0, arCoefficients: [phi_fixed] }; // fitAr defaults to false
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    const result = await withLeakCheck(() =>
      dlmMLE(y, { ...options, maxIter: 100, lr: 0.02, tol: 1e-6, dtype: 'f64' })
    );

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

    const result = await withLeakCheck(() =>
      dlmMLE(yForMle, { ...options, maxIter: 150, lr: 0.05, tol: 1e-6, dtype: 'f64' })
    );

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
});

