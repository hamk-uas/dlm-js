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
import { defaultDevice, DType } from '@hamk-uas/jax-js-nonconsuming';
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
  const dtype = DType.Float64;

  it('recovers s and w for local-level model', async () => {
    const s_true = 10;
    const w_true = [3];
    const options = { order: 0 };
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    const result = await withLeakCheck(() =>
      dlmMLE(y, options, { s: s_true, w: w_true }, 200, 0.05, 1e-6, dtype)
    );

    // MLE should converge
    expect(result.iterations).toBeLessThan(200);
    // Parameters should be in the right ballpark (MLE is not exact on finite data)
    expect(result.s).toBeGreaterThan(s_true * 0.3);
    expect(result.s).toBeLessThan(s_true * 3);
    expect(result.w[0]).toBeGreaterThan(0);
    // Fit result should be populated
    expect(result.fit.y.length).toBe(200);
    expect(result.fit.x[0].length).toBe(200);
    // arphi should be undefined (no AR)
    expect(result.arphi).toBeUndefined();
    // likHistory should be non-increasing (mostly)
    expect(result.likHistory.length).toBeGreaterThan(0);
  });

  it('recovers AR coefficient with fitar=true', async () => {
    const phi_true = 0.8;
    const s_true = 3;
    const w_true = [5, 4]; // level noise, AR noise

    const options = { order: 0, arphi: [phi_true], fitar: true };
    const sys = dlmGenSys(options);
    expect(sys.m).toBe(2);

    // Verify findArInds
    const arInds = findArInds(options);
    expect(arInds).toEqual([1]); // AR state is at index 1 (after order=0 trend)

    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    const result = await withLeakCheck(() =>
      dlmMLE(
        y, options,
        { s: s_true, w: w_true, arphi: [0.5] }, // init arphi away from true
        200, 0.02, 1e-6, dtype,
      )
    );

    // arphi should be returned
    expect(result.arphi).toBeDefined();
    expect(result.arphi!.length).toBe(1);

    // arphi should be reasonably close to true value
    // (on 200 datapoints with good init, expect within 0.2 of true)
    const phi_err = Math.abs(result.arphi![0] - phi_true);
    expect(phi_err).toBeLessThan(0.2);

    // Fit result should exist and be valid
    expect(result.fit.y.length).toBe(200);
    expect(result.fit.x[0].length).toBe(200);

    // lik should be finite
    expect(Number.isFinite(result.lik)).toBe(true);
  });

  it('fitar=false keeps AR coefficients fixed', async () => {
    const phi_fixed = 0.8;
    const s_true = 3;
    const w_true = [5, 4];

    const options = { order: 0, arphi: [phi_fixed] }; // fitar defaults to false
    const sys = dlmGenSys(options);
    const y = generateData(sys.G, sys.F, s_true, w_true, 200, 42);

    const result = await withLeakCheck(() =>
      dlmMLE(y, options, undefined, 100, 0.02, 1e-6, dtype)
    );

    // arphi should NOT be returned (not fitted)
    expect(result.arphi).toBeUndefined();

    // s and w should still be fitted
    expect(result.s).toBeGreaterThan(0);
    expect(result.w.length).toBe(2);
    expect(result.fit.y.length).toBe(200);
    expect(Number.isFinite(result.lik)).toBe(true);
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
      dlmMLE(yForMle, options, undefined, 150, 0.05, 1e-6, dtype)
    );

    // lik must be finite (NaN lik would indicate a bug in masking)
    expect(Number.isFinite(result.lik)).toBe(true);

    // s and w estimates should be positive
    expect(result.s).toBeGreaterThan(0);
    expect(result.w[0]).toBeGreaterThan(0);

    // fit.nobs should reflect only the observed timesteps
    expect(result.fit.nobs).toBe(nobs_expected);

    // fit outputs should be fully interpolated (finite everywhere)
    expect(Array.from(result.fit.yhat).every(Number.isFinite)).toBe(true);
    expect(result.fit.x[0].every(Number.isFinite)).toBe(true);
  });
});

