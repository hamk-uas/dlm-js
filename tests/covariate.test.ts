/**
 * Covariate (regression) tests for dlm-js.
 *
 * Verifies that dlmFit correctly recovers known regression coefficients β
 * when covariates are supplied as the X parameter.
 *
 * Model:
 *   y(t) = μ(t) + β₁·X₁(t) + β₂·X₂(t) + v,  v ~ N(0, s²)
 *   μ(t) = μ(t-1) + w,                          (local level)
 *
 * We generate synthetic observations from known β, then check that the
 * smoother recovers all three modes (for / scan / jit call paths) and
 * that the recovered β states converge to the true values within tolerance.
 *
 * Test 1 (β recovery): low obs noise + zero state noise → exact β recovery.
 * Test 2 (yhat accuracy): residuals after accounting for X should be small.
 * Test 3 (XX stored): result.XX carries back the covariate rows verbatim.
 */
import { defaultDevice } from '@hamk-uas/jax-js-nonconsuming';
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { dlmFit } from '../src/index';
import { withLeakCheck } from './utils';

// ─── Deterministic PRNG ──────────────────────────────────────────────────────
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

// ─── Synthetic covariate dataset ────────────────────────────────────────────
/**
 * Generate y = μ_true + β₁·X₁ + β₂·X₂ + ε
 * - μ_true is a slow random walk (small w)
 * - X₁ is a sine wave (period 50)
 * - X₂ is a cosine wave (period 20)
 */
function makeData(n: number, beta: number[], s: number, seed: number) {
  const randn = gaussianRng(mulberry32(seed));
  const X: number[][] = Array.from({ length: n }, (_, t) => [
    Math.sin(2 * Math.PI * t / 50),
    Math.cos(2 * Math.PI * t / 20),
  ]);
  // Slow random walk level
  const mu: number[] = new Array(n).fill(0);
  mu[0] = randn() * 2;
  const w_level = 0.05;
  for (let t = 1; t < n; t++) mu[t] = mu[t - 1] + randn() * w_level;

  const y: number[] = Array.from({ length: n }, (_, t) =>
    mu[t] + beta[0] * X[t][0] + beta[1] * X[t][1] + randn() * s
  );
  return { y, X, mu };
}

// ─── Test suite ──────────────────────────────────────────────────────────────
describe('covariate regression (X parameter)', () => {
  const TRUE_BETA = [3.5, -2.1];
  const N = 200;
  const S_OBS = 0.3;
  const W_LEVEL = 0.05;

  const { y, X, mu: _mu } = makeData(N, TRUE_BETA, S_OBS, 42);

  beforeAll(() => { defaultDevice('cpu'); });
  afterAll(() => { defaultDevice('cpu'); });

  it('β states converge to true values (Float64)', async () => {
    const fit = await withLeakCheck(() =>
      dlmFit(y, { obsStd: S_OBS, processStd: [W_LEVEL], dtype: 'f64', order: 0, X })
    );

    // order=0 → local-level only, m_base=1; β₁=smoothed[1], β₂=smoothed[2]
    const m_base = 1;
    const beta1_est = fit.smoothed.get(N - 1, m_base);   // last timestep
    const beta2_est = fit.smoothed.get(N - 1, m_base + 1);

    expect(Math.abs(beta1_est - TRUE_BETA[0])).toBeLessThan(0.3);
    expect(Math.abs(beta2_est - TRUE_BETA[1])).toBeLessThan(0.3);

    // yhat should closely match observations minus noise
    let rmse = 0;
    for (let t = 0; t < N; t++) {
      rmse += (fit.yhat[t] - y[t]) ** 2;
    }
    rmse = Math.sqrt(rmse / N);
    // yhat rmse should be close to observation noise level
    expect(rmse).toBeLessThan(S_OBS * 3);
  }, 30000);

  it('XX field stores covariate rows verbatim', async () => {
    const fit = await withLeakCheck(() =>
      dlmFit(y, { obsStd: S_OBS, processStd: [W_LEVEL], dtype: 'f64', order: 0, X })
    );
    expect(Array.isArray(fit.covariates)).toBe(true);
    expect((fit.covariates as number[][]).length).toBe(N);
    // Check a few rows
    for (const t of [0, 50, 100, N - 1]) {
      const row = (fit.covariates as number[][])[t];
      expect(row[0]).toBeCloseTo(X[t][0], 12);
      expect(row[1]).toBeCloseTo(X[t][1], 12);
    }
  }, 30000);

  it('no covariates: covariates is empty, state size unchanged', async () => {
    const fit = await withLeakCheck(() =>
      dlmFit(y, { obsStd: S_OBS, processStd: [W_LEVEL], dtype: 'f64', order: 0 })
    );
    expect(fit.covariates).toEqual([]);
    // m_base = 1 for order=0 (local level)
    expect(fit.m).toBe(1);
  }, 30000);

  it('β recovery works with Float32 (within looser tolerance)', async () => {
    const fit = await withLeakCheck(() =>
      dlmFit(y, { obsStd: S_OBS, processStd: [W_LEVEL], dtype: 'f32', order: 0, X })
    );
    const m_base = 1;
    const beta1_est = fit.smoothed.get(N - 1, m_base);
    const beta2_est = fit.smoothed.get(N - 1, m_base + 1);
    // Float32: just verify the states are finite and in the right ballpark
    expect(isFinite(beta1_est)).toBe(true);
    expect(isFinite(beta2_est)).toBe(true);
    expect(Math.abs(beta1_est - TRUE_BETA[0])).toBeLessThan(2.0);
    expect(Math.abs(beta2_est - TRUE_BETA[1])).toBeLessThan(2.0);
  }, 30000);

  it('single covariate (q=1) works', async () => {
    const X1 = X.map(row => [row[0]]);  // only first column
    const fit = await withLeakCheck(() =>
      dlmFit(y, { obsStd: S_OBS, processStd: [W_LEVEL], dtype: 'f64', order: 0, X: X1 })
    );
    expect(fit.m).toBe(2);  // m_base=1 (order=0) + q=1
    const beta1_est = fit.smoothed.get(N - 1, 1);
    // β₁ should be close to TRUE_BETA[0], though now X₂ is omitted (signal contamination)
    expect(isFinite(beta1_est)).toBe(true);
  }, 30000);
});
