/**
 * Tests for dlmForecast — h-step-ahead prediction via jax-js lax.scan.
 *
 * Coverage:
 * 1. Local level (order=0)         — ystd monotone, yhat finite
 * 2. Local trend (order=1)         — ystd monotone, yhat follows slope
 * 3. Trigonometric seasonal        — cyclic pattern continues
 * 4. AR(2)                         — AR dynamics propagate correctly
 * 5. Covariate model               — X_forecast rows affect yhat linearly
 * 6. Smoke: all device × dtype     — all finite, correct shape
 */
import { checkLeaks, DType } from '@hamk-uas/jax-js-nonconsuming';
import { describe, it, expect, beforeAll } from 'vitest';
import { dlmFit, dlmForecast } from '../src/index';
import { getTestConfigs, applyConfig, assertAllFinite, type TestConfig } from './test-matrix';

// ─── Deterministic PRNG ───────────────────────────────────────────────────────

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

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Assert monotonically non-decreasing (allows very small floating-point plateau). */
function assertMonotoneNonDecreasing(arr: ArrayLike<number>, name: string): void {
  for (let k = 1; k < arr.length; k++) {
    expect((arr[k] as number), `${name}[${k}] >= ${name}[${k - 1}]`).toBeGreaterThanOrEqual(
      (arr[k - 1] as number) - 1e-9,
    );
  }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('dlmForecast', () => {
  let configs: TestConfig[];

  beforeAll(async () => {
    configs = await getTestConfigs();
  });

  // ── 1. Local level (order=0) ─────────────────────────────────────────────

  it('local level: ystd monotone + all finite | cpu/f64', async () => {
    const cfg = configs.find(c => c.device === 'cpu' && c.dtype === DType.Float64);
    if (!cfg) return;
    applyConfig(cfg);

    const randn = gaussianRng(mulberry32(1));
    const n = 80, s = 1.0, qW = 0.2;
    const obs = new Float64Array(n);
    let mu = 5;
    for (let t = 0; t < n; t++) { mu += qW * randn(); obs[t] = mu + s * randn(); }

    checkLeaks.start();
    const fit = await dlmFit(obs, s, [qW * qW], DType.Float64, { order: 0 });
    const h = 20;
    const fc = await dlmForecast(fit, s, h, DType.Float64);
    checkLeaks.stop();

    assertAllFinite(fc.yhat);
    assertAllFinite(fc.ystd);
    expect(fc.h).toBe(h);
    expect(fc.m).toBe(1);
    assertMonotoneNonDecreasing(fc.ystd, 'ystd');
    expect(fc.ystd[h - 1]).toBeGreaterThan(s * 0.9);
  });

  // ── 2. Local trend (order=1) ──────────────────────────────────────────────

  it('local trend: ystd monotone + yhat extrapolates trend | cpu/f64', async () => {
    const cfg = configs.find(c => c.device === 'cpu' && c.dtype === DType.Float64);
    if (!cfg) return;
    applyConfig(cfg);

    const randn = gaussianRng(mulberry32(2));
    const n = 100, s = 1.0;
    const obs = new Float64Array(n);
    let mu = 0, v = 0.5;
    for (let t = 0; t < n; t++) {
      v += 0.01 * randn(); mu += v + 0.05 * randn(); obs[t] = mu + s * randn();
    }

    checkLeaks.start();
    const fit = await dlmFit(obs, s, [0.01, 0.0025], DType.Float64, { order: 1 });
    const h = 15;
    const fc = await dlmForecast(fit, s, h, DType.Float64);
    checkLeaks.stop();

    assertAllFinite(fc.yhat);
    assertAllFinite(fc.ystd);
    assertMonotoneNonDecreasing(fc.ystd, 'ystd');
    expect(Math.abs((fc.yhat[h - 1] as number) - (fc.yhat[0] as number))).toBeGreaterThan(0.05);
  });

  // ── 3. Trigonometric seasonal ──────────────────────────────────────────────

  it('trig seasonal: cyclic yhat + ystd monotone | cpu/f64', async () => {
    const cfg = configs.find(c => c.device === 'cpu' && c.dtype === DType.Float64);
    if (!cfg) return;
    applyConfig(cfg);

    const randn = gaussianRng(mulberry32(3));
    const ns = 12, n = ns * 8, s = 0.5;
    const nHarmonics = Math.floor(ns / 2);
    const nW = 1 + 2 * nHarmonics;
    const obs = new Float64Array(n);
    for (let t = 0; t < n; t++)
      obs[t] = 3 * Math.sin(2 * Math.PI * t / ns) + s * randn();

    checkLeaks.start();
    const fit = await dlmFit(obs, s, new Array(nW).fill(0.01), DType.Float64, {
      order: 0, trig: nHarmonics, ns,
    });
    const h = ns * 2;
    const fc = await dlmForecast(fit, s, h, DType.Float64);
    checkLeaks.stop();

    assertAllFinite(fc.yhat);
    assertAllFinite(fc.ystd);
    assertMonotoneNonDecreasing(fc.ystd, 'ystd');
    expect(Math.abs((fc.yhat[0] as number) - (fc.yhat[ns] as number))).toBeLessThan(1.0);
  });

  // ── 4. AR(2) model ─────────────────────────────────────────────────────────

  it('AR(2): finite outputs + ystd monotone | cpu/f64', async () => {
    const cfg = configs.find(c => c.device === 'cpu' && c.dtype === DType.Float64);
    if (!cfg) return;
    applyConfig(cfg);

    const phi = [0.6, -0.2];
    const randn = gaussianRng(mulberry32(4));
    const n = 150, s = 1.0;
    const obs = new Float64Array(n);
    const state = [0, 0];
    for (let t = 0; t < n; t++) {
      const x = phi[0] * state[0] + phi[1] * state[1] + 0.3 * randn();
      state[1] = state[0]; state[0] = x;
      obs[t] = x + s * randn();
    }

    checkLeaks.start();
    const fit = await dlmFit(obs, s, [0.09, 0], DType.Float64, { order: 0, arphi: phi });
    const h = 20;
    const fc = await dlmForecast(fit, s, h, DType.Float64);
    checkLeaks.stop();

    assertAllFinite(fc.yhat);
    assertAllFinite(fc.ystd);
    assertMonotoneNonDecreasing(fc.ystd, 'ystd');
    expect(fc.m).toBe(1 + phi.length);
  });

  // ── 5. Covariate model ──────────────────────────────────────────────────────

  it('covariate model: X_forecast affects yhat linearly | cpu/f64', async () => {
    const cfg = configs.find(c => c.device === 'cpu' && c.dtype === DType.Float64);
    if (!cfg) return;
    applyConfig(cfg);

    const randn = gaussianRng(mulberry32(5));
    const n = 80, beta_true = 2.5, s = 0.5;
    const X_train: ArrayLike<number>[] = Array.from({ length: n }, (_, t) => [Math.sin(t * 0.3)]);
    const obs = new Float64Array(n);
    for (let t = 0; t < n; t++)
      obs[t] = 3 + beta_true * (X_train[t] as number[])[0] + s * randn();

    checkLeaks.start();
    const fit = await dlmFit(obs, s, [0.01], DType.Float64, { order: 0 }, X_train);

    const h = 10;
    const X_low  = Array.from({ length: h }, () => [-1.0] as ArrayLike<number>);
    const X_high = Array.from({ length: h }, () => [+1.0] as ArrayLike<number>);
    const fc_low  = await dlmForecast(fit, s, h, DType.Float64, X_low);
    const fc_high = await dlmForecast(fit, s, h, DType.Float64, X_high);
    checkLeaks.stop();

    assertAllFinite(fc_low.yhat);
    assertAllFinite(fc_high.yhat);
    for (let k = 0; k < h; k++)
      expect((fc_high.yhat[k] as number)).toBeGreaterThan((fc_low.yhat[k] as number));
    const diff = (fc_high.yhat[0] as number) - (fc_low.yhat[0] as number);
    expect(diff).toBeGreaterThan(2.0);
    expect(diff).toBeLessThan(10.0);
  });

  // ── 6. Smoke: all device × dtype configs ────────────────────────────────────

  it('all finite + correct shape | all devices', async () => {
    const randn = gaussianRng(mulberry32(99));
    const n = 60, h = 12, s = 1.0;
    const obs = new Float64Array(n);
    let mu = 0;
    for (let t = 0; t < n; t++) { mu += 0.1 * randn(); obs[t] = mu + s * randn(); }

    for (const cfg of configs) {
      applyConfig(cfg);
      const fit = await dlmFit(obs, s, [0.01], cfg.dtype, { order: 0 });
      const fc  = await dlmForecast(fit, s, h, cfg.dtype);

      expect(fc.h, `${cfg.label}: h`).toBe(h);
      expect(fc.m, `${cfg.label}: m`).toBe(1);
      expect(fc.yhat.length, `${cfg.label}: yhat.length`).toBe(h);
      expect(fc.ystd.length, `${cfg.label}: ystd.length`).toBe(h);
      assertAllFinite(fc.yhat);
      assertAllFinite(fc.ystd);
    }
  });
});
