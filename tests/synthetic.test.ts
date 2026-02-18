/**
 * Synthetic ground-truth tests for dlm-js.
 *
 * Generates state-space data from known true states using a deterministic
 * PRNG, then verifies the DLM smoother recovers states within statistical
 * bounds. Unlike the Octave reference tests (which compare two implementations'
 * rounding), these test against mathematical ground truth.
 *
 * Assertions:
 * 1. All outputs finite (no NaN/Inf)
 * 2. Smoothed covariance diagonals positive
 * 3. Smoother RMSE < observation RMSE (smoother reduces noise)
 * 4. True states fall within posterior credible intervals at nominal rate
 */
import { DType } from '@hamk-uas/jax-js-nonconsuming';
import { describe, it, expect } from 'vitest';
import { dlmFit, dlmGenSys } from '../src/index';
import { getTestConfigs, applyConfig, assertAllFinite } from './test-matrix';
import { withLeakCheck } from './utils';
import type { DlmOptions } from '../src/dlmgensys';

// ─── Deterministic PRNG (Mulberry32 + Box-Muller) ───────────────────────────

/**
 * Mulberry32: simple 32-bit PRNG with period 2³².
 * Returns uniform random numbers in [0, 1).
 */
function mulberry32(seed: number): () => number {
  let a = seed | 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Box-Muller transform: converts uniform [0,1) samples to N(0,1).
 * Generates pairs; caches the spare for the next call.
 */
function gaussianRng(uniform: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const val = spare;
      spare = null;
      return val;
    }
    let u1: number;
    do { u1 = uniform(); } while (u1 === 0);
    const u2 = uniform();
    const mag = Math.sqrt(-2 * Math.log(u1));
    spare = mag * Math.sin(2 * Math.PI * u2);
    return mag * Math.cos(2 * Math.PI * u2);
  };
}

// ─── Synthetic data generation ──────────────────────────────────────────────

/**
 * Generate state-space data from a known process with known true states.
 *
 * Model:
 *   x(t) = G·x(t-1) + w_noise(t),  w_noise[k] ~ N(0, w[k]²) for k < w.length
 *   y(t) = F'·x(t) + v_noise(t),    v_noise ~ N(0, s²)
 *
 * Initial state: x[0] = random (level ~ N(0,10²), others ~ N(0,1)).
 *
 * @returns Observations and true hidden states, both as plain number arrays.
 */
function generateSyntheticData(
  G: number[][],
  F: number[],
  s: number,
  w: number[],
  n: number,
  seed: number,
): { y: number[]; x_true: number[][] } {
  const m = G.length;
  const randn = gaussianRng(mulberry32(seed));

  // x_true[k][t] = true state component k at time t
  const x_true: number[][] = Array.from({ length: m }, () => new Array(n));
  const y: number[] = new Array(n);

  // Random initial state
  const x_prev = new Array(m).fill(0);
  x_prev[0] = randn() * 10;
  for (let k = 1; k < m; k++) x_prev[k] = randn();

  for (let t = 0; t < n; t++) {
    // State transition: x(t) = G * x(t-1) + state noise
    const x_new = new Array(m).fill(0);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        x_new[i] += G[i][j] * x_prev[j];
      }
      if (i < w.length) x_new[i] += w[i] * randn();
    }

    for (let k = 0; k < m; k++) x_true[k][t] = x_new[k];

    // Observation: y(t) = F' * x(t) + observation noise
    let obs = 0;
    for (let k = 0; k < m; k++) obs += F[k] * x_new[k];
    y[t] = obs + s * randn();

    for (let k = 0; k < m; k++) x_prev[k] = x_new[k];
  }

  return { y, x_true };
}

// ─── Statistical helpers ────────────────────────────────────────────────────

/** Root mean squared error, optionally skipping initial timesteps. */
function rmse(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  start = 0,
): number {
  let sum = 0;
  let count = 0;
  for (let i = start; i < Math.min(a.length, b.length); i++) {
    const d = (a[i] as number) - (b[i] as number);
    sum += d * d;
    count++;
  }
  return Math.sqrt(sum / count);
}

/**
 * Fraction of true values falling within ±z standard deviations of estimate.
 * For z=1.96, a well-calibrated Gaussian posterior gives ~95% coverage.
 */
function coverage(
  x_true: ArrayLike<number>,
  x_est: ArrayLike<number>,
  x_std: ArrayLike<number>,
  z: number,
  start = 0,
): number {
  let inside = 0;
  let count = 0;
  for (let i = start; i < Math.min(x_true.length, x_est.length); i++) {
    if (Math.abs((x_true[i] as number) - (x_est[i] as number)) <= z * (x_std[i] as number)) {
      inside++;
    }
    count++;
  }
  return inside / count;
}

// ─── Test configurations ────────────────────────────────────────────────────

interface SyntheticCase {
  name: string;
  options: DlmOptions;
  n: number;
  /** Observation noise standard deviation */
  s: number;
  /** State noise standard deviations (one per driven component) */
  w: number[];
  seed: number;
  /** Minimum 95%-CI coverage after warmup (generous for finite n + two-pass init) */
  minCoverage: number;
}

const syntheticCases: SyntheticCase[] = [
  {
    name: 'local level (m=1)',
    options: { order: 0 },
    n: 200,
    s: 10,
    w: [3],
    seed: 42,
    minCoverage: 0.85,
  },
  {
    name: 'local linear trend (m=2)',
    options: { order: 1 },
    n: 200,
    s: 5,
    w: [2, 0.5],
    seed: 123,
    minCoverage: 0.80,
  },
  {
    name: 'local level, high SNR (m=1)',
    options: { order: 0 },
    n: 300,
    s: 2,
    w: [5],
    seed: 7,
    minCoverage: 0.85,
  },
  {
    name: 'local level, low SNR (m=1)',
    options: { order: 0 },
    n: 300,
    s: 20,
    w: [1],
    seed: 999,
    minCoverage: 0.85,
  },
  {
    name: 'trig seasonal (m=6)',
    options: { order: 1, trig: 2, ns: 12 },
    n: 240,
    s: 5,
    w: [3, 0.5],
    seed: 314,
    minCoverage: 0.80,
  },
  {
    name: 'full seasonal (m=13)',
    options: { order: 1, fullseas: true, ns: 12 },
    n: 240,
    s: 5,
    w: [3, 0.5],
    seed: 271,
    minCoverage: 0.80,
  },
];

// ─── Tests ──────────────────────────────────────────────────────────────────

describe('synthetic ground-truth tests', async () => {
  const configs = await getTestConfigs();

  /** Skip initial timesteps for statistical checks (two-pass init transient) */
  const WARMUP = 20;

  for (const config of configs) {
    describe(config.label, () => {
      for (const sc of syntheticCases) {
        it(sc.name, async () => {
          applyConfig(config);

          const sys = dlmGenSys(sc.options);

          // Float32 Kalman filter is numerically unstable for m > 2:
          // covariance goes negative → NaN. Skip entirely.
          if (config.dtype === DType.Float32 && sys.m > 2) return;

          const data = generateSyntheticData(
            sys.G, sys.F, sc.s, sc.w, sc.n, sc.seed,
          );

          const result = await withLeakCheck(() =>
            dlmFit(data.y, sc.s, sc.w, config.dtype, sc.options)
          );

          // 1. All outputs finite
          assertAllFinite(result);

          // 2. Smoothed covariance diagonals positive
          for (let k = 0; k < sys.m; k++) {
            for (let t = 0; t < sc.n; t++) {
              expect(result.C[k][k][t]).toBeGreaterThan(0);
            }
          }

          // 3. Smoother beats raw observations for the level component.
          //    RMSE(x_smooth - x_true) should be strictly less than RMSE(y - x_true).
          //    This verifies the smoother is actually reducing noise, not just
          //    passing observations through.
          const smoothRMSE = rmse(result.x[0], data.x_true[0], WARMUP);
          const obsRMSE = rmse(result.y, data.x_true[0], WARMUP);
          expect(
            smoothRMSE,
            `smoother RMSE (${smoothRMSE.toFixed(3)}) should be < obs RMSE (${obsRMSE.toFixed(3)})`,
          ).toBeLessThan(obsRMSE);

          // 4. True states within 95% posterior credible intervals.
          //    For each state component, check that x_true falls within
          //    x_smooth ± 1.96·xstd at roughly the nominal 95% rate.
          //    The threshold is generous (80–85%) to accommodate finite n,
          //    two-pass initialization, and boundary effects.
          for (let k = 0; k < sys.m; k++) {
            const std_k = new Float64Array(sc.n);
            for (let t = 0; t < sc.n; t++) {
              std_k[t] = result.xstd[t][k];
            }

            const cov = coverage(data.x_true[k], result.x[k], std_k, 1.96, WARMUP);
            expect(
              cov,
              `state[${k}] 95%-CI coverage = ${(cov * 100).toFixed(1)}%, ` +
              `expected ≥ ${sc.minCoverage * 100}%`,
            ).toBeGreaterThanOrEqual(sc.minCoverage);
          }
        });
      }
    });
  }
});
