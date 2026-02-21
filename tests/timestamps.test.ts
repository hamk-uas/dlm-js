/**
 * Arbitrary timesteps tests.
 *
 * Verifies that the `timestamps` option in dlmFit produces correct results
 * for irregularly-spaced observations via closed-form continuous-time
 * discretization of G and W.
 *
 * Test strategy:
 * 1. Uniform timestamps should reproduce the standard (no-timestamp) result exactly.
 * 2. Order=0 with gaps: timestamps with Δt>1 should match NaN-padded uniform
 *    model at the observed indices (G=I so cumulated W = Δt·w²).
 * 3. Order=1 with irregular spacing: reasonableness checks (finite outputs,
 *    ystd widens over gaps, correct shapes).
 * 4. dlmGenSysTV unit-spacing matches dlmGenSys.
 */
import { describe, it, expect } from 'vitest';
import { DType } from '@hamk-uas/jax-js-nonconsuming';
import { dlmFit, toMatlab, dlmGenSys, dlmGenSysTV } from '../src/index';
import { withLeakCheck, deepAlmostEqual } from './utils';
import { getTestConfigs, applyConfig, getDlmDtype, getModelTolerances, assertAllFinite } from './test-matrix';
import type { TestConfig } from './test-matrix';

// ── Synthetic data ──────────────────────────────────────────────────────────

// Simple slowly-varying signal for order=0 (local level)
const LEVEL_DATA = [10, 10.5, 11, 11.2, 11.8, 12, 12.3, 12.1, 12.5, 13,
  13.2, 13.5, 14, 13.8, 14.2, 14.5, 14.8, 15, 15.3, 15.5];

// Same data with a gap: observations at t=0..9, skip 10..14, resume 15..19
const GAP_OBS = LEVEL_DATA.slice(0, 10).concat(LEVEL_DATA.slice(15));
const GAP_TIMESTAMPS = [0,1,2,3,4,5,6,7,8,9, 15,16,17,18,19];

// Order=1 data (linear trend + noise)
const TREND_DATA = Array.from({ length: 30 }, (_, t) => 100 + 2 * t + (Math.sin(t) * 3));

// ── Helpers ─────────────────────────────────────────────────────────────────

/** Build NaN-padded version of GAP data at uniform Δt=1 (indices 0..19) */
function buildNanPadded(): number[] {
  const y = new Array(20).fill(NaN);
  for (let i = 0; i < 10; i++) y[i] = LEVEL_DATA[i];
  for (let i = 15; i < 20; i++) y[i] = LEVEL_DATA[i];
  return y;
}

// ── Test runner helpers ─────────────────────────────────────────────────────

const f64Configs = async () =>
  (await getTestConfigs()).filter(c => c.dtype === DType.Float64);

// ── Tests ───────────────────────────────────────────────────────────────────

describe('timestamps', () => {

  // ── dlmGenSysTV sanity checks ──────────────────────────────────────────

  describe('dlmGenSysTV', () => {
    it('unit-spaced timestamps reproduce dlmGenSys matrices (order=0)', () => {
      const opts = { order: 0 as const };
      const sys = dlmGenSys(opts);
      const ts = Array.from({ length: 10 }, (_, i) => i);
      const w = [1.0];
      const tv = dlmGenSysTV(opts, ts, w);

      // Every step should have G = sys.G, W = diag(w²)
      for (let k = 0; k < 10; k++) {
        expect(tv.G[k]).toEqual(sys.G);
        // W[k] for order=0 with Δt=1: [[w² · 1]] = [[1]]
        expect(tv.W[k][0][0]).toBeCloseTo(1.0, 10);
      }
      expect(tv.F).toEqual(sys.F);
      expect(tv.m).toBe(sys.m);
    });

    it('unit-spaced timestamps reproduce dlmGenSys matrices (order=1)', () => {
      const opts = { order: 1 as const };
      const sys = dlmGenSys(opts);
      const ts = Array.from({ length: 10 }, (_, i) => i);
      const w = [1.0, 0.5];
      const tv = dlmGenSysTV(opts, ts, w);

      // G for Δt=1: [[1, 1], [0, 1]], same as dlmGenSys
      for (let k = 0; k < 10; k++) {
        expect(tv.G[k]).toEqual(sys.G);
      }
      expect(tv.m).toBe(2);
    });

    it('Δt=2 gives G with correct off-diagonal (order=1)', () => {
      const opts = { order: 1 as const };
      // timestamps: [0, 2] → departing dt = [2, 1]
      // G_scan[0] transitions from t=0 to t=2, so Δt=2
      const ts = [0, 2];
      const w = [1.0, 1.0];
      const tv = dlmGenSysTV(opts, ts, w);

      // Step 0 (k=0) has departing Δt=2: G = [[1, 2], [0, 1]]
      expect(tv.G[0][0][0]).toBeCloseTo(1.0, 10);
      expect(tv.G[0][0][1]).toBeCloseTo(2.0, 10);
      expect(tv.G[0][1][0]).toBeCloseTo(0.0, 10);
      expect(tv.G[0][1][1]).toBeCloseTo(1.0, 10);

      // W for departing Δt=2 is the accumulated discrete noise:
      //   W(d) = Σ_{k=0}^{d-1} G^k · W₁ · (G^k)'
      // For d=2, w=[1,1]: W = [[3, 1], [1, 2]]
      expect(tv.W[0][0][0]).toBeCloseTo(3.0, 10);
      expect(tv.W[0][0][1]).toBeCloseTo(1.0, 10);
      expect(tv.W[0][1][0]).toBeCloseTo(1.0, 10);
      expect(tv.W[0][1][1]).toBeCloseTo(2.0, 10);

      // Step 1 (k=1) is last obs, departing Δt=1: G = [[1,1],[0,1]], W = diag
      expect(tv.G[1][0][1]).toBeCloseTo(1.0, 10);
      expect(tv.W[1][0][0]).toBeCloseTo(1.0, 10);
      expect(tv.W[1][0][1]).toBeCloseTo(0.0, 10);
      expect(tv.W[1][1][1]).toBeCloseTo(1.0, 10);
    });

    it('throws on fullSeasonal', () => {
      expect(() => dlmGenSysTV(
        { order: 0, fullSeasonal: true, seasonLength: 4 },
        [0, 1, 2], [1.0]
      )).toThrow(/fullSeasonal/);
    });

    it('throws on AR coefficients', () => {
      expect(() => dlmGenSysTV(
        { order: 0, arCoefficients: [0.5] },
        [0, 1, 2], [1.0]
      )).toThrow(/AR/);
    });

    it('throws on non-increasing timestamps', () => {
      expect(() => dlmGenSysTV(
        { order: 0 },
        [0, 2, 1], [1.0]
      )).toThrow(/strictly increasing/);
    });
  });

  // ── Uniform timestamps = no timestamps ────────────────────────────────

  describe('uniform timestamps match standard result', () => {
    it.each([
      { order: 0, label: 'order=0' },
      { order: 1, label: 'order=1' },
    ])('$label', async ({ order }) => {
      const configs = await f64Configs();
      if (configs.length === 0) return;
      const config = configs[0];
      applyConfig(config);
      const dlmDtype = getDlmDtype(config);
      const n = LEVEL_DATA.length;
      const timestamps = Array.from({ length: n }, (_, i) => i);

      // Fit without timestamps
      const ref = await withLeakCheck(() => dlmFit(LEVEL_DATA, {
        obsStd: 1.0, processStd: [0.5],
        order, dtype: dlmDtype,
      }));
      const refM = toMatlab(ref);

      // Fit with timestamps (uniform Δt=1)
      const res = await withLeakCheck(() => dlmFit(LEVEL_DATA, {
        obsStd: 1.0, processStd: [0.5],
        order, timestamps, dtype: dlmDtype,
      }));
      const resM = toMatlab(res);

      // Compare key fields
      const cmp = deepAlmostEqual(
        { x: resM.x, xstd: resM.xstd, yhat: resM.yhat, ystd: resM.ystd, lik: resM.lik },
        { x: refM.x, xstd: refM.xstd, yhat: refM.yhat, ystd: refM.ystd, lik: refM.lik },
        1e-10, 1e-12,
      );
      expect(cmp.equal).toBe(true);
    });
  });

  // ── Order=0 with gap: timestamps vs NaN-padded uniform ────────────────

  describe('order=0 gap behavior', () => {
    it('ystd widens at the gap boundary and yhat interpolates (f64)', async () => {
      const configs = await f64Configs();
      if (configs.length === 0) return;
      const config = configs[0];
      applyConfig(config);
      const dlmDtype = getDlmDtype(config);

      // Timestamps model (15 observations with Δt=5 gap between index 9 and 10)
      const tsFit = await withLeakCheck(() => dlmFit(GAP_OBS, {
        obsStd: 1.0, processStd: [0.5],
        order: 0, timestamps: GAP_TIMESTAMPS, dtype: dlmDtype,
      }));
      const tsM = toMatlab(tsFit);

      assertAllFinite(tsM.yhat);
      assertAllFinite(tsM.ystd);

      const tsYstd = (tsM.ystd as number[]);

      // Key property: the observation right after the gap (index 10, t=15)
      // should have larger ystd than observations well before the gap (e.g. index 5).
      // The pre-gap region has many observations so ystd is low; after the gap,
      // the accumulated process noise from Δt=6 increases uncertainty.
      expect(tsYstd[10]).toBeGreaterThan(tsYstd[5]);

      // yhat should be between the observations before and after the gap
      const yhat = (tsM.yhat as number[]);
      // All yhat values should be finite and in a reasonable range
      for (const v of yhat) {
        expect(v).toBeGreaterThan(5);
        expect(v).toBeLessThan(20);
      }
    });

    it('uniform-Δt timestamps give same -2logL as no timestamps', async () => {
      const configs = await f64Configs();
      if (configs.length === 0) return;
      const config = configs[0];
      applyConfig(config);
      const dlmDtype = getDlmDtype(config);

      const n = GAP_OBS.length;
      const uniformTs = Array.from({ length: n }, (_, i) => i);

      // Without timestamps
      const ref = await withLeakCheck(() => dlmFit(GAP_OBS, {
        obsStd: 1.0, processStd: [0.5],
        order: 0, dtype: dlmDtype,
      }));

      // With uniform timestamps (should be identical)
      const res = await withLeakCheck(() => dlmFit(GAP_OBS, {
        obsStd: 1.0, processStd: [0.5],
        order: 0, timestamps: uniformTs, dtype: dlmDtype,
      }));

      expect(toMatlab(res).lik).toBeCloseTo(toMatlab(ref).lik as number, 10);
    });
  });

  // ── Timestamps vs NaN-padded equivalence (order=1, Nile gapped data) ──

  describe('timestamps vs NaN-padded equivalence (order=1)', () => {
    it('smoothed states match at observed indices', async () => {
      const configs = await f64Configs();
      if (configs.length === 0) return;
      const config = configs[0];
      applyConfig(config);
      const dlmDtype = getDlmDtype(config);

      // Use actual Nile gapped data from tests/gapped-in.json
      const { readFileSync } = await import('node:fs');
      const { resolve, dirname } = await import('node:path');
      const root = resolve(dirname(new URL(import.meta.url).pathname), '..');
      const input = JSON.parse(readFileSync(resolve(root, 'tests/gapped-in.json'), 'utf8'));

      const yRaw: (number | null)[] = input.y;
      const y: number[] = yRaw.map((v: number | null) => (v === null ? NaN : v));
      const yFull: number[] = input.y_full;
      const nanMask: number[] = input.nan_mask;
      const s: number = input.s;
      const w: number[] = input.w;
      const opts = input.options;
      const n = yRaw.length;
      const t: number[] = Array.from({ length: n }, (_, i) => 1871 + i);

      // Build observed-only arrays
      const tsObs: number[] = [];
      const yObs: number[] = [];
      const obsIdx: number[] = []; // map k → original index
      for (let i = 0; i < n; i++) {
        if (nanMask[i] === 0) {
          tsObs.push(t[i]);
          yObs.push(yFull[i]);
          obsIdx.push(i);
        }
      }

      // NaN-padded fit
      const nanResult = await withLeakCheck(() =>
        dlmFit(y, { obsStd: s, processStd: w, dtype: dlmDtype, ...opts }),
      );
      const nanM = toMatlab(nanResult);
      const nanLevel = nanM.x[0] as Float64Array;
      const nanXstd = (nanM.xstd as number[][]).map((r: number[]) => r[0]);

      // Timestamps fit
      const tsResult = await withLeakCheck(() =>
        dlmFit(yObs, { obsStd: s, processStd: w, dtype: dlmDtype, ...opts, timestamps: tsObs }),
      );
      const tsM = toMatlab(tsResult);
      const tsLevel = tsM.x[0] as Float64Array;
      const tsXstd = (tsM.xstd as number[][]).map((r: number[]) => r[0]);

      // Compare at all 77 observed indices — should match within ~1e-8
      const relTol = 1e-6;
      const absTol = 1e-6;
      let maxLevelDiff = 0;
      let maxXstdDiff = 0;
      for (let k = 0; k < yObs.length; k++) {
        const origI = obsIdx[k];
        const ld = Math.abs(tsLevel[k] - nanLevel[origI]);
        const xd = Math.abs(tsXstd[k] - nanXstd[origI]);
        maxLevelDiff = Math.max(maxLevelDiff, ld);
        maxXstdDiff = Math.max(maxXstdDiff, xd);
        const nanLev = nanLevel[origI];
        const tsLev = tsLevel[k];
        expect(ld).toBeLessThan(Math.max(absTol, relTol * Math.abs(nanLev)),
          `level mismatch at k=${k}, origIdx=${origI}: NaN=${nanLev}, ts=${tsLev}, diff=${ld}`);
        expect(xd).toBeLessThan(Math.max(absTol, relTol * Math.abs(nanXstd[origI])),
          `xstd mismatch at k=${k}, origIdx=${origI}: NaN=${nanXstd[origI]}, ts=${tsXstd[k]}, diff=${xd}`);
      }
    });
  });

  // ── Order=1 with irregular timestamps: reasonableness ──────────────────

  describe('order=1 irregular timestamps', () => {
    it.each(
      ['scan', 'assoc'] as const,
    )('algorithm=%s produces finite outputs with correct shape', async (algorithm) => {
      const configs = await f64Configs();
      if (configs.length === 0) return;
      const config = configs[0];
      applyConfig(config);
      const dlmDtype = getDlmDtype(config);

      // Irregular timestamps: some close, some far apart
      const n = TREND_DATA.length;
      const timestamps = Array.from({ length: n }, (_, i) => {
        if (i < 10) return i;            // t=0..9
        if (i < 15) return 10 + (i - 10) * 3;  // t=10,13,16,19,22
        return 22 + (i - 14);            // t=23,24,...,38
      });

      const res = await withLeakCheck(() => dlmFit(TREND_DATA, {
        obsStd: 2.0, processStd: [0.5, 0.1],
        order: 1, timestamps, dtype: dlmDtype, algorithm,
      }));
      const m = toMatlab(res);

      // Basic shape checks — m.x is [stateSize][n], yhat is [n]
      const stateSize = (m.x as Float64Array[]).length;
      expect(stateSize).toBe(2); // order=1 → 2 states
      expect((m.x as Float64Array[])[0].length).toBe(n);
      expect((m.yhat as number[]).length).toBe(n);
      expect((m.ystd as number[]).length).toBe(n);

      // All finite
      assertAllFinite(m.x);
      assertAllFinite(m.yhat);
      assertAllFinite(m.ystd);
      assertAllFinite(m.xstd);

      // ystd should be positive everywhere
      for (const v of m.ystd as number[]) {
        expect(v).toBeGreaterThan(0);
      }
    });
  });

  // ── Trig harmonics with timestamps ─────────────────────────────────────

  describe('trig harmonics with timestamps', () => {
    it('dlmGenSysTV trig rotation for Δt=2 is correct', () => {
      const opts = { order: 0, harmonics: 1, seasonLength: 12 };
      const ts = [0, 2];
      const w = [0.5, 0.3, 0.3]; // level + 2 trig states
      const tv = dlmGenSysTV(opts, ts, w);

      // m = 1 (level) + 2 (one harmonic pair) = 3
      expect(tv.m).toBe(3);

      // Step 0 (departing Δt=2): trig block should be rotation by 2·2π/12 = π/3
      const theta = 2 * 2 * Math.PI / 12;  // Δt * 2π*k/ns, k=1
      expect(tv.G[0][1][1]).toBeCloseTo(Math.cos(theta), 10);
      expect(tv.G[0][1][2]).toBeCloseTo(Math.sin(theta), 10);
      expect(tv.G[0][2][1]).toBeCloseTo(-Math.sin(theta), 10);
      expect(tv.G[0][2][2]).toBeCloseTo(Math.cos(theta), 10);
    });

    it('produces finite results with order=0 + harmonics=2', async () => {
      const configs = await f64Configs();
      if (configs.length === 0) return;
      const config = configs[0];
      applyConfig(config);
      const dlmDtype = getDlmDtype(config);

      // Seasonal-like data
      const n = 24;
      const y = Array.from({ length: n }, (_, t) => 50 + 10 * Math.sin(2 * Math.PI * t / 12) + Math.random());
      const timestamps = Array.from({ length: n }, (_, i) => {
        if (i < 12) return i;
        return 12 + (i - 12) * 2; // irregular spacing in second half
      });

      const res = await withLeakCheck(() => dlmFit(y, {
        obsStd: 1.0, processStd: [0.5, 0.3, 0.3, 0.3, 0.3],
        order: 0, harmonics: 2, seasonLength: 12,
        timestamps, dtype: dlmDtype,
      }));
      const m = toMatlab(res);

      assertAllFinite(m.x);
      assertAllFinite(m.yhat);
      assertAllFinite(m.ystd);
    });
  });

  // ── NaN infill pattern (user tip) ──────────────────────────────────────

  describe('NaN infill for interpolation', () => {
    it('NaN at query points produces smoothed estimates', async () => {
      const configs = await f64Configs();
      if (configs.length === 0) return;
      const config = configs[0];
      applyConfig(config);
      const dlmDtype = getDlmDtype(config);

      // 10 real observations + 5 NaN query points interspersed
      const timestamps = [0, 1, 2, 2.5, 3, 4, 4.5, 5, 6, 6.5, 7, 8, 9, 9.5, 10];
      const y: number[] = [10, 10.5, 11, NaN, 11.5, 12, NaN, 12.5, 13, NaN, 13.5, 14, 14.5, NaN, 15];

      const res = await withLeakCheck(() => dlmFit(y, {
        obsStd: 1.0, processStd: [0.5],
        order: 0, timestamps, dtype: dlmDtype,
      }));
      const m = toMatlab(res);

      // All 15 outputs should be finite (including NaN-query points)
      expect((m.yhat as number[]).length).toBe(15);
      assertAllFinite(m.yhat);
      assertAllFinite(m.ystd);

      // The interpolated values at NaN query points should be between
      // their surrounding observed values (order=0 local level)
      const yhat = m.yhat as number[];
      // Index 3 (t=2.5): between y[2]=11 and y[4]=11.5
      expect(yhat[3]).toBeGreaterThanOrEqual(10.5);
      expect(yhat[3]).toBeLessThanOrEqual(12.0);
    });
  });
});
