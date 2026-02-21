/**
 * Test utility functions for dlm-js
 */

import { checkLeaks } from '@hamk-uas/jax-js-nonconsuming';

/**
 * Run `fn` inside a checkLeaks guard. Throws if any np.Array objects leak.
 * Use this to wrap every `dlmFit`/`dlmSmo`/`dlmForecast`/`dlmMLE` call in tests.
 *
 * @example
 * const result = await withLeakCheck(() => dlmFit(y, s, w, dtype));
 */
export const withLeakCheck = async <T>(fn: () => Promise<T>): Promise<T> => {
  const guard = checkLeaks.start();
  try {
    return await fn();
  } finally {
    checkLeaks.stop(guard);
  }
};

/**
 * Filter object by allowed keys (shallow)
 */
export function filterKeys<T extends object>(obj: T, keys: string[]): Partial<T> {
  if (!obj || typeof obj !== 'object') return obj;
  const filtered: Partial<T> = {};
  for (const k of keys) {
    if (Object.prototype.hasOwnProperty.call(obj, k)) {
      (filtered as Record<string, unknown>)[k] = (obj as Record<string, unknown>)[k];
    }
  }
  return filtered;
}

export interface ComparisonResult {
  equal: boolean;
  path?: string;
  a?: unknown;
  b?: unknown;
}

/**
 * Deep comparison with percentage tolerance and path reporting.
 * @param absoluteTolerance - If set, values differing by less than this
 *   are considered equal regardless of relative error (handles near-zero values).
 */
export function deepAlmostEqual(
  a: unknown,
  b: unknown,
  relativeTolerance = 0.001,
  path = '',
  absoluteTolerance = 0,
): ComparisonResult {
  if (typeof a === 'number' && typeof b === 'number') {
    if (isNaN(a) && isNaN(b)) return { equal: true };
    if (!isFinite(a) || !isFinite(b)) return { equal: a === b, path, a, b };
    const diff = Math.abs(a - b);
    if (absoluteTolerance > 0 && diff < absoluteTolerance) return { equal: true };
    const maxAbs = Math.max(Math.abs(a), Math.abs(b), 1e-12);
    if (diff / maxAbs > relativeTolerance) {
      return { equal: false, path, a, b };
    }
    return { equal: true };
  }
  // Normalize TypedArrays to plain arrays for comparison
  const aArr = ArrayBuffer.isView(a) ? Array.from(a as Float64Array) : a;
  const bArr = ArrayBuffer.isView(b) ? Array.from(b as Float64Array) : b;
  if (Array.isArray(aArr) && Array.isArray(bArr) && aArr.length === bArr.length) {
    for (let i = 0; i < aArr.length; i++) {
      const res = deepAlmostEqual(aArr[i], bArr[i], relativeTolerance, `${path}[${i}]`, absoluteTolerance);
      if (!res.equal) return res;
    }
    return { equal: true };
  }
  if (aArr && bArr && typeof aArr === 'object' && typeof bArr === 'object') {
    const aKeys = Object.keys(aArr);
    const bKeys = Object.keys(bArr);
    if (aKeys.length !== bKeys.length) {
      return { equal: false, path: path + ' (key length mismatch)', a: aKeys, b: bKeys };
    }
    for (const k of aKeys) {
      const res = deepAlmostEqual(
        (aArr as Record<string, unknown>)[k],
        (bArr as Record<string, unknown>)[k],
        relativeTolerance,
        path ? `${path}.${k}` : k,
        absoluteTolerance,
      );
      if (!res.equal) return res;
    }
    return { equal: true };
  }
  if (aArr !== bArr) {
    return { equal: false, path, a, b };
  }
  return { equal: true };
}

/**
 * Normalize MATLAB JSON output to match JS output format.
 *
 * MATLAB/Octave collapses 1×n matrices to flat vectors and m×m matrices
 * to flat nested arrays — for m=1 models the shapes differ from the JS output.
 * This mirrors the helper in gensys.test.ts for shared use.
 *
 * Call with the parsed (and null-normalised) reference object.
 */
export function normalizeMatlabOutput(
  obj: Record<string, unknown>,
  m: number,
): Record<string, unknown> {
  const result = { ...obj };

  // Scalar 1×1 fields that MATLAB squeezes to numbers
  if (typeof result.G === 'number') result.G = [[result.G]];
  if (typeof result.F === 'number') result.F = [result.F];
  if (typeof result.W === 'number') result.W = [[result.W]];
  if (typeof result.x0 === 'number') result.x0 = [result.x0];
  if (typeof result.C0 === 'number') result.C0 = [[result.C0]];

  // State arrays: MATLAB m=1 gives flat [n] instead of [[...]] (1 row × n timesteps)
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
  // xstd: MATLAB gives [n] (one std per timestep) instead of [[std1],[std2],...] for m=1
  if (m === 1 && Array.isArray(result.xstd) && typeof result.xstd[0] === 'number') {
    result.xstd = (result.xstd as number[]).map((v: number) => [v]);
  }

  // Remove MATLAB-only fields not present in JS output
  for (const k of ['options', 's', 'ss', 'xr', 'xrd', 'xrp', 'yrp']) {
    delete result[k];
  }

  return result;
}

/**
 * Recursively replace JSON `null` values with `NaN`.
 *
 * Octave's jsonencode serialises NaN as `null` (JSON has no NaN literal).
 * This helper normalises a parsed reference object so that NaN-aware numeric
 * comparisons (e.g. deepAlmostEqual's `isNaN(a) && isNaN(b)` branch) work
 * correctly when comparing JS TypedArray outputs against Octave references.
 */
export function normalizeNulls(val: unknown): unknown {
  if (val === null) return NaN;
  if (Array.isArray(val)) return val.map(normalizeNulls);
  if (val && typeof val === 'object') {
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(val)) {
      out[k] = normalizeNulls((val as Record<string, unknown>)[k]);
    }
    return out;
  }
  return val;
}

/**
 * Check if an array is close to expected values
 */
export async function arrayClose(
  arr: any,
  expected: number[],
  tolerance = 1e-6
): Promise<void> {
  const { expect } = await import('vitest');
  // Use data() method to get the underlying data array
  const data = await arr.data();
  expect(data.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(data[i]).toBeCloseTo(expected[i], -Math.log10(tolerance));
  }
}
