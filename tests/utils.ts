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
  if (Array.isArray(a) && Array.isArray(b) && a.length === b.length) {
    for (let i = 0; i < a.length; i++) {
      const res = deepAlmostEqual(a[i], b[i], relativeTolerance, `${path}[${i}]`, absoluteTolerance);
      if (!res.equal) return res;
    }
    return { equal: true };
  }
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    const aKeys = Object.keys(a);
    const bKeys = Object.keys(b);
    if (aKeys.length !== bKeys.length) {
      return { equal: false, path: path + ' (key length mismatch)', a: aKeys, b: bKeys };
    }
    for (const k of aKeys) {
      const res = deepAlmostEqual(
        (a as Record<string, unknown>)[k],
        (b as Record<string, unknown>)[k],
        relativeTolerance,
        path ? `${path}.${k}` : k,
        absoluteTolerance,
      );
      if (!res.equal) return res;
    }
    return { equal: true };
  }
  if (a !== b) {
    return { equal: false, path, a, b };
  }
  return { equal: true };
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
