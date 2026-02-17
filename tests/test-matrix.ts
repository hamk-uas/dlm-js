/**
 * Shared device × dtype test matrix for dlm-js tests.
 *
 * Provides parameterized test configurations with dtype-specific tolerances.
 * Test files use `describe.each(getTestConfigs())` to run against all
 * available device/dtype combinations.
 */
import { init, defaultDevice, DType, type Device } from '@jax-js-nonconsuming/jax';

/** A single (device, dtype) test configuration with tolerances */
export interface TestConfig {
  device: Device;
  dtype: DType;
  /** Human-readable label for test names */
  label: string;
  /** Relative tolerance for numeric comparison */
  relativeTolerance: number;
  /** Absolute tolerance floor for near-zero values */
  absoluteTolerance: number;
}

/** Tolerances returned by getModelTolerances, or null for smoke-test-only */
export interface Tolerances {
  relativeTolerance: number;
  absoluteTolerance: number;
}

/**
 * All (device, dtype) pairs we'd like to test.
 * webgpu + Float64 is excluded (not supported).
 *
 * Tolerances are based on measured maximum errors across all test models.
 * The tolerances are applied in combination: values pass if EITHER
 *   |a - b| < absoluteTolerance  OR  |a - b| / max(|a|, |b|) < relativeTolerance.
 *
 * Float64 measured maxima (jax-js v0.2.1, Kahan summation for f64 reductions):
 *   - Largest relErr: 4.8e-3 (trig Cf[0][4], m=6, absErr=2.6e-7)
 *   - Seasonal (m=13) improved vs v0.2.0: worst 2.9e-5 → 1.8e-5
 *   - Near-zero values: absErr ≤ 7.1e-9 (trig C[5][4], m=6)
 *   Kahan helps the seasonal model (m=13) but shifts rounding in trig (m=6);
 *   the dominant error source remains catastrophic cancellation in C - C·N·C
 *   which Kahan cannot fix.
 *   Setting: relTol=2e-3, absTol=1e-6 → absTol catches the trig Cf outlier.
 *
 * Float32 measured maxima (m ≤ 2 only; m > 2 diverges):
 *   - m=1: relErr ≤ 7e-6
 *   - m=2: relErr ≤ 6e-3
 *   Setting: relTol=1e-2, absTol=1e-4 → ~1.7x headroom.
 */
const FULL_MATRIX: Omit<TestConfig, 'label'>[] = [
  {
    device: 'cpu',
    dtype: DType.Float64,
    relativeTolerance: 2e-3,
    absoluteTolerance: 1e-6,
  },
  {
    device: 'cpu',
    dtype: DType.Float32,
    relativeTolerance: 1e-2,
    absoluteTolerance: 1e-4,
  },
  {
    device: 'wasm',
    dtype: DType.Float64,
    relativeTolerance: 2e-3,
    absoluteTolerance: 1e-6,
  },
  {
    device: 'wasm',
    dtype: DType.Float32,
    relativeTolerance: 1e-2,
    absoluteTolerance: 1e-4,
  },
  {
    device: 'webgpu',
    dtype: DType.Float32,
    relativeTolerance: 1e-2,
    absoluteTolerance: 1e-4,
  },
  // webgpu + Float64 intentionally omitted — not supported
];

/** Cache so init() is called only once across all test files */
let cachedConfigs: TestConfig[] | null = null;

/**
 * Discover available devices and return the filtered test matrix.
 * Results are cached — safe to call multiple times.
 */
export async function getTestConfigs(): Promise<TestConfig[]> {
  if (cachedConfigs) return cachedConfigs;

  const availableDevices = await init();

  cachedConfigs = FULL_MATRIX
    .filter(c => availableDevices.includes(c.device))
    .map(c => ({
      ...c,
      label: `${c.device}/${c.dtype === DType.Float32 ? 'f32' : 'f64'}`,
    }));

  return cachedConfigs;
}

/**
 * Set the default device for a test config.
 * Call at the start of each test or describe block.
 */
export function applyConfig(config: TestConfig): void {
  defaultDevice(config.device);
}

/**
 * Get tolerances appropriate for a specific model dimension.
 *
 * Float64 is precise across all state dimensions.
 * Float32 Kalman filters accumulate substantial roundoff for m > 2 due to
 * matrix inversions and 2-pass initialization — element-wise comparison
 * against Float64 reference is not meaningful.
 *
 * @returns Tolerances for precision comparison, or null if only a smoke test
 *          (no NaN/Inf, correct shape) is appropriate.
 */
export function getModelTolerances(
  config: TestConfig,
  stateSize: number,
): Tolerances | null {
  if (config.dtype === DType.Float64) {
    return {
      relativeTolerance: config.relativeTolerance,
      absoluteTolerance: config.absoluteTolerance,
    };
  }
  // Float32: small models (m ≤ 2) maintain enough precision for comparison
  if (stateSize <= 2) {
    return {
      relativeTolerance: config.relativeTolerance,
      absoluteTolerance: config.absoluteTolerance,
    };
  }
  // m > 2 in Float32: skip precision comparison, only smoke-test
  return null;
}

/**
 * Walk an object/array tree and throw if any number is NaN or ±Infinity.
 */
export function assertAllFinite(obj: unknown, path = ''): void {
  if (typeof obj === 'number') {
    if (!isFinite(obj)) {
      throw new Error(`Non-finite value at ${path}: ${obj}`);
    }
    return;
  }
  if (Array.isArray(obj)) {
    for (let i = 0; i < obj.length; i++) {
      assertAllFinite(obj[i], `${path}[${i}]`);
    }
    return;
  }
  if (obj && typeof obj === 'object') {
    // Handle TypedArrays
    if (ArrayBuffer.isView(obj)) {
      const arr = obj as Float64Array | Float32Array;
      for (let i = 0; i < arr.length; i++) {
        if (!isFinite(arr[i])) {
          throw new Error(`Non-finite value at ${path}[${i}]: ${arr[i]}`);
        }
      }
      return;
    }
    for (const [k, v] of Object.entries(obj)) {
      assertAllFinite(v, path ? `${path}.${k}` : k);
    }
  }
}
