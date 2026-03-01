import { DType } from "@hamk-uas/jax-js-nonconsuming";
import type { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

// ─── Foundation types ────────────────────────────────────────────────────────

/** TypedArray type for float data - either Float32Array or Float64Array based on dtype */
export type FloatArray = Float32Array | Float64Array;

/** TypedArray constructor type */
export type FloatArrayConstructor = typeof Float32Array | typeof Float64Array;

/** User-facing dtype specification (avoids importing DType from jax-js) */
export type DlmDtype = 'f32' | 'f64';

/** Algorithm selection for the Kalman filter/smoother */
export type DlmAlgorithm = 'scan' | 'assoc' | 'sqrt-assoc' | 'ud';

/**
 * Describes the layout of the `params` vector passed to {@link DlmLossFn}.
 *
 * `params = [s?, w₀, …, w_{m-1}, φ₁, …, φ_p]`  (natural scale)
 * - Observation std `s` is present only when `nObs === 1` (absent when `obsStdFixed` is set).
 * - Process std devs `wᵢ` (`nProcess` elements).
 * - AR coefficients `φⱼ` (`nAr` elements, 0 when `fitAr` is off).
 */
export interface DlmParamMeta {
  /** 1 if observation std (s) is estimated, 0 if fixed. */
  nObs: number;
  /** Number of process std dev parameters (= state dimension m). */
  nProcess: number;
  /** Number of AR coefficients (0 unless `fitAr: true`). */
  nAr: number;
}

/**
 * Custom loss function type for MAP estimation.
 *
 * Receives the Kalman deviance (−2·logL, scalar `np.Array`), the current
 * parameter vector in **natural scale** (`np.Array`), and a layout
 * descriptor ({@link DlmParamMeta}).  Must return a scalar `np.Array`.
 * All operations on the inputs must be AD-safe (jax-js ops only).
 *
 * `params` layout: `[s?, w₀, …, w_{m-1}, φ₁, …, φ_p]`
 * - `s` and `wᵢ` are positive std devs (not log-transformed).
 * - `φⱼ` are unconstrained AR coefficients.
 */
export type DlmLossFn = (deviance: np.Array, params: np.Array, meta: DlmParamMeta) => np.Array;

/** Map user-facing dtype string to internal DType enum */
export function parseDtype(d?: DlmDtype): DType {
  if (d === 'f32') return DType.Float32;
  return DType.Float64; // default
}

/** Get the appropriate TypedArray constructor based on DType. */
export function getFloatArrayType(dtype: DType): FloatArrayConstructor {
  return dtype === DType.Float32 ? Float32Array : Float64Array;
}

// ─── Option validation ──────────────────────────────────────────────────────

/** MATLAB DLM → dlm-js name mapping for helpful error messages. */
const MATLAB_HINTS: Record<string, string> = {
  trig: 'harmonics',
  ns: 'seasonLength',
  fullseas: 'fullSeasonal',
  arphi: 'arCoefficients',
  fitar: 'fitAr',
  sfixed: 'obsStdFixed',
  sFixed: 'obsStdFixed',
};

/**
 * Throw if `opts` contains keys not in `validKeys`.
 * Prevents silent typos and MATLAB-name mismatches (e.g. `trig` vs `harmonics`).
 */
export function checkUnknownKeys(
  opts: Record<string, unknown>,
  validKeys: ReadonlySet<string>,
  fnName: string,
): void {
  for (const key of Object.keys(opts)) {
    if (!validKeys.has(key)) {
      const hint = MATLAB_HINTS[key] ?? MATLAB_HINTS[key.toLowerCase()];
      const suffix = hint
        ? ` (MATLAB DLM name — use '${hint}' instead)`
        : '';
      throw new Error(
        `${fnName}: unknown option '${key}'${suffix}. Valid options: ${[...validKeys].join(', ')}`,
      );
    }
  }
}

/** Valid keys for {@link DlmFitOptions}. */
export const DLM_FIT_KEYS: ReadonlySet<string> = new Set([
  'obsStd', 'processStd',
  'order', 'harmonics', 'seasonLength', 'fullSeasonal', 'arCoefficients', 'spline',
  'F', 'X', 'timestamps',
  'dtype', 'algorithm', 'stabilization',
]);

/** Valid keys for {@link DlmStabilizationFlags}. */
export const DLM_STABILIZATION_KEYS: ReadonlySet<string> = new Set([
  'nSym', 'nDiag', 'nDiagAbs', 'nLeak',
  'cDiag', 'cDiagAbs', 'cTriuSym', 'cSmoAbsDiag',
]);

/** Named stabilization presets for {@link DlmStabilization}. */
export const DLM_STABILIZATION_PRESETS: ReadonlySet<string> = new Set([
  'matlab', 'none',
]);

/** Valid keys for {@link DlmMleOptions}. */
export const DLM_MLE_KEYS: ReadonlySet<string> = new Set([
  'order', 'harmonics', 'seasonLength', 'fullSeasonal', 'arCoefficients', 'fitAr',
  'X', 'init', 'maxIter', 'lr', 'tol', 'obsStdFixed', 'callbacks', 'adamOpts',
  'optimizer', 'naturalOpts', 'loss',
  'dtype', 'algorithm', 'checkpoint',
]);

/** Valid keys for {@link DlmMleOptions.init}. */
export const DLM_MLE_INIT_KEYS: ReadonlySet<string> = new Set([
  'obsStd', 'processStd', 'arCoefficients',
]);

/** Valid keys for {@link DlmMleOptions.callbacks}. */
export const DLM_MLE_CALLBACKS_KEYS: ReadonlySet<string> = new Set([
  'onInit', 'onIteration',
]);

/** Valid keys for {@link DlmMleOptions.adamOpts}. */
export const DLM_MLE_ADAM_KEYS: ReadonlySet<string> = new Set([
  'b1', 'b2', 'eps',
]);

/** Valid keys for {@link DlmMleOptions.naturalOpts}. */
export const DLM_MLE_NATURAL_KEYS: ReadonlySet<string> = new Set([
  'hessian', 'lambdaInit', 'lambdaShrink', 'lambdaGrow', 'fdStep',
]);

/** Valid keys for {@link DlmForecastOptions}. */
export const DLM_FORECAST_KEYS: ReadonlySet<string> = new Set([
  'dtype', 'X', 'obsStd', 'timestamps',
]);

/** Valid keys for {@link DlmOptions} (dlmGenSys). */
export const DLM_GENSYS_KEYS: ReadonlySet<string> = new Set([
  'order', 'fullSeasonal', 'harmonics', 'seasonLength', 'arCoefficients', 'spline', 'fitAr',
]);

/** Valid keys for {@link DlmPriorSpec}. */
export const DLM_PRIOR_KEYS: ReadonlySet<string> = new Set([
  'obsVar', 'processVar', 'arCoef',
]);

// ─── StateMatrix & CovMatrix ────────────────────────────────────────────────

/**
 * Lightweight wrapper around a flat row-major [n, m] TypedArray buffer.
 * Zero-copy construction from `consumeData()` output — no transpose.
 *
 * Provides three access patterns:
 * - `at(t)`: zero-copy subarray view of all states at time t (length m)
 * - `series(i)`: copied time series of state component i (length n)
 * - `get(t, i)`: single scalar element
 */
export class StateMatrix {
  /** Flat row-major [n, m] buffer */
  readonly data: FloatArray;
  /** Number of timesteps */
  readonly n: number;
  /** State dimension */
  readonly m: number;

  constructor(data: FloatArray, n: number, m: number) {
    this.data = data;
    this.n = n;
    this.m = m;
  }

  /** All states at time t — zero-copy subarray view, length m */
  at(t: number): FloatArray {
    return this.data.subarray(t * this.m, (t + 1) * this.m);
  }

  /** Time series of state component i across all timesteps — copied, length n */
  series(i: number): FloatArray {
    const Ctor = this.data.constructor as FloatArrayConstructor;
    const arr = new Ctor(this.n);
    for (let t = 0; t < this.n; t++) arr[t] = this.data[t * this.m + i];
    return arr;
  }

  /** Single element: state i at time t */
  get(t: number, i: number): number {
    return this.data[t * this.m + i];
  }
}

/**
 * Lightweight wrapper around a flat row-major [n, m, m] TypedArray buffer.
 * Zero-copy construction from `consumeData()` output — no transpose.
 *
 * Provides four access patterns:
 * - `at(t)`: zero-copy subarray view of the m×m covariance at time t
 * - `series(i, j)`: copied time series of Cov(i, j) (length n)
 * - `get(t, i, j)`: single scalar element
 * - `variance(t, i)`: diagonal element Var(state_i) at time t
 */
export class CovMatrix {
  /** Flat row-major [n, m, m] buffer */
  readonly data: FloatArray;
  /** Number of timesteps */
  readonly n: number;
  /** State dimension */
  readonly m: number;

  constructor(data: FloatArray, n: number, m: number) {
    this.data = data;
    this.n = n;
    this.m = m;
  }

  /** m×m covariance at time t — zero-copy subarray view, length m*m, row-major */
  at(t: number): FloatArray {
    const mm = this.m * this.m;
    return this.data.subarray(t * mm, (t + 1) * mm);
  }

  /** Single element: Cov(i, j) at time t */
  get(t: number, i: number, j: number): number {
    return this.data[t * this.m * this.m + i * this.m + j];
  }

  /** Var(state_i) at time t — shorthand for get(t, i, i) */
  variance(t: number, i: number): number {
    return this.data[t * this.m * this.m + i * this.m + i];
  }

  /** Time series of Cov(i, j) across all t — copied, length n */
  series(i: number, j: number): FloatArray {
    const mm = this.m * this.m;
    const Ctor = this.data.constructor as FloatArrayConstructor;
    const arr = new Ctor(this.n);
    for (let t = 0; t < this.n; t++) arr[t] = this.data[t * mm + i * this.m + j];
    return arr;
  }
}

// ─── Internal smoother result (not exported) ────────────────────────────────

/**
 * Result from the DLM smoother function (dlmSmo).
 * All tensor outputs are np.Arrays returned directly from the JIT core.
 * Caller is responsible for reading (.data()) and disposing these arrays.
 *
 * For a model with state dimension m, observation dimension p, and n observations:
 * - State/covariance tensors are stacked: x [n,m,1], C [n,m,m]
 * - Observation-space diagnostics are [n*p] vectors (row-major)
 * - Scalar diagnostics are 0-d tensors
 *
 * @internal - Used only within the library implementation.
 * Internal names are kept short; the rename to JS-idiomatic names happens
 * at the DlmSmoResult → DlmFitResult / DlmTensorResult boundary.
 */
export interface DlmSmoResult {
  /** Smoothed states [n, m, 1] */
  x: np.Array;
  /** Smoothed covariances [n, m, m] */
  C: np.Array;
  /** Filtered (predicted) states [n, m, 1] */
  xf: np.Array;
  /** Filtered covariances [n, m, m] */
  Cf: np.Array;
  /** Filter predictions yhat = F·xf. [n] for p=1, [n*p] row-major for p>1. */
  yhat: np.Array;
  /** Prediction standard deviations. [n] for p=1, [n*p] row-major for p>1. */
  ystd: np.Array;
  /** Innovations. [n] for p=1, [n*p] row-major for p>1. */
  v: np.Array;
  /** Innovation covariances. [n] for p=1, [n*p*p] row-major for p>1. */
  Cp: np.Array;
  /** Raw residuals. [n] for p=1, [n*p] row-major for p>1. */
  resid0: np.Array;
  /** Scaled residuals. [n] for p=1, [n*p] row-major for p>1. */
  resid: np.Array;
  /** Standardized residuals. [n] for p=1, [n*p] row-major for p>1. */
  resid2: np.Array;
  /** Sum of squared raw residuals (scalar) */
  ssy: np.Array;
  /** -2 * log likelihood (scalar) */
  lik: np.Array;
  /** Residual variance (scalar) */
  s2: np.Array;
  /** Mean squared error (scalar) */
  mse: np.Array;
  /** Mean absolute percentage error (scalar) */
  mape: np.Array;
  /** Number of non-NaN observations (scalar np.Array, consumed by dlmFit) */
  nobs: np.Array;
  /** State dimension */
  m: number;
  /** Observation dimension */
  p: number;
}

// ─── Public result types ────────────────────────────────────────────────────

/**
 * Result from dlmFit — materialized TypedArrays with JS-idiomatic names.
 *
 * State estimates use {@link StateMatrix} and {@link CovMatrix} wrappers
 * over contiguous [n, m] / [n, m, m] row-major buffers — zero-copy from
 * the JIT output with no transpose.
 *
 * MATLAB DLM users: call {@link toMatlab} to get the familiar `x[state][time]`
 * layout and single-letter field names.
 */
export interface DlmFitResult {
  // ── State estimates ──

  /** Smoothed state means [n, m]. In MATLAB DLM, this is `x`. */
  smoothed: StateMatrix;
  /** Filtered state means [n, m]. In MATLAB DLM, this is `xf`. */
  filtered: StateMatrix;
  /** Smoothed state covariances [n, m, m]. In MATLAB DLM, this is `C`. */
  smoothedCov: CovMatrix;
  /** Filtered state covariances [n, m, m]. In MATLAB DLM, this is `Cf`. */
  filteredCov: CovMatrix;
  /** Smoothed state standard deviations [n, m] = sqrt(diag(smoothedCov)). In MATLAB DLM, this is `xstd`. */
  smoothedStd: StateMatrix;

  // ── Observation diagnostics (length n) ──

  /** Fitted values: yhat = F · filtered state. */
  yhat: FloatArray;
  /** Prediction standard deviations: sqrt(F·C·F' + V²). */
  ystd: FloatArray;
  /** Innovations (one-step-ahead prediction errors). In MATLAB DLM, this is `v`. */
  innovations: FloatArray;
  /** Innovation variances. In MATLAB DLM, this is `Cp`. */
  innovationVar: FloatArray;
  /** Raw residuals: y - yhat. In MATLAB DLM, this is `resid0`. */
  rawResiduals: FloatArray;
  /** Scaled residuals: (y - yhat) / V. In MATLAB DLM, this is `resid`. */
  scaledResiduals: FloatArray;
  /** Standardized residuals: innovation / sqrt(innovationVar). In MATLAB DLM, this is `resid2`. */
  standardizedResiduals: FloatArray;

  // ── Scalar diagnostics ──

  /** Deviance: -2 · log-likelihood. In MATLAB DLM, this is `lik`. */
  deviance: number;
  /** Residual variance. In MATLAB DLM, this is `s2`. */
  residualVariance: number;
  /** Mean squared error */
  mse: number;
  /** Mean absolute percentage error */
  mape: number;
  /** Residual sum of squares. In MATLAB DLM, this is `ssy`. */
  rss: number;
  /** Number of non-NaN observations */
  nobs: number;

  // ── Model matrices (standard notation) ──

  /** State transition matrix G (m × m) */
  G: number[][];
  /** Observation matrix F. p=1: [m] row vector (backward compat). p>1: [p, m]. */
  F: number[] | number[][];
  /** State noise covariance W (m × m) */
  W: number[][];
  /** Initial state mean (after first smoother pass). In MATLAB DLM, this is `x0`. */
  initialState: number[];
  /** Initial state covariance (scaled). In MATLAB DLM, this is `C0`. */
  initialCov: number[][];
  /** Observations */
  y: FloatArray;
  /** Observation noise standard deviations. In MATLAB DLM, this is `V`. */
  obsNoise: FloatArray;
  /** Covariate matrix X [n × q] (empty array when no covariates). In MATLAB DLM, this is `XX`. */
  covariates: number[][];

  // ── Shape ──

  /** Number of observations */
  n: number;
  /** State dimension (m_base + q for covariates) */
  m: number;
  /** Observation dimension (1 for univariate, >1 for multivariate) */
  p: number;
}

/**
 * On-device tensor result from dlmFitTensor.
 * All arrays are np.Array tensors that stay on-device — no data transfer to JS.
 * Implements Disposable for automatic cleanup via `using`.
 */
export interface DlmTensorResult extends Disposable {
  /** Smoothed states [n, m] */
  smoothed: np.Array;
  /** Filtered states [n, m] */
  filtered: np.Array;
  /** Smoothed covariances [n, m, m] */
  smoothedCov: np.Array;
  /** Filtered covariances [n, m, m] */
  filteredCov: np.Array;
  /** Smoothed state standard deviations [n, m] */
  smoothedStd: np.Array;

  /** Fitted values [n] */
  yhat: np.Array;
  /** Prediction standard deviations [n] */
  ystd: np.Array;
  /** Innovations [n] */
  innovations: np.Array;
  /** Innovation variances [n] */
  innovationVar: np.Array;
  /** Raw residuals [n] */
  rawResiduals: np.Array;
  /** Scaled residuals [n] */
  scaledResiduals: np.Array;
  /** Standardized residuals [n] */
  standardizedResiduals: np.Array;

  /** Deviance: -2 · log-likelihood (scalar) */
  deviance: np.Array;
  /** Residual variance (scalar) */
  residualVariance: np.Array;
  /** Mean squared error (scalar) */
  mse: np.Array;
  /** Mean absolute percentage error (scalar) */
  mape: np.Array;
  /** Residual sum of squares (scalar) */
  rss: np.Array;
  /** Number of non-NaN observations (scalar) */
  nobs: np.Array;

  /** Number of observations */
  n: number;
  /** State dimension */
  m: number;
}

/**
 * MATLAB DLM-compatible result layout and names.
 * Produced by {@link toMatlab}. State arrays use the MATLAB convention:
 * `x[stateIdx][timeIdx]`, `C[i][j][timeIdx]`, `xstd[timeIdx][stateIdx]`.
 */
export interface DlmFitResultMatlab {
  /** Smoothed states: x[state][time] */
  x: FloatArray[];
  /** Filtered states: xf[state][time] */
  xf: FloatArray[];
  /** Smoothed covariances: C[i][j][time] */
  C: FloatArray[][];
  /** Filtered covariances: Cf[i][j][time] */
  Cf: FloatArray[][];
  /** Smoothed state std devs: xstd[time][state] */
  xstd: FloatArray[];

  /** Innovations */
  v: FloatArray;
  /** Innovation covariances */
  Cp: FloatArray;
  /** Raw residuals */
  resid0: FloatArray;
  /** Scaled residuals */
  resid: FloatArray;
  /** Standardized residuals */
  resid2: FloatArray;

  /** -2 · log-likelihood */
  lik: number;
  /** Residual variance */
  s2: number;
  /** Sum of squared residuals */
  ssy: number;

  /** State transition matrix */
  G: number[][];
  /** Observation matrix. p=1: [m] row vector (backward compat). p>1: [p, m]. */
  F: number[] | number[][];
  /** State noise covariance */
  W: number[][];
  /** Observation noise std devs */
  V: FloatArray;
  /** Initial state mean */
  x0: number[];
  /** Initial state covariance */
  C0: number[][];
  /** Covariates matrix */
  XX: number[][];

  /** Observations */
  y: FloatArray;
  /** Fitted values */
  yhat: FloatArray;
  /** Prediction standard deviations */
  ystd: FloatArray;
  /** Mean squared error */
  mse: number;
  /** Mean absolute percentage error */
  mape: number;
  /** Number of non-NaN observations */
  nobs: number;
  /** Number of observations */
  n: number;
  /** State dimension */
  m: number;
  /** Observation dimension */
  p: number;
  /** Class identifier */
  class: 'dlmfit';
}

/**
 * Result from dlmForecast: h-step-ahead predictions with uncertainty.
 * Uses {@link StateMatrix} and {@link CovMatrix} for predicted state trajectories.
 */
export interface DlmForecastResult {
  /** Predicted observation means (length h). yhat[k] = F · predicted state at step k+1. */
  yhat: FloatArray;
  /** Predicted observation std devs (length h). ystd[k] = sqrt(F·C·F' + s²). Monotonically increasing. */
  ystd: FloatArray;
  /** Predicted state means [h, m]. */
  predicted: StateMatrix;
  /** Predicted state covariances [h, m, m]. */
  predictedCov: CovMatrix;
  /** Predicted state std devs [h, m] = sqrt(diag(predictedCov)). */
  predictedStd: StateMatrix;
  /** Forecast horizon */
  h: number;
  /** State dimension */
  m: number;
}

// ─── Options types ──────────────────────────────────────────────────────────

/**
 * Named presets for the `stabilization` option in {@link DlmFitOptions}.
 *
 * - `'matlab'` — MATLAB `dlmsmo.m` exact match: `triu+triu'` symmetrize +
 *   `abs(diag(C_smooth))` after the backward smoother. Works for both f32
 *   and f64. Equivalent to `{ cTriuSym: true, cSmoAbsDiag: true }`.
 * - `'none'` — Disable all optional stabilization flags. For f32, the
 *   unconditional Joseph form + symmetrize + `C += 1e-6·I` baseline still
 *   applies (those cannot be disabled). Equivalent to `{ cTriuSym: false }`.
 */
export type DlmStabilizationPreset = 'matlab' | 'none';

/**
 * Stabilization setting for the sequential scan smoother.
 *
 * - **Preset string** — `'matlab'` or `'none'` (see {@link DlmStabilizationPreset}).
 * - **Flags object** — fine-grained control for research / exploration
 *   (see {@link DlmStabilizationFlags}).
 * - **`undefined`** (omit the option) — library defaults: f64 uses
 *   `triu+triu'` symmetrize; f32 uses Joseph form + `(C+C')/2` + `C += 1e-6·I`.
 */
export type DlmStabilization = DlmStabilizationPreset | DlmStabilizationFlags;

/**
 * Fine-grained stabilization flags for the sequential scan smoother.
 *
 * Most flags are f32-only (silently ignored for f64) and sit on top of the
 * default Joseph + symmetrize + `C += 1e-6·I` baseline. Two flags —
 * `cTriuSym` and `cSmoAbsDiag` — also work for f64: they mirror the
 * stabilization steps in MATLAB's `dlmsmo.m` (`triu+triu'` symmetrize +
 * `abs(diag)` on smoother output) and reduce the f64 max error vs the Octave
 * reference by ~3000×.
 *
 * Applicable to the sequential scan path only; the assoc/WebGPU path has its
 * own internal stabilization.
 *
 * For most users, use the preset strings `'matlab'` or `'none'` instead.
 * Use `pnpm run stab:search:full` to exhaustively search all flag combinations.
 */
export interface DlmStabilizationFlags {
  /**
   * Symmetrize N after each backward step: N = 0.5*(N + N').
   * N is an information matrix and should be symmetric, but f32 rounding in
   * the L'·N·L einsum introduces asymmetries that compound over many steps.
   */
  nSym?: boolean;
  /**
   * Clamp diagonal of N to >= 0 after each backward step.
   * N is an information (Fisher) matrix and should be PSD; negative diagonal
   * entries from f32 rounding cause C·N·C to undercorrect, widening C_smooth.
   */
  nDiag?: boolean;
  /**
   * Take abs of diagonal of N after each backward step: diag(N) = |diag(N)|.
   * Stronger than nDiag: rather than flooring at 0, sign-flips negative entries.
   * Rationale: if rounding makes N_ii barely negative the true value was near
   * zero and the magnitude is still informative; reflecting it back to +|N_ii|
   * preserves the scale of the correction rather than discarding it.
   */
  nDiagAbs?: boolean;
  /**
   * Multiply N by (1 - 1e-5) after each backward step.
   * Slight forgetting that prevents N from accumulating unboundedly over long
   * series, which would cause C·N·C to overshoot and produce negative C_smooth.
   */
  nLeak?: boolean;
  /**
   * Clamp diagonal of C_smooth to >= 1e-7 after symmetrize.
   * Off-diagonal entries (which can legitimately be negative) are left intact.
   * Prevents negative variances from causing NaN in sqrt(diag(C_smooth)).
   */
  cDiag?: boolean;
  /**
   * Take abs of diagonal of C_smooth after symmetrize: diag(C) = |diag(C)|.
   * Unlike cDiag (clamp to 1e-7) this is magnitude-preserving: if catastrophic
   * cancellation yields C_ii = -0.003 the true value was ~+0.003, and abs
   * gives back that physically meaningful value. Off-diagonal entries (which
   * can legitimately be negative, representing correlation) are left intact.
   */
  cDiagAbs?: boolean;
  /**
   * Use `triu(C) + triu(C,1)'` for covariance symmetrization instead of
   * `(C + C') / 2`.  The upper triangle is taken as authoritative and mirrored
   * to the lower — matching MATLAB's `dlmsmo.m` line 77 (`triu(C) + triu(C,1)'`).
   *
   * Unlike `(C+C')/2` which averages both triangles, `triu+triu'` discards the
   * lower triangle entirely. For f64 on the forward filter and backward smoother
   * this is what MATLAB does; combined with `cSmoAbsDiag` it reduces max |Δ|
   * from ~3.78e-8 to ~9e-11 without the Joseph-form overhead.
   *
   * Valid for **both f32 and f64** (unlike most other flags which are f32-only).
   * For f32 it replaces the default `(C+C')/2` symmetrize step.
   * **Defaults to `true` for f64** (matches MATLAB `dlmsmo.m`; ~500× better Octave
   * agreement vs unsymmetrized). Set `cTriuSym: false` to disable.
   */
  cTriuSym?: boolean;
  /**
   * Apply `abs(diag(C_smooth))` after the backward smoother symmetrize step.
   * Matches MATLAB's `dlmsmo.m` lines 114-115.
   *
   * Valid for **both f32 and f64**. For f64, `cTriuSym` is on by default so
   * `cSmoAbsDiag` can be used independently.
   * For f32 it acts on top of the existing symmetrize+cEps default.
   */
  cSmoAbsDiag?: boolean;
}

// ─── Shared model specification ─────────────────────────────────────────────

/**
 * Shared model specification fields used by {@link DlmFitOptions},
 * {@link DlmMleOptions}, and {@link DlmOptions} (dlmGenSys).
 *
 * These fields describe the structural components of the state-space model:
 * polynomial trend, seasonal/harmonic, AR, and spline.  A single
 * `DlmModelSpec` object can be threaded through the full pipeline:
 *
 * ```ts
 * const model: DlmModelSpec = { order: 1, harmonics: 2, seasonLength: 12 };
 * const sys   = dlmGenSys(model);
 * const fit   = await dlmFit(y, { ...model, obsStd: 120, processStd: [40, 10] });
 * const mle   = await dlmMLE(y, model);
 * ```
 */
export interface DlmModelSpec {
  /** Polynomial trend order: 0 (level), 1 (level + slope), 2 (level + slope + acceleration). Default: 1. */
  order?: number;
  /** Number of trigonometric harmonic pairs. In MATLAB DLM, this is `trig`. */
  harmonics?: number;
  /** Seasons per cycle (period length). In MATLAB DLM, this is `ns`. Default: 12. */
  seasonLength?: number;
  /** Full seasonal component (ns-1 dummy variables). In MATLAB DLM, this is `fullseas`. */
  fullSeasonal?: boolean;
  /** AR coefficients (initial values). In MATLAB DLM, this is `arphi`. */
  arCoefficients?: number[];
  /** Spline mode for order=1: modifies W for integrated random walk. */
  spline?: boolean;
}

/**
 * Options for {@link dlmFit} and dlmFitTensor.
 */
export interface DlmFitOptions extends DlmModelSpec {
  // ── Noise (required) ──

  /**
   * Observation noise std dev(s).
   *
   * **Univariate (p=1)**:
   * - `number`: same std for all timesteps.
   * - `ArrayLike<number>` of length n: per-timestep observation noise.
   *
   * **Multivariate (p>1)**: inferred when `F` is provided.
   * - `number`: same std for all series and timesteps → V² = s²·Iₚ.
   * - `number[]` of length p: per-series → V²(t) = diag(s₀², …, sₚ₋₁²).
   * - `number[][]` of shape [n, p]: per-timestep per-series.
   */
  obsStd: number | ArrayLike<number> | number[][];
  /** Process noise std devs (diagonal of √W). Length determines which states have noise. */
  processStd: number[];

  // ── Multivariate observations ──

  /**
   * Observation matrix F [p, m]. When provided, p is inferred from F.length
   * (first dimension) and y must be 2D [n, p].
   * When omitted, F is derived from dlmGenSys (p=1, univariate).
   */
  F?: number[][];

  // ── Covariates ──

  /** Covariate matrix: n rows × q columns. X[t] is the covariate row at time t. */
  X?: ArrayLike<number>[];

  // ── Timestamps ──

  /**
   * Observation timestamps (length n). When provided, G and W become
   * time-varying: G(Δt_k) and W(Δt_k) are computed via closed-form
   * continuous-time discretization for each step Δt_k = t[k] - t[k-1].
   *
   * Supported model components: polynomial trend (order 0, 1, 2) and
   * trigonometric harmonics. Throws if fullSeasonal or AR components
   * are used (these are purely discrete-time constructs).
   *
   * When omitted, all timesteps use Δt = 1 (uniform spacing, equivalent
   * to the standard DLM convention).
   *
   * **Tip — interpolation at query points:** To obtain smoothed estimates
   * at times where no observation exists (e.g., regular grid over an
   * irregular series), insert `NaN` observations at those timestamps.
   * The smoother treats NaN as missing data (pure prediction step with
   * widening covariance), giving you interpolated state estimates and
   * uncertainty bands at arbitrary query points — no separate forecast
   * call needed.
   */
  timestamps?: number[];

  // ── Runtime ──

  /** Computation precision. Default: `'f64'`. */
  dtype?: DlmDtype;
  /** Algorithm selection. `'scan'` = sequential, `'assoc'` = parallel associative scan. Default: auto-select from device/dtype. */
  algorithm?: DlmAlgorithm;
  /**
   * Stabilization mode for the backward smoother.
   *
   * - `'matlab'` — MATLAB `dlmsmo.m` exact match (`cTriuSym` + `cSmoAbsDiag`).
   * - `'none'`   — Disable all optional stabilization flags.
   * - Object     — Fine-grained {@link DlmStabilizationFlags} for research.
   * - `undefined` (omit) — Library defaults per dtype.
   *
   * See {@link DlmStabilization} and {@link DlmStabilizationFlags}.
   */
  stabilization?: DlmStabilization;
}

/**
 * Options for {@link dlmMLE}.
 */
export interface DlmMleOptions extends DlmModelSpec {
  // ── Model specification (inherited from DlmModelSpec) ──

  /** Estimate AR coefficients via MLE. In MATLAB DLM, this is `fitar`. */
  fitAr?: boolean;

  // ── Covariates ──

  /** Covariate matrix: n rows × q columns. */
  X?: ArrayLike<number>[];

  // ── Optimizer ──

  /** Initial parameter guess. */
  init?: { obsStd?: number; processStd?: number[]; arCoefficients?: number[] };
  /** Maximum optimizer iterations. Default: 200. */
  maxIter?: number;
  /** Adam learning rate. Default: 0.05. */
  lr?: number;
  /** Convergence tolerance on relative deviance change. Default: 1e-6. */
  tol?: number;
  /** Per-observation σ array (length n). When provided, obsStd is fixed and not estimated. In MATLAB DLM, this is `sFixed`. */
  obsStdFixed?: ArrayLike<number>;
  /** Callbacks for monitoring optimization progress. */
  callbacks?: {
    /** Called before iteration 0 with the initial theta. */
    onInit?: (theta: FloatArray) => void;
    /** Called after each iteration with updated theta and deviance. */
    onIteration?: (iter: number, theta: FloatArray, deviance: number) => void;
  };
  /** Adam hyperparameters. Default: b1=0.9, b2=0.9, eps=1e-8. */
  adamOpts?: { b1?: number; b2?: number; eps?: number };
  /**
   * Optimizer selection. Default: `'natural'` for f64, `'adam'` for f32.
   * - `'adam'`: optax Adam (first-order, diagonal curvature approximation).
   * - `'natural'`: Newton / Fisher scoring (second-order, full Hessian).
   *   Solves `(H + λI)⁻¹ g` with adaptive Levenberg-Marquardt damping.
   *   Converges in far fewer iterations than Adam but each step costs more.
   *   Best for small parameter spaces (nParams ≤ ~10).
   *   Uses `lr` as step size (default 1.0 for Newton).
   */
  optimizer?: 'adam' | 'natural';
  /**
   * Options for the `'natural'` optimizer.  Ignored when `optimizer` is `'adam'`.
   */
  naturalOpts?: {
    /**
     * How to compute the Hessian.  Default: `'fd'`.
     * - `'fd'`: central finite differences of the JIT'd gradient
     *   (2·nParams extra gradient evaluations per step, no extra JIT trace).
     * - `'exact'`: exact AD Hessian via `jit(hessian(lossFn))`
     *   (`jacfwd(grad)`).  More accurate and avoids FD step-size tuning,
     *   but the first call incurs a large JIT compilation overhead
     *   (~20 s for nParams=2 on WASM as of jax-js v0.7.8).
     *   May become competitive as jax-js JIT improves.
     */
    hessian?: 'fd' | 'exact';
    /** Initial λ scale: `λ₀ = lambdaInit · max(diag(H))`.  Default: `0.1`. */
    lambdaInit?: number;
    /** λ shrink factor on accepted step.  Default: `0.5`. */
    lambdaShrink?: number;
    /** λ grow factor on rejected step.  Default: `2`. */
    lambdaGrow?: number;
    /** Finite-difference step size for `hessian='fd'`.  Default: `1e-5`. */
    fdStep?: number;
  };

  // ── Objective ──

  /**
   * Custom loss function for MAP estimation or other regularised objectives.
   * Default: `'ml'` (standard Kalman prediction-error likelihood).
   *
   * When a function is provided it receives:
   *   1. `deviance` — Kalman −2·logL (scalar `np.Array`).
   *   2. `params`   — natural-scale parameter vector (1-D `np.Array`).
   *   3. `meta`     — {@link DlmParamMeta} describing the params layout.
   *
   * `params` layout: `[s?, w₀, …, w_{m-1}, φ₁, …, φ_p]`
   *   - `s` and `wᵢ` are positive std devs (not log-transformed).
   *   - AR coefficients (when `fitAr`) are unconstrained.
   *   - When `obsStdFixed` is set, the leading `s` slot is absent.
   *
   * The entire chain — Kalman scan + custom penalty + AD backward pass +
   * optimizer update — is wrapped in a single `jit()` call.
   *
   * Tip: use {@link dlmPrior} to create a callback with MATLAB DLM-style
   * Inverse-Gamma priors on variances — no manual coding needed.
   *
   * @example Manual L2 prior on process std devs
   * ```ts
   * const result = await dlmMLE(y, {
   *   order: 1,
   *   loss: (deviance, params, meta) => {
   *     // Split params into [obsStd, processStd] using meta
   *     const parts = np.split(params, [meta.nObs], 0);
   *     const processStd = parts[meta.nObs > 0 ? 1 : 0];
   *     const prior = np.multiply(np.array(0.1), np.sum(np.square(processStd)));
   *     return np.add(deviance, prior);
   *   },
   * });
   * ```
   */
  loss?: 'ml' | DlmLossFn;

  // ── Runtime ──

  /** Computation precision. Default: `'f64'`. */
  dtype?: DlmDtype;
  /** Algorithm selection. Default: auto-select from device/dtype. */
  algorithm?: DlmAlgorithm;
  /**
   * Gradient checkpointing for `lax.scan` backward pass.
   * - `true`: √N segment checkpointing (O(√N) memory, ~2× compute).
   * - `false` (default): store all N carries (O(N) memory, fastest backward pass).
   * - number: explicit segment size.
   *
   * Only affects the sequential scan loss path; ignored for assocScan.
   */
  checkpoint?: boolean | number;
}

/**
 * Options for {@link dlmForecast}.
 */
export interface DlmForecastOptions {
  /** Computation precision (should match dtype used in dlmFit). Default: `'f64'`. */
  dtype?: DlmDtype;
  /** Covariate rows for forecast steps (h rows × q cols). */
  X?: ArrayLike<number>[];
  /**
   * Observation noise std dev (scalar). Overrides the value from the fit result.
   * If omitted, defaults to `fit.obsNoise[0]` (the first observation's noise std dev).
   * Useful for scenario analysis ("what if observation noise were different?").
   */
  obsStd?: number;
  /**
   * Timestamps for forecast steps (length h). When provided, forecast-step
   * intervals Δt_k = timestamps[k] - timestamps[k-1] (or timestamps[0] - last
   * fit timestamp) are used to compute time-varying G(Δt_k) and W(Δt_k).
   * When omitted, all forecast steps use Δt = 1 (uniform spacing).
   */
  timestamps?: number[];
}

// adSafeInv removed in v0.7.8: np.linalg.inv now has a correct VJP.
// All call sites use np.linalg.inv(X) directly.
