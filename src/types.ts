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
export type DlmAlgorithm = 'scan' | 'assoc';

/**
 * Custom loss function type for MAP estimation.
 * Receives the traceable Kalman loss (scalar np.Array) and the current
 * parameter vector theta (np.Array), and returns a scalar np.Array loss.
 * Must be AD-safe: only use jax-js ops on the inputs.
 */
export type DlmLossFn = (kalmanLoss: np.Array, theta: np.Array) => np.Array;

/** Map user-facing dtype string to internal DType enum */
export function parseDtype(d?: DlmDtype): DType {
  if (d === 'f32') return DType.Float32;
  return DType.Float64; // default
}

/** Get the appropriate TypedArray constructor based on DType. */
export function getFloatArrayType(dtype: DType): FloatArrayConstructor {
  return dtype === DType.Float32 ? Float32Array : Float64Array;
}

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
 * For a model with state dimension m and n observations:
 * - State/covariance tensors are stacked: x [n,m,1], C [n,m,m]
 * - Observation-space diagnostics are [n] vectors
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
  /** Filter predictions yhat = F·xf [n] */
  yhat: np.Array;
  /** Prediction standard deviations [n] */
  ystd: np.Array;
  /** Innovations [n] */
  v: np.Array;
  /** Innovation covariances [n] */
  Cp: np.Array;
  /** Raw residuals [n] */
  resid0: np.Array;
  /** Scaled residuals [n] */
  resid: np.Array;
  /** Standardized residuals [n] */
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
  /** Observation vector F (length m) */
  F: number[];
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
  /** Observation vector */
  F: number[];
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
 * Incremental stabilization flags for the Float32 backward smoother.
 *
 * Each flag adds one operation on top of the default joseph+symmetrize baseline.
 * All flags are f32-only (silently ignored for f64) and apply to the sequential
 * scan path only (the assoc/WebGPU path has its own internal stabilization).
 *
 * Use {@link scripts/stab-search.ts} to run a greedy search finding the best
 * combination for your model dimension and series length.
 */
export interface DlmStabilization {
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
   * Add 1e-6·I to C_smooth after symmetrize.
   * Stronger PSD guarantee than cDiag (shifts all eigenvalues up by 1e-6)
   * at the cost of a small bias in the smoothed covariance estimates.
   */
  cEps?: boolean;
}

/**
 * Options for {@link dlmFit} and dlmFitTensor.
 */
export interface DlmFitOptions {
  // ── Noise (required) ──

  /** Observation noise std dev: scalar (all timesteps) or per-observation array (length n). */
  obsStd: number | ArrayLike<number>;
  /** Process noise std devs (diagonal of √W). Length determines which states have noise. */
  processStd: number[];

  // ── Model specification (optional, defaults to local linear trend) ──

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

  // ── Covariates ──

  /** Covariate matrix: n rows × q columns. X[t] is the covariate row at time t. */
  X?: ArrayLike<number>[];

  // ── Runtime ──

  /** Computation precision. Default: `'f64'`. */
  dtype?: DlmDtype;
  /** Algorithm selection. `'scan'` = sequential, `'assoc'` = parallel associative scan. Default: auto-select from device/dtype. */
  algorithm?: DlmAlgorithm;
  /**
   * @experimental Incremental Float32 stabilization flags for the backward smoother.
   * Stacks on top of the default joseph+symmetrize. See {@link DlmStabilization}.
   * Ignored when dtype is 'f64' or algorithm is 'assoc'.
   */
  stabilization?: DlmStabilization;
}

/**
 * Options for {@link dlmMLE}.
 */
export interface DlmMleOptions {
  // ── Model specification ──

  /** Polynomial trend order. Default: 1. */
  order?: number;
  /** Number of trigonometric harmonic pairs. In MATLAB DLM, this is `trig`. */
  harmonics?: number;
  /** Seasons per cycle. In MATLAB DLM, this is `ns`. Default: 12. */
  seasonLength?: number;
  /** Full seasonal component. In MATLAB DLM, this is `fullseas`. */
  fullSeasonal?: boolean;
  /** AR coefficients (initial values when fitAr=true). In MATLAB DLM, this is `arphi`. */
  arCoefficients?: number[];
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

  // ── Runtime ──

  /** Computation precision. Default: `'f64'`. */
  dtype?: DlmDtype;
  /** Algorithm selection. Default: auto-select from device/dtype. */
  algorithm?: DlmAlgorithm;
}

/**
 * Options for {@link dlmForecast}.
 */
export interface DlmForecastOptions {
  /** Computation precision (should match dtype used in dlmFit). Default: `'f64'`. */
  dtype?: DlmDtype;
  /** Covariate rows for forecast steps (h rows × q cols). */
  X?: ArrayLike<number>[];
}

// adSafeInv removed in v0.7.8: np.linalg.inv now has a correct VJP.
// All call sites use np.linalg.inv(X) directly.
