import { numpy as np, DType } from "@jax-js-nonconsuming/jax";

/** TypedArray type for float data - either Float32Array or Float64Array based on dtype */
export type FloatArray = Float32Array | Float64Array;

/** TypedArray constructor type */
export type FloatArrayConstructor = typeof Float32Array | typeof Float64Array;

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
  /** Number of observations */
  nobs: number;
  /** State dimension */
  m: number;
}

/**
 * Result from the DLM fit function (dlmFit).
 * Numeric arrays use TypedArrays (Float32Array or Float64Array based on dtype).
 */
export interface DlmFitResult {
  /** Filtered state means [state_dim][time] */
  xf: FloatArray[];
  /** Filtered covariances [row][col][time] */
  Cf: FloatArray[][];
  /** Smoothed state means [state_dim][time] */
  x: FloatArray[];
  /** Smoothed covariances [row][col][time] */
  C: FloatArray[][];
  /** Smoothed state standard deviations [time][state_dim] */
  xstd: FloatArray[];
  /** State transition matrix G (m×m) */
  G: number[][];
  /** Observation vector F (1×m flattened) */
  F: number[];
  /** State noise covariance W (m×m) */
  W: number[][];
  /** Observations */
  y: FloatArray;
  /** Observation noise standard deviations */
  V: FloatArray;
  /** Initial state mean (after first smoother pass) */
  x0: number[];
  /** Initial state covariance (scaled) */
  C0: number[][];
  /** Covariates (empty for basic model) */
  XX: number[];
  /** Filter predictions */
  yhat: FloatArray;
  /** Prediction standard deviations */
  ystd: FloatArray;
  /** Raw residuals */
  resid0: FloatArray;
  /** Scaled residuals */
  resid: FloatArray;
  /** Sum of squared residuals */
  ssy: number;
  /** Innovations */
  v: FloatArray;
  /** Innovation covariances */
  Cp: FloatArray;
  /** Residual variance */
  s2: number;
  /** Number of observations */
  nobs: number;
  /** -2 * log likelihood */
  lik: number;
  /** Mean squared error */
  mse: number;
  /** Mean absolute percentage error */
  mape: number;
  /** Standardized residuals */
  resid2: FloatArray;
  /** Class identifier */
  class: 'dlmfit';
}

/**
 * Get the appropriate TypedArray constructor based on DType.
 */
export function getFloatArrayType(dtype: DType): FloatArrayConstructor {
  return dtype === DType.Float32 ? Float32Array : Float64Array;
}
