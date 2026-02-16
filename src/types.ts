import { numpy as np, DType } from "@jax-js-nonconsuming/jax";

/** TypedArray type for float data - either Float32Array or Float64Array based on dtype */
export type FloatArray = Float32Array | Float64Array;

/** TypedArray constructor type */
export type FloatArrayConstructor = typeof Float32Array | typeof Float64Array;

/**
 * Result from the DLM smoother function (dlmSmo).
 * All tensor outputs are np.Arrays returned directly from the JIT core.
 * Caller is responsible for reading (.data()) and disposing these arrays.
 * @internal - Used only within the library implementation.
 */
export interface DlmSmoResult {
  /** Smoothed state: level component [n] */
  x_0: np.Array;
  /** Smoothed state: slope component [n] */
  x_1: np.Array;
  /** Smoothed covariance components [n] */
  C_00: np.Array; C_01: np.Array; C_10: np.Array; C_11: np.Array;
  /** Filtered state: level component [n] */
  xf_0: np.Array;
  /** Filtered state: slope component [n] */
  xf_1: np.Array;
  /** Filtered covariance components [n] */
  Cf_00: np.Array; Cf_01: np.Array; Cf_10: np.Array; Cf_11: np.Array;
  /** Filter predictions [n] */
  yhat: np.Array;
  /** Prediction standard deviations [n] */
  ystd: np.Array;
  /** Smoothed state std devs [n] */
  xstd_0: np.Array; xstd_1: np.Array;
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
  /** State transition matrix G (2x2) */
  G: number[][];
  /** Observation matrix F (1x2 flattened) */
  F: number[];
  /** State noise covariance W (2x2) */
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
