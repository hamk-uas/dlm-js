import { numpy as np } from "@jax-js/jax";
import type { DType } from "@jax-js/jax";

/**
 * Result from the DLM smoother function (dlmSmo).
 * Contains both filtered and smoothed estimates.
 * @internal - Used only within the library implementation.
 */
export interface DlmSmoResult {
  /** Smoothed state means - array of 2x1 np.Arrays */
  x: np.Array[];
  /** Smoothed state covariances - array of 2x2 np.Arrays */
  C: np.Array[];
  /** Filtered (one-step-ahead prediction) state means - array of 2x1 np.Arrays */
  xf: np.Array[];
  /** Filtered (one-step-ahead prediction) state covariances - array of 2x2 np.Arrays */
  Cf: np.Array[];
  /** Filter predictions (F * x_pred) */
  yhat: number[];
  /** Prediction standard deviations */
  ystd: number[];
  /** Smoothed state standard deviations [sqrt(C[0,0]), sqrt(C[1,1])] per timestep */
  xstd: [number, number][];
  /** Raw residuals (y - yhat) */
  resid0: number[];
  /** Scaled residuals (resid0 / V) */
  resid: number[];
  /** Standardized prediction residuals (v / sqrt(Cp)) */
  resid2: number[];
  /** Innovations (prediction errors) */
  v: number[];
  /** Innovation covariances */
  Cp: number[];
  /** Sum of squared raw residuals */
  ssy: number;
  /** Residual variance */
  s2: number;
  /** Number of observations */
  nobs: number;
  /** -2 * log likelihood */
  lik: number;
  /** Mean squared error of standardized residuals */
  mse: number;
  /** Mean absolute percentage error */
  mape: number;
}

/**
 * Result from the DLM fit function (dlmFit).
 * All arrays are plain JavaScript arrays (no np.Arrays).
 */
export interface DlmFitResult {
  /** Filtered state means [state_dim][time] */
  xf: number[][];
  /** Filtered covariances [row][col][time] */
  Cf: number[][][];
  /** Smoothed state means [state_dim][time] */
  x: number[][];
  /** Smoothed covariances [row][col][time] */
  C: number[][][];
  /** Smoothed state standard deviations [time][state_dim] */
  xstd: number[][];
  /** State transition matrix G (2x2) */
  G: number[][];
  /** Observation matrix F (1x2 flattened) */
  F: number[];
  /** State noise covariance W (2x2) */
  W: number[][];
  /** Observations */
  y: number[];
  /** Observation noise standard deviations */
  V: number[];
  /** Initial state mean (after first smoother pass) */
  x0: number[];
  /** Initial state covariance (scaled) */
  C0: number[][];
  /** Covariates (empty for basic model) */
  XX: number[];
  /** Filter predictions */
  yhat: number[];
  /** Prediction standard deviations */
  ystd: number[];
  /** Raw residuals */
  resid0: number[];
  /** Scaled residuals */
  resid: number[];
  /** Sum of squared residuals */
  ssy: number;
  /** Innovations */
  v: number[];
  /** Innovation covariances */
  Cp: number[];
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
  resid2: number[];
  /** Class identifier */
  class: 'dlmfit';
}

/**
 * Helper to dispose multiple np.Arrays at once.
 */
export function disposeAll(...arrays: (np.Array | undefined | null)[]): void {
  for (const arr of arrays) {
    if (arr) {
      arr.dispose();
    }
  }
}
