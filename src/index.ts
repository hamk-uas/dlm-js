import { DType, numpy as np } from "@jax-js/jax";
import type { DlmSmoResult, DlmFitResult, FloatArray } from "./types";
import { disposeAll, getFloatArrayType } from "./types";

// Public type exports
export type { DlmFitResult, FloatArray } from "./types";

/**
 * DLM Smoother - Kalman filter (forward) + Rauch-Tung-Striebel smoother (backward).
 *
 * Implements the state-space model:
 *
 *   y(t) = F·x(t) + v,    observation equation
 *   x(t) = G·x(t-1) + w,  state transition equation
 *
 * where:
 *   x(1) ~ N(x0, C0)      initial state distribution
 *   v ~ N(0, V)           observation noise
 *   w ~ N(0, W)           state evolution noise
 *
 * The forward Kalman filter computes one-step-ahead predictions.
 * The backward RTS smoother refines estimates using all observations.
 *
 * Reference: Durbin & Koopman (2012), "Time Series Analysis by State Space Methods"
 *
 * @param y - Observations (n×1)
 * @param F - Observation matrix (1×m), maps state to observation
 * @param V_std - Observation noise std devs (n×1)
 * @param x0_data - Initial state mean (m×1 as nested array)
 * @param G - State transition matrix (m×m)
 * @param W - State noise covariance (m×m)
 * @param C0_data - Initial state covariance (m×m as nested array)
 * @param dtype - Computation precision
 * @returns Smoothed and filtered state estimates with diagnostics
 * @internal
 */
const dlmSmo = async (
  y: FloatArray,
  F: np.Array,
  V_std: FloatArray,
  x0_data: number[][],
  G: np.Array,
  W: np.Array,
  C0_data: number[][],
  dtype: DType = DType.Float64
): Promise<DlmSmoResult> => {
  const n = y.length;
  const FA = getFloatArrayType(dtype);

  // ─────────────────────────────────────────────────────────────────────────
  // Storage allocation for Kalman filter outputs
  // ─────────────────────────────────────────────────────────────────────────
  const x_pred: np.Array[] = new Array(n);  // One-step-ahead state predictions (m×1 each)
  const C_pred: np.Array[] = new Array(n);  // Prediction covariances (m×m each)
  const K_array: np.Array[] = new Array(n); // Kalman gains (m×1 each)
  const v_array = new FA(n);                // Innovation residuals (scalar per timestep)
  const Cp_array = new FA(n);               // Innovation covariances (scalar per timestep)

  // Initialize state from prior distribution x(1) ~ N(x0, C0)
  x_pred[0] = np.array(x0_data, { dtype });
  C_pred[0] = np.array(C0_data, { dtype });

  // Precompute F' for reuse in both filter and smoother loops
  const Ft = np.transpose(F.ref);

  // ─────────────────────────────────────────────────────────────────────────
  // Forward Kalman Filter
  // Computes one-step-ahead predictions: x(t|t-1) and C(t|t-1)
  // ─────────────────────────────────────────────────────────────────────────
  for (let i = 0; i < n; i++) {
    const xi = x_pred[i];  // x(t|t-1): predicted state
    const Ci = C_pred[i];  // C(t|t-1): predicted covariance

    // Innovation (prediction error): v(t) = y(t) - F·x(t|t-1)
    const v = np.subtract(np.array([y[i]], { dtype }), np.matmul(F.ref, xi.ref));
    v_array[i] = (await v.ref.data())[0];

    // Innovation covariance: Cp(t) = F·C(t|t-1)·F' + V(t)
    const Cp = np.add(
      np.einsum('ij,jk,lk->il', F.ref, Ci.ref, F.ref),
      np.array([[V_std[i] ** 2]], { dtype })
    );
    Cp_array[i] = (await Cp.ref.data())[0];

    // Kalman gain: K(t) = G·C(t|t-1)·F' / Cp(t)
    // This determines how much the innovation corrects the state prediction
    const K = np.divide(
      np.einsum('ij,jk,lk->il', G.ref, Ci.ref, F.ref),
      Cp_array[i]
    );
    K_array[i] = np.reshape(K.ref, [2, 1]);

    // Propagate state prediction to next timestep (except at final observation)
    if (i < n - 1) {
      // L(t) = G - K(t)·F  (used for covariance propagation)
      const L = np.subtract(G.ref, np.matmul(K.ref, F.ref));

      // State prediction: x(t+1|t) = G·x(t|t-1) + K(t)·v(t)
      x_pred[i + 1] = np.add(
        np.matmul(G.ref, xi.ref),
        np.matmul(K.ref, np.array([[v_array[i]]], { dtype }))
      );

      // Covariance prediction: C(t+1|t) = G·C(t|t-1)·L' + W
      C_pred[i + 1] = np.add(
        np.einsum('ij,jk,lk->il', G.ref, Ci.ref, L.ref),
        W.ref
      );
    }

    disposeAll(K, v, Cp);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Backward RTS (Rauch-Tung-Striebel) Smoother
  // Refines state estimates using future observations: x(t|n) and C(t|n)
  // Uses the "information form" with auxiliary variables r and N
  // ─────────────────────────────────────────────────────────────────────────
  let r = np.array([[0.0], [0.0]], { dtype });   // Smoothing residual (m×1)
  let N = np.array([[0.0, 0.0], [0.0, 0.0]], { dtype }); // Smoothing precision (m×m)
  const x_smooth: np.Array[] = new Array(n);
  const C_smooth: np.Array[] = new Array(n);

  for (let i = n - 1; i >= 0; i--) {
    const xi = x_pred[i];  // x(t|t-1)
    const Ci = C_pred[i];  // C(t|t-1)
    const Ki = K_array[i]; // K(t)

    // L(t) = G - K(t)·F  (transition minus Kalman correction)
    const L = np.subtract(G.ref, np.matmul(Ki.ref, F.ref));

    // F'·Cp⁻¹ — weighted observation operator
    const FtCpInv = np.divide(Ft.ref, Cp_array[i]);

    // Smoother residual update: r(t-1) = F'·Cp⁻¹·v(t) + L'·r(t)
    const r_new = np.add(
      np.multiply(FtCpInv.ref, v_array[i]),
      np.matmul(np.transpose(L.ref), r.ref)
    );

    // Smoother precision update: N(t-1) = F'·Cp⁻¹·F + L'·N(t)·L
    const N_new = np.add(
      np.matmul(FtCpInv, F.ref),
      np.einsum('ji,jk,kl->il', L.ref, N.ref, L.ref)
    );

    // Smoothed state: x(t|n) = x(t|t-1) + C(t|t-1)·r(t-1)
    x_smooth[i] = np.add(xi.ref, np.matmul(Ci.ref, r_new.ref));

    // Smoothed covariance: C(t|n) = C(t|t-1) - C(t|t-1)·N(t-1)·C(t|t-1)
    C_smooth[i] = np.subtract(
      Ci.ref,
      np.einsum('ij,jk,kl->il', Ci.ref, N_new.ref, Ci.ref)
    );

    disposeAll(r, N);
    r = r_new;
    N = N_new;
  }

  // Cleanup temporary arrays
  disposeAll(r, N, Ft);
  for (let i = 0; i < n; i++) K_array[i].dispose();

  // ─────────────────────────────────────────────────────────────────────────
  // Compute output statistics and diagnostics
  // ─────────────────────────────────────────────────────────────────────────
  const yhat = new FA(n);   // Filter predictions: F·x(t|t-1)
  const ystd = new FA(n);   // Prediction std: sqrt(C[0,0] + V²)
  const xstd: [number, number][] = new Array(n);  // Smoothed state stds
  const resid0 = new FA(n); // Raw residuals: y - yhat
  const resid = new FA(n);  // Scaled residuals: resid0 / V
  const resid2 = new FA(n); // Standardized prediction residuals: v / sqrt(Cp)
  let ssy = 0;  // Sum of squared raw residuals
  let lik = 0;  // -2 × log-likelihood (for model comparison)

  for (let i = 0; i < n; i++) {
    // Extract filter prediction (first state component = level)
    const yhat_i = (await x_pred[i].ref.data())[0];
    const C_s = await C_smooth[i].ref.data();

    yhat[i] = yhat_i;
    // State uncertainty: sqrt of diagonal elements of C(t|n)
    xstd[i] = [Math.sqrt(Math.abs(C_s[0])), Math.sqrt(Math.abs(C_s[3]))];
    // Observation uncertainty: state + observation noise
    ystd[i] = Math.sqrt(C_s[0] + V_std[i] ** 2);

    // Residual diagnostics
    const r0 = y[i] - yhat_i;
    resid0[i] = r0;                              // Raw residual
    resid[i] = r0 / V_std[i];                    // Scaled by obs noise
    resid2[i] = v_array[i] / Math.sqrt(Cp_array[i]); // Standardized innovation

    // Accumulate likelihood components
    ssy += r0 * r0;
    lik += (v_array[i] ** 2) / Cp_array[i] + Math.log(Cp_array[i]);
  }

  // Aggregate error metrics
  let s2 = 0, mse = 0, mape = 0;
  for (let i = 0; i < n; i++) {
    s2 += resid[i] ** 2;
    mse += resid2[i] ** 2;
    mape += Math.abs(resid2[i]) / Math.abs(y[i]);
  }
  s2 /= n;   // Residual variance
  mse /= n;  // Mean squared error of standardized residuals
  mape /= n; // Mean absolute percentage error

  return {
    x: x_smooth, C: C_smooth, xf: x_pred, Cf: C_pred,
    yhat, ystd, xstd, resid0, resid, resid2,
    v: v_array, Cp: Cp_array, ssy, s2, nobs: n, lik, mse, mape,
  };
};

/**
 * Fit a local linear trend Dynamic Linear Model (DLM).
 *
 * Implements a two-pass estimation procedure:
 * 1. Initial pass with diffuse prior to estimate starting values
 * 2. Final pass with refined initial state from smoothed estimates
 *
 * The local linear trend model has state x = [level, slope]':
 *   y(t) = level(t) + v(t),           v ~ N(0, s²)
 *   level(t) = level(t-1) + slope(t-1) + w₁(t),  w₁ ~ N(0, w[0]²)
 *   slope(t) = slope(t-1) + w₂(t),    w₂ ~ N(0, w[1]²)
 *
 * System matrices:
 *   F = [1, 0]        (observation extracts level)
 *   G = [[1, 1],      (level evolves with slope)
 *        [0, 1]]      (slope is random walk)
 *   W = diag(w[0]², w[1]²)  (state noise covariance)
 *
 * @param y - Observations (n×1 array)
 * @param s - Observation noise standard deviation
 * @param w - State noise standard deviations [level, slope]
 * @param dtype - Computation precision (default: Float64)
 * @returns Complete model fit with smoothed estimates and diagnostics
 */
export const dlmFit = async (
  y: ArrayLike<number>,
  s: number,
  w: [number, number],
  dtype: DType = DType.Float64
): Promise<DlmFitResult> => {
  const n = y.length;
  const FA = getFloatArrayType(dtype);

  // Convert input to TypedArray if needed
  const yArr = y instanceof FA ? y : FA.from(y);
  // Observation noise std dev (constant for all timesteps)
  const V_std = new FA(n).fill(s);

  // ─────────────────────────────────────────────────────────────────────────
  // Define system matrices for local linear trend model
  // ─────────────────────────────────────────────────────────────────────────
  // State transition: x(t) = G·x(t-1) + w
  const G = np.array([[1.0, 1.0], [0.0, 1.0]], { dtype });
  // Observation: y(t) = F·x(t) + v
  const F = np.array([[1.0, 0.0]], { dtype });
  // State noise covariance
  const W = np.array([[w[0] ** 2, 0.0], [0.0, w[1] ** 2]], { dtype });

  // ─────────────────────────────────────────────────────────────────────────
  // Initialize state with diffuse prior
  // Level initialized to mean of first observations; slope initialized to 0
  // ─────────────────────────────────────────────────────────────────────────
  let sum = 0;
  const count = Math.min(12, n);
  for (let i = 0; i < count; i++) sum += yArr[i];
  const mean_y = sum / count;
  // Initial covariance: large uncertainty (diffuse prior)
  const c0_val = (Math.abs(mean_y) * 0.5) ** 2;
  const c0 = c0_val === 0 ? 1e7 : c0_val;
  const x0_data = [[mean_y], [0.0]];  // [level, slope]
  const C0_data = [[c0, 0.0], [0.0, c0]];

  // ─────────────────────────────────────────────────────────────────────────
  // Pass 1: Initial smoother to refine starting values
  // ─────────────────────────────────────────────────────────────────────────
  const out1 = await dlmSmo(yArr, F.ref, V_std, x0_data, G.ref, W.ref, C0_data, dtype);

  // Update initial state from smoothed estimate at t=1
  const x0_new = await out1.x[0].ref.data();
  const C0_new = await out1.C[0].ref.data();
  const x0_updated = [[x0_new[0]], [x0_new[1]]];
  // Scale initial covariance by 100 for second pass (following MATLAB dlmfit)
  const C0_scaled = [
    [C0_new[0] * 100, C0_new[1] * 100],
    [C0_new[2] * 100, C0_new[3] * 100],
  ];

  // Dispose Pass 1 arrays (no longer needed)
  for (let i = 0; i < n; i++) {
    disposeAll(out1.x[i], out1.C[i], out1.xf[i], out1.Cf[i]);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Pass 2: Final smoother with refined initial state
  // ─────────────────────────────────────────────────────────────────────────
  const out2 = await dlmSmo(yArr, F, V_std, x0_updated, G, W, C0_scaled, dtype);

  // ─────────────────────────────────────────────────────────────────────────
  // Convert np.Array outputs to TypedArray format for API
  // Layout: [state_dim][time] for vectors, [row][col][time] for matrices
  // ─────────────────────────────────────────────────────────────────────────
  const xf = [new FA(n), new FA(n)];         // Filtered states
  const Cf = [[new FA(n), new FA(n)], [new FA(n), new FA(n)]];  // Filtered covariances
  const x = [new FA(n), new FA(n)];          // Smoothed states
  const C = [[new FA(n), new FA(n)], [new FA(n), new FA(n)]];   // Smoothed covariances
  const xstd: FloatArray[] = new Array(n);   // Smoothed state std devs

  for (let i = 0; i < n; i++) {
    // Extract filtered state: xf[state_dim][time]
    const xfi = await out2.xf[i].ref.data();
    xf[0][i] = xfi[0]; xf[1][i] = xfi[1];
    out2.xf[i].dispose();

    // Extract filtered covariance: Cf[row][col][time]
    const Cfi = await out2.Cf[i].ref.data();
    Cf[0][0][i] = Cfi[0]; Cf[0][1][i] = Cfi[1];
    Cf[1][0][i] = Cfi[2]; Cf[1][1][i] = Cfi[3];
    out2.Cf[i].dispose();

    // Extract smoothed state: x[state_dim][time]
    const xi = await out2.x[i].ref.data();
    x[0][i] = xi[0]; x[1][i] = xi[1];
    out2.x[i].dispose();

    // Extract smoothed covariance: C[row][col][time]
    const Ci = await out2.C[i].ref.data();
    C[0][0][i] = Ci[0]; C[0][1][i] = Ci[1];
    C[1][0][i] = Ci[2]; C[1][1][i] = Ci[3];
    out2.C[i].dispose();

    // State std devs: xstd[time][state_dim] (matches MATLAB format)
    xstd[i] = new FA([out2.xstd[i][0], out2.xstd[i][1]]);
  }

  return {
    // State estimates
    xf, Cf, x, C, xstd,
    // System matrices (plain arrays for easy serialization)
    G: [[1.0, 1.0], [0.0, 1.0]],
    F: [1.0, 0.0],
    W: [[w[0] ** 2, 0.0], [0.0, w[1] ** 2]],
    // Input data
    y: yArr, V: V_std,
    // Initial state (after Pass 1 refinement)
    x0: [x0_updated[0][0], x0_updated[1][0]],
    C0: C0_scaled,
    // Covariates (empty for basic model)
    XX: [],
    // Predictions and residuals
    yhat: out2.yhat, ystd: out2.ystd,
    resid0: out2.resid0, resid: out2.resid, resid2: out2.resid2,
    // Diagnostics
    ssy: out2.ssy,   // Sum of squared residuals
    v: out2.v,       // Innovations (filter prediction errors)
    Cp: out2.Cp,     // Innovation covariances
    s2: out2.s2,     // Residual variance
    nobs: out2.nobs, // Number of observations
    lik: out2.lik,   // -2×log-likelihood
    mse: out2.mse,   // Mean squared error
    mape: out2.mape, // Mean absolute percentage error
    class: 'dlmfit',
  };
};
