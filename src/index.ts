import { DType, numpy as np, lax, jit } from "@jax-js/jax";
import type { DlmSmoResult, DlmFitResult, FloatArray } from "./types";
import { disposeAll, getFloatArrayType } from "./types";

// Public type exports
export type { DlmFitResult, FloatArray } from "./types";

/**
 * Execution mode for the DLM smoother.
 * - 'scan': Use lax.scan primitive (better for debugging)
 * - 'jit': Use jit-compiled lax.scan (fast)
 */
export type DlmMode = 'scan' | 'jit';

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
 * @param mode - Execution mode: 'jit' for jitted scan (faster) or 'scan' (better for debugging)
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
  dtype: DType = DType.Float64,
  mode: DlmMode = 'jit'
): Promise<DlmSmoResult> => {
  const n = y.length;
  const FA = getFloatArrayType(dtype);

  // Stack observations: shape [n, 1, 1] for matmul compatibility
  const y_arr = np.array(Array.from(y).map(yi => [[yi]]), { dtype });
  // Stack V² (variance): shape [n, 1, 1]
  const V2_arr = np.array(Array.from(V_std).map(v => [[v * v]]), { dtype });
  
  // Initial state
  const x0 = np.array(x0_data, { dtype });
  const C0 = np.array(C0_data, { dtype });
  
  // Precompute F' for reuse
  const Ft = np.transpose(F.ref);

  // ─────────────────────────────────────────────────────────────────────────
  // Forward Kalman Filter using lax.scan
  // ─────────────────────────────────────────────────────────────────────────
  
  type ForwardCarry = { x: np.Array; C: np.Array };
  type ForwardX = { y: np.Array; V2: np.Array };
  type ForwardY = { x_pred: np.Array; C_pred: np.Array; K: np.Array; v: np.Array; Cp: np.Array };
  
  const forwardStep = (
    carry: ForwardCarry,
    inp: ForwardX
  ): [ForwardCarry, ForwardY] => {
    const { x: xi, C: Ci } = carry;
    const { y: yi, V2: V2i } = inp;
    
    // Innovation: v = y - F·x
    const v = np.subtract(yi.ref, np.matmul(F.ref, xi.ref));
    
    // Innovation covariance: Cp = F·C·F' + V²
    const Cp = np.add(
      np.einsum('ij,jk,lk->il', F.ref, Ci.ref, F.ref),
      V2i.ref
    );
    
    // Kalman gain: K = G·C·F' / Cp
    const GCFt = np.einsum('ij,jk,lk->il', G.ref, Ci.ref, F.ref);
    const K = np.divide(GCFt.ref, Cp.ref);
    
    // L = G - K·F
    const L = np.subtract(G.ref, np.matmul(K.ref, F.ref));
    
    // Next state prediction: x_next = G·x + K·v
    const x_next = np.add(
      np.matmul(G.ref, xi.ref),
      np.matmul(K.ref, v.ref)
    );
    
    // Next covariance: C_next = G·C·L' + W
    const C_next = np.add(
      np.einsum('ij,jk,lk->il', G.ref, Ci.ref, L.ref),
      W.ref
    );
    
    const output: ForwardY = {
      x_pred: xi.ref,
      C_pred: Ci.ref,
      K: K.ref,
      v: v.ref,
      Cp: Cp.ref,
    };
    
    // Note: Don't dispose K, v, Cp - they are returned via .ref in output
    // The scan will manage their lifecycle through the output pytree
    disposeAll(GCFt, L);
    
    return [{ x: x_next, C: C_next }, output];
  };
  
  // Run forward scan (optionally jit-compiled)
  const [finalCarry, forwardOutputs] = mode === 'jit'
    ? await jit((x0: np.Array, C0: np.Array, y_arr: np.Array, V2_arr: np.Array) =>
        lax.scan(forwardStep, { x: x0, C: C0 }, { y: y_arr, V2: V2_arr })
      )(x0, C0, y_arr, V2_arr)
    : lax.scan(forwardStep, { x: x0, C: C0 }, { y: y_arr, V2: V2_arr });
  
  const { x_pred: x_pred_stacked, C_pred: C_pred_stacked, K: K_stacked, v: v_stacked, Cp: Cp_stacked } = forwardOutputs;
  
  // Convert v and Cp to TypedArrays
  const v_data = await v_stacked.ref.data();
  const Cp_data = await Cp_stacked.ref.data();
  const v_array = new FA(n);
  const Cp_array = new FA(n);
  for (let i = 0; i < n; i++) {
    v_array[i] = v_data[i];
    Cp_array[i] = Cp_data[i];
  }
  
  // ─────────────────────────────────────────────────────────────────────────
  // Backward RTS Smoother using lax.scan with reverse
  // ─────────────────────────────────────────────────────────────────────────
  
  const x_pred_rev = np.flip(x_pred_stacked.ref, 0);
  const C_pred_rev = np.flip(C_pred_stacked.ref, 0);
  const K_rev = np.flip(K_stacked.ref, 0);
  const v_rev = np.flip(v_stacked.ref, 0);
  const Cp_rev = np.flip(Cp_stacked.ref, 0);
  
  type BackwardCarry = { r: np.Array; N: np.Array };
  type BackwardX = { x_pred: np.Array; C_pred: np.Array; K: np.Array; v: np.Array; Cp: np.Array };
  type BackwardY = { x_smooth: np.Array; C_smooth: np.Array };
  
  const backwardStep = (
    carry: BackwardCarry,
    inp: BackwardX
  ): [BackwardCarry, BackwardY] => {
    const { r, N } = carry;
    const { x_pred: xi, C_pred: Ci, K: Ki, v: vi, Cp: Cpi } = inp;
    
    // L = G - K·F
    const L = np.subtract(G.ref, np.matmul(Ki.ref, F.ref));
    
    // F'·Cp⁻¹
    const FtCpInv = np.divide(Ft.ref, Cpi.ref);
    
    // r_new = F'·Cp⁻¹·v + L'·r
    const r_new = np.add(
      np.multiply(FtCpInv.ref, vi.ref),
      np.matmul(np.transpose(L.ref), r.ref)
    );
    
    // N_new = F'·Cp⁻¹·F + L'·N·L
    const N_new = np.add(
      np.matmul(FtCpInv.ref, F.ref),
      np.einsum('ji,jk,kl->il', L.ref, N.ref, L.ref)
    );
    
    // x_smooth = x_pred + C_pred·r_new
    const x_smooth = np.add(xi.ref, np.matmul(Ci.ref, r_new.ref));
    
    // C_smooth = C_pred - C_pred·N_new·C_pred
    const C_smooth = np.subtract(
      Ci.ref,
      np.einsum('ij,jk,kl->il', Ci.ref, N_new.ref, Ci.ref)
    );
    
    disposeAll(L, FtCpInv);
    
    return [{ r: r_new, N: N_new }, { x_smooth, C_smooth }];
  };
  
  const r0 = np.array([[0.0], [0.0]], { dtype });
  const N0 = np.array([[0.0, 0.0], [0.0, 0.0]], { dtype });
  
  // Run backward scan (optionally jit-compiled)
  const [finalBackward, backwardOutputs] = mode === 'jit'
    ? await jit((
        r0: np.Array, N0: np.Array,
        x_pred_rev: np.Array, C_pred_rev: np.Array, K_rev: np.Array, v_rev: np.Array, Cp_rev: np.Array
      ) => lax.scan(
        backwardStep,
        { r: r0, N: N0 },
        { x_pred: x_pred_rev, C_pred: C_pred_rev, K: K_rev, v: v_rev, Cp: Cp_rev }
      ))(r0, N0, x_pred_rev, C_pred_rev, K_rev, v_rev, Cp_rev)
    : lax.scan(
        backwardStep,
        { r: r0, N: N0 },
        { x_pred: x_pred_rev, C_pred: C_pred_rev, K: K_rev, v: v_rev, Cp: Cp_rev }
      );
  
  // Reverse smoothed outputs back to forward order
  const x_smooth_stacked = np.flip(backwardOutputs.x_smooth.ref, 0);
  const C_smooth_stacked = np.flip(backwardOutputs.C_smooth.ref, 0);
  
  // Cleanup (lax.scan consumed init and xs inputs)
  disposeAll(finalCarry.x, finalCarry.C, finalBackward.r, finalBackward.N);
  disposeAll(Ft);
  disposeAll(backwardOutputs.x_smooth, backwardOutputs.C_smooth);
  disposeAll(v_stacked, Cp_stacked, K_stacked);
  
  // Convert stacked outputs to per-timestep arrays
  const x_pred: np.Array[] = new Array(n);
  const C_pred: np.Array[] = new Array(n);
  const x_smooth: np.Array[] = new Array(n);
  const C_smooth: np.Array[] = new Array(n);
  
  for (let i = 0; i < n; i++) {
    x_pred[i] = np.reshape(x_pred_stacked.ref.slice(i), [2, 1]);
    C_pred[i] = np.reshape(C_pred_stacked.ref.slice(i), [2, 2]);
    x_smooth[i] = np.reshape(x_smooth_stacked.ref.slice(i), [2, 1]);
    C_smooth[i] = np.reshape(C_smooth_stacked.ref.slice(i), [2, 2]);
  }
  
  disposeAll(x_pred_stacked, C_pred_stacked, x_smooth_stacked, C_smooth_stacked);

  // Compute output statistics
  const yhat = new FA(n);
  const ystd = new FA(n);
  const xstd: [number, number][] = new Array(n);
  const resid0 = new FA(n);
  const resid = new FA(n);
  const resid2 = new FA(n);
  let ssy = 0;
  let lik = 0;

  for (let i = 0; i < n; i++) {
    const yhat_i = (await x_pred[i].ref.data())[0];
    const C_s = await C_smooth[i].ref.data();

    yhat[i] = yhat_i;
    xstd[i] = [Math.sqrt(Math.abs(C_s[0])), Math.sqrt(Math.abs(C_s[3]))];
    ystd[i] = Math.sqrt(C_s[0] + V_std[i] ** 2);

    const r0_val = y[i] - yhat_i;
    resid0[i] = r0_val;
    resid[i] = r0_val / V_std[i];
    resid2[i] = v_array[i] / Math.sqrt(Cp_array[i]);

    ssy += r0_val * r0_val;
    lik += (v_array[i] ** 2) / Cp_array[i] + Math.log(Cp_array[i]);
  }

  let s2 = 0, mse = 0, mape = 0;
  for (let i = 0; i < n; i++) {
    s2 += resid[i] ** 2;
    mse += resid2[i] ** 2;
    mape += Math.abs(resid2[i]) / Math.abs(y[i]);
  }
  s2 /= n;
  mse /= n;
  mape /= n;

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
 * @param mode - Execution mode: 'jit' (faster) or 'scan' (better for debugging)
 * @returns Complete model fit with smoothed estimates and diagnostics
 */
export const dlmFit = async (
  y: ArrayLike<number>,
  s: number,
  w: [number, number],
  dtype: DType = DType.Float64,
  mode: DlmMode = 'jit'
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
  const out1 = await dlmSmo(yArr, F.ref, V_std, x0_data, G.ref, W.ref, C0_data, dtype, mode);

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
  const out2 = await dlmSmo(yArr, F, V_std, x0_updated, G, W, C0_scaled, dtype, mode);

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
