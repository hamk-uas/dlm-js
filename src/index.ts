import { DType, numpy as np, lax, jit, tree } from "@jax-js/jax";
import type { DlmSmoResult, DlmFitResult, FloatArray } from "./types";
import { getFloatArrayType } from "./types";

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
  dtype: DType = DType.Float64,
): Promise<DlmSmoResult & Disposable> => {
  const n = y.length;
  const FA = getFloatArrayType(dtype);

  // Stack observations: shape [n, 1, 1] for matmul compatibility
  using y_arr = np.array(Array.from(y).map(yi => [[yi]]), { dtype });
  // Stack V² (variance): shape [n, 1, 1]
  using V2_arr = np.array(Array.from(V_std).map(v => [[v * v]]), { dtype });
  
  // Initial state
  using x0 = np.array(x0_data, { dtype });
  using C0 = np.array(C0_data, { dtype });

  // Initial backward state (zeros)
  using r0 = np.array([[0.0], [0.0]], { dtype });
  using N0 = np.array([[0.0, 0.0], [0.0, 0.0]], { dtype });

  // Precompute F' for reuse in backward step
  using Ft = np.transpose(F);

  // ─────────────────────────────────────────────────────────────────────────
  // Step functions (close over F, G, W, Ft as constants for JIT)
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
    const v = np.subtract(yi, np.matmul(F, xi));
    
    // Innovation covariance: Cp = F·C·F' + V²
    const Cp = np.add(
      np.einsum('ij,jk,lk->il', F, Ci, F),
      V2i
    );
    
    // Kalman gain: K = G·C·F' / Cp
    using GCFt = np.einsum('ij,jk,lk->il', G, Ci, F);
    const K = np.divide(GCFt, Cp);
    
    // L = G - K·F
    using L = np.subtract(G, np.matmul(K, F));
    
    // Next state prediction: x_next = G·x + K·v
    const x_next = np.add(
      np.matmul(G, xi),
      np.matmul(K, v)
    );
    
    // Next covariance: C_next = G·C·L' + W
    const C_next = np.add(
      np.einsum('ij,jk,lk->il', G, Ci, L),
      W
    );
    
    return [
      { x: x_next, C: C_next },
      { x_pred: xi, C_pred: Ci, K, v, Cp } as ForwardY,
    ];
  };
  
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
    using L = np.subtract(G, np.matmul(Ki, F));
    
    // F'·Cp⁻¹
    using FtCpInv = np.divide(Ft, Cpi);
    
    // r_new = F'·Cp⁻¹·v + L'·r
    const r_new = np.add(
      np.multiply(FtCpInv, vi),
      np.matmul(np.transpose(L), r)
    );
    
    // N_new = F'·Cp⁻¹·F + L'·N·L
    const N_new = np.add(
      np.matmul(FtCpInv, F),
      np.einsum('ji,jk,kl->il', L, N, L)
    );
    
    // x_smooth = x_pred + C_pred·r_new
    const x_smooth = np.add(xi, np.matmul(Ci, r_new));
    
    // C_smooth = C_pred - C_pred·N_new·C_pred
    const C_smooth = np.subtract(
      Ci,
      np.einsum('ij,jk,kl->il', Ci, N_new, Ci)
    );
    
    return [{ r: r_new, N: N_new }, { x_smooth, C_smooth }];
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Jittable core: forward Kalman filter + backward RTS smoother +
  // all diagnostics computed with vectorized numpy ops.
  // F, G, W, Ft are captured as constants by the JIT compiler.
  // ─────────────────────────────────────────────────────────────────────────
  
  const core = (
    x0: np.Array, C0: np.Array,
    y_arr: np.Array, V2_arr: np.Array,
    r0: np.Array, N0: np.Array
  ) => {
    // Derive flat [n] inputs for diagnostics
    using y_1d = np.squeeze(y_arr);      // [n] from [n,1,1]
    using V2_flat = np.squeeze(V2_arr);  // [n]
    using V_flat = np.sqrt(V2_flat);     // [n] observation noise std dev

    // ─── Forward Kalman Filter ───
    const [fwdCarry, fwd] = lax.scan(
      forwardStep,
      { x: x0, C: C0 },
      { y: y_arr, V2: V2_arr }
    );
    tree.dispose(fwdCarry);
    
    // ─── Backward RTS Smoother (reversed forward outputs) ───
    using x_pred_rev = np.flip(fwd.x_pred, 0);
    using C_pred_rev = np.flip(fwd.C_pred, 0);
    using K_rev = np.flip(fwd.K, 0);
    using v_rev = np.flip(fwd.v, 0);
    using Cp_rev = np.flip(fwd.Cp, 0);

    const [bwdCarry, bwd] = lax.scan(
      backwardStep,
      { r: r0, N: N0 },
      {
        x_pred: x_pred_rev,
        C_pred: C_pred_rev,
        K: K_rev,
        v: v_rev,
        Cp: Cp_rev,
      }
    );
    tree.dispose(bwdCarry);
    
    using x_smooth = np.flip(bwd.x_smooth, 0);
    using C_smooth = np.flip(bwd.C_smooth, 0);
    tree.dispose(bwd);

    // ─── Extract per-component 1D arrays from stacked tensors ───

    // Filtered state [n,2,1] → [n] per component
    // yhat = F·x_pred = [1,0]·[level,slope]' = level = xf_0
    const xf_0 = fwd.x_pred.slice([], 0, 0);
    const xf_1 = fwd.x_pred.slice([], 1, 0);

    // Filtered covariance [n,2,2] → [n] per element
    const Cf_00 = fwd.C_pred.slice([], 0, 0);
    const Cf_01 = fwd.C_pred.slice([], 0, 1);
    const Cf_10 = fwd.C_pred.slice([], 1, 0);
    const Cf_11 = fwd.C_pred.slice([], 1, 1);

    // Innovations [n,1,1] → [n]
    const v_flat = np.squeeze(fwd.v);
    const Cp_flat = np.squeeze(fwd.Cp);

    // Forward scan outputs fully extracted — dispose
    tree.dispose(fwd);

    // Smoothed state [n,2,1] → [n] per component
    const x_0 = x_smooth.slice([], 0, 0);
    const x_1 = x_smooth.slice([], 1, 0);

    // Smoothed covariance [n,2,2] → [n] per element
    const C_00 = C_smooth.slice([], 0, 0);
    const C_01 = C_smooth.slice([], 0, 1);
    const C_10 = C_smooth.slice([], 1, 0);
    const C_11 = C_smooth.slice([], 1, 1);

    // ─── Diagnostics (all vectorized numpy ops) ───

    // State std devs from smoothed covariance diagonals
    const xstd_0 = np.sqrt(np.abs(C_00));
    const xstd_1 = np.sqrt(np.abs(C_11));
    const ystd = np.sqrt(np.add(C_00, V2_flat));

    // Residuals
    const resid0 = np.subtract(y_1d, xf_0);
    const resid = np.divide(resid0, V_flat);
    const resid2 = np.divide(v_flat, np.sqrt(Cp_flat));

    // Scalar reductions
    const ssy = np.sum(np.square(resid0));
    const lik = np.sum(np.add(
      np.divide(np.square(v_flat), Cp_flat),
      np.log(Cp_flat)
    ));
    const s2 = np.mean(np.square(resid));
    const mse = np.mean(np.square(resid2));
    const mape = np.mean(np.divide(np.abs(resid2), np.abs(y_1d)));
    
    return {
      xf_0, xf_1, Cf_00, Cf_01, Cf_10, Cf_11,
      x_0, x_1, C_00, C_01, C_10, C_11,
      yhat: xf_0, ystd, xstd_0, xstd_1,
      v: v_flat, Cp: Cp_flat,
      resid0, resid, resid2,
      ssy, lik, s2, mse, mape,
    };
  };
  
  // Run core — one jit wrapping both scans + all diagnostics
  const coreResult = await jit(core)(x0, C0, y_arr, V2_arr, r0, N0);

  return tree.makeDisposable({ ...coreResult, nobs: n }) as DlmSmoResult & Disposable;
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
  using G = np.array([[1.0, 1.0], [0.0, 1.0]], { dtype });
  // Observation: y(t) = F·x(t) + v
  using F = np.array([[1.0, 0.0]], { dtype });
  // State noise covariance
  using W = np.array([[w[0] ** 2, 0.0], [0.0, w[1] ** 2]], { dtype });

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
  let x0_0: number, x0_1: number, c00: number, c01: number, c10: number, c11: number;
  { // Block scope — `using` auto-disposes all Pass 1 arrays at block end
    using out1 = await dlmSmo(yArr, F, V_std, x0_data, G, W, C0_data, dtype);
    x0_0 = (await out1.x_0.data())[0];
    x0_1 = (await out1.x_1.data())[0];
    c00 = (await out1.C_00.data())[0];
    c01 = (await out1.C_01.data())[0];
    c10 = (await out1.C_10.data())[0];
    c11 = (await out1.C_11.data())[0];
  }

  const x0_updated = [[x0_0], [x0_1]];
  // Scale initial covariance by 100 for second pass (following MATLAB dlmfit)
  const C0_scaled = [
    [c00 * 100, c01 * 100],
    [c10 * 100, c11 * 100],
  ];

  // ─────────────────────────────────────────────────────────────────────────
  // Pass 2: Final smoother with refined initial state
  // ─────────────────────────────────────────────────────────────────────────
  const out2 = await dlmSmo(yArr, F, V_std, x0_updated, G, W, C0_scaled, dtype);

  // ─────────────────────────────────────────────────────────────────────────
  // Convert np.Array results to TypedArrays via consumeData (read + dispose)
  // ─────────────────────────────────────────────────────────────────────────
  const toFA = async (a: np.Array) => new FA(await a.consumeData() as ArrayLike<number>);
  const toNum = async (a: np.Array) => (await a.consumeData() as ArrayLike<number>)[0];

  // State estimates
  const xf_0 = await toFA(out2.xf_0);
  const xf_1 = await toFA(out2.xf_1);
  const Cf_00 = await toFA(out2.Cf_00);
  const Cf_01 = await toFA(out2.Cf_01);
  const Cf_10 = await toFA(out2.Cf_10);
  const Cf_11 = await toFA(out2.Cf_11);
  const x_0 = await toFA(out2.x_0);
  const x_1 = await toFA(out2.x_1);
  const C_00 = await toFA(out2.C_00);
  const C_01 = await toFA(out2.C_01);
  const C_10 = await toFA(out2.C_10);
  const C_11 = await toFA(out2.C_11);

  // Diagnostics
  const yhat = await toFA(out2.yhat);
  const ystd = await toFA(out2.ystd);
  const xstd_0_data = await toFA(out2.xstd_0);
  const xstd_1_data = await toFA(out2.xstd_1);
  const v = await toFA(out2.v);
  const Cp = await toFA(out2.Cp);
  const resid0 = await toFA(out2.resid0);
  const resid = await toFA(out2.resid);
  const resid2 = await toFA(out2.resid2);

  // Scalar diagnostics
  const ssy = await toNum(out2.ssy);
  const lik = await toNum(out2.lik);
  const s2 = await toNum(out2.s2);
  const mse = await toNum(out2.mse);
  const mape = await toNum(out2.mape);

  // Reconstruct xstd as per-timestep [2] arrays (MATLAB format)
  const xstd: FloatArray[] = new Array(n);
  for (let i = 0; i < n; i++) {
    xstd[i] = new FA([xstd_0_data[i], xstd_1_data[i]]);
  }

  return {
    // State estimates
    xf: [xf_0, xf_1],
    Cf: [[Cf_00, Cf_01], [Cf_10, Cf_11]],
    x: [x_0, x_1],
    C: [[C_00, C_01], [C_10, C_11]],
    xstd,
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
    yhat, ystd, resid0, resid, resid2,
    // Diagnostics
    ssy, v, Cp, s2,
    nobs: out2.nobs,
    lik, mse, mape,    class: 'dlmfit',
  };
};