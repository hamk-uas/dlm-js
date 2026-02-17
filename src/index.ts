import { DType, numpy as np, lax, jit, tree } from "@hamk-uas/jax-js-nonconsuming";
import type { DlmSmoResult, DlmFitResult, FloatArray } from "./types";
import { getFloatArrayType } from "./types";
import { dlmGenSys } from "./dlmgensys";
import type { DlmOptions } from "./dlmgensys";

// Public type exports
export type { DlmFitResult, FloatArray } from "./types";
export type { DlmOptions, DlmSystem } from "./dlmgensys";
export { dlmGenSys } from "./dlmgensys";
export { dlmMLE } from "./mle";
export type { DlmMleResult } from "./mle";

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
 * @param stateSize - State dimension m
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
  stateSize: number,
  dtype: DType = DType.Float64,
): Promise<DlmSmoResult & Disposable> => {
  const n = y.length;

  // Stack observations: shape [n, 1, 1] for matmul compatibility
  using y_arr = np.array(Array.from(y).map(yi => [[yi]]), { dtype });
  // Stack V² (variance): shape [n, 1, 1]
  using V2_arr = np.array(Array.from(V_std).map(v => [[v * v]]), { dtype });
  
  // Initial state
  using x0 = np.array(x0_data, { dtype });
  using C0 = np.array(C0_data, { dtype });

  // Initial backward state (zeros) — size depends on state dimension m
  const r0_data: number[][] = Array.from({ length: stateSize }, () => [0.0]);
  const N0_data: number[][] = Array.from({ length: stateSize }, () =>
    new Array(stateSize).fill(0.0)
  );
  using r0 = np.array(r0_data, { dtype });
  using N0 = np.array(N0_data, { dtype });

  // Precompute F' for reuse in backward step
  using Ft = np.transpose(F);

  // ─────────────────────────────────────────────────────────────────────────
  // Step functions (close over F, G, W, Ft as constants for JIT)
  // All matrix ops are generic in m — matmul and einsum handle any dimension.
  // ─────────────────────────────────────────────────────────────────────────
  
  type ForwardCarry = { x: np.Array; C: np.Array };
  type ForwardX = { y: np.Array; V2: np.Array };
  type ForwardY = {
    x_pred: np.Array; C_pred: np.Array;
    K: np.Array; v: np.Array; Cp: np.Array;
  };
  
  const forwardStep = (
    carry: ForwardCarry,
    inp: ForwardX
  ): [ForwardCarry, ForwardY] => {
    const { x: xi, C: Ci } = carry;
    const { y: yi, V2: V2i } = inp;
    
    // Innovation: v = y - F·x  [1,1]
    const v = np.subtract(yi, np.matmul(F, xi));
    
    // Innovation covariance: Cp = F·C·F' + V²  [1,1]
    const Cp = np.add(
      np.einsum('ij,jk,lk->il', F, Ci, F),
      V2i
    );
    
    // Kalman gain: K = G·C·F' / Cp  [m,1]
    using GCFt = np.einsum('ij,jk,lk->il', G, Ci, F);
    const K = np.divide(GCFt, Cp);
    
    // L = G - K·F  [m,m]
    using L = np.subtract(G, np.matmul(K, F));
    
    // Next state prediction: x_next = G·x + K·v  [m,1]
    const x_next = np.add(
      np.matmul(G, xi),
      np.matmul(K, v)
    );
    
    // Next covariance: C_next = G·C·L' + W  [m,m]
    //
    // NUMERICAL PRECISION NOTE:
    // einsum('ij,jk,lk->il', G, C, L) decomposes into two sequential
    // dot products: tmp = G·C, then result = tmp·L'. Since jax-js-nonconsuming
    // v0.2.1, Float64 reductions use Kahan compensated summation,
    // reducing per-dot rounding from O(m·ε) to O(ε²). For m = 13
    // (full seasonal) this improved worst-case relative error from
    // ~3e-5 to ~1.8e-5 vs MATLAB. However, the dominant error
    // source remains the subtraction C - C·N·C in the backward
    // step (see below), which Kahan cannot fix.
    //
    // POTENTIAL DLM-SIDE IMPROVEMENT (not yet implemented):
    // The Joseph form covariance update is numerically more stable:
    //   C_next = L·C·L' + K·V²·K' + W
    // It guarantees positive semi-definiteness by construction and
    // avoids the subtraction in G·C·L' = (G - K·F)·C·G' that can
    // cause cancellation. However, it costs more matrix multiplies
    // per step and would deviate from the MATLAB DLM reference
    // implementation (dlmfit.m). We should only adopt it after the
    // port is fully validated against MATLAB outputs.
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
  type BackwardX = {
    x_pred: np.Array; C_pred: np.Array;
    K: np.Array; v: np.Array; Cp: np.Array;
  };
  type BackwardY = { x_smooth: np.Array; C_smooth: np.Array };
  
  const backwardStep = (
    carry: BackwardCarry,
    inp: BackwardX
  ): [BackwardCarry, BackwardY] => {
    const { r, N } = carry;
    const { x_pred: xi, C_pred: Ci, K: Ki, v: vi, Cp: Cpi } = inp;
    
    // L = G - K·F  [m,m]
    using L = np.subtract(G, np.matmul(Ki, F));
    
    // F'·Cp⁻¹  [m,1] (scalar division valid for p=1)
    using FtCpInv = np.divide(Ft, Cpi);
    
    // r_new = F'·Cp⁻¹·v + L'·r  [m,1]
    const r_new = np.add(
      np.multiply(FtCpInv, vi),
      np.matmul(np.transpose(L), r)
    );
    
    // N_new = F'·Cp⁻¹·F + L'·N·L  [m,m]
    //
    // NUMERICAL PRECISION NOTE:
    // The L'·N·L product via einsum uses two pairwise dot() calls.
    // Since jax-js-nonconsuming v0.2.1, Float64 uses Kahan compensated summation
    // in each dot, but errors still propagate into C_smooth via the
    // C·N·C product below. N accumulates information over the
    // backward pass, so rounding compounds across timesteps.
    const N_new = np.add(
      np.matmul(FtCpInv, F),
      np.einsum('ji,jk,kl->il', L, N, L)
    );
    
    // x_smooth = x_pred + C_pred·r_new  [m,1]
    const x_smooth = np.add(xi, np.matmul(Ci, r_new));
    
    // C_smooth = C_pred - C_pred·N_new·C_pred  [m,m]
    //
    // NUMERICAL PRECISION NOTE — MOST SENSITIVE OPERATION:
    // This subtraction is the single largest source of numerical
    // error in the DLM. When the smoothing correction C·N·C is
    // nearly equal to C_pred, we subtract two similar-magnitude
    // quantities to get a small result — classic catastrophic
    // cancellation. Measured worst case (jax-js-nonconsuming v0.2.1 with Kahan):
    // trig model (m=6), C[5][4] ≈ 2e-7 shows ~3.5% relative error
    // vs MATLAB (absolute error ~7e-9). Kahan compensated summation
    // in dot products (v0.2.1) improved the seasonal model (m=13)
    // but shifted the rounding pattern for trig (m=6), making this
    // specific element worse. This confirms that the bottleneck is
    // the subtraction itself, not the dot product accuracy.
    //
    // The einsum('ij,jk,kl->il', C, N, C) still decomposes into
    // two pairwise dot() calls; Kahan helps each individual dot
    // but cannot prevent the cancellation in the outer subtraction.
    //
    // POTENTIAL DLM-SIDE IMPROVEMENTS (not yet implemented):
    //
    // Option A — Joseph form for the backward step:
    //   C_smooth = (I - C·N)·C·(I - C·N)' + C·N·(tolerance term)
    //   This avoids the large subtraction by reformulating as a
    //   product of smaller corrections. It's used in some modern
    //   Kalman filter implementations for exactly this reason.
    //
    // Option B — Symmetrize after subtraction:
    //   C_smooth = 0.5 * (C_smooth + C_smooth')
    //   Cheap, doesn't fix the cancellation but ensures symmetry
    //   is preserved despite asymmetric rounding.
    //
    // Option C — Clamp negative diagonal entries:
    //   Covariance diagonals must be ≥ 0. In Float32 with m > 2,
    //   this subtraction can produce negative variances, causing
    //   NaN when sqrt is taken later. Clamping would be a band-aid.
    //
    // We intentionally match the MATLAB DLM reference (dlmsmo.m)
    // formulation for now. Deviating should only happen after the
    // port is fully validated and the improvement is justified by
    // a specific downstream need (e.g., Float32 support for m > 2,
    // or very large state dimensions).
    const C_smooth = np.subtract(
      Ci,
      np.einsum('ij,jk,kl->il', Ci, N_new, Ci)
    );
    
    return [{ r: r_new, N: N_new }, { x_smooth, C_smooth }];
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Jittable core: forward Kalman filter + backward RTS smoother +
  // diagnostics computed with vectorized numpy ops.
  // F, G, W, Ft are captured as constants by the JIT compiler.
  // Returns stacked tensors for arbitrary state dimension m.
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
    
    const x_smooth = np.flip(bwd.x_smooth, 0);  // [n, m, 1]
    const C_smooth = np.flip(bwd.C_smooth, 0);  // [n, m, m]
    tree.dispose(bwd);

    // ─── Observation-space diagnostics ───

    // yhat = F @ xf: F:[1,m] @ xf:[n,m,1] → [n,1,1] → [n]
    using yhat_3d = np.matmul(F, fwd.x_pred);
    const yhat = np.squeeze(yhat_3d);

    // ystd = sqrt(F @ C_smooth @ F' + V²)
    using FCFt_3d = np.einsum('ij,njk,lk->nil', F, C_smooth, F);
    using FCFt_flat = np.squeeze(FCFt_3d);
    const ystd = np.sqrt(np.add(FCFt_flat, V2_flat));

    // Innovations [n,1,1] → [n]
    const v_flat = np.squeeze(fwd.v);
    const Cp_flat = np.squeeze(fwd.Cp);

    // Dispose fwd.K (no longer needed)
    fwd.K.dispose();

    // Residuals
    const resid0 = np.subtract(y_1d, yhat);
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
    const mape = np.mean(np.divide(np.abs(resid2), y_1d));
    
    return {
      x: x_smooth, C: C_smooth,
      xf: fwd.x_pred, Cf: fwd.C_pred,
      yhat, ystd,
      v: v_flat, Cp: Cp_flat,
      resid0, resid, resid2,
      ssy, lik, s2, mse, mape,
    };
  };
  
  // Run core — one jit wrapping both scans + all diagnostics
  const coreResult = await jit(core)(x0, C0, y_arr, V2_arr, r0, N0);

  return tree.makeDisposable({
    ...coreResult, nobs: n, m: stateSize,
  }) as DlmSmoResult & Disposable;
};

/**
 * Fit a Dynamic Linear Model (DLM).
 *
 * Implements a two-pass estimation procedure:
 * 1. Initial pass with diffuse prior to estimate starting values
 * 2. Final pass with refined initial state from smoothed estimates
 *
 * Model components are determined by the options parameter:
 * - Polynomial trend (order 0/1/2)
 * - Full or trigonometric seasonal
 * - AR(p) components
 *
 * System matrices G and F are generated by dlmGenSys().
 * State noise covariance W = diag(w[0]², w[1]², ...) with zeros for
 * states beyond w.length.
 *
 * @param y - Observations (n×1 array)
 * @param s - Observation noise standard deviation
 * @param w - State noise standard deviations (diagonal of sqrt(W))
 * @param dtype - Computation precision (default: Float64)
 * @param options - Model specification (default: order=1, no seasonal)
 * @returns Complete model fit with smoothed estimates and diagnostics
 */
export const dlmFit = async (
  y: ArrayLike<number>,
  s: number,
  w: number[],
  dtype: DType = DType.Float64,
  options: DlmOptions = {},
): Promise<DlmFitResult> => {
  const n = y.length;
  const FA = getFloatArrayType(dtype);

  // Convert input to TypedArray if needed
  const yArr = y instanceof FA ? y : FA.from(y);
  // Observation noise std dev (constant for all timesteps)
  const V_std = new FA(n).fill(s);

  // ─────────────────────────────────────────────────────────────────────────
  // Generate system matrices from options
  // ─────────────────────────────────────────────────────────────────────────
  const sys = dlmGenSys(options);
  const m = sys.m;

  // State transition: x(t) = G·x(t-1) + w
  using G = np.array(sys.G, { dtype });
  // Observation: y(t) = F·x(t) + v  (F is [1, m])
  using F = np.array([sys.F], { dtype });

  // State noise covariance: W = diag(w[0]², w[1]², ..., 0, ...)
  const W_data: number[][] = Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => {
      if (i === j && i < w.length) return w[i] ** 2;
      return 0;
    })
  );

  // Spline mode: modifies W for order=1
  if (options.spline && (options.order ?? 1) === 1 && w.length >= 2) {
    W_data[0][0] = w[1] ** 2 * (1 / 3);
    W_data[0][1] = w[1] ** 2 * (1 / 2);
    W_data[1][0] = w[1] ** 2 * (1 / 2);
    W_data[1][1] = w[1] ** 2 * 1;
  }

  using W = np.array(W_data, { dtype });

  // ─────────────────────────────────────────────────────────────────────────
  // Initialize state with diffuse prior
  // x0[0] = mean of first ns observations (level); rest = 0
  // ─────────────────────────────────────────────────────────────────────────
  const ns = options.ns ?? 12;
  let initSum = 0;
  const count = Math.min(ns, n);
  for (let i = 0; i < count; i++) initSum += Number(yArr[i]);
  const mean_y = initSum / count;
  // Initial covariance: diagonal with large uncertainty (diffuse prior)
  const c0_val = (Math.abs(mean_y) * 0.5) ** 2;
  const c0 = c0_val === 0 ? 1e7 : c0_val;
  const x0_data: number[][] = Array.from({ length: m }, (_, i) =>
    [i === 0 ? mean_y : 0.0]
  );
  const C0_data: number[][] = Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => (i === j ? c0 : 0.0))
  );

  // ─────────────────────────────────────────────────────────────────────────
  // Pass 1: Initial smoother to refine starting values
  // ─────────────────────────────────────────────────────────────────────────
  let x0_updated: number[][];
  let C0_scaled: number[][];
  { // Block scope — `using` auto-disposes all Pass 1 arrays at block end
    using out1 = await dlmSmo(yArr, F, V_std, x0_data, G, W, C0_data, m, dtype);
    // out1.x is [n, m, 1] — extract first timestep
    const x_data = await out1.x.data() as Float64Array | Float32Array;
    const C_data = await out1.C.data() as Float64Array | Float32Array;
    x0_updated = Array.from({ length: m }, (_, i) => [x_data[i]]);
    // C is stored as [n, m, m] → first m×m block
    C0_scaled = Array.from({ length: m }, (_, i) =>
      Array.from({ length: m }, (_, j) => C_data[i * m + j] * 100)
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Pass 2: Final smoother with refined initial state
  // ─────────────────────────────────────────────────────────────────────────
  const out2 = await dlmSmo(yArr, F, V_std, x0_updated, G, W, C0_scaled, m, dtype);

  // ─────────────────────────────────────────────────────────────────────────
  // Convert np.Array results to TypedArrays via consumeData (read + dispose)
  // ─────────────────────────────────────────────────────────────────────────
  const toFA = async (a: np.Array) =>
    new FA(await a.consumeData() as ArrayLike<number>);
  const toNum = async (a: np.Array) =>
    (await a.consumeData() as ArrayLike<number>)[0];

  // Read stacked tensors and extract per-component arrays
  const xf_raw = await out2.xf.consumeData() as ArrayLike<number>; // [n,m,1]
  const Cf_raw = await out2.Cf.consumeData() as ArrayLike<number>; // [n,m,m]
  const x_raw = await out2.x.consumeData() as ArrayLike<number>;   // [n,m,1]
  const C_raw = await out2.C.consumeData() as ArrayLike<number>;   // [n,m,m]

  // xf[k][t] = xf_raw[t * m + k]  (m×1 per timestep, flattened)
  const xf: FloatArray[] = Array.from({ length: m }, (_, k) => {
    const arr = new FA(n);
    for (let t = 0; t < n; t++) arr[t] = xf_raw[t * m + k] as number;
    return arr;
  });

  // Cf[i][j][t] = Cf_raw[t * m * m + i * m + j]
  const Cf: FloatArray[][] = Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => {
      const arr = new FA(n);
      for (let t = 0; t < n; t++) arr[t] = Cf_raw[t * m * m + i * m + j] as number;
      return arr;
    })
  );

  // Smoothed state: x[k][t]
  const x_out: FloatArray[] = Array.from({ length: m }, (_, k) => {
    const arr = new FA(n);
    for (let t = 0; t < n; t++) arr[t] = x_raw[t * m + k] as number;
    return arr;
  });

  // Smoothed covariance: C_out[i][j][t]
  const C_out: FloatArray[][] = Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => {
      const arr = new FA(n);
      for (let t = 0; t < n; t++) arr[t] = C_raw[t * m * m + i * m + j] as number;
      return arr;
    })
  );

  // xstd[t][k] = sqrt(|C[k][k][t]|) — smoothed state std devs
  const xstd: FloatArray[] = Array.from({ length: n }, (_, t) => {
    const arr = new FA(m);
    for (let k = 0; k < m; k++) {
      arr[k] = Math.sqrt(Math.abs(C_raw[t * m * m + k * m + k] as number));
    }
    return arr;
  });

  // Diagnostics
  const yhat = await toFA(out2.yhat);
  const ystd = await toFA(out2.ystd);
  const v = await toFA(out2.v);
  const Cp_arr = await toFA(out2.Cp);
  const resid0 = await toFA(out2.resid0);
  const resid = await toFA(out2.resid);
  const resid2 = await toFA(out2.resid2);

  // Scalar diagnostics
  const ssy = await toNum(out2.ssy);
  const lik = await toNum(out2.lik);
  const s2 = await toNum(out2.s2);
  const mse = await toNum(out2.mse);
  const mape = await toNum(out2.mape);

  return {
    // State estimates
    xf, Cf, x: x_out, C: C_out, xstd,
    // System matrices (plain arrays for easy serialization)
    G: sys.G,
    F: sys.F,
    W: W_data,
    // Input data
    y: yArr, V: V_std,
    // Initial state (after Pass 1 refinement)
    x0: x0_updated.map(row => row[0]),
    C0: C0_scaled,
    // Covariates (empty for basic model)
    XX: [],
    // Predictions and residuals
    yhat, ystd, resid0, resid, resid2,
    // Diagnostics
    ssy, v, Cp: Cp_arr, s2,
    nobs: out2.nobs,
    lik, mse, mape,
    class: 'dlmfit',
  };
};
