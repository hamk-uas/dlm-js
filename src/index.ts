import { DType, numpy as np, lax, jit, tree, defaultDevice } from "@hamk-uas/jax-js-nonconsuming";
import type { DlmSmoResult, DlmFitResult, DlmForecastResult, FloatArray } from "./types";
import { getFloatArrayType } from "./types";
import { dlmGenSys } from "./dlmgensys";
import type { DlmOptions } from "./dlmgensys";

// Public type exports
export type { DlmFitResult, DlmForecastResult, FloatArray } from "./types";
export type { DlmOptions, DlmSystem } from "./dlmgensys";
export { dlmGenSys, findArInds } from "./dlmgensys";
export { dlmMLE } from "./mle";
export type { DlmMleResult } from "./mle";

/**
 * Solve the Discrete Algebraic Riccati Equation (DARE) to find the
 * steady-state Kalman gain K_ss and associated matrices.
 *
 * Iterates the standard Kalman predict-update cycle starting from C0
 * until the predicted covariance converges. Uses the Joseph form for
 * the filtered covariance update to maintain positive-definiteness
 * in Float32.
 *
 * Result:
 *   K_ss  [m, 1]  — steady-state Kalman gain (standard formulation)
 *   A_ss  [m, m]  — (I - K_ss·F)·G  — state transition after update
 *   Sigma_base [m, m] — (I - K_ss·F)·W·(I - K_ss·F)' — base covariance
 *   KKt   [m, m]  — K_ss · K_ss' — for per-timestep V²-scaled term
 *
 * The returned K_ss uses the standard formulation:
 *   x̄_t = G·x_{t-1}  (predict)
 *   K_t  = C̄_t·F' / (F·C̄_t·F' + V²)  (gain from predicted cov)
 *   x_t  = x̄_t + K_t·(y_t - F·x̄_t)   = (I - K_t·F)·G·x_{t-1} + K_t·y_t
 *
 * This is NOT the same as the MATLAB DLM convention used in the sequential
 * filter (K_m = G·C·F'/Cp). The two produce algebraically different gains
 * but converge to the same steady state. The standard formulation is used
 * here because it gives a symmetric compose rule for associativeScan:
 *   (A₂, b₂, Σ₂) ⊕ (A₁, b₁, Σ₁) = (A₂·A₁, A₂·b₁ + b₂, A₂·Σ₁·A₂' + Σ₂)
 *
 * @internal
 */
const solveDAREForKss = async (
  G_data: number[][],
  F_data: number[],
  W_data: number[][],
  C0_data: number[][],
  V2_scalar: number,
  m: number,
  dtype: DType,
  maxIter: number = 50,
): Promise<{
  K_ss: np.Array;        // [m, 1]
  A_ss: np.Array;        // [m, m]
  Sigma_base: np.Array;  // [m, m]
  KKt: np.Array;         // [m, m]
}> => {
  // Work eagerly (no jit) — this is a cheap O(maxIter·m³) pre-computation
  // Uses MATLAB DLM convention: K = G·C·F'/(F·C·F'+V²) directly from filtered cov C,
  // so that the assocScan forward pass (A_ss = G−K_ss·F) is consistent with the
  // sequential MATLAB filter and the backward smoother K recovery.
  using G = np.array(G_data, { dtype });
  using F_mat = np.array([F_data], { dtype });        // [1, m]
  using W_mat = np.array(W_data, { dtype });
  using V2 = np.array([[V2_scalar]], { dtype });       // [1, 1]

  let C = np.array(C0_data, { dtype });                // [m, m] filtered cov

  for (let i = 0; i < maxIter; i++) {
    // MATLAB convention: Cp = F·C·F' + V²  (using filtered C, not predicted)
    using Cp = np.add(np.einsum('ij,jk,lk->il', F_mat, C, F_mat), V2);

    // MATLAB gain: K = G·C·F' / Cp  [m,1]
    using GCFt = np.einsum('ij,jk,lk->il', G, C, F_mat);
    using K = np.divide(GCFt, Cp);

    // L = G − K·F  (MATLAB convention smoother matrix)
    using KF = np.matmul(K, F_mat);                    // [m, m]
    using L = np.subtract(G, KF);                      // [m, m]

    // Joseph form update: C_new = L·C·L' + K·V²·K' + W
    using LCLt = np.einsum('ij,jk,lk->il', L, C, L);
    using KV2Kt = np.multiply(V2, np.matmul(K, np.transpose(K)));
    using sum1 = np.add(LCLt, KV2Kt);
    // jax-js-lint: allow-non-using
    const C_new = np.add(sum1, W_mat);
    C.dispose();
    C = C_new;
  }

  // Compute converged K_ss and A_ss using MATLAB convention
  using Cp_ss = np.add(np.einsum('ij,jk,lk->il', F_mat, C, F_mat), V2);
  using GCFt_ss = np.einsum('ij,jk,lk->il', G, C, F_mat);
  const K_ss = np.divide(GCFt_ss, Cp_ss);              // [m, 1]

  // A_ss = G − K_ss·F  (MATLAB convention: the smoother matrix L_ss)
  using KF_ss = np.matmul(K_ss, F_mat);
  const A_ss = np.subtract(G, KF_ss);                  // [m, m]

  // Sigma_base = W  (MATLAB Joseph form: C_new = A_ss·C·A_ss' + K·V²·K' + W,
  // so the V²-independent part contributed each step is just W)
  const Sigma_base = np.add(W_mat, np.zerosLike(W_mat)); // copy of W [m, m]

  // KKt = K_ss · K_ss'  (for V²-scaled measurement noise part)
  const KKt = np.matmul(K_ss, np.transpose(K_ss));     // [m, m]

  C.dispose();
  return { K_ss, A_ss, Sigma_base, KKt };
};

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
 * When covariates are provided (FF_arr), F is time-varying:
 *
 *   y(t) = FF_t·x_ext(t) + v,  where FF_t = [F_base, X[t,:]]
 *
 * The extended state x_ext includes the covariate regression coefficients β,
 * which evolve as static states (identity block in G, zero block in W).
 *
 * The forward Kalman filter computes one-step-ahead predictions.
 * The backward RTS smoother refines estimates using all observations.
 *
 * Reference: Durbin & Koopman (2012), "Time Series Analysis by State Space Methods"
 *
 * @param y - Observations (n×1)
 * @param F - Observation matrix (1×m), maps state to observation. When FF_arr
 *            is provided, this is the base F (1×m_base) and the effective F at
 *            each timestep is read from FF_arr instead.
 * @param V_std - Observation noise std devs (n×1)
 * @param x0_data - Initial state mean (m×1 as nested array)
 * @param G - State transition matrix (m×m)
 * @param W - State noise covariance (m×m)
 * @param C0_data - Initial state covariance (m×m as nested array)
 * @param stateSize - State dimension m (extended: m_base + q for covariates)
 * @param dtype - Computation precision
 * @param FF_arr - Optional time-varying observation matrix [n, 1, m] (for covariates)
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
  FF_arr?: np.Array,  // [n, 1, m_ext] time-varying F (covariates)
  /** Plain-JS system matrix data — only needed for associativeScan DARE.
   *  Passed from dlmFit which already has them. */
  sysData?: { G_data: number[][]; F_data: number[]; W_data: number[][] },
): Promise<DlmSmoResult & Disposable> => {
  const n = y.length;

  // ─────────────────────────────────────────────────────────────────────────
  // Branch selection: three execution paths based on device + dtype
  //
  //   wasm/cpu + Float64  →  current sequential scan (no extra stabilization)
  //   cpu      + Float32  →  sequential scan + Joseph form + symmetrize + clamp
  //   webgpu   + Float32  →  associativeScan forward + Joseph form/symmetrize/clamp
  //
  // The Float32 stabilization (Joseph form covariance update, symmetrization,
  // diagonal clamping) adds ~m² extra ops per step but prevents the covariance
  // from going non-positive-definite for m > 2, which is the main float32
  // failure mode. Float64 is unaffected — it takes the fast path without any
  // extra work.
  //
  // The associativeScan path (webgpu) reformulates the forward Kalman filter
  // as an associative prefix scan per Särkkä & García-Fernández (2020),
  // reducing sequential depth from O(n) to O(log n) on parallel hardware.
  // ─────────────────────────────────────────────────────────────────────────
  const device = defaultDevice();
  const f32 = dtype === DType.Float32;
  const useAssocScan = f32 && device === 'webgpu';

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

  // ─────────────────────────────────────────────────────────────────────────
  // Build FF_scan: [n, 1, m] pytree input for the scan.
  // Without covariates: broadcast static F to shape [n, 1, m].
  // With covariates: caller provides FF_arr [n, 1, m_ext] directly.
  // Both cases use the same step functions — no branching inside scan.
  // ─────────────────────────────────────────────────────────────────────────
  const FF_scan: np.Array = FF_arr !== undefined
    ? FF_arr
    : np.tile(np.reshape(F, [1, 1, stateSize]), [n, 1, 1]);

  // ─────────────────────────────────────────────────────────────────────────
  // Pre-compute steady-state Kalman gain for associativeScan path (webgpu).
  // Done eagerly before core (outside JIT) — cheap O(50·m³).
  // ─────────────────────────────────────────────────────────────────────────
  let dare: { K_ss: np.Array; A_ss: np.Array; Sigma_base: np.Array; KKt: np.Array } | null = null;
  if (useAssocScan && sysData) {
    // Use mean V² across timesteps for the DARE
    let v2sum = 0;
    for (let i = 0; i < V_std.length; i++) v2sum += V_std[i] * V_std[i];
    const v2mean = v2sum / V_std.length;
    dare = await solveDAREForKss(
      sysData.G_data, sysData.F_data, sysData.W_data,
      C0_data, v2mean, stateSize, dtype,
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Step functions receive FF_t ([1, m]) from the scan pytree.
  // G and W are still captured as constants (not time-varying).
  // ─────────────────────────────────────────────────────────────────────────

  // Constant [1,1] ones tensor captured by forwardStep closure for NaN masking.
  using const_one_11 = np.array([[1.0]], { dtype });

  type ForwardCarry = { x: np.Array; C: np.Array };
  type ForwardX = { y: np.Array; V2: np.Array; FF: np.Array };
  type ForwardY = {
    x_pred: np.Array; C_pred: np.Array;
    K: np.Array; v: np.Array; Cp: np.Array; FF: np.Array;
    mask: np.Array;  // [1,1]: 1.0 if observed, 0.0 if NaN
  };
  
  const forwardStep = (
    carry: ForwardCarry,
    inp: ForwardX
  ): [ForwardCarry, ForwardY] => {
    const { x: xi, C: Ci } = carry;
    const { y: yi, V2: V2i, FF: FFi } = inp;

    // NaN masking: mask = 1.0 if y is observed, 0.0 if y is NaN.
    // When y is NaN, the measurement update is skipped entirely:
    //   x_{t|t} = x_{t|t-1}  (no correction)
    //   C_{t|t} = C_{t|t-1}  (no reduction)
    // This is achieved by zeroing K and v, which makes L = G and
    // the update terms vanish without branching inside the scan.
    using is_nan = np.isnan(yi);                                   // [1,1] bool
    using zero_11 = np.zerosLike(yi);                              // [1,1]
    const mask_t = np.where(is_nan, zero_11, const_one_11);       // [1,1]: 0 if NaN
    using y_safe = np.where(is_nan, zero_11, yi);                  // [1,1]: 0 if NaN

    // Innovation: v = (y_safe - FF·x) * mask  [1,1]  (0 when NaN)
    using FFxi = np.matmul(FFi, xi);
    using v_raw = np.subtract(y_safe, FFxi);
    const v = np.multiply(mask_t, v_raw);

    // Innovation covariance: Cp = FF·C·FF' + V²  [1,1]
    const Cp = np.add(
      np.einsum('ij,jk,lk->il', FFi, Ci, FFi),
      V2i
    );

    // Kalman gain: K = mask * (G·C·FF' / Cp)  [m,1]  (0 when NaN)
    using GCFFt = np.einsum('ij,jk,lk->il', G, Ci, FFi);
    using K_raw = np.divide(GCFFt, Cp);
    const K = np.multiply(mask_t, K_raw);  // [1,1]×[m,1] → [m,1] by broadcast

    // L = G - K·FF  [m,m]  (= G when NaN, since K=0)
    using L = np.subtract(G, np.matmul(K, FFi));

    // Next state prediction: x_next = G·x + K·v  [m,1]
    // When NaN: x_next = G·x (no measurement update)
    const x_next = np.add(
      np.matmul(G, xi),
      np.matmul(K, v)
    );

    // Next covariance: C_next depends on dtype branch.
    //
    // Float64 path (fast, matches MATLAB DLM reference):
    //   C_next = G·C·L' + W
    //
    // Float32 path (Joseph form — numerically stable):
    //   C_next = L·C·L' + K·V²·K' + W
    //   followed by symmetrization: C_next = 0.5·(C_next + C_next')
    //
    // The Joseph form guarantees positive semi-definiteness by construction
    // and avoids the implicit subtraction in G·C·L' that causes catastrophic
    // cancellation for m > 2 in Float32. The extra cost (~2 matmuls + transpose)
    // is negligible compared to the stability gain. Float64 skips this because
    // Kahan-compensated dot products (v0.2.1) keep errors manageable and we
    // want exact MATLAB DLM reference matching.
    let C_next: np.Array;
    if (f32) {
      // Joseph form: L·C·L' + K·V²·K' + W
      using LCLt = np.einsum('ij,jk,lk->il', L, Ci, L);
      using KV2Kt = np.multiply(V2i, np.matmul(K, np.transpose(K)));  // [1,1]·([m,1]·[1,m]) = [m,m]
      using sum1 = np.add(LCLt, KV2Kt);
      using sum2 = np.add(sum1, W);
      // Symmetrize: C = 0.5·(C + C')
      using sum2t = np.transpose(sum2);
      using sumBoth = np.add(sum2, sum2t);
      C_next = np.multiply(np.array(0.5, { dtype }), sumBoth);
    } else {
      // Standard form (matches MATLAB dlmsmo.m): G·C·L' + W
      C_next = np.add(
        np.einsum('ij,jk,lk->il', G, Ci, L),
        W
      );
    }
    
    return [
      { x: x_next, C: C_next },
      // Pass FFi and mask through so backward pass can reuse them
      { x_pred: xi, C_pred: Ci, K, v, Cp, FF: FFi, mask: mask_t } as ForwardY,
    ];
  };
  
  type BackwardCarry = { r: np.Array; N: np.Array };
  type BackwardX = {
    x_pred: np.Array; C_pred: np.Array;
    K: np.Array; v: np.Array; Cp: np.Array; FF: np.Array;
    mask: np.Array;  // [1,1]: 1.0 if observed, 0.0 if NaN (mirrors forwardStep)
  };
  type BackwardY = { x_smooth: np.Array; C_smooth: np.Array };
  
  const backwardStep = (
    carry: BackwardCarry,
    inp: BackwardX
  ): [BackwardCarry, BackwardY] => {
    const { r, N } = carry;
    const { x_pred: xi, C_pred: Ci, K: Ki, v: vi, Cp: Cpi, FF: FFi, mask: maski } = inp;

    // L = G - K·FF  [m,m]  (K=0 when NaN → L=G, propagating prior)
    using L = np.subtract(G, np.matmul(Ki, FFi));

    // FF'·Cp⁻¹  [m,1] (scalar division valid for p=1)
    using FFt = np.transpose(FFi);
    using FtCpInv = np.divide(FFt, Cpi);

    // r_new = F'·Cp⁻¹·v + L'·r  [m,1]
    // vi is already 0 at NaN positions (zeroed in forwardStep), so
    // FtCpInv·vi contributes 0 automatically at missing timesteps.
    const r_new = np.add(
      np.multiply(FtCpInv, vi),
      np.matmul(np.transpose(L), r)
    );

    // N_new = mask·(FF'·Cp⁻¹·FF) + L'·N·L  [m,m]
    // The outer-product term must be masked: at NaN timesteps it would
    // otherwise add spurious Fisher information to N, causing the smoother
    // to over-shrink state uncertainty at and around missing observations.
    //
    // NUMERICAL PRECISION NOTE:
    // The L'·N·L product via einsum uses two pairwise dot() calls.
    // Since jax-js-nonconsuming v0.2.1, Float64 uses Kahan compensated summation
    // in each dot, but errors still propagate into C_smooth via the
    // C·N·C product below. N accumulates information over the
    // backward pass, so rounding compounds across timesteps.
    using FtCpInvFF = np.matmul(FtCpInv, FFi);  // [m,m]
    const N_new = np.add(
      np.multiply(maski, FtCpInvFF),  // [1,1]×[m,m] → [m,m]: 0 when NaN
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
    // cancellation. Float64 keeps errors manageable via Kahan
    // compensated summation (v0.2.1); Float32 can produce negative
    // variances for m > 2.
    //
    // Float32 stabilization (applied when f32=true):
    //   1. Symmetrize: C_smooth = 0.5·(C_smooth + C_smooth')
    //   2. Clamp diagonal: max(diag, 0) to prevent NaN from sqrt
    // This doesn't fix the cancellation but prevents downstream NaN.
    // A future parallel smoother (Särkkä §4) will replace this
    // entire backward step.
    let C_smooth: np.Array;
    {
      using C_raw = np.subtract(
        Ci,
        np.einsum('ij,jk,kl->il', Ci, N_new, Ci)
      );
      if (f32) {
        // Symmetrize: C = 0.5·(C + C') — fixes tiny asymmetries from f32 rounding.
        // NOTE: We do NOT clamp entries to ≥ 0 here. Off-diagonal entries of a
        // covariance matrix CAN be negative (negative correlations). Clamping all
        // entries would destroy the covariance structure for m > 1.
        using Ct = np.transpose(C_raw);
        using sumC = np.add(C_raw, Ct);
        C_smooth = np.multiply(np.array(0.5, { dtype }), sumC);
      } else {
        // Float64: use raw result (matches MATLAB dlmsmo.m reference)
        C_smooth = np.add(C_raw, np.zeros([stateSize, stateSize], { dtype })); // copy to own
      }
    }
    
    return [{ r: r_new, N: N_new }, { x_smooth, C_smooth }];
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Jittable core: forward Kalman filter + backward RTS smoother +
  // diagnostics computed with vectorized numpy ops.
  // G and W are captured as constants by the JIT compiler.
  // FF_scan [n,1,m] is threaded through scan for time-varying F support.
  // Returns stacked tensors for arbitrary state dimension m.
  // ─────────────────────────────────────────────────────────────────────────
  
  const core = (
    x0: np.Array, C0: np.Array,
    y_arr: np.Array, V2_arr: np.Array,
    FF_scan: np.Array,
    r0: np.Array, N0: np.Array
  ) => {
    // Derive flat [n] inputs for diagnostics
    using y_1d = np.squeeze(y_arr);      // [n] from [n,1,1]
    using V2_flat = np.squeeze(V2_arr);  // [n]
    using V_flat = np.sqrt(V2_flat);     // [n] observation noise std dev

    // ─── Forward Kalman Filter ───
    // Two paths: sequential lax.scan (cpu/wasm) or parallel associativeScan (webgpu).
    // Both produce the same `fwd` structure consumed by the backward smoother:
    //   fwd.x_pred [n,m,1]  — carry entering step t (= x_{t-1|t-1})
    //   fwd.C_pred [n,m,m]  — cov entering step t
    //   fwd.K      [n,m,1]  — Kalman gain (MATLAB convention)
    //   fwd.v      [n,1,1]  — innovation (masked to 0 at NaN)
    //   fwd.Cp     [n,1,1]  — innovation variance
    //   fwd.FF     [n,1,m]  — observation matrix per step
    //   fwd.mask   [n,1,1]  — 1.0 observed, 0.0 NaN

    let fwd: ForwardY;
    let x_smooth: np.Array;
    let C_smooth: np.Array;

    if (useAssocScan && dare) {
      // ─── Associative Scan Forward Filter (webgpu + f32) ───
      //
      // Uses steady-state Kalman gain from DARE to construct per-timestep
      // elements (A, b, Σ) and composes them via lax.associativeScan.
      //
      // Standard Kalman formulation (NOT the MATLAB convention):
      //   x̄_t = G·x_{t-1|t-1}           (predict)
      //   K_t  = C̄_t·F'/(F·C̄_t·F'+V²)   (gain)
      //   x_{t|t} = (I-K_t·F)·G · x_{t-1|t-1} + K_t·y_t
      //   C_{t|t} = A·C_{t-1|t-1}·A' + Σ  where A=(I-K·F)·G
      //
      // The steady-state approximation uses K_ss for all steps (converges
      // within ~5 steps for typical DLMs), giving constant A_ss and Σ_base.
      // NaN steps use A=G, b=0, Σ=W (prediction only, no update).

      const { K_ss, A_ss, Sigma_base, KKt } = dare!;

      // Build NaN mask [n, 1, 1]: 1.0 observed, 0.0 NaN
      using is_nan = np.isnan(y_arr);                    // [n,1,1] bool
      using zero_n11 = np.zerosLike(y_arr);              // [n,1,1]
      using one_n11 = np.onesLike(y_arr);                // [n,1,1]
      // jax-js-lint: allow-non-using — stored in fwd.mask, disposed after backward pass
      const mask_arr = np.where(is_nan, zero_n11, one_n11); // [n,1,1]

      // NaN-safe y: replace NaN with 0
      using y_safe_arr = np.where(is_nan, zero_n11, y_arr); // [n,1,1]

      // Per-timestep b = mask · K_ss · y_safe  [n, m, 1]
      // K_ss is [m,1], y_safe is [n,1,1] → via einsum: [n,m,1]
      // b[t] = mask[t] · K_ss · y_safe[t]  (0 when NaN)
      using y_safe_flat = np.squeeze(y_safe_arr, [2]);    // [n, 1]
      using Ky_all = np.einsum('ij,nj->ni', K_ss, y_safe_flat);  // [m,1]@... → [n, m]
      using Ky_expand = np.expandDims(Ky_all, 2);       // [n, m, 1]
      using b_arr = np.multiply(mask_arr, Ky_expand);    // [n, m, 1]: 0 at NaN

      // Per-timestep A [n, m, m]: A_ss where observed, G where NaN
      // mask_mm [n,1,1] broadcasts to [n,m,m]
      using A_ss_expanded = np.tile(np.reshape(A_ss, [1, stateSize, stateSize]), [n, 1, 1]);
      using G_expanded = np.tile(np.reshape(G, [1, stateSize, stateSize]), [n, 1, 1]);
      // is_nan [n,1,1] bool — NaN→G, observed→A_ss (branches swapped vs float-mask form)
      using A_arr = np.where(
        np.tile(is_nan, [1, stateSize, stateSize]),
        G_expanded,
        A_ss_expanded,
      );                                                  // [n, m, m]

      // Per-timestep Σ [n, m, m]:
      //   observed: Sigma_base + V²[t]·KKt
      //   NaN:      W
      using V2_mm = np.reshape(V2_arr, [n, 1, 1]);       // [n,1,1]
      using KKt_expanded = np.tile(np.reshape(KKt, [1, stateSize, stateSize]), [n, 1, 1]);
      using Sigma_V2 = np.multiply(V2_mm, KKt_expanded); // [n,m,m]
      using Sigma_base_exp = np.tile(np.reshape(Sigma_base, [1, stateSize, stateSize]), [n, 1, 1]);
      using Sigma_obs = np.add(Sigma_base_exp, Sigma_V2); // [n,m,m]
      using W_expanded = np.tile(np.reshape(W, [1, stateSize, stateSize]), [n, 1, 1]);
      // is_nan [n,1,1] bool — NaN→W, observed→Sigma_obs (branches swapped vs float-mask form)
      using Sigma_arr = np.where(np.tile(is_nan, [1, stateSize, stateSize]), W_expanded, Sigma_obs); // [n, m, m]

      // ─── Compose function for associativeScan ───
      // Pytree: { A: [k,m,m], b: [k,m,1], S: [k,m,m] }
      // compose(earlier, later) chains the affine maps:
      //   A_comp = A_later · A_earlier
      //   b_comp = A_later · b_earlier + b_later
      //   S_comp = A_later · S_earlier · A_later' + S_later
      type ScanElem = { A: np.Array; b: np.Array; S: np.Array };

      const compose = (a: ScanElem, b_elem: ScanElem): ScanElem => {
        // A_comp = B.A @ A.A  — batched [k,m,m]
        const A_comp = np.einsum('nij,njk->nik', b_elem.A, a.A);
        // b_comp = B.A @ A.b + B.b  — [k,m,1]
        using Ab = np.einsum('nij,njk->nik', b_elem.A, a.b);
        const b_comp = np.add(Ab, b_elem.b);
        // S_comp = B.A @ A.S @ B.A' + B.S  — [k,m,m]
        using ASAt = np.einsum('nij,njk,nlk->nil', b_elem.A, a.S, b_elem.A);
        const S_comp = np.add(ASAt, b_elem.S);
        return { A: A_comp, b: b_comp, S: S_comp };
      };

      // Run associative scan — O(log n) depth
      const scanned = lax.associativeScan(
        compose,
        { A: A_arr, b: b_arr, S: Sigma_arr },
      ) as ScanElem;

      // scanned.A[t], scanned.b[t], scanned.S[t] are the composed transformations
      // from step 0..t inclusive. To get filtered state:
      //   x_filt[t] = scanned.A[t] · x0 + scanned.b[t]
      //   C_filt[t] = scanned.A[t] · C0 · scanned.A[t]' + scanned.S[t]

      // x_filt = A_comp @ x0 + b_comp  [n, m, 1]
      using x0_exp = np.tile(np.reshape(x0, [1, stateSize, 1]), [n, 1, 1]);
      using Ax0 = np.einsum('nij,njk->nik', scanned.A, x0_exp);
      const x_filt = np.add(Ax0, scanned.b);             // [n, m, 1]

      // C_filt = A_comp @ C0 @ A_comp' + S_comp  [n, m, m]
      using C0_exp = np.tile(np.reshape(C0, [1, stateSize, stateSize]), [n, 1, 1]);
      using AC0At = np.einsum('nij,njk,nlk->nil', scanned.A, C0_exp, scanned.A);
      using C_filt_raw = np.add(AC0At, scanned.S);

      // Symmetrize C_filt (f32 stabilization — always true on this path)
      using C_filt_t = np.einsum('nij->nji', C_filt_raw);
      using C_filt_sum = np.add(C_filt_raw, C_filt_t);
      const C_filt = np.multiply(np.array(0.5, { dtype }), C_filt_sum); // [n,m,m]

      tree.dispose(scanned);

      // ─── Recover sequential-convention diagnostics from filtered results ───
      // x_pred[t] = x_{t-1|t-1} → prepend x0, drop last
      // C_pred[t] = C_{t-1|t-1} → prepend C0, drop last
      const xFiltParts = np.split(x_filt, [n - 1], 0);
      xFiltParts[1].dispose();
      using x_filt_head = xFiltParts[0];  // [n-1, m, 1]
      using x0_1 = np.reshape(x0, [1, stateSize, 1]);
      // jax-js-lint: allow-non-using — stored in fwd.x_pred, disposed by caller
      const x_pred_arr = np.concatenate([x0_1, x_filt_head], 0);  // [n, m, 1]

      const cFiltParts = np.split(C_filt, [n - 1], 0);
      cFiltParts[1].dispose();
      using C_filt_head = cFiltParts[0];  // [n-1, m, m]
      using C0_1 = np.reshape(C0, [1, stateSize, stateSize]);
      // jax-js-lint: allow-non-using — stored in fwd.C_pred, disposed by caller
      const C_pred_arr = np.concatenate([C0_1, C_filt_head], 0);  // [n, m, m]

      // v[t] = mask · (y - F·x_pred)  [n,1,1]
      using Fx_pred = np.einsum('nij,njk->nik', FF_scan, x_pred_arr); // [n,1,1]
      using v_raw = np.subtract(y_safe_arr, Fx_pred);
      // jax-js-lint: allow-non-using — stored in fwd.v, disposed by caller
      const v_arr = np.multiply(mask_arr, v_raw);         // [n,1,1]

      // Cp[t] = F·C_pred·F' + V²  [n,1,1]
      using FCFt = np.einsum('nij,njk,nlk->nil', FF_scan, C_pred_arr, FF_scan);
      // jax-js-lint: allow-non-using — stored in fwd.Cp, disposed by caller
      const Cp_arr = np.add(FCFt, V2_arr);                // [n,1,1]

      // K[t] = mask · G·C_pred·F' / Cp  [n,m,1]  (MATLAB convention for backward pass)
      using GCFt = np.einsum('ij,njk,nlk->nil', G, C_pred_arr, FF_scan); // [n,m,1]
      using K_raw = np.divide(GCFt, Cp_arr);
      // jax-js-lint: allow-non-using — stored in fwd.K, disposed by caller
      const K_arr = np.multiply(mask_arr, K_raw);         // [n,m,1]

      fwd = {
        x_pred: x_pred_arr, C_pred: C_pred_arr,
        K: K_arr, v: v_arr, Cp: Cp_arr,
        FF: FF_scan, mask: mask_arr,
      } as unknown as ForwardY;

      // ─── Parallel Backward Smoother (Särkkä & García-Fernández 2020) ─────
      //
      // Reformulates the RTS backward pass as an associative suffix scan,
      // reducing sequential depth from O(n) to O(log n) dispatches on WebGPU.
      //
      // Each smoother element a_k = (E_k, g_k, L_k) satisfies:
      //   a_k(x_k | x_{k+1}) = N(x_k; E_k·x_{k+1} + g_k, L_k)
      //
      // Composition (Lemma 6): identical structure to forward compose:
      //   (E_ij, g_ij, L_ij) = (E_i·E_j, E_i·g_j + g_i, E_i·L_j·E_i' + L_i)
      //
      // Terminal: E_{n-1}=0, g_{n-1}=x̄_{n-1}, L_{n-1}=C_{filt,n-1}.
      // After composition with terminal, all E values → 0.
      // Smoothed density: x_smooth = g_comp, C_smooth = L_comp.
      // ────────────────────────────────────────────────────────────────────────
      {
        // S_k = G · C_filt,k · G' + W  [n, m, m]
        using GCGt = np.einsum('ij,njk,lk->nil', G, C_filt, G);
        using W_bcast = np.tile(np.reshape(W, [1, stateSize, stateSize]), [n, 1, 1]);
        using S_mat = np.add(GCGt, W_bcast);

        // Batched matrix inverse S^{-1}  [n, m, m]
        using S_inv = np.linalg.inv(S_mat);

        // E_k = C_filt,k · G' · S_k^{-1}  [n, m, m]
        using CGt = np.einsum('nij,kj->nik', C_filt, G);
        using E_raw = np.einsum('nij,njk->nik', CGt, S_inv);

        // Terminal masking: E[n-1] = 0
        using term_mask = np.array(
          Array.from({ length: n }, (_, t) => [[t < n - 1 ? 1.0 : 0.0]]),
          { dtype }
        );  // [n, 1, 1]
        // jax-js-lint: allow-non-using — E_all disposed after scan
        const E_all = np.multiply(E_raw, term_mask);  // [n, m, m]

        // ImEG = I - E_k · G  [n, m, m]
        using EG = np.einsum('nij,jk->nik', E_all, G);
        using I_eye = np.eye(stateSize, undefined, { dtype });
        using I_exp = np.tile(np.reshape(I_eye, [1, stateSize, stateSize]), [n, 1, 1]);
        using ImEG = np.subtract(I_exp, EG);

        // g_k = (I - E_k·G) · x̄_k  [n, m, 1]
        // jax-js-lint: allow-non-using — g_all disposed after scan
        const g_all = np.einsum('nij,njk->nik', ImEG, x_filt);

        // L_k (Joseph form — guaranteed PSD):
        //   L_k = (I - E_k·G) · C_filt,k · (I - E_k·G)' + E_k · W · E_k'
        using ImEG_C_ImEGt = np.einsum('nij,njk,nlk->nil', ImEG, C_filt, ImEG);
        using EWEt = np.einsum('nij,jk,nlk->nil', E_all, W, E_all);
        using L_raw = np.add(ImEG_C_ImEGt, EWEt);
        // Symmetrize (f32 stabilization)
        using L_raw_t = np.einsum('nij->nji', L_raw);
        using L_sum = np.add(L_raw, L_raw_t);
        // jax-js-lint: allow-non-using — L_all disposed after scan
        const L_all = np.multiply(np.array(0.5, { dtype }), L_sum);

        // Suffix scan via reverse associativeScan (same compose as forward)
        const smoothed = lax.associativeScan(
          compose,
          { A: E_all, b: g_all, S: L_all },
          { reverse: true }
        ) as ScanElem;

        // Smoothed estimates: x_smooth = g_comp, C_smooth = L_comp
        x_smooth = smoothed.b;      // [n, m, 1]
        C_smooth = smoothed.S;      // [n, m, m]
        smoothed.A.dispose();       // All-zero E values (not needed)
        E_all.dispose();
        g_all.dispose();
        L_all.dispose();
      }

      x_filt.dispose();
      C_filt.dispose();

    } else {
      // ─── Sequential Forward Filter (cpu/wasm) ───
      // fwdSeq fields are disposed individually via fwd.K.dispose() etc.
      // eslint-disable-next-line jax-js/require-scan-result-dispose
      const [fwdCarry, fwdSeq] = lax.scan(
        forwardStep,
        { x: x0, C: C0 },
        { y: y_arr, V2: V2_arr, FF: FF_scan }
      );
      tree.dispose(fwdCarry);
      fwd = fwdSeq as unknown as ForwardY;

      // ─── Sequential Backward RTS Smoother (cpu/wasm) ───
      using x_pred_rev = np.flip(fwd.x_pred, 0);
      using C_pred_rev = np.flip(fwd.C_pred, 0);
      using K_rev = np.flip(fwd.K, 0);
      using v_rev = np.flip(fwd.v, 0);
      using Cp_rev = np.flip(fwd.Cp, 0);
      using FF_rev = np.flip(fwd.FF, 0);
      using mask_rev = np.flip(fwd.mask, 0);

      const [bwdCarry, bwd] = lax.scan(
        backwardStep,
        { r: r0, N: N0 },
        {
          x_pred: x_pred_rev,
          C_pred: C_pred_rev,
          K: K_rev,
          v: v_rev,
          Cp: Cp_rev,
          FF: FF_rev,
          mask: mask_rev,
        }
      );
      tree.dispose(bwdCarry);

      x_smooth = np.flip(bwd.x_smooth, 0);  // [n, m, 1]
      C_smooth = np.flip(bwd.C_smooth, 0);  // [n, m, m]
      tree.dispose(bwd);
    }

    // ─── Observation-space diagnostics ───

    // NaN observation mask [n]: 1.0 where observed, 0.0 where missing.
    // Squeezed from the [n,1,1] mask stored by forwardStep.
    using mask_flat = np.squeeze(fwd.mask);   // [n]

    // yhat = FF @ xf: FF:[n,1,m] @ xf:[n,m,1] → [n,1,1] → [n]
    using yhat_3d = np.matmul(FF_scan, fwd.x_pred);
    const yhat = np.squeeze(yhat_3d);

    // ystd = sqrt(diag(FF @ C_smooth @ FF') + V²)
    // einsum 'nij,njk,nlk->nil' but p=1, so result is [n,1,1] which we squeeze
    using FCFt_3d = np.einsum('nij,njk,nlk->nil', FF_scan, C_smooth, FF_scan);
    using FCFt_flat = np.squeeze(FCFt_3d);
    const ystd = np.sqrt(np.add(FCFt_flat, V2_flat));

    // Innovations [n,1,1] → [n]
    // v is already zeroed at NaN positions (K=0 in forwardStep when NaN)
    const v_flat = np.squeeze(fwd.v);
    const Cp_flat = np.squeeze(fwd.Cp);

    // Dispose fwd.K, fwd.FF, fwd.mask (no longer needed after squeeze)
    fwd.K.dispose();
    fwd.FF.dispose();
    fwd.mask.dispose();

    // y_safe: replace NaN with 0 for numerically safe reductions
    using is_nan_y = np.isnan(y_1d);       // [n] bool
    using y_safe = np.where(is_nan_y, np.zerosLike(y_1d), y_1d);  // [n]

    // Residuals: naturally NaN at missing positions (y_1d has NaN there)
    const resid0 = np.subtract(y_1d, yhat);    // [n]: NaN at missing obs
    const resid  = np.divide(resid0, V_flat);  // [n]: NaN at missing obs
    // Standardised prediction residuals: NaN at missing positions (matching MATLAB)
    using resid2_raw = np.divide(v_flat, np.sqrt(Cp_flat));   // [n]: 0 at NaN pos
    using nan_arr = np.full([n], NaN, { dtype });              // [n] all NaN
    const resid2 = np.where(is_nan_y, nan_arr, resid2_raw);   // [n]: NaN at missing

    // NaN-safe scalar reductions — use mask_flat to exclude missing timesteps
    using resid0_safe = np.subtract(y_safe, yhat);           // [n]: 0 at missing pos
    using resid_safe  = np.divide(resid0_safe, V_flat);      // [n]: 0 at missing pos
    const nobs = np.sum(mask_flat);                          // scalar: count of valid obs
    const ssy  = np.sum(np.multiply(mask_flat, np.square(resid0_safe)));
    const lik  = np.sum(np.multiply(mask_flat, np.add(
      np.divide(np.square(v_flat), Cp_flat),
      np.log(Cp_flat)
    )));
    const s2   = np.divide(
      np.sum(np.multiply(mask_flat, np.square(resid_safe))), nobs);
    const mse  = np.divide(
      np.sum(np.multiply(mask_flat, np.square(resid2_raw))), nobs);
    // Sign-preserving guard: tiny epsilon prevents NaN when y contains exact
    // zeros, but preserves sign (negative y → negative MAPE, matching MATLAB).
    const mape = np.divide(
      np.sum(np.multiply(mask_flat,
        np.divide(np.abs(resid2_raw), np.add(y_safe, np.array(1e-30, { dtype }))))),
      nobs
    );

    return {
      x: x_smooth, C: C_smooth,
      xf: fwd.x_pred, Cf: fwd.C_pred,
      yhat, ystd,
      v: v_flat, Cp: Cp_flat,
      resid0, resid, resid2,
      ssy, lik, s2, mse, mape, nobs,
    };
  };
  
  // Run core — one jit wrapping both scans + all diagnostics
  const coreResult = await jit(core)(x0, C0, y_arr, V2_arr, FF_scan, r0, N0);

  // Dispose DARE tensors — they live outside jit, so manual cleanup is needed.
  if (dare) {
    dare.K_ss.dispose();
    dare.A_ss.dispose();
    dare.Sigma_base.dispose();
    dare.KKt.dispose();
  }

  if (FF_arr === undefined) FF_scan.dispose(); // we own it (created by np.tile)

  return tree.makeDisposable({
    ...coreResult, m: stateSize,
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
 * When X is provided (n×q covariate matrix), the observation equation becomes:
 *   y(t) = F_base·x(t) + X[t,:]·β + v
 *
 * The β coefficients are appended to the state vector as static states
 * (identity evolution, zero process noise), matching the MATLAB DLM convention.
 *
 * System matrices G and F are generated by dlmGenSys().
 * State noise covariance W = diag(w[0]², w[1]², ...) with zeros for
 * states beyond w.length.
 *
 * @param y - Observations (n×1 array)
 * @param s - Observation noise standard deviation: scalar (same for all timesteps)
 *            or array of length n (per-observation sigma, e.g. satellite uncertainty)
 * @param w - State noise standard deviations (diagonal of sqrt(W))
 * @param dtype - Computation precision (default: Float64)
 * @param options - Model specification (default: order=1, no seasonal)
 * @param X - Optional covariate matrix (n rows × q cols); each row is X[t,:]
 * @returns Complete model fit with smoothed estimates and diagnostics
 */
export const dlmFit = async (
  y: ArrayLike<number>,
  s: number | ArrayLike<number>,
  w: number[],
  dtype: DType = DType.Float64,
  options: DlmOptions = {},
  X?: ArrayLike<number>[],  // n×q: X[t] is the covariate row at time t
): Promise<DlmFitResult> => {
  const n = y.length;
  const FA = getFloatArrayType(dtype);

  // Convert input to TypedArray if needed
  const yArr = y instanceof FA ? y : FA.from(y);
  // Observation noise std dev — scalar or per-observation array
  const V_std: InstanceType<typeof FA> = (() => {
    if (typeof s === "number") return new FA(n).fill(s);
    const arr = new FA(n);
    for (let i = 0; i < n; i++) arr[i] = (s as ArrayLike<number>)[i];
    return arr;
  })();

  // ─────────────────────────────────────────────────────────────────────────
  // Generate system matrices from options
  // ─────────────────────────────────────────────────────────────────────────
  const sys = dlmGenSys(options);
  const m_base = sys.m;
  const q = X ? X[0].length : 0;
  const m = m_base + q;  // extended state dimension (includes β)

  // Validate covariate matrix dimensions
  if (X) {
    if (X.length !== n) {
      throw new Error(`X must have ${n} rows (one per observation), got ${X.length}`);
    }
    for (let t = 0; t < n; t++) {
      if (X[t].length !== q) {
        throw new Error(`X[${t}] has ${X[t].length} columns, expected ${q}`);
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Extend G, W for covariate β states (static: identity in G, zero in W)
  // When q=0 this is a no-op and we use the base matrices directly.
  // ─────────────────────────────────────────────────────────────────────────
  const G_data: number[][] = m === m_base
    ? sys.G
    : [
        ...sys.G.map(row => [...row, ...new Array(q).fill(0)]),
        ...Array.from({ length: q }, (_, k) =>
          [...new Array(m_base).fill(0), ...Array.from({ length: q }, (_, j) => j === k ? 1 : 0)]
        ),
      ];

  const W_data: number[][] = Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => {
      // β states (indices m_base..m-1) have zero process noise
      if (i >= m_base || j >= m_base) return 0;
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

  using G = np.array(G_data, { dtype });
  // Base F (1×m_base) — used only for building FF_arr; not passed to dlmSmo directly
  using F_base = np.array([sys.F], { dtype });
  using W = np.array(W_data, { dtype });

  // ─────────────────────────────────────────────────────────────────────────
  // Build time-varying FF_arr [n, 1, m] when covariates present.
  // FF_arr[t] = [[F_base[0], F_base[1], ..., X[t,0], ..., X[t,q-1]]]
  // When q=0, pass undefined so dlmSmo builds FF_scan from static F.
  // ─────────────────────────────────────────────────────────────────────────
  let FF_arr: np.Array | undefined;
  if (q > 0 && X) {
    const FF_data: number[][][] = Array.from({ length: n }, (_, t) => [
      [...sys.F, ...Array.from(X[t])]
    ]);
    FF_arr = np.array(FF_data, { dtype });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Initialize state with diffuse prior
  // x0[0] = mean of first ns observations (level); rest = 0
  // β states start at 0 with large uncertainty (diffuse prior)
  // ─────────────────────────────────────────────────────────────────────────
  const ns = options.ns ?? 12;
  let initSum = 0, initCount = 0;
  const count = Math.min(ns, n);
  for (let i = 0; i < count; i++) {
    const v = Number(yArr[i]);
    if (!isNaN(v)) { initSum += v; initCount++; }
  }
  // NaN-safe mean: use available observations; fall back to 0 if all missing
  const mean_y = initCount > 0 ? initSum / initCount : 0;
  // Initial covariance: diagonal with large uncertainty (diffuse prior)
  const c0_val = (Math.abs(mean_y) * 0.5) ** 2;
  const c0 = c0_val === 0 ? 1e7 : c0_val;
  const x0_data: number[][] = Array.from({ length: m }, (_, i) =>
    [i === 0 ? mean_y : 0.0]
  );
  const C0_data: number[][] = Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => (i === j ? c0 : 0.0))
  );

  // Plain-JS system data for DARE (associativeScan steady-state Kalman gain).
  // F_data is extended to m dimensions (base F + zeros for covariate states).
  const F_data_ext: number[] = q > 0
    ? [...sys.F, ...new Array(q).fill(0)]
    : sys.F;
  const sysData = { G_data, F_data: F_data_ext, W_data };

  // ─────────────────────────────────────────────────────────────────────────
  // Pass 1: Initial smoother to refine starting values
  // ─────────────────────────────────────────────────────────────────────────
  let x0_updated: number[][];
  let C0_scaled: number[][];
  { // Block scope — `using` auto-disposes all Pass 1 arrays at block end
    using out1 = await dlmSmo(yArr, F_base, V_std, x0_data, G, W, C0_data, m, dtype, FF_arr, sysData);
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
  const out2 = await dlmSmo(yArr, F_base, V_std, x0_updated, G, W, C0_scaled, m, dtype, FF_arr, sysData);

  FF_arr?.dispose();

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
  const nobs = Math.round(await toNum(out2.nobs));  // count of non-NaN observations

  return {
    // State estimates (m = m_base + q; last q states are β coefficients)
    xf, Cf, x: x_out, C: C_out, xstd,
    // System matrices (plain arrays for easy serialization)
    G: G_data,
    F: sys.F,
    W: W_data,
    // Input data
    y: yArr, V: V_std,
    // Initial state (after Pass 1 refinement)
    x0: x0_updated.map(row => row[0]),
    C0: C0_scaled,
    // Covariates matrix (stored as row vectors; empty array when X not provided)
    XX: X ? Array.from({ length: n }, (_, t) => Array.from(X[t]) as number[]) : [],
    // Predictions and residuals
    yhat, ystd, resid0, resid, resid2,
    // Diagnostics
    ssy, v, Cp: Cp_arr, s2,
    nobs,
    lik, mse, mape,
    class: 'dlmfit',
  };
};

/**
 * Forecast h steps ahead from the end of a fitted DLM.
 *
 * Starting from the last smoothed state (`fit.x[:][n-1]`, `fit.C[:][:][-1]`),
 * iterates the state-space model forward h times with no observations:
 *
 *   x_pred(k+1) = G · x_pred(k)                      (state mean)
 *   C_pred(k+1) = G · C_pred(k) · G' + W              (state covariance)
 *   yhat(k)     = FF_k · x_pred(k)                    (observation mean)
 *   ystd(k)     = sqrt(FF_k · C_pred(k) · FF_k' + s²) (observation std)
 *
 * This is the standard Kalman prediction step with no measurement update —
 * equivalent to appending NaN observations and running dlmFit on the extended
 * series, but cheaper (O(h) vs O(n+h)) because it skips the full filter+smoother
 * pass over the already-fitted data.
 *
 * **Equivalence with NaN-extended dlmFit:**
 * Appending NaN values to `y` and calling `dlmFit` on the extended series
 * produces numerically identical `yhat`/`ystd` for the appended steps, because
 * the RTS smoother propagates no new information backwards through NaN steps.
 * Use that pattern instead when:
 *   - You have *some* known future observations (partial future data, revised
 *     estimates, scenario constraints) — mix real values and NaN freely.
 *   - You want the smoothed state trajectory to continue into the forecast window
 *     as part of the same `DlmFitResult` (e.g. for plotting continuity).
 *
 * All model types are supported: local level/trend, full/trigonometric seasonal,
 * AR(p), and covariate (β) models. Covariate states (static β blocks in G/W)
 * are propagated correctly; pass X_forecast for their observation contributions.
 *
 * The jittable core uses `lax.scan` over h steps, capturing G and W as
 * constants. The scan input is a time-varying FF_scan [h,1,m] so that
 * covariate F rows are included inside the same compiled body.
 *
 * @param fit - DlmFitResult from dlmFit (provides G, F, W, last smoothed state)
 * @param s - Observation noise std dev (scalar, same as used in dlmFit)
 * @param h - Forecast horizon (number of steps ahead)
 * @param dtype - Computation precision (should match the dtype used in dlmFit)
 * @param X_forecast - Optional covariate rows for forecast steps (h rows × q cols).
 *                     If omitted (or shorter than h), missing covariate entries are
 *                     treated as zero in the current implementation. This gives a
 *                     baseline conditional forecast where unknown driver effects are
 *                     set to zero. For scientifically neutral use, center drivers
 *                     before fitting so zero means "typical" driver level.
 * @returns Predicted state means, covariances, and observation predictions for steps 1…h
 */
export const dlmForecast = async (
  fit: DlmFitResult,
  s: number,
  h: number,
  dtype: DType = DType.Float64,
  X_forecast?: ArrayLike<number>[],
): Promise<DlmForecastResult> => {
  const { G: G_data, F: F_data, W: W_data } = fit;
  const m = G_data.length;
  const q = fit.XX && (fit.XX as number[][])[0]?.length > 0
    ? (fit.XX as number[][])[0].length
    : 0;
  const n = fit.x[0].length;
  const FA = getFloatArrayType(dtype);

  // ── Build constant np.Arrays for G and W (captured by jit core) ──────────
  using G_np = np.array(G_data, { dtype });
  using W_np = np.array(W_data, { dtype });

  // ── Initial state: last smoothed timestep ─────────────────────────────────
  const x0_data: number[][] = Array.from({ length: m }, (_, i) => [fit.x[i][n - 1]]);
  const C0_data: number[][] = Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => fit.C[i][j][n - 1])
  );
  using x0 = np.array(x0_data, { dtype });
  using C0 = np.array(C0_data, { dtype });

  // ── FF_scan [h,1,m]: observation matrix for each forecast step ────────────
  // Base F is always the same; covariate rows are appended from X_forecast.
  const FF_data: number[][][] = Array.from({ length: h }, (_, k) => {
    const row = [...F_data];
    if (q > 0) {
      const xrow = X_forecast ? X_forecast[k] : null;
      for (let qi = 0; qi < q; qi++) row.push(xrow ? Number(xrow[qi]) : 0);
    }
    return [row];  // shape [1, m]
  });
  using FF_scan = np.array(FF_data, { dtype });

  // s² as constant scalar array [1,1]
  using s2_arr = np.array([[s * s]], { dtype });

  // ── Jittable prediction step (no measurement update) ─────────────────────
  // carry: { x: [m,1], C: [m,m] }
  // input: { FF: [1,m] }  — one row per forecast step
  // output per step: { x: [m,1], C: [m,m], yhat: [1,1], ystd: [1,1] }
  type PredCarry = { x: np.Array; C: np.Array };
  type PredInp   = { FF: np.Array };
  type PredOut   = { x: np.Array; C: np.Array; yhat: np.Array; ystd: np.Array };

  const predStep = (carry: PredCarry, inp: PredInp): [PredCarry, PredOut] => {
    const { x: xi, C: Ci } = carry;
    const { FF: FFi } = inp;

    // x_new = G · x  [m,1]
    const x_new = np.matmul(G_np, xi);

    // C_new = G · C · G' + W  [m,m]
    const C_new = np.add(
      np.einsum('ij,jk,lk->il', G_np, Ci, G_np),
      W_np
    );

    // yhat = FF · x_new  [1,1]
    const yhat = np.matmul(FFi, x_new);

    // ystd = sqrt(FF·C_new·FF' + s²)  [1,1]
    using FCFt = np.einsum('ij,jk,lk->il', FFi, C_new, FFi);
    const ystd = np.sqrt(np.add(FCFt, s2_arr));

    return [{ x: x_new, C: C_new }, { x: x_new, C: C_new, yhat, ystd }];
  };

  // ── Jittable core: scan over h steps ─────────────────────────────────────
  const core = (x0: np.Array, C0: np.Array, FF_scan: np.Array) => {
    const [finalCarry, outputs] = lax.scan(
      predStep,
      { x: x0, C: C0 },
      { FF: FF_scan }
    );
    tree.dispose(finalCarry);
    return outputs;  // { x: [h,m,1], C: [h,m,m], yhat: [h,1,1], ystd: [h,1,1] }
  };

  const out = await jit(core)(x0, C0, FF_scan);

  // ── Extract results into TypedArrays ─────────────────────────────────────
  const x_raw    = await out.x.consumeData()    as ArrayLike<number>;  // [h,m,1]
  const C_raw    = await out.C.consumeData()    as ArrayLike<number>;  // [h,m,m]
  const yhat_raw = await out.yhat.consumeData() as ArrayLike<number>;  // [h,1,1]
  const ystd_raw = await out.ystd.consumeData() as ArrayLike<number>;  // [h,1,1]

  const yhat_out = new FA(h);
  const ystd_out = new FA(h);
  for (let k = 0; k < h; k++) {
    yhat_out[k] = yhat_raw[k] as number;
    ystd_out[k] = ystd_raw[k] as number;
  }

  // x[i][k] = x_raw[k * m + i]
  const x_out: FloatArray[] = Array.from({ length: m }, (_, i) => {
    const arr = new FA(h);
    for (let k = 0; k < h; k++) arr[k] = x_raw[k * m + i] as number;
    return arr;
  });

  // C[i][j][k] = C_raw[k * m * m + i * m + j]
  const C_out: FloatArray[][] = Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => {
      const arr = new FA(h);
      for (let k = 0; k < h; k++) arr[k] = C_raw[k * m * m + i * m + j] as number;
      return arr;
    })
  );

  // xstd[k][i] = sqrt(|C[i][i][k]|)
  const xstd_out: FloatArray[] = Array.from({ length: h }, (_, k) => {
    const arr = new FA(m);
    for (let i = 0; i < m; i++)
      arr[i] = Math.sqrt(Math.abs(C_raw[k * m * m + i * m + i] as number));
    return arr;
  });

  return { yhat: yhat_out, ystd: ystd_out, x: x_out, C: C_out, xstd: xstd_out, h, m };
};
