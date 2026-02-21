import { DType, numpy as np, lax, jit, tree, defaultDevice } from "@hamk-uas/jax-js-nonconsuming";
import type { DlmSmoResult, FloatArray } from "./types";
import {
  getFloatArrayType, parseDtype,
  StateMatrix, CovMatrix,
} from "./types";
import type {
  DlmFitResult, DlmForecastResult, DlmTensorResult,
  DlmFitOptions, DlmForecastOptions, DlmFitResultMatlab,
  DlmStabilization,
} from "./types";
import { dlmGenSys, dlmGenSysTV } from "./dlmgensys";
import type { DlmOptions } from "./dlmgensys";

// Public type exports
export type {
  DlmFitResult, DlmForecastResult, DlmTensorResult,
  DlmFitOptions, DlmForecastOptions, DlmFitResultMatlab,
  DlmDtype, DlmAlgorithm, DlmLossFn, DlmStabilization,
  FloatArray,
} from "./types";
export { StateMatrix, CovMatrix } from "./types";
export type { DlmOptions, DlmSystem, DlmSystemTV } from "./dlmgensys";
export { dlmGenSys, dlmGenSysTV, findArInds } from "./dlmgensys";
export { dlmMLE, toMatlabMle } from "./mle";
export type { DlmMleResult, DlmMleResultMatlab, DlmMleOptions } from "./mle";

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
  G_scan: np.Array,   // [n, m, m] per-step transition matrix
  W_scan: np.Array,   // [n, m, m] per-step noise covariance
  C0_data: number[][],
  stateSize: number,
  dtype: DType = DType.Float64,
  FF_arr?: np.Array,  // [n, 1, m_ext] time-varying F (covariates)
  forceAssocScan?: boolean,
  stabilization?: DlmStabilization,
): Promise<DlmSmoResult & Disposable> => {
  const n = y.length;

  // ─────────────────────────────────────────────────────────────────────────
  // Branch selection: three execution paths based on device + dtype
  //
  //   wasm/cpu + Float64  →  sequential scan + triu+triu' symmetrization (default)
  //   cpu      + Float32  →  sequential scan + Joseph form + triu/avg sym + cEps
  //   webgpu   + Float32  →  associativeScan forward + Joseph form/sym/cEps
  //
  // Float64 default: triu(C)+triu(C,1)' symmetrization after each filter and
  // smoother step (matches MATLAB dlmsmo.m line 77). Reduces max relative error
  // vs Octave reference from ~2e-9 to ~4e-12 — ~500× improvement at negligible
  // cost (two np.triu calls per step). Disable with stabilization:{cTriuSym:false}.
  //
  // Float32 default: Joseph form (L·C·L' + K·V²·K' + W), (C+C')/2 symmetrize,
  // and C += 1e-6·I (cEps) — prevents covariance from going non-PD for m > 2.
  //
  // The associativeScan path (webgpu) reformulates the forward Kalman filter
  // as an associative prefix scan per Särkkä & García-Fernández (2020),
  // reducing sequential depth from O(n) to O(log n) on parallel hardware.
  // ─────────────────────────────────────────────────────────────────────────
  const device = defaultDevice();
  const f32 = dtype === DType.Float32;
  const useAssocScan = forceAssocScan || (f32 && device === 'webgpu');

  // ── Stabilization flags (f32 sequential backward step only) ──────────────────
  // Flags are captured as JS constants — each unique combination produces a
  // different JIT-compiled kernel (acceptable for research/exploration).
  const stabNSym     = stabilization?.nSym      ?? false;
  const stabNDiag    = stabilization?.nDiag     ?? false;
  const stabNDiagAbs = stabilization?.nDiagAbs  ?? false;
  const stabNLeak    = stabilization?.nLeak     ?? false;
  const stabCDiag    = stabilization?.cDiag     ?? false;
  const stabCEps     = stabilization?.cEps      ?? false;  // no-op (cEps now unconditional for f32)
  const stabCDiagAbs = stabilization?.cDiagAbs  ?? false;
  // cTriuSym: default true for f64 (matches MATLAB dlmsmo.m triu+triu' sym),
  //           default false for f32 (uses (C+C')/2 instead; triu has no benefit for f32).
  // Override with stabilization: { cTriuSym: false } to disable for f64.
  const stabCTriuSym    = stabilization?.cTriuSym    ?? !f32;
  const stabCSmoAbsDiag = stabilization?.cSmoAbsDiag ?? false;  // abs(diag(C_smooth)) (f32+f64)
  // Pre-computed [m,m] constant tensors for stabilization ops.
  // Created unconditionally to avoid conditional `using` complexity.
  // Captured by backwardStep closure; disposed when dlmSmo scope exits after jit.
  using stab_I_eye       = np.eye(stateSize, undefined, { dtype });               // [m,m]
  using stab_off_I       = np.subtract(np.ones([stateSize, stateSize], { dtype }), stab_I_eye);
  using stab_nLeak_fact  = np.array(1.0 - 1e-5, { dtype });                       // scalar
  using stab_cDiag_eps_I = np.multiply(np.array(1e-7, { dtype }), stab_I_eye);   // [m,m]
  using stab_cEps_I      = np.multiply(np.array(1e-6, { dtype }), stab_I_eye);   // [m,m]

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
  // Step functions receive FF_t ([1, m]), G_t ([m, m]), and W_t ([m, m])
  // from the scan pytree. All three can be time-varying.
  // ─────────────────────────────────────────────────────────────────────────

  // Constant [1,1] ones tensor captured by forwardStep closure for NaN masking.
  using const_one_11 = np.array([[1.0]], { dtype });

  type ForwardCarry = { x: np.Array; C: np.Array };
  type ForwardX = { y: np.Array; V2: np.Array; FF: np.Array; Gt: np.Array; Wt: np.Array };
  type ForwardY = {
    x_pred: np.Array; C_pred: np.Array;
    K: np.Array; v: np.Array; Cp: np.Array; FF: np.Array;
    Gt: np.Array;  // [m,m] per-step transition — passed through for backward step
    mask: np.Array;  // [1,1]: 1.0 if observed, 0.0 if NaN
  };
  
  const forwardStep = (
    carry: ForwardCarry,
    inp: ForwardX
  ): [ForwardCarry, ForwardY] => {
    const { x: xi, C: Ci } = carry;
    const { y: yi, V2: V2i, FF: FFi, Gt: G_t, Wt: W_t } = inp;

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
    using GCFFt = np.einsum('ij,jk,lk->il', G_t, Ci, FFi);
    using K_raw = np.divide(GCFFt, Cp);
    const K = np.multiply(mask_t, K_raw);  // [1,1]×[m,1] → [m,1] by broadcast

    // L = G - K·FF  [m,m]  (= G when NaN, since K=0)
    using L = np.subtract(G_t, np.matmul(K, FFi));

    // Next state prediction: x_next = G·x + K·v  [m,1]
    // When NaN: x_next = G·x (no measurement update)
    const x_next = np.add(
      np.matmul(G_t, xi),
      np.matmul(K, v)
    );

    // Next covariance: C_next depends on dtype + stabilization flags.
    //
    // Float64 default (matches MATLAB DLM reference formula):
    //   C_next = G·C·L' + W            (+ triu+triu' sym if cTriuSym is set)
    //
    // Float32 (Joseph form — numerically stable):
    //   C_next = L·C·L' + K·V²·K' + W  (+ sym)
    //
    // The Joseph form guarantees positive semi-definiteness by construction
    // and avoids the implicit subtraction in G·C·L' that causes catastrophic
    // cancellation for m > 2 in Float32.
    //
    // Symmetrization options (applied after the covariance formula):
    //   default f32:  (C + C') / 2
    //   cTriuSym f32: triu(C) + triu(C,1)'  — upper triangle authoritative
    //   cTriuSym f64: triu(C) + triu(C,1)'  — mirrors MATLAB dlmsmo.m line 77
    let C_next: np.Array;
    {
      // Compute raw forward covariance.
      // jax-js-lint: allow-non-using — sym branch takes ownership below
      let C_fwd_raw: np.Array;
      if (f32) {
        // Joseph form: L·C·L' + K·V²·K' + W
        using LCLt = np.einsum('ij,jk,lk->il', L, Ci, L);
        using KV2Kt = np.multiply(V2i, np.matmul(K, np.transpose(K)));
        using sum1 = np.add(LCLt, KV2Kt);
        C_fwd_raw = np.add(sum1, W_t);
      } else {
        // Standard form (matches MATLAB dlmsmo.m): G·C·L' + W
        C_fwd_raw = np.add(np.einsum('ij,jk,lk->il', G_t, Ci, L), W_t);
      }

      // Apply symmetrization for f32 (always) or f64+cTriuSym.
      if (f32 || stabCTriuSym) {
        if (stabCTriuSym) {
          // triu(C) + triu(C,1)': upper triangle authoritative, mirrored to lower.
          // Matches MATLAB dlmsmo.m line 77. Works for both f32 and f64.
          using C_upper = np.triu(C_fwd_raw);
          using C_sup   = np.triu(C_fwd_raw, 1);
          using C_sup_t = np.transpose(C_sup);
          C_next = np.add(C_upper, C_sup_t);
        } else {
          // f32 default: average both triangles (C + C') / 2
          using Ct      = np.transpose(C_fwd_raw);
          using sumBoth = np.add(C_fwd_raw, Ct);
          C_next = np.multiply(np.array(0.5, { dtype }), sumBoth);
        }
        C_fwd_raw.dispose();
      } else {
        // f64 default: raw (no symmetrization)
        C_next = C_fwd_raw;
      }
    }
    
    return [
      { x: x_next, C: C_next },
      // Pass FFi, Gt, and mask through so backward pass can reuse them
      { x_pred: xi, C_pred: Ci, K, v, Cp, FF: FFi, Gt: G_t, mask: mask_t } as ForwardY,
    ];
  };
  
  type BackwardCarry = { r: np.Array; N: np.Array };
  type BackwardX = {
    x_pred: np.Array; C_pred: np.Array;
    K: np.Array; v: np.Array; Cp: np.Array; FF: np.Array;
    Gt: np.Array;   // [m,m] per-step transition matrix (from forward pass)
    mask: np.Array;  // [1,1]: 1.0 if observed, 0.0 if NaN (mirrors forwardStep)
  };
  type BackwardY = { x_smooth: np.Array; C_smooth: np.Array };
  
  const backwardStep = (
    carry: BackwardCarry,
    inp: BackwardX
  ): [BackwardCarry, BackwardY] => {
    const { r, N } = carry;
    const { x_pred: xi, C_pred: Ci, K: Ki, v: vi, Cp: Cpi, FF: FFi, Gt: G_t, mask: maski } = inp;

    // L = G - K·FF  [m,m]  (K=0 when NaN → L=G, propagating prior)
    using L = np.subtract(G_t, np.matmul(Ki, FFi));

    // FF'·Cp⁻¹  [m,1] (scalar division valid for p=1)
    using FFt = np.transpose(FFi);
    using FtCpInv = np.divide(FFt, Cpi);

    // r_new = F'·Cp⁻¹·v + L'·r  [m,1]
    // vi is already 0 at NaN positions (zeroed in forwardStep), so
    // FtCpInv·vi contributes 0 automatically at gapped timesteps.
    const r_new = np.add(
      np.multiply(FtCpInv, vi),
      np.matmul(np.transpose(L), r)
    );

    // N_new = mask·(FF'·Cp⁻¹·FF) + L'·N·L  [m,m]
    // The outer-product term must be masked: at NaN timesteps it would
    // otherwise add spurious Fisher information to N, causing the smoother
    // to over-shrink state uncertainty at and around gappedobservations.
    //
    // NUMERICAL PRECISION NOTE:
    // The L'·N·L product via einsum uses two pairwise dot() calls.
    // Since jax-js-nonconsuming v0.2.1, Float64 uses Kahan compensated summation
    // in each dot, but errors still propagate into C_smooth via the
    // C·N·C product below. N accumulates information over the
    // backward pass, so rounding compounds across timesteps.
    using FtCpInvFF = np.matmul(FtCpInv, FFi);  // [m,m]
    // jax-js-lint: allow-non-using — N stabilization below may replace this binding
    let N_new = np.add(
      np.multiply(maski, FtCpInvFF),  // [1,1]×[m,m] → [m,m]: 0 when NaN
      np.einsum('ji,jk,kl->il', L, N, L)
    );

    // ── N stabilization (f32 only, sequential path) ─────────────────────────
    // Applied in order: nSym → nDiag → nLeak. Each may replace N_new with a
    // stabilized version (accumulator-swap pattern). Ignored on f64 / assoc path.
    if (f32) {
      if (stabNSym) {
        // Symmetrize: N = 0.5*(N + N').
        // N is mathematically symmetric but f32 rounding in L'·N·L breaks this;
        // asymmetries compound each step because the result feeds back as input.
        using Nt = np.transpose(N_new);
        using Nsum = np.add(N_new, Nt);
        // jax-js-lint: allow-non-using — accumulator-swap: N_new.dispose() + N_new = N_stab below
        const N_stab = np.multiply(np.array(0.5, { dtype }), Nsum);
        N_new.dispose();
        N_new = N_stab;
      }
      if (stabNDiag) {
        // Clamp diagonal of N to >= 0.
        // N is an information matrix (should be PSD); f32 rounding can push
        // diagonal entries negative, causing C·N·C to undercorrect.
        // Strategy: split N into diagonal (N*I) and off-diagonal (N*(1-I)) parts,
        // clamp the diagonal part to zero, recombine.
        // max(N*I, 0) correctly clamps diagonal; off-diag: max(0,0)=0 (no change).
        using N_d = np.multiply(N_new, stab_I_eye);
        using N_o = np.multiply(N_new, stab_off_I);
        using N_d_c = np.maximum(N_d, np.zerosLike(N_d));
        // jax-js-lint: allow-non-using — accumulator-swap: N_new.dispose() + N_new = N_stab below
        const N_stab = np.add(N_d_c, N_o);
        N_new.dispose();
        N_new = N_stab;
      }
      if (stabNDiagAbs) {
        // Abs diagonal of N: diag(N) = |diag(N)|.
        // Stronger than nDiag: sign-flips barely-negative entries rather than
        // zeroing them, preserving their magnitude as an information estimate.
        using N_d = np.multiply(N_new, stab_I_eye);
        using N_o = np.multiply(N_new, stab_off_I);
        using N_d_a = np.abs(N_d);
        // jax-js-lint: allow-non-using — accumulator-swap: N_new.dispose() + N_new = N_stab below
        const N_stab = np.add(N_d_a, N_o);
        N_new.dispose();
        N_new = N_stab;
      }
      if (stabNLeak) {
        // Slight forgetting: N *= (1 - 1e-5) per step.
        // Prevents N from accumulating unboundedly, which would cause
        // C·N·C to overshoot and produce negative variances in C_smooth.
        // jax-js-lint: allow-non-using — accumulator-swap: N_new.dispose() + N_new = N_stab below
        const N_stab = np.multiply(stab_nLeak_fact, N_new);
        N_new.dispose();
        N_new = N_stab;
      }
    }

    // x_smooth = x_pred + C_pred·r_new  [m,1]
    const x_smooth = np.add(xi, np.matmul(Ci, r_new));

    // C_smooth = C_pred - C_pred·N_new·C_pred  [m,m]
    //
    // NUMERICAL PRECISION NOTE — MOST SENSITIVE OPERATION:
    // This subtraction is the single largest source of numerical error in the DLM.
    // When the smoothing correction C·N·C ≈ C_pred, catastrophic cancellation
    // produces a small result with large relative error.
    //
    // f32 stabilization (always applied, in order):
    //   1. symmetrize: (C+C')/2 default, or triu(C)+triu(C,1)' if cTriuSym
    //   2. cEps: C += 1e-6·I  (unconditional; reduces kaisaniemi m=4 err 1.37e-2→9.66e-3)
    //   3. optional: cDiag | cDiagAbs | cSmoAbsDiag  (all magnitude-preserving variants)
    //
    // f64 + cTriuSym + cSmoAbsDiag = MATLAB dlmsmo.m exact stabilization:
    //   triu(C)+triu(C,1)' (line 77 analog) + abs(diag(C)) (lines 114-115)
    //   reduces max |Δ| vs Octave reference from ~3.78e-8 to ~9e-11.
    let C_smooth: np.Array;
    {
      using C_raw = np.subtract(
        Ci,
        np.einsum('ij,jk,kl->il', Ci, N_new, Ci)
      );
      if (f32) {
        // ── f32 backward smoother ────────────────────────────────────────────
        // Step 1: symmetrize
        // jax-js-lint: allow-non-using — cEps step takes ownership below
        let C_sym: np.Array;
        if (stabCTriuSym) {
          // triu+triu': upper triangle authoritative, mirrors MATLAB dlmsmo.m
          using C_upper = np.triu(C_raw);
          using C_sup   = np.triu(C_raw, 1);
          using C_sup_t = np.transpose(C_sup);
          C_sym = np.add(C_upper, C_sup_t);
        } else {
          // Default: average both triangles
          using Ct   = np.transpose(C_raw);
          using sumC = np.add(C_raw, Ct);
          C_sym = np.multiply(np.array(0.5, { dtype }), sumC);
        }
        // Step 2: always add cEps (unconditional for f32)
        // jax-js-lint: allow-non-using — post-cEps branch takes ownership below
        const C_eps = np.add(C_sym, stab_cEps_I);
        C_sym.dispose();
        // Step 3: optional post-correction
        const useAbsDiag = stabCDiagAbs || stabCSmoAbsDiag;
        if (stabCDiag) {
          using C_d = np.multiply(C_eps, stab_I_eye);
          using C_o = np.multiply(C_eps, stab_off_I);
          using C_d_c = np.maximum(C_d, stab_cDiag_eps_I);
          C_smooth = np.add(C_d_c, C_o);
          C_eps.dispose();
        } else if (useAbsDiag) {
          // abs(diag): magnitude-preserving sign-flip on diagonal, off-diag intact.
          // Covers both cDiagAbs and cSmoAbsDiag (same operation).
          using C_d = np.multiply(C_eps, stab_I_eye);
          using C_o = np.multiply(C_eps, stab_off_I);
          using C_d_a = np.abs(C_d);
          C_smooth = np.add(C_d_a, C_o);
          C_eps.dispose();
        } else {
          C_smooth = C_eps;  // default: sym + cEps only
        }
      } else if (stabCTriuSym) {
        // ── f64 + cTriuSym: mirrors MATLAB dlmsmo.m triu+triu' symmetrize ───
        using C_upper = np.triu(C_raw);
        using C_sup   = np.triu(C_raw, 1);
        using C_sup_t = np.transpose(C_sup);
        // jax-js-lint: allow-non-using — cSmoAbsDiag branch may take ownership
        const C_sym = np.add(C_upper, C_sup_t);
        if (stabCSmoAbsDiag) {
          // abs(diag(C_smooth)): matches MATLAB dlmsmo.m lines 114-115.
          using C_d = np.multiply(C_sym, stab_I_eye);
          using C_o = np.multiply(C_sym, stab_off_I);
          using C_d_a = np.abs(C_d);
          C_smooth = np.add(C_d_a, C_o);
          C_sym.dispose();
        } else {
          C_smooth = C_sym;
        }
      } else {
        // f64 default: raw result (matches MATLAB dlmsmo.m formula, no corrective steps)
        C_smooth = np.add(C_raw, np.zeros([stateSize, stateSize], { dtype }));
      }
    }

    return [{ r: r_new, N: N_new }, { x_smooth, C_smooth }];
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Jittable core: forward Kalman filter + backward RTS smoother +
  // diagnostics computed with vectorized numpy ops.
  // G_scan [n,m,m] and W_scan [n,m,m] are time-varying (or tiled uniform).
  // FF_scan [n,1,m] is threaded through scan for time-varying F support.
  // Returns stacked tensors for arbitrary state dimension m.
  // ─────────────────────────────────────────────────────────────────────────
  
  const core = (
    x0: np.Array, C0: np.Array,
    y_arr: np.Array, V2_arr: np.Array,
    FF_scan: np.Array,
    G_scan: np.Array, W_scan: np.Array,
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

    if (useAssocScan) {
      // ─── Exact Parallel Forward Filter (Särkkä & García-Fernández 2020, Lemmas 1–2) ───
      // 5-tuple elements: (A, b, C, eta, J)
      type ForwardElem = { A: np.Array; b: np.Array; C: np.Array; eta: np.Array; J: np.Array };
      type BackwardElem = { A: np.Array; b: np.Array; S: np.Array };

      using is_nan = np.isnan(y_arr);                    // [n,1,1] bool
      using zero_n11 = np.zerosLike(y_arr);              // [n,1,1]
      using one_n11 = np.onesLike(y_arr);                // [n,1,1]
      // jax-js-lint: allow-non-using — stored in fwd.mask, disposed after backward pass
      const mask_arr = np.where(is_nan, zero_n11, one_n11); // [n,1,1]
      using y_safe_arr = np.where(is_nan, zero_n11, y_arr); // [n,1,1]

      using I_eye = np.eye(stateSize, undefined, { dtype });
      using I_exp = np.tile(np.reshape(I_eye, [1, stateSize, stateSize]), [n, 1, 1]);

      // ─── Arriving vs departing G/W convention ───
      //
      // G_scan[k] / W_scan[k] encode the departing transition from obs k to
      // obs k+1:  Δt = T[k+1] − T[k].  (Used by backward smoother.)
      //
      // The forward element at position k needs the arriving transition
      // (from obs k−1 to obs k):  Δt = T[k] − T[k−1].
      //
      //   G_arriving[k] = G_departing[k−1] = G_scan[k−1]   for k ≥ 1
      //   G_arriving[0] = G(1) (prior → first obs, unit step; value discarded
      //     because element 0 is overwritten with the exact first element)
      //
      // Build G_arriving / W_arriving by prepending G(1)/W(1) and taking
      // G_scan[0:n−1].
      const gArrParts = np.split(G_scan, [n - 1], 0);
      const wArrParts = np.split(W_scan, [n - 1], 0);
      using G_head_arr = gArrParts[0];  // G_scan[0..n-2] = arriving for steps 1..n-1
      gArrParts[1].dispose();
      using W_head_arr = wArrParts[0];  // W_scan[0..n-2] = arriving for steps 1..n-1
      wArrParts[1].dispose();
      // G/W for arriving step 0: unit Δt (matches uniform prior convention)
      using G_unit_1 = np.reshape(np.tile(np.reshape(I_eye, [1, stateSize, stateSize]), [1, 1, 1]), [1, stateSize, stateSize]);
      // For G_arriving[0], use the identity-like uniform G. Since element 0 is
      // overwritten by exact initialization, we just need a valid [1,m,m] tensor.
      // Use G_scan[n-1] which encodes Δt=1 (the last departing step with unit Δt).
      const gLastParts = np.split(G_scan, [n - 1], 0);
      using G_last = gLastParts[1];     // G_scan[n-1], shape [1,m,m]
      gLastParts[0].dispose();
      const wLastParts = np.split(W_scan, [n - 1], 0);
      using W_last = wLastParts[1];     // W_scan[n-1], shape [1,m,m]
      wLastParts[0].dispose();
      using G_arriving = np.concatenate([G_last, G_head_arr], 0);  // [n, m, m]
      using W_arriving = np.concatenate([W_last, W_head_arr], 0);  // [n, m, m]

      // Per-step observed-element construction (Lemma 1, using arriving G/W)
      using S_obs = np.add(np.einsum('nij,njk,nlk->nil', FF_scan, W_arriving, FF_scan), V2_arr); // [n,1,1]
      using WFt = np.einsum('nij,nkj->nik', W_arriving, FF_scan);                               // [n,m,1]
      using K_obs = np.divide(WFt, S_obs);                                                  // [n,m,1]

      using KF_obs = np.einsum('nij,njk->nik', K_obs, FF_scan);                             // [n,m,m]
      using ImKF_obs = np.subtract(I_exp, KF_obs);                                           // [n,m,m]
      using A_obs = np.einsum('nij,njk->nik', ImKF_obs, G_arriving);                         // [n,m,m]
      using C_obs = np.einsum('nij,njk->nik', ImKF_obs, W_arriving);                         // [n,m,m]
      using b_obs = np.multiply(K_obs, y_safe_arr);                                         // [n,m,1]

      using Ft = np.einsum('nij->nji', FF_scan);                                             // [n,m,1]
      using Ft_over_S = np.divide(Ft, S_obs);                                                // [n,m,1]
      using Gt_batch = np.einsum('nij->nji', G_arriving);                                    // [n,m,m]
      using eta_obs_base = np.einsum('nij,njk->nik', Gt_batch, Ft_over_S);                   // [n,m,1]
      using eta_obs = np.multiply(eta_obs_base, y_safe_arr);                                 // [n,m,1]
      using FtF_over_S = np.einsum('nij,njk->nik', Ft_over_S, FF_scan);                      // [n,m,m]
      using J_obs = np.einsum('nij,njk,nkl->nil', Gt_batch, FtF_over_S, G_arriving);         // [n,m,m]

      // NaN handling for k>=2 elements: pure prediction for gapped y
      using A_all = np.where(np.tile(is_nan, [1, stateSize, stateSize]), G_arriving, A_obs);
      using b_all = np.multiply(mask_arr, b_obs);
      using C_all = np.where(np.tile(is_nan, [1, stateSize, stateSize]), W_arriving, C_obs);
      using zero_nmm = np.zerosLike(J_obs);
      using eta_all = np.multiply(mask_arr, eta_obs);
      using J_all = np.where(np.tile(is_nan, [1, stateSize, stateSize]), zero_nmm, J_obs);

      // First element (k=1): exact initialization from prior (A1=0, b1/C1 from x0/C0)
      const F_parts = np.split(FF_scan, [1], 0);
      const V2_parts = np.split(V2_arr, [1], 0);
      const y_parts = np.split(y_safe_arr, [1], 0);
      const mask_parts = np.split(mask_arr, [1], 0);
      const A_parts = np.split(A_all, [1], 0);
      const b_parts = np.split(b_all, [1], 0);
      const C_parts = np.split(C_all, [1], 0);
      const eta_parts = np.split(eta_all, [1], 0);
      const J_parts = np.split(J_all, [1], 0);

      using F1 = F_parts[0];
      using V2_1 = V2_parts[0];
      using y1 = y_parts[0];
      using mask1 = mask_parts[0];

      using C0_first = np.reshape(C0, [1, stateSize, stateSize]);
      using x0_first = np.reshape(x0, [1, stateSize, 1]);

      using S1 = np.add(np.einsum('nij,njk,nlk->nil', F1, C0_first, F1), V2_1);          // [1,1,1]
      using C0Ft1 = np.einsum('ij,nkj->nik', C0, F1);                                     // [1,m,1]
      using K1_obs = np.divide(C0Ft1, S1);                                                // [1,m,1]
      using K1 = np.multiply(mask1, K1_obs);                                              // [1,m,1]

      using Fx0_1 = np.einsum('nij,njk->nik', F1, x0_first);                              // [1,1,1]
      using innov1 = np.subtract(y1, Fx0_1);                                              // [1,1,1]
      using Kinnov1 = np.multiply(K1, innov1);                                            // [1,m,1]
      const b1 = np.add(x0_first, Kinnov1);                                               // [1,m,1]

      using K1S1 = np.multiply(K1, S1);                                                   // [1,m,1]
      using K1S1K1t = np.einsum('nij,nkj->nik', K1S1, K1);                                // [1,m,m]
      const C1 = np.subtract(C0_first, K1S1K1t);                                           // [1,m,m]

      const A1 = np.zeros([1, stateSize, stateSize], { dtype });                         // [1,m,m]
      const eta1 = np.zeros([1, stateSize, 1], { dtype });                                // [1,m,1]
      const J1 = np.zeros([1, stateSize, stateSize], { dtype });                          // [1,m,m]

      // Replace timestep 0 with exact first element; keep k>=2 elements from Lemma 1
      const A_arr = np.concatenate([A1, A_parts[1]], 0);
      const b_arr = np.concatenate([b1, b_parts[1]], 0);
      const C_arr = np.concatenate([C1, C_parts[1]], 0);
      const eta_arr = np.concatenate([eta1, eta_parts[1]], 0);
      const J_arr = np.concatenate([J1, J_parts[1]], 0);

      F_parts[1].dispose();
      V2_parts[1].dispose();
      y_parts[1].dispose();
      mask_parts[1].dispose();
      A_parts[0].dispose();
      A_parts[1].dispose();
      b_parts[0].dispose();
      b_parts[1].dispose();
      C_parts[0].dispose();
      C_parts[1].dispose();
      eta_parts[0].dispose();
      eta_parts[1].dispose();
      J_parts[0].dispose();
      J_parts[1].dispose();

      const composeForward = (a: ForwardElem, b_elem: ForwardElem): ForwardElem => {
        // Compose later (j=b_elem) after earlier (i=a)
        // M = (I + C_i J_j)^-1
        using I1 = np.reshape(np.eye(stateSize, undefined, { dtype }), [1, stateSize, stateSize]);
        using inv_eps = np.array(dtype === DType.Float32 ? 1e-6 : 1e-12, { dtype });
        using regI = np.multiply(np.reshape(inv_eps, [1, 1, 1]), I1);
        using CiJj = np.einsum('nij,njk->nik', a.C, b_elem.J);
        using X_reg = np.add(np.add(I1, CiJj), regI);
        using M = np.linalg.inv(X_reg);

        // A_ij = A_j M A_i
        using AjM = np.einsum('nij,njk->nik', b_elem.A, M);
        const A_comp = np.einsum('nij,njk->nik', AjM, a.A);

        // b_ij = A_j M (b_i + C_i eta_j) + b_j
        using CiEtaj = np.einsum('nij,njk->nik', a.C, b_elem.eta);
        using bi_plus = np.add(a.b, CiEtaj);
        using AjM_b = np.einsum('nij,njk->nik', AjM, bi_plus);
        const b_comp = np.add(AjM_b, b_elem.b);

        // C_ij = A_j M C_i A_j' + C_j
        using AjMCi = np.einsum('nij,njk->nik', AjM, a.C);
        using C_tmp = np.einsum('nij,njk->nik', AjMCi, np.einsum('nij->nji', b_elem.A));
        const C_comp = np.add(C_tmp, b_elem.C);

        // eta_ij = A_i' (I + J_j C_i)^-1 (eta_j - J_j b_i) + eta_i
        // Derive (I + J_j C_i)^-1 via push-through identity:
        // N = I - J_j (I + C_i J_j)^-1 C_i = I - J_j M C_i
        using JjCi = np.einsum('nij,njk->nik', b_elem.J, a.C);
        JjCi.dispose();
        using MCi = np.einsum('nij,njk->nik', M, a.C);
        using JjMCi = np.einsum('nij,njk->nik', b_elem.J, MCi);
        using N = np.subtract(I1, JjMCi);
        using Jjbi = np.einsum('nij,njk->nik', b_elem.J, a.b);
        using eta_diff = np.subtract(b_elem.eta, Jjbi);
        using N_eta = np.einsum('nij,njk->nik', N, eta_diff);
        using AtNeta = np.einsum('nji,njk->nik', a.A, N_eta);
        const eta_comp = np.add(AtNeta, a.eta);

        // J_ij = A_i' (I + J_j C_i)^-1 J_j A_i + J_i
        using NJ = np.einsum('nij,njk->nik', N, b_elem.J);
        using NJAi = np.einsum('nij,njk->nik', NJ, a.A);
        using AtNJAi = np.einsum('nji,njk->nik', a.A, NJAi);
        const J_comp = np.add(AtNJAi, a.J);

        return { A: A_comp, b: b_comp, C: C_comp, eta: eta_comp, J: J_comp };
      };

      const scanned = lax.associativeScan(
        composeForward,
        { A: A_arr, b: b_arr, C: C_arr, eta: eta_arr, J: J_arr },
      ) as ForwardElem;

      using x0_exp = np.tile(np.reshape(x0, [1, stateSize, 1]), [n, 1, 1]);
      using Ax0 = np.einsum('nij,njk->nik', scanned.A, x0_exp);
      const x_filt = np.add(Ax0, scanned.b);             // [n, m, 1]

      using C0_exp = np.tile(np.reshape(C0, [1, stateSize, stateSize]), [n, 1, 1]);
      using AC0At = np.einsum('nij,njk,nlk->nil', scanned.A, C0_exp, scanned.A);
      using C_filt_raw = np.add(AC0At, scanned.C);

      using C_filt_t = np.einsum('nij->nji', C_filt_raw);
      using C_filt_sum = np.add(C_filt_raw, C_filt_t);
      const C_filt = np.multiply(np.array(0.5, { dtype }), C_filt_sum); // [n,m,m]

      scanned.A.dispose();
      scanned.b.dispose();
      scanned.C.dispose();
      scanned.eta.dispose();
      scanned.J.dispose();

      A1.dispose();
      b1.dispose();
      C1.dispose();
      eta1.dispose();
      J1.dispose();
      A_arr.dispose();
      b_arr.dispose();
      C_arr.dispose();
      eta_arr.dispose();
      J_arr.dispose();

      // ─── Recover sequential-convention diagnostics from filtered results ───
      //
      // The assocScan forward pass produces the standard Kalman filtered state
      // x_{t|t} and C_{t|t}. But the MATLAB DLM sequential convention carries
      // a hybrid predict+update state:
      //
      //   carry_{t+1} = G · x_{t|t}           (state prediction for next step)
      //   carry_C_{t+1} = G · C_{t|t} · G' + W  (covariance prediction)
      //
      // The sequential path stores x_pred[t] = carry entering step t. So:
      //   x_pred[0] = x0
      //   x_pred[t] = G · x_filt[t-1]         for t >= 1
      //   C_pred[0] = C0
      //   C_pred[t] = G · C_filt[t-1] · G' + W  for t >= 1
      // Prediction recovery uses the arriving transition:
      //   x_pred[t] = G_arriving[t] · x_filt[t-1]   for t >= 1
      //   C_pred[t] = G_arriving[t] · C[t-1] · G_arriving[t]' + W_arriving[t]
      //
      // G_arriving[t] = G_scan[t-1] (departing from t-1), so we need
      // G_scan[0:n-1] and W_scan[0:n-1].
      const gParts = np.split(G_scan, [n - 1], 0);
      using G_arr_head = gParts[0];   // [n-1, m, m]  G_scan[0..n-2] = arriving for steps 1..n-1
      gParts[1].dispose();
      const wParts = np.split(W_scan, [n - 1], 0);
      using W_arr_head = wParts[0];   // [n-1, m, m]  W_scan[0..n-2] = arriving for steps 1..n-1
      wParts[1].dispose();

      const xFiltParts = np.split(x_filt, [n - 1], 0);
      xFiltParts[1].dispose();
      using x_filt_head = xFiltParts[0];  // [n-1, m, 1]
      // Apply G_arriving[t] to get predicted state  [n-1, m, 1]
      using x_filt_pred = np.einsum('nij,njk->nik', G_arr_head, x_filt_head);
      using x0_1 = np.reshape(x0, [1, stateSize, 1]);
      // jax-js-lint: allow-non-using — stored in fwd.x_pred, disposed by caller
      const x_pred_arr = np.concatenate([x0_1, x_filt_pred], 0);  // [n, m, 1]

      const cFiltParts = np.split(C_filt, [n - 1], 0);
      cFiltParts[1].dispose();
      using C_filt_head = cFiltParts[0];  // [n-1, m, m]
      // Apply G_arriving[t]·C·G_arriving[t]' + W_arriving[t]  [n-1, m, m]
      using GCGt = np.einsum('nij,njk,nlk->nil', G_arr_head, C_filt_head, G_arr_head);
      using C_filt_pred = np.add(GCGt, W_arr_head);
      using C0_1 = np.reshape(C0, [1, stateSize, stateSize]);
      // jax-js-lint: allow-non-using — stored in fwd.C_pred, disposed by caller
      const C_pred_arr = np.concatenate([C0_1, C_filt_pred], 0);  // [n, m, m]

      // v[t] = mask · (y - F·x_pred)  [n,1,1]
      using Fx_pred = np.einsum('nij,njk->nik', FF_scan, x_pred_arr); // [n,1,1]
      using v_raw = np.subtract(y_safe_arr, Fx_pred);
      // jax-js-lint: allow-non-using — stored in fwd.v, disposed by caller
      const v_arr = np.multiply(mask_arr, v_raw);         // [n,1,1]

      // Cp[t] = F·C_pred·F' + V²  [n,1,1]
      using FCFt = np.einsum('nij,njk,nlk->nil', FF_scan, C_pred_arr, FF_scan);
      // jax-js-lint: allow-non-using — stored in fwd.Cp, disposed by caller
      const Cp_arr = np.add(FCFt, V2_arr);                // [n,1,1]

      // K[t] = mask · G[t]·C_pred[t]·F[t]' / Cp[t]  [n,m,1]  (MATLAB convention for backward pass)
      using GCFt = np.einsum('nij,njk,nlk->nil', G_scan, C_pred_arr, FF_scan); // [n,m,1]
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
        // S_k = G[k] · C_filt,k · G[k]' + W[k]  [n, m, m]
        using GCGt = np.einsum('nij,njk,nlk->nil', G_scan, C_filt, G_scan);
        using S_mat = np.add(GCGt, W_scan);

        // Batched matrix inverse S^{-1}  [n, m, m]
        using S_inv = np.linalg.inv(S_mat);

        // E_k = C_filt,k · G[k]' · S_k^{-1}  [n, m, m]
        using Gt_bwd = np.einsum('nij->nji', G_scan);  // batched transpose
        using CGt = np.einsum('nij,njk->nik', C_filt, Gt_bwd);
        using E_raw = np.einsum('nij,njk->nik', CGt, S_inv);

        // Terminal masking: E[n-1] = 0
        using term_mask = np.array(
          Array.from({ length: n }, (_, t) => [[t < n - 1 ? 1.0 : 0.0]]),
          { dtype }
        );  // [n, 1, 1]
        // jax-js-lint: allow-non-using — E_all disposed after scan
        const E_all = np.multiply(E_raw, term_mask);  // [n, m, m]

        // ImEG = I - E_k · G[k]  [n, m, m]
        using EG = np.einsum('nij,njk->nik', E_all, G_scan);
        using I_eye = np.eye(stateSize, undefined, { dtype });
        using I_exp = np.tile(np.reshape(I_eye, [1, stateSize, stateSize]), [n, 1, 1]);
        using ImEG = np.subtract(I_exp, EG);

        // g_k = (I - E_k·G[k]) · x̄_k  [n, m, 1]
        // jax-js-lint: allow-non-using — g_all disposed after scan
        const g_all = np.einsum('nij,njk->nik', ImEG, x_filt);

        // L_k (Joseph form — guaranteed PSD):
        //   L_k = (I - E_k·G[k]) · C_filt,k · (I - E_k·G[k])' + E_k · W[k] · E_k'
        using ImEG_C_ImEGt = np.einsum('nij,njk,nlk->nil', ImEG, C_filt, ImEG);
        using EWEt = np.einsum('nij,njk,nlk->nil', E_all, W_scan, E_all);
        using L_raw = np.add(ImEG_C_ImEGt, EWEt);
        // Symmetrize (f32 stabilization)
        using L_raw_t = np.einsum('nij->nji', L_raw);
        using L_sum = np.add(L_raw, L_raw_t);
        // jax-js-lint: allow-non-using — L_all disposed after scan
        const L_all = np.multiply(np.array(0.5, { dtype }), L_sum);

        const composeBackward = (a: BackwardElem, b_elem: BackwardElem): BackwardElem => {
          const A_comp = np.einsum('nij,njk->nik', b_elem.A, a.A);
          using Ab = np.einsum('nij,njk->nik', b_elem.A, a.b);
          const b_comp = np.add(Ab, b_elem.b);
          using ASAt = np.einsum('nij,njk,nlk->nil', b_elem.A, a.S, b_elem.A);
          const S_comp = np.add(ASAt, b_elem.S);
          return { A: A_comp, b: b_comp, S: S_comp };
        };

        // Suffix scan via reverse associativeScan (RTS backward compose)
        const smoothed = lax.associativeScan(
          composeBackward,
          { A: E_all, b: g_all, S: L_all },
          { reverse: true }
        ) as BackwardElem;

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
        { y: y_arr, V2: V2_arr, FF: FF_scan, Gt: G_scan, Wt: W_scan }
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
      using Gt_rev = np.flip(fwd.Gt, 0);
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
          Gt: Gt_rev,
          mask: mask_rev,
        }
      );
      tree.dispose(bwdCarry);

      x_smooth = np.flip(bwd.x_smooth, 0);  // [n, m, 1]
      C_smooth = np.flip(bwd.C_smooth, 0);  // [n, m, m]
      tree.dispose(bwd);
    }

    // ─── Observation-space diagnostics ───

    // NaN observation mask [n]: 1.0 where observed, 0.0 where gapped.
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

    // Dispose fwd.K, fwd.FF, fwd.Gt (seq only), fwd.mask (no longer needed after squeeze)
    fwd.K.dispose();
    fwd.FF.dispose();
    if (fwd.Gt) fwd.Gt.dispose();
    fwd.mask.dispose();

    // y_safe: replace NaN with 0 for numerically safe reductions
    using is_nan_y = np.isnan(y_1d);       // [n] bool
    using y_safe = np.where(is_nan_y, np.zerosLike(y_1d), y_1d);  // [n]

    // Residuals: naturally Nan at missing positions (y_1d has NaN there)
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
  const coreResult = await jit(core)(x0, C0, y_arr, V2_arr, FF_scan, G_scan, W_scan, r0, N0);

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
 * states beyond processStd.length.
 *
 * @param y - Observations (n×1 array)
 * @param opts - Model and runtime options (see {@link DlmFitOptions})
 * @returns Complete model fit with smoothed estimates and diagnostics
 */
export const dlmFit = async (
  y: ArrayLike<number>,
  opts: DlmFitOptions,
): Promise<DlmFitResult> => {
  const {
    obsStd: s, processStd: w,
    order, harmonics, seasonLength, fullSeasonal, arCoefficients, spline,
    X, algorithm, stabilization,
  } = opts;
  const dtype = parseDtype(opts.dtype);
  const forceAssocScan = algorithm === 'assoc' ? true : undefined;

  // Build DlmOptions for dlmGenSys
  const genSysOpts: DlmOptions = {
    order, harmonics, seasonLength, fullSeasonal, arCoefficients, spline,
  };

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
  const sys = dlmGenSys(genSysOpts);
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
  if (spline && (order ?? 1) === 1 && w.length >= 2) {
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
  // Build G_scan [n, m, m] and W_scan [n, m, m] for dlmSmo.
  //
  // Without timestamps: tile uniform G/W to [n,m,m] (standard DLM convention).
  // With timestamps: dlmGenSysTV computes per-step G(Δt_k), W(Δt_k) via
  //   closed-form continuous-time discretization. When covariates are present,
  //   each per-step matrix is extended with identity/zero blocks for β states.
  // ─────────────────────────────────────────────────────────────────────────
  const timestamps = opts.timestamps;
  let G_scan: np.Array;
  let W_scan: np.Array;
  if (timestamps) {
    const tv = dlmGenSysTV(genSysOpts, timestamps, w, spline);
    // tv.G and tv.W are [n, m_base, m_base] as JS arrays.
    // Extend for covariates if q > 0.
    let G_tv_data: number[][][];
    let W_tv_data: number[][][];
    if (q > 0) {
      G_tv_data = tv.G.map(Gk => [
        ...Gk.map(row => [...row, ...new Array(q).fill(0)]),
        ...Array.from({ length: q }, (_, k) =>
          [...new Array(m_base).fill(0), ...Array.from({ length: q }, (_, j) => j === k ? 1 : 0)]
        ),
      ]);
      W_tv_data = tv.W.map(Wk => [
        ...Wk.map(row => [...row, ...new Array(q).fill(0)]),
        ...Array.from({ length: q }, () => new Array(m).fill(0)),
      ]);
    } else {
      G_tv_data = tv.G;
      W_tv_data = tv.W;
    }
    G_scan = np.array(G_tv_data, { dtype });
    W_scan = np.array(W_tv_data, { dtype });
  } else {
    // Uniform timesteps: tile constant G/W to [n, m, m]
    G_scan = np.tile(np.reshape(G, [1, m, m]), [n, 1, 1]);
    W_scan = np.tile(np.reshape(W, [1, m, m]), [n, 1, 1]);
  }

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
  const ns = seasonLength ?? 12;
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

  // ─────────────────────────────────────────────────────────────────────────
  // Pass 1: Initial smoother to refine starting values
  // ─────────────────────────────────────────────────────────────────────────
  let x0_updated: number[][];
  let C0_scaled: number[][];
  { // Block scope — `using` auto-disposes all Pass 1 arrays at block end
    using out1 = await dlmSmo(yArr, F_base, V_std, x0_data, G_scan, W_scan, C0_data, m, dtype, FF_arr, forceAssocScan, stabilization);
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
  const out2 = await dlmSmo(yArr, F_base, V_std, x0_updated, G_scan, W_scan, C0_scaled, m, dtype, FF_arr, forceAssocScan, stabilization);

  FF_arr?.dispose();
  G_scan.dispose();
  W_scan.dispose();

  // ─────────────────────────────────────────────────────────────────────────
  // Convert np.Array results to TypedArrays via consumeData (read + dispose).
  // Zero-cost buffer wrapping: consumeData() returns [n,m,1] row-major data,
  // which after flattening the trailing-1 dimension is [n,m] — exactly the
  // layout StateMatrix expects. No transpose needed.
  // ─────────────────────────────────────────────────────────────────────────
  const toFA = async (a: np.Array) =>
    new FA(await a.consumeData() as ArrayLike<number>);
  const toNum = async (a: np.Array) =>
    (await a.consumeData() as ArrayLike<number>)[0];

  // State and covariance tensors — zero-copy wrapping
  const xf_raw = new FA(await out2.xf.consumeData() as ArrayLike<number>); // [n,m,1] → [n*m]
  const Cf_raw = new FA(await out2.Cf.consumeData() as ArrayLike<number>); // [n,m,m] → [n*m*m]
  const x_raw = new FA(await out2.x.consumeData() as ArrayLike<number>);   // [n,m,1] → [n*m]
  const C_raw = new FA(await out2.C.consumeData() as ArrayLike<number>);   // [n,m,m] → [n*m*m]

  const smoothed = new StateMatrix(x_raw, n, m);
  const filtered = new StateMatrix(xf_raw, n, m);
  const smoothedCov = new CovMatrix(C_raw, n, m);
  const filteredCov = new CovMatrix(Cf_raw, n, m);

  // smoothedStd[t, k] = sqrt(|C[t, k, k]|) — contiguous [n, m] buffer
  const stdData = new FA(n * m);
  for (let t = 0; t < n; t++) {
    for (let k = 0; k < m; k++) {
      stdData[t * m + k] = Math.sqrt(Math.abs(C_raw[t * m * m + k * m + k]));
    }
  }
  const smoothedStd = new StateMatrix(stdData, n, m);

  // Diagnostics
  const yhat = await toFA(out2.yhat);
  const ystd = await toFA(out2.ystd);
  const innovations = await toFA(out2.v);
  const innovationVar = await toFA(out2.Cp);
  const rawResiduals = await toFA(out2.resid0);
  const scaledResiduals = await toFA(out2.resid);
  const standardizedResiduals = await toFA(out2.resid2);

  // Scalar diagnostics
  const rss = await toNum(out2.ssy);
  const deviance = await toNum(out2.lik);
  const residualVariance = await toNum(out2.s2);
  const mse = await toNum(out2.mse);
  const mape = await toNum(out2.mape);
  const nobs = Math.round(await toNum(out2.nobs));  // count of non-NaN observations

  return {
    // State estimates (m = m_base + q; last q states are β coefficients)
    smoothed, filtered, smoothedCov, filteredCov, smoothedStd,
    // System matrices (plain arrays for easy serialization)
    G: G_data,
    F: sys.F,
    W: W_data,
    // Input data
    y: yArr, obsNoise: V_std,
    // Initial state (after Pass 1 refinement)
    initialState: x0_updated.map(row => row[0]),
    initialCov: C0_scaled,
    // Covariates matrix (stored as row vectors; empty array when X not provided)
    covariates: X ? Array.from({ length: n }, (_, t) => Array.from(X[t]) as number[]) : [],
    // Predictions and residuals
    yhat, ystd, rawResiduals, scaledResiduals, standardizedResiduals,
    // Diagnostics
    rss, innovations, innovationVar, residualVariance,
    nobs,
    deviance, mse, mape,
    // Shape
    n, m,
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
 * @param obsStd - Observation noise std dev (scalar, same as used in dlmFit)
 * @param h - Forecast horizon (number of steps ahead)
 * @param opts - Optional forecast options (dtype, covariates)
 * @returns Predicted state means, covariances, and observation predictions for steps 1…h
 */
export const dlmForecast = async (
  fit: DlmFitResult,
  obsStd: number,
  h: number,
  opts?: DlmForecastOptions,
): Promise<DlmForecastResult> => {
  const { G: G_data, F: F_data, W: W_data } = fit;
  const m = G_data.length;
  const q = fit.covariates && (fit.covariates as number[][])[0]?.length > 0
    ? (fit.covariates as number[][])[0].length
    : 0;
  const dtype = parseDtype(opts?.dtype);
  const X_forecast = opts?.X;
  const n = fit.n;
  const FA = getFloatArrayType(dtype);

  // ── Build constant np.Arrays for G and W (captured by jit core) ──────────
  using G_np = np.array(G_data, { dtype });
  using W_np = np.array(W_data, { dtype });

  // ── Initial state: last smoothed timestep ─────────────────────────────────
  const x0_data: number[][] = Array.from({ length: m }, (_, i) => [fit.smoothed.get(n - 1, i)]);
  const C0_data: number[][] = Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => fit.smoothedCov.get(n - 1, i, j))
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
  using s2_arr = np.array([[obsStd * obsStd]], { dtype });

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

  // ── Extract results — zero-copy StateMatrix/CovMatrix wrapping ────────────
  const x_raw    = new FA(await out.x.consumeData()    as ArrayLike<number>);  // [h,m,1] → [h*m]
  const C_raw    = new FA(await out.C.consumeData()    as ArrayLike<number>);  // [h,m,m] → [h*m*m]
  const yhat_raw = await out.yhat.consumeData() as ArrayLike<number>;  // [h,1,1]
  const ystd_raw = await out.ystd.consumeData() as ArrayLike<number>;  // [h,1,1]

  const yhat_out = new FA(h);
  const ystd_out = new FA(h);
  for (let k = 0; k < h; k++) {
    yhat_out[k] = yhat_raw[k] as number;
    ystd_out[k] = ystd_raw[k] as number;
  }

  const predicted = new StateMatrix(x_raw, h, m);
  const predictedCov = new CovMatrix(C_raw, h, m);

  // predictedStd[k, i] = sqrt(|C[k, i, i]|)
  const stdData = new FA(h * m);
  for (let k = 0; k < h; k++) {
    for (let i = 0; i < m; i++) {
      stdData[k * m + i] = Math.sqrt(Math.abs(C_raw[k * m * m + i * m + i]));
    }
  }
  const predictedStd = new StateMatrix(stdData, h, m);

  return { yhat: yhat_out, ystd: ystd_out, predicted, predictedCov, predictedStd, h, m };
};

/**
 * Convert a JS-idiomatic DlmFitResult to MATLAB DLM layout and names.
 *
 * This function serves two purposes:
 * 1. **Name restoration**: `smoothed` → `x`, `deviance` → `lik`, etc.
 * 2. **Axis transposition**: time-major [n, m] → state-major `x[state][time]`
 *
 * Use this for MATLAB DLM migration and for comparing against Octave reference
 * output that uses MATLAB naming conventions.
 *
 * @param result - JS-idiomatic DlmFitResult from dlmFit
 * @returns MATLAB DLM-compatible result with transposed layout and old names
 */
export const toMatlab = (result: DlmFitResult): DlmFitResultMatlab => {
  const { n, m } = result;

  // Transpose smoothed [n, m] → x[m][n]
  const x: FloatArray[] = Array.from({ length: m }, (_, k) => result.smoothed.series(k));
  const xf: FloatArray[] = Array.from({ length: m }, (_, k) => result.filtered.series(k));

  // Transpose covariance [n, m, m] → C[m][m][n]
  const C: FloatArray[][] = Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => result.smoothedCov.series(i, j))
  );
  const Cf: FloatArray[][] = Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => result.filteredCov.series(i, j))
  );

  // xstd [n][m] — same layout as original MATLAB (time-major)
  const xstd: FloatArray[] = Array.from({ length: n }, (_, t) => {
    const Ctor = result.smoothedStd.data.constructor as typeof Float32Array | typeof Float64Array;
    const arr = new Ctor(m);
    for (let k = 0; k < m; k++) arr[k] = result.smoothedStd.get(t, k);
    return arr;
  });

  return {
    x, xf, C, Cf, xstd,
    v: result.innovations,
    Cp: result.innovationVar,
    resid0: result.rawResiduals,
    resid: result.scaledResiduals,
    resid2: result.standardizedResiduals,
    lik: result.deviance,
    s2: result.residualVariance,
    ssy: result.rss,
    G: result.G,
    F: result.F,
    W: result.W,
    V: result.obsNoise,
    x0: result.initialState,
    C0: result.initialCov,
    XX: result.covariates,
    y: result.y,
    yhat: result.yhat,
    ystd: result.ystd,
    mse: result.mse,
    mape: result.mape,
    nobs: result.nobs,
    n, m,
    class: 'dlmfit',
  };
};
