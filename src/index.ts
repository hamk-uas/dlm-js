import { DType, numpy as np, lax, jit, tree, defaultDevice } from "@hamk-uas/jax-js-nonconsuming";
import type { DlmSmoResult, FloatArray } from "./types";
import {
  getFloatArrayType, parseDtype,
  StateMatrix, CovMatrix,
  checkUnknownKeys, DLM_FIT_KEYS, DLM_STABILIZATION_KEYS, DLM_FORECAST_KEYS,
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
  DlmDtype, DlmAlgorithm, DlmLossFn, DlmParamMeta, DlmStabilization,
  FloatArray,
} from "./types";
export { StateMatrix, CovMatrix } from "./types";
export type { DlmOptions, DlmSystem, DlmSystemTV } from "./dlmgensys";
export { dlmGenSys, dlmGenSysTV, findArInds } from "./dlmgensys";
export { dlmMLE, toMatlabMle } from "./mle";
export type { DlmMleResult, DlmMleResultMatlab } from "./mle";
export type { DlmMleOptions } from "./types";
export { dlmPrior } from "./priors";
export type { InverseGammaPrior, NormalPrior, DlmPriorSpec } from "./priors";

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
 * @param y_arr - Observations tensor [n, p, 1] (p=1 for univariate)
 * @param V2_arr - Observation noise covariance [n, p, p] (diagonal for independent noise)
 * @param x0_data - Initial state mean (m×1 as nested array)
 * @param G_scan - Per-step transition matrices [n, m, m]
 * @param W_scan - Per-step noise covariances [n, m, m]
 * @param C0_data - Initial state covariance (m×m as nested array)
 * @param stateSize - State dimension m (extended: m_base + q for covariates)
 * @param obsSize - Observation dimension p (1 for univariate)
 * @param dtype - Computation precision
 * @param FF_scan - Observation matrix [n, p, m] (time-varying or tiled static)
 * @returns Smoothed and filtered state estimates with diagnostics
 * @internal
 */
const dlmSmo = async (
  y_arr: np.Array,     // [n, p, 1] observations
  V2_arr: np.Array,    // [n, p, p] observation noise covariance
  x0_data: number[][],
  G_scan: np.Array,   // [n, m, m] per-step transition matrix
  W_scan: np.Array,   // [n, m, m] per-step noise covariance
  C0_data: number[][],
  stateSize: number,
  obsSize: number,
  dtype: DType = DType.Float64,
  FF_scan: np.Array,  // [n, p, m] observation matrix
  forceAssocScan?: boolean,
  stabilization?: DlmStabilization,
  forceSqrtAssocScan?: boolean,
  forceUdScan?: boolean,
): Promise<DlmSmoResult & Disposable> => {
  const n = y_arr.shape[0];
  const p = obsSize;

  // ─────────────────────────────────────────────────────────────────────────
  // Branch selection: four execution paths based on device + dtype
  //
  //   wasm/cpu + Float64  →  sequential scan + triu+triu' symmetrization (default)
  //   cpu      + Float32  →  sequential scan + Joseph form + triu/avg sym + cEps
  //   webgpu   + Float32  →  associativeScan forward + Joseph form/sym/cEps
  //   ud       + any      →  sequential scan + UD (Bierman) factorization in forward
  //
  // Float64 default: triu(C)+triu(C,1)' symmetrization after each filter and
  // smoother step (matches MATLAB dlmsmo.m line 77). Reduces max relative error
  // vs Octave reference from ~2e-9 to ~4e-12 — ~500× improvement at negligible
  // cost (two np.triu calls per step). Disable with stabilization:{cTriuSym:false}.
  //
  // Float32 default: Joseph form (L·C·L' + K·V²·K' + W), (C+C')/2 symmetrize,
  // and C += 1e-6·I (cEps) — prevents covariance from going non-PD for m > 2.
  //
  // The assoc path (webgpu) reformulates the forward Kalman filter
  // as an associative prefix scan per Särkkä & García-Fernández (2020),
  // reducing sequential depth from O(n) to O(log n) on parallel hardware.
  //
  // The UD path (Bierman 1977) factors C = U·D·U' (unit upper-triangular U,
  // positive diagonal D). The Bierman measurement update avoids squaring
  // condition numbers, improving Float32 precision.
  // ─────────────────────────────────────────────────────────────────────────
  const device = defaultDevice();
  const f32 = dtype === DType.Float32;
  const useUdScan = forceUdScan ?? false;
  const useSqrtAssocScan = !useUdScan && (forceSqrtAssocScan ?? false);
  const useAssocScan = !useUdScan && !useSqrtAssocScan && (forceAssocScan ?? (f32 && device === 'webgpu'));

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
  using _stab_ones       = np.ones([stateSize, stateSize], { dtype });             // [m,m]
  using stab_off_I       = np.subtract(_stab_ones, stab_I_eye);
  using stab_nLeak_fact  = np.array(1.0 - 1e-5, { dtype });                       // scalar
  using _stab_eps7       = np.array(1e-7, { dtype });                              // scalar
  using stab_cDiag_eps_I = np.multiply(_stab_eps7, stab_I_eye);                   // [m,m]
  using _stab_eps6       = np.array(1e-6, { dtype });                              // scalar
  using stab_cEps_I      = np.multiply(_stab_eps6, stab_I_eye);                   // [m,m]
  // Shared scalar constants used inside jit(core) — hoisted here to avoid
  // inline np.array() temporaries that the leak checker flags (rc=1, never disposed).
  using half             = np.array(0.5, { dtype });                               // scalar
  using zeros_mm         = np.zeros([stateSize, stateSize], { dtype });            // [m,m]
  using zeros_m          = np.zeros([stateSize], { dtype });                       // [m]

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
  // Step functions receive FF_t ([1, m]), G_t ([m, m]), and W_t ([m, m])
  // from the scan pytree. All three can be time-varying.
  // ─────────────────────────────────────────────────────────────────────────

  // NaN mask constants: [1,1] scalar mask broadcasts to [p,1] and [m,p] via multiply.
  // All-or-nothing per timestep: if ANY yi component is NaN, entire step is skipped.
  using const_one_11 = np.array([[1.0]], { dtype });  // [1,1] observed
  using zero_11 = np.array([[0.0]], { dtype });        // [1,1] NaN

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

    // NaN masking: mask = 1.0 if observation is valid, 0.0 if NaN.
    // p=1: isnan(yi) directly.  p>1: sum propagates NaN → all-or-nothing.
    // mask_t is always [1,1] scalar, broadcasts to [p,1] and [m,p].
    let is_nan_11: np.Array;
    if (p === 1) {
      is_nan_11 = np.isnan(yi);                                     // [1,1] bool
    } else {
      using yi_sum = np.reshape(np.sum(yi), [1, 1]);                // [1,1]
      is_nan_11 = np.isnan(yi_sum);                                 // [1,1] bool
    }
    using _is_nan = is_nan_11;
    const mask_t = np.where(_is_nan, zero_11, const_one_11);       // [1,1]: 0 or 1
    using y_safe = np.where(_is_nan, np.zerosLike(yi), yi);         // [p,1]: 0 if NaN

    // Innovation: v = mask * (y_safe - FF·x)  [p,1]  (0 when NaN)
    using FFxi = np.matmul(FFi, xi);                                // [p,m]·[m,1] → [p,1]
    using v_raw = np.subtract(y_safe, FFxi);
    const v = np.multiply(mask_t, v_raw);                           // [1,1]×[p,1] broadcast

    // Innovation covariance: Cp = FF·C·FF' + V²  [p,p]
    const Cp = np.add(
      np.einsum('ij,jk,lk->il', FFi, Ci, FFi),                    // [p,p]
      V2i
    );

    // Kalman gain: K = mask * (G·C·FF' · Cp⁻¹)  [m,p]  (0 when NaN)
    using GCFFt = np.einsum('ij,jk,lk->il', G_t, Ci, FFi);        // [m,p]
    if (p === 1) {
      using K_raw = np.divide(GCFFt, Cp);                          // scalar division
      var K = np.multiply(mask_t, K_raw);                           // [1,1]×[m,1] broadcast
    } else {
      using CpInv = np.linalg.inv(Cp);                             // [p,p]
      using K_raw = np.matmul(GCFFt, CpInv);                       // [m,p]
      var K = np.multiply(mask_t, K_raw);                           // [1,1]×[m,p] broadcast
    }

    // L = G - K·FF  [m,m]  (= G when NaN, since K=0)
    using L = np.subtract(G_t, np.matmul(K, FFi));                  // [m,p]·[p,m] → [m,m]

    // Next state prediction: x_next = G·x + K·v  [m,1]
    // When NaN: x_next = G·x (no measurement update)
    const x_next = np.add(
      np.matmul(G_t, xi),
      np.matmul(K, v)                                               // [m,p]·[p,1] → [m,1]
    );

    // Next covariance: C_next depends on dtype + stabilization flags.
    //
    // Float64 default (matches MATLAB DLM reference formula):
    //   C_next = G·C·L' + W            (+ triu+triu' sym if cTriuSym is set)
    //
    // Float32 (Joseph form — numerically stable):
    //   C_next = L·C·L' + K·V²·K' + W  (+ sym)
    let C_next: np.Array;
    {
      // jax-js-lint: allow-non-using — sym branch takes ownership below
      let C_fwd_raw: np.Array;
      if (f32) {
        // Joseph form: L·C·L' + K·V²·K' + W
        using LCLt = np.einsum('ij,jk,lk->il', L, Ci, L);
        let KV2Kt: np.Array;
        if (p === 1) {
          KV2Kt = np.multiply(V2i, np.matmul(K, np.transpose(K)));     // scalar V² × KK'
        } else {
          KV2Kt = np.einsum('ij,jk,lk->il', K, V2i, K);               // [m,p]·[p,p]·[p,m]→[m,m]
        }
        using _kv2kt = KV2Kt;
        using sum1 = np.add(LCLt, KV2Kt);
        C_fwd_raw = np.add(sum1, W_t);
      } else {
        // Standard form (matches MATLAB dlmsmo.m): G·C·L' + W
        C_fwd_raw = np.add(np.einsum('ij,jk,lk->il', G_t, Ci, L), W_t);
      }

      // Apply symmetrization for f32 (always) or f64+cTriuSym.
      if (f32 || stabCTriuSym) {
        if (stabCTriuSym) {
          using C_upper = np.triu(C_fwd_raw);
          using C_sup   = np.triu(C_fwd_raw, 1);
          using C_sup_t = np.transpose(C_sup);
          C_next = np.add(C_upper, C_sup_t);
        } else {
          using Ct      = np.transpose(C_fwd_raw);
          using sumBoth = np.add(C_fwd_raw, Ct);
          C_next = np.multiply(half, sumBoth);
        }
        C_fwd_raw.dispose();
      } else {
        C_next = C_fwd_raw;
      }
    }
    
    return [
      { x: x_next, C: C_next },
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

    // FF'·Cp⁻¹  [m,p]: p=1 uses scalar division, p>1 uses matrix inverse
    using FFt = np.transpose(FFi);
    if (p === 1) {
      var FtCpInv = np.divide(FFt, Cpi);                   // [m,1] / [1,1]
    } else {
      using CpiInv = np.linalg.inv(Cpi);                   // [p,p]
      var FtCpInv = np.matmul(FFt, CpiInv);                // [m,p]·[p,p] → [m,p]
    }
    using _FtCpInv = FtCpInv;

    // r_new = F'·Cp⁻¹·v + L'·r  [m,1]
    // vi is already 0 at NaN positions (zeroed in forwardStep), so
    // FtCpInv·vi contributes 0 automatically at gapped timesteps.
    const r_new = np.add(
      np.matmul(FtCpInv, vi),                              // [m,p]·[p,1] → [m,1]
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
        const N_stab = np.multiply(half, Nsum);
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
        using _N_d_zeros = np.zerosLike(N_d);
        using N_d_c = np.maximum(N_d, _N_d_zeros);
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
          C_sym = np.multiply(half, sumC);
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
        C_smooth = np.add(C_raw, zeros_mm);
      }
    }

    return [{ r: r_new, N: N_new }, { x_smooth, C_smooth }];
  };

  // ─────────────────────────────────────────────────────────────────────────
  // UD (Bierman/Thornton) Forward Filter
  //
  // Factors covariance C = U·D·U' where U is unit lower-triangular and D is
  // a positive diagonal vector.  The Bierman measurement update processes
  // observation components via an unrolled column loop (j = m-1..0), avoiding
  // the O(m³) matrix products that square the condition number in the Joseph
  // form.  D stays positive by construction (ratio of positive alphas).
  //
  // Time update: Thornton modified weighted Gram-Schmidt (MWGS) propagates
  // the UD factors directly — no Cholesky roundtrip.  Given Bierman-updated
  // U_filt, D_filt and transition G, noise W, it factors:
  //   C_{t+1|t} = G·U_filt·D_filt·(G·U_filt)' + W
  // by orthogonalizing the augmented [G·U_filt, L_W] matrix with weights
  // [D_filt, D_W].  No epsilon regularization needed since factors stay
  // accurate.  C_true carry field is eliminated.
  //
  // Reference: Bierman (1977), "Factorization Methods for Discrete Sequential
  // Estimation", Ch. 6; Grewal & Andrews (2001), Algorithm 6.1.
  // ─────────────────────────────────────────────────────────────────────────

  // Precomputed one-hot vectors and column masks for the Bierman inner loop.
  // Created unconditionally (cheap) to avoid conditional `using` complexity.
  const ud_oneHot: np.Array[] = [];        // [m] one-hot vectors: e_j
  const ud_oneHotCol: np.Array[] = [];     // [m,1] column one-hot: e_j reshaped
  const ud_oneHotRow: np.Array[] = [];     // [1,m] row one-hot: e_j reshaped
  const ud_gtMask: np.Array[] = [];        // [m] masks: gtMask[j][i] = (i > j ? 1 : 0)
  if (useUdScan) {
    for (let j = 0; j < stateSize; j++) {
      const hot = new Array(stateSize).fill(0);
      hot[j] = 1;
      ud_oneHot.push(np.array(hot, { dtype }));
      ud_oneHotCol.push(np.reshape(ud_oneHot[j], [stateSize, 1]));
      ud_oneHotRow.push(np.reshape(ud_oneHot[j], [1, stateSize]));
      const gt = new Array(stateSize).fill(0);
      for (let i = j + 1; i < stateSize; i++) gt[i] = 1;
      ud_gtMask.push(np.array(gt, { dtype }));
    }
  }

  type UDForwardCarry = { x: np.Array; U: np.Array; D: np.Array };

  const forwardStepUD = (
    carry: UDForwardCarry,
    inp: ForwardX
  ): [UDForwardCarry, ForwardY] => {
    const { x: x_pred, U: Ui, D: Di } = carry;
    const { y: yi, V2: V2i, FF: FFi, Gt: G_t, Wt: W_t } = inp;
    const m = stateSize;

    // ── NaN masking (identical to standard forwardStep) ──
    using is_nan_11 = np.isnan(yi);                                 // [1,1] bool
    const mask_t = np.where(is_nan_11, zero_11, const_one_11);     // [1,1]: 0 or 1
    using y_safe = np.where(is_nan_11, np.zerosLike(yi), yi);       // [1,1]: 0 if NaN

    // ── Reconstruct C_pred = U·diag(D)·U' for backward smoother output ──
    // With Thornton time update, U,D are the exact factors of C_pred (no ε).
    using D_diag = np.multiply(np.reshape(Di, [1, m]), stab_I_eye); // [m,m]
    using UD_mat = np.matmul(Ui, D_diag);                           // [m,m]
    using C_pred_raw = np.matmul(UD_mat, np.transpose(Ui));         // [m,m]
    // Symmetrize: (C + C') / 2  (floating-point rounding may introduce asymmetry)
    using C_pred_t = np.transpose(C_pred_raw);
    using C_pred_sum = np.add(C_pred_raw, C_pred_t);
    const C_pred = np.multiply(half, C_pred_sum);                    // [m,m]

    // ── Clone carry's U,D as Bierman working arrays ──
    // Clone needed: accumulator-swap in the Bierman loop disposes old values.
    // jax-js-lint: allow-non-using — Bierman loop mutates U_bar/D_bar below
    let U_bar = np.add(Ui, zeros_mm);                               // [m,m] clone
    let D_bar = np.add(Di, zeros_m);                                // [m] clone

    // ── Innovation (using predicted state from carry directly) ──
    using Fx_pred = np.matmul(FFi, x_pred);                         // [1,1]
    using v_raw = np.subtract(y_safe, Fx_pred);
    const v = np.multiply(mask_t, v_raw);                            // [1,1]

    // ── Bierman measurement update (p=1 scalar observation) ──
    // U_bar is unit LOWER-triangular: C_pred = U·D·U'.
    // f = U'·H' (U' is upper-tri × column vec).
    // Column loop runs in REVERSE order (j = m-1 down to 0) because for
    // lower-tri U the free entries in column j are at rows i > j.  Reverse
    // processing ensures g[j] is still its original value when used in the
    // α update (inner loops only modify g at positions i > current_j, and
    // future columns have smaller j).
    // When NaN, f_bar = 0 → entire Bierman loop becomes a no-op
    // (all alphas stay at R, D/U unchanged, K = 0).
    using f_raw = np.matmul(np.transpose(U_bar), np.transpose(FFi));// [m,1]
    using f_vec_raw = np.reshape(f_raw, [m]);                       // [m]
    using f_vec = np.multiply(np.reshape(mask_t, []), f_vec_raw);   // 0 when NaN

    // g = D_bar ⊙ f (element-wise product)
    // jax-js-lint: allow-non-using — g_vec mutated in column loop below
    let g_vec = np.multiply(D_bar, f_vec);                           // [m]

    // R = observation variance (scalar)
    using R_scalar = np.reshape(V2i, []);                            // scalar

    // Process column m-1 first (no inner-loop entries: no rows i > m-1)
    // α₀ = R + f_{m-1}·g_{m-1}
    using f_last = np.sum(np.multiply(f_vec, ud_oneHot[m - 1]));    // scalar
    using g_last = np.sum(np.multiply(g_vec, ud_oneHot[m - 1]));    // scalar
    // jax-js-lint: allow-non-using — alpha mutated in column loop below
    let alpha = np.add(R_scalar, np.multiply(f_last, g_last));       // scalar

    // D_new[m-1] = D_bar[m-1] · R / α
    {
      using D_last = np.sum(np.multiply(D_bar, ud_oneHot[m - 1]));  // scalar
      using D_last_new = np.multiply(D_last, np.divide(R_scalar, alpha));
      using delta_last = np.subtract(D_last_new, D_last);
      // jax-js-lint: allow-non-using — accumulator-swap: D_bar updated
      const D_new = np.add(D_bar, np.multiply(delta_last, ud_oneHot[m - 1]));
      D_bar.dispose();
      D_bar = D_new;
    }

    // Column loop j = m-2 down to 0: update D[j], U[:,j], g
    for (let j = m - 2; j >= 0; j--) {
      // Extract scalar values for column j
      using f_j = np.sum(np.multiply(f_vec, ud_oneHot[j]));         // scalar
      using g_j = np.sum(np.multiply(g_vec, ud_oneHot[j]));         // scalar

      // α_new = α_old + f_j · g_j
      // jax-js-lint: allow-non-using — accumulator-swap for alpha
      const alpha_old = alpha;
      alpha = np.add(alpha_old, np.multiply(f_j, g_j));

      // λ_j = -f_j / α_old
      using lambda_j = np.divide(np.negative(f_j), alpha_old);      // scalar
      alpha_old.dispose();

      // D_new[j] = D_bar[j] · α_old / α_new  (= D_bar[j] · (α_new - f_j·g_j) / α_new)
      {
        using D_j = np.sum(np.multiply(D_bar, ud_oneHot[j]));
        using alpha_minus = np.subtract(alpha, np.multiply(f_j, g_j));
        using D_j_new = np.multiply(D_j, np.divide(alpha_minus, alpha));
        using delta_j = np.subtract(D_j_new, D_j);
        // jax-js-lint: allow-non-using — accumulator-swap: D_bar updated
        const D_new = np.add(D_bar, np.multiply(delta_j, ud_oneHot[j]));
        D_bar.dispose();
        D_bar = D_new;
      }

      // Extract old column j of U_bar: [m,1]
      using u_col_old = np.matmul(U_bar, ud_oneHotCol[j]);          // [m,1]

      // U_bar[:,j] += λ_j · g   (only rows i > j, masked by gtMask)
      using delta_u = np.multiply(
        lambda_j,
        np.multiply(np.reshape(g_vec, [m, 1]), np.reshape(ud_gtMask[j], [m, 1]))
      );                                                             // [m,1]
      using rank1_u = np.matmul(delta_u, ud_oneHotRow[j]);          // [m,m]
      // jax-js-lint: allow-non-using — accumulator-swap: U_bar updated
      const U_new = np.add(U_bar, rank1_u);
      U_bar.dispose();
      U_bar = U_new;

      // g[i] += U_old[i,j] · g_j   (only rows i > j, masked by gtMask)
      using g_update = np.multiply(
        g_j,
        np.multiply(np.reshape(u_col_old, [m]), ud_gtMask[j])
      );                                                             // [m]
      // jax-js-lint: allow-non-using — accumulator-swap: g_vec updated
      const g_new = np.add(g_vec, g_update);
      g_vec.dispose();
      g_vec = g_new;
    }

    // ── Gains and innovation covariance from Bierman loop ──
    // alpha_final = V² + F·U·D·U'·F' = Cp (innovation covariance)
    const Cp = np.reshape(alpha, [1, 1]);                            // [1,1]

    // Bierman gain: K_bierman = g_final / alpha = C_pred·F'/Cp  [m,1]
    // With Thornton maintaining exact U,D (no ε), this is exact.
    using K_bierman = np.reshape(
      np.divide(g_vec, alpha),                                       // [m]
      [m, 1]                                                         // [m,1]
    );
    g_vec.dispose();

    // Filtered state: x_filt = x_pred + K_bierman · v  = x_{t|t}
    using x_filt = np.add(x_pred, np.matmul(K_bierman, v));         // [m,1]

    // x_{t+1|t} = G · x_{t|t}
    const x_next_pred = np.matmul(G_t, x_filt);                     // [m,1]

    // Predicted-convention gain for the backward RTS smoother:
    // K = mask · G · K_bierman = mask · G · C_pred · F' / Cp  [m,1]
    using K_G = np.matmul(G_t, K_bierman);                          // [m,1]
    const K = np.multiply(mask_t, K_G);                              // [m,1] (0 when NaN)

    // ── Thornton time update (modified weighted Gram-Schmidt) ──
    // Factors C_{t+1|t} = G·C_filt·G' + W directly in UD form.
    //
    // Augmented decomposition:
    //   C_{t+1|t} = [G·U_filt, L_W] · diag(D_filt, D_W) · [G·U_filt, L_W]'
    // where W = L_W·D_W·L_W' (LDL from Cholesky).
    //
    // MWGS processes columns j=0..m-1, orthogonalizing and extracting the
    // unit lower-triangular factor U_next and positive diagonal D_next.

    // Factor W_t via Cholesky → LDL
    using W_sym = np.multiply(half, np.add(W_t, np.transpose(W_t)));
    using W_reg = np.add(W_sym, stab_cEps_I);                       // ε for Cholesky only
    using L_W = np.linalg.cholesky(W_reg);
    using L_Wd = np.einsum('ii->i', L_W);                           // [m] diagonal
    using L_Wdi = np.reciprocal(L_Wd);
    using L_Wdim = np.multiply(np.reshape(L_Wdi, [1, m]), stab_I_eye);
    using L_W_unit_raw = np.matmul(L_W, L_Wdim);                    // unit lower-tri
    using D_W = np.multiply(L_Wd, L_Wd);                            // [m] diagonal

    // Form B = G · U_filt (Bierman-updated U)
    using GU_filt = np.matmul(G_t, U_bar);                          // [m,m]
    U_bar.dispose();

    // MWGS working copies (deflated during the loop)
    // jax-js-lint: allow-non-using — accumulator-swap in MWGS loop
    let B_work = np.add(GU_filt, zeros_mm);                         // [m,m] clone
    let L_work = np.add(L_W_unit_raw, zeros_mm);                    // [m,m] clone

    // Output accumulators
    // jax-js-lint: allow-non-using — accumulated in MWGS loop
    let U_next = np.add(stab_I_eye, zeros_mm);                      // identity (unit diag)
    let D_next = np.add(Di, zeros_m);                                // placeholder, overwritten

    for (let j = 0; j < m; j++) {
      // Extract row j from B_work and L_work: [m]
      using b_j = np.einsum('ij,i->j', B_work, ud_oneHot[j]);      // [m]
      using l_j = np.einsum('ij,i->j', L_work, ud_oneHot[j]);      // [m]

      // D_next[j] = Σ_k D_filt[k]·b_j[k]² + Σ_k D_W[k]·l_j[k]²
      using b_sq = np.multiply(b_j, b_j);
      using l_sq = np.multiply(l_j, l_j);
      using term_B = np.sum(np.multiply(D_bar, b_sq));              // scalar
      using term_L = np.sum(np.multiply(D_W, l_sq));                // scalar
      using d_j_val = np.add(term_B, term_L);                       // scalar

      // Scatter d_j into D_next at position j
      {
        using d_old = np.sum(np.multiply(D_next, ud_oneHot[j]));    // scalar
        using delta_d = np.subtract(d_j_val, d_old);
        // jax-js-lint: allow-non-using — accumulator-swap: D_next updated
        const D_new = np.add(D_next, np.multiply(delta_d, ud_oneHot[j]));
        D_next.dispose();
        D_next = D_new;
      }

      // Compute u_{i,j} for all i > j (weighted inner products / d_j)
      // scores[i] = Σ_k D_filt[k]·B[i,k]·b_j[k] + Σ_k D_W[k]·L[i,k]·l_j[k]
      using wb_j = np.multiply(D_bar, b_j);                         // [m] weighted
      using wl_j = np.multiply(D_W, l_j);                           // [m] weighted
      using scores_B = np.matmul(B_work, np.reshape(wb_j, [m, 1]));// [m,1]
      using scores_L = np.matmul(L_work, np.reshape(wl_j, [m, 1]));// [m,1]
      using scores = np.add(scores_B, scores_L);                    // [m,1]
      using u_col_raw = np.divide(scores, np.reshape(d_j_val, [1, 1])); // [m,1]
      // Mask: only rows i > j
      using u_col = np.multiply(
        u_col_raw,
        np.reshape(ud_gtMask[j], [m, 1])
      );                                                             // [m,1]

      // Set U_next column j (sub-diagonal entries)
      using u_rank1 = np.matmul(u_col, ud_oneHotRow[j]);            // [m,m]
      // jax-js-lint: allow-non-using — accumulator-swap: U_next updated
      const U_next_new = np.add(U_next, u_rank1);
      U_next.dispose();
      U_next = U_next_new;

      // Deflate B_work: B[i,:] -= u_{i,j} · b_j  for all i > j
      using b_j_row = np.reshape(b_j, [1, m]);                      // [1,m]
      using deflate_B = np.matmul(u_col, b_j_row);                  // [m,m]
      // jax-js-lint: allow-non-using — accumulator-swap: B_work deflated
      const B_new = np.subtract(B_work, deflate_B);
      B_work.dispose();
      B_work = B_new;

      // Deflate L_work: L[i,:] -= u_{i,j} · l_j  for all i > j
      using l_j_row = np.reshape(l_j, [1, m]);                      // [1,m]
      using deflate_L = np.matmul(u_col, l_j_row);                  // [m,m]
      // jax-js-lint: allow-non-using — accumulator-swap: L_work deflated
      const L_new = np.subtract(L_work, deflate_L);
      L_work.dispose();
      L_work = L_new;
    }
    B_work.dispose();
    L_work.dispose();
    D_bar.dispose();

    return [
      { x: x_next_pred, U: U_next, D: D_next },
      { x_pred, C_pred, K, v, Cp, FF: FFi, Gt: G_t, mask: mask_t } as ForwardY,
    ];
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
    // Derive flat [n*p] inputs for diagnostics (n*p = n when p=1)
    using y_flat = np.reshape(y_arr, [n * p]);                    // [n*p]
    let V2_flat: np.Array;
    let V_flat: np.Array;
    if (p === 1) {
      V2_flat = np.squeeze(V2_arr);                               // [n]
      V_flat = np.sqrt(V2_flat);                                  // [n]
    } else {
      // V2_arr is [n,p,p] diagonal — extract per-component variance
      using V2_diag2d = np.einsum('nii->ni', V2_arr);            // [n,p] batch diagonal
      V2_flat = np.reshape(V2_diag2d, [n * p]);                  // [n*p]
      V_flat = np.sqrt(V2_flat);                                  // [n*p]
    }
    using _V2_flat = V2_flat;
    using _V_flat = V_flat;

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

    if (useSqrtAssocScan) {
      if (p > 1) throw new Error('sqrt-assoc algorithm not yet supported for multivariate observations (p > 1)');
      // ─── Square-Root Parallel Forward Filter (Yaghoobi et al. 2022, arXiv:2207.00426) ───
      // Reformulates the 5-tuple associative scan in Cholesky factor space.
      // Covariance matrices C, J (forward) and L (backward) are replaced by their
      // Cholesky factors U, Z, D.  Composition via block tria() ensures PSD by
      // construction — no Joseph form, symmetrize, or ε·I needed.
      //
      // tria(A) = R' where [Q, R] = qr(A') — proper QR-based triangular factor.
      // This avoids squaring the condition number (unlike the old cholesky(A·A'+ε·I)
      // fallback), enabling larger state dimensions (m > 6, e.g. fullSeasonal m=13).
      // Reference: EEA-sensors/sqrt-parallel-smoothers (JAX).
      type SqrtForwardElem = { A: np.Array; b: np.Array; U: np.Array; eta: np.Array; Z: np.Array };
      type SqrtBackwardElem = { g: np.Array; E: np.Array; D: np.Array };

      const tria_eps = f32 ? 1e-6 : 1e-12;

      using is_nan = np.isnan(y_arr);                    // [n,1,1] bool
      using zero_n11 = np.zerosLike(y_arr);              // [n,1,1]
      using one_n11 = np.onesLike(y_arr);                // [n,1,1]
      // jax-js-lint: allow-non-using — stored in fwd.mask, disposed after backward pass
      const mask_arr = np.where(is_nan, zero_n11, one_n11); // [n,1,1]
      using y_safe_arr = np.where(is_nan, zero_n11, y_arr); // [n,1,1]

      using I_eye = np.eye(stateSize, undefined, { dtype });
      using _I_eye_1mm = np.reshape(I_eye, [1, stateSize, stateSize]);
      using I_exp = np.tile(_I_eye_1mm, [n, 1, 1]);
      const gArrParts = np.split(G_scan, [n - 1], 0);
      const wArrParts = np.split(W_scan, [n - 1], 0);
      using G_head_arr = gArrParts[0];
      gArrParts[1].dispose();
      using W_head_arr = wArrParts[0];
      wArrParts[1].dispose();
      const gLastParts = np.split(G_scan, [n - 1], 0);
      using G_last = gLastParts[1];
      gLastParts[0].dispose();
      const wLastParts = np.split(W_scan, [n - 1], 0);
      using W_last = wLastParts[1];
      wLastParts[0].dispose();
      using G_arriving = np.concatenate([G_last, G_head_arr], 0);  // [n, m, m]
      using W_arriving = np.concatenate([W_last, W_head_arr], 0);  // [n, m, m]

      // cholW = cholesky(W_arriving + ε·I)  [n, m, m]
      using tria_eps_arr = np.array(tria_eps, { dtype });
      using regI_m = np.multiply(tria_eps_arr, I_exp);             // [n, m, m]
      using W_reg = np.add(W_arriving, regI_m);
      using cholW = np.linalg.cholesky(W_reg);                    // [n, m, m]

      // ─── Element construction for k >= 1 (zero prior: m0=0, L0=0) ───
      // With L0=0: N1_ = tria([G·0, cholW]) = cholW (predicted chol = cholW)
      //
      // Psi block [n, 1+m, m+1]:
      //   [[H·cholW, cholR],     H = FF_scan [n,1,m], cholR = V_std [n,1,1]
      //    [cholW,   zeros ]]
      using HcholW = np.einsum('nij,njk->nik', FF_scan, cholW);   // [n, 1, m]
      using V_std_arr = np.sqrt(V2_arr);                           // [n, 1, 1]
      using zeroCol = np.zeros([n, stateSize, 1], { dtype });
      using psi_top = np.concatenate([HcholW, V_std_arr], -1);     // [n, 1, m+1]
      using psi_bot = np.concatenate([cholW, zeroCol], -1);        // [n, m, m+1]
      using Psi = np.concatenate([psi_top, psi_bot], -2);          // [n, 1+m, m+1]

      // tria(Psi) via QR: [Q, R] = qr(Psi'), triaPsi = R'
      using PsiT = np.einsum('nij->nji', Psi);                    // [n, m+1, 1+m]
      const [Q_psi, R_psi] = np.linalg.qr(PsiT);
      Q_psi.dispose();
      using triaPsi = np.einsum('nij->nji', R_psi);               // [n, 1+m, 1+m]
      R_psi.dispose();

      // Extract sub-blocks via split
      // Split along axis -2 at index 1: top [n,1,1+m], bottom [n,m,1+m]
      const rowParts = np.split(triaPsi, [1], -2);
      using triaPsi_top = rowParts[0];                              // [n, 1, 1+m]
      using triaPsi_bot = rowParts[1];                              // [n, m, 1+m]
      // From top row: Psi11 = triaPsi_top[:, :, 0:1]  [n, 1, 1]
      const topColParts = np.split(triaPsi_top, [1], -1);
      using Psi11 = topColParts[0];                                 // [n, 1, 1]
      topColParts[1].dispose();
      // From bottom row: Psi21 = triaPsi_bot[:, :, 0:1], U = triaPsi_bot[:, :, 1:]
      const botColParts = np.split(triaPsi_bot, [1], -1);
      using Psi21 = botColParts[0];                                 // [n, m, 1]
      using U_obs = botColParts[1];                                 // [n, m, m]

      // K = Psi21 / Psi11  [n, m, 1]
      using K_sqrt = np.divide(Psi21, Psi11);                      // [n, m, 1]
      // A = G - K·H·G  [n, m, m]
      using KHG = np.einsum('nij,njk,nkl->nil', K_sqrt, FF_scan, G_arriving);
      using A_obs = np.subtract(G_arriving, KHG);                   // [n, m, m]
      // b = K·y  [n, m, 1]  (since m1=0, c=0)
      using b_obs = np.multiply(K_sqrt, y_safe_arr);                // [n, m, 1]

      // Z construction: solve_tri(Psi11, H·G, lower=True).T
      // For p=1: Psi11 is [n,1,1], H·G is [n,1,m]
      using HG = np.einsum('nij,njk->nik', FF_scan, G_arriving);   // [n, 1, m]
      using HG_over_psi = np.divide(HG, Psi11);                    // [n, 1, m]
      using Z_thin = np.einsum('nij->nji', HG_over_psi);           // [n, m, 1]
      // Pad Z to [n, m, m] (rank-1 for p=1 when m > 1)
      let Z_obs: np.Array;
      if (stateSize > 1) {
        using Z_pad = np.zeros([n, stateSize, stateSize - 1], { dtype });
        Z_obs = np.concatenate([Z_thin, Z_pad], -1);               // [n, m, m]
      } else {
        using _one_scalar = np.array(1.0, { dtype });
        Z_obs = np.multiply(_one_scalar, Z_thin);     // [n, 1, 1] copy
      }

      // eta = (Z_thin / Psi11) · y  [n, m, 1]
      using eta_factor = np.divide(Z_thin, Psi11);                 // [n, m, 1]
      using eta_obs = np.multiply(eta_factor, y_safe_arr);          // [n, m, 1]

      // ─── NaN handling: pure prediction elements ───
      // NaN: A = G_arriving, b = 0, U = cholW, Z = 0, eta = 0
      using nan_mm = np.tile(is_nan, [1, stateSize, stateSize]);
      using A_all = np.where(nan_mm, G_arriving, A_obs);
      using _b_zeros = np.zerosLike(b_obs);
      using b_all = np.where(is_nan, _b_zeros, b_obs);
      using U_all = np.where(nan_mm, cholW, U_obs);
      using zero_nmm = np.zerosLike(Z_obs);
      using _eta_zeros = np.zerosLike(eta_obs);
      using eta_all = np.where(is_nan, _eta_zeros, eta_obs);
      using Z_all_raw = np.where(nan_mm, zero_nmm, Z_obs);
      Z_obs.dispose();

      // ─── First element (k=0): exact initialization from prior ───
      // Compute filtered state at t=0 from prior (x0, C0) and first observation y[0].
      // For sqrt: cholC0 = cholesky(C0), then Psi block with L0 = cholC0.
      const F_parts = np.split(FF_scan, [1], 0);
      const V2_parts = np.split(V2_arr, [1], 0);
      const y_parts = np.split(y_safe_arr, [1], 0);
      const mask_parts = np.split(mask_arr, [1], 0);
      const A_parts = np.split(A_all, [1], 0);
      const b_parts = np.split(b_all, [1], 0);
      const U_parts = np.split(U_all, [1], 0);
      const eta_parts = np.split(eta_all, [1], 0);
      const Z_parts = np.split(Z_all_raw, [1], 0);

      using F1 = F_parts[0];
      using V2_1 = V2_parts[0];
      using y1 = y_parts[0];
      using mask1 = mask_parts[0];

      using C0_first = np.reshape(C0, [1, stateSize, stateSize]);
      using x0_first = np.reshape(x0, [1, stateSize, 1]);

      // cholC0 = cholesky(C0 + ε·I)
      using C0_reg = np.add(C0, np.multiply(tria_eps_arr, I_eye));
      using cholC0 = np.linalg.cholesky(C0_reg);                  // [m, m]
      using cholC0_1 = np.reshape(cholC0, [1, stateSize, stateSize]);

      // Psi for k=0: N1_ = cholC0 (no prediction step at k=0; prior IS the prediction)
      // Psi0 = [[H·cholC0, cholR], [cholC0, 0]]
      using F1_cholC0 = np.einsum('nij,jk->nik', F1, cholC0);     // [1, 1, m]
      using V_std_1 = np.sqrt(V2_1);                               // [1, 1, 1]
      using zeroCol_1 = np.zeros([1, stateSize, 1], { dtype });
      using psi0_top = np.concatenate([F1_cholC0, V_std_1], -1);   // [1, 1, m+1]
      using psi0_bot = np.concatenate([cholC0_1, zeroCol_1], -1);  // [1, m, m+1]
      using Psi0 = np.concatenate([psi0_top, psi0_bot], -2);       // [1, 1+m, m+1]

      // tria(Psi0) via QR
      using Psi0T = np.einsum('nij->nji', Psi0);                  // [1, m+1, 1+m]
      const [Q_psi0, R_psi0] = np.linalg.qr(Psi0T);
      Q_psi0.dispose();
      using triaPsi0 = np.einsum('nij->nji', R_psi0);             // [1, 1+m, 1+m]
      R_psi0.dispose();

      const row0Parts = np.split(triaPsi0, [1], -2);
      using triaPsi0_top = row0Parts[0];
      using triaPsi0_bot = row0Parts[1];
      const top0ColParts = np.split(triaPsi0_top, [1], -1);
      using Psi0_11 = top0ColParts[0];                             // [1, 1, 1]
      top0ColParts[1].dispose();
      const bot0ColParts = np.split(triaPsi0_bot, [1], -1);
      using Psi0_21 = bot0ColParts[0];                             // [1, m, 1]
      using U1_init = bot0ColParts[1];                              // [1, m, m]

      // K at k=0
      using _K0_div = np.divide(Psi0_21, Psi0_11);
      using K0 = np.multiply(mask1, _K0_div); // [1, m, 1]
      // b at k=0: x0 + K0·(y[0] - H·x0)
      using Fx0_1 = np.einsum('nij,njk->nik', F1, x0_first);
      using innov0 = np.subtract(y1, Fx0_1);
      using K0innov = np.multiply(K0, innov0);
      const b1 = np.add(x0_first, K0innov);                       // [1, m, 1]

      // First element: A=0, eta=0, Z=0 (prior doesn't propagate A-contribution)
      const A1 = np.zeros([1, stateSize, stateSize], { dtype });
      const eta1 = np.zeros([1, stateSize, 1], { dtype });
      const Z1 = np.zeros([1, stateSize, stateSize], { dtype });

      // Replace timestep 0 with exact first element
      const A_arr = np.concatenate([A1, A_parts[1]], 0);
      const b_arr = np.concatenate([b1, b_parts[1]], 0);
      const U_arr = np.concatenate([U1_init, U_parts[1]], 0);
      const eta_arr = np.concatenate([eta1, eta_parts[1]], 0);
      const Z_arr = np.concatenate([Z1, Z_parts[1]], 0);

      F_parts[1].dispose(); V2_parts[1].dispose(); y_parts[1].dispose(); mask_parts[1].dispose();
      A_parts[0].dispose(); A_parts[1].dispose();
      b_parts[0].dispose(); b_parts[1].dispose();
      U_parts[0].dispose(); U_parts[1].dispose();
      eta_parts[0].dispose(); eta_parts[1].dispose();
      Z_parts[0].dispose(); Z_parts[1].dispose();

      // ─── Square-root forward composition operator ───
      // Hoist I1 outside compose — vmap broadcasts [1,m,m] correctly (jax-js ≥65cb449)
      using _I1_eye = np.eye(stateSize, undefined, { dtype });
      using I1_sqrt = np.reshape(_I1_eye, [1, stateSize, stateSize]);

      const composeSqrtForward = (a: SqrtForwardElem, b_elem: SqrtForwardElem): SqrtForwardElem => {
        const m = stateSize;

        // Xi block [n, 2m, 2m]:
        //   [[U1' · Z2,  I],
        //    [Z2,         0]]
        using U1t = np.einsum('nij->nji', a.U);                     // [n, m, m]
        using U1tZ2 = np.einsum('nij,njk->nik', U1t, b_elem.Z);    // [n, m, m]
        using _zeros_U = np.zerosLike(a.U);                         // [n, m, m]
        using I_batch = np.add(_zeros_U, I1_sqrt);                   // [n, m, m] via broadcast
        using zero_mm = np.zerosLike(a.U);                          // [n, m, m]
        // Xi top: [U1'Z2, I] → [n, m, 2m]
        using xi_top = np.concatenate([U1tZ2, I_batch], -1);
        // Xi bot: [Z2, 0] → [n, m, 2m]
        using xi_bot = np.concatenate([b_elem.Z, zero_mm], -1);
        // Xi: [n, 2m, 2m]
        using Xi = np.concatenate([xi_top, xi_bot], -2);

        // tria(Xi) via QR: [Q, R] = qr(Xi'), tria_xi = R'
        using XiT = np.einsum('nij->nji', Xi);                     // [n, 2m, 2m]
        const [Q_xi, R_xi] = np.linalg.qr(XiT);
        Q_xi.dispose();
        using tria_xi = np.einsum('nij->nji', R_xi);               // [n, 2m, 2m]
        R_xi.dispose();

        // Extract Xi11 [n,m,m], Xi21 [n,m,m], Xi22 [n,m,m]
        const xiRowParts = np.split(tria_xi, [m], -2);
        using xi_top_row = xiRowParts[0];                             // [n, m, 2m]
        using xi_bot_row = xiRowParts[1];                             // [n, m, 2m]
        const xiTopCols = np.split(xi_top_row, [m], -1);
        using Xi11 = xiTopCols[0];                                    // [n, m, m]
        xiTopCols[1].dispose();
        const xiBotCols = np.split(xi_bot_row, [m], -1);
        using Xi21 = xiBotCols[0];                                    // [n, m, m]
        using Xi22 = xiBotCols[1];                                    // [n, m, m]

        // S1 = triangularSolve(Xi11, U1'·A2')  [n, m, m]  (Xi11 is lower triangular from tria)
        using A2t = np.einsum('nij->nji', b_elem.A);
        using U1tA2t = np.einsum('nij,njk->nik', U1t, A2t);
        using S1 = lax.linalg.triangularSolve(Xi11, U1tA2t, { leftSide: true, lower: true });
        using S1t = np.einsum('nij->nji', S1);                       // [n, m, m]

        // A_comp = A2·A1 - S1'·Xi21'·A1
        using A2A1 = np.einsum('nij,njk->nik', b_elem.A, a.A);
        using Xi21t = np.einsum('nij->nji', Xi21);
        using S1tXi21tA1 = np.einsum('nij,njk,nkl->nil', S1t, Xi21t, a.A);
        const A_comp = np.subtract(A2A1, S1tXi21tA1);

        // For b_comp: A2·(I - R'·Xi21')·(b1 + U1·U1'·eta2) + b2
        // where R = triangularSolve(Xi11, U1')  (NOTE: different from S1 which uses U1'·A2')
        using R_b = lax.linalg.triangularSolve(Xi11, U1t, { leftSide: true, lower: true });
        using R_bt = np.einsum('nij->nji', R_b);                     // U1·Xi11⁻ᵀ
        using RbtXi21t = np.einsum('nij,njk->nik', R_bt, Xi21t);    // [n, m, m]
        using ImRX = np.subtract(I_batch, RbtXi21t);                 // [n, m, m]
        using U1U1t = np.einsum('nij,nkj->nik', a.U, a.U);          // [n, m, m] = C1
        using U1U1tEta2 = np.einsum('nij,njk->nik', U1U1t, b_elem.eta); // [n, m, 1]
        using b1_plus = np.add(a.b, U1U1tEta2);
        using A2ImRX = np.einsum('nij,njk->nik', b_elem.A, ImRX);
        using A2ImRX_b1 = np.einsum('nij,njk->nik', A2ImRX, b1_plus);
        const b_comp = np.add(A2ImRX_b1, b_elem.b);

        // U_comp = tria([S1', U2]) via QR
        using U_wide = np.concatenate([S1t, b_elem.U], -1);         // [n, m, 2m]
        using U_wideT = np.einsum('nij->nji', U_wide);               // [n, 2m, m]
        const [Q_u, R_u] = np.linalg.qr(U_wideT);
        Q_u.dispose();
        const U_comp = np.einsum('nij->nji', R_u);                   // [n, m, m]
        R_u.dispose();

        // eta_comp = A1'·(I - triangularSolve(Xi11, Xi21', trans=True)'·U1')·(eta2 - Z2·Z2'·b1) + eta1
        // triangularSolve with transposeA=true: Xi11' \ Xi21'
        using R1 = lax.linalg.triangularSolve(Xi11, Xi21t, { leftSide: true, lower: true, transposeA: true });
        using R1t = np.einsum('nij->nji', R1);                       // [n, m, m]
        using R1tU1t = np.einsum('nij,njk->nik', R1t, U1t);         // [n, m, m]
        using ImRU = np.subtract(I_batch, R1tU1t);                 // [n, m, m]
        using Z2Z2t = np.einsum('nij,nkj->nik', b_elem.Z, b_elem.Z); // [n, m, m] = J2
        using Z2Z2tb1 = np.einsum('nij,njk->nik', Z2Z2t, a.b);      // [n, m, 1]
        using eta_diff = np.subtract(b_elem.eta, Z2Z2tb1);           // [n, m, 1]
        using ImRU_eta = np.einsum('nij,njk->nik', ImRU, eta_diff);  // [n, m, 1]
        using A1t = np.einsum('nij->nji', a.A);
        using A1t_ImRU_eta = np.einsum('nij,njk->nik', A1t, ImRU_eta);
        const eta_comp = np.add(A1t_ImRU_eta, a.eta);

        // Z_comp = tria([A1'·Xi22, Z1]) via QR
        using A1tXi22 = np.einsum('nij,njk->nik', A1t, Xi22);      // [n, m, m]
        using Z_wide = np.concatenate([A1tXi22, a.Z], -1);          // [n, m, 2m]
        using Z_wideT = np.einsum('nij->nji', Z_wide);               // [n, 2m, m]
        const [Q_z, R_z] = np.linalg.qr(Z_wideT);
        Q_z.dispose();
        const Z_comp = np.einsum('nij->nji', R_z);                   // [n, m, m]
        R_z.dispose();

        return { A: A_comp, b: b_comp, U: U_comp, eta: eta_comp, Z: Z_comp };
      };

      const scanned = lax.associativeScan(
        composeSqrtForward,
        { A: A_arr, b: b_arr, U: U_arr, eta: eta_arr, Z: Z_arr },
      ) as SqrtForwardElem;

      // Recover filtered state: x_filt = A·x0 + b
      using _x0_col = np.reshape(x0, [stateSize, 1]);
      using Ax0 = np.matmul(scanned.A, _x0_col);  // [n,m,m]×[m,1] → [n,m,1]
      const x_filt = np.add(Ax0, scanned.b);                        // [n, m, 1]

      // Recover filtered covariance: C_filt = A·C0·A' + U·U'   (where C = U U')
      using AC0At = np.einsum('nij,jk,nlk->nil', scanned.A, C0, scanned.A);  // broadcast C0 [m,m]
      using UUt = np.einsum('nij,nkj->nik', scanned.U, scanned.U);
      const C_filt = np.add(AC0At, UUt);                            // [n, m, m]

      scanned.A.dispose(); scanned.b.dispose(); scanned.U.dispose();
      scanned.eta.dispose(); scanned.Z.dispose();
      A1.dispose(); b1.dispose(); U_arr.dispose(); eta1.dispose(); Z1.dispose();
      A_arr.dispose(); b_arr.dispose(); eta_arr.dispose(); Z_arr.dispose();

      // ─── Recover sequential-convention diagnostics ───
      // (Same recovery logic as the standard assoc path)
      const gParts = np.split(G_scan, [n - 1], 0);
      using G_arr_head = gParts[0];
      gParts[1].dispose();
      const wParts = np.split(W_scan, [n - 1], 0);
      using W_arr_head = wParts[0];
      wParts[1].dispose();

      const xFiltParts = np.split(x_filt, [n - 1], 0);
      xFiltParts[1].dispose();
      using x_filt_head = xFiltParts[0];
      using x_filt_pred = np.einsum('nij,njk->nik', G_arr_head, x_filt_head);
      using x0_1 = np.reshape(x0, [1, stateSize, 1]);
      // jax-js-lint: allow-non-using — stored in fwd.x_pred
      const x_pred_arr = np.concatenate([x0_1, x_filt_pred], 0);

      const cFiltParts = np.split(C_filt, [n - 1], 0);
      cFiltParts[1].dispose();
      using C_filt_head = cFiltParts[0];
      using GCGt_pred = np.einsum('nij,njk,nlk->nil', G_arr_head, C_filt_head, G_arr_head);
      using C_filt_pred = np.add(GCGt_pred, W_arr_head);
      using C0_1 = np.reshape(C0, [1, stateSize, stateSize]);
      // jax-js-lint: allow-non-using — stored in fwd.C_pred
      const C_pred_arr = np.concatenate([C0_1, C_filt_pred], 0);

      using Fx_pred = np.einsum('nij,njk->nik', FF_scan, x_pred_arr);
      using v_raw = np.subtract(y_safe_arr, Fx_pred);
      // jax-js-lint: allow-non-using — stored in fwd.v
      const v_arr = np.multiply(mask_arr, v_raw);

      using FCFt = np.einsum('nij,njk,nlk->nil', FF_scan, C_pred_arr, FF_scan);
      // jax-js-lint: allow-non-using — stored in fwd.Cp
      const Cp_arr = np.add(FCFt, V2_arr);

      using GCFt2 = np.einsum('nij,njk,nlk->nil', G_scan, C_pred_arr, FF_scan);
      using K_raw2 = np.divide(GCFt2, Cp_arr);
      // jax-js-lint: allow-non-using — stored in fwd.K
      const K_arr = np.multiply(mask_arr, K_raw2);

      fwd = {
        x_pred: x_pred_arr, C_pred: C_pred_arr,
        K: K_arr, v: v_arr, Cp: Cp_arr,
        FF: FF_scan, mask: mask_arr,
      } as unknown as ForwardY;

      // ─── Square-root Parallel Backward Smoother (Yaghoobi et al. 2022) ───
      {
        // Compute Cholesky of filtered covariance for backward elements
        using C_filt_reg = np.add(C_filt, regI_m);
        using cholC_filt = np.linalg.cholesky(C_filt_reg);          // [n, m, m]

        // Backward element construction: Phi block → E, g, D
        // Phi = [[G·cholC, cholW], [cholC, 0]]   size [n, 2m, 2m]
        using GcholC = np.einsum('nij,njk->nik', G_scan, cholC_filt); // [n, m, m]
        // Recompute cholW for departing G/W (already using original G_scan/W_scan)
        using W_scan_reg = np.add(W_scan, regI_m);
        using cholW_dep = np.linalg.cholesky(W_scan_reg);            // [n, m, m]
        using zero_bwd = np.zeros([n, stateSize, stateSize], { dtype });
        using phi_top = np.concatenate([GcholC, cholW_dep], -1);      // [n, m, 2m]
        using phi_bot = np.concatenate([cholC_filt, zero_bwd], -1);   // [n, m, 2m]
        using Phi = np.concatenate([phi_top, phi_bot], -2);           // [n, 2m, 2m]

        // tria(Phi) via QR: [Q, R] = qr(Phi'), triaPhi = R'
        using PhiT = np.einsum('nij->nji', Phi);                   // [n, 2m, 2m]
        const [Q_phi, R_phi] = np.linalg.qr(PhiT);
        Q_phi.dispose();
        using triaPhi = np.einsum('nij->nji', R_phi);               // [n, 2m, 2m]
        R_phi.dispose();

        // Extract Phi11, Phi21, D
        const phiRowParts = np.split(triaPhi, [stateSize], -2);
        using phi_top_row = phiRowParts[0];                           // [n, m, 2m]
        using phi_bot_row = phiRowParts[1];                           // [n, m, 2m]
        const phiTopCols = np.split(phi_top_row, [stateSize], -1);
        using Phi11 = phiTopCols[0];                                  // [n, m, m]
        phiTopCols[1].dispose();
        const phiBotCols = np.split(phi_bot_row, [stateSize], -1);
        using Phi21 = phiBotCols[0];                                  // [n, m, m]
        // jax-js-lint: allow-non-using — D_raw disposed after terminal masking
        const D_raw = phiBotCols[1];                                  // [n, m, m]

        // E = triangularSolve(Phi11, Phi21', transposeA=true)'  [n, m, m]
        using Phi21t = np.einsum('nij->nji', Phi21);
        using E_solve = lax.linalg.triangularSolve(Phi11, Phi21t, { leftSide: true, lower: true, transposeA: true });
        using E_raw = np.einsum('nij->nji', E_solve);                // [n, m, m]

        // g = x_filt - E·(G·x_filt + b)  = (I - E·G)·x_filt  (b=0 for DLM)
        using EG = np.einsum('nij,njk->nik', E_raw, G_scan);
        using ImEG = np.subtract(I_exp, EG);
        using g_raw = np.einsum('nij,njk->nik', ImEG, x_filt);      // [n, m, 1]

        // Terminal element: E[n-1]=0, g[n-1]=x_filt[n-1], D[n-1]=cholC_filt[n-1]
        using term_bool = np.array(
          Array.from({ length: n }, (_, t) => [[t < n - 1]]),
        );  // [n, 1, 1] bool
        using term_bool_mm = np.tile(term_bool, [1, stateSize, stateSize]); // [n, m, m] bool
        // jax-js-lint: allow-non-using — E_all, g_all, D_all disposed after scan
        using _E_zeros_sqrt = np.zerosLike(E_raw);
        const E_all = np.where(term_bool, E_raw, _E_zeros_sqrt);
        // For g at terminal: g[n-1] = x_filt[n-1] (not (I-E·G)·x_filt which would be same since E=0)
        // Since E[n-1] gets zeroed, g[n-1] = (I - 0)·x_filt[n-1] = x_filt[n-1]. Correct.
        const g_all = np.where(term_bool, g_raw, x_filt);
        // D[n-1] = cholC_filt[n-1], D[k] = D_raw[k] for k < n-1
        // Terminal D: cholC_filt at last timestep
        const D_all = np.where(term_bool_mm, D_raw, cholC_filt);
        D_raw.dispose();

        // ─── Square-root backward composition operator ───
        const composeSqrtBackward = (a: SqrtBackwardElem, b_elem: SqrtBackwardElem): SqrtBackwardElem => {
          // g = E2·g1 + g2
          using E2g1 = np.einsum('nij,njk->nik', b_elem.E, a.g);
          const g_comp = np.add(E2g1, b_elem.g);
          // E = E2·E1
          const E_comp = np.einsum('nij,njk->nik', b_elem.E, a.E);
          // D = tria([E2·D1, D2]) via QR
          using E2D1 = np.einsum('nij,njk->nik', b_elem.E, a.D);
          using D_wide = np.concatenate([E2D1, b_elem.D], -1);     // [n, m, 2m]
          using D_wideT = np.einsum('nij->nji', D_wide);             // [n, 2m, m]
          const [Q_d, R_d] = np.linalg.qr(D_wideT);
          Q_d.dispose();
          const D_comp = np.einsum('nij->nji', R_d);                 // [n, m, m]
          R_d.dispose();
          return { g: g_comp, E: E_comp, D: D_comp };
        };

        const smoothed = lax.associativeScan(
          composeSqrtBackward,
          { g: g_all, E: E_all, D: D_all },
          { reverse: true }
        ) as SqrtBackwardElem;

        // Smoothed estimates: x_smooth = g_comp, C_smooth = D·D'
        x_smooth = smoothed.g;                                      // [n, m, 1]
        // jax-js-lint: allow-non-using — ownership transferred to C_smooth (returned)
        C_smooth = np.einsum('nij,nkj->nik', smoothed.D, smoothed.D); // [n, m, m] — PSD by construction
        smoothed.E.dispose();
        smoothed.D.dispose();
        E_all.dispose();
        g_all.dispose();
        D_all.dispose();
      }

      x_filt.dispose();
      C_filt.dispose();

    } else if (useAssocScan) {
      // ─── Exact Parallel Forward Filter (Särkkä & García-Fernández 2020, Lemmas 1–2) ───
      // 5-tuple elements: (A, b, C, eta, J)
      type ForwardElem = { A: np.Array; b: np.Array; C: np.Array; eta: np.Array; J: np.Array };
      type BackwardElem = { A: np.Array; b: np.Array; S: np.Array };

      // All-or-nothing NaN mask: [n,1,1] bool (if ANY p-component is NaN, skip step)
      let is_nan_n11: np.Array;
      if (p === 1) {
        is_nan_n11 = np.isnan(y_arr);                                              // [n,1,1]
      } else {
        // Sum over p: NaN propagates → [n,1,1]
        using y_2d = np.reshape(y_arr, [n, p]);                                    // [n,p]
        using y_rowsum = np.sum(y_2d, [1]);                                        // [n]
        using y_3d = np.reshape(y_rowsum, [n, 1, 1]);                              // [n,1,1]
        is_nan_n11 = np.isnan(y_3d);                                               // [n,1,1]
      }
      using is_nan = is_nan_n11;
      using zero_np1 = np.zerosLike(y_arr);              // [n,p,1]
      using one_n11 = np.ones([n, 1, 1], { dtype });     // [n,1,1]
      using zero_n11 = np.zeros([n, 1, 1], { dtype });   // [n,1,1]
      // jax-js-lint: allow-non-using — stored in fwd.mask, disposed after backward pass
      const mask_arr = np.where(is_nan, zero_n11, one_n11); // [n,1,1]
      using y_safe_arr = np.where(is_nan, zero_np1, y_arr); // [n,p,1] (broadcast [n,1,1] → [n,p,1])

      using I_eye = np.eye(stateSize, undefined, { dtype });
      using _I_eye_1mm = np.reshape(I_eye, [1, stateSize, stateSize]);
      using I_exp = np.tile(_I_eye_1mm, [n, 1, 1]);
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
      using G_unit_1 = np.reshape(np.tile(_I_eye_1mm, [1, 1, 1]), [1, stateSize, stateSize]);
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
      using S_obs = np.add(np.einsum('nij,njk,nlk->nil', FF_scan, W_arriving, FF_scan), V2_arr); // [n,p,p]
      using WFt = np.einsum('nij,nkj->nik', W_arriving, FF_scan);                               // [n,m,p]
      // K_obs = W·F'·S⁻¹: p=1 scalar division, p>1 batched matrix inverse
      if (p === 1) {
        var K_obs = np.divide(WFt, S_obs);                                                      // [n,m,1]
      } else {
        using S_inv = np.linalg.inv(S_obs);                                                      // [n,p,p]
        var K_obs = np.einsum('nij,njk->nik', WFt, S_inv);                                      // [n,m,p]
      }
      using _K_obs = K_obs;

      using KF_obs = np.einsum('nij,njk->nik', K_obs, FF_scan);                             // [n,m,m]
      using ImKF_obs = np.subtract(I_exp, KF_obs);                                           // [n,m,m]
      using A_obs = np.einsum('nij,njk->nik', ImKF_obs, G_arriving);                         // [n,m,m]
      using C_obs = np.einsum('nij,njk->nik', ImKF_obs, W_arriving);                         // [n,m,m]
      using b_obs = np.einsum('nij,njk->nik', K_obs, y_safe_arr);                           // [n,m,1]

      using Ft = np.einsum('nij->nji', FF_scan);                                             // [n,m,p]
      // Ft_over_S = F'·S⁻¹: p=1 scalar division, p>1 batched matrix inverse
      if (p === 1) {
        var Ft_over_S = np.divide(Ft, S_obs);                                                // [n,m,1]
      } else {
        using S_inv2 = np.linalg.inv(S_obs);                                                 // [n,p,p]
        var Ft_over_S = np.einsum('nij,njk->nik', Ft, S_inv2);                              // [n,m,p]
      }
      using _Ft_over_S = Ft_over_S;
      using Gt_batch = np.einsum('nij->nji', G_arriving);                                    // [n,m,m]
      using eta_obs_base = np.einsum('nij,njk->nik', Gt_batch, Ft_over_S);                   // [n,m,1]
      using eta_obs = np.einsum('nij,njk->nik', eta_obs_base, y_safe_arr);                  // [n,m,1]
      using FtF_over_S = np.einsum('nij,njk->nik', Ft_over_S, FF_scan);                      // [n,m,m]
      using J_obs = np.einsum('nij,njk,nkl->nil', Gt_batch, FtF_over_S, G_arriving);         // [n,m,m]

      // NaN handling for k>=2 elements: pure prediction for gapped y
      using nan_mm = np.tile(is_nan, [1, stateSize, stateSize]);
      using A_all = np.where(nan_mm, G_arriving, A_obs);
      using _b_zeros = np.zerosLike(b_obs);
      using b_all = np.where(is_nan, _b_zeros, b_obs);
      using C_all = np.where(nan_mm, W_arriving, C_obs);
      using zero_nmm = np.zerosLike(J_obs);
      using _eta_zeros = np.zerosLike(eta_obs);
      using eta_all = np.where(is_nan, _eta_zeros, eta_obs);
      using J_all = np.where(nan_mm, zero_nmm, J_obs);

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

      using S1 = np.add(np.einsum('nij,njk,nlk->nil', F1, C0_first, F1), V2_1);          // [1,p,p]
      using C0Ft1 = np.einsum('ij,nkj->nik', C0, F1);                                     // [1,m,p]
      // K1_obs = C0·F'·S1⁻¹: p=1 scalar division, p>1 matrix inverse
      if (p === 1) {
        using K1_obs = np.divide(C0Ft1, S1);                                              // [1,m,1]
        var K1 = np.multiply(mask1, K1_obs);                                              // [1,m,1]
      } else {
        using S1_inv = np.linalg.inv(S1);                                                 // [1,p,p]
        using K1_obs = np.einsum('nij,njk->nik', C0Ft1, S1_inv);                         // [1,m,p]
        var K1 = np.multiply(mask1, K1_obs);                                              // [1,m,p]
      }
      using _K1 = K1;

      using Fx0_1 = np.einsum('nij,njk->nik', F1, x0_first);                              // [1,p,1]
      using innov1 = np.subtract(y1, Fx0_1);                                              // [1,p,1]
      using Kinnov1 = np.einsum('nij,njk->nik', K1, innov1);                              // [1,m,1]
      const b1 = np.add(x0_first, Kinnov1);                                               // [1,m,1]

      using K1S1 = np.einsum('nij,njk->nik', K1, S1);                                     // [1,m,p]
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

      // Hoist constants outside compose — vmap broadcasts [1,m,m] correctly (jax-js ≥65cb449)
      using _I1_eye_std = np.eye(stateSize, undefined, { dtype });
      using I1 = np.reshape(_I1_eye_std, [1, stateSize, stateSize]);
      using inv_eps = np.array(dtype === DType.Float32 ? 1e-6 : 1e-12, { dtype });
      using _inv_eps_1 = np.reshape(inv_eps, [1, 1, 1]);
      using regI = np.multiply(_inv_eps_1, I1);

      const composeForward = (a: ForwardElem, b_elem: ForwardElem): ForwardElem => {
        // Compose later (j=b_elem) after earlier (i=a)
        // M = (I + C_i J_j)^-1
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

      using _x0_col = np.reshape(x0, [stateSize, 1]);
      using Ax0 = np.matmul(scanned.A, _x0_col);  // [n,m,m]×[m,1] → [n,m,1]
      const x_filt = np.add(Ax0, scanned.b);             // [n, m, 1]

      using AC0At = np.einsum('nij,jk,nlk->nil', scanned.A, C0, scanned.A);  // broadcast C0 [m,m]
      using C_filt_raw = np.add(AC0At, scanned.C);

      using C_filt_t = np.einsum('nij->nji', C_filt_raw);
      using C_filt_sum = np.add(C_filt_raw, C_filt_t);
      const C_filt = np.multiply(half, C_filt_sum); // [n,m,m]

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

      // v[t] = mask · (y - F·x_pred)  [n,p,1]
      using Fx_pred = np.einsum('nij,njk->nik', FF_scan, x_pred_arr); // [n,p,1]
      using v_raw = np.subtract(y_safe_arr, Fx_pred);
      // jax-js-lint: allow-non-using — stored in fwd.v, disposed by caller
      const v_arr = np.multiply(mask_arr, v_raw);         // [n,p,1]

      // Cp[t] = F·C_pred·F' + V²  [n,p,p]
      using FCFt = np.einsum('nij,njk,nlk->nil', FF_scan, C_pred_arr, FF_scan);
      // jax-js-lint: allow-non-using — stored in fwd.Cp, disposed by caller
      const Cp_arr = np.add(FCFt, V2_arr);                // [n,p,p]

      // K[t] = mask · G[t]·C_pred[t]·F[t]'·Cp[t]⁻¹  [n,m,p]  (MATLAB convention for backward pass)
      using GCFt = np.einsum('nij,njk,nlk->nil', G_scan, C_pred_arr, FF_scan); // [n,m,p]
      if (p === 1) {
        using K_raw = np.divide(GCFt, Cp_arr);
        // jax-js-lint: allow-non-using — stored in fwd.K, disposed by caller
        var K_arr = np.multiply(mask_arr, K_raw);          // [n,m,1]
      } else {
        using Cp_inv = np.linalg.inv(Cp_arr);              // [n,p,p]
        using K_raw = np.einsum('nij,njk->nik', GCFt, Cp_inv); // [n,m,p]
        // jax-js-lint: allow-non-using — stored in fwd.K, disposed by caller
        var K_arr = np.multiply(mask_arr, K_raw);          // [n,m,p]
      }

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
        using term_bool = np.array(
          Array.from({ length: n }, (_, t) => [[t < n - 1]]),
        );  // [n, 1, 1] bool
        // jax-js-lint: allow-non-using — E_all disposed after scan
        using _E_zeros = np.zerosLike(E_raw);
        const E_all = np.where(term_bool, E_raw, _E_zeros);  // [n, m, m]

        // ImEG = I - E_k · G[k]  [n, m, m]
        using EG = np.einsum('nij,njk->nik', E_all, G_scan);
        using I_eye = np.eye(stateSize, undefined, { dtype });
        using _I_eye_bwd = np.reshape(I_eye, [1, stateSize, stateSize]);
        using I_exp = np.tile(_I_eye_bwd, [n, 1, 1]);
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
        const L_all = np.multiply(half, L_sum);

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

    } else if (useUdScan) {
      // ─── UD (Bierman) Forward Filter ───
      // Carries U (unit lower-triangular) and D (positive diagonal) instead of C.
      // Bierman column-by-column measurement update avoids condition number squaring.
      // Only supports p=1 (scalar observations).
      if (p > 1) throw new Error('ud algorithm not yet supported for multivariate observations (p > 1)');

      // Initial UD: C0 = L_unit·D·L_unit' via Cholesky → LDL
      // Store L_unit (unit lower-tri) as U so that C = U·D·U' holds.
      using C0sym = np.add(C0, np.transpose(C0));
      using C0half = np.multiply(half, C0sym);
      using C0reg = np.add(C0half, stab_cEps_I);
      using L0 = np.linalg.cholesky(C0reg);
      using L0d = np.einsum('ii->i', L0);
      using L0di = np.reciprocal(L0d);
      using L0dim = np.multiply(np.reshape(L0di, [1, stateSize]), stab_I_eye);
      using U0_init = np.matmul(L0, L0dim);                         // unit lower-tri
      using D0_init = np.multiply(L0d, L0d);

      // eslint-disable-next-line jax-js/require-scan-result-dispose
      const [udCarry, udSeq] = lax.scan(
        forwardStepUD,
        { x: x0, U: U0_init, D: D0_init },
        { y: y_arr, V2: V2_arr, FF: FF_scan, Gt: G_scan, Wt: W_scan }
      );
      tree.dispose(udCarry);
      fwd = udSeq as unknown as ForwardY;

      // ─── Sequential Backward RTS Smoother (reuses standard backwardStep) ───
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

    // NaN observation mask: [n] (per-timestep, same for all p components)
    using mask_n = np.squeeze(fwd.mask);                // [n] from [n,1,1]
    // For p>1 diagnostics, expand to [n*p] (repeat each timestep mask p times)
    let mask_flat: np.Array;
    if (p === 1) {
      mask_flat = np.add(mask_n, np.zerosLike(mask_n));  // [n] (copy to avoid alias)
    } else {
      using mask_n1 = np.reshape(mask_n, [n, 1]);        // [n,1]
      using mask_np = np.tile(mask_n1, [1, p]);           // [n,p]
      mask_flat = np.reshape(mask_np, [n * p]);           // [n*p]
    }
    using _mask_flat = mask_flat;

    // yhat = FF @ x_pred: [n,p,m] @ [n,m,1] → [n,p,1] → [n*p]
    using yhat_3d = np.matmul(FF_scan, fwd.x_pred);
    const yhat = np.reshape(yhat_3d, [n * p]);

    // ystd = sqrt(diag(FF @ C_smooth @ FF') + diag(V²))  [n*p]
    using FCFt_3d = np.einsum('nij,njk,nlk->nil', FF_scan, C_smooth, FF_scan); // [n,p,p]
    if (p === 1) {
      using FCFt_sq = np.squeeze(FCFt_3d);                // [n]
      var ystd = np.sqrt(np.add(FCFt_sq, V2_flat));       // [n]
    } else {
      using FCFt_diag = np.einsum('nii->ni', FCFt_3d);    // [n,p] batch diagonal
      using FCFt_np = np.reshape(FCFt_diag, [n * p]);     // [n*p]
      var ystd = np.sqrt(np.add(FCFt_np, V2_flat));        // [n*p]
    }

    // Innovations [n,p,1] → [n*p]
    const v_flat = np.reshape(fwd.v, [n * p]);
    // Cp: [n,p,p] for matrix, [n] for scalar (p=1)
    if (p === 1) {
      var Cp_out = np.squeeze(fwd.Cp);                    // [n]
    } else {
      // Keep as [n,p,p] (needed for lik, returned as-is)
      var Cp_out = np.add(fwd.Cp, np.zerosLike(fwd.Cp));  // copy
    }

    // Dispose forward arrays no longer needed
    fwd.K.dispose();
    fwd.FF.dispose();
    if (fwd.Gt) fwd.Gt.dispose();
    fwd.mask.dispose();

    // y_safe: replace NaN with 0 for numerically safe reductions [n*p]
    using is_nan_y = np.isnan(y_flat);                     // [n*p] bool
    using _zeros_y = np.zerosLike(y_flat);                 // [n*p]
    using y_safe = np.where(is_nan_y, _zeros_y, y_flat);   // [n*p]

    // Residuals: NaN at missing positions [n*p]
    const resid0 = np.subtract(y_flat, yhat);              // [n*p]: NaN at missing obs
    const resid  = np.divide(resid0, V_flat);              // [n*p]: NaN at missing obs

    // Standardised prediction residuals: NaN at missing positions [n*p]
    if (p === 1) {
      using Cp_flat_1d = np.squeeze(fwd.Cp);
      using Cp_sqrt = np.sqrt(Cp_flat_1d);                // [n]
      var resid2_raw = np.divide(v_flat, Cp_sqrt);        // [n]: 0 at NaN pos (v=0)
    } else {
      // For p>1: per-component standardize using diag(Cp)
      using Cp_diag = np.einsum('nii->ni', fwd.Cp);      // [n,p] batch diagonal
      using Cp_diag_flat = np.reshape(Cp_diag, [n * p]);  // [n*p]
      using Cp_sqrt = np.sqrt(Cp_diag_flat);
      var resid2_raw = np.divide(v_flat, Cp_sqrt);        // [n*p]
    }
    using _resid2_raw = resid2_raw;
    using nan_arr = np.full([n * p], NaN, { dtype });
    const resid2 = np.where(is_nan_y, nan_arr, resid2_raw);

    // NaN-safe scalar reductions — use mask_flat to exclude missing timesteps
    using resid0_safe = np.subtract(y_safe, yhat);
    using resid_safe  = np.divide(resid0_safe, V_flat);
    const nobs = np.sum(mask_n);                          // scalar: count of valid timesteps

    const ssy  = np.sum(np.multiply(mask_flat, np.square(resid0_safe)));

    // Likelihood: −2·log L  (scalar)
    // p=1: sum(mask * (v²/Cp + log(Cp)))
    // p>1: sum_t(mask_t * (v_t'·Cp_t⁻¹·v_t + log|det(Cp_t)|))
    let lik: np.Array;
    if (p === 1) {
      using Cp_1d = np.squeeze(fwd.Cp);
      lik = np.sum(np.multiply(mask_n, np.add(
        np.divide(np.square(v_flat), Cp_1d),
        np.log(Cp_1d)
      )));
    } else {
      // Quadratic form v'·Cp⁻¹·v per timestep
      using Cp_inv = np.linalg.inv(fwd.Cp);               // [n,p,p]
      using v_3d = np.reshape(fwd.v, [n, p, 1]);          // [n,p,1]
      using Cpinv_v = np.einsum('nij,njk->nik', Cp_inv, v_3d); // [n,p,1]
      using vCpinv_v = np.einsum('nji,njk->ni', v_3d, Cpinv_v); // [n,1]
      using quad = np.reshape(vCpinv_v, [n]);              // [n]
      // Log determinant per timestep
      const [slogdet_sign, slogdet_logabs] = np.linalg.slogdet(fwd.Cp); // [sign: [n], logabsdet: [n]]
      using _sld_sign = slogdet_sign;
      using _sld_logabs = slogdet_logabs;
      using logdet = np.multiply(slogdet_sign, slogdet_logabs); // [n] (sign should be +1 for PD)
      lik = np.sum(np.multiply(mask_n, np.add(quad, logdet)));
    }

    const s2   = np.divide(
      np.sum(np.multiply(mask_flat, np.square(resid_safe))), nobs);
    const mse  = np.divide(
      np.sum(np.multiply(mask_flat, np.square(resid2_raw))), nobs);
    using _eps_mape = np.array(1e-30, { dtype });
    const mape = np.divide(
      np.sum(np.multiply(mask_flat,
        np.divide(np.abs(resid2_raw), np.add(y_safe, _eps_mape)))),
      nobs
    );

    return {
      x: x_smooth, C: C_smooth,
      xf: fwd.x_pred, Cf: fwd.C_pred,
      yhat, ystd,
      v: v_flat, Cp: Cp_out,
      resid0, resid, resid2,
      ssy, lik, s2, mse, mape, nobs,
    };
  };
  
  // Run core — one jit wrapping both scans + all diagnostics
  const coreResult = await jit(core)(x0, C0, y_arr, V2_arr, FF_scan, G_scan, W_scan, r0, N0);

  // Dispose UD precomputed arrays (no-op if empty)
  for (const a of ud_oneHot) a.dispose();
  for (const a of ud_oneHotCol) a.dispose();
  for (const a of ud_oneHotRow) a.dispose();
  for (const a of ud_gtMask) a.dispose();

  return tree.makeDisposable({
    ...coreResult, m: stateSize, p,
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
  y: ArrayLike<number> | number[][],
  opts: DlmFitOptions,
): Promise<DlmFitResult> => {
  checkUnknownKeys(opts as unknown as Record<string, unknown>, DLM_FIT_KEYS, 'dlmFit');
  if (opts.stabilization) {
    checkUnknownKeys(opts.stabilization as unknown as Record<string, unknown>, DLM_STABILIZATION_KEYS, 'dlmFit (stabilization)');
  }
  const {
    obsStd: s, processStd: w,
    order, harmonics, seasonLength, fullSeasonal, arCoefficients, spline,
    X, algorithm, stabilization,
  } = opts;
  const dtype = parseDtype(opts.dtype);
  const forceAssocScan = algorithm === 'assoc' ? true : algorithm === 'scan' ? false : undefined;
  const forceSqrtAssocScan = algorithm === 'sqrt-assoc' ? true : undefined;
  const forceUdScan = algorithm === 'ud' ? true : undefined;

  // Detect observation dimension p from opts.F (p>1) or default (p=1).
  const obsF: number[][] | undefined = opts.F;  // [p, m_ext] observation matrix
  const obsP = obsF ? obsF.length : 1;          // p: observation dimension

  // Build DlmOptions for dlmGenSys
  const genSysOpts: DlmOptions = {
    order, harmonics, seasonLength, fullSeasonal, arCoefficients, spline,
  };

  // Determine n from y: flat array for p=1, n×p matrix for p>1
  const n = obsP === 1 ? (y as ArrayLike<number>).length : (y as number[][]).length;
  const FA = getFloatArrayType(dtype);

  // Convert y to flat TypedArray (p=1) or keep as number[][] (p>1)
  let yArr: FloatArray;
  let yMat: number[][] | undefined;
  if (obsP === 1) {
    const yFlat = y as ArrayLike<number>;
    yArr = yFlat instanceof FA ? yFlat as FloatArray : FA.from(yFlat) as FloatArray;
  } else {
    yMat = y as number[][];
    // Flatten for the result return
    const flatArr = new FA(n * obsP);
    for (let t = 0; t < n; t++) {
      for (let j = 0; j < obsP; j++) flatArr[t * obsP + j] = yMat[t][j];
    }
    yArr = flatArr as FloatArray;
  }

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
  using W = np.array(W_data, { dtype });

  // ─────────────────────────────────────────────────────────────────────────
  // Build observation matrix F for output and tensor construction.
  // p=1: F = sys.F (1D array from dlmGenSys)
  // p>1: F = opts.F (p×m matrix, user-provided)
  // When covariates present: augmented with X columns per timestep.
  // ─────────────────────────────────────────────────────────────────────────
  const F_out: number[] | number[][] = obsP === 1 ? sys.F : obsF!;
  // F_base: [p, m_base] as nested array
  const F_base_data: number[][] = obsP === 1 ? [sys.F] : obsF!;

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
    const tv = dlmGenSysTV(genSysOpts, timestamps, w);
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
    {
      using G_3d = np.reshape(G, [1, m, m]);
      G_scan = np.tile(G_3d, [n, 1, 1]);
    }
    {
      using W_3d = np.reshape(W, [1, m, m]);
      W_scan = np.tile(W_3d, [n, 1, 1]);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Build pre-shaped tensors for dlmSmo:
  //   y_arr    [n, p, 1]   observations
  //   V2_arr   [n, p, p]   observation noise covariance (diagonal)
  //   FF_scan  [n, p, m]   observation matrix (time-varying when covariates)
  // ─────────────────────────────────────────────────────────────────────────
  let y_arr: np.Array;
  let V2_arr: np.Array;
  let FF_scan_obs: np.Array;

  if (obsP === 1) {
    // Univariate: y [n,1,1], V2 [n,1,1], FF [n,1,m]
    y_arr = np.array(Array.from(yArr!).map(yi => [[yi]]), { dtype });
    // Observation noise: scalar or per-observation
    const V_std_arr = typeof s === 'number'
      ? new FA(n).fill(s)
      : FA.from(s as ArrayLike<number>);
    V2_arr = np.array(Array.from(V_std_arr).map(v => [[v * v]]), { dtype });
    // FF_scan: time-varying (covariates) or static
    if (q > 0 && X) {
      const FF_data: number[][][] = Array.from({ length: n }, (_, t) => [
        [...sys.F, ...Array.from(X[t])]
      ]);
      FF_scan_obs = np.array(FF_data, { dtype });
    } else {
      using F_3d = np.array([[sys.F]], { dtype });       // [1, 1, m_base]
      FF_scan_obs = np.tile(F_3d, [n, 1, 1]);           // [n, 1, m_base]
    }
  } else {
    // Multivariate (p>1): y [n,p,1], V2 [n,p,p], FF [n,p,m]
    y_arr = np.array((yMat!).map(row => row.map(v => [v])), { dtype });  // [n,p,1]
    // Build V2: [n,p,p] diagonal from obsStd
    if (typeof s === 'number') {
      // Scalar s → same variance s² for all components
      const V2_t: number[][] = Array.from({ length: obsP }, (_, i) =>
        Array.from({ length: obsP }, (_, j) => i === j ? s * s : 0));
      const V2_data = Array.from({ length: n }, () => V2_t);
      V2_arr = np.array(V2_data, { dtype });
    } else if (Array.isArray(s) && Array.isArray((s as unknown[])[0])) {
      // number[][] → per-component std dev matrix [p,p], same for all t
      const sMat = s as number[][];
      // Build V2 = sMat * sMat (element-wise squaring for diagonal)
      const V2_t: number[][] = Array.from({ length: obsP }, (_, i) =>
        Array.from({ length: obsP }, (_, j) => sMat[i][j] * sMat[i][j]));
      const V2_data = Array.from({ length: n }, () => V2_t);
      V2_arr = np.array(V2_data, { dtype });
    } else {
      // ArrayLike<number> → per-component std dev (length p), same for all t
      const sArr = Array.from(s as ArrayLike<number>);
      const V2_t: number[][] = Array.from({ length: obsP }, (_, i) =>
        Array.from({ length: obsP }, (_, j) => i === j ? sArr[i] * sArr[i] : 0));
      const V2_data = Array.from({ length: n }, () => V2_t);
      V2_arr = np.array(V2_data, { dtype });
    }
    // FF_scan: p>1 observation matrix [n,p,m]
    // Validate F dimensions
    if (!obsF || obsF.length !== obsP) {
      throw new Error(`opts.F must have ${obsP} rows for p=${obsP}`);
    }
    if (q > 0 && X) {
      // Time-varying: augment each row of F with covariate columns
      const FF_data: number[][][] = Array.from({ length: n }, (_, t) =>
        obsF.map(row => [...row, ...Array.from(X[t])])
      );
      FF_scan_obs = np.array(FF_data, { dtype });
    } else {
      using F_3d = np.array([obsF], { dtype });            // [1, p, m_base]
      FF_scan_obs = np.tile(F_3d, [n, 1, 1]);             // [n, p, m_base]
    }
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
    // For p>1, use mean of first component; for p=1, use scalar
    const v = obsP === 1 ? Number(yArr![i]) : Number(yMat![i][0]);
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
    using out1 = await dlmSmo(y_arr, V2_arr, x0_data, G_scan, W_scan, C0_data, m, obsP, dtype, FF_scan_obs, forceAssocScan, stabilization, forceSqrtAssocScan, forceUdScan);
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
  const out2 = await dlmSmo(y_arr, V2_arr, x0_updated, G_scan, W_scan, C0_scaled, m, obsP, dtype, FF_scan_obs, forceAssocScan, stabilization, forceSqrtAssocScan, forceUdScan);

  FF_scan_obs.dispose();
  y_arr.dispose();
  V2_arr.dispose();
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
    F: F_out,
    W: W_data,
    // Input data
    y: yArr!, obsNoise: obsP === 1
      ? (() => { const a = new FA(n); if (typeof s === 'number') a.fill(s); else for (let i = 0; i < n; i++) a[i] = (s as ArrayLike<number>)[i]; return a; })()
      : new FA(0), // placeholder for p>1 (obsNoise is per-component; full V2 captured in out2)
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
    n, m, p: obsP,
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
  if (opts) {
    checkUnknownKeys(opts as unknown as Record<string, unknown>, DLM_FORECAST_KEYS, 'dlmForecast');
  }
  if (fit.p && fit.p > 1) {
    throw new Error('dlmForecast does not yet support multivariate observations (p > 1)');
  }
  const { G: G_data, W: W_data } = fit;
  // F_data is always number[] when p=1 (p>1 throws above)
  const F_data = fit.F as number[];
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
    p: result.p ?? 1,
    class: 'dlmfit',
  };
};
