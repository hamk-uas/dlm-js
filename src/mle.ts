/* eslint-disable jax-js/no-nested-array-leak */
import { DType, numpy as np, lax, jit, valueAndGrad, hessian as adHessian, tree, defaultDevice } from "@hamk-uas/jax-js-nonconsuming";
import { adam, applyUpdates, type ScaleByAdamOptions } from "@hamk-uas/jax-js-nonconsuming/optax";
import type { DlmFitResult, FloatArray, DlmMleOptions } from "./types";
import { getFloatArrayType, parseDtype, checkUnknownKeys, DLM_MLE_KEYS, DLM_MLE_INIT_KEYS, DLM_MLE_CALLBACKS_KEYS, DLM_MLE_ADAM_KEYS, DLM_MLE_NATURAL_KEYS } from "./types";
import { dlmGenSys, findArInds } from "./dlmgensys";
import type { DlmOptions } from "./dlmgensys";
import { dlmFit } from "./index";

/**
 * Result of converting integer-step timestamped observations to a NaN-stuffed unit grid.
 *
 * `y` is expanded to the full unit grid, missing steps are filled with `NaN`, and
 * optional `X` / `obsStdFixed` inputs are expanded to matching arrays.
 */
export interface StuffedMleInput {
  y: number[];
  timestamps: number[];
  X?: number[][];
  obsStdFixed?: number[];
}

/**
 * Expand integer-step timestamped observations to a NaN-stuffed unit grid.
 *
 * This is intended for workflows such as MLE that operate on the same objective
 * as a NaN-stuffed `dlmFit` run. Timestamp offsets must be positive integers
 * relative to the initial timestamp.
 *
 * @param y - Observed values on the compressed timestamp grid.
 * @param timestamps - Integer-step timestamps aligned with `y`.
 * @param X - Optional covariate rows aligned with `y`.
 * @param obsStdFixed - Optional per-observation fixed std devs aligned with `y`.
 * @returns Expanded arrays on the unit grid, with missing steps filled in.
 */
export function stuffIntegerTimestamps(
  y: ArrayLike<number>,
  timestamps: number[],
  X: ArrayLike<number>[] | undefined,
  obsStdFixed: ArrayLike<number> | undefined,
): StuffedMleInput {
  const n = y.length;
  if (timestamps.length !== n) {
    throw new Error(`stuffIntegerTimestamps: timestamps must have length ${n}, got ${timestamps.length}`);
  }
  if (X && X.length !== n) {
    throw new Error(`stuffIntegerTimestamps: X must have ${n} rows when timestamps are provided, got ${X.length}`);
  }
  if (obsStdFixed && obsStdFixed.length !== n) {
    throw new Error(`stuffIntegerTimestamps: obsStdFixed must have length ${n} when timestamps are provided, got ${obsStdFixed.length}`);
  }
  if (n === 0) return { y: [], timestamps: [] };

  const idx = new Array(n);
  const t0 = timestamps[0];
  idx[0] = 0;
  for (let i = 1; i < n; i++) {
    const dt = timestamps[i] - timestamps[i - 1];
    if (dt <= 0) throw new Error('stuffIntegerTimestamps: timestamps must be strictly increasing');
    const raw = timestamps[i] - t0;
    const rounded = Math.round(raw);
    if (Math.abs(raw - rounded) > 1e-8 * Math.max(1, Math.abs(raw))) {
      throw new Error(
        'stuffIntegerTimestamps: timestamp steps must be positive integers relative to the initial timestamp.'
      );
    }
    idx[i] = rounded;
  }

  const stuffedLength = idx[n - 1] + 1;
  const stuffedY = new Array(stuffedLength).fill(NaN);
  let stuffedX: number[][] | undefined;
  let stuffedObsStd: number[] | undefined;

  if (X) {
    const q = X[0]?.length ?? 0;
    stuffedX = Array.from({ length: stuffedLength }, () => new Array(q).fill(0));
  }
  if (obsStdFixed) {
    stuffedObsStd = new Array(stuffedLength).fill(1.0);
  }

  for (let i = 0; i < n; i++) {
    const j = idx[i];
    stuffedY[j] = Number(y[i]);
    if (stuffedX && X) stuffedX[j] = Array.from(X[i], Number);
    if (stuffedObsStd && obsStdFixed) stuffedObsStd[j] = Number(obsStdFixed[i]);
  }

  return {
    y: stuffedY,
    timestamps: idx,
    ...(stuffedX ? { X: stuffedX } : {}),
    ...(stuffedObsStd ? { obsStdFixed: stuffedObsStd } : {}),
  };
}

/**
 * Result from MLE estimation with JS-idiomatic names.
 */
export interface DlmMleResult {
  /** Estimated observation noise std dev. In MATLAB DLM, this is `s`. */
  obsStd: number;
  /** Estimated state noise std devs (diagonal of √W). In MATLAB DLM, this is `w`. */
  processStd: number[];
  /** Estimated AR coefficients (only when fitAr=true). In MATLAB DLM, this is `arphi`. */
  arCoefficients?: number[];
  /** Deviance: -2 · log-likelihood at optimum. In MATLAB DLM, this is `lik`. */
  deviance: number;
  /** Number of optimizer iterations */
  iterations: number;
  /** Full DLM fit result using the estimated parameters */
  fit: DlmFitResult;
  /** Optimization history: deviance at each iteration. In MATLAB DLM, this is `likHistory`. */
  devianceHistory: number[];
  /** Wall-clock time in ms (total: setup + all iterations + final dlmFit) */
  elapsed: number;
  /** Wall-clock time in ms for the first optimizer step (JIT compilation + one gradient pass). In MATLAB DLM, this is `jitMs`. */
  compilationMs: number;
  /** Prior penalty at the optimum: −2·log p(θ*).  Only present when a custom `loss` function is used. */
  priorPenalty?: number;
}

/**
 * MATLAB DLM-compatible MLE result.
 * Produced by {@link toMatlabMle}.
 */
export interface DlmMleResultMatlab {
  /** Estimated observation noise std dev */
  s: number;
  /** Estimated state noise std devs */
  w: number[];
  /** Estimated AR coefficients */
  arphi?: number[];
  /** -2 · log-likelihood */
  lik: number;
  /** Number of iterations */
  iterations: number;
  /** Full DLM fit result (MATLAB layout) */
  fit: DlmFitResult;
  /** Prior penalty at optimum (only when custom loss) */
  priorPenalty?: number;
  /** Optimization history */
  likHistory: number[];
  /** Wall-clock time in ms */
  elapsed: number;
  /** JIT compilation time in ms */
  jitMs: number;
}

/**
 * Convert a JS-idiomatic DlmMleResult to MATLAB DLM-compatible names.
 */
export const toMatlabMle = (result: DlmMleResult): DlmMleResultMatlab => ({
  s: result.obsStd,
  w: result.processStd,
  arphi: result.arCoefficients,
  lik: result.deviance,
  iterations: result.iterations,
  fit: result.fit,
  priorPenalty: result.priorPenalty,
  likHistory: result.devianceHistory,
  elapsed: result.elapsed,
  jitMs: result.compilationMs,
});

/**
 * Build a diagonal matrix W = diag(w²) from the theta parameter vector
 * in an AD-compatible way using rank-1 updates:
 *   W = Σᵢ w[i]² · eᵢ · eᵢᵀ
 *
 * Each eᵢ is a constant unit vector, and w[i] is extracted from expTheta
 * via a dot-product mask (no np.slice/np.take needed).
 *
 * @internal
 */
const buildDiagW = (
  expTheta: np.Array, m: number, dtype: DType, _nTheta: number,
  /** Index in theta where the w entries start (0 when s is fixed, 1 otherwise). */
  wOffset: number = 1,
): np.Array => {
  // Extract contiguous w entries via structural slice, then diag(w²)
  using wFlat = lax.sliceInDim(expTheta, wOffset, wOffset + m, 0);  // [m]
  using w2 = np.square(wFlat);
  return np.diag(w2);
};

/**
 * Build G matrix with AR coefficients from the theta parameter vector.
 *
 * G_effective = G_base + \u03a3\u1d62 arphi[i] \u00b7 e_{arInds[i]} \u00b7 e_{arCol}\u1d40
 *
 * G_base has the AR column zeroed; this function adds the trainable
 * AR coefficients back using rank-1 updates (AD-compatible).
 *
 * @internal
 */
const buildG = (
  G_base: np.Array, theta: np.Array,
  arInds: number[], m: number, nSwParams: number, _nTheta: number,
  _dtype: DType,
): np.Array => {
  const arCol = arInds[0];
  const nar = arInds.length;

  // Start from G_base (AR column zeroed) and set each AR coefficient entry
  // directly via ND dynamicUpdateSlice — no one-hot algebra needed.
  // jax-js-lint: allow-non-using — accumulator-swap: G updated in loop
  let G = np.add(G_base, np.zerosLike(G_base));  // clone
  for (let i = 0; i < nar; i++) {
    // Extract arphi[i] from theta (NOT exp-transformed) via structural index
    using phi_i = lax.dynamicIndexInDim(theta, nSwParams + i, 0);  // scalar
    using phi_11 = np.reshape(phi_i, [1, 1]);
    // Set G[arInds[i], arCol] = phi_i (base was zeroed, so set = add)
    // jax-js-lint: allow-non-using — accumulator-swap
    const G_new = lax.dynamicUpdateSlice(G, phi_11, [arInds[i], arCol]);
    G.dispose();
    G = G_new;
  }

  return G;
};

/**
 * Build the Kalman filter log-likelihood loss function.
 *
 * Returns a function θ → -2·log L that can be differentiated with grad().
 *
 * The forward filter uses `lax.scan`, which supports autodiff in
 * jax-js-nonconsuming with the following AD-compatibility constraints:
 *
 * - Use `lax.dynamicIndexInDim` / `lax.sliceInDim` for element extraction from theta.
 * - Use matmul chains where each individual matmul has at least one operand
 *   with exactly 1 column (inner dims can be any size).
 *
 * @internal
 */
const makeKalmanLoss = (
  F: np.Array, G_base: np.Array, Ft: np.Array,
  x0: np.Array, C0: np.Array,
  y_arr: np.Array,
  n: number, m: number, dtype: DType,
  arInds: number[] = [],
  /** When provided, V² is taken from this array and s is NOT in theta. */
  fixedV2_arr?: np.Array,
  /** Gradient checkpointing for lax.scan backward pass.
   * - `true` (default): √N segment checkpointing (O(√N) memory, ~2× compute).
   * - `false`: store all N carries (O(N) memory, fastest backward pass).
   * - number: explicit segment size.
   */
  checkpoint?: boolean | number,
  /** Optional per-timestep NaN mask [n, 1, 1]: 1 = observed, 0 = missing.
   * When undefined, all timesteps are treated as observed (default).
   * y_arr must have NaN replaced with 0 before being passed when using this. */
  mask_arr?: np.Array,
) => {
  const nar = arInds.length;
  // When s is fixed, theta = [w₀…w_{m-1}, arphi…]  (no leading s slot)
  const fixS = fixedV2_arr !== undefined;
  const nSwParams = (fixS ? 0 : 1) + m;
  const nTheta = nSwParams + nar;

  return (theta: np.Array): np.Array => {
    // Build effective G: constant if no AR fitting, theta-dependent if fitting
    const G = nar > 0
      ? buildG(G_base, theta, arInds, m, nSwParams, nTheta, dtype)
      : G_base;

    // Step function defined here so it captures the correct G
    type Carry = { x: np.Array; C: np.Array };
    type ScanInp = { y: np.Array; V2: np.Array; W: np.Array; mask: np.Array };

    const step = (carry: Carry, inp: ScanInp): [Carry, np.Array] => {
      const { x, C } = carry;
      const { y: yi, V2: V2i, W, mask: mask_t } = inp;

      // Innovation: v_raw = y - F·x  [1,1]; mask to 0 at missing timesteps
      using v_raw = np.subtract(yi, np.matmul(F, x));
      using v = np.multiply(mask_t, v_raw);

      // C·Fᵀ: [m,m]@[m,1] → [m,1]
      using CFt = np.matmul(C, Ft);

      // Innovation covariance: Cp = F·(C·Fᵀ) + V²  [1,1]
      const Cp = np.add(np.matmul(F, CFt), V2i);

      // Kalman gain: K_raw = G·(C·Fᵀ)/Cp, masked to 0 at missing steps [m,1]
      using K_raw = np.divide(np.matmul(G, CFt), Cp);
      using K = np.multiply(mask_t, K_raw);

      // Next state: x_next = G·x + K·v  [m,1]
      const x_next = np.add(np.matmul(G, x), np.matmul(K, v));

      // L = G - K·F  [m,m]
      using L = np.subtract(G, np.matmul(K, F));
      using Lt = np.transpose(L);

      // Next covariance: C_next = G·(C·Lᵀ) + W  [m,m]
      using CLt = np.matmul(C, Lt);
      const C_next = np.add(np.matmul(G, CLt), W);

      // Per-step -2·loglik: mask · (v²/Cp + log(Cp)) — zero at missing steps
      using lik_raw = np.add(np.divide(np.square(v_raw), Cp), np.log(Cp));
      return [{ x: x_next, C: C_next }, np.multiply(mask_t, np.squeeze(lik_raw))];
    };

    using expTheta = np.exp(theta);

    // V2_arr: either fixed (known per-timestep σ²) or estimated from theta[0]
    let V2_arr: np.Array;
    if (fixS) {
      // fixedV2_arr is a constant captured by the closure — shape [n,1,1]
      V2_arr = fixedV2_arr!;
    } else {
      // s = exp(theta[0]) via dot mask
      using sVal = lax.dynamicIndexInDim(expTheta, 0, 0);  // scalar: s = exp(θ₀)
      using _sVal2 = np.square(sVal);
      using V2 = np.reshape(_sVal2, [1, 1]);
      using _V2_ones = np.ones([n, 1, 1], { dtype });
      using _V2_1 = np.reshape(V2, [1, 1, 1]);
      V2_arr = np.multiply(_V2_ones, _V2_1);
    }

    // W = diag(w²) from theta[0..m-1] (fixS) or theta[1..m] (estimating s)
    using W = buildDiagW(expTheta, m, dtype, nTheta, fixS ? 0 : 1);

    // Broadcast W to [n, ...] for scan
    using _W_ones = np.ones([n, 1, 1], { dtype });
    using _W_1mm = np.reshape(W, [1, m, m]);
    using W_arr = np.multiply(_W_ones, _W_1mm);

    // Build mask for scan: use provided mask or create all-ones (no NaN masking)
    const mask_for_scan = mask_arr ?? np.ones([n, 1, 1], { dtype });
    const ownsMask = mask_arr === undefined;

    const [fc, likTerms] = lax.scan(
      step,
      { x: x0, C: C0 },
      { y: y_arr, V2: V2_arr, W: W_arr, mask: mask_for_scan },
      checkpoint !== undefined ? { checkpoint } : undefined,
    );
    if (ownsMask) mask_for_scan.dispose();
    if (!fixS) V2_arr.dispose();
    tree.dispose(fc);
    const total = np.sum(likTerms);
    likTerms.dispose();
    return total;
  };
};

/**
 * Build a `valueAndGrad`-compatible Kalman log-likelihood loss function
 * using `lax.associativeScan` for the forward filter.
 *
 * Designed for the WebGPU + Float32 path where sequential `lax.scan`
 * dispatches O(n) GPU kernels. The associative scan reduces depth to
 * O(log n) (⌈log₂N⌉+1 Kogge-Stone rounds).
 *
 * Uses the exact parallel forward Kalman filter algebra from Särkkä &
 * García-Fernández (2020), Lemmas 1–2, with 5-tuple scan elements
 * `(A, b, C, eta, J)`.
 *
 * Missing observations (NaN) are handled by masking each timestep to the
 * no-observation transition: `A=G, b=0, C=W, eta=0, J=0`.
 *
 * The first element is initialized exactly from the prior `(x0, C0)`
 * (A₁=0, b₁/C₁ from the one-step update), matching the sequential filter.
 *
 * ## Prediction-error likelihood
 *
 * After the prefix scan recovers filtered states x_filt[t] and
 * covariances C_filt[t], one-step-ahead predictions are obtained by
 * shifting: x_pred[0] = x0, x_pred[t] = G·x_filt[t−1]. The
 * prediction-error decomposition −2 log L = Σ_t mask_t · [v²/Cp + log Cp]
 * matches the sequential `makeKalmanLoss` objective exactly.
 *
 * ## References
 *
 * - Särkkä & García-Fernández (2020), IEEE TAC 66(1), §3 — associative
 *   affine-map composition for the forward filter.
 * - Särkkä & García-Fernández (2020), Lemmas 1–2.
 *
 * @internal
 */
const makeKalmanLossAssoc = (
  F: np.Array,        // [1, m]
  G_base: np.Array,   // [m, m]
  x0: np.Array,       // [m, 1]
  C0: np.Array,       // [m, m]
  y_arr: np.Array,    // [n, 1, 1] — NaN replaced with 0
  n: number,
  m: number,
  dtype: DType,
  arInds: number[] = [],
  fixedV2_arr?: np.Array,  // [n, 1, 1] when s is fixed
  mask_arr?: np.Array,     // [n, 1, 1]: 1=observed, 0=NaN; undefined = all observed
) => {
  const nar = arInds.length;
  const fixS = fixedV2_arr !== undefined;
  const nSwParams = (fixS ? 0 : 1) + m;
  const nTheta = nSwParams + nar;

  type ForwardElem = { A: np.Array; b: np.Array; C: np.Array; eta: np.Array; J: np.Array };

  return (theta: np.Array): np.Array => {
    // Hoist constants outside compose — vmap broadcasts [1,m,m] correctly (jax-js ≥65cb449)
    // Created per-call (once per optimizer iteration) instead of O(N log N) times in compose.
    using _eye = np.eye(m, undefined, { dtype });
    using I1 = np.reshape(_eye, [1, m, m]);
    using inv_eps = np.array(dtype === DType.Float32 ? 1e-6 : 1e-12, { dtype });
    using _inv_eps_3d = np.reshape(inv_eps, [1, 1, 1]);
    using regI = np.multiply(_inv_eps_3d, I1);

    const composeForward = (a: ForwardElem, b_elem: ForwardElem): ForwardElem => {
      using CiJj = np.einsum('nij,njk->nik', a.C, b_elem.J);
      using X_reg = np.add(np.add(I1, CiJj), regI);
      using M = np.linalg.inv(X_reg);

      using AjM = np.einsum('nij,njk->nik', b_elem.A, M);
      const A_comp = np.einsum('nij,njk->nik', AjM, a.A);

      using CiEtaj = np.einsum('nij,njk->nik', a.C, b_elem.eta);
      using bi_plus = np.add(a.b, CiEtaj);
      using AjM_b = np.einsum('nij,njk->nik', AjM, bi_plus);
      const b_comp = np.add(AjM_b, b_elem.b);

      using AjMCi = np.einsum('nij,njk->nik', AjM, a.C);
      using C_tmp = np.einsum('nij,njk->nik', AjMCi, np.einsum('nij->nji', b_elem.A));
      const C_comp = np.add(C_tmp, b_elem.C);

      using MCi = np.einsum('nij,njk->nik', M, a.C);
      using JjMCi = np.einsum('nij,njk->nik', b_elem.J, MCi);
      using N = np.subtract(I1, JjMCi);
      using Jjbi = np.einsum('nij,njk->nik', b_elem.J, a.b);
      using eta_diff = np.subtract(b_elem.eta, Jjbi);
      using N_eta = np.einsum('nij,njk->nik', N, eta_diff);
      using AtNeta = np.einsum('nji,njk->nik', a.A, N_eta);
      const eta_comp = np.add(AtNeta, a.eta);

      using NJ = np.einsum('nij,njk->nik', N, b_elem.J);
      using NJAi = np.einsum('nij,njk->nik', NJ, a.A);
      using AtNJAi = np.einsum('nji,njk->nik', a.A, NJAi);
      const J_comp = np.add(AtNJAi, a.J);

      return { A: A_comp, b: b_comp, C: C_comp, eta: eta_comp, J: J_comp };
    };

    // ─── Parameter extraction (traced) ───
    using expTheta = np.exp(theta);

    // Build effective G (traced when fitting AR coefficients)
    const G = nar > 0
      ? buildG(G_base, theta, arInds, m, nSwParams, nTheta, dtype)
      : G_base;

    // V2 scalar [1, 1] for estimated-s case
    let V2: np.Array | undefined;
    if (!fixS) {
      using sVal = lax.dynamicIndexInDim(expTheta, 0, 0);  // scalar: s = exp(θ₀)
      using _sVal2 = np.square(sVal);
      V2 = np.reshape(_sVal2, [1, 1]);
    }

    // W = diag(w²) [m, m] — traced
    const W = buildDiagW(expTheta, m, dtype, nTheta, fixS ? 0 : 1);

    // ─── Build per-timestep exact forward elements [n, ...] ───
    // Mask: use provided mask or create all-ones
    const mask_n = mask_arr ?? np.ones([n, 1, 1], { dtype });
    const ownsMask = mask_arr === undefined;

    // Per-step V2(t): fixed array or broadcast scalar
    let V2_3d: np.Array | undefined;
    if (!fixS) V2_3d = np.reshape(V2!, [1, 1, 1]);
    const V2_per_step = fixS
      ? fixedV2_arr!
      : np.tile(V2_3d!, [n, 1, 1]);
    V2_3d?.dispose();
    const ownsV2ps = !fixS;

    using y_flat = np.squeeze(y_arr, [2]);                     // [n,1]
    // Boolean NaN mask for np.where blending (observed vs missing-step elements)
    using _zeros_111 = np.zeros([1, 1, 1], { dtype });
    using is_nan_n = np.equal(mask_n, _zeros_111);  // [n,1,1] bool
    using is_nan_mm = np.tile(is_nan_n, [1, m, m]);                     // [n,m,m] bool

    using _G_3d = np.reshape(G, [1, m, m]);
    using G_exp = np.tile(_G_3d, [n, 1, 1]);
    using _W_3d = np.reshape(W, [1, m, m]);
    using W_exp = np.tile(_W_3d, [n, 1, 1]);
    using _F_3d = np.reshape(F, [1, 1, m]);
    using F_exp = np.tile(_F_3d, [n, 1, 1]);
    using Ft = np.transpose(F);
    using _Ft_3d = np.reshape(Ft, [1, m, 1]);
    using Ft_exp = np.tile(_Ft_3d, [n, 1, 1]);
    using _eye_I = np.eye(m, undefined, { dtype });
    using _eye_3d = np.reshape(_eye_I, [1, m, m]);
    using I_exp = np.tile(_eye_3d, [n, 1, 1]);

    // Lemma 1 observed-step terms
    using FW = np.einsum('nij,njk->nik', F_exp, W_exp);        // [n,1,m]
    using FWFt = np.einsum('nij,njk->nik', FW, Ft_exp);        // [n,1,1]
    using S_obs = np.add(FWFt, V2_per_step);                   // [n,1,1]

    using WFt = np.einsum('nij,njk->nik', W_exp, Ft_exp);      // [n,m,1]
    using K_obs_raw = np.divide(WFt, S_obs);                   // [n,m,1]
    using _K_zeros = np.zerosLike(K_obs_raw);
    using K_obs = np.where(is_nan_n, _K_zeros, K_obs_raw); // [n,m,1]

    using KF = np.einsum('nij,njk->nik', K_obs, F_exp);        // [n,m,m]
    using ImKF = np.subtract(I_exp, KF);

    using A_obs = np.einsum('nij,njk->nik', ImKF, G_exp);      // [n,m,m]
    using C_obs = np.einsum('nij,njk->nik', ImKF, W_exp);      // [n,m,m]

    using Ky = np.einsum('nij,nj->ni', K_obs, y_flat);         // [n,m]
    using b_obs = np.expandDims(Ky, 2);                        // [n,m,1]

    using y_col = np.reshape(y_flat, [n, 1, 1]);
    using FG = np.einsum('nij,njk->nik', F_exp, G_exp);        // [n,1,m]
    using FGt = np.einsum('nij->nji', FG);                     // [n,m,1]
    using eta_num = np.multiply(FGt, y_col);                   // [n,m,1]
    using eta_obs_raw = np.divide(eta_num, S_obs);             // [n,m,1]
    using _eta_zeros = np.zerosLike(eta_obs_raw);
    using eta_obs = np.where(is_nan_n, _eta_zeros, eta_obs_raw); // [n,m,1]

    using J_num = np.einsum('nij,njk->nik', FGt, FG);          // [n,m,m]
    using J_obs_raw = np.divide(J_num, S_obs);                 // [n,m,m]
    using _J_zeros = np.zerosLike(J_obs_raw);
    using J_obs = np.where(is_nan_mm, _J_zeros, J_obs_raw); // [n,m,m]

    // Missing-step blend: A=G when NaN, A_obs when observed; etc.
    using A_all = np.where(is_nan_mm, G_exp, A_obs);           // [n,m,m]
    using _b_zeros = np.zerosLike(b_obs);
    using b_all = np.where(is_nan_n, _b_zeros, b_obs); // [n,m,1]
    using C_all = np.where(is_nan_mm, W_exp, C_obs);           // [n,m,m]
    using _eta_all_zeros = np.zerosLike(eta_obs);
    using eta_all = np.where(is_nan_n, _eta_all_zeros, eta_obs); // [n,m,1]
    using _J_all_zeros = np.zerosLike(J_obs);
    using J_all = np.where(is_nan_mm, _J_all_zeros, J_obs); // [n,m,m]

    // First element (k=1): exact initialization from prior
    const F_parts = np.split(F_exp, [1], 0);
    const V2_parts = np.split(V2_per_step, [1], 0);
    const y_parts = np.split(y_arr, [1], 0);
    const mask_parts = np.split(mask_n, [1], 0);
    const A_parts = np.split(A_all, [1], 0);
    const b_parts = np.split(b_all, [1], 0);
    const C_parts = np.split(C_all, [1], 0);
    const eta_parts = np.split(eta_all, [1], 0);
    const J_parts = np.split(J_all, [1], 0);

    using F1 = F_parts[0];
    using V2_1 = V2_parts[0];
    using y1 = y_parts[0];
    using mask1 = mask_parts[0];

    using C0_first = np.reshape(C0, [1, m, m]);
    using x0_first = np.reshape(x0, [1, m, 1]);

    using _F1C0F1t = np.einsum('nij,njk,nlk->nil', F1, C0_first, F1);
    using S1 = np.add(_F1C0F1t, V2_1);
    using C0Ft1 = np.einsum('ij,nkj->nik', C0, F1);
    using K1_obs = np.divide(C0Ft1, S1);
    using K1 = np.multiply(mask1, K1_obs);

    using Fx0_1 = np.einsum('nij,njk->nik', F1, x0_first);
    using innov1 = np.subtract(y1, Fx0_1);
    using Kinnov1 = np.multiply(K1, innov1);
    const b1 = np.add(x0_first, Kinnov1);

    using K1S1 = np.multiply(K1, S1);
    using K1S1K1t = np.einsum('nij,nkj->nik', K1S1, K1);
    const C1 = np.subtract(C0_first, K1S1K1t);

    const A1 = np.zeros([1, m, m], { dtype });
    const eta1 = np.zeros([1, m, 1], { dtype });
    const J1 = np.zeros([1, m, m], { dtype });

    // jax-js-lint: allow-non-using — disposed after scan
    const A_arr = np.concatenate([A1, A_parts[1]], 0);
    // jax-js-lint: allow-non-using — disposed after scan
    const b_arr = np.concatenate([b1, b_parts[1]], 0);
    // jax-js-lint: allow-non-using — disposed after scan
    const C_arr = np.concatenate([C1, C_parts[1]], 0);
    // jax-js-lint: allow-non-using — disposed after scan
    const eta_arr = np.concatenate([eta1, eta_parts[1]], 0);
    // jax-js-lint: allow-non-using — disposed after scan
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

    // ─── Associative prefix scan ───
    const scanned = lax.associativeScan(
      composeForward,
      { A: A_arr, b: b_arr, C: C_arr, eta: eta_arr, J: J_arr },
    ) as ForwardElem;

    // x_filt[t] = scanned.A[t] · x0 + scanned.b[t]  [n, m, 1]
    // Note: np.add creates NEW arrays — x_filt and C_filt are independent of
    // scanned.b / scanned.S, so tree.dispose(scanned) below is safe.
    using _x0_col = np.reshape(x0, [m, 1]);
    using Ax0 = np.matmul(scanned.A, _x0_col);    // [n,m,m]×[m,1] → [n,m,1] (batch broadcast)
    using x_filt = np.add(Ax0, scanned.b);

    // C_filt[t] = scanned.A[t]·C0·scanned.A[t]' + scanned.S[t]  [n, m, m]
    using AC0At = np.einsum('nij,jk,nlk->nil', scanned.A, C0, scanned.A);  // broadcast C0 [m,m]
    using C_filt_raw = np.add(AC0At, scanned.C);
    using C_filt_t = np.einsum('nij->nji', C_filt_raw);
    using C_filt_sum = np.add(C_filt_raw, C_filt_t);
    using _half = np.array(0.5, { dtype });
    using C_filt = np.multiply(_half, C_filt_sum);

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

    // ─── Compute -2·logL from filtered outputs ───
    // Build one-step-ahead predictions for all n timesteps:
    //   x_pred[0] = x0,  x_pred[t] = G·x_filt[t-1]   for t=1..n-1
    //   C_pred[0] = C0,  C_pred[t] = G·C_filt[t-1]·G' + W
    const xfSplit = np.split(x_filt, [n - 1], 0);
    xfSplit[1].dispose();
    using x_filt_head = xfSplit[0];                                 // [n-1, m, 1]

    const cfSplit = np.split(C_filt, [n - 1], 0);
    cfSplit[1].dispose();
    using C_filt_head = cfSplit[0];                                 // [n-1, m, m]

    using x_pred_rest = np.einsum('ij,njk->nik', G, x_filt_head);  // [n-1, m, 1]
    using x0_batch = np.reshape(x0, [1, m, 1]);
    using x_pred = np.concatenate([x0_batch, x_pred_rest], 0);     // [n, m, 1]

    using GCGt = np.einsum('ij,njk,lk->nil', G, C_filt_head, G);   // [n-1, m, m]
    using _W_3d_pred = np.reshape(W, [1, m, m]);
    using C_pred_rest = np.add(GCGt, _W_3d_pred);    // broadcast [1,m,m] → [n-1,m,m]
    using C0_batch = np.reshape(C0, [1, m, m]);
    using C_pred = np.concatenate([C0_batch, C_pred_rest], 0);     // [n, m, m]

    // Innovations and prediction covariance (batched over all n timesteps)
    using Fx_pred = np.einsum('ij,njk->nik', F, x_pred);           // [n, 1, 1]
    using v = np.subtract(y_arr, Fx_pred);                          // [n, 1, 1]
    using FCFt = np.einsum('ij,njk,lk->nil', F, C_pred, F);        // [n, 1, 1]
    using Cp = np.add(FCFt, V2_per_step);                           // [n, 1, 1]

    // -2·logL = Σ mask_t · (v²/Cp + log(Cp))
    using v_sq = np.square(v);
    using v_over_Cp = np.divide(v_sq, Cp);
    using log_Cp = np.log(Cp);
    using lik_raw = np.add(v_over_Cp, log_Cp);
    using lik_masked = np.multiply(mask_n, lik_raw);

    if (ownsMask) mask_n.dispose();
    if (ownsV2ps) V2_per_step.dispose();
    if (!fixS) V2!.dispose();
    W.dispose();
    if (nar > 0) G.dispose();

    return np.sum(lik_masked);
  };
};

// ── Plain-JS linear algebra for the natural gradient optimizer ──────────────
//
// Solve (H + λI) δ = g  for δ  where H is nTheta×nTheta (symmetric),
// g is nTheta×1, and λ is the Levenberg-Marquardt damping.
// Uses Gaussian elimination with partial pivoting.  For nTheta ≤ 10 this
// is exact and trivially fast in plain JS (no GPU needed).

/**
 * Solve `(H + λI) x = b` via Gaussian elimination with partial pivoting.
 * @returns solution vector x
 * @internal
 */
const solveRegularized = (
  H: number[][], b: number[], lambda: number,
): number[] => {
  const n = b.length;
  // Build augmented matrix [H + λI | b]
  const A = H.map((row, i) => {
    const r = [...row];
    r[i] += lambda;
    return [...r, b[i]];
  });

  // Forward elimination with partial pivoting
  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxVal = Math.abs(A[col][col]);
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(A[row][col]) > maxVal) {
        maxVal = Math.abs(A[row][col]);
        maxRow = row;
      }
    }
    if (maxRow !== col) { const tmp = A[col]; A[col] = A[maxRow]; A[maxRow] = tmp; }

    const pivot = A[col][col];
    if (Math.abs(pivot) < 1e-30) {
      // Near-singular: return gradient direction (steepest descent fallback)
      const gNorm = Math.sqrt(b.reduce((s, v) => s + v * v, 0)) || 1;
      return b.map(v => v / gNorm);
    }

    for (let row = col + 1; row < n; row++) {
      const factor = A[row][col] / pivot;
      for (let j = col; j <= n; j++) A[row][j] -= factor * A[col][j];
    }
  }

  // Back substitution
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = A[i][n];
    for (let j = i + 1; j < n; j++) x[i] -= A[i][j] * x[j];
    x[i] /= A[i][i];
  }
  return x;
};

/**
 * Compute the Hessian of a scalar function via central finite differences
 * of the gradient.  Uses the already-JIT'd gradFn to evaluate grad(θ ± h·eⱼ)
 * at 2·nTheta perturbed points — no extra JIT tracing needed.
 *
 * Cost: 2·nTheta gradient evaluations × ~50 ms each (on WASM+f64).
 * For nTheta = 2–6, that's 200–600 ms — far cheaper than JIT-compiling
 * `jacfwd(grad(loss))` which takes ~24 s for the trace alone.
 *
 * Accuracy: O(h²) with per-parameter relative step hⱼ = ε·max(|θⱼ|, 1).
 * Relative stepping keeps the Hessian accurate when parameters drift to
 * extreme log-space values (e.g. w → 0 ⇒ θ → −7).
 * Result is explicitly symmetrized: H → (H + Hᵀ) / 2.
 *
 * @internal
 */
const computeHessianFD = async (
  gradFn: (theta: np.Array) => unknown,
  thetaData: number[],
  nTheta: number,
  dtype: DType,
  fdStepSize: number = 1e-5,
): Promise<number[][]> => {
  const H: number[][] = Array.from({ length: nTheta }, () => new Array(nTheta).fill(0));

  for (let j = 0; j < nTheta; j++) {
    // Relative FD step: scale by |θⱼ| so the perturbation stays meaningful
    // even when parameters drift to extreme log-space values (e.g. θ → −7).
    const hj = fdStepSize * Math.max(Math.abs(thetaData[j]), 1.0);

    // θ + hⱼ·eⱼ
    const thetaPlus = [...thetaData];
    thetaPlus[j] += hj;
    using tpArr = np.array(thetaPlus, { dtype });
    const [lpArr, gpArr] = gradFn(tpArr) as [np.Array, np.Array];
    const gPlus = Array.from(await gpArr.data() as Float64Array | Float32Array);
    lpArr.dispose(); gpArr.dispose();

    // θ - hⱼ·eⱼ
    const thetaMinus = [...thetaData];
    thetaMinus[j] -= hj;
    using tmArr = np.array(thetaMinus, { dtype });
    const [lmArr, gmArr] = gradFn(tmArr) as [np.Array, np.Array];
    const gMinus = Array.from(await gmArr.data() as Float64Array | Float32Array);
    lmArr.dispose(); gmArr.dispose();

    // H[:,j] = (g⁺ - g⁻) / (2hⱼ)
    for (let i = 0; i < nTheta; i++) {
      H[i][j] = (gPlus[i] - gMinus[i]) / (2 * hj);
    }
  }

  // Symmetrize: H ← (H + Hᵀ) / 2
  for (let i = 0; i < nTheta; i++) {
    for (let j = i + 1; j < nTheta; j++) {
      const avg = (H[i][j] + H[j][i]) / 2;
      H[i][j] = avg;
      H[j][i] = avg;
    }
  }
  return H;
};

/**
 * Estimate DLM parameters (obsStd, processStd, and optionally arCoefficients)
 * by maximum likelihood via autodiff.
 *
 * The entire optimization step — `valueAndGrad(loss)` (Kalman filter forward
 * pass + AD backward pass) and optax Adam moment/parameter updates — is
 * wrapped in a single `jit()` call, so every iteration runs from compiled code.
 *
 * The parameterization maps unconstrained reals → positive values:
 *   obsStd = exp(θ_s),  processStd[i] = exp(θ_{w,i})
 * AR coefficients (when `fitAr = true`) are optimized directly
 * (unconstrained — not log-transformed, matching MATLAB DLM behavior).
 *
 * When `obsStdFixed` is supplied (a per-timestep σ array, e.g. known measurement
 * uncertainties), the observation noise is **not estimated** — it is treated as
 * a known constant.  Only processStd (and optionally arCoefficients) are optimized.
 * The returned `obsStd` field will be `NaN` in this case.
 *
 * @param y - Observations (n×1)
 * @param opts - MLE options: model specification, optimizer settings, runtime config
 * @returns MLE result with estimated parameters and full DLM fit
 */
export const dlmMLE = async (
  y: ArrayLike<number>,
  opts?: DlmMleOptions,
): Promise<DlmMleResult> => {
  if (opts) {
    checkUnknownKeys(opts as unknown as Record<string, unknown>, DLM_MLE_KEYS, 'dlmMLE');
    if (opts.init) checkUnknownKeys(opts.init as unknown as Record<string, unknown>, DLM_MLE_INIT_KEYS, 'dlmMLE (init)');
    if (opts.callbacks) checkUnknownKeys(opts.callbacks as unknown as Record<string, unknown>, DLM_MLE_CALLBACKS_KEYS, 'dlmMLE (callbacks)');
    if (opts.adamOpts) checkUnknownKeys(opts.adamOpts as unknown as Record<string, unknown>, DLM_MLE_ADAM_KEYS, 'dlmMLE (adamOpts)');
    if (opts.naturalOpts) checkUnknownKeys(opts.naturalOpts as unknown as Record<string, unknown>, DLM_MLE_NATURAL_KEYS, 'dlmMLE (naturalOpts)');
  }
  const {
    order, harmonics, seasonLength, fullSeasonal, arCoefficients, fitAr,
    X, init,
    obsStdFixed: sFixed, callbacks, adamOpts,
    algorithm, optimizer: optimizerChoice, naturalOpts,
    loss: lossOption, checkpoint: checkpointOpt,
  } = opts ?? {};
  const dtype = parseDtype(opts?.dtype);
  // Default optimizer: natural for f64 (fast second-order), Adam for f32 (FD Hessian too noisy)
  const useNatural = optimizerChoice === 'natural' ||
    (optimizerChoice === undefined && dtype === DType.Float64);
  const hessianMode = naturalOpts?.hessian ?? 'fd';
  const lmInit = naturalOpts?.lambdaInit ?? 0.1;
  const lmShrink = naturalOpts?.lambdaShrink ?? 0.5;
  const lmGrow = naturalOpts?.lambdaGrow ?? 2;
  const fdStep = naturalOpts?.fdStep ?? 1e-5;
  // Default lr: 0.05 for Adam, 1.0 for Newton (full step)
  const lr = opts?.lr ?? (useNatural ? 1.0 : 0.05);
  const maxIter = opts?.maxIter ?? (useNatural ? 50 : 200);
  const tol = opts?.tol ?? 1e-6;
  const forceAssocScan = algorithm === 'assoc' ? true : undefined;
  const options: DlmOptions = { order, harmonics, seasonLength, fullSeasonal, arCoefficients, fitAr };
  const t0 = performance.now();
  const FA = getFloatArrayType(dtype);
  const yOrig = y instanceof FA ? y as FloatArray : FA.from(y);
  const mleSFixed = sFixed;
  const yArr = FA.from(Array.from(yOrig));
  const n = yArr.length;

  // Generate system matrices
  const sys = dlmGenSys(options);
  const m = sys.m;

  // AR fitting setup
  const arphi_orig = arCoefficients ?? [];
  const doFitAr = !!(fitAr && arphi_orig.length > 0);
  const arInds = doFitAr ? findArInds(options) : [];
  const nar = arInds.length;

  // Build G: if fitting AR, zero the AR column (those values come from theta)
  let G_data = sys.G;
  if (nar > 0) {
    G_data = sys.G.map(row => [...row]);
    const arCol = arInds[0];
    for (const idx of arInds) G_data[idx][arCol] = 0;
  }

  // System matrices (constants — captured by closure, not differentiated)
  using G = np.array(G_data, { dtype });
  using F = np.array([sys.F], { dtype }); // [1, m]
  using Ft = np.transpose(F);

  // Detect missing observations (NaN) and build a mask [n, 1, 1]: 1 = observed, 0 = NaN
  const yList = Array.from(yArr);
  const hasMissing = yList.some(v => isNaN(v));
  const mleMaskArr = hasMissing
    ? np.array(yList.map(yi => [[isNaN(yi) ? 0 : 1]]), { dtype })
    : undefined;

  // Stack observations: [n, 1, 1] — NaN replaced with 0 (masked out in scan step)
  using y_arr = np.array(yList.map(yi => [[isNaN(yi) ? 0 : yi]]), { dtype });

  // Initial state — initialised from data mean/variance, NaN-safe
  const ns = seasonLength ?? 12;
  let initSum = 0;
  let initCount = 0;
  const nsActual = Math.min(ns, n);
  for (let i = 0; i < nsActual; i++) {
    const vi = Number(yArr[i]);
    if (!isNaN(vi)) { initSum += vi; initCount++; }
  }
  const mean_y = initCount > 0 ? initSum / initCount : 0;
  const c0_val = (Math.abs(mean_y) * 0.5) ** 2;
  const c0 = c0_val === 0 ? 1e7 : c0_val;
  const x0_data: number[][] = Array.from({ length: m }, (_, i) =>
    [i === 0 ? mean_y : 0.0]
  );
  const C0_data: number[][] = Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => (i === j ? c0 : 0.0))
  );
  using x0 = np.array(x0_data, { dtype });
  using C0 = np.array(C0_data, { dtype });

  // Initial parameter guess (NaN-safe variance)
  const yObs = Array.from(yArr).filter(v => !isNaN(v));
  const nObs = yObs.length || 1;
  const variance = yObs.reduce((s, v) => s + v * v, 0) / nObs
    - (yObs.reduce((s, v) => s + v, 0) / nObs) ** 2;
  const s_init = init?.obsStd ?? (Math.sqrt(Math.abs(variance)) || 1.0);
  const w_init = init?.processStd ?? new Array(m).fill(s_init * 0.1);
  const arphi_init = init?.arCoefficients ?? arphi_orig;

  // Build fixed V2_arr when sFixed is provided
  const fixS = sFixed !== undefined;
  let fixedV2_arr: np.Array | undefined;
  if (fixS) {
    // V2_t = sFixed[t]² — shape [n, 1, 1]
    const v2data = Array.from(mleSFixed!).map(si => [[si * si]]);
    fixedV2_arr = np.array(v2data, { dtype });
  }

  // theta = [log(s)?, log(w0).., log(w_{m-1}), arphi..]
  const theta_init = [
    ...(fixS ? [] : [Math.log(s_init)]),
    ...w_init.map(wi => Math.log(Math.abs(wi) || 0.01)),
    ...(doFitAr ? arphi_init : []),  // unconstrained (not log-transformed); only when fitting AR
  ];

  // Build loss & JIT the entire optimization step:
  //   (theta, optState) → (newTheta, newOptState, likValue)
  // One jit() wrapping: valueAndGrad (Kalman scan + AD) + optax Adam update.
  // Traces once, then every iteration is compiled.

  // Device/dtype dispatch: WebGPU + Float32 uses associativeScan-based loss
  // (O(log n) depth), otherwise sequential lax.scan (O(n) depth).
  // forceAssocScan overrides to enable benchmarking the parallel path on any backend.
  const useAssocScanLoss = forceAssocScan || (defaultDevice() === 'webgpu' && dtype === DType.Float32);

  // checkpoint: false stores all N carries — no recomputation on backward pass.
  // Benchmarks show ~25–30% speedup over default √N checkpointing for typical
  // DLM dataset sizes (n ≲ few hundred), where carry memory is negligible.
  const kalmanLoss = useAssocScanLoss
    ? makeKalmanLossAssoc(F, G, x0, C0, y_arr, n, m, dtype, arInds, fixedV2_arr, mleMaskArr)
    : makeKalmanLoss(F, G, Ft, x0, C0, y_arr, n, m, dtype, arInds, fixedV2_arr, checkpointOpt ?? false, mleMaskArr);

  // Custom loss wrapping (MAP / regularised objectives).
  // When a DlmLossFn is provided, it wraps the Kalman −2·logL so that
  //   objectiveFn(θ) = userLoss(deviance(θ), naturalParams(θ), meta)
  // Natural-scale params: exp(θ) for variance slots, raw θ for AR slots.
  const hasCustomLoss = typeof lossOption === 'function';
  const wOffset = fixS ? 0 : 1;
  const nSwParams = wOffset + m; // number of variance params (s? + w's)
  let paramMeta: import('./types').DlmParamMeta | undefined;
  if (hasCustomLoss) {
    paramMeta = { nObs: fixS ? 0 : 1, nProcess: m, nAr: nar };
  }
  const lossFn = hasCustomLoss
    ? (theta: np.Array): np.Array => {
        using kl = kalmanLoss(theta);
        // Compute natural-scale params: exp for variances, raw for AR
        let params: np.Array;
        if (nar > 0) {
          const parts = np.split(theta, [nSwParams], 0);
          using varPart = np.exp(parts[0]);
          parts[0].dispose();
          params = np.concatenate([varPart, parts[1]], 0);
          parts[1].dispose();
        } else {
          params = np.exp(theta);
        }
        return (lossOption as import('./types').DlmLossFn)(kl, params, paramMeta!);
      }
    : kalmanLoss;

  // ════════════════════════════════════════════════════════════════════════════
  // Natural gradient / Levenberg-Marquardt optimizer path
  // ════════════════════════════════════════════════════════════════════════════
  //
  // Solves  Δθ = −η · (H + λI)⁻¹ · g  where H is the Hessian of the
  // Kalman −2·logL (≈ observed Fisher information on log-scale params)
  // computed via central finite differences of the gradient, and g is
  // the gradient.  This is Amari's natural gradient (1998) with
  // Levenberg-Marquardt damping (Marquardt 1963).
  //
  // LM damping adapts λ: shrink when loss decreases (trust the quadratic
  // model), grow when it increases (fall back toward gradient descent).
  // For nTheta ≤ ~10 the (nTheta×nTheta) solve is done in plain JS —
  // effectively free.
  //
  // Cost per iteration:
  //   hessian='fd' (default): one valueAndGrad + 2·nTheta perturbed gradient
  //     evals (all using the same JIT-compiled gradFn — no extra trace).
  //   hessian='exact': one valueAndGrad + one jit(hessian(lossFn)) call
  //     (exact AD; first call incurs a large JIT trace, ~20 s on WASM v0.7.8).
  // Convergence is quadratic near the optimum: 3–15 steps vs 50–200 for Adam.
  //
  if (useNatural) {
    const nTheta = theta_init.length;
    // JIT the valueAndGrad call (only one trace, ~300 ms).
    const gradFn = jit(valueAndGrad(lossFn));
    // Exact Hessian (lazy): only JIT-traced when hessianMode === 'exact'.
    // The first call takes ~20 s (jax-js v0.7.8) but warm calls are fast.
    const exactHessFn = hessianMode === 'exact' ? jit(adHessian(lossFn)) : undefined;

    let theta = np.array(theta_init, { dtype });

    // Notify callback with initial theta
    if (callbacks?.onInit) {
      const initData = await theta.data() as Float64Array | Float32Array;
      callbacks.onInit(initData);
    }

    const likHistory: number[] = [];
    let prevLik = Infinity;
    let lambda = -1;  // sentinel: will be set from Hessian diagonal on first iteration
    let jitMs = 0;
    let iter = 0;

    for (iter = 0; iter < maxIter; iter++) {
      const tStep = iter === 0 ? performance.now() : NaN;

      // ── 1. Evaluate loss + gradient at current θ ──
      const [likArr, gradArr] = gradFn(theta) as [np.Array, np.Array];
      const likNum = (await likArr.data() as Float64Array | Float32Array)[0];
      const g = Array.from(await gradArr.data() as Float64Array | Float32Array);
      likArr.dispose(); gradArr.dispose();

      if (iter === 0) jitMs = Math.round(performance.now() - tStep);
      likHistory.push(likNum);

      // Read thetaData for FD Hessian and line search
      const thetaData = Array.from(await theta.data() as Float64Array | Float32Array);

      // Notify callback
      if (callbacks?.onIteration) {
        const td = await theta.data() as Float64Array | Float32Array;
        callbacks.onIteration(iter + 1, td, likNum);
      }

      // ── 2. Check convergence ──
      // Require BOTH small relative loss change AND small gradient norm.
      // Loss-only checks are fooled by plateaus (the optimizer may be far from
      // the optimum but moving along a flat ridge).
      if (iter > 0) {
        const relChange = Math.abs((likNum - prevLik) / (Math.abs(prevLik) + 1e-30));
        const gNorm = Math.sqrt(g.reduce((s, v) => s + v * v, 0));
        if (relChange < tol && gNorm < 1.0) { prevLik = likNum; iter++; break; }
      }
      prevLik = likNum;

      // ── 3. Compute Hessian ──
      //   'fd' (default): 2·nTheta perturbed gradient evaluations using the
      //     same JIT'd gradFn.  For nTheta ≤ 10, ~50 ms × 2·nTheta.
      //   'exact': single call to jit(hessian(lossFn)).  Accurate, but first
      //     call incurs a large JIT trace (~20 s on WASM, jax-js v0.7.8).
      let H: number[][];
      if (exactHessFn) {
        const hessArr = exactHessFn(theta) as np.Array;
        const hessData = await hessArr.data() as Float64Array | Float32Array;
        H = Array.from({ length: nTheta }, (_, i) =>
          Array.from({ length: nTheta }, (_, j) => hessData[i * nTheta + j])
        );
        hessArr.dispose();
      } else {
        H = await computeHessianFD(gradFn, thetaData, nTheta, dtype, fdStep);
      }

      // Marquardt initialization: λ = τ · max(diag(H))
      // Start close to gradient descent (large λ) when far from optimum.
      if (lambda < 0) {
        let maxDiag = 0;
        for (let i = 0; i < nTheta; i++) maxDiag = Math.max(maxDiag, Math.abs(H[i][i]));
        lambda = Math.max(maxDiag * lmInit, 1e-6);
      }

      // ── 4. Solve (H + λI) δ = g  via Gaussian elimination ──
      //   For nTheta ≤ 10 this is trivially cheap in plain JS.
      const delta = solveRegularized(H, g, lambda);

      // ── 5. Backtracking line search with LM damping fallback ──
      //   Try the Newton step; if loss doesn't decrease, increase λ and
      //   re-solve up to 5 times (damps toward gradient descent).
      let accepted = false;
      for (let attempt = 0; attempt < 6; attempt++) {
        const d = attempt === 0 ? delta : solveRegularized(H, g, lambda);
        const thetaNew = thetaData.map((t, i) => t - lr * d[i]);
        using tNewArr = np.array(thetaNew, { dtype });
        const [lNewArr, gNewArr] = gradFn(tNewArr) as [np.Array, np.Array];
        const lNew = (await lNewArr.data() as Float64Array | Float32Array)[0];
        lNewArr.dispose(); gNewArr.dispose();

        if (lNew < likNum) {
          // Good step — accept and reduce damping
          theta.dispose();
          theta = np.array(thetaNew, { dtype });
          lambda = Math.max(lambda * lmShrink, 1e-10);
          accepted = true;
          break;
        }
        // Increase damping for next attempt
        lambda *= lmGrow;
      }

      if (!accepted) {
        // All attempts failed — take a small gradient descent step
        const gNorm = Math.sqrt(g.reduce((s, v) => s + v * v, 0)) || 1;
        const thetaGD = thetaData.map((t, i) => t - 0.01 * g[i] / gNorm);
        theta.dispose();
        theta = np.array(thetaGD, { dtype });
      }
    }

    // ── Extract results ──
    // When a custom loss is used, prevLik is the MAP objective.
    // Evaluate pure Kalman −2·logL at the optimum to separate prior penalty.
    let pureDeviance = prevLik;
    let priorPenalty: number | undefined;
    if (hasCustomLoss) {
      using devArr = kalmanLoss(theta);
      pureDeviance = (await devArr.data() as Float64Array | Float32Array)[0];
      priorPenalty = prevLik - pureDeviance;
    }

    const thetaData = await theta.data() as Float64Array | Float32Array;
    theta.dispose();
    gradFn.dispose();
    exactHessFn?.dispose();
    fixedV2_arr?.dispose();
    mleMaskArr?.dispose();

    const wOff = fixS ? 0 : 1;
    const s_opt = fixS ? NaN : Math.exp(thetaData[0]);
    const w_opt = Array.from({ length: m }, (_, i) => Math.exp(thetaData[wOff + i]));
    const arphi_opt = nar > 0
      ? Array.from({ length: nar }, (_, i) => thetaData[wOff + m + i])
      : undefined;

    const { fitAr: _fa1, ...fitOpts1 } = arphi_opt ? { ...options, arCoefficients: arphi_opt } : options;
    const sForFit: number | ArrayLike<number> = fixS ? sFixed! : s_opt;
    const fit = await dlmFit(yOrig, {
      obsStd: sForFit, processStd: w_opt, ...fitOpts1,
      X, dtype: opts?.dtype, algorithm: opts?.algorithm,
    });

    const elapsed = performance.now() - t0;
    return {
      obsStd: s_opt, processStd: w_opt, arCoefficients: arphi_opt,
      deviance: pureDeviance, iterations: iter, fit,
      devianceHistory: likHistory, elapsed, compilationMs: jitMs,
      ...(priorPenalty !== undefined ? { priorPenalty } : {}),
    };
  }

  // ════════════════════════════════════════════════════════════════════════════
  // Adam optimizer path (default)
  // ════════════════════════════════════════════════════════════════════════════
  const optimizer = adam(lr, { b2: 0.9, ...adamOpts });

  // ── On-device batched training loop ──
  // Instead of returning to JS after every step, run INNER_STEPS optimizer
  // iterations on-device via lax.scan before reading back the deviance.
  // The scan carries {theta, optState, lastLik} and returns null outputs
  // (skip stacking).
  //
  // Architecture:
  //   jit((theta, optState) =>
  //     lax.scan(step, {theta, optState, lastLik}, null, {length: INNER_STEPS})
  //   )
  // Each on-device block runs INNER_STEPS forward+backward+Adam passes
  // compiled into a single native program. Loss and convergence are checked
  // only at block boundaries (every INNER_STEPS iters).
  // INNER_STEPS: number of Adam iterations per on-device block.
  // For the sequential loss path we batch 10 steps via lax.scan to minimise
  // JS↔Wasm round-trips.  For the associative-scan loss path we cannot nest
  // an outer lax.scan around an inner lax.associativeScan (causes
  // UseAfterFreeError during JIT _inlineLiterals), so we fall back to a
  // single-step JIT compiled block (INNER_STEPS=1).
  const INNER_STEPS = useAssocScanLoss ? 1 : 10;

  type OptCarry = { theta: np.Array; optState: any; lastLik: np.Array };

  // Used only by the sequential-scan optimBlock below.
  const innerStep = (carry: OptCarry, _x: null): [OptCarry, null] => {
    const [likVal, grad] = valueAndGrad(lossFn)(carry.theta);
    const [updates, newOptState] = optimizer.update(grad, carry.optState);
    const newTheta = applyUpdates(carry.theta, updates);
    return [{ theta: newTheta, optState: newOptState, lastLik: likVal }, null];
  };

  const optimBlock = useAssocScanLoss
    ? jit((theta: np.Array, optState: any, _lastLik: np.Array) => {
        // Single Adam step — avoids nesting lax.scan around lax.associativeScan.
        const [likVal, grad] = valueAndGrad(lossFn)(theta);
        const [updates, newOptState] = optimizer.update(grad, optState);
        const newTheta = applyUpdates(theta, updates);
        return { theta: newTheta, optState: newOptState, lastLik: likVal } as OptCarry;
      })
    : jit((theta: np.Array, optState: any, lastLik: np.Array) => {
        const [finalCarry, _ys] = lax.scan(
          innerStep,
          { theta, optState, lastLik } as OptCarry,
          null,
          { length: INNER_STEPS },
        );
        return finalCarry;
      });

  // Initialize
  let theta = np.array(theta_init, { dtype });
  let optState: any = optimizer.init(theta);
  let lastLik = np.array(Infinity, { dtype });

  // Notify callback with initial theta
  if (callbacks?.onInit) {
    const initData = await theta.data() as Float64Array | Float32Array;
    callbacks.onInit(initData);
  }

  const likHistory: number[] = [];
  let prevLik = Infinity;
  let iter = 0;
  let patienceCount = 0;
  let jitMs = 0;
  // Each block runs INNER_STEPS iterations on-device, so a single converged
  // block already represents 10 consecutive near-zero-change steps. PATIENCE=2
  // blocks (= 20 steps) is more than the previous per-step PATIENCE=5.
  const PATIENCE = 2;
  const nBlocks = Math.ceil(maxIter / INNER_STEPS);

  for (let block = 0; block < nBlocks; block++) {
    const t0block = block === 0 ? performance.now() : NaN;
    const result = optimBlock(theta, optState, lastLik) as OptCarry;
    const likNum = (await result.lastLik.data() as Float64Array | Float32Array)[0];
    if (block === 0) jitMs = Math.round(performance.now() - t0block);
    likHistory.push(likNum);

    // Notify callback with updated theta (at block boundary)
    if (callbacks?.onIteration) {
      const td = await result.theta.data() as Float64Array | Float32Array;
      const iterNum = (block + 1) * INNER_STEPS;
      callbacks.onIteration(iterNum, td, likNum);
    }

    // Dispose old state
    theta.dispose(); tree.dispose(optState); lastLik.dispose();
    theta = result.theta; optState = result.optState; lastLik = result.lastLik;

    iter = (block + 1) * INNER_STEPS;

    // Check convergence: require PATIENCE consecutive blocks below tol
    const relChange = Math.abs((likNum - prevLik) / (Math.abs(prevLik) + 1e-30));
    if (block > 0) {
      if (relChange < tol) {
        patienceCount++;
        if (patienceCount >= PATIENCE) {
          prevLik = likNum;
          break;
        }
      } else {
        patienceCount = 0;
      }
    }
    prevLik = likNum;
  }

  // Extract optimized parameters
  // When a custom loss is used, prevLik is the MAP objective.
  // Evaluate pure Kalman −2·logL at the optimum to separate prior penalty.
  let pureDeviance = prevLik;
  let priorPenalty: number | undefined;
  if (hasCustomLoss) {
    using devArr = kalmanLoss(theta);
    pureDeviance = (await devArr.data() as Float64Array | Float32Array)[0];
    priorPenalty = prevLik - pureDeviance;
  }

  const thetaData = await theta.data() as Float64Array | Float32Array;
  theta.dispose(); tree.dispose(optState); lastLik.dispose();
  fixedV2_arr?.dispose();
  mleMaskArr?.dispose();
  optimBlock.dispose();

  const wOff = fixS ? 0 : 1;
  const s_opt = fixS ? NaN : Math.exp(thetaData[0]);
  const w_opt = Array.from({ length: m }, (_, i) => Math.exp(thetaData[wOff + i]));
  const arphi_opt = nar > 0
    ? Array.from({ length: nar }, (_, i) => thetaData[wOff + m + i])
    : undefined;

  // Run full dlmFit with optimized parameters (including fitted arCoefficients if applicable)
  const { fitAr: _fa2, ...fitOpts2 } = arphi_opt ? { ...options, arCoefficients: arphi_opt } : options;
  // When s was fixed, pass the original sFixed array to dlmFit; otherwise use scalar s_opt
  const sForFit: number | ArrayLike<number> = fixS ? sFixed! : s_opt;
  const fit = await dlmFit(yOrig, {
    obsStd: sForFit,
    processStd: w_opt,
    ...fitOpts2,
    X,
    dtype: opts?.dtype,
    algorithm: opts?.algorithm,
  });

  const elapsed = performance.now() - t0;

  return {
    obsStd: s_opt,
    processStd: w_opt,
    arCoefficients: arphi_opt,
    deviance: pureDeviance,
    iterations: iter,
    fit,
    devianceHistory: likHistory,
    elapsed,
    compilationMs: jitMs,
    ...(priorPenalty !== undefined ? { priorPenalty } : {}),
  };
};
