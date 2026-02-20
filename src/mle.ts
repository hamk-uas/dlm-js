import { DType, numpy as np, lax, jit, valueAndGrad, tree, defaultDevice } from "@hamk-uas/jax-js-nonconsuming";
import { adam, applyUpdates, type ScaleByAdamOptions } from "@hamk-uas/jax-js-nonconsuming/optax";
import type { DlmFitResult, FloatArray } from "./types";
import { getFloatArrayType, adSafeInv } from "./types";
import { dlmGenSys, findArInds } from "./dlmgensys";
import type { DlmOptions } from "./dlmgensys";
import { dlmFit } from "./index";

/**
 * Result from MLE estimation.
 */
export interface DlmMleResult {
  /** Estimated observation noise std dev */
  s: number;
  /** Estimated state noise std devs (diagonal of sqrt(W)) */
  w: number[];  /** Estimated AR coefficients (only when fitar=true) */
  arphi?: number[];  /** -2 · log-likelihood at optimum */
  lik: number;
  /** Number of optimizer iterations */
  iterations: number;
  /** Full DLM fit result using the estimated parameters */
  fit: DlmFitResult;
  /** Optimization history: lik at each iteration */
  likHistory: number[];
  /** Wall-clock time in ms */
  elapsed: number;
}

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
  expTheta: np.Array, m: number, dtype: DType, nTheta: number,
  /** Index in theta where the w entries start (0 when s is fixed, 1 otherwise). */
  wOffset: number = 1,
): np.Array => {
  let W = np.zeros([m, m], { dtype });
  for (let i = 0; i < m; i++) {
    const maskData = new Array(nTheta).fill(0);
    maskData[wOffset + i] = 1;
    using mask = np.array(maskData, { dtype });
    using wi = np.dot(expTheta, mask);
    using wi2 = np.square(wi);

    const eiData = Array.from({ length: m }, (_, j) => j === i ? [1] : [0]);
    using ei = np.array(eiData, { dtype });
    using eit = np.transpose(ei);
    using outer = np.matmul(ei, eit);
    using scaled = np.multiply(np.reshape(wi2, [1, 1]), outer);
    // Accumulator pattern: transfer ownership from old W to new W
    // jax-js-lint: allow-non-using
    const W_new = np.add(W, scaled);
    W.dispose();
    W = W_new;
  }
  return W;
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
  arInds: number[], m: number, nSwParams: number, nTheta: number,
  dtype: DType,
): np.Array => {
  const arCol = arInds[0];
  const nar = arInds.length;

  // Build AR contribution as a sum of rank-1 updates
  let arContrib = np.zeros([m, m], { dtype });
  for (let i = 0; i < nar; i++) {
    // Extract arphi[i] from theta (NOT exp-transformed) via mask
    const maskData = new Array(nTheta).fill(0);
    maskData[nSwParams + i] = 1;
    using mask = np.array(maskData, { dtype });
    using phi_i = np.dot(theta, mask);

    // Rank-1 update at G[arInds[i], arCol]
    const eiData = Array.from({ length: m }, (_, j) => j === arInds[i] ? [1] : [0]);
    const ejData = Array.from({ length: m }, (_, j) => [j === arCol ? 1 : 0]);
    using ei = np.array(eiData, { dtype });     // [m, 1]
    using ejt = np.transpose(np.array(ejData, { dtype })); // [1, m]
    using outer = np.matmul(ei, ejt);            // [m, m]
    using scaled = np.multiply(np.reshape(phi_i, [1, 1]), outer);
    // jax-js-lint: allow-non-using
    const newContrib = np.add(arContrib, scaled);
    arContrib.dispose();
    arContrib = newContrib;
  }

  // G = G_base + arContrib
  const G = np.add(G_base, arContrib);
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
 * - Use `np.dot(vector, mask)` for element extraction from theta.
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
      using mask_s = np.array([1, ...new Array(nTheta - 1).fill(0)], { dtype });
      using sVal = np.dot(expTheta, mask_s);
      using V2 = np.reshape(np.square(sVal), [1, 1]);
      V2_arr = np.multiply(
        np.ones([n, 1, 1], { dtype }),
        np.reshape(V2, [1, 1, 1]),
      );
    }

    // W = diag(w²) from theta[0..m-1] (fixS) or theta[1..m] (estimating s)
    using W = buildDiagW(expTheta, m, dtype, nTheta, fixS ? 0 : 1);

    // Broadcast W to [n, ...] for scan
    using W_arr = np.multiply(
      np.ones([n, 1, 1], { dtype }),
      np.reshape(W, [1, m, m]),
    );

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

  const composeForward = (a: ForwardElem, b_elem: ForwardElem): ForwardElem => {
    using I1 = np.reshape(np.eye(m, undefined, { dtype }), [1, m, m]);
    using inv_eps = np.array(dtype === DType.Float32 ? 1e-6 : 1e-12, { dtype });
    using regI = np.multiply(np.reshape(inv_eps, [1, 1, 1]), I1);
    using CiJj = np.einsum('nij,njk->nik', a.C, b_elem.J);
    using X_reg = np.add(np.add(I1, CiJj), regI);
    using M = adSafeInv(X_reg, m, dtype);

    using AjM = np.einsum('nij,njk->nik', b_elem.A, M);
    const A_comp = np.einsum('nij,njk->nik', AjM, a.A);

    using CiEtaj = np.einsum('nij,njk->nik', a.C, b_elem.eta);
    using bi_plus = np.add(a.b, CiEtaj);
    using AjM_b = np.einsum('nij,njk->nik', AjM, bi_plus);
    const b_comp = np.add(AjM_b, b_elem.b);

    using AjMCi = np.einsum('nij,njk->nik', AjM, a.C);
    using C_tmp = np.einsum('nij,njk->nik', AjMCi, np.einsum('nij->nji', b_elem.A));
    const C_comp = np.add(C_tmp, b_elem.C);

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

    using NJ = np.einsum('nij,njk->nik', N, b_elem.J);
    using NJAi = np.einsum('nij,njk->nik', NJ, a.A);
    using AtNJAi = np.einsum('nji,njk->nik', a.A, NJAi);
    const J_comp = np.add(AtNJAi, a.J);

    return { A: A_comp, b: b_comp, C: C_comp, eta: eta_comp, J: J_comp };
  };

  return (theta: np.Array): np.Array => {
    // ─── Parameter extraction (traced) ───
    using expTheta = np.exp(theta);

    // Build effective G (traced when fitting AR coefficients)
    const G = nar > 0
      ? buildG(G_base, theta, arInds, m, nSwParams, nTheta, dtype)
      : G_base;

    // V2 scalar [1, 1] for estimated-s case
    let V2: np.Array | undefined;
    if (!fixS) {
      using mask_s = np.array([1, ...new Array(nTheta - 1).fill(0)], { dtype });
      using sVal = np.dot(expTheta, mask_s);
      V2 = np.reshape(np.square(sVal), [1, 1]);
    }

    // W = diag(w²) [m, m] — traced
    const W = buildDiagW(expTheta, m, dtype, nTheta, fixS ? 0 : 1);

    // ─── Build per-timestep exact forward elements [n, ...] ───
    // Mask: use provided mask or create all-ones
    const mask_n = mask_arr ?? np.ones([n, 1, 1], { dtype });
    const ownsMask = mask_arr === undefined;

    // Per-step V2(t): fixed array or broadcast scalar
    const V2_per_step = fixS
      ? fixedV2_arr!
      : np.tile(np.reshape(V2!, [1, 1, 1]), [n, 1, 1]);
    const ownsV2ps = !fixS;

    using y_flat = np.squeeze(y_arr, [2]);                     // [n,1]
    using mask_mm = np.tile(mask_n, [1, m, m]);               // [n,m,m]
    using one_mm = np.onesLike(mask_mm);
    using inv_mask_mm = np.subtract(one_mm, mask_mm);

    using G_exp = np.tile(np.reshape(G, [1, m, m]), [n, 1, 1]);
    using W_exp = np.tile(np.reshape(W, [1, m, m]), [n, 1, 1]);
    using F_exp = np.tile(np.reshape(F, [1, 1, m]), [n, 1, 1]);
    using Ft = np.transpose(F);
    using Ft_exp = np.tile(np.reshape(Ft, [1, m, 1]), [n, 1, 1]);
    using I_exp = np.tile(np.reshape(np.eye(m, undefined, { dtype }), [1, m, m]), [n, 1, 1]);

    // Lemma 1 observed-step terms
    using FW = np.einsum('nij,njk->nik', F_exp, W_exp);        // [n,1,m]
    using FWFt = np.einsum('nij,njk->nik', FW, Ft_exp);        // [n,1,1]
    using S_obs = np.add(FWFt, V2_per_step);                   // [n,1,1]

    using WFt = np.einsum('nij,njk->nik', W_exp, Ft_exp);      // [n,m,1]
    using K_obs_raw = np.divide(WFt, S_obs);                   // [n,m,1]
    using K_obs = np.multiply(mask_n, K_obs_raw);              // [n,m,1]

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
    using eta_obs = np.multiply(mask_n, eta_obs_raw);          // [n,m,1]

    using J_num = np.einsum('nij,njk->nik', FGt, FG);          // [n,m,m]
    using J_obs_raw = np.divide(J_num, S_obs);                 // [n,m,m]
    using J_obs = np.multiply(mask_mm, J_obs_raw);             // [n,m,m]

    // Missing-step blend: A=G, b=0, C=W, eta=0, J=0
    using A_obs_mask = np.multiply(mask_mm, A_obs);
    using A_nan = np.multiply(inv_mask_mm, G_exp);
    using A_all = np.add(A_obs_mask, A_nan);

    using b_all = np.multiply(mask_n, b_obs);

    using C_obs_mask = np.multiply(mask_mm, C_obs);
    using C_nan = np.multiply(inv_mask_mm, W_exp);
    using C_all = np.add(C_obs_mask, C_nan);

    using eta_all = np.multiply(mask_n, eta_obs);
    using J_all = np.multiply(mask_mm, J_obs);

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

    using S1 = np.add(np.einsum('nij,njk,nlk->nil', F1, C0_first, F1), V2_1);
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

    if (ownsV2ps) V2_per_step.dispose();

    // ─── Associative prefix scan ───
    const scanned = lax.associativeScan(
      composeForward,
      { A: A_arr, b: b_arr, C: C_arr, eta: eta_arr, J: J_arr },
    ) as ForwardElem;

    // x_filt[t] = scanned.A[t] · x0 + scanned.b[t]  [n, m, 1]
    // Note: np.add creates NEW arrays — x_filt and C_filt are independent of
    // scanned.b / scanned.S, so tree.dispose(scanned) below is safe.
    using x0_exp = np.tile(np.reshape(x0, [1, m, 1]), [n, 1, 1]);
    using Ax0 = np.einsum('nij,njk->nik', scanned.A, x0_exp);
    using x_filt = np.add(Ax0, scanned.b);

    // C_filt[t] = scanned.A[t]·C0·scanned.A[t]' + scanned.S[t]  [n, m, m]
    using C0_exp = np.tile(np.reshape(C0, [1, m, m]), [n, 1, 1]);
    using AC0At = np.einsum('nij,njk,nlk->nil', scanned.A, C0_exp, scanned.A);
    using C_filt_raw = np.add(AC0At, scanned.C);
    using C_filt_t = np.einsum('nij->nji', C_filt_raw);
    using C_filt_sum = np.add(C_filt_raw, C_filt_t);
    using C_filt = np.multiply(np.array(0.5, { dtype }), C_filt_sum);

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

    // ─── Compute -2·logL from filtered outputs (split, no concatenate) ───
    // np.concatenate lacks VJP in jax-js, so we compute the likelihood in
    // two parts: step 0 (uses x0, C0) and steps 1..n-1 (uses shifted
    // x_filt, C_filt).  Only np.split (which has working VJP) is used.

    // ── Split observations and mask ──
    const ySplit = np.split(y_arr, [1], 0);
    using y0 = ySplit[0];               // [1, 1, 1]
    const y_rest = ySplit[1];            // [n-1, 1, 1]  — disposed after use

    const maskSplit = mask_arr !== undefined
      ? np.split(mask_n, [1], 0) : undefined;
    using mask0 = maskSplit ? maskSplit[0] : np.ones([1, 1, 1], { dtype });
    const mask_rest = maskSplit
      ? maskSplit[1]
      : (mask_arr === undefined ? np.ones([n - 1, 1, 1], { dtype }) : undefined);

    // ── V² for each part ──
    let V2_0: np.Array;    // [1, 1] for step 0
    let V2_rest: np.Array; // [n-1, 1, 1] for steps 1..n-1
    if (fixS) {
      const v2Split = np.split(fixedV2_arr!, [1], 0);
      V2_0 = np.reshape(v2Split[0], [1, 1]);
      v2Split[0].dispose();
      V2_rest = v2Split[1];
    } else {
      V2_0 = np.reshape(V2!, [1, 1]);
      V2_rest = np.tile(np.reshape(V2!, [1, 1, 1]), [n - 1, 1, 1]);
    }

    // ── Step 0: v₀ = y₀ − F·x₀,  Cp₀ = F·C₀·F' + V² ──
    using Fx0 = np.matmul(F, x0);                          // [1, 1]
    using y0_11 = np.reshape(y0, [1, 1]);
    using v0 = np.subtract(y0_11, Fx0);                    // [1, 1]
    using FC0Ft = np.einsum('ij,jk,lk->il', F, C0, F);    // [1, 1]
    using Cp0 = np.add(FC0Ft, V2_0);                       // [1, 1]
    if (!fixS) { /* V2_0 aliases V2 via reshape — don't dispose */ } else V2_0.dispose();

    using v0_sq = np.square(v0);
    using v0_over_Cp = np.divide(v0_sq, Cp0);
    using log_Cp0 = np.log(Cp0);
    using lik0_raw = np.add(v0_over_Cp, log_Cp0);
    using mask0_11 = np.reshape(mask0, [1, 1]);
    using lik0_val = np.multiply(mask0_11, lik0_raw);
    using lik0 = np.sum(lik0_val);                          // scalar

    // ── Steps 1..n-1: x_pred = G·x_filt[0..n-2],  C_pred = G·C_filt·G'+W ──
    const xfSplit = np.split(x_filt, [n - 1], 0);
    xfSplit[1].dispose();
    using x_filt_head = xfSplit[0];                         // [n-1, m, 1]

    const cfSplit = np.split(C_filt, [n - 1], 0);
    cfSplit[1].dispose();
    using C_filt_head = cfSplit[0];                         // [n-1, m, m]

    using x_pred_r = np.einsum('ij,njk->nik', G, x_filt_head);     // [n-1, m, 1]
    using GCGt_r = np.einsum('ij,njk,lk->nil', G, C_filt_head, G); // [n-1, m, m]
    using W_tiled = np.tile(np.reshape(W, [1, m, m]), [n - 1, 1, 1]);
    using C_pred_r = np.add(GCGt_r, W_tiled);                      // [n-1, m, m]

    using Fx_r = np.einsum('ij,njk->nik', F, x_pred_r);            // [n-1, 1, 1]
    using v_rest_raw = np.subtract(y_rest, Fx_r);                   // [n-1, 1, 1]
    y_rest.dispose();

    using FCFt_r = np.einsum('ij,njk,lk->nil', F, C_pred_r, F);   // [n-1, 1, 1]
    using Cp_rest = np.add(FCFt_r, V2_rest);                       // [n-1, 1, 1]
    V2_rest.dispose();

    using vr_sq = np.square(v_rest_raw);
    using vr_over_Cp = np.divide(vr_sq, Cp_rest);
    using log_Cp_r = np.log(Cp_rest);
    using lik_r_raw = np.add(vr_over_Cp, log_Cp_r);
    using lik_r_masked = np.multiply(mask_rest!, lik_r_raw);
    using lik_rest = np.sum(lik_r_masked);                          // scalar
    mask_rest!.dispose();

    if (ownsMask) mask_n.dispose();
    if (!fixS) V2!.dispose();
    W.dispose();
    if (nar > 0) G.dispose();

    return np.add(lik0, lik_rest);
  };
};

/**
 * Estimate DLM parameters (s, w, and optionally arphi) by maximum likelihood
 * via autodiff.
 *
 * The entire optimization step — `valueAndGrad(loss)` (Kalman filter forward
 * pass + AD backward pass) and optax Adam moment/parameter updates — is
 * wrapped in a single `jit()` call, so every iteration runs from compiled code.
 *
 * The parameterization maps unconstrained reals → positive values:
 *   s = exp(θ_s),  w[i] = exp(θ_{w,i})
 * AR coefficients (when `options.fitar = true`) are optimized directly
 * (unconstrained — not log-transformed, matching MATLAB DLM behavior).
 *
 * When `sFixed` is supplied (a per-timestep σ array, e.g. known measurement
 * uncertainties), the observation noise is **not estimated** — it is treated as
 * a known constant.  Only W (and optionally arphi) are optimized.  The
 * returned `s` field will be `NaN` in this case.
 *
 * @param y - Observations (n×1)
 * @param options - Model specification (order, trig, ns, arphi, fitar, etc.)
 * @param init - Initial guess for parameters (optional; arphi defaults to options.arphi)
 * @param maxIter - Maximum optimizer iterations (default: 200)
 * @param lr - Learning rate for Adam (default: 0.05)
 * @param tol - Convergence tolerance on relative lik change (default: 1e-6).
 *   Requires 5 consecutive steps below this threshold before stopping, to guard
 *   against transient near-zero steps during oscillation (e.g. assocScan path).
 * @param dtype - Computation precision (default: Float64)
 * @param X - Optional covariate matrix (n rows × q cols), passed through to dlmFit
 * @param sFixed - Optional per-timestep σ array (length n). When provided, s is fixed
 *   and not estimated; only W is optimized.
 * @param adamOpts - Optional Adam hyperparameters (b1, b2, eps). Default: b1=0.9, b2=0.9, eps=1e-8.
 *   The b2=0.9 default is much faster to adapt than the canonical 0.999 on DLM likelihoods
 *   (measured: reaches same loss in ~3× fewer iterations on Nile and ozone benchmarks).
 * @returns MLE result with estimated parameters and full DLM fit
 */
export const dlmMLE = async (
  y: ArrayLike<number>,
  options: DlmOptions = {},
  init?: { s?: number; w?: number[]; arphi?: number[] },
  maxIter: number = 200,
  lr: number = 0.05,
  tol: number = 1e-6,
  dtype: DType = DType.Float64,
  callbacks?: {
    /** Called before iteration 0 with the initial theta. */
    onInit?: (theta: Float64Array | Float32Array) => void;
    /** Called after each iteration with the updated theta and lik. */
    onIteration?: (iter: number, theta: Float64Array | Float32Array, lik: number) => void;
  },
  X?: ArrayLike<number>[],  // n×q covariate matrix, passed through to dlmFit
  sFixed?: ArrayLike<number>, // per-timestep σ (fixes V; s is not estimated)
  adamOpts?: ScaleByAdamOptions, // Adam b1/b2/eps overrides
  /** Force use of `makeKalmanLossAssoc` (associativeScan-based loss) regardless
   *  of device/dtype. Useful for benchmarking the parallel loss path on
   *  CPU/WASM backends where it would not normally be selected. */
  forceAssocScan?: boolean,
): Promise<DlmMleResult> => {
  const t0 = performance.now();
  const n = y.length;
  const FA = getFloatArrayType(dtype);
  const yArr = y instanceof FA ? y as FloatArray : FA.from(y);

  // Generate system matrices
  const sys = dlmGenSys(options);
  const m = sys.m;

  // AR fitting setup
  const arphi_orig = options.arphi ?? [];
  const fitar = !!(options.fitar && arphi_orig.length > 0);
  const arInds = fitar ? findArInds(options) : [];
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
  const ns = options.ns ?? 12;
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
  const s_init = init?.s ?? (Math.sqrt(Math.abs(variance)) || 1.0);
  const w_init = init?.w ?? new Array(m).fill(s_init * 0.1);
  const arphi_init = init?.arphi ?? arphi_orig;

  // Build fixed V2_arr when sFixed is provided
  const fixS = sFixed !== undefined;
  let fixedV2_arr: np.Array | undefined;
  if (fixS) {
    // V2_t = sFixed[t]² — shape [n, 1, 1]
    const v2data = Array.from(sFixed!).map(si => [[si * si]]);
    fixedV2_arr = np.array(v2data, { dtype });
  }

  // theta = [log(s)?, log(w0).., log(w_{m-1}), arphi..]
  const theta_init = [
    ...(fixS ? [] : [Math.log(s_init)]),
    ...w_init.map(wi => Math.log(Math.abs(wi) || 0.01)),
    ...(fitar ? arphi_init : []),  // unconstrained (not log-transformed); only when fitting AR
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
  const lossFn = useAssocScanLoss
    ? makeKalmanLossAssoc(F, G, x0, C0, y_arr, n, m, dtype, arInds, fixedV2_arr, mleMaskArr)
    : makeKalmanLoss(F, G, Ft, x0, C0, y_arr, n, m, dtype, arInds, fixedV2_arr, false, mleMaskArr);
  const optimizer = adam(lr, { b2: 0.9, ...adamOpts });

  // One jit() wrapping: valueAndGrad (Kalman scan + AD) + optax Adam update.
  // Traces once, then every iteration is compiled.
  const optimStep = jit((theta: np.Array, optState: any): [np.Array, any, np.Array] => {
    const [likVal, grad] = valueAndGrad(lossFn)(theta);
    const [updates, newOptState] = optimizer.update(grad, optState);
    const newTheta = applyUpdates(theta, updates);
    return [newTheta, newOptState, likVal];
  });

  // Initialize
  let theta = np.array(theta_init, { dtype });
  let optState: any = optimizer.init(theta);

  // Notify callback with initial theta
  if (callbacks?.onInit) {
    const initData = await theta.data() as Float64Array | Float32Array;
    callbacks.onInit(initData);
  }

  const likHistory: number[] = [];
  let prevLik = Infinity;
  let iter = 0;
  let patienceCount = 0;
  const PATIENCE = 5; // require 5 consecutive steps below tol before stopping

  for (iter = 0; iter < maxIter; iter++) {
    const [newTheta, newOptState, likVal] = optimStep(theta, optState);
    const likNum = (await likVal.consumeData() as Float64Array | Float32Array)[0];
    likHistory.push(likNum);

    // Notify callback with updated theta
    if (callbacks?.onIteration) {
      const td = await newTheta.data() as Float64Array | Float32Array;
      callbacks.onIteration(iter, td, likNum);
    }

    // Dispose old state
    theta.dispose(); tree.dispose(optState);
    theta = newTheta; optState = newOptState;

    // Check convergence: require PATIENCE consecutive steps below tol to guard
    // against transient near-zero steps during oscillation (assocScan path).
    const relChange = Math.abs((likNum - prevLik) / (Math.abs(prevLik) + 1e-30));
    if (iter > 0) {
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
  const thetaData = await theta.data() as Float64Array | Float32Array;
  theta.dispose(); tree.dispose(optState);
  fixedV2_arr?.dispose();
  mleMaskArr?.dispose();

  const wOffset = fixS ? 0 : 1;
  const s_opt = fixS ? NaN : Math.exp(thetaData[0]);
  const w_opt = Array.from({ length: m }, (_, i) => Math.exp(thetaData[wOffset + i]));
  const arphi_opt = nar > 0
    ? Array.from({ length: nar }, (_, i) => thetaData[wOffset + m + i])
    : undefined;

  // Run full dlmFit with optimized parameters (including fitted arphi if applicable)
  const fitOptions = arphi_opt ? { ...options, arphi: arphi_opt } : options;
  // When s was fixed, pass the original sFixed array to dlmFit; otherwise use scalar s_opt
  const sForFit: number | ArrayLike<number> = fixS ? sFixed! : s_opt;
  const fit = await dlmFit(yArr, sForFit, w_opt, dtype, fitOptions, X);

  const elapsed = performance.now() - t0;

  return {
    s: s_opt,
    w: w_opt,
    arphi: arphi_opt,
    lik: prevLik,
    iterations: iter,
    fit,
    likHistory,
    elapsed,
  };
};
