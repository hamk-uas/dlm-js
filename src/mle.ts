import { DType, numpy as np, lax, jit, valueAndGrad, tree } from "@hamk-uas/jax-js-nonconsuming";
import { adam, applyUpdates } from "@hamk-uas/jax-js-nonconsuming/optax";
import type { DlmFitResult, FloatArray } from "./types";
import { getFloatArrayType } from "./types";
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
// AD-traced: array lifetimes managed by the tracer, not `using`
/* eslint-disable jax-js/require-using */
const buildDiagW = (
  expTheta: np.Array, m: number, dtype: DType, nTheta: number,
): np.Array => {
  let W = np.zeros([m, m], { dtype });
  for (let i = 0; i < m; i++) {
    const maskData = new Array(nTheta).fill(0);
    maskData[1 + i] = 1;
    const mask = np.array(maskData, { dtype });
    const wi = np.dot(expTheta, mask);
    const wi2 = np.square(wi);

    const eiData = Array.from({ length: m }, (_, j) => j === i ? [1] : [0]);
    const ei = np.array(eiData, { dtype });
    const eit = np.transpose(ei);
    const outer = np.matmul(ei, eit);
    const scaled = np.multiply(np.reshape(wi2, [1, 1]), outer);
    // Accumulator pattern: transfer ownership from old W to new W
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
    const mask = np.array(maskData, { dtype });
    const phi_i = np.dot(theta, mask);

    // Rank-1 update at G[arInds[i], arCol]
    const eiData = Array.from({ length: m }, (_, j) => j === arInds[i] ? [1] : [0]);
    const ejData = Array.from({ length: m }, (_, j) => [j === arCol ? 1 : 0]);
    const ei = np.array(eiData, { dtype });     // [m, 1]
    const ejt = np.transpose(np.array(ejData, { dtype })); // [1, m]
    const outer = np.matmul(ei, ejt);            // [m, m]
    const scaled = np.multiply(np.reshape(phi_i, [1, 1]), outer);
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
 * jax-js-nonconsuming v0.2.2 with the following AD-compatibility constraints:
 *
 * - Use `np.diag(np.ones([m]))` instead of `np.eye(m)` everywhere (np.eye
 *   produces abstract tracer shapes under grad, breaking matmul).
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
) => {
  const nar = arInds.length;
  const nSwParams = 1 + m;
  const nTheta = nSwParams + nar;

  return (theta: np.Array): np.Array => { // AD-traced
    // Build effective G: constant if no AR fitting, theta-dependent if fitting
    const G = nar > 0
      ? buildG(G_base, theta, arInds, m, nSwParams, nTheta, dtype)
      : G_base;

    // Step function defined here so it captures the correct G
    type Carry = { x: np.Array; C: np.Array };
    type ScanInp = { y: np.Array; V2: np.Array; W: np.Array };

    const step = (carry: Carry, inp: ScanInp): [Carry, np.Array] => {
      const { x, C } = carry;
      const { y: yi, V2: V2i, W } = inp;

      // Innovation: v = y - F·x  [1,1]
      const v = np.subtract(yi, np.matmul(F, x));

      // C·Fᵀ: [m,m]@[m,1] → [m,1]
      const CFt = np.matmul(C, Ft);

      // Innovation covariance: Cp = F·(C·Fᵀ) + V²  [1,1]
      const Cp = np.add(np.matmul(F, CFt), V2i);

      // Kalman gain: K = G·(C·Fᵀ) / Cp  [m,1]
      const K = np.divide(np.matmul(G, CFt), Cp);

      // Next state: x_next = G·x + K·v  [m,1]
      const x_next = np.add(np.matmul(G, x), np.matmul(K, v));

      // L = G - K·F  [m,m]
      const L = np.subtract(G, np.matmul(K, F));
      const Lt = np.transpose(L);

      // Next covariance: C_next = G·(C·Lᵀ) + W  [m,m]
      const CLt = np.matmul(C, Lt);
      const C_next = np.add(np.matmul(G, CLt), W);

      // Per-step -2·loglik: v²/Cp + log(Cp)
      const lik_t = np.add(np.divide(np.square(v), Cp), np.log(Cp));
      return [{ x: x_next, C: C_next }, np.squeeze(lik_t)];
    };

    const expTheta = np.exp(theta);

    // s = exp(theta[0]) via dot mask
    const mask_s = np.array([1, ...new Array(nTheta - 1).fill(0)], { dtype });
    const sVal = np.dot(expTheta, mask_s);
    const V2 = np.reshape(np.square(sVal), [1, 1]);

    // W = diag(w²) from theta[1..m]
    const W = buildDiagW(expTheta, m, dtype, nTheta);

    // Broadcast to [n, ...] for scan inputs
    const V2_arr = np.multiply(
      np.ones([n, 1, 1], { dtype }),
      np.reshape(V2, [1, 1, 1]),
    );
    const W_arr = np.multiply(
      np.ones([n, 1, 1], { dtype }),
      np.reshape(W, [1, m, m]),
    );

    const [fc, likTerms] = lax.scan(
      step,
      { x: x0, C: C0 },
      { y: y_arr, V2: V2_arr, W: W_arr },
    );
    tree.dispose(fc);
    const total = np.sum(likTerms);
    likTerms.dispose();
    return total;
  };
};
/* eslint-enable jax-js/require-using */

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
 * @param y - Observations (n×1)
 * @param options - Model specification (order, trig, ns, arphi, fitar, etc.)
 * @param init - Initial guess for parameters (optional; arphi defaults to options.arphi)
 * @param maxIter - Maximum optimizer iterations (default: 200)
 * @param lr - Learning rate for Adam (default: 0.05)
 * @param tol - Convergence tolerance on relative lik change (default: 1e-6)
 * @param dtype - Computation precision (default: Float64)
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

  // Stack observations: [n, 1, 1]
  using y_arr = np.array(Array.from(yArr).map(yi => [[yi]]), { dtype });

  // Initial state — use np.array, NOT np.eye (AD bug in jax-js v0.2.2)
  const ns = options.ns ?? 12;
  let initSum = 0;
  const count = Math.min(ns, n);
  for (let i = 0; i < count; i++) initSum += Number(yArr[i]);
  const mean_y = initSum / count;
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

  // Initial parameter guess
  const variance = Array.from(yArr).reduce((s, v) => s + v * v, 0) / n
    - (Array.from(yArr).reduce((s, v) => s + v, 0) / n) ** 2;
  const s_init = init?.s ?? (Math.sqrt(Math.abs(variance)) || 1.0);
  const w_init = init?.w ?? new Array(m).fill(s_init * 0.1);
  const arphi_init = init?.arphi ?? arphi_orig;
  const theta_init = [
    Math.log(s_init),
    ...w_init.map(wi => Math.log(Math.abs(wi) || 0.01)),
    ...(fitar ? arphi_init : []),  // unconstrained (not log-transformed); only when fitting AR
  ];

  // Build loss & JIT the entire optimization step:
  //   (theta, optState) → (newTheta, newOptState, likValue)
  // One jit() wrapping: valueAndGrad (Kalman scan + AD) + optax Adam update.
  // Traces once, then every iteration is compiled.

  const lossFn = makeKalmanLoss(F, G, Ft, x0, C0, y_arr, n, m, dtype, arInds);
  const optimizer = adam(lr);

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

    // Check convergence
    const relChange = Math.abs((likNum - prevLik) / (Math.abs(prevLik) + 1e-30));
    if (iter > 0 && relChange < tol) {
      prevLik = likNum;
      break;
    }
    prevLik = likNum;
  }

  // Extract optimized parameters
  const thetaData = await theta.data() as Float64Array | Float32Array;
  theta.dispose(); tree.dispose(optState);

  const s_opt = Math.exp(thetaData[0]);
  const w_opt = Array.from({ length: m }, (_, i) => Math.exp(thetaData[1 + i]));
  const arphi_opt = nar > 0
    ? Array.from({ length: nar }, (_, i) => thetaData[1 + m + i])
    : undefined;

  // Run full dlmFit with optimized parameters (including fitted arphi if applicable)
  const fitOptions = arphi_opt ? { ...options, arphi: arphi_opt } : options;
  const fit = await dlmFit(yArr, s_opt, w_opt, dtype, fitOptions);

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
