import { DType, numpy as np } from "@jax-js/jax";
import type { DlmSmoResult, DlmFitResult } from "./types";
import { disposeAll } from "./types";

// Public type exports
export type { DlmFitResult } from "./types";

/**
 * DLM Smoother - Kalman filter (forward) + RTS smoother (backward).
 * @internal
 */
const dlmSmo = async (
  y: number[],
  F: np.Array,
  V_std: number[],
  x0_data: number[][],
  G: np.Array,
  W: np.Array,
  C0_data: number[][],
  dtype: DType = DType.Float64
): Promise<DlmSmoResult> => {
  const n = y.length;

  // Storage for filter outputs
  const x_pred: np.Array[] = new Array(n);
  const C_pred: np.Array[] = new Array(n);
  const K_array: np.Array[] = new Array(n);
  const v_array: number[] = new Array(n);
  const Cp_array: number[] = new Array(n);

  // Initialize from input data (already correct shapes from nested arrays)
  x_pred[0] = np.array(x0_data, { dtype });
  C_pred[0] = np.array(C0_data, { dtype });

  // === Forward Kalman Filter ===
  for (let i = 0; i < n; i++) {
    const xi = x_pred[i];
    const Ci = C_pred[i];

    // Innovation: v = y - F*x
    const v = np.subtract(np.array([y[i]], { dtype }), np.matmul(F.ref, xi.ref));
    v_array[i] = (await v.ref.data())[0];

    // Innovation covariance: Cp = F*C*F' + V
    const Cp = np.add(
      np.matmul(np.matmul(F.ref, Ci.ref), np.transpose(F.ref)),
      np.array([[V_std[i] ** 2]], { dtype })
    );
    Cp_array[i] = (await Cp.ref.data())[0];

    // Kalman gain: K = G*C*F' / Cp (result is [2,1])
    const K = np.matmul(
      np.matmul(np.matmul(G.ref, Ci.ref), np.transpose(F.ref)),
      np.linalg.solve(Cp.ref, np.eye(1, 1, { dtype }))
    );
    // Store K - need to keep it for backward pass, use .ref to preserve
    K_array[i] = np.reshape(K.ref, [2, 1]);

    // Predict next state (except last timestep)
    if (i < n - 1) {
      // L = G - K*F
      const L = np.subtract(G.ref, np.matmul(K.ref, F.ref));

      // x_next = G*x + K*v (result is [2,1])
      x_pred[i + 1] = np.add(
        np.matmul(G.ref, xi.ref),
        np.matmul(K.ref, np.array([[v_array[i]]], { dtype }))
      );

      // C_next = G*C*L' + W (result is [2,2])
      C_pred[i + 1] = np.add(
        np.matmul(np.matmul(G.ref, Ci.ref), np.transpose(L)),
        W.ref
      );
    }

    disposeAll(K, v, Cp);
  }

  // === Backward RTS Smoother ===
  let r = np.array([[0.0], [0.0]], { dtype });
  let N = np.array([[0.0, 0.0], [0.0, 0.0]], { dtype });
  const x_smooth: np.Array[] = new Array(n);
  const C_smooth: np.Array[] = new Array(n);

  for (let i = n - 1; i >= 0; i--) {
    const xi = x_pred[i];
    const Ci = C_pred[i];
    const Ki = K_array[i];

    // L = G - K*F, Lt = L'
    const L = np.subtract(G.ref, np.matmul(Ki.ref, F.ref));
    const Lt = np.transpose(L.ref);

    // F'/Cp for F=[1,0]: [[1/Cp], [0]]
    const FtCpInv = np.array([[1.0 / Cp_array[i]], [0.0]], { dtype });

    // r_new = F'/Cp * v + L' * r
    const r_new = np.add(
      np.multiply(FtCpInv.ref, v_array[i]),
      np.matmul(Lt.ref, r.ref)
    );

    // N_new = F'/Cp * F + L' * N * L
    const N_new = np.add(
      np.matmul(FtCpInv, F.ref),
      np.matmul(np.matmul(Lt, N.ref), L)
    );

    // x_smooth = x_pred + C * r (result is [2,1])
    x_smooth[i] = np.add(xi.ref, np.matmul(Ci.ref, r_new.ref));

    // C_smooth = C_pred - C * N * C (result is [2,2])
    C_smooth[i] = np.subtract(
      Ci.ref,
      np.matmul(np.matmul(Ci.ref, N_new.ref), Ci.ref)
    );

    disposeAll(r, N);
    r = r_new;
    N = N_new;
  }

  disposeAll(r, N);
  for (let i = 0; i < n; i++) K_array[i].dispose();

  // === Compute statistics ===
  const yhat: number[] = [];
  const ystd: number[] = [];
  const xstd: [number, number][] = [];
  const resid0: number[] = [];
  const resid: number[] = [];
  const resid2: number[] = [];
  let ssy = 0;
  let lik = 0;

  for (let i = 0; i < n; i++) {
    const yhat_i = (await x_pred[i].ref.data())[0];
    const C_s = await C_smooth[i].ref.data();

    yhat.push(yhat_i);
    xstd.push([Math.sqrt(Math.abs(C_s[0])), Math.sqrt(Math.abs(C_s[3]))]);
    ystd.push(Math.sqrt(C_s[0] + V_std[i] ** 2));

    const r0 = y[i] - yhat_i;
    resid0.push(r0);
    resid.push(r0 / V_std[i]);
    resid2.push(v_array[i] / Math.sqrt(Cp_array[i]));

    ssy += r0 * r0;
    lik += (v_array[i] ** 2) / Cp_array[i] + Math.log(Cp_array[i]);
  }

  const nobs = n;
  const s2 = resid.reduce((s, r) => s + r * r, 0) / nobs;
  const mse = resid2.reduce((s, r) => s + r * r, 0) / nobs;
  const mape = resid2.reduce((s, r, i) => s + Math.abs(r) / Math.abs(y[i]), 0) / nobs;

  return {
    x: x_smooth, C: C_smooth, xf: x_pred, Cf: C_pred,
    yhat, ystd, xstd, resid0, resid, resid2,
    v: v_array, Cp: Cp_array, ssy, s2, nobs, lik, mse, mape,
  };
};

/**
 * Fit a local linear trend DLM model.
 *
 * @param y - Observations
 * @param s - Observation noise std dev
 * @param w - State noise std devs [level, slope]
 * @param dtype - Computation dtype (default: Float64)
 */
export const dlmFit = async (
  y: number[],
  s: number,
  w: [number, number],
  dtype: DType = DType.Float64
): Promise<DlmFitResult> => {
  const n = y.length;

  // System matrices
  const G = np.array([[1.0, 1.0], [0.0, 1.0]], { dtype });
  const F = np.array([[1.0, 0.0]], { dtype });
  const W = np.array([[w[0] ** 2, 0.0], [0.0, w[1] ** 2]], { dtype });
  const V_std = new Array<number>(n).fill(s);

  // Initial state (mean of first 12 observations)
  const mean_y = y.slice(0, 12).reduce((a, b) => a + b, 0) / 12;
  const c0_val = (Math.abs(mean_y) * 0.5) ** 2;
  const c0 = c0_val === 0 ? 1e7 : c0_val;
  const x0_data = [[mean_y], [0.0]];
  const C0_data = [[c0, 0.0], [0.0, c0]];

  // Run 1: Initial smoother pass (use .ref to preserve matrices)
  const out1 = await dlmSmo(y, F.ref, V_std, x0_data, G.ref, W.ref, C0_data, dtype);

  // Update initial values from smoothed estimates
  const x0_new = await out1.x[0].ref.data();
  const C0_new = await out1.C[0].ref.data();
  const x0_updated = [[x0_new[0]], [x0_new[1]]];
  const C0_scaled = [
    [C0_new[0] * 100, C0_new[1] * 100],
    [C0_new[2] * 100, C0_new[3] * 100],
  ];

  // Dispose Run 1 arrays
  for (let i = 0; i < n; i++) {
    disposeAll(out1.x[i], out1.C[i], out1.xf[i], out1.Cf[i]);
  }

  // Run 2: Final pass (matrices consumed here)
  const out2 = await dlmSmo(y, F, V_std, x0_updated, G, W, C0_scaled, dtype);

  // Convert to plain JS arrays
  const xf: number[][] = [[], []];
  const Cf: number[][][] = [[[], []], [[], []]];
  const x: number[][] = [[], []];
  const C: number[][][] = [[[], []], [[], []]];
  const xstd: number[][] = [];

  for (let i = 0; i < n; i++) {
    const xfi = await out2.xf[i].ref.data();
    xf[0].push(xfi[0]); xf[1].push(xfi[1]);
    out2.xf[i].dispose();

    const Cfi = await out2.Cf[i].ref.data();
    Cf[0][0].push(Cfi[0]); Cf[0][1].push(Cfi[1]);
    Cf[1][0].push(Cfi[2]); Cf[1][1].push(Cfi[3]);
    out2.Cf[i].dispose();

    const xi = await out2.x[i].ref.data();
    x[0].push(xi[0]); x[1].push(xi[1]);
    out2.x[i].dispose();

    const Ci = await out2.C[i].ref.data();
    C[0][0].push(Ci[0]); C[0][1].push(Ci[1]);
    C[1][0].push(Ci[2]); C[1][1].push(Ci[3]);
    out2.C[i].dispose();

    xstd.push([out2.xstd[i][0], out2.xstd[i][1]]);
  }

  return {
    xf, Cf, x, C, xstd,
    G: [[1.0, 1.0], [0.0, 1.0]],
    F: [1.0, 0.0],
    W: [[w[0] ** 2, 0.0], [0.0, w[1] ** 2]],
    y, V: V_std,
    x0: [x0_updated[0][0], x0_updated[1][0]],
    C0: C0_scaled,
    XX: [],
    yhat: out2.yhat, ystd: out2.ystd,
    resid0: out2.resid0, resid: out2.resid, resid2: out2.resid2,
    ssy: out2.ssy, v: out2.v, Cp: out2.Cp,
    s2: out2.s2, nobs: out2.nobs, lik: out2.lik, mse: out2.mse, mape: out2.mape,
    class: 'dlmfit',
  };
};
