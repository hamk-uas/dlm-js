import { numpy as np, DType } from "@hamk-uas/jax-js-nonconsuming";

/** TypedArray type for float data - either Float32Array or Float64Array based on dtype */
export type FloatArray = Float32Array | Float64Array;

/** TypedArray constructor type */
export type FloatArrayConstructor = typeof Float32Array | typeof Float64Array;

/**
 * Result from the DLM smoother function (dlmSmo).
 * All tensor outputs are np.Arrays returned directly from the JIT core.
 * Caller is responsible for reading (.data()) and disposing these arrays.
 *
 * For a model with state dimension m and n observations:
 * - State/covariance tensors are stacked: x [n,m,1], C [n,m,m]
 * - Observation-space diagnostics are [n] vectors
 * - Scalar diagnostics are 0-d tensors
 *
 * @internal - Used only within the library implementation.
 */
export interface DlmSmoResult {
  /** Smoothed states [n, m, 1] */
  x: np.Array;
  /** Smoothed covariances [n, m, m] */
  C: np.Array;
  /** Filtered (predicted) states [n, m, 1] */
  xf: np.Array;
  /** Filtered covariances [n, m, m] */
  Cf: np.Array;
  /** Filter predictions yhat = F·xf [n] */
  yhat: np.Array;
  /** Prediction standard deviations [n] */
  ystd: np.Array;
  /** Innovations [n] */
  v: np.Array;
  /** Innovation covariances [n] */
  Cp: np.Array;
  /** Raw residuals [n] */
  resid0: np.Array;
  /** Scaled residuals [n] */
  resid: np.Array;
  /** Standardized residuals [n] */
  resid2: np.Array;
  /** Sum of squared raw residuals (scalar) */
  ssy: np.Array;
  /** -2 * log likelihood (scalar) */
  lik: np.Array;
  /** Residual variance (scalar) */
  s2: np.Array;
  /** Mean squared error (scalar) */
  mse: np.Array;
  /** Mean absolute percentage error (scalar) */
  mape: np.Array;
  /** Number of non-NaN observations (scalar np.Array, consumed by dlmFit) */
  nobs: np.Array;
  /** State dimension */
  m: number;
}

/**
 * Result from the DLM fit function (dlmFit).
 * Numeric arrays use TypedArrays (Float32Array or Float64Array based on dtype).
 */
export interface DlmFitResult {
  /** Filtered state means [state_dim][time] */
  xf: FloatArray[];
  /** Filtered covariances [row][col][time] */
  Cf: FloatArray[][];
  /** Smoothed state means [state_dim][time] */
  x: FloatArray[];
  /** Smoothed covariances [row][col][time] */
  C: FloatArray[][];
  /** Smoothed state standard deviations [time][state_dim] */
  xstd: FloatArray[];
  /** State transition matrix G (m×m) */
  G: number[][];
  /** Observation vector F (1×m flattened) */
  F: number[];
  /** State noise covariance W (m×m) */
  W: number[][];
  /** Observations */
  y: FloatArray;
  /** Observation noise standard deviations */
  V: FloatArray;
  /** Initial state mean (after first smoother pass) */
  x0: number[];
  /** Initial state covariance (scaled) */
  C0: number[][];
  /** Covariates matrix: XX[t] is the covariate row at time t (empty array when no covariates) */
  XX: number[][] | number[];
  /** Filter predictions */
  yhat: FloatArray;
  /** Prediction standard deviations */
  ystd: FloatArray;
  /** Raw residuals */
  resid0: FloatArray;
  /** Scaled residuals */
  resid: FloatArray;
  /** Sum of squared residuals */
  ssy: number;
  /** Innovations */
  v: FloatArray;
  /** Innovation covariances */
  Cp: FloatArray;
  /** Residual variance */
  s2: number;
  /** Number of observations */
  nobs: number;
  /** -2 * log likelihood */
  lik: number;
  /** Mean squared error */
  mse: number;
  /** Mean absolute percentage error */
  mape: number;
  /** Standardized residuals */
  resid2: FloatArray;
  /** Class identifier */
  class: 'dlmfit';
}

/**
 * Result from dlmForecast: h-step-ahead predictions with uncertainty.
 *
 * All arrays have length h (the forecast horizon).
 */
export interface DlmForecastResult {
  /**
   * Predicted observation means (h×1).
   * yhat[k] = F · x_pred[k]
   */
  yhat: FloatArray;
  /**
   * Predicted observation standard deviations (h×1).
   * ystd[k] = sqrt(F · C_pred[k] · F' + s²)
   */
  ystd: FloatArray;
  /**
   * Predicted state means (state_dim × h).
   * x[i][k] = i-th state component at forecast step k+1.
   */
  x: FloatArray[];
  /**
   * Predicted state covariances (state_dim × state_dim × h).
   * C[i][j][k] = covariance between states i and j at forecast step k+1.
   */
  C: FloatArray[][];
  /**
   * Predicted state standard deviations (h × state_dim).
   * xstd[k][i] = sqrt(C[i][i][k])
   */
  xstd: FloatArray[];
  /** Forecast horizon */
  h: number;
  /** State dimension */
  m: number;
}

/**
 * Get the appropriate TypedArray constructor based on DType.
 */
export function getFloatArrayType(dtype: DType): FloatArrayConstructor {
  return dtype === DType.Float32 ? Float32Array : Float64Array;
}

/**
 * AD-safe batched matrix inverse for [n, m, m] tensors.
 *
 * Uses analytic cofactor-expansion formulas (m ≤ 3) and a recursive
 * Schur-complement block inversion (m ≥ 4) to avoid the broken
 * `np.linalg.inv` VJP in jax-js-nonconsuming (which adds a spurious
 * −inv(X) term to the gradient). All constituent ops — multiply, subtract,
 * add, divide, split, concatenate, einsum — have correct VJPs.
 *
 * For m ≥ 4 the matrix is partitioned as X = [[A, B], [C, D]] with
 * k = ⌊m/2⌋, j = m − k and the Woodbury / Schur complement formula applied:
 *
 *   X⁻¹ = [[  S⁻¹,              −S⁻¹ B D⁻¹ ],
 *           [ −D⁻¹ C S⁻¹,   D⁻¹ + D⁻¹ C S⁻¹ B D⁻¹ ]]
 *
 * where S = A − B D⁻¹ C.  Both sub-inverses recurse into this function
 * with smaller dimension, bottoming out at the m ≤ 3 analytic cases.
 * Maximum recursion depth: ⌈log₂(m/2)⌉ for the D block.
 *
 * Float32 sub-block stabilization: before inverting D, an extra ε·I is added
 * to prevent near-singularity.  The outer ε·I that callers add to the full
 * m×m matrix does not protect extracted sub-blocks when float32 rounding
 * makes (C·J) diagonal entries approach −1.
 *
 * @param X - Batched matrix [n, m, m]
 * @param m - Matrix dimension (state size)
 * @param dtype - Computation dtype
 * @returns inv(X) with shape [n, m, m]
 * @internal
 */
export function adSafeInv(X: np.Array, m: number, dtype: DType): np.Array {
  if (m === 1) {
    // inv([[x]]) = [[1/x]]
    using ones = np.ones(X.shape, { dtype });
    return np.divide(ones, X);
  }

  if (m === 2) {
    // X = [[a, b], [c, d]]  →  inv = [[d, -b], [-c, a]] / (ad - bc)
    const rows = np.split(X, [1], 1);       // [n,1,2], [n,1,2]
    const r0c = np.split(rows[0], [1], 2);   // a=[n,1,1], b=[n,1,1]
    const r1c = np.split(rows[1], [1], 2);   // c=[n,1,1], d=[n,1,1]
    rows[0].dispose(); rows[1].dispose();
    const a = r0c[0], b = r0c[1], c = r1c[0], d = r1c[1];

    using ad = np.multiply(a, d);
    using bc = np.multiply(b, c);
    using det = np.subtract(ad, bc);

    using neg1 = np.array(-1, { dtype });
    using neg_b = np.multiply(b, neg1);
    using neg_c = np.multiply(c, neg1);
    using adj_r0 = np.concatenate([d, neg_b], 2);   // [n,1,2]
    using adj_r1 = np.concatenate([neg_c, a], 2);    // [n,1,2]
    using adj = np.concatenate([adj_r0, adj_r1], 1); // [n,2,2]

    a.dispose(); b.dispose(); c.dispose(); d.dispose();

    return np.divide(adj, det);  // broadcast [n,2,2] / [n,1,1]
  }

  if (m === 3) {
    // 3×3 cofactor expansion.
    // X = [[a,b,c],[d,e,f],[g,h,i]]
    // det = a(ei-fh) - b(di-fg) + c(dh-eg)
    // adj = transpose of cofactor matrix
    const rows = np.split(X, [1, 2], 1);         // 3 × [n,1,3]
    const r0c = np.split(rows[0], [1, 2], 2);     // a,b,c each [n,1,1]
    const r1c = np.split(rows[1], [1, 2], 2);     // d,e,f
    const r2c = np.split(rows[2], [1, 2], 2);     // g,h,i
    rows[0].dispose(); rows[1].dispose(); rows[2].dispose();
    const [a, b, c] = r0c;
    const [d, e, f] = r1c;
    const [g, h, i] = r2c;

    using neg1 = np.array(-1, { dtype });

    // 2×2 minors for cofactor matrix
    using ei = np.multiply(e, i); using fh = np.multiply(f, h);
    using di = np.multiply(d, i); using fg = np.multiply(f, g);
    using dh = np.multiply(d, h); using eg = np.multiply(e, g);
    using bi = np.multiply(b, i); using ch = np.multiply(c, h);
    using ai = np.multiply(a, i); using cg = np.multiply(c, g);
    using ah = np.multiply(a, h); using bg = np.multiply(b, g);
    using bf = np.multiply(b, f); using ce = np.multiply(c, e);
    using af = np.multiply(a, f); using cd = np.multiply(c, d);
    using ae = np.multiply(a, e); using bd = np.multiply(b, d);

    // Cofactors: cof[i][j] = (-1)^{i+j} * minor_ij
    // Adjugate: adj[j][i] = cof[i][j]  (transpose of cofactor)
    using cof00 = np.subtract(ei, fh);                                     // ei-fh
    using cof01_raw = np.subtract(di, fg);
    using cof01 = np.multiply(cof01_raw, neg1);                            // -(di-fg)
    using cof02 = np.subtract(dh, eg);                                     // dh-eg
    using cof10_raw = np.subtract(bi, ch);
    using cof10 = np.multiply(cof10_raw, neg1);                            // -(bi-ch)
    using cof11 = np.subtract(ai, cg);                                     // ai-cg
    using cof12_raw = np.subtract(ah, bg);
    using cof12 = np.multiply(cof12_raw, neg1);                            // -(ah-bg)
    using cof20 = np.subtract(bf, ce);                                     // bf-ce
    using cof21_raw = np.subtract(af, cd);
    using cof21 = np.multiply(cof21_raw, neg1);                            // -(af-cd)
    using cof22 = np.subtract(ae, bd);                                     // ae-bd

    // det = a*cof00 + b*cof01 + c*cof02
    using d1 = np.multiply(a, cof00);
    using d2 = np.multiply(b, cof01);
    using d3 = np.multiply(c, cof02);
    using det = np.add(np.add(d1, d2), d3);

    // adj[j][i] = cof[i][j]
    using adj_r0 = np.concatenate([cof00, cof10, cof20], 2); // row 0 of adj
    using adj_r1 = np.concatenate([cof01, cof11, cof21], 2);
    using adj_r2 = np.concatenate([cof02, cof12, cof22], 2);
    using adj = np.concatenate([adj_r0, adj_r1, adj_r2], 1); // [n,3,3]

    a.dispose(); b.dispose(); c.dispose();
    d.dispose(); e.dispose(); f.dispose();
    g.dispose(); h.dispose(); i.dispose();

    return np.divide(adj, det);  // broadcast [n,3,3] / [n,1,1]
  }

  // m ≥ 4: Schur-complement block inversion — fully AD-safe (no np.linalg.inv).
  //
  // Partition X = [[A, B], [C, D]]  with  k = ⌊m/2⌋,  j = m − k.
  // Then:
  //   S    = A − B·D⁻¹·C            (Schur complement of D, size k×k)
  //   X⁻¹  = [[  S⁻¹,             −S⁻¹·B·D⁻¹           ],
  //            [ −D⁻¹·C·S⁻¹,   D⁻¹ + D⁻¹·C·S⁻¹·B·D⁻¹  ]]
  // Both recursive calls land on smaller m, eventually reaching m ≤ 3.
  const k = Math.floor(m / 2);
  const j = m - k;

  const rowBlocks = np.split(X, [k], 1);              // [n,k,m], [n,j,m]
  const topCols   = np.split(rowBlocks[0], [k], 2);   // A=[n,k,k], B=[n,k,j]
  const botCols   = np.split(rowBlocks[1], [k], 2);   // C=[n,j,k], D=[n,j,j]
  rowBlocks[0].dispose(); rowBlocks[1].dispose();
  const A = topCols[0], B = topCols[1];
  const C = botCols[0], D = botCols[1];

  // D⁻¹  [n,j,j]
  //
  // Float32: add another ε·I before recursing into the j×j block.  The outer
  // ε·I in composeForward regularises the full m×m matrix, but after Schur
  // partitioning the extracted D sub-block can still be near-singular when
  // float32 rounding makes (C·J) diagonal entries approach −1.  One extra level
  // of regularisation here prevents single-iteration loss spikes.
  // jax-js-lint: allow-non-using
  const D_inv_src = (dtype === DType.Float32)
    ? (() => {
        using v  = np.array(1e-6, { dtype });
        using Ij = np.reshape(np.eye(j, undefined, { dtype }), [1, j, j]);
        return np.add(D, np.multiply(v, Ij));
      })()
    : D;
  using Dinv = adSafeInv(D_inv_src, j, dtype);
  if (D_inv_src !== D) D_inv_src.dispose();

  // B·D⁻¹  [n,k,j]
  using BDinv = np.einsum('nij,njk->nik', B, Dinv);

  // Schur complement S = A − B·D⁻¹·C  [n,k,k]
  using BDinvC = np.einsum('nij,njk->nik', BDinv, C);
  using S     = np.subtract(A, BDinvC);
  using Sinv  = adSafeInv(S, k, dtype);

  // R₀₁ = −S⁻¹·B·D⁻¹  [n,k,j]
  using neg1       = np.array(-1, { dtype });
  using SinvBDinv  = np.einsum('nij,njk->nik', Sinv, BDinv);
  using R01        = np.multiply(SinvBDinv, neg1);

  // D⁻¹·C  [n,j,k]
  using DinvC = np.einsum('nij,njk->nik', Dinv, C);

  // D⁻¹·C·S⁻¹  [n,j,k]  — used for both R₁₀ and R₁₁
  using DinvCSinv = np.einsum('nij,njk->nik', DinvC, Sinv);

  // R₁₀ = −D⁻¹·C·S⁻¹  [n,j,k]
  using R10 = np.multiply(DinvCSinv, neg1);

  // R₁₁ = D⁻¹ + D⁻¹·C·S⁻¹·B·D⁻¹  [n,j,j]
  using DinvCSinv_BDinv = np.einsum('nij,njk->nik', DinvCSinv, BDinv);
  using R11 = np.add(Dinv, DinvCSinv_BDinv);

  // Assemble [[S⁻¹, R₀₁], [R₁₀, R₁₁]]  [n,m,m]
  using R_top = np.concatenate([Sinv, R01], 2);   // [n,k,m]
  using R_bot = np.concatenate([R10,  R11], 2);   // [n,j,m]
  A.dispose(); B.dispose(); C.dispose(); D.dispose();
  return np.concatenate([R_top, R_bot], 1);        // [n,m,m]
}
