/**
 * Generate DLM system matrices G and F.
 *
 * Constructs the state transition matrix G and observation vector F for a
 * Dynamic Linear Model by combining polynomial trend, seasonal, and AR components.
 *
 * This is a direct port of the MATLAB `dlmgensys.m` function.
 *
 * @param options - Model specification
 * @returns System matrices G (m×m) and F (1×m), plus state dimension m
 */

/** DLM model configuration options */
export interface DlmOptions {
  /** Polynomial trend order: 0 = level, 1 = level+slope, 2 = quadratic (default: 1) */
  order?: number;
  /** Use full seasonal component with seasonLength-1 states (default: false). In MATLAB DLM, this is `fullseas`. */
  fullSeasonal?: boolean;
  /** Number of trigonometric harmonic pairs; overrides fullSeasonal if > 0 (default: 0). In MATLAB DLM, this is `trig`. */
  harmonics?: number;
  /** Number of seasons per cycle (default: 12). In MATLAB DLM, this is `ns`. */
  seasonLength?: number;
  /** AR coefficients; adds AR(p) block where p = arCoefficients.length (default: []). In MATLAB DLM, this is `arphi`. */
  arCoefficients?: number[];
  /** Spline mode: modifies W for order=1 (default: false) */
  spline?: boolean;
  /** Fit AR coefficients via MLE optimization (default: false). In MATLAB DLM, this is `fitar`. */
  fitAr?: boolean;
}

/**
 * Find the indices of AR states in the full state vector.
 *
 * These are the row/column positions in G where the AR coefficients
 * appear (first column of the AR companion block).
 *
 * @param options - Model specification with arphi
 * @returns Array of 0-based state indices, or empty if no AR component
 */
export function findArInds(options: DlmOptions): number[] {
  const arCoefficients = options.arCoefficients ?? [];
  const nar = arCoefficients.length;
  if (nar === 0) return [];

  const order = options.order ?? 1;
  const harmonics = options.harmonics ?? 0;
  const seasonLength = options.seasonLength ?? 12;
  const fullSeasonal = harmonics > 0 ? false : (options.fullSeasonal ?? false);

  let offset = order + 1;  // trend block size
  if (fullSeasonal) {
    offset += seasonLength - 1;
  } else if (harmonics > 0) {
    offset += Math.min(seasonLength - 1, harmonics * 2);
  }

  return Array.from({ length: nar }, (_, i) => offset + i);
}

/** Generated system matrices */
export interface DlmSystem {
  /** State transition matrix [m×m] */
  G: number[][];
  /** Observation vector [m] (row of F for p=1) */
  F: number[];
  /** State dimension */
  m: number;
}

/**
 * Block-diagonal concatenation of two matrices.
 * stack(A, B) = [ A  0 ]
 *               [ 0  B ]
 */
function stackMatrices(a: number[][], b: number[][]): number[][] {
  if (a.length === 0) return b;
  if (b.length === 0) return a;
  const ra = a.length, ca = a[0].length;
  const rb = b.length, cb = b[0].length;
  const result: number[][] = [];
  for (let i = 0; i < ra; i++) {
    result.push([...a[i], ...new Array(cb).fill(0)]);
  }
  for (let i = 0; i < rb; i++) {
    result.push([...new Array(ca).fill(0), ...b[i]]);
  }
  return result;
}

export function dlmGenSys(options: DlmOptions = {}): DlmSystem {
  const order = options.order ?? 1;
  const harmonics = options.harmonics ?? 0;
  const seasonLength = options.seasonLength ?? 12;
  const arCoefficients = options.arCoefficients ?? [];

  // If harmonics > 0, disable fullSeasonal (MATLAB convention)
  const fullSeasonal = harmonics > 0 ? false : (options.fullSeasonal ?? false);

  if (harmonics > seasonLength / 2) {
    throw new Error(`harmonics must be between 0 and seasonLength/2, got harmonics=${harmonics} with seasonLength=${seasonLength}`);
  }

  // ─── Polynomial trend component ───
  // G_trend = I + superdiagonal (Jordan block)
  // F_trend = [1, 0, ..., 0]
  const trendSize = order + 1;
  const Gt: number[][] = Array.from({ length: trendSize }, (_, i) => {
    const row = new Array(trendSize).fill(0);
    row[i] = 1;
    if (i + 1 < trendSize) row[i + 1] = 1;
    return row;
  });
  const Ft: number[] = new Array(trendSize).fill(0);
  Ft[0] = 1;

  // ─── Seasonal component ───
  let Gs: number[][] = [];
  let Fs: number[] = [];

  if (fullSeasonal) {
    // Full seasonal: (seasonLength-1)×(seasonLength-1) companion matrix
    // First row all -1, subdiagonal = 1
    const seasSize = seasonLength - 1;
    Gs = Array.from({ length: seasSize }, (_, i) => {
      const row = new Array(seasSize).fill(0);
      if (i === 0) {
        row.fill(-1);
      } else {
        row[i - 1] = 1;
      }
      return row;
    });
    Fs = new Array(seasSize).fill(0);
    Fs[0] = 1;
  } else if (harmonics > 0) {
    // Trigonometric harmonics: k rotation blocks
    for (let k = 1; k <= harmonics; k++) {
      const angle = (2 * Math.PI * k) / seasonLength;
      const cos_k = Math.cos(angle);
      const sin_k = Math.sin(angle);

      // Expand Gs with a 2×2 harmonic block
      const offset = Gs.length;
      const newSize = offset + 2;
      for (let i = 0; i < offset; i++) {
        Gs[i].push(0, 0);
      }
      const row1 = new Array(newSize).fill(0);
      const row2 = new Array(newSize).fill(0);
      row1[offset] = cos_k;
      row1[offset + 1] = sin_k;
      row2[offset] = -sin_k;
      row2[offset + 1] = cos_k;
      Gs.push(row1, row2);
      Fs.push(1, 0);
    }
    // If harmonics == seasonLength/2 and seasonLength is even, remove last element (redundant)
    if (harmonics === seasonLength / 2) {
      const lastIdx = Gs.length - 1;
      Gs = Gs.slice(0, lastIdx).map(row => row.slice(0, lastIdx));
      Fs = Fs.slice(0, lastIdx);
    }
  }

  // ─── AR component ───
  let Gar: number[][] = [];
  let Far: number[] = [];
  const nar = arCoefficients.length;
  if (nar > 0) {
    // Companion form: first column = arCoefficients, rest = shifted identity
    // [a1, 1, 0, ...]
    // [a2, 0, 1, ...]
    // [ap, 0, 0, ...]
    Gar = Array.from({ length: nar }, (_, i) => {
      const row = new Array(nar).fill(0);
      row[0] = arCoefficients[i];
      if (i + 1 < nar) row[i + 1] = 1;
      return row;
    });
    Far = new Array(nar).fill(0);
    Far[0] = 1;
  }

  // ─── Combine: block-diagonal G, concatenated F ───
  const G = stackMatrices(stackMatrices(Gt, Gs), Gar);
  const F = [...Ft, ...Fs, ...Far];
  const m = F.length;

  return { G, F, m };
}

// ─────────────────────────────────────────────────────────────────────────
// Time-varying system matrices for arbitrary timesteps
// ─────────────────────────────────────────────────────────────────────────

/**
 * Time-varying system matrices for non-uniform timesteps.
 */
export interface DlmSystemTV {
  /** Per-step transition matrices G(Δt_k) — shape [n, m, m] as nested JS arrays */
  G: number[][][];
  /** Per-step noise covariances W(Δt_k) — shape [n, m, m] as nested JS arrays */
  W: number[][][];
  /** Observation vector [m] (row of F for p=1) — same for all steps */
  F: number[];
  /** State dimension */
  m: number;
}

/**
 * Build an m×m zero matrix.
 */
function zeroMatrix(m: number): number[][] {
  return Array.from({ length: m }, () => new Array(m).fill(0));
}

/**
 * In-place block-diagonal write: write `block` into `target` at offset (r, c).
 */
function writeBlock(target: number[][], block: number[][], r: number, c: number): void {
  for (let i = 0; i < block.length; i++) {
    for (let j = 0; j < block[0].length; j++) {
      target[r + i][c + j] = block[i][j];
    }
  }
}

/**
 * Generate time-varying DLM system matrices G(Δt) and W(Δt) for non-uniform
 * timesteps, using closed-form continuous-time discretization.
 *
 * The underlying continuous-time model is:
 *   dx = F_c · x · dt + L · dw   (state SDE)
 *   y_k = F · x(t_k) + v_k       (discrete observations)
 *
 * where F_c is the continuous-time system matrix. The discrete-time matrices
 * at step k are:
 *   G_k = expm(F_c · Δt_k)                          (state transition)
 *   W_k = ∫₀^{Δt} expm(F_c·s) · Q_c · expm(F_c·s)' ds  (process noise cov)
 *
 * For polynomial trend and trigonometric harmonics, G_k and W_k have closed-form
 * expressions (no matrix exponential needed).
 *
 * **Supported components:** polynomial trend (order 0, 1, 2) and trigonometric
 * harmonics. Throws if fullSeasonal or AR components are requested (these are
 * purely discrete-time constructs with no natural continuous-time extension).
 *
 * @param options - Model specification (same as dlmGenSys)
 * @param timestamps - Observation times (length n). Timesteps Δt_k = t_{k} - t_{k-1}
 *   are computed from consecutive differences. The first Δt is t[1]-t[0] (t[0] is
 *   treated as the baseline; the first transition is from prior to t[0] so Δt[0] is
 *   set to 1.0, matching the uniform-timestep convention).
 * @param processStd - Process noise std devs (diagonal of √Q_c per unit time).
 *   Length determines which states have noise (same as DlmFitOptions.processStd).
 * @param spline - If true and order=1, use integrated random walk W(Δt) instead of
 *   diagonal W.
 * @returns Time-varying system matrices { G[n,m,m], W[n,m,m], F[m], m }
 */
export function dlmGenSysTV(
  options: DlmOptions,
  timestamps: number[],
  processStd: number[],
): DlmSystemTV {
  const order = options.order ?? 1;
  const harmonics = options.harmonics ?? 0;
  const seasonLength = options.seasonLength ?? 12;
  const arCoefficients = options.arCoefficients ?? [];
  const fullSeasonal = harmonics > 0 ? false : (options.fullSeasonal ?? false);

  // ── Validate: only polynomial + trig harmonics supported ──
  if (fullSeasonal) {
    throw new Error(
      'dlmGenSysTV: fullSeasonal is a purely discrete-time construct with no ' +
      'natural continuous-time extension. Use harmonics instead, or use ' +
      'uniform timesteps (omit timestamps).'
    );
  }
  if (arCoefficients.length > 0) {
    throw new Error(
      'dlmGenSysTV: AR components are discrete-time constructs. Arbitrary ' +
      'timesteps with AR are not supported. Consider the binary-power ' +
      'decomposition approach for future support.'
    );
  }

  const n = timestamps.length;
  if (n < 2) {
    throw new Error(`dlmGenSysTV: need at least 2 timestamps, got ${n}`);
  }

  // Use the static system to get F and m
  const sys = dlmGenSys(options);
  const m = sys.m;
  const F = sys.F;

  // Compute departing Δt per scan index.
  //
  // G_scan[k] and W_scan[k] encode the transition FROM the filtered state
  // at observation k TO the predicted state at observation k+1.  In the DLM
  // forward step the carry output is x_next = G_k · x_filt(k), so G_k must
  // use the time span *departing* from k to k+1:
  //
  //   dt[k] = T[k+1] − T[k]   for k = 0 … n−2
  //   dt[n−1] = 1.0            (last step: no observation follows; arbitrary)
  //
  // Validation of strictly-increasing timestamps is done while computing dt.
  const dt = new Array(n);
  for (let k = 0; k < n - 1; k++) {
    dt[k] = timestamps[k + 1] - timestamps[k];
    if (dt[k] <= 0) {
      throw new Error(
        `dlmGenSysTV: timestamps must be strictly increasing, but ` +
        `t[${k + 1}]=${timestamps[k + 1]} <= t[${k}]=${timestamps[k]}`
      );
    }
  }
  dt[n - 1] = 1.0;  // departing from last observation — no next obs

  // Component block sizes
  const trendSize = order + 1;
  let trigSize = 0;
  if (harmonics > 0) {
    trigSize = Math.min(seasonLength - 1, harmonics * 2);
    // If harmonics == seasonLength/2 and even seasonLength, one fewer
    if (harmonics === seasonLength / 2) trigSize -= 1;
  }

  const w = processStd;

  const G_arr: number[][][] = new Array(n);
  const W_arr: number[][][] = new Array(n);

  for (let k = 0; k < n; k++) {
    const d = dt[k];
    const G_k = zeroMatrix(m);
    const W_k = zeroMatrix(m);

    // ─── Trend block: expm of Jordan block ───
    // Order 0: G = [1], W = d·w₀²
    // Order 1: G = [[1,d],[0,1]], W depends on spline flag
    // Order 2: G = [[1,d,d²/2],[0,1,d],[0,0,1]], W = diag scaling
    for (let i = 0; i < trendSize; i++) {
      // G: upper triangular with d^j/j! on superdiagonal j
      for (let j = i; j < trendSize; j++) {
        const exp = j - i;  // superdiagonal offset
        G_k[i][j] = dPower(d, exp);
      }
    }

    // W for trend block — accumulated discrete-step noise
    //
    // The DLM applies x_{t+1} = G·x_t + w_t (w_t ~ N(0, W₁)) at each
    // integer timestep.  For a gap of Δt steps with no intermediate
    // observations, the total process noise is the sum:
    //   W(Δt) = Σ_{k=0}^{Δt-1} G^k · W₁ · (G^k)'
    //
    // Because G is a Jordan block (upper-triangular 1's), G^k has polynomial
    // entries in k and the sum has closed-form via Faulhaber sums:
    //   S₁ = Σk = d(d-1)/2,  S₂ = Σk² = d(d-1)(2d-1)/6, etc.
    //
    // These polynomial formulas interpolate smoothly for non-integer d and
    // collapse to the unit-step identity W(1) = W₁ for all orders.
    if (order === 0) {
      // G = [1], so W(d) = d · w₀²
      if (w.length > 0) W_k[0][0] = d * (w[0] ** 2);
    } else if (order === 1) {
      const w0sq = w.length > 0 ? w[0] ** 2 : 0;
      const w1sq = w.length > 1 ? w[1] ** 2 : 0;
      // Faulhaber sums
      const S1 = d * (d - 1) / 2;
      const S2 = d * (d - 1) * (2 * d - 1) / 6;
      W_k[0][0] = d * w0sq + w1sq * S2;
      W_k[0][1] = w1sq * S1;
      W_k[1][0] = w1sq * S1;
      W_k[1][1] = d * w1sq;
    } else if (order === 2) {
      const w0sq = w.length > 0 ? w[0] ** 2 : 0;
      const w1sq = w.length > 1 ? w[1] ** 2 : 0;
      const w2sq = w.length > 2 ? w[2] ** 2 : 0;
      const S1 = d * (d - 1) / 2;
      const S2 = d * (d - 1) * (2 * d - 1) / 6;
      const S3 = S1 * S1;  // [d(d-1)/2]²
      const S4 = d * (d - 1) * (2 * d - 1) * (3 * d * d - 3 * d - 1) / 30;
      W_k[0][0] = d * w0sq + w1sq * S2 + w2sq * S4 / 4;
      W_k[0][1] = w1sq * S1 + w2sq * S3 / 2;
      W_k[0][2] = w2sq * S2 / 2;
      W_k[1][0] = W_k[0][1];
      W_k[1][1] = d * w1sq + w2sq * S2;
      W_k[1][2] = w2sq * S1;
      W_k[2][0] = W_k[0][2];
      W_k[2][1] = W_k[1][2];
      W_k[2][2] = d * w2sq;
    }

    // ─── Trigonometric harmonics block ───
    let trigOffset = trendSize;
    if (harmonics > 0) {
      // If harmonics == ns/2, the last harmonic is a single cosine state
      const lastHarmonicIsSingle = (harmonics === seasonLength / 2);
      const fullPairs = lastHarmonicIsSingle ? harmonics - 1 : harmonics;

      for (let h = 1; h <= fullPairs; h++) {
        const angle = (2 * Math.PI * h) / seasonLength;
        const theta = d * angle;
        const cos_t = Math.cos(theta);
        const sin_t = Math.sin(theta);
        G_k[trigOffset][trigOffset] = cos_t;
        G_k[trigOffset][trigOffset + 1] = sin_t;
        G_k[trigOffset + 1][trigOffset] = -sin_t;
        G_k[trigOffset + 1][trigOffset + 1] = cos_t;
        // W: independent noise per harmonic pair, scaled by Δt
        const wIdx = trigOffset;
        if (wIdx < w.length) W_k[trigOffset][trigOffset] = d * (w[wIdx] ** 2);
        if (wIdx + 1 < w.length) W_k[trigOffset + 1][trigOffset + 1] = d * (w[wIdx + 1] ** 2);
        trigOffset += 2;
      }
      // Last harmonic if it's a single cosine (ns/2 with even ns)
      if (lastHarmonicIsSingle) {
        const angle = (2 * Math.PI * harmonics) / seasonLength;
        const theta = d * angle;
        G_k[trigOffset][trigOffset] = Math.cos(theta);
        if (trigOffset < w.length) W_k[trigOffset][trigOffset] = d * (w[trigOffset] ** 2);
        trigOffset += 1;
      }
    }

    G_arr[k] = G_k;
    W_arr[k] = W_k;
  }

  return { G: G_arr, W: W_arr, F, m };
}

/**
 * Compute d^exp / exp! (Taylor coefficient for matrix exponential of Jordan block).
 *   exp=0 → 1, exp=1 → d, exp=2 → d²/2, etc.
 */
function dPower(d: number, exp: number): number {
  if (exp === 0) return 1;
  if (exp === 1) return d;
  if (exp === 2) return (d * d) / 2;
  // General case (order > 2 not currently supported, but future-proof)
  let val = 1;
  for (let i = 1; i <= exp; i++) val *= d / i;
  return val;
}
