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
  /** Use full seasonal component with ns-1 states (default: false) */
  fullseas?: boolean;
  /** Number of trigonometric harmonic pairs; overrides fullseas if > 0 (default: 0) */
  trig?: number;
  /** Number of seasons per cycle (default: 12) */
  ns?: number;
  /** AR coefficients; adds AR(p) block where p = arphi.length (default: []) */
  arphi?: number[];
  /** Spline mode: modifies W for order=1 (default: false) */
  spline?: boolean;
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

export function dlmgensys(options: DlmOptions = {}): DlmSystem {
  const order = options.order ?? 1;
  const trig = options.trig ?? 0;
  const ns = options.ns ?? 12;
  const arphi = options.arphi ?? [];

  // If trig > 0, disable fullseas (MATLAB convention)
  const fullseas = trig > 0 ? false : (options.fullseas ?? false);

  if (trig > ns / 2) {
    throw new Error(`trig must be between 0 and ns/2, got trig=${trig} with ns=${ns}`);
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

  if (fullseas) {
    // Full seasonal: (ns-1)×(ns-1) companion matrix
    // First row all -1, subdiagonal = 1
    const seasSize = ns - 1;
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
  } else if (trig > 0) {
    // Trigonometric harmonics: k rotation blocks
    for (let k = 1; k <= trig; k++) {
      const angle = (2 * Math.PI * k) / ns;
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
    // If trig == ns/2 and ns is even, remove last element (redundant)
    if (trig === ns / 2) {
      const lastIdx = Gs.length - 1;
      Gs = Gs.slice(0, lastIdx).map(row => row.slice(0, lastIdx));
      Fs = Fs.slice(0, lastIdx);
    }
  }

  // ─── AR component ───
  let Gar: number[][] = [];
  let Far: number[] = [];
  const nar = arphi.length;
  if (nar > 0) {
    // Companion form: first column = arphi, rest = shifted identity
    // [a1, 1, 0, ...]
    // [a2, 0, 1, ...]
    // [ap, 0, 0, ...]
    Gar = Array.from({ length: nar }, (_, i) => {
      const row = new Array(nar).fill(0);
      row[0] = arphi[i];
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
