# dlm-js API Overhaul Plan

## Design principles

1. **JAX-idiomatic axis order**: time axis 0, state axis 1 — matches `lax.scan`, numpy, Equinox, and the internal tensor layout already in use.
2. **Two-tier output**: on-device `np.Array` tensors (power users, MLE internals) and materialized TypedArrays (typical JS/TS users).
3. **Options objects** for all function parameters beyond `y`.
4. **Zero-cost materialization**: the internal `[n, m]` row-major layout from `consumeData()` is used directly — no transpose.
5. **Abstract away jax-js-nonconsuming** for casual users; expose it cleanly for power users.
6. **No streaming/incremental API**: incompatible with the `lax.scan` / `associativeScan` parallel model; see §11.
7. **Forward-compatible with time-varying system matrices**: current G, W are constant; the API and internal structures must accommodate per-timestep `G_t`, `W_t` for arbitrary time steps (see §13).
8. **Composable loss functions**: the tensor API and loss function design should support custom losses for MAP estimation and future MCMC (see §14, §15).
9. **JS-idiomatic naming**: replace single-letter and MATLAB-abbreviated field names with descriptive camelCase names. System matrices (G, F, W) retain their standard notation. Provide a `toMatlab()` translator that restores both MATLAB names and MATLAB axis layout (see §5, §17).

---

## 1. Axis conventions

| Context | States | Covariance | Rationale |
|---------|--------|------------|-----------|
| Internal jax-js tensors | `[n, m]` | `[n, m, m]` | Required by `lax.scan` (time = scan axis 0) |
| On-device API (`DlmTensorResult`) | `[n, m]` np.Array | `[n, m, m]` np.Array | Zero-cost passthrough from jit output |
| Materialized API (`DlmFitResult`) | `[n, m]` flat buffer | `[n, m, m]` flat buffer | Same row-major layout as `consumeData()` — no transpose |
| MATLAB compat helper | `[m][n]` nested arrays | `[m][m][n]` nested arrays | On-demand transpose for migration users only |

**Key change**: the current code performs an O(n·m) transpose per state component after `consumeData()` to produce the MATLAB-like `x[state][time]` layout. The redesign eliminates this entirely — `consumeData()` already returns `[n, m]` row-major data; we wrap it directly.

**Inconsistency fixed**: the current API has `x[state][time]` but `xstd[time][state]` — a direct inheritance of MATLAB's own axis inconsistency between `out.x` (m×n) and `out.xstd` (n×m). The redesign uses `[n, m]` (time-major) consistently for all state and std-dev outputs, eliminating the trap.

---

## 2. On-device tensor API

For power users who work with jax-js directly and for internal MLE use. Returns `np.Array` tensors that stay on-device. No data transfer to JS.

```ts
interface DlmTensorResult extends Disposable {
  // States (see §17 for naming rationale)
  smoothed: np.Array;       // [n, m] — smoothed states
  filtered: np.Array;       // [n, m] — filtered states
  smoothedCov: np.Array;    // [n, m, m] — smoothed covariances
  filteredCov: np.Array;    // [n, m, m] — filtered covariances
  smoothedStd: np.Array;    // [n, m] — state std devs (sqrt of diagonal)

  // Observation diagnostics — [n]
  yhat: np.Array;
  ystd: np.Array;
  innovations: np.Array;
  innovationVar: np.Array;
  rawResiduals: np.Array;
  scaledResiduals: np.Array;
  standardizedResiduals: np.Array;

  // Scalars
  deviance: np.Array;        // -2 · log-likelihood
  residualVariance: np.Array;
  mse: np.Array;
  mape: np.Array;
  rss: np.Array;             // residual sum of squares
  nobs: np.Array;

  // Metadata (plain JS)
  n: number;
  m: number;
}
```

Usage:
```ts
import { dlmFitTensor } from 'dlm-js';

{
  using result = await dlmFitTensor(y, { obsStd: 120, processStd: [40, 10], order: 1 });
  // result.smoothed is np.Array [n, m] — stays on device
  const trend = np.slice(result.smoothed, [0, 0], [n, 1]);  // level component [n, 1]
  // Auto-disposed at block end
}
```

**Implementation**: the existing `dlmSmo` already returns `[n, m, 1]` and `[n, m, m]` tensors from jit. The tensor API just squeezes the trailing `1` on the state tensor and returns — no `consumeData()` call at all. Current `dlmFit` is `dlmSmo` + `consumeData()` + transpose; the tensor API is just `dlmSmo` + `squeeze`.

**MLE benefit**: `dlmMLE` calls `dlmFit` at the end to produce the final fit. With `dlmFitTensor`, that final call stays on-device. The materialization to JS `StateMatrix` / `FloatArray` happens once at the very end when constructing `DlmMleResult.fit`, using zero-copy buffer wrapping (see §3).

**Future**: `dlmFitTensor` and `dlmMLE` can also accept `y: np.Array` to skip the initial data upload if observations are already on-device.

---

## 3. Materialized API — `StateMatrix` and `CovMatrix`

Lightweight wrappers around flat row-major TypedArray buffers. Single contiguous allocation per matrix. No intermediate data copies on construction.

```ts
class StateMatrix {
  /** Flat row-major [n, m] buffer */
  readonly data: FloatArray;
  readonly n: number;
  readonly m: number;

  /** All states at time t — zero-copy subarray view, length m */
  at(t: number): FloatArray;

  /** Time series of state i across all timesteps — copied, length n */
  series(i: number): FloatArray;

  /** Single element */
  get(t: number, i: number): number;
}

class CovMatrix {
  /** Flat row-major [n, m, m] buffer */
  readonly data: FloatArray;
  readonly n: number;
  readonly m: number;

  /** m×m covariance at time t — zero-copy subarray view, length m*m, row-major */
  at(t: number): FloatArray;

  /** Single element */
  get(t: number, i: number, j: number): number;

  /** Var(state_i) at time t */
  variance(t: number, i: number): number;

  /** Time series of Cov(i, j) across all t — copied, length n */
  series(i: number, j: number): FloatArray;
}
```

**`DlmFitResult`:**
```ts
interface DlmFitResult {
  // State estimates (see §17 for naming rationale)
  smoothed: StateMatrix;       // [n, m] smoothed states
  filtered: StateMatrix;       // [n, m] filtered states
  smoothedCov: CovMatrix;      // [n, m, m] smoothed covariances
  filteredCov: CovMatrix;      // [n, m, m] filtered covariances
  smoothedStd: StateMatrix;    // [n, m] smoothed state std devs

  // 1-D time series (plain FloatArray, length n)
  yhat: FloatArray;
  ystd: FloatArray;
  innovations: FloatArray;
  innovationVar: FloatArray;
  rawResiduals: FloatArray;
  scaledResiduals: FloatArray;
  standardizedResiduals: FloatArray;

  // Scalars
  deviance: number;              // -2 · log-likelihood
  residualVariance: number;
  mse: number;
  mape: number;
  rss: number;                   // residual sum of squares
  nobs: number;

  // Model matrices (standard notation — see §17)
  G: number[][];                 // state transition [m × m]
  F: number[];                   // observation vector [m]
  W: number[][];                 // state noise covariance [m × m]
  initialState: number[];        // x₀ after first smoother pass
  initialCov: number[][];        // C₀ (scaled)
  y: FloatArray;                 // observations
  obsNoise: FloatArray;          // observation noise std devs
  covariates: number[][];        // covariate matrix X [n × q] (empty when q = 0)

  // Shape
  n: number;
  m: number;
}
```

**Materialization becomes zero-cost:**
```ts
// Current code — O(n·m) transpose per state:
const x_out: FloatArray[] = Array.from({ length: m }, (_, k) => {
  const arr = new FA(n);
  for (let t = 0; t < n; t++) arr[t] = x_raw[t * m + k];
  return arr;
});

// New code — O(1) buffer wrap, no copy:
const x_out = new StateMatrix(new FA(x_raw), n, m);
// x_raw is already [n, m] row-major from consumeData() — wrap directly.
```

**Common usage patterns:**
```ts
const fit = await dlmFit(y, { obsStd: 120, processStd: [40, 10], order: 1 });

// Level trend for plotting (common case):
const level = fit.smoothed.series(0);        // FloatArray [n]
plotLine(level);

// All states at a specific timestep (zero-copy view):
const stateVec = fit.smoothed.at(42);       // FloatArray view [m]

// State uncertainty:
const levelStd = fit.smoothedStd.series(0); // FloatArray [n]

// Raw buffer for bulk GPU upload:
fit.smoothed.data                            // FloatArray of length n*m, row-major [n, m]

// Observation predictions (unchanged — already FloatArray [n]):
plotBand(fit.yhat, fit.ystd);
```

**Comparison to current API:**
```ts
// CURRENT — MATLAB names, state-major, raw arrays, inconsistent xstd
fit.x[0]        // FloatArray [n] — zero-copy (convenient)
fit.x[0][42]    // state 0 at time 42
fit.xstd[42]    // FloatArray [m] — INCONSISTENT axis order vs x!
fit.Cp          // innovation covariances — cryptic
fit.ssy         // sum of squared residuals — cryptic
fit.V           // observation noise std devs — confusing (V usually = variance)
fit.XX          // covariates — MATLAB-ism

// NEW — JS-idiomatic names, time-major, StateMatrix, consistent
fit.smoothed.series(0) // FloatArray [n] — descriptive and consistent
fit.smoothed.get(42, 0)    // state 0 at time 42
fit.smoothed.at(42)        // FloatArray [m] — zero-copy view
fit.smoothedStd.at(42)     // FloatArray [m] — SAME convention as smoothed ✓
fit.smoothedStd.series(0)  // FloatArray [n] — level std dev time series
fit.innovationVar          // self-documenting
fit.rss                    // universally understood
fit.obsNoise               // clear: observation noise std devs
fit.covariates             // descriptive
```

`DlmForecastResult` uses the same `StateMatrix` and `CovMatrix` classes with `h` replacing `n`.

---

## 4. Function signatures — options objects

**`dlmFit`:**
```ts
interface DlmFitOptions {
  // Noise (required) — see §17 for naming rationale
  obsStd: number | ArrayLike<number>;      // observation noise std dev (scalar or per-obs)
  processStd: number[];                     // process noise std devs (diagonal of √W)

  // Model (optional, defaults to local linear trend)
  order?: number;              // polynomial trend order: 0, 1, 2 — default: 1
  harmonics?: number;          // trigonometric harmonic pairs (was: trig)
  seasonLength?: number;       // seasons per cycle (was: ns) — default: 12
  fullSeasonal?: boolean;      // full seasonal component (was: fullseas)
  arCoefficients?: number[];   // AR coefficients (was: arphi)
  spline?: boolean;            // spline mode for order=1

  // Covariates
  X?: ArrayLike<number>[];     // n rows × q cols

  // Arbitrary time steps (see §13)
  dt?: ArrayLike<number>;      // [n] inter-observation intervals

  // Runtime
  dtype?: 'f32' | 'f64';              // default: 'f64'
  algorithm?: 'scan' | 'assoc';       // default: auto-select from device/dtype
  stabilization?: 'joseph' | 'none';  // default: auto-select from dtype
}

async function dlmFit(
  y: ArrayLike<number>,
  opts: DlmFitOptions,
): Promise<DlmFitResult>

async function dlmFitTensor(
  y: ArrayLike<number> | np.Array,
  opts: DlmFitOptions,
): Promise<DlmTensorResult>
```

**`dlmMLE`:**
```ts
interface DlmMleOptions {
  // Model (same names as DlmFitOptions — see §17)
  order?: number;
  harmonics?: number;
  seasonLength?: number;
  fullSeasonal?: boolean;
  arCoefficients?: number[];
  fitAr?: boolean;              // fit AR coefficients via MLE (was: fitar)

  // Covariates
  X?: ArrayLike<number>[];

  // Arbitrary time steps (see §13)
  dt?: ArrayLike<number>;

  // Loss function (see §14)
  loss?: 'ml' | DlmLossFn;     // default: 'ml' (standard Kalman prediction-error likelihood)

  // Optimizer
  maxIter?: number;             // default: 200
  lr?: number;                  // Adam learning rate — default: 0.05
  tol?: number;                 // convergence tolerance — default: 1e-6
  init?: { obsStd?: number; processStd?: number[]; arCoefficients?: number[] };
  adamOpts?: { b1?: number; b2?: number; eps?: number };  // default b2=0.9
  obsStdFixed?: ArrayLike<number>;   // per-obs σ (fixes V; obsStd not estimated) (was: sFixed)
  callbacks?: {
    onInit?: (theta: FloatArray) => void;
    onIteration?: (iter: number, theta: FloatArray, deviance: number) => void;
  };

  // Runtime
  dtype?: 'f32' | 'f64';
  algorithm?: 'scan' | 'assoc';
}

async function dlmMLE(
  y: ArrayLike<number>,
  opts?: DlmMleOptions,
): Promise<DlmMleResult>
```

**`dlmForecast`:**
```ts
interface DlmForecastOptions {
  dtype?: 'f32' | 'f64';
  X?: ArrayLike<number>[];    // future covariate rows [h × q]
  dt?: ArrayLike<number>;     // [h] inter-step intervals for forecast (see §13)
}

async function dlmForecast(
  fit: DlmFitResult,
  obsStd: number,
  h: number,
  opts?: DlmForecastOptions,
): Promise<DlmForecastResult>
```

**`dtype` abstraction**: `'f32' | 'f64'` string literals map internally to `DType.Float32` / `DType.Float64`. Users never need to import from `@hamk-uas/jax-js-nonconsuming` for routine use. Power users who work with jax-js tensors directly still import what they need from jax-js-nonconsuming for their own code; dlm-js just stops requiring it as part of the basic `dlmFit` call.

**Algorithm auto-selection** (unchanged from current behavior):
- `'f64'` → `scan`, `none`
- `'f32'` on cpu/wasm → `scan`, `joseph`
- `'f32'` on webgpu → `assoc`, `none` (assoc has built-in stability)
- `algorithm: 'assoc'` + `stabilization: 'joseph'` → runtime error (invalid combination)

---

## 5. MATLAB DLM compat helpers

`toMatlab()` converts both **field names** and **axis layout** from the new JS-idiomatic API to MATLAB DLM conventions. This is a single function that fully restores the MATLAB experience:

```ts
interface DlmFitResultMatlab {
  // State estimates — MATLAB DLM names + state-major layout
  x: FloatArray[];       // x[stateIdx][timeIdx] — like MATLAB m×n
  xf: FloatArray[];
  C: FloatArray[][];     // C[i][j][timeIdx]
  Cf: FloatArray[][];
  xstd: FloatArray[];    // xstd[timeIdx][stateIdx] — MATLAB's n×m layout

  // Diagnostics — MATLAB DLM names
  v: FloatArray;         // innovations
  Cp: FloatArray;        // innovation covariances
  resid0: FloatArray;    // raw residuals
  resid: FloatArray;     // scaled residuals
  resid2: FloatArray;    // standardized residuals

  // Scalars — MATLAB DLM names
  lik: number;           // -2 · log-likelihood
  s2: number;            // residual variance
  ssy: number;           // sum of squared residuals

  // Model matrices — same letters
  G: number[][];
  F: number[];
  W: number[][];
  V: FloatArray;         // observation noise std devs (MATLAB name)
  x0: number[];          // initial state
  C0: number[][];        // initial covariance
  XX: number[][];        // covariates

  // Pass-through
  y: FloatArray;
  yhat: FloatArray;
  ystd: FloatArray;
  mse: number;
  mape: number;
  nobs: number;
  n: number;
  m: number;
  class: 'dlmfit';       // MATLAB-style class tag
}

/** Convert JS-idiomatic DlmFitResult to MATLAB DLM layout + names */
function toMatlab(result: DlmFitResult): DlmFitResultMatlab;
```

This creates the transpose copies that current `dlmFit` does eagerly — but only on explicit request. MATLAB DLM migration users call this once; everyone else never pays the cost.

A corresponding `toMatlabMle(result: DlmMleResult): DlmMleResultMatlab` restores `s`, `w`, `arphi`, `lik`, `likHistory` from their JS-idiomatic counterparts.

The Octave-comparison test harness calls `toMatlab()` internally before comparing against reference JSON, keeping numerical test logic unchanged.

---

## 6. Impact on MLE training loop

**Current data flow:**
```
JS data → np.Array → jit(optimStep) loop (on-device) → theta to JS →
  dlmFit() → np.Array (recreate G,W,F) → dlmSmo (jit) →
  consumeData() → O(n·m²) transpose → DlmFitResult
```

**New data flow:**
```
JS data → np.Array → jit(optimStep) loop (on-device) → theta to JS →
  dlmFitTensor() → np.Array (recreate G,W,F) → dlmSmo (jit) →
  squeeze → DlmTensorResult → zero-copy wrap → DlmFitResult
```

Concrete benefits:
- No transpose in the final fit — `DlmTensorResult` wraps jit output directly.
- Materialization is O(1) buffer wrapping, not O(n·m²) copy loops.
- Path is open to `y: np.Array` inputs for fully on-device MLE workflows.

---

## 7. `DlmRunConfig` as a real type

The README currently documents `DlmRunConfig` with `algorithm` and `stabilization` fields, but the actual code uses a bare `forceAssocScan?: boolean`. The redesign makes this a first-class part of each options object (inlined, not nested) and removes the undocumented `forceAssocScan` escape hatch:

```ts
// In DlmFitOptions, DlmMleOptions, DlmForecastOptions:
dtype?: 'f32' | 'f64';
algorithm?: 'scan' | 'assoc';
stabilization?: 'joseph' | 'none';
```

---

## 8. `dlmGenSys` — options renamed, output unchanged

`dlmGenSys` returns `{ G: number[][], F: number[], m: number }`. Output is already idiomatic; no axis-ordering issues.

**Input options renamed** to match `DlmFitOptions` (see §17): `trig` → `harmonics`, `ns` → `seasonLength`, `fullseas` → `fullSeasonal`, `arphi` → `arCoefficients`, `fitar` → `fitAr`.

`findArInds` input options change correspondingly.

---

## 9. Breaking changes summary

| Change | What breaks | Migration path |
|--------|-------------|----------------|
| **Naming** (§17) | | |
| `x` → `smoothed` | `fit.x[0]` | `fit.smoothed.series(0)` or `toMatlab(fit).x[0]` |
| `xf` → `filtered` | `fit.xf[0]` | `fit.filtered.series(0)` or `toMatlab(fit).xf[0]` |
| `C` → `smoothedCov` | `fit.C[i][j]` | `fit.smoothedCov.series(i, j)` or `toMatlab(fit).C[i][j]` |
| `Cf` → `filteredCov` | `fit.Cf[i][j]` | `fit.filteredCov.series(i, j)` |
| `xstd` → `smoothedStd` | `fit.xstd[t]` | `fit.smoothedStd.at(t)` |
| `v` → `innovations` | `fit.v` | `fit.innovations` or `toMatlab(fit).v` |
| `Cp` → `innovationVar` | `fit.Cp` | `fit.innovationVar` |
| `resid0/resid/resid2` → descriptive | `fit.resid0` | `fit.rawResiduals` etc. |
| `lik` → `deviance` | `fit.lik` | `fit.deviance` or `toMatlab(fit).lik` |
| `s2` → `residualVariance` | `fit.s2` | `fit.residualVariance` |
| `ssy` → `rss` | `fit.ssy` | `fit.rss` or `toMatlab(fit).ssy` |
| `V` → `obsNoise` | `fit.V` | `fit.obsNoise` |
| `x0` → `initialState` | `fit.x0` | `fit.initialState` |
| `C0` → `initialCov` | `fit.C0` | `fit.initialCov` |
| `XX` → `covariates` | `fit.XX` | `fit.covariates` |
| `class` field removed | `fit.class` | Use `instanceof` or `toMatlab(fit).class` |
| `s`/`w` → `obsStd`/`processStd` input names | `dlmFit(y, s, w, ...)` | `{ obsStd: s, processStd: w }` |
| `ns` → `seasonLength` | `{ ns: 12 }` | `{ seasonLength: 12 }` |
| `trig` → `harmonics` | `{ trig: 3 }` | `{ harmonics: 3 }` |
| `fullseas` → `fullSeasonal` | `{ fullseas: true }` | `{ fullSeasonal: true }` |
| `arphi` → `arCoefficients` | `{ arphi: [0.5] }` | `{ arCoefficients: [0.5] }` |
| `fitar` → `fitAr` | `{ fitar: true }` | `{ fitAr: true }` |
| `sFixed` → `obsStdFixed` | `sFixed: [...]` | `obsStdFixed: [...]` |
| MLE result `s`/`w`/`lik`/`jitMs` | `mle.s`, `mle.w`, `mle.lik` | `mle.obsStd`, `mle.processStd`, `mle.deviance` |
| **Structure** | | |
| `StateMatrix` / `CovMatrix` wrappers | direct array indexing | `.series()`, `.at()`, `.get()` |
| All args in options bag | Positional `(y, s, w, dtype, opts, X)` | `(y, { obsStd, processStd, dtype, ...opts, X })` |
| `dtype: DType` → `dtype: 'f32' \| 'f64'` | `DType.Float64` import from jax-js | `'f64'` string |
| `forceAssocScan` → `algorithm: 'assoc'` | Internal boolean flag | `{ algorithm: 'assoc' }` |
| New function `dlmFitTensor` | n/a — new addition | — |
| `toMatlabLayout` → `toMatlab` | n/a — new function | — |

---

## 10. What stays the same

- `dlmGenSys` output shape (`{ G, F, m }`) — unchanged
- `findArInds` — unchanged
- `yhat`, `ystd` field names — universally understood, keep as-is
- `mse`, `mape`, `nobs` — universally understood abbreviations
- `G`, `F`, `W` result fields — standard system-matrix notation (well-documented via JSDoc)
- Internal numerics, all three execution branches, tolerances — no algorithmic changes
- `DlmMleResult` fields `iterations`, `elapsed`, `fit` — already descriptive; `fit` field becomes the new `DlmFitResult` shape

---

## 11. Streaming/incremental API — explicitly out of scope

`lax.scan` and `lax.associativeScan` process the full time axis as a single traced operation compiled by jit. Breaking this into incremental updates would either:
- (a) Exit jit between updates — losing compilation and defeating the WASM/WebGPU performance model, or
- (b) Maintain jit-compiled state across calls — not supported by jax-js-nonconsuming.

The parallel-scan model's value (O(log n) depth, GPU efficiency, JIT compilation) comes from seeing the whole series at once. For real-time use, the practical pattern is to re-run `dlmFit` on the extended series — at n < 10k on WASM this is ~150ms, fast enough for interactive dashboards.

---

## 12. Future work enabled by this design

| Feature | How this design enables it |
|---------|----------------------------|
| Batched fitting (B pixels) | `y: np.Array [B, n]` input to `dlmFitTensor`; scan axis remains 0; batch dim prepends naturally as `[B, n, m]` tensors |
| Fully on-device MLE | `dlmFitTensor` + `y: np.Array` input avoids JS↔GPU data movement entirely |
| Multivariate observations (p > 1) | `StateMatrix` / `CovMatrix` wrappers are p-agnostic; the filter core would expand from `[n]` to `[n, p]` diagnostics |
| Arbitrary time steps | `dt` option + time-varying G_t, W_t threaded through scan inputs; see §13 |
| MCMC sampling | On-device tensor API + custom loss → stochastic gradient MCMC feasible; see §15 |
| MAP estimation | Custom loss with log-prior penalty; see §14 |

---

## 13. Arbitrary time steps

### Motivation

The current DLM assumes uniform time spacing: G and W are constant across all timesteps. Many real-world time series — satellite observations with cloud gaps, irregular sensor readings, financial tick data — have non-uniform observation times. Supporting arbitrary time steps requires making G and W time-varying.

### Background: continuous-time formulation

For a continuous-time linear state-space model:

$$\dot{x}(t) = F_c \, x(t) + L \, w(t), \quad y(t_k) = H \, x(t_k) + v_k$$

the discrete-time transition matrices between observation times $t_{k-1}$ and $t_k$ with spacing $\Delta t_k = t_k - t_{k-1}$ are:

$$G_k = \exp(F_c \, \Delta t_k), \quad W_k = \int_0^{\Delta t_k} \exp(F_c \, \tau) \, L \, Q_c \, L^\top \, \exp(F_c \, \tau)^\top \, d\tau$$

For the polynomial trend models used in dlm-js (order 0, 1, 2), these have closed-form expressions. For order = 1 (local linear trend):

$$G_k = \begin{pmatrix} 1 & \Delta t_k \\ 0 & 1 \end{pmatrix}, \quad W_k = q_c \begin{pmatrix} \Delta t_k^3/3 & \Delta t_k^2/2 \\ \Delta t_k^2/2 & \Delta t_k \end{pmatrix}$$

where $q_c$ is the continuous-time spectral density of the process noise.

Razavi, García-Fernández & Särkkä (2025) [1] extend the parallel associative scan framework from Särkkä & García-Fernández (2020) [2] to continuous-time models. Their key insight is that the sub-interval solutions (obtained via ODE integration within each block) produce associative elements with the same 5-tuple structure — so the prefix scan composition rules are unchanged. This means the `assoc` path in dlm-js can support arbitrary time steps with no changes to the scan composition logic — only the per-timestep element construction changes.

### API design

A `dt` option in `DlmFitOptions` provides the inter-observation intervals:

```ts
const fit = await dlmFit(y, { obsStd: 120, processStd: [40, 10], order: 1 });
```

When `dt` is absent, all spacings are 1 (current behavior — uniform time steps).

### Internal architecture

When `dt` is provided, the internal flow changes:

1. **`dlmGenSys` extended**: a new function `dlmGenSysTV(options, dt)` returns per-timestep `G_t[n, m, m]` and `W_t[n, m, m]` tensors instead of constant matrices.
2. **`dlmSmo` change**: G and W become scan inputs `[n, m, m]` threaded alongside `FF_scan`, `y_arr`, `V2_arr` — instead of constants captured by closure. The `lax.scan` step function receives `(G_t, W_t)` per step.
3. **`associativeScan` path**: the per-timestep 5-tuple elements already use G and W to construct $(A_k, b_k, C_k, \eta_k, J_k)$. Making them time-varying only changes element construction — the Lemma 2 composition rule is unchanged, as shown in [1].
4. **`dlmForecast`**: accepts `dt` for forecast-step intervals.

The key constraint is that G_t and W_t must be **pre-computed as full `[n, m, m]` tensors** before entering the scan. This is necessary because `lax.scan` requires fixed-shape inputs and cannot branch on per-step metadata. For polynomial trend + trigonometric seasonal models, the closed-form matrix exponential is cheap; for more complex continuous-time models, the ODE integration to produce each sub-interval's G_t and W_t can be done in parallel (embarrassingly parallel across timesteps), then the result fed into the scan.

### Parameterization for MLE with arbitrary time steps

When `dt` is provided, the noise parameters `processStd` in `DlmMleOptions` are reinterpreted as **continuous-time spectral densities** $q_c$ rather than discrete-time standard deviations. The per-timestep $W_k = f(q_c, \Delta t_k)$ is recomputed from $q_c$ and $\Delta t_k$ inside the loss function. Since the closed-form $W_k(\Delta t_k, q_c)$ is differentiable w.r.t. $q_c$, autodiff propagates through the matrix construction naturally.

### Impact on scan input sizes

Threading `G_t[n, m, m]` and `W_t[n, m, m]` as scan inputs increases memory from O(m²) (captured constants) to O(n·m²) (stacked per step). For typical dlm-js models (m ≤ 13, n ≤ 10k), this is 13² × 10k × 8 bytes ≈ 13 MB — well within budget. For future large-n or high-m applications, a hybrid approach could compute G_t and W_t inside the scan step from a compact `dt[n]` vector + constant $F_c$, avoiding the full stack.

### References

- [1] Razavi, García-Fernández & Särkkä (2025). "Temporal Parallelisation of Continuous-Time MAP Trajectory Estimation." [arXiv:2512.13319](https://arxiv.org/abs/2512.13319)
- [2] Särkkä & García-Fernández (2020). "Temporal Parallelization of Bayesian Smoothers." IEEE TAC 66(1).

---

## 14. Custom loss functions and MAP estimation

### Motivation

The current MLE minimizes the standard Kalman prediction-error likelihood $-2 \log L$. For regularized estimation (MAP), users want to add a log-prior penalty on parameters — for example, a log-normal prior on $W$ entries to prevent degenerate solutions on real data (the ozone problem documented in the README). MATLAB DLM supports this via MCMC with priors; dlm-js currently offers MLE only.

### Design

The loss function used by `dlmMLE` becomes a configurable component:

```ts
/**
 * Custom loss function type.
 * Receives the traceable Kalman loss (scalar np.Array) and the current
 * parameter vector theta (np.Array), and returns a scalar np.Array loss.
 * Must be AD-safe: only use jax-js ops on the inputs.
 */
type DlmLossFn = (kalmanLoss: np.Array, theta: np.Array) => np.Array;
```

Usage in `DlmMleOptions`:
```ts
// Standard MLE (default):
const mle = await dlmMLE(y, { order: 1 });

// MAP with log-normal prior on W entries:
const mle = await dlmMLE(y, {
  order: 1,
  loss: (deviance, theta) => {
    // theta = [log(obsStd), log(processStd0), log(processStd1), ...]
    // Add N(0, σ²) prior on log(processStd) entries (= log-normal prior on processStd)
    const logW = np.slice(theta, [1], [3]);   // processStd entries
    const prior = np.sum(np.square(logW));     // -2 log p(processStd)
    return np.add(deviance, np.multiply(np.array(0.1), prior));
  },
});
```

### Architecture

The loss function is injected into the `jit(valueAndGrad(...))` wrapper:

```
Current:  jit(valueAndGrad(makeKalmanLoss(...)))(theta)
New:      jit(valueAndGrad(theta => userLoss(makeKalmanLoss(theta), theta)))(theta)
```

When `loss: 'ml'` (default), the wrapper is identity: `(deviance, _theta) => deviance`. The entire chain — Kalman scan + custom penalty + AD backward pass + Adam update — remains inside a single `jit()` call. No performance change for the default path.

### Why this is sufficient for MAP

MAP estimation is just MLE with a prior penalty added to the loss. The custom loss hook lets users express arbitrary differentiable priors without changing any internals. For the ozone regularization case, a 3-line loss function replaces what currently requires MCMC in MATLAB DLM.

---

## 15. MCMC sampling

### Motivation

Full Bayesian posterior inference provides uncertainty quantification over parameters (not just states). MATLAB DLM supports this via Adaptive Metropolis (`mcmcrun`). For dlm-js, MCMC is a near-term goal.

### Feasibility within jax-js

MCMC is compatible with the parallel scan model — each MCMC iteration runs a full `dlmFit` (or `dlmFitTensor`) to evaluate the posterior, and iterations are inherently sequential (each proposal depends on the previous accept/reject). The parallel benefit comes from each iteration's filter/smoother being fast.

Two practical approaches:

**1. Stochastic Gradient MCMC (SGLD / SGHMC)**

Uses the existing `valueAndGrad` infrastructure. Instead of Adam updates, the gradient is used for a Langevin dynamics step with injected noise:

$$\theta_{k+1} = \theta_k - \epsilon \nabla \log p(\theta | y) + \sqrt{2\epsilon} \, \eta_k, \quad \eta_k \sim \mathcal{N}(0, I)$$

This fits directly into the existing jitted optimization loop — replace the Adam update with a Langevin step. The `jit(valueAndGrad(loss))` wrapper is reused. The custom loss function (§14) supplies the prior, making the gradient $\nabla(-\log p(\theta | y))$.

**Pros**: trivial implementation delta from MLE; reuses JIT compilation; posterior samples are a natural extension of the MLE training loop.

**Cons**: SGLD is approximate — the discretization error means samples are not from the exact posterior. Diagnostics (effective sample size, R-hat) are needed.

**2. Metropolis-Hastings with dlmFitTensor**

A more classical approach: propose $\theta' \sim q(\theta' | \theta)$, evaluate $\log p(\theta' | y)$ via `dlmFitTensor` (which returns `deviance` as an on-device scalar), accept/reject. The on-device tensor API avoids materializing the full fit result on each MCMC iteration — only `deviance` is read back to JS.

```ts
// Sketch of MH loop using existing tensor API:
for (let k = 0; k < nSamples; k++) {
  const thetaPrime = propose(theta);
  using fit = await dlmFitTensor(y, buildOpts(thetaPrime));
  const devPrime = (await fit.deviance.data())[0];
  const logPrior = evaluatePrior(thetaPrime);
  if (Math.log(Math.random()) < logPosterior(devPrime, logPrior) - logPosteriorCurrent) {
    theta = thetaPrime;
    // ...
  }
}
```

**Pros**: exact posterior (up to MCMC mixing); conceptually simple.

**Cons**: each iteration re-enters jit (no cross-iteration compilation benefit); slower than SGLD for large models because `dlmFitTensor` includes the full two-pass smoother, not just the forward filter needed for likelihood.

### API sketch

```ts
interface DlmMcmcOptions extends DlmMleOptions {
  method?: 'sgld' | 'mh';     // default: 'sgld'
  nSamples?: number;           // default: 1000
  burnIn?: number;             // default: 200
  prior?: DlmLossFn;           // reuses the custom loss type from §14
  proposalScale?: number;      // for MH: proposal std dev
  // ... diagnostics callbacks
}

interface DlmMcmcResult {
  samples: {
    obsStd: Float64Array;          // [nSamples]
    processStd: Float64Array[];    // [m][nSamples]
    arCoefficients?: Float64Array[]; // [p][nSamples]
  };
  acceptance: number;            // acceptance rate (MH only)
  fit: DlmFitResult;             // fit at posterior mean
  devianceHistory: Float64Array;
}

async function dlmMCMC(
  y: ArrayLike<number>,
  opts?: DlmMcmcOptions,
): Promise<DlmMcmcResult>
```

### Relationship to custom loss (§14)

The `prior` option in `DlmMcmcOptions` reuses the `DlmLossFn` type. For SGLD, the prior is added to the Kalman loss inside `jit(valueAndGrad(...))` — identical to MAP estimation, but with noise injection. For MH, the prior is evaluated separately in JavaScript alongside the Kalman likelihood.

This means MAP estimation (§14) and MCMC (§15) share the same prior specification mechanism. A user who starts with MAP can switch to full MCMC by changing one option (`loss` → `prior`, call `dlmMCMC` instead of `dlmMLE`).

---

## 16. Summary of all planned API exports

```ts
// ─── Core functions ───
dlmFit(y, opts)                    // → DlmFitResult (materialized)
dlmFitTensor(y, opts)              // → DlmTensorResult (on-device)
dlmForecast(fit, obsStd, h, opts?) // → DlmForecastResult
dlmGenSys(opts?)                   // → DlmSystem  { G, F, m }
findArInds(opts)                   // → number[]

// ─── Parameter estimation ───
dlmMLE(y, opts?)                   // → DlmMleResult
dlmMCMC(y, opts?)                  // → DlmMcmcResult (future)

// ─── MATLAB compat ───
toMatlab(result)                   // → DlmFitResultMatlab (names + layout)
toMatlabMle(result)                // → DlmMleResultMatlab (names)

// ─── Types ───
DlmFitOptions                     // options for dlmFit / dlmFitTensor
DlmMleOptions                     // options for dlmMLE
DlmMcmcOptions                    // options for dlmMCMC (future)
DlmForecastOptions                // options for dlmForecast
DlmFitResult                      // materialized: StateMatrix / CovMatrix, JS-idiomatic names
DlmTensorResult                   // on-device: np.Array, JS-idiomatic names
DlmForecastResult                 // forecast with StateMatrix / CovMatrix
DlmMleResult                      // MLE result (obsStd, processStd, deviance, ...)
DlmMcmcResult                     // MCMC result (future)
DlmFitResultMatlab                // MATLAB-compatible names + layout
DlmSystem                         // { G, F, m } from dlmGenSys
DlmLossFn                         // custom loss function type
StateMatrix                       // [n, m] wrapper with at/series/get
CovMatrix                         // [n, m, m] wrapper with at/series/get/variance
FloatArray                        // Float32Array | Float64Array
```

---

## 17. Naming conventions

### Design rationale

The current API inherits MATLAB DLM naming: single-letter result fields (`x`, `v`, `C`), abbreviated options (`ns`, `fullseas`, `arphi`, `fitar`), and MATLAB-specific names (`ssy`, `s2`, `XX`). This is natural for MATLAB users but opaque for JS/TS developers who discover the library via npm/TypeDoc. JS conventions favor descriptive camelCase names that are self-documenting and discoverable via autocomplete.

**Guiding principle**: A JS developer who has never seen a Kalman filter paper should be able to read result field names and understand what they contain.

**System matrix letters (G, F, W) are the exception**: these are standard notation across the state-space literature. Renaming them to verbose names would make it harder to map documentation and papers to code. They are kept as-is but with comprehensive JSDoc comments.

### Complete rename mapping

#### Result fields (`DlmFitResult`, `DlmTensorResult`)

| Current | New | Category | Rationale |
|---------|-----|----------|----------|
| `x` | `smoothed` | State estimates | "Smoothed states" — what it is |
| `xf` | `filtered` | State estimates | "Filtered states" — what it is |
| `C` | `smoothedCov` | State estimates | Explicit |
| `Cf` | `filteredCov` | State estimates | Explicit |
| `xstd` | `smoothedStd` | State estimates | Consistent with `smoothed` |
| `v` | `innovations` | Diagnostics | Single-letter → descriptive |
| `Cp` | `innovationVar` | Diagnostics | MATLAB abbreviation → descriptive |
| `resid0` | `rawResiduals` | Diagnostics | Unclear suffix → descriptive |
| `resid` | `scaledResiduals` | Diagnostics | Ambiguous → qualified |
| `resid2` | `standardizedResiduals` | Diagnostics | Unclear suffix → descriptive |
| `lik` | `deviance` | Scalars | Misleading (`lik` sounds like likelihood but is −2logL). "Deviance" is the standard statistics term. |
| `s2` | `residualVariance` | Scalars | MATLAB abbreviation → descriptive |
| `ssy` | `rss` | Scalars | MATLAB-specific → universally understood |
| `V` | `obsNoise` | Model matrices | Confusing (V suggests variance but stores std devs) |
| `x0` | `initialState` | Model matrices | Semi-clear → explicit |
| `C0` | `initialCov` | Model matrices | MATLAB abbreviation → explicit |
| `XX` | `covariates` | Model matrices | MATLAB-ism → descriptive |
| `class` | *(removed)* | Metadata | TypeScript has `instanceof`; MATLAB-ism. Restored by `toMatlab()`. |
| `G`, `F`, `W` | `G`, `F`, `W` | Model matrices | **Keep** — standard notation |
| `y` | `y` | Data | Universal |
| `yhat`, `ystd` | `yhat`, `ystd` | Diagnostics | Universal in stats/ML |
| `mse`, `mape`, `nobs` | `mse`, `mape`, `nobs` | Scalars | Universal abbreviations |
| `n`, `m` | `n`, `m` | Shape | Standard dimension descriptors |

#### Input options (`DlmFitOptions`, `DlmMleOptions`, `DlmGenSysOptions`)

| Current | New | Rationale |
|---------|-----|----------|
| `s` | `obsStd` | "Observation standard deviation" — self-documenting |
| `w` | `processStd` | "Process noise standard deviations" — self-documenting |
| `trig` | `harmonics` | Describes what the number counts |
| `ns` | `seasonLength` | Describes what the number means |
| `fullseas` | `fullSeasonal` | Unabbreviate |
| `arphi` | `arCoefficients` | Descriptive |
| `fitar` | `fitAr` | Slightly cleaner |
| `sFixed` | `obsStdFixed` | Consistent with `obsStd` |
| `order` | `order` | Already clear |
| `spline` | `spline` | Already clear |
| `X` | `X` | Standard notation for covariates (design matrix) |

#### MLE result (`DlmMleResult`)

| Current | New | Rationale |
|---------|-----|----------|
| `s` | `obsStd` | Consistent with input option |
| `w` | `processStd` | Consistent with input option |
| `arphi` | `arCoefficients` | Consistent with input option |
| `lik` | `deviance` | Consistent with fit result |
| `likHistory` | `devianceHistory` | Consistent |
| `jitMs` | `compilationMs` | Descriptive; jax-jargon → generic |
| `iterations` | `iterations` | Already descriptive |
| `elapsed` | `elapsed` | Already descriptive |
| `fit` | `fit` | Already descriptive |

#### System output (`DlmSystem`)

| Current | New | Rationale |
|---------|-----|----------|
| `G` | `G` | Standard notation |
| `F` | `F` | Standard notation |
| `m` | `m` | Standard dimension |

### Where old names live on: `toMatlab()` and `toMatlabMle()`

`toMatlab(result)` serves **two purposes simultaneously**: axis transposition (time-major → state-major) AND name restoration. This means users migrating from MATLAB DLM can write:

```ts
const m = toMatlab(fit);
m.x[0]       // level time series — exactly like MATLAB
m.Cp         // innovation covariances — exactly like MATLAB
m.lik        // -2logL — exactly like MATLAB
m.ssy        // sum of squared residuals
m.V          // observation noise std devs
m.XX         // covariates
m.class      // 'dlmfit'
```

`toMatlabMle(mleResult)` does the same for MLE:
```ts
const m = toMatlabMle(mle);
m.s          // estimated obs noise std dev
m.w          // estimated state noise std devs
m.arphi      // AR coefficients
m.lik        // -2logL
m.likHistory // optimization trace
```

The Octave test harness calls `toMatlab()` before comparing against reference JSON, so test logic is unaffected. The reference JSON files use MATLAB DLM names natively.

### Internal vs public naming

The internal `DlmSmoResult` (returned by `dlmSmo`, never exported) may keep short names for compactness since only library code reads it. The rename happens at the `DlmSmoResult` → `DlmTensorResult` / `DlmFitResult` boundary.

---

## 18. Documentation changes

### Scope of documentation updates

The naming overhaul (§17), axis changes (§1), and structural changes (§2–4) require coordinated documentation updates across the entire project. Below is the full plan.

### 1. README.md

| Section | Changes needed |
|---------|---------------|
| Quick start / Usage examples | All `dlmFit(y, s, w, dtype, opts)` calls → `dlmFit(y, { obsStd, processStd, ... })`. All result access: `fit.x[0]` → `fit.smoothed.series(0)`, `fit.lik` → `fit.deviance`, etc. |
| MLE examples | `dlmMLE(y, opts, init, ...)` positional → `dlmMLE(y, { ... })`. Result: `mle.s` → `mle.obsStd`, `mle.lik` → `mle.deviance`, `mle.jitMs` → `mle.compilationMs`. |
| Forecast examples | `dlmForecast(fit, s, h, dtype, X)` → `dlmForecast(fit, obsStd, h, opts?)`. |
| API reference tables | Full field-name update. Add `DlmFitOptions`, `DlmMleOptions`, `DlmForecastOptions` interface tables. |
| DlmGenSys section | `{ trig: 3, ns: 12, arphi: [...] }` → `{ harmonics: 3, seasonLength: 12, arCoefficients: [...] }`. |
| MATLAB comparison table | Add column mapping MATLAB names → JS names. Reference `toMatlab()` for migration. |
| Performance / benchmark sections | `lik` → `deviance` in timing tables and prose. |
| Timing markers | `lik` field references in `<!-- computed:... -->` expressions → update. |

### 2. TypeDoc / JSDoc comments (src/*.ts)

| File | Changes |
|------|---------|
| `src/types.ts` | All interfaces renamed field-by-field per §17 tables. JSDoc comments updated. `DlmFitResultMatlab` interface added. |
| `src/index.ts` | `dlmFit` and `dlmForecast` signatures + JSDoc. Internal `dlmSmo` → `DlmTensorResult` mapping code. Add `dlmFitTensor`. Add `toMatlab()`. |
| `src/mle.ts` | `DlmMleResult` interface, `dlmMLE` signature + JSDoc. Add `toMatlabMle()`. Callback parameter names. |
| `src/dlmgensys.ts` | `DlmOptions` field renames + JSDoc. `dlmGenSys` and `findArInds` parameter JSDoc. |

All JSDoc gets updated with:
- `@param` descriptions for new option names
- `@returns` descriptions with new field names
- `@example` blocks showing new API
- Cross-references: "In MATLAB DLM, this is called `x`. See `toMatlab()`." on key fields

### 3. Test files

| File | Changes |
|------|---------|
| `tests/utils.ts` | Add `toMatlab()` import. Any test helpers that reference old field names. |
| `tests/niledemo.test.ts` | Wrap result with `toMatlab()` before comparing against MATLAB reference JSON. |
| `tests/gensys.test.ts` | Same `toMatlab()` wrapping. Update option names in `dlmFit` calls. |
| `tests/synthetic.test.ts` | Update `fit.x[i]` → `fit.smoothed.series(i)` access patterns (or use `toMatlab()`). |
| `tests/mle.test.ts` | Update `result.s`, `result.w`, `result.lik` → new names. |
| `tests/covariate.test.ts` | `fit.x[m_base]` → `fit.smoothed.series(m_base)`. `fit.XX` → `fit.covariates`. |
| `tests/forecast.test.ts` | Update forecast result access patterns. |
| `tests/missing.test.ts` | Same `toMatlab()` wrapping for Octave comparison. |
| `tests/assocscan.test.ts` | Same `toMatlab()` wrapping for Octave comparison. |
| `tests/ozone.test.ts` | Update result access patterns. |
| `tests/test-matrix.ts` | No changes expected (device/dtype configs, not field names). |

**Strategy for Octave-comparison tests**: The MATLAB reference JSON files (e.g. `niledemo-out-m.json`) use MATLAB DLM field names. Rather than renaming all reference files, the test harness calls `toMatlab(result)` before the comparison, keeping the reference files and numeric comparison logic untouched. This is the minimal-risk approach.

### 4. Script files

| File | Changes |
|------|---------|
| All `scripts/gen-*.ts` | `fit.x[0]` → `fit.smoothed.series(0)`. Option names in `dlmFit`/`dlmMLE` calls. |
| All `scripts/collect-*.ts` | `mle.lik` → `mle.deviance`. `fit.x` access patterns. |
| `scripts/collect-mle-benchmark.ts` | `nileOrder1.lik` → `nileOrder1.deviance`. |
| `scripts/bench-*.ts` | Option names in `dlmFit` calls. |

### 5. copilot-instructions.md

Update field name references throughout:
- `DlmFitResult` field descriptions
- `DlmMleResult` field descriptions  
- `DlmOptions` descriptions
- Example prompts for agents
- Test suite descriptions
- Troubleshooting checklist

### 6. api-overhaul-plan.md

This document (already updated in this commit).

### 7. Migration guide (new document)

Create `MIGRATION.md` as part of the release:
- Side-by-side old → new name table
- `toMatlab()` usage for MATLAB DLM users
- Search-and-replace patterns for common migrations
- Before/after code examples for `dlmFit`, `dlmMLE`, `dlmForecast`

### Documentation execution order

1. **Types first**: Update `src/types.ts` interfaces with new names + JSDoc.
2. **Implementation**: Update `src/index.ts`, `src/mle.ts`, `src/dlmgensys.ts`.
3. **MATLAB compat**: Implement `toMatlab()` and `toMatlabMle()`.
4. **Tests**: Update test harness to use `toMatlab()` for Octave comparisons; update direct field access in other tests.
5. **Scripts**: Update all SVG generators and benchmark collectors.
6. **README**: Full example rewrite.
7. **copilot-instructions.md**: Update field references.
8. **TypeDoc generation**: `pnpm run docs` to regenerate API docs.
9. **MIGRATION.md**: Write as final step before release.
