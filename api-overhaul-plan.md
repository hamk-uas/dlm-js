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
  // States
  x: np.Array;     // [n, m] — smoothed states (squeezed from internal [n, m, 1])
  xf: np.Array;    // [n, m] — filtered states
  C: np.Array;     // [n, m, m] — smoothed covariances
  Cf: np.Array;    // [n, m, m] — filtered covariances
  xstd: np.Array;  // [n, m] — state std devs (sqrt of C diagonal)

  // Observation diagnostics — [n]
  yhat: np.Array;
  ystd: np.Array;
  v: np.Array;
  Cp: np.Array;
  resid0: np.Array;
  resid: np.Array;
  resid2: np.Array;

  // Scalars
  lik: np.Array;
  s2: np.Array;
  mse: np.Array;
  mape: np.Array;
  ssy: np.Array;
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
  using result = await dlmFitTensor(y, { s: 120, w: [40, 10], order: 1 });
  // result.x is np.Array [n, m] — stays on device
  const trend = np.slice(result.x, [0, 0], [n, 1]);  // level component [n, 1]
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
  // State estimates
  x: StateMatrix;       // [n, m] smoothed
  xf: StateMatrix;      // [n, m] filtered
  C: CovMatrix;         // [n, m, m] smoothed
  Cf: CovMatrix;        // [n, m, m] filtered
  xstd: StateMatrix;    // [n, m] smoothed state std devs

  // 1-D time series (plain FloatArray, length n)
  yhat: FloatArray;
  ystd: FloatArray;
  v: FloatArray;
  Cp: FloatArray;
  resid0: FloatArray;
  resid: FloatArray;
  resid2: FloatArray;

  // Scalars
  lik: number;
  s2: number;
  mse: number;
  mape: number;
  ssy: number;
  nobs: number;

  // Model matrices (plain JS arrays for easy serialization)
  G: number[][];
  F: number[];
  W: number[][];
  x0: number[];
  C0: number[][];
  y: FloatArray;
  V: FloatArray;
  XX: number[][];

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
const fit = await dlmFit(y, { s: 120, w: [40, 10], order: 1 });

// Level trend for plotting (common case):
const level = fit.x.series(0);        // FloatArray [n]
plotLine(level);

// All states at a specific timestep (zero-copy view):
const stateVec = fit.x.at(42);       // FloatArray view [m]

// State uncertainty:
const levelStd = fit.xstd.series(0); // FloatArray [n]

// Raw buffer for bulk GPU upload:
fit.x.data                            // FloatArray of length n*m, row-major [n, m]

// Observation predictions (unchanged — already FloatArray [n]):
plotBand(fit.yhat, fit.ystd);
```

**Comparison to current API:**
```ts
// CURRENT — state-major, raw arrays, inconsistent xstd
fit.x[0]        // FloatArray [n] — zero-copy (convenient)
fit.x[0][42]    // state 0 at time 42
fit.xstd[42]    // FloatArray [m] — INCONSISTENT axis order vs x!

// NEW — time-major, StateMatrix, consistent
fit.x.series(0) // FloatArray [n] — copy (one allocation, negligible vs filter runtime)
fit.x.get(42, 0)    // state 0 at time 42
fit.x.at(42)        // FloatArray [m] — zero-copy view
fit.xstd.at(42)     // FloatArray [m] — SAME convention as x ✓
fit.xstd.series(0)  // FloatArray [n] — level std dev time series
```

`DlmForecastResult` uses the same `StateMatrix` and `CovMatrix` classes with `h` replacing `n`.

---

## 4. Function signatures — options objects

**`dlmFit`:**
```ts
interface DlmFitOptions {
  // Noise (required)
  s: number | ArrayLike<number>;   // observation noise std dev (scalar or per-obs array)
  w: number[];                      // state noise std devs (diagonal of sqrt(W))

  // Model (optional, defaults to local linear trend)
  order?: number;          // polynomial trend order: 0, 1, 2 — default: 1
  trig?: number;           // trigonometric harmonic pairs
  ns?: number;             // seasons per cycle — default: 12
  fullseas?: boolean;      // full seasonal component
  arphi?: number[];        // AR coefficients
  spline?: boolean;        // spline mode for order=1

  // Covariates
  X?: ArrayLike<number>[];  // n rows × q cols

  // Arbitrary time steps (see §13)
  dt?: ArrayLike<number>;   // [n] inter-observation intervals; enables time-varying G_t, W_t

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
  // Model
  order?: number;
  trig?: number;
  ns?: number;
  fullseas?: boolean;
  arphi?: number[];
  fitar?: boolean;          // fit AR coefficients via MLE

  // Covariates
  X?: ArrayLike<number>[];

  // Arbitrary time steps (see §13)
  dt?: ArrayLike<number>;

  // Loss function (see §14)
  loss?: 'ml' | DlmLossFn;   // default: 'ml' (standard Kalman prediction-error likelihood)

  // Optimizer
  maxIter?: number;         // default: 200
  lr?: number;              // Adam learning rate — default: 0.05
  tol?: number;             // convergence tolerance — default: 1e-6
  init?: { s?: number; w?: number[]; arphi?: number[] };
  adamOpts?: { b1?: number; b2?: number; eps?: number };  // default b2=0.9
  sFixed?: ArrayLike<number>;   // per-obs sigma (fixes V; s not estimated)
  callbacks?: {
    onInit?: (theta: FloatArray) => void;
    onIteration?: (iter: number, theta: FloatArray, lik: number) => void;
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
  s: number,
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

An exported function converts the new time-major layout to MATLAB-like layout on demand:

```ts
interface DlmFitResultMatlab {
  x: FloatArray[];       // x[stateIdx][timeIdx] — state-major, like MATLAB m×n
  xf: FloatArray[];
  C: FloatArray[][];     // C[i][j][timeIdx]
  Cf: FloatArray[][];
  xstd: FloatArray[];    // xstd[timeIdx][stateIdx] — MATLAB's n×m layout
  // All 1-D fields pass through unchanged
}

function toMatlabLayout(result: DlmFitResult): DlmFitResultMatlab;
```

This creates the transpose copies that current `dlmFit` does eagerly — but only on explicit request. MATLAB DLM migration users call this once; everyone else never pays the cost.

The Octave-comparison test harness calls `toMatlabLayout()` internally before comparing against reference JSON, keeping numerical test logic unchanged.

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

## 8. `dlmGenSys` — no change

`dlmGenSys` returns `{ G: number[][], F: number[], m: number }`. Already idiomatic; no axis-ordering or ergonomics issues. Unchanged.

`findArInds` unchanged.

---

## 9. Breaking changes summary

| Change | What breaks | Migration path |
|--------|-------------|----------------|
| `x`, `xf` from `FloatArray[]` to `StateMatrix` | `fit.x[0]` | `fit.x.series(0)` or `toMatlabLayout(fit).x[0]` |
| `C`, `Cf` from `FloatArray[][]` to `CovMatrix` | `fit.C[i][j]` | `fit.C.series(i, j)` or `toMatlabLayout(fit).C[i][j]` |
| `xstd` from `FloatArray[]` to `StateMatrix` (axis flip) | `fit.xstd[t]` (was [time][state]) | `fit.xstd.at(t)` (same semantics) |
| All noise/model/runtime args in options bag | Positional `(y, s, w, dtype, opts, X)` | `(y, { s, w, dtype, ...opts, X })` |
| `dtype: DType` → `dtype: 'f32' \| 'f64'` | `DType.Float64` import from jax-js | `'f64'` string |
| `forceAssocScan` → `algorithm: 'assoc'` | Internal boolean flag | `{ algorithm: 'assoc' }` |
| New function `dlmFitTensor` | n/a — new addition | — |

---

## 10. What stays the same

- `dlmGenSys` and `findArInds` — unchanged
- All 1-D diagnostic outputs (`yhat`, `ystd`, `v`, `Cp`, `resid0`, `resid`, `resid2`) — already `FloatArray` of length `n`, no axis issues, no change required
- All scalar diagnostics (`lik`, `s2`, `mse`, `mape`, `ssy`, `nobs`)
- Internal numerics, all three execution branches, tolerances — no algorithmic changes
- `DlmMleResult` fields `s`, `w`, `arphi`, `lik`, `iterations`, `elapsed`, `jitMs`, `likHistory` — unchanged; `fit` field becomes the new `DlmFitResult` shape

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
const fit = await dlmFit(y, {
  s: 120,
  w: [40, 10],     // now interpreted as continuous-time spectral densities (q_c)
  order: 1,
  dt: [1, 1, 1, 3, 1, 1, 7, 1, ...],  // days between observations
});
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

When `dt` is provided, the noise parameters `w` in `DlmMleOptions` are reinterpreted as **continuous-time spectral densities** $q_c$ rather than discrete-time standard deviations. The per-timestep $W_k = f(q_c, \Delta t_k)$ is recomputed from $q_c$ and $\Delta t_k$ inside the loss function. Since the closed-form $W_k(\Delta t_k, q_c)$ is differentiable w.r.t. $q_c$, autodiff propagates through the matrix construction naturally.

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
  loss: (lik, theta) => {
    // theta = [log(s), log(w0), log(w1), ...]
    // Add N(0, σ²) prior on log(w) entries (= log-normal prior on w)
    const logW = np.slice(theta, [1], [3]);   // w entries
    const prior = np.sum(np.square(logW));     // -2 log p(w)
    return np.add(lik, np.multiply(np.array(0.1), prior));
  },
});
```

### Architecture

The loss function is injected into the `jit(valueAndGrad(...))` wrapper:

```
Current:  jit(valueAndGrad(makeKalmanLoss(...)))(theta)
New:      jit(valueAndGrad(theta => userLoss(makeKalmanLoss(theta), theta)))(theta)
```

When `loss: 'ml'` (default), the wrapper is identity: `(lik, _theta) => lik`. The entire chain — Kalman scan + custom penalty + AD backward pass + Adam update — remains inside a single `jit()` call. No performance change for the default path.

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

A more classical approach: propose $\theta' \sim q(\theta' | \theta)$, evaluate $\log p(\theta' | y)$ via `dlmFitTensor` (which returns `lik` as an on-device scalar), accept/reject. The on-device tensor API avoids materializing the full fit result on each MCMC iteration — only `lik` is read back to JS.

```ts
// Sketch of MH loop using existing tensor API:
for (let k = 0; k < nSamples; k++) {
  const thetaPrime = propose(theta);
  using fit = await dlmFitTensor(y, buildOpts(thetaPrime));
  const likPrime = (await fit.lik.data())[0];
  const logPrior = evaluatePrior(thetaPrime);
  if (Math.log(Math.random()) < logPosterior(likPrime, logPrior) - logPosteriorCurrent) {
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
    s: Float64Array;     // [nSamples]
    w: Float64Array[];   // [m][nSamples]
    arphi?: Float64Array[]; // [p][nSamples]
  };
  acceptance: number;    // acceptance rate (MH only)
  fit: DlmFitResult;     // fit at posterior mean
  likHistory: Float64Array;
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
dlmForecast(fit, s, h, opts?)      // → DlmForecastResult
dlmGenSys(opts?)                   // → DlmSystem  { G, F, m }
findArInds(opts)                   // → number[]

// ─── Parameter estimation ───
dlmMLE(y, opts?)                   // → DlmMleResult
dlmMCMC(y, opts?)                  // → DlmMcmcResult (future)

// ─── Utilities ───
toMatlabLayout(result)             // → DlmFitResultMatlab (compat helper)

// ─── Types ───
DlmFitOptions                     // options bag for dlmFit / dlmFitTensor
DlmMleOptions                     // options bag for dlmMLE
DlmMcmcOptions                    // options bag for dlmMCMC (future)
DlmForecastOptions                // options bag for dlmForecast
DlmFitResult                      // materialized result with StateMatrix / CovMatrix
DlmTensorResult                   // on-device result with np.Array
DlmForecastResult                 // forecast result with StateMatrix / CovMatrix
DlmMleResult                      // MLE result
DlmMcmcResult                     // MCMC result (future)
DlmSystem                         // { G, F, m } from dlmGenSys
DlmLossFn                         // custom loss function type
StateMatrix                       // [n, m] wrapper with at/series/get
CovMatrix                         // [n, m, m] wrapper with at/series/get/variance
FloatArray                        // Float32Array | Float64Array
```
