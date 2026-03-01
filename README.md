# dlm-js — TypeScript Kalman filter/smoother with autodiff MLE

<strong>
  <a href="https://hamk-uas.github.io/dlm-js/">API Reference</a> |
  <a href="https://github.com/hamk-uas/dlm-js">GitHub</a> |
  <a href="https://mjlaine.github.io/dlm/">Original MATLAB DLM Docs</a> |
  <a href="https://github.com/mjlaine/dlm">Original MATLAB DLM GitHub</a>
</strong>

A TypeScript Kalman filter + RTS smoother library using [jax-js-nonconsuming](https://github.com/hamk-uas/jax-js-nonconsuming), inspired by [dynamic linear model](https://mjlaine.github.io/dlm/dlmtut.html) (MATLAB). Extends the original with autodiff-based MLE via `jit(valueAndGrad + Adam)` and an exact O(log N) parallel filter+smoother via `lax.associativeScan` (Särkkä & García-Fernández 2020).

🤖 AI generated code & documentation with gentle human supervision.

### Features at a glance
- **Kalman filter & RTS smoother**: Sequential (`scan`), exact O(log N) parallel (`assoc`), and square-root parallel (`sqrt-assoc`) algorithms.
- **Autodiff MLE & MAP**: Jointly estimate observation noise, process noise, and AR coefficients via `jit(valueAndGrad + Adam)`. MAP estimation with `dlmPrior` (Inverse-Gamma / Normal priors) or custom loss callbacks.
- **Multiple backends**: Runs on CPU, WASM (recommended for speed), and WebGPU.
- **Multivariate observations**: Vector-valued observations (p > 1) with full matrix algebra (inv, matmul, einsum, slogdet).
- **Missing data & irregular timestamps**: Built-in support for NaN observations and arbitrary time steps.
- **Forecasting**: Propagate states $h$ steps ahead with calibrated uncertainty bounds.
- **Cross-platform**: Works in Node.js and the browser (ESM & CommonJS).

> ⚠️ **Warning:** The API is not yet frozen and may change before the 1.0 release.

## Installation

dlm-js is not yet published to npm. Install directly from GitHub:

```shell
# npm
npm install github:hamk-uas/dlm-js

# pnpm
pnpm add github:hamk-uas/dlm-js
```

This also installs the `@hamk-uas/jax-js-nonconsuming` dependency automatically.

## Usage

dlm-js works in **both Node.js and the browser** — the library has no platform-specific code. It ships ESM, CommonJS, and TypeScript declarations.

Naming convention: exported JS/TS APIs use camelCase (for example `dlmFit`, `dlmGenSys`), while original MATLAB functions are lowercase (for example `dlmfit`, `dlmsmo`, `dlmgensys`).

### ESM (Node.js / browser bundler)

```js
import { dlmFit, dlmGenSys } from "dlm-js";

// Nile river annual flow data (excerpt)
const y = [1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140];

// Fit a local linear trend model (order=1, state dim m=2)
const result = await dlmFit(y, { obsStd: 120, processStd: [40, 10], order: 1, dtype: 'f64' });

console.log(result.yhat);       // smoothed predictions [n]
console.log(result.smoothed);   // smoothed states (StateMatrix: .series(i), .get(t,i))
console.log(result.deviance);   // -2·log-likelihood
// Also available: result.smoothedStd (StateMatrix), result.ystd [n], result.innovations [n],
//   result.standardizedResiduals [n], result.mse, result.mape, result.rss, result.residualVariance, result.nobs
```

For an order=1 model with `spline: true`, the W covariance is scaled to produce an integrated random walk (matches MATLAB `dlmfit` spline mode).

### CommonJS (Node.js)

```js
const { dlmFit } = require("dlm-js");
```

### Generate system matrices only

```js
import { dlmGenSys } from "dlm-js";

const sys = dlmGenSys({ order: 1, harmonics: 2, seasonLength: 12 });
console.log(sys.G);  // state transition matrix (m×m)
console.log(sys.F);  // observation vector (1×m)
console.log(sys.m);  // state dimension
```

### Shared model specification (`DlmModelSpec`)

The `DlmModelSpec` type captures the structural model description (trend order, seasonality, AR) and is accepted by `dlmGenSys`, `dlmFit`, and `dlmMLE`. Define the model once and thread it through the pipeline:

```ts
import { dlmFit, dlmMLE, dlmGenSys, type DlmModelSpec } from "dlm-js";

const model: DlmModelSpec = { order: 1, harmonics: 2, seasonLength: 12 };

const sys = dlmGenSys(model);
const fit = await dlmFit(y, { ...model, obsStd: 120, processStd: [40, 10], dtype: 'f64' });
const mle = await dlmMLE(y, { ...model, dtype: 'f64' });
```

### h-step-ahead forecasting

Propagate the last smoothed state h steps forward with no new observations:

```js
import { dlmFit, dlmForecast } from "dlm-js";

const y = [1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140];

// Fit a local linear trend model
const fit = await dlmFit(y, { obsStd: 120, processStd: [40, 10], order: 1, dtype: 'f64' });

// Forecast 12 steps ahead
const fc = await dlmForecast(fit, 12, { dtype: 'f64' });

console.log(fc.yhat);       // predicted observation means [h] = F·x_pred
console.log(fc.ystd);       // observation prediction std devs [h] — grows monotonically
console.log(fc.predicted);  // state trajectories (StateMatrix)
console.log(fc.h);          // 12
console.log(fc.m);          // 2 (state dimension)
```

`fc.yhat` is the full observation prediction `F·x_pred`. For pure trend models (no seasonality) this equals the level state and is appropriate to plot directly. For seasonal or AR models, `yhat` oscillates with the harmonics/AR dynamics in the forecast horizon — if you want a smooth trendline, use the level state `fc.predicted.series(0)` directly:

```js
// For seasonal/AR models: plot level state, not yhat
const trend = fc.predicted.series(0);           // smooth trend mean
const trendStd = fc.predictedStd.series(0);     // level state std dev
```

With covariates, pass `X_forecast` rows for each forecast step:

```js
// Forecast 3 steps ahead with known future covariate values
const fc = await dlmForecast(fit, 3, { dtype: 'f64', X: [
  [solarProxy[n], qbo1[n], qbo2[n]],    // step n+1
  [solarProxy[n+1], qbo1[n+1], qbo2[n+1]], // step n+2
  [solarProxy[n+2], qbo1[n+2], qbo2[n+2]], // step n+3
] });
```

Current behavior for unknown future covariates: if `X_forecast` is omitted (or does not provide a row/entry), dlm-js uses `0` for the missing covariate value in that step. Interpret this as a **baseline conditional forecast** (unknown driver effects set to zero), not a full unconditional forecast.

For a more neutral assumption in practice, center covariates before fitting so that `0` represents a typical/historical-average driver level. Then the default forecast corresponds to “no expected driver anomaly.”

For decision use, prefer scenario forecasting: provide multiple plausible `X_forecast` paths (e.g. low/base/high) and compare resulting forecast bands.

### Missing data (NaN observations)

Place `NaN` in the observation vector `y` wherever a measurement is absent. `dlmFit` automatically skips those timesteps in the Kalman gain and residual calculations (K and v are zeroed), so the smoother interpolates through the gaps without any extra configuration:

```js
import { dlmFit } from "dlm-js";

// Nile data with a gap in years 30–39 and every 7th observation missing
const y = [1120, 1160, 963, NaN, 1210, 1160, 1160, NaN, 813, /* ... */];

const result = await dlmFit(y, { obsStd: 120, processStd: [40, 10], order: 1, dtype: 'f64' });

// nobs: number of non-NaN observations actually used
console.log(result.nobs);   // e.g. 77 when 23 of 100 values are NaN

// yhat, smoothed, smoothedStd, ystd: fully interpolated — finite at every timestep
console.log(result.yhat);                    // smoothed observation mean [n] — no NaN
console.log(result.smoothed);                // smoothed state trajectories (StateMatrix) — no NaN
console.log(result.smoothedStd);             // smoothed state std devs (StateMatrix) — no NaN
console.log(result.ystd);                    // smoothed observation std devs [n] — no NaN

// innovations and standardizedResiduals: NaN at missing positions (consistent with MATLAB dlmsmo)
console.log(result.innovations);             // innovations [n] — NaN at missing timesteps
console.log(result.standardizedResiduals);   // squared normalised residuals [n] — NaN at missing timesteps

// deviance is the log-likelihood summed only over observed timesteps
console.log(result.deviance);
```

Missing observations are handled identically to MATLAB's `dlmsmo` (`ig = not(isnan(y(i,:)))` logic): the filter propagates through the gap using only the prior, and the RTS smoother then distributes the information from surrounding observations backward and forward. `ystd` grows wider over the gap, reflecting higher uncertainty where no data was seen.

### Irregular timestamps

When observations are not uniformly spaced, pass a `timestamps` array to `dlmFit`. The transition matrix $G(\Delta t)$ and process noise covariance $W(\Delta t)$ are then computed per-step in closed form: $G$ via matrix-exponential discretization of the continuous-time state matrix; $W$ via analytic continuation of the discrete-time noise accumulation sum evaluated at non-integer $\Delta t$ (Faulhaber sums). No numerical ODE solver or matrix exponential of $W$ needed.

```js
import { dlmFit } from "dlm-js";

// Observations at irregular times (e.g. years with gaps)
const y   = [1120, 1160, 963, 1210, 1160,   969, 831, 456, 824, 702];
const ts  = [1871, 1872, 1873, 1874, 1875,  1910, 1911, 1913, 1914, 1915];
//                                        ↑ 35-year gap

const result = await dlmFit(y, {
  obsStd: 120, processStd: [40, 10], order: 1,
  timestamps: ts, dtype: 'f64',
});

console.log(result.yhat);      // smoothed predictions at each timestamp
console.log(result.ystd);      // observation std devs — wider after gaps
```

**Supported components:** polynomial trend (order 0, 1, 2), trigonometric harmonics, and AR components (integer-spaced timestamps only). `fullSeasonal` throws because it is a purely discrete-time construct with no natural continuous-time extension. AR components are supported when every $\Delta t$ is a positive integer: the companion matrix is raised to the $d$-th power via binary exponentiation ($G_{AR}^d$) and the noise is accumulated as $W_{AR}(d) = \sum_{k=0}^{d-1} C^k \, W_1 \, (C^k)^\top$. Non-integer $\Delta t$ with AR still throws.

**Tip — non-integer base step with AR:** If your data's natural step is non-integer (e.g. months as fractions of a year: $\Delta t = 1/12$), scale `timestamps` so the base step becomes 1. This makes all gaps integer multiples and satisfies the AR requirement:

```js
const baseStep = 1 / 12;  // monthly data in year units
const ts_raw = [0, 1/12, 2/12, 6/12, 7/12];  // 3-month gap after March

const result = await dlmFit(y, {
  timestamps: ts_raw.map(t => t / baseStep),  // [0, 1, 2, 6, 7]
  arCoefficients: [0.8], processStd: [10, 5], obsStd: 20,
  dtype: 'f64',
});
```

After rescaling, the model's time unit is one base step (one month), so `processStd` and slope states are per-month quantities. Adjust `seasonLength` accordingly (e.g. `seasonLength: 12` for an annual cycle in monthly data).

#### How it works

The standard DLM propagates through each unit timestep as $x_{t+1} = G \cdot x_t + w_t$ with $w_t \sim \mathcal{N}(0, W)$.  For a gap of $\Delta t$ steps between consecutive observations, `dlmGenSysTV` computes:

**Transition matrix** $G(\Delta t) = \text{expm}(F_c \cdot \Delta t)$ — for a polynomial trend this is just the Jordan block with entries $\Delta t^j / j!$ on the $j$-th superdiagonal:

$$G(\Delta t) = \begin{bmatrix} 1 & \Delta t & \Delta t^2/2 \\ 0 & 1 & \Delta t \\ 0 & 0 & 1 \end{bmatrix} \quad \text{(order 2)}$$

**Process noise** $W(\Delta t)$ is the accumulated discrete noise from $\Delta t$ unit steps:

$$W(\Delta t) = \sum_{k=0}^{\Delta t - 1} G^k \; W_1 \; (G^k)^\top$$

This is a discrete-time sum, not a continuous-time noise integral. Because $G$ is a Jordan block, $G^k$ has polynomial entries in $k$ and the sum reduces to closed-form Faulhaber sums. Evaluating those polynomials at non-integer $\Delta t$ is an **analytic continuation** of the discrete-time formula.  For order 1 with process noise std devs $w_0, w_1$:

$$W(\Delta t) = \begin{bmatrix} \Delta t \, w_0^2 + w_1^2 \, S_2 & w_1^2 \, S_1 \\ w_1^2 \, S_1 & \Delta t \, w_1^2 \end{bmatrix}$$

where $S_1 = \Delta t(\Delta t - 1)/2$ and $S_2 = \Delta t(\Delta t - 1)(2\Delta t - 1)/6$.  These formulas interpolate smoothly for non-integer $\Delta t$ and collapse to $W(1) = \text{diag}(w_0^2, w_1^2)$ at unit spacing.  The off-diagonal terms capture the physical coupling: slope noise accumulates into level variance over multi-step gaps.

**Trigonometric harmonics** scale the rotation angle: $\theta_k = \Delta t \cdot 2\pi h / n_s$, and process noise scales linearly: $W_{hh} = \Delta t \cdot w_h^2$.

**Indexing convention.** `G_scan[k]` encodes the *departing* transition from obs $k$ to obs $k+1$ ($\Delta t = T[k+1] - T[k]$).  The forward Kalman step applies $G_k$ *after* incorporating observation $k$ to predict the state at $k+1$.  In the associative scan path, element construction at position $k$ needs the *arriving* transition ($G_\text{scan}[k-1]$), which is built by shifting the array.

**Tip — interpolation at query points:** To obtain smoothed estimates at times where no measurement exists, insert `NaN` observations at those timestamps. The smoother treats NaN as a pure prediction step, giving interpolated state estimates and uncertainty bands at arbitrary query points.

**Related work.** [4] extends the parallel scan framework of [1] to continuous-time models, deriving exact associative elements for MAP trajectory estimation on irregularly-sampled data. dlm-js uses a simpler approach — closed-form $G$ discretization and analytic continuation of $W$ fed to the existing filter — but the parallel scan composition is directly applicable if continuous-time MAP estimation is needed in the future. The EEA-sensors group provides two JAX reference implementations of the same parallel scan framework: [5] implements parallel EKF/CKF/IEKS for non-linear models, and [6] reformulates the same algorithms in square-root (Cholesky factor) form for improved numerical stability — see [square-root parallel smoother](#square-root-parallel-smoother-sqrt-assoc).

### MLE parameter estimation

Estimate observation noise `obsStd`, process noise `processStd`, and optionally AR coefficients by maximizing the Kalman filter log-likelihood via autodiff:

```js
import { dlmMLE } from "dlm-js";
import { defaultDevice } from "@hamk-uas/jax-js-nonconsuming";

defaultDevice("wasm"); // recommended: ~10–20× faster than "cpu"

const y = [1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140 /* ... */];

// Basic: estimate obsStd and processStd
const mle = await dlmMLE(
  y,
  { order: 1, maxIter: 300, lr: 0.05, tol: 1e-6, dtype: 'f64' },
);

console.log(mle.obsStd);      // estimated observation noise std dev
console.log(mle.processStd);  // estimated process noise std devs
console.log(mle.deviance);    // -2·log-likelihood at optimum
console.log(mle.iterations);  // iterations to convergence
console.log(mle.elapsed);     // wall-clock ms
console.log(mle.fit);         // full DlmFitResult with optimized parameters

// With AR fitting: estimate obsStd, processStd, and AR coefficients jointly
const mleAR = await dlmMLE(
  y,
  { order: 0, arCoefficients: [0.5], fitAr: true, maxIter: 300, lr: 0.02, tol: 1e-6, dtype: 'f64' },
);
console.log(mleAR.arCoefficients);  // estimated AR coefficients (e.g. [0.81])
```

`dlmMLE` uses Adam with automatic differentiation (autodiff) to optimize the Kalman filter log-likelihood. The entire optimization step — forward filter + AD backward pass + Adam update — is compiled via `jit()` for speed. See [How MLE works](#how-mle-works) for technical details and [Parameter estimation: MATLAB DLM vs dlm-js](#parameter-estimation-mle--map-matlab-dlm-vs-dlm-js) for a detailed comparison.

`dlmMLE` also supports missing data — the Kalman loss scan zeros K, v, and the log-likelihood contribution at NaN timesteps, so autodiff and Adam optimization work correctly through the gaps:

```js
const mle = await dlmMLE(y, { order: 1, maxIter: 200, lr: 0.05 });
// mle.deviance is the log-likelihood summed only over observed timesteps
// mle.fit.nobs reports the count of non-NaN observations used
```

### MAP estimation (custom loss / priors)

By default `dlmMLE` minimizes the Kalman −2·log-likelihood (pure MLE). The `loss` option lets you supply a custom objective that wraps the Kalman loss, enabling MAP estimation, regularization, or any user-defined penalty. The entire chain — Kalman scan + custom penalty + AD backward pass + optimizer update — is compiled in a single `jit()` call with no extra dispatch overhead.

Inspired by [dynamax](https://probml.github.io/dynamax/) (`log_prior + marginal_loglik`), [GPJax](https://docs.jaxgaussianprocesses.com/) (composable objectives), and [BayesNewton](https://github.com/AaltoML/BayesNewton) (energy-based optimization). Unlike these Python libraries which use class hierarchies, dlm-js uses a single callback — simpler and equally expressive when the prior is a jax-js function.

#### `dlmPrior` — Inverse-Gamma / Normal priors (MATLAB DLM style)

The easiest way to add priors. `dlmPrior` creates the loss callback for you, matching MATLAB/R DLM `dlmGibbsDIG` conventions: Inverse-Gamma on variances, Normal on AR coefficients.

```js
import { dlmMLE, dlmPrior } from "dlm-js";

// Weakly informative IG(2, 1) on both obs and process variances
const result = await dlmMLE(y, {
  order: 1,
  loss: dlmPrior({
    obsVar:     { shape: 2, rate: 1 },   // IG(α=2, β=1) on s²
    processVar: { shape: 2, rate: 1 },   // IG(α=2, β=1) on each wᵢ²
  }),
});

console.log(result.deviance);      // pure −2·logL at the MAP optimum
console.log(result.priorPenalty);   // MAP_objective − pure_deviance (≥ 0)
```

Available prior specs:
- `obsVar: { shape, rate }` — Inverse-Gamma on observation variance s².
- `processVar: { shape, rate }` — IG on process variance(s) wᵢ² (single spec recycled for all components, or array for per-component priors).
- `arCoef: { mean, std }` — Normal on AR coefficients (requires `fitAr: true`).

#### Custom loss callback

For full control, pass a function directly. The `loss` callback receives three arguments: `deviance` (scalar Kalman −2·logL), `params` (natural-scale 1-D parameter vector), and `meta` (layout descriptor). Use only jax-js ops inside — it runs inside the traced AD graph.

```js
import { dlmMLE } from "dlm-js";
import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

// Manual L2 prior on natural-scale params
const result = await dlmMLE(y, {
  order: 1,
  loss: (deviance, params, meta) => {
    // params = [s, w0, w1] for order=1 (natural scale, not log-transformed)
    // L2 penalty pulling std devs toward 1:
    const target = np.array([1, 1, 1]);
    const diff = np.subtract(params, target);
    const penalty = np.multiply(np.array(0.1), np.sum(np.square(diff)));
    return np.add(deviance, penalty);
  },
});
```

`params` layout: `[s?, w₀, …, w_{m-1}, φ₁, …, φ_p]` — `s` and `wᵢ` are positive std devs; `φⱼ` are unconstrained AR coefficients. The `meta` object (`{ nObs, nProcess, nAr }`) describes how many of each type are present.

## Fit

### Demos

All demos can be regenerated locally with `pnpm run gen:svg`. The `assoc` and `webgpu` variants use an exact O(log N) parallel filter+smoother (Särkkä & García-Fernández 2020) and match the sequential `scan` results to within numerical tolerance (validated by `assocscan.test.ts`). The `sqrt-assoc` variant implements the square-root form in Cholesky-factor space (Yaghoobi et al. 2022) — same O(log N) depth, validated by `sqrtassoc.test.ts` (24 tests on wasm — f64: precision comparison for all models including fullSeasonal m=13, f32: all-outputs-finite smoke test for all models).

#### Nile River Flow (Local Linear Trend)

<p align="center">
  <img alt="Nile demo (sequential scan)" src="assets/niledemo-scan.svg" width="100%" />
  <br/><br/>
  <img alt="Nile demo (associative scan)" src="assets/niledemo-assoc.svg" width="100%" />
  <br/><br/>
  <img alt="Nile demo (sqrt-assoc, f64)" src="assets/niledemo-sqrt-assoc.svg" width="100%" />
  <br/><br/>
  <img alt="Nile demo (sqrt-assoc, f32)" src="assets/niledemo-sqrt-assoc-f32.svg" width="100%" />
</p>

*First smoothed state (level) `smoothed[0]` from dlm-js (solid blue) vs MATLAB/Octave dlm (dashed red), with ± 2σ bands from `smoothedStd[:,0]` (state uncertainty, not observation prediction intervals).*

#### Kaisaniemi Monthly Temperatures (Seasonal)

<p align="center">
  <img alt="Kaisaniemi demo (sequential scan)" src="assets/kaisaniemi-scan.svg" width="100%" />
  <br/><br/>
  <img alt="Kaisaniemi demo (associative scan)" src="assets/kaisaniemi-assoc.svg" width="100%" />
</p>

*Top panel: level state `smoothed[0] ± 2σ`. Bottom panel: covariance-aware combined signal `smoothed[0]+smoothed[2] ± 2σ`, using `Var(x0+x2)=Var(x0)+Var(x2)+2Cov(x0,x2)`. Model settings: `order=1`, `harmonics=1`, `obsStd=2`, `processStd=[0,0.005,0.4,0.4]`.*

#### Energy Demand (Seasonal + AR)

<p align="center">
  <img alt="Energy demand demo (sequential scan)" src="assets/trigar-scan.svg" width="100%" />
  <br/><br/>
  <img alt="Energy demand demo (associative scan)" src="assets/trigar-assoc.svg" width="100%" />
</p>

*Synthetic 10-year monthly data. Panels top to bottom: smoothed level `smoothed[0] ± 2σ`, trigonometric seasonal `smoothed[2] ± 2σ`, AR(1) state `smoothed[4] ± 2σ`, and covariance-aware combined signal `F·x = smoothed[0]+smoothed[2]+smoothed[4] ± 2σ`. True hidden states (green dashed) are overlaid. Model settings: `order=1`, `harmonics=1`, `seasonLength=12`, `arCoefficients=[0.85]`, `obsStd=1.5`, `processStd=[0.3,0.02,0.02,0.02,2.5]`, m=5.*

#### Stratospheric Ozone Trend Analysis

<p align="center">
  <img alt="Stratospheric ozone demo (sequential scan)" src="assets/ozone-demo-scan.svg" width="100%" />
  <br/><br/>
  <img alt="Stratospheric ozone demo (associative scan)" src="assets/ozone-demo-assoc.svg" width="100%" />
</p>

*Top panel: O₃ density (SAGE II / GOMOS observations, 1984–2011) with smoothed level state ± 2σ and a 15-year `dlmForecast` trend extrapolation. Bottom panel: proxy covariate contributions — solar cycle (β̂·X_solar, amber) and QBO (β̂_qbo1·X₁ + β̂_qbo2·X₂, purple). Model: `order=1`, `harmonics=2`, `seasonLength=12`, 3 static-β covariates, state dimension m=9.*

#### Missing Data (NaN observations)

<p align="center">
  <img alt="Gapped-data demo (sequential scan)" src="assets/gapped-demo-scan.svg" width="100%" />
  <br/><br/>
  <img alt="gapped-data demo (associative scan)" src="assets/gapped-demo-assoc.svg" width="100%" />
</p>

*Nile flow (n=100) with 23 NaN observations. Gray bands mark missing timesteps. Outer light band: observation prediction interval `F·x_smooth ± 2·ystd`; inner opaque band: state uncertainty `smoothed[0] ± 2·smoothedStd[0]`. The smoother interpolates continuously through all gaps with no extra configuration.*

#### Multivariate Observations (p = 2)

<p align="center">
  <img alt="Multivariate demo (sequential scan)" src="assets/multivariate-demo-scan.svg" width="100%" />
  <br/><br/>
  <img alt="Multivariate demo (associative scan)" src="assets/multivariate-demo-assoc.svg" width="100%" />
</p>

*Two sensors observing the same local linear trend state (order=1, m=2, p=2). Each panel shows one sensor's observations with the shared smoothed level ± 2σ bands. Sensor 2 has higher observation noise — the smoother correctly down-weights it. Both `scan` and `assoc` algorithms support p > 1; `sqrt-assoc`, `ud`, and `dlmForecast` throw for p > 1. Validated against Octave reference (`multivariate.test.ts`: 48 tests — order=1, order=0, gapped, order=1+AR(2) × scan/assoc × all backends).*

#### Algorithm compatibility

| Feature | `scan` | `assoc` | `sqrt-assoc` | `ud` |
|---------|--------|---------|--------------|------|
| Multivariate ($p > 1$) | ✅ | ✅ | ❌ throws | ❌ throws |
| `dlmMLE` | ✅ | ✅ | ❌ not wired | ❌ not wired |
| `dlmForecast` | ✅ | ✅ | ✅ | ✅ |
| Float32 stabilization | Joseph form (auto) | built-in | built-in (Cholesky) | built-in (UD factors) |
| Parallel ($O(\log N)$) | ❌ sequential | ✅ | ✅ | ❌ sequential |
| Default backend | cpu, wasm | webgpu | — (explicit) | — (explicit) |

### scan algorithm

`algorithm: 'scan'` uses sequential `lax.scan` for both the Kalman forward filter and RTS backward smoother. It is the default when `algorithm` is not set and the `assoc` path is not auto-selected.

The dominant error source is **not** summation accuracy — it is catastrophic cancellation in the RTS backward smoother step `C_smooth = C - C·N·C`. When the smoothing correction nearly equals the prior covariance, the subtraction amplifies any rounding in the operands. Kahan summation in jax-js-nonconsuming cannot fix this because it only improves the individual dot products, not the outer subtraction. See detailed comments in `src/index.ts`.

**Float32 stabilization (Joseph form):** When `dtype: 'f32'`, the scan path automatically uses Joseph-form stabilization, replacing the standard covariance update `C_filt = C_pred - K·F·C_pred` with:

$$C_{\text{filt}} = (I - K F) \, C_{\text{pred}} \, (I - K F)^\top + K \, V^2 \, K^\top$$

This is algebraically equivalent but numerically more stable — it guarantees a positive semi-definite result even with rounding. Combined with explicit symmetrization (`(C + C') / 2`), this prevents the covariance from going non-positive-definite for m ≤ 2. Without Joseph form, Float32 + scan is numerically unstable for m ≥ 4. Float32 is still skipped in tests for m > 2 even with Joseph form, due to accumulated rounding in the smoother.

**Approaches attempted and rejected:**

- **Diagonal preconditioning** — Rescale the state space by $D = \text{diag}(\sqrt{|\text{diag}(C_0)|} + \epsilon)$ so the filter operates on a better-conditioned system, then untransform the results. Improved joseph+triu precision 2–5× for polynomial models (m ≤ 2), but **degraded** m = 1 models by ~85× and catastrophically broke trigonometric/seasonal models (max |Δ|% > 300,000%) because diagonal scaling distorts the rotation blocks in non-triangular G matrices. Reverted.
- **Iterative refinement** — Re-run the smoother using the smoothed state x[0] and covariance C[0] as the initial state for each additional pass. Three C₀ strategies tested (×100, ×1, x₀-only). All variants were consistently 5–4,550× **worse** across all five benchmark models. The Float32 precision bottleneck is per-timestep rounding accumulation in the forward/backward recursions, not initialization quality — the existing two-pass scheme (diffuse prior → refined) already handles initialization optimally. Reverted.

**MATLAB DLM comparison:** The `dlmsmo.m` reference uses the standard covariance update formula (not Joseph form), combined with explicit `triu + triu'` symmetrization after each filter step (line 77) and `abs(diag(C))` diagonal correction on smoother output (line 114). The improved match seen in the benchmark between dlm-js+joseph/f64 and the Octave reference (9.38e-11 vs 3.78e-8 max |Δ| for f64 without joseph) is not a coincidence — both approaches enforce numerical stability in similar ways, pushing both implementations toward the same stable numerical attractor.

### assoc algorithm

`algorithm: 'assoc'` uses `lax.associativeScan` to evaluate the **exact O(log N) parallel Kalman filter + smoother** from Särkkä & García-Fernández (2020) [1], Lemmas 1–4. Pass `{ algorithm: 'assoc' }` in the options to use it on any backend and any dtype. The assoc path always applies its own numerically stable formulation.

Both passes dispatch ⌈log₂N⌉+1 kernel rounds (Kogge-Stone), giving O(log n) total depth. Results are numerically equivalent to `scan` to within floating-point reordering (validated by `assocscan.test.ts`).

- **Forward filter** (exact 5-tuple from [1, Lemmas 1–2]): Constructs per-timestep 5-tuple elements $(A_k, b_k, C_k, \eta_k, J_k)$ with exact Kalman gains per Lemma 1, composed via `lax.associativeScan` using Lemma 2 with regularized inverse and push-through identity. No approximation — produces the same filtered states as sequential `scan`, up to floating-point reordering.
- **Backward smoother** ([1], Lemmas 3–4; see also Theorem 2): Exact per-timestep smoother gains $E_k = C_{filt,k} G^\top (G C_{filt,k} G^\top + W)^{-1}$ per Lemma 3, computed from the forward-filtered covariances via batched `np.linalg.inv`. Smoother elements $(E_k, g_k, L_k)$ — dlm-js replaces the standard-form $L_k = P_k - E_k G P_k$ (Lemma 3) with a Joseph-form variant for numerical stability — composed via `lax.associativeScan(compose, elems, { reverse: true })` (suffix scan). No accuracy loss — the backward smoother is algebraically equivalent to sequential RTS.

Combined with a WebGPU backend, this provides two orthogonal dimensions of parallelism: across time steps (O(log n) depth via `associativeScan`) and within each step's matrix operations (GPU ALUs). The same technique is used in production by Pyro's `GaussianHMM` [2] and NumPyro's parallel HMM inference [3]. Both EEA-sensors JAX implementations [5][6] use the same 5-tuple forward / 3-tuple backward formulation and `0.5*(C + C')` symmetrization after each composition — confirming the numerical stability approach used by dlm-js.

#### Design choices

| Aspect | Choice | Rationale |
|--------|--------|----------|
| Exact 5-tuple elements | Per-timestep $(A, b, C, \eta, J)$ from Lemma 1 | Each timestep has its own Kalman gain — exact, no approximation. |
| Regularized inverse in compose | $(I + C_i J_j + \epsilon I)^{-1}$ | Guards against near-singular matrices at degenerate (NaN/zero-J) compose steps. $\epsilon = 10^{-6}$ (Float32) or $10^{-12}$ (Float64). |
| Push-through identity | $N = I - J_j M C_i$ | Derives the second inverse from the first — only one `np.linalg.inv` call per compose step. |
| `np.where` NaN blending | $A = \text{where}(\text{isNaN}, G, A_{\text{obs}})$ | Boolean-conditioned blending between observed and missing-step elements. AD-safe since jax-js `297f93a` (np.where VJP). Remaining scalar zeroing sites use `np.multiply(mask, x)`. |
| Scan output disposal | Individual `scanned.*.dispose()` after `x_filt` and `C_filt` recovery | Safe because `np.add` produces new arrays — `x_filt` and `C_filt` are independent of the scan pytree. |

#### Square-root parallel smoother (`sqrt-assoc`)

`algorithm: 'sqrt-assoc'` implements the square-root associative-scan smoother from Yaghoobi et al. [6], reformulating the parallel filter + smoother in **Cholesky-factor space**. The forward 5-tuple becomes $(A, b, U, \eta, Z)$ where $C = U U^\top$ and $J = Z Z^\top$; the backward 3-tuple becomes $(g, E, D)$ where $L = D D^\top$. Covariances are never formed explicitly during composition — they remain PSD by construction, eliminating the need for Joseph form, `(C+C')/2` symmetrization, and `cEps` regularization.

Both passes use `lax.associativeScan` with ⌈log₂N⌉+1 rounds, same as the standard `assoc` path. Results are validated against the same Octave ground truth (`sqrtassoc.test.ts`; 24 tests on wasm — f64: precision comparison for all models including fullSeasonal m=13, f32: all-outputs-finite smoke test for all models including fullSeasonal m=13) with max relative error ~3e-5 (f64).

```ts
const result = await dlmFit(y, {
  obsStd: 10,
  processStd: [5, 2],
  algorithm: 'sqrt-assoc',
});
```

##### Deviations from the reference

The [6] reference implementation (JAX, [EEA-sensors/sqrt-parallel-smoothers](https://github.com/EEA-sensors/sqrt-parallel-smoothers)) uses `jnp.linalg.qr`, `jnp.block`, and `jax.scipy.linalg.solve_triangular`. dlm-js now matches the reference for `tria()` and triangular solves; remaining deviations:

| Operation | Reference [6] | dlm-js | Reason | Impact |
|-----------|--------------|--------|--------|--------|
| `tria(A)` | `_, R = qr(Aᵀ); Rᵀ` | `_, R = qr(Aᵀ); Rᵀ` | ✅ Same | Proper QR-based tria — no condition-number squaring. Works for all state dimensions (m=13 fullSeasonal verified). |
| Block matrices | `jnp.block([[A, B], [C, D]])` | `np.concatenate` along last two axes | jax-js `np.block` doesn't handle batch dims | Equivalent; verbose but correct. |
| Triangular solve | `solve_triangular(L, B)` | `lax.linalg.triangularSolve(L, B)` | ✅ Same | Proper triangular back-substitution; faster than general LU solve. |
| $Z$ padding | Dense $m \times m$ | Rank-1 column + zero padding | Observation dimension $p = 1$ | $J = Z Z^\top$ has rank $\min(p, m) = 1$; zero-padding avoids allocating unused columns. |
| Scalar $1/\Psi_{11}$ | `solve_triangular(Ψ_{11}, ·)` | `np.divide(·, Ψ_{11})` | $p=1$ ⇒ $\Psi_{11}$ is $[n,1,1]$ scalar | Exact; avoids unnecessary matrix solve for scalar denominator. |

##### Known limitations

- **$p > 1$**: Throws for multivariate observations. The Cholesky-factor composition assumes scalar $\Psi_{11}$.
- **MLE**: The sqrt-assoc path is not yet wired into `dlmMLE` / `makeKalmanLoss`. Use `algorithm: 'scan'` or `'assoc'` for MLE.

#### UD factorization (`ud`)

`algorithm: 'ud'` replaces the dense covariance matrix $C$ with its $U D U^\top$ factorization (Bierman 1977; Thornton 1976), where $U$ is unit triangular and $D$ is a positive diagonal vector. The **Bierman measurement update** processes each state component via a scalar column loop, updating $D$ and $U$ in-place without ever forming or inverting a full covariance matrix. The **Thornton time update** (modified weighted Gram-Schmidt) propagates the UD factors through the prediction step directly — no Cholesky re-factorization or epsilon regularization is needed at any stage. Together, these two algorithms maintain the UD factorization purely in factor space, avoiding condition-number squaring and guaranteeing positive-definite $D$ by construction.

The UD path uses sequential `lax.scan` for the forward filter (like `scan`) and reuses the standard RTS backward smoother unchanged. It targets the same use case as `scan` with Joseph form — **Float32 stabilization** — but aims for lower numerical error by avoiding the $O(m^2)$ outer products that amplify rounding.

```ts
const result = await dlmFit(y, {
  obsStd: 120,
  processStd: [40, 10],
  order: 1,
  dtype: 'f32',
  algorithm: 'ud',
});
```

##### Forward step (Bierman measurement + Thornton time update)

Each forward step takes the carry $(x_{t|t-1}, U_{t|t-1}, D_{t|t-1})$ — the predicted state and its UD factors — and observation $y_t$, producing the updated carry plus diagnostic outputs for the backward smoother.

**1. Reconstruct $C_{t|t-1}$.** Since the Thornton time update maintains exact UD factors (no epsilon), the predicted covariance is reconstructed as:

$$C_{t|t-1} = U_{t|t-1} \, \text{diag}(D_{t|t-1}) \, U_{t|t-1}^\top$$

This is emitted as $\mathtt{C\_pred}$ for the backward smoother.

**2. Innovation.** $v_t = y_t - F \, x_{t|t-1}$.

**3. Bierman measurement update (scalar observation, $p=1$).** Clone the carry's $U_{t|t-1}, D_{t|t-1}$ as working arrays. Define $f = U^\top F^\top$ and $g = D \odot f$ (element-wise). Initialize $\alpha = V^2$. For $j = m{-}1$ down to $0$:

$$\alpha^{\text{new}} = \alpha^{\text{old}} + f_j \, g_j$$

$$d_j^+ = d_j \cdot \frac{\alpha^{\text{old}}}{\alpha^{\text{new}}}, \qquad \lambda_j = -\frac{f_j}{\alpha^{\text{old}}}$$

$$\bar{U}_{i,j}^+ = \bar{U}_{i,j} + \lambda_j \, g_i \quad (i > j), \qquad g_i^+ = g_i + \bar{U}_{i,j} \, g_j \quad (i > j)$$

After the loop, $\alpha_{\text{final}} = C_p^{(t)}$ (innovation covariance) and the Kalman gain is $K_{\text{Bierman}} = g_{\text{final}} / \alpha_{\text{final}} = C_{t|t-1} F^\top / C_p^{(t)}$.

> **Note:** The classical Bierman/Thornton presentation uses unit *upper*-triangular $U$. We use unit *lower*-triangular because `np.linalg.cholesky` returns lower-triangular $L$, making $C = L_{\text{unit}} \, D \, L_{\text{unit}}^\top$ the natural factorization without any transpose. The column-loop direction is reversed accordingly ($j = m{-}1$ down to $0$).

**4. State update.** $x_{t|t} = x_{t|t-1} + K_{\text{Bierman}} \, v_t$, then $x_{t+1|t} = G \, x_{t|t}$.

**5. Thornton time update (modified weighted Gram-Schmidt).** Given the Bierman-updated factors $U_t^+, D_t^+$ and process noise $W_t$:

1. Decompose $W_t = L_W \, D_W \, L_W^\top$. For diagonal $W$ (all dlm-js models), $L_W = I$ and $D_W = \text{diag}(W_t)$, extracted directly without Cholesky to avoid $\epsilon$ contamination.
2. Form the augmented matrix $B = G \, U_t^+$ and its companion $L = L_W$, with weight vectors $D_t^+$ and $D_W$.
3. MWGS: for $j = 0, \ldots, m{-}1$:

$$d_j^* = \sum_k {D_t^+}_k \, B_{j,k}^2 + \sum_k {D_W}_k \, L_{j,k}^2$$

$$U_{i,j}^* = \left(\sum_k {D_t^+}_k \, B_{i,k} \, B_{j,k} + \sum_k {D_W}_k \, L_{i,k} \, L_{j,k}\right) / d_j^* \quad (i > j)$$

$$B_{i,k} \leftarrow B_{i,k} - U_{i,j}^* \, B_{j,k}, \qquad L_{i,k} \leftarrow L_{i,k} - U_{i,j}^* \, L_{j,k} \quad (i > j)$$

The output $U^*, D^*$ satisfies $U^* \, \text{diag}(D^*) \, (U^*)^\top = G \, C_{t|t} \, G^\top + W_t = C_{t+1|t}$.

**6. Carry and outputs.** The updated carry is $(x_{t+1|t}, U^*, D^*)$. For the backward smoother:

$$\mathtt{x\_pred} = x_{t|t-1}, \quad \mathtt{C\_pred} = U \, D \, U^\top, \quad \mathtt{K} = G \, K_{\text{Bierman}}, \quad \mathtt{v} = v_t, \quad \mathtt{Cp} = \alpha_{\text{final}}$$

##### Backward smoother

The backward pass is the standard sequential RTS smoother (same function as the `scan` path), operating on the $(x_{\text{pred}}, C_{\text{pred}}, K, v, C_p)$ sequence emitted by the UD forward pass. Since the Thornton time update maintains exact UD factors (no epsilon), the reconstructed $C_{t|t-1}$ is the true predicted covariance.

##### Design choices

| Aspect | Choice | Rationale |
|--------|--------|----------|
| Factorization convention | Unit lower-triangular $U$ | Cholesky gives lower-tri $L$ natively; avoids transpose |
| Measurement update | Bierman column loop ($j = m{-}1$ down to $0$) | Scalar-observation processing; produces gain $K$ and $C_p$ as byproducts |
| Time update | Thornton MWGS on augmented $[G U_{\text{filt}}, L_W]$ | Propagates UD factors directly; no Cholesky re-factorization of $C$; no epsilon needed |
| $W$ factorization | Explicit diagonal extraction of $W_t$ each step | $W$ is diagonal in dlm-js; avoids $\epsilon$ contamination from Cholesky |
| Column loop order | $j = m{-}1$ down to $0$ (Bierman), $j = 0$ to $m{-}1$ (Thornton) | Lower-tri convention: free entries at $i > j$ |
| Backward smoother gain $K$ | $K = G \, K_{\text{Bierman}}$ | Predicted-convention gain from Bierman's exact $C_{\text{pred}} F' / C_p$ |
| NaN masking | $f \leftarrow 0$ when observation is NaN | Entire Bierman loop becomes a no-op: $D, U$ unchanged, $K = 0$ |
| Scope | $p = 1$ only | Bierman's column loop is scalar-observation-specific; $p > 1$ would need a different formulation |

##### Known limitations

- **$p > 1$**: UD factorization currently supports scalar observations only ($p = 1$). Throws for multivariate observations.
- **MLE**: Not yet wired into `dlmMLE`. Use `algorithm: 'scan'` or `'assoc'` for MLE.

##### References

- Bierman, G. J. (1977). *Factorization Methods for Discrete Sequential Estimation*. Academic Press.
- Thornton, C. L. (1976). *Triangular Covariance Factorizations for Kalman Filtering*. Ph.D. Thesis, UCLA. NASA Tech. Memo. 33-798.
- Grewal, M. S. & Andrews, A. P. (2001). *Kalman Filtering: Theory and Practice Using MATLAB*. Wiley. Algorithm 6.1.

### Backend performance

`dlmFit` warm-run timings (jitted core, second of two sequential runs) and maximum errors vs. the Octave/MATLAB reference (worst case across all 5 models and all outputs: yhat, ystd, smoothed, smoothedStd) for each backend × dtype × algorithm × stabilization combination. Regenerate with `pnpm run bench:full`. **Bold rows** are the auto-selected default per backend × dtype.

Stab column: `triu` = `triu(C)+triu(C,1)'` symmetrize (f64 default, matches MATLAB); `joseph` = Joseph-form covariance update + `(C+C')/2` symmetrize (f32/scan default, always on); `joseph+triu` = Joseph form + triu symmetrize instead of `(C+C')/2` (explicit `stabilization:{cTriuSym:true}`); `built-in` = assoc/sqrt-assoc/ud paths use their own exact per-timestep formulation (no external flag); `off` = explicit override disabling the default (`stabilization:{cTriuSym:false}`).

Models: Nile order=0 (n=100, m=1) · Nile order=1 (n=100, m=2) · Kaisaniemi trig (n=117, m=4) · Energy trig+AR (n=120, m=5) · Gapped order=1 (n=100, m=2, 23 NaN). Benchmarked on: <!-- computed:static("machine") -->Intel(R) Core(TM) Ultra 5 125H, 62 GB RAM<!-- /computed --> · GPU: <!-- computed:static("gpu") -->GeForce RTX 4070 Ti SUPER (WebGPU adapter)<!-- /computed -->.

<!-- generated:bench-full-table -->
| backend | dtype | algorithm | stab | Nile o=0 | rel err | Nile o=1 | rel err | Kaisaniemi | rel err | Energy | rel err | Gapped | rel err |
|---------|-------|-----------|------|-------|------|-------|------|-------|------|-------|------|-------|------|
| **cpu** | **f64** | **scan** | **triu** | **251 ms** | **1.73e-16** | **432 ms** | **4.78e-13** | **562 ms** | **4.20e-11** | **601 ms** | **1.19e-11** | **486 ms** | **2.52e-13** |
|  |  | scan | off | 215 ms | 1.73e-16 | 349 ms | 1.40e-11 | 457 ms | 5.85e-10 | 505 ms | 1.06e-6 | 460 ms | 1.03e-12 |
|  |  | assoc | built-in | 147 ms | 4.87e-12 | 325 ms | 4.36e-9 | 1320 ms | 3.17e-8 | 1769 ms | 2.17e-7 | 349 ms | 1.29e-9 |
|  |  | ud | built-in | 383 ms | 1.30e-14 | 879 ms | 1.43e-12 | 1711 ms | 4.01e-5 | 2270 ms | 1.56e-10 | 949 ms | 1.02e-13 |
|  | **f32** | **scan** | **joseph** | **216 ms** | **9.93e-7** | **397 ms** | **2.25e-4** | **509 ms** | **2.93e-3** | **559 ms** | **6.18e-4** | **424 ms** | **1.74e-5** |
|  |  | scan | joseph+triu | 247 ms | 9.93e-7 | 451 ms | 1.09e-4 | 570 ms | 8.98e-3 | 635 ms | 1.41e-3 | 480 ms | 1.40e-4 |
|  |  | assoc | built-in | 156 ms | 4.74e-6 | 330 ms | 3.93e-3 | 1282 ms | 0.03 | 1882 ms | 0.20 | 352 ms | 1.51e-3 |
|  |  | ud | built-in | 358 ms | 1.06e-6 | 874 ms | 7.48e-4 | 1703 ms | 0.02 | 2263 ms | 1.02e-3 | 903 ms | 2.73e-5 |
| **wasm** | **f64** | **scan** | **triu** | **26 ms** | **1.73e-16** | **26 ms** | **4.78e-13** | **27 ms** | **4.20e-11** | **29 ms** | **1.19e-11** | **29 ms** | **2.52e-13** |
|  |  | scan | off | 20 ms | 1.73e-16 | 22 ms | 1.40e-11 | 21 ms | 5.85e-10 | 24 ms | 1.06e-6 | 25 ms | 1.03e-12 |
|  |  | assoc | built-in | 77 ms | 4.87e-12 | 99 ms | 4.36e-9 | 219 ms | 3.17e-8 | 107 ms | 2.17e-7 | 101 ms | 1.29e-9 |
|  |  | sqrt-assoc | built-in | 136 ms | 1.82e-15 | 141 ms | 6.84e-12 | 166 ms | 2.03e-6 | 262 ms | 3.10e-10 | 150 ms | 6.78e-13 |
|  |  | ud | built-in | 32 ms | 1.30e-14 | 52 ms | 1.43e-12 | 74 ms | 4.01e-5 | 94 ms | 1.56e-10 | 50 ms | 1.02e-13 |
|  | **f32** | **scan** | **joseph** | **19 ms** | **1.06e-6** | **24 ms** | **2.59e-4** | **25 ms** | **0.01** | **26 ms** | **2.08e-3** | **25 ms** | **1.32e-4** |
|  |  | scan | joseph+triu | 23 ms | 1.06e-6 | 28 ms | 7.11e-4 | 28 ms | 0.02 | 30 ms | 1.34e-3 | 30 ms | 2.48e-4 |
|  |  | assoc | built-in | 73 ms | 4.74e-6 | 100 ms | 4.53e-3 | 210 ms | 0.03 | 106 ms | 0.22 | 112 ms | 1.19e-3 |
|  |  | sqrt-assoc | built-in | 132 ms | 2.91e-7 | 141 ms | 1.34e-3 | 166 ms | 1.99 | 211 ms | 0.03 | 154 ms | 3.83e-5 |
|  |  | ud | built-in | 29 ms | 1.06e-6 | 49 ms | 7.96e-4 | 72 ms | 3.74e-3 | 82 ms | 1.30e-3 | 50 ms | 1.41e-5 |
| **webgpu** | **f32** | **assoc** | **built-in** | **370 ms** | **4.74e-6** | **456 ms** | **5.01e-3** | **678 ms** | **0.03** | **562 ms** | **0.21** | **556 ms** | **4.87e-4** |
|  |  | scan | joseph | 617 ms | 1.06e-6 | 972 ms | 6.34e-5 | 1217 ms | 9.85e-3 | 1802 ms | 1.06e-3 | 1681 ms | 2.56e-5 |
|  |  | scan | joseph+triu | 924 ms | 1.06e-6 | 1019 ms | 6.34e-5 | 1677 ms | 0.03 | 1870 ms | 2.88e-3 | 1631 ms | 8.20e-5 |
|  |  | ud | built-in | 1056 ms | 1.06e-6 | 1894 ms | 7.70e-4 | 5162 ms | 9.20e-3 | 8065 ms | 3.12e-4 | 3373 ms | 4.61e-5 |
<!-- /generated -->

Each cell shows warm timing and max relative error vs Octave. Errors are per-model, per output variable (yhat, ystd, smoothed, smoothedStd); the Octave reference value is the denominator. Percentages >1% in `assoc` and `sqrt-assoc` rows come from small smoothedStd values (not from yhat/ystd). The `sqrt-assoc` path uses QR-based `tria()` and `lax.linalg.triangularSolve` — covariances are stored as Cholesky factors, ensuring PSD by construction. On cpu, sqrt-assoc has large errors for m > 1 due to the JS interpreter's numerical behaviour; use wasm.

**Key findings** (see table above for exact numbers):
- **WASM is ~10–20× faster than CPU** for these small models. The JS interpreter has significant per-operation overhead.
- **`assoc` on CPU is faster for small m, slower for large m.** For m≥4 the extra matrix compositions dominate (~3× slower than `scan`).
- **`assoc` on WASM is slower than `scan`** — the 5-tuple composition overhead dominates at small n. Prefer `scan` on WASM unless the parallel formulation is explicitly required.
- **`sqrt-assoc` matches reference precision on wasm/f64** and maintains PSD covariances structurally (Cholesky factors) — no Joseph form or symmetrization needed. Slower than `scan` due to QR decompositions per composition step.
- **Stabilization is auto-selected per dtype** — f64 uses `cTriuSym` (triu symmetrize, matching MATLAB `dlmsmo.m`), f32 uses Joseph-form update. The `assoc`/`sqrt-assoc` paths use their own exact formulation regardless of dtype. Overhead is negligible on WASM. Disable f64 symmetrization with `stabilization: { cTriuSym: false }`.
- **f32 precision is limited to ~1–4% max error for large models.** Use f64 when accuracy matters; f32 is safe for all state dimensions with the default stabilization.
- **WebGPU `assoc` is faster than `scan`** — `assoc` dispatches O(log n) rounds via a single `queue.submit()`, while `scan` dispatches O(n) sequential rounds. At small n (100–120), `assoc` is ~1.5–2× faster; the gap widens with larger n (see scaling table below).
- **WASM stays flat up to N≈3200, then scales linearly** (~<!-- timing:scale:wasm-f64:n1638400 -->1902 ms (warm)<!-- /timing --> at N=1.6M). WebGPU scales sub-linearly at small N but approaches O(N) at large N (<!-- timing:scale:webgpu-f32:n100 -->541 ms (cold)<!-- /timing --> → <!-- timing:scale:webgpu-f32:n102400 -->1827 ms (cold)<!-- /timing --> for a 1024× increase). No crossover was observed up to N=1.6M; see scaling table.
- **WebGPU numerical differences** vs WASM/f64 are from Float32 precision and parallel scan reordering, not algorithmic approximation — both paths use exact per-timestep Kalman gains.

For background on the Nile and Kaisaniemi demos and the original model formulation, see [Marko Laine's DLM page](https://mjlaine.github.io/dlm/). The energy demand demo uses synthetic data generated for this project. The gapped-data demo uses the same Nile dataset with 23 observations removed.


#### WebGPU/f32/assoc vs WASM/f64/scan warm-run benchmark

`dlmFit` warm-run timings (jitted core, second of two runs):

| Model | $n$ | $m$ | wasm / f64 / scan (warm) | webgpu / f32 / assoc (warm) |
|-------|-----|-----|--------------------------|------------------------------|
| Nile, order=0 | 100 | 1 | <!-- timing:bb:nile-o0:wasm-f64 -->24 ms<!-- /timing --> | <!-- timing:bb:nile-o0:webgpu-f32 -->339 ms<!-- /timing --> |
| Nile, order=1 | 100 | 2 | <!-- timing:bb:nile-o1:wasm-f64 -->32 ms<!-- /timing --> | <!-- timing:bb:nile-o1:webgpu-f32 -->349 ms<!-- /timing --> |
| Kaisaniemi, trig | 117 | 4 | <!-- timing:bb:kaisaniemi:wasm-f64 -->27 ms<!-- /timing --> | <!-- timing:bb:kaisaniemi:webgpu-f32 -->510 ms<!-- /timing --> |
| Energy, trig+AR | 120 | 5 | <!-- timing:bb:trigar:wasm-f64 -->26 ms<!-- /timing --> | <!-- timing:bb:trigar:webgpu-f32 -->515 ms<!-- /timing --> |

**WebGPU/f32/assoc vs WASM/f64/scan scaling: O(log n) vs O(n).**

A scaling benchmark (Nile order=1, m=2) measures `dlmFit` at exponentially increasing N. WASM `jit()` is polymorphic in N (one JIT compilation covers all array sizes), so WASM timings are warm (4 runs, median). WebGPU `jit()` is not polymorphic in N (each new N recompiles), so WebGPU timings are cold (single first call including JIT). WASM uses sequential `scan`; WebGPU uses `assoc` (both forward filter and backward smoother via `lax.associativeScan`):

| N | wasm/f64/scan (warm) | webgpu/f32/assoc (cold) | ratio |
|---|----------------------|-------------------------|-------|
| 100 | <!-- timing:scale:wasm-f64:n100 -->29 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n100 -->541 ms<!-- /timing --> | 17.9× |
| 200 | <!-- timing:scale:wasm-f64:n200 -->26 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n200 -->500 ms<!-- /timing --> | 18.4× |
| 400 | <!-- timing:scale:wasm-f64:n400 -->31 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n400 -->503 ms<!-- /timing --> | 18.3× |
| 800 | <!-- timing:scale:wasm-f64:n800 -->28 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n800 -->543 ms<!-- /timing --> | 19.6× |
| 1600 | <!-- timing:scale:wasm-f64:n1600 -->29 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n1600 -->557 ms<!-- /timing --> | 19.7× |
| 3200 | <!-- timing:scale:wasm-f64:n3200 -->30 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n3200 -->615 ms<!-- /timing --> | 21.2× |
| 6400 | <!-- timing:scale:wasm-f64:n6400 -->31 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n6400 -->652 ms<!-- /timing --> | 21.2× |
| 12800 | <!-- timing:scale:wasm-f64:n12800 -->36 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n12800 -->738 ms<!-- /timing --> | 20.1× |
| 25600 | <!-- timing:scale:wasm-f64:n25600 -->47 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n25600 -->899 ms<!-- /timing --> | 14.8× |
| 51200 | <!-- timing:scale:wasm-f64:n51200 -->82 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n51200 -->1203 ms<!-- /timing --> | 15.4× |
| 102400 | <!-- timing:scale:wasm-f64:n102400 -->132 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n102400 -->1827 ms<!-- /timing --> | 13.8× |
| 204800 | <!-- timing:scale:wasm-f64:n204800 -->244 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n204800 -->2943 ms<!-- /timing --> | 12.3× |
| 409600 | <!-- timing:scale:wasm-f64:n409600 -->484 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n409600 -->4853 ms<!-- /timing --> | 10.2× |
| 819200 | <!-- timing:scale:wasm-f64:n819200 -->969 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n819200 -->9376 ms<!-- /timing --> | 10.0× |
| 1638400 | <!-- timing:scale:wasm-f64:n1638400 -->1902 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n1638400 -->17978 ms<!-- /timing --> | 9.5× |

Three findings:

1. **WASM stays flat up to N≈3200**, then grows roughly linearly (O(n)). The per-step cost asymptotes around ~1.2 µs/step (<!-- timing:scale:wasm-f64:n1638400 -->1902 ms (warm)<!-- /timing --> at N=1638400). The flat region reflects fixed JIT/dispatch overhead, not compute. WASM OOM previously occurred at N=3276800 due to a signed 32-bit integer overflow in the page-count calculation — fixed upstream in jax-js-nonconsuming (bb126c6+a6d37da).

2. **WebGPU cold timings include per-N JIT recompilation** (WebGPU `jit()` is not polymorphic in N). At small N the ~500 ms JIT overhead dominates; at large N the GPU arithmetic dominates. Each associativeScan pass dispatches ⌈log₂N⌉+1 Kogge-Stone rounds, each operating on all N elements in parallel — so total GPU work is O(N log N), while WASM sequential scan is O(N). A 1024× increase from N=100 to N=102400 roughly triples the runtime (<!-- timing:scale:webgpu-f32:n100 -->541 ms (cold)<!-- /timing --> → <!-- timing:scale:webgpu-f32:n102400 -->1827 ms (cold)<!-- /timing -->).

3. **The cold-WebGPU-to-warm-WASM ratio decreases as N grows** (~18× at small N where JIT overhead dominates, ~10× at N=1.6M where GPU execution time dominates). No crossover was observed up to N=1638400.


## MLE

### Demos

#### Nile MLE Optimization

<p align="center">
  <img alt="Nile MLE optimization (sequential scan)" src="assets/nile-mle-anim-scan.svg" width="100%" />
  <br/><br/>
  <img alt="Nile MLE optimization (associative scan)" src="assets/nile-mle-anim-assoc.svg" width="100%" />
  <br/><br/>
  <img alt="Nile MLE optimization (WebGPU)" src="assets/nile-mle-anim-webgpu.svg" width="100%" />
</p>

*Parameter estimation via autodiff (`dlmMLE`). The animated SVG shows iteration-by-iteration convergence of the level ± 2σ bands. WASM variants use natural gradient (second-order); WebGPU uses natural gradient for Nile. Estimated observation noise s = 121.1 (known: 122.9), −2·log-likelihood = 1105.0. WebGPU is slower than WASM at n=100 due to dispatch overhead, but pays off at large N.*

#### Energy MLE Optimization (with AR estimation)

<p align="center">
  <img alt="Energy MLE optimization (sequential scan)" src="assets/energy-mle-anim-scan.svg" width="100%" />
  <br/><br/>
  <img alt="Energy MLE optimization (associative scan)" src="assets/energy-mle-anim-assoc.svg" width="100%" />
  <br/><br/>
  <img alt="Energy MLE optimization (WebGPU)" src="assets/energy-mle-anim-webgpu.svg" width="100%" />
</p>

*Joint estimation of observation noise s, state variances w, and AR(1) coefficient φ via autodiff (`dlmMLE` with `fitAr: true`). WASM variants use natural gradient; WebGPU uses Adam (FD Hessian too noisy in f32 for 7 parameters). Two sparklines track convergence: −2·log-likelihood (amber) and AR coefficient φ (green, 0.50 → 0.68, true: 0.85).*

The Nile MLE demo estimates `obsStd` and `processStd` on the classic Nile dataset; the energy MLE demo jointly estimates `obsStd`, `processStd`, and AR coefficient `φ` on the synthetic energy model (`fitAr: true`). See [Parameter estimation (MLE & MAP): MATLAB DLM vs dlm-js](#parameter-estimation-mle--map-matlab-dlm-vs-dlm-js) for details.

### How MLE works

Noise parameters are optimized in log-space: $s = e^{\theta_s}$, $w_i = e^{\theta_{w,i}}$. AR coefficients are optimized directly (unconstrained, not log-transformed — matching MATLAB DLM behavior). Two optimizers are available: natural gradient (second-order, default for f64) and Adam (first-order, default for f32); see [Optimizers](#optimizers) for details.

The entire optimization step is wrapped in a single `jit()` call. For Adam, this includes `valueAndGrad(loss)` (Kalman filter forward pass + AD backward pass) and optax Adam parameter update; for natural gradient, the `jit(valueAndGrad(loss))` call is reused for both the gradient and finite-difference Hessian evaluations. The `jit()` compilation happens on the first iteration; subsequent iterations run from compiled code.

**Performance**: on the `wasm` backend, one Nile MLE run (100 observations, m = 2) converges in <!-- timing:mle-bench:nile-order1:iterations -->190<!-- /timing --> iterations (~<!-- timing:mle-bench:nile-order1:elapsed -->3259 ms<!-- /timing -->) with Adam (b2=0.9), or <!-- timing:nat-mle-bench:nile-order1:iterations -->15<!-- /timing --> iterations (~<!-- timing:nat-mle-bench:nile-order1:elapsed -->2117 ms<!-- /timing -->) with the natural gradient optimizer.

**Two loss paths:** `dlmMLE` dispatches between two loss functions based on the `dtype` and backend:

- **CPU/WASM (any dtype):** `makeKalmanLoss` — sequential `lax.scan` forward filter (O(n) depth per iteration). For the energy demo (n=120, <!-- timing:energy-mle:iterations -->19<!-- /timing --> iters, ~<!-- timing:energy-mle:elapsed -->5.7 s<!-- /timing --> on WASM).
- **WebGPU + Float32:** `makeKalmanLossAssoc` — `lax.associativeScan` forward filter (O(log n) depth per iteration). Details below.

Both paths are wrapped in `jit(valueAndGrad(lossFn))`. The final refit after convergence calls `dlmFit` (which itself uses the parallel path on WebGPU).

#### `makeKalmanLossAssoc` — parallel MLE loss via associative scan

The parallel MLE loss function replaces the sequential Kalman forward pass inside `valueAndGrad` with the exact 5-tuple forward filter from [1, Lemmas 1–2]. Each timestep produces per-step Kalman gains directly. Gradients propagate through $\theta \to (W, V^2) \to$ scan elements $\to$ loss naturally because the element construction uses standard differentiable ops.

**Step-by-step derivation:**

1. **Parameter extraction (traced):** $\theta \xrightarrow{\exp} (s, w_0 \ldots w_{m-1}, \phi_1 \ldots \phi_p)$. Observation variance $V^2 = s^2$ (scalar); state noise $W = \text{diag}(w_i^2)$; $G$ updated with AR coefficients if `fitAr: true`.

2. **Per-timestep 5-tuple elements (Lemma 1):** For each timestep $t = 1 \ldots n$:

   $$S_t = F W F^\top + V^2, \quad K_t = W F^\top / S_t$$
   $$A_t = (I - K_t F) G, \quad b_t = K_t y_t, \quad C_t = (I - K_t F) W$$
   $$\eta_t = G^\top F^\top y_t / S_t, \quad J_t = G^\top F^\top F G / S_t$$

   Missing timesteps ($\text{mask}_t = 0$): $A_t = G$, $b_t = 0$, $C_t = W$, $\eta_t = 0$, $J_t = 0$.

   Blending uses float-mask arithmetic for clean autodiff behavior.

3. **First element (exact prior initialization):** $A_1 = 0$, $b_1 = x_0 + K_1 (y_1 - F x_0)$, $C_1 = C_0 - K_1 S_1 K_1^\top$, $\eta_1 = 0$, $J_1 = 0$.

4. **Prefix scan:** `lax.associativeScan(composeForward, {A, b, C, η, J})` composes all $n$ elements in O(log n) depth using Lemma 2:

   $$M = (I + C_i J_j + \epsilon I)^{-1}$$
   $$A_{ij} = A_j M A_i, \quad b_{ij} = A_j M (b_i + C_i \eta_j) + b_j, \quad C_{ij} = A_j M C_i A_j^\top + C_j$$
   $$\eta_{ij} = A_i^\top N (\eta_j - J_j b_i) + \eta_i, \quad J_{ij} = A_i^\top N J_j A_i + J_i$$

   where $N = I - J_j M C_i$ (push-through identity — only one matrix inverse per compose step).

5. **Filtered state/covariance recovery:**

   $$x_{\text{filt},t} = A_{\text{comp},t} \, x_0 + b_{\text{comp},t}$$
   $$C_{\text{filt},t} = A_{\text{comp},t} \, C_0 \, A_{\text{comp},t}^\top + C_{\text{comp},t} \quad \text{(symmetrized)}$$

   Note: $x_{\text{filt}}$ and $C_{\text{filt}}$ are new arrays produced by `np.add`, not aliases of the scan output — the scan pytree is safely disposed immediately after.

6. **One-step-ahead predictions (shift):** The prediction-error likelihood requires $x_{t|t-1}$ and $C_{t|t-1}$ (the *predicted* state before observing $y_t$):

   $$x_{\text{pred},0} = x_0, \quad x_{\text{pred},t} = G \, x_{\text{filt},t-1} \quad (t \geq 1)$$
   $$C_{\text{pred},0} = C_0, \quad C_{\text{pred},t} = G \, C_{\text{filt},t-1} \, G^\top + W \quad (t \geq 1)$$

7. **Log-likelihood (prediction-error decomposition):**

   $$v_t = y_t - F \, x_{\text{pred},t}, \quad C_p^{(t)} = F \, C_{\text{pred},t} \, F^\top + V^2$$
   $$-2 \log L = \sum_{t=1}^{n} \text{mask}_t \left[ \frac{v_t^2}{C_p^{(t)}} + \log C_p^{(t)} \right]$$

   This is the same objective as the sequential path — both minimize the Kalman filter prediction-error decomposition.

### Optimizers

`dlmMLE` supports two optimizers, selected via `optimizer: 'natural'` or `optimizer: 'adam'`. The default depends on dtype: **natural gradient** for f64 (second-order, fast convergence), **Adam** for f32 (FD Hessian too noisy in single precision). Both operate on the same differentiable loss function — the Kalman filter prediction-error decomposition $\ell(\theta) = -2 \log L(\theta)$ — and both use `jit(valueAndGrad(lossFn))` for gradients.

#### Steepest descent framework

Both optimizers can be understood as instances of **steepest descent under a norm** (Bernstein & Newhouse 2024). Given a loss $\ell(\theta)$ with gradient $g = \nabla_\theta \ell$ and a norm $\|\cdot\|$ on the parameter space, the optimal update minimizing a quadratic model is:

$$\Delta\theta^* = \operatorname{arg\,min}_{\Delta\theta} \left[ g^\top \Delta\theta + \frac{\lambda}{2} \|\Delta\theta\|^2 \right] = -\frac{\|g\|^\dagger}{\lambda} \cdot \mathcal{D}_{\|\cdot\|}\, g$$

where $\|g\|^\dagger$ is the dual norm and $\mathcal{D}_{\|\cdot\|}$ is the *duality map* — it converts the gradient (a member of the dual space) into a primal-space update direction. The choice of norm determines the geometry:

| Norm on $\Delta\theta$ | Duality map | Update rule | Method |
|------------------------|-------------|-------------|--------|
| Euclidean: $\|\Delta\theta\|_2$ | $\mathcal{D}\,g = g / \|g\|_2$ | $\Delta\theta = -\alpha\, g$ | Gradient descent |
| Hessian: $\|\Delta\theta\|_H^2 = \Delta\theta^\top H \Delta\theta$ | $\mathcal{D}\,g = H^{-1} g$ | $\Delta\theta = -(H + \lambda I)^{-1} g$ | Newton / natural gradient |

The Euclidean norm treats all parameter directions equally; the Hessian norm uses local curvature to rescale directions, yielding faster convergence on ill-conditioned objectives.

#### Adam (first-order, default for f32)

Adam (Kingma & Ba 2015) maintains exponential moving averages of the gradient and squared gradient:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_{t+1} = \theta_t - \alpha \, \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

This is a diagonal approximation to natural gradient — $\hat{v}_t$ estimates per-parameter curvature, but ignores cross-parameter correlations.

**dlm-js defaults:** `b1=0.9`, `b2=0.9`, `eps=1e-8`, `lr=0.05`, `maxIter=200`. The non-standard `b2=0.9` was selected empirically — it converges ~3× faster than the canonical 0.999 on DLM log-likelihoods (measured across Nile, Kaisaniemi, ozone benchmarks). The entire Adam step (forward filter + AD backward pass + Adam state update) is wrapped in a single `jit()` call.

```js
const mle = await dlmMLE(y, {
  order: 1,
  optimizer: 'adam',      // explicit (default for f32)
  maxIter: 300,
  lr: 0.05,
  dtype: 'f32',
});
```

#### Natural gradient (second-order, default for f64, `optimizer: 'natural'`)

The natural gradient optimizer uses second-order curvature information to take Newton-type steps. In the steepest-descent framework above, the norm on $\Delta\theta$ is the quadratic form induced by the Hessian $H = \nabla^2_\theta \ell$, which gives the update:

$$\Delta\theta = -\eta \cdot (H + \lambda I)^{-1} g$$

where $g = \nabla_\theta \ell$ is the gradient, $H$ is the Hessian of the log-likelihood, and $\lambda > 0$ is a Levenberg-Marquardt damping parameter that interpolates between Newton ($\lambda \to 0$) and gradient descent ($\lambda \to \infty$).

**Step-by-step algorithm:**

1. **Gradient evaluation.** Compute $({\ell(\theta),\, g})$ via the JIT-compiled `valueAndGrad(lossFn)`.

2. **Hessian approximation.** Two modes:
   - `hessian: 'fd'` (default): central finite differences of the gradient. For each parameter $j = 1 \ldots p$:

     $$H_{\cdot,j} \approx \frac{g(\theta + h\, e_j) - g(\theta - h\, e_j)}{2h}, \quad h = 10^{-5}$$

     Cost: $2p$ gradient evaluations reusing the same JIT-compiled `gradFn` — no extra tracing. For $p \leq 7$ (typical DLM), this is ~200–700 ms on WASM. Result is symmetrized: $H \leftarrow (H + H^\top) / 2$.

   - `hessian: 'exact'`: `jit(hessian(lossFn))` via AD. Exact but expensive — the first call incurs ~20 s of JIT tracing (jax-js v0.7.8); subsequent calls are fast.

3. **Marquardt initialization** (first iteration only): $\lambda_0 = \tau \cdot \max_i |H_{ii}|$, with $\tau = 0.1$ (default). This starts with moderate damping to avoid overshooting into wrong basins.

4. **Linear solve.** Solve $(H + \lambda I)\,\delta = g$ via Gaussian elimination with partial pivoting. For $p \leq 10$, this is exact and trivially fast in plain JS.

5. **Backtracking line search.** Try candidate $\theta' = \theta - \eta\, \delta$ (default $\eta = 1$, full Newton step):
   - If $\ell(\theta') < \ell(\theta)$: accept, shrink damping $\lambda \leftarrow \lambda \cdot \rho_{\text{shrink}}$ (trust the quadratic model more).
   - If $\ell(\theta') \geq \ell(\theta)$: reject, grow damping $\lambda \leftarrow \lambda \cdot \rho_{\text{grow}}$, re-solve, retry (up to 6 attempts).
   - If all attempts fail: fall back to a small gradient-descent step.

   Defaults: $\rho_{\text{shrink}} = 0.5$, $\rho_{\text{grow}} = 2$.

6. **Convergence.** Stop when the relative change in $-2\log L$ falls below `tol` (default $10^{-6}$) or `maxIter` is reached (default 50).

**Why this is effective for DLM MLE:** The log-likelihood $\ell(\theta) = -2 \log L(\theta)$ of a DLM is smooth and close to quadratic near the optimum (it is a sum-of-squares-like prediction-error decomposition). The Hessian captures cross-parameter curvature — e.g. the coupling between observation noise $s$ and state noise $w$ — that diagonal methods like Adam miss. Convergence is typically quadratic near the optimum: 3–15 iterations vs 100–300 for Adam.

**Cost trade-off:** Each natural gradient iteration is more expensive ($1 + 2p$ gradient evaluations for FD Hessian) but far fewer iterations are needed. For $p \leq 5$ and $n \leq 200$, natural gradient is ~1.5–2× faster in wall-clock time; for larger $p$ the FD cost grows and Adam becomes competitive.

```js
const mle = await dlmMLE(y, {
  order: 1,
  optimizer: 'natural',   // default for f64
  maxIter: 50,
  naturalOpts: {
    hessian: 'fd',         // 'fd' (default) or 'exact'
    lambdaInit: 0.1,       // τ for Marquardt init
    lambdaShrink: 0.5,     // ρ_shrink: on success
    lambdaGrow: 2,         // ρ_grow: on failure
    fdStep: 1e-5,          // h for central differences
  },
  dtype: 'f64',
});
```

### Parameter estimation (MLE & MAP): MATLAB DLM vs dlm-js

This comparison focuses on the univariate estimation workflow ($p=1$). `dlmMLE` supports both pure MLE (default) and MAP estimation via the `loss` option — see [MAP estimation](#map-estimation-custom-loss--priors) for API details. For the original MATLAB DLM, see the [tutorial](https://mjlaine.github.io/dlm/dlmtut.html) and [source](https://github.com/mjlaine/dlm).

#### Objective function

Both optimize the same scalar likelihood form (for $p=1$ observations) — $-2 \log L$ from the Kalman filter prediction error decomposition:

$$-2 \log L = \sum_{t=1}^{n} \left[ \frac{v_t^2}{C_p^{(t)}} + \log C_p^{(t)} \right]$$

where $v_t = y_t - F x_{t|t-1}$ is the innovation and $C_p^{(t)} = F C_{t|t-1} F' + V^2$ is the innovation covariance. The `dlm_costfun` function (inside `dlmfit.m`) calls `dlmsmo(...,0,0)` (filter only, no smoother, no sample) and returns `out.lik`; we ran this under Octave. The dlm-js `makeKalmanLoss` in `src/mle.ts` computes the same per-step terms via `lax.scan` over the forward filter steps.

In practice, exact numeric equality is not expected because initialization and optimization procedures differ (e.g., `dlmfit` uses a two-pass prefit for initial state/covariance before optional optimization, as run under Octave).

#### Parameterization

| Aspect | dlm-js | MATLAB DLM |
|--------|--------|-----------|
| Observation noise $s$ | Always fitted: $s = e^{\theta_s}$ | Optionally fitted as a multiplicative factor $V \cdot e^{\theta_v}$ (controlled by `options.fitv`) |
| State noise $w$ | $W_{ii} = (e^{\theta_{w,i}})^2$ via `buildDiagW` | $W_{ii} = (e^{\theta_{w,i}})^2$ |
| AR coefficients | Directly optimized (not log-transformed): $G(\text{arInds}) = \theta_\phi$ via `buildG` rank-1 update (AD-safe) | Directly optimized (not log-transformed): $G(\text{arInds}) = \theta_\phi$ |
| Parameter grouping | Each $W_{ii}$ is an independent parameter | `options.winds` maps $\text{diag}(W)$ entries to shared parameters (e.g., `winds=[1,1,2,2]` ties states 1&2 and 3&4) |

Both use the same positivity enforcement: log-space for variance parameters, then $e^{(\cdot)}$ to map back. The MATLAB version has an extra feature — `winds` — that lets you **tie** $\text{diag}(W)$ entries to shared parameters, reducing the optimization dimension when multiple states should share the same noise variance.

#### Optimizer

| Aspect | dlm-js | MATLAB DLM |
|--------|--------|-----------|
| **Algorithm** | Adam (1st-order) or natural gradient (2nd-order, LM-damped Newton) — see [Optimizers](#optimizers) | `fminsearch` (Nelder-Mead simplex) |
| **Gradient computation** | **Autodiff** via `valueAndGrad()` + reverse-mode AD through `lax.scan` | **None** — derivative-free |
| **Typical run budget** | Adam: 200 iterations; natural: 50 iterations | 400 function evaluations (`options.maxfuneval`) |
| **Compilation** | `jit()`-traced optimization step (forward filter + AD + parameter update) | None (interpreted; tested under Octave, or optional `dlmmex` C MEX) |
| **WASM performance** | Adam: ~<!-- timing:ckpt:nile:false-s -->2.3 s<!-- /timing --> for 60 iters (Nile, n=100, m=2, `checkpoint: false`); natural: ~1.5 s for 5 iters | N/A |

**Key tradeoff**: Nelder-Mead needs only function evaluations (no gradients), making it robust on non-smooth surfaces but expensive as parameter dimension grows. Adam uses first-order gradients; natural gradient uses second-order curvature (Hessian) for faster convergence on smooth objectives like the DLM log-likelihood. See [Optimizers](#optimizers) for equations and algorithmic details.

##### MLE, MAP, and MCMC: three estimation approaches

Pure MLE minimises $-2 \log L$ without any prior on $W$. On real data such as satellite ozone measurements, this can produce degenerate solutions — e.g. most seasonal noise variances collapse to near-zero while one or two grow large — because the likelihood surface has a wide, nearly flat ridge.

**MAP estimation** (dlm-js `loss` option) adds prior penalties to the likelihood, regularising the solution without the cost of full posterior sampling. dlm-js provides `dlmPrior` for MATLAB DLM-style Inverse-Gamma priors on variances and Normal priors on AR coefficients (matching `dlmGibbsDIG` conventions), or accepts a custom callback for arbitrary penalties. The entire chain — Kalman loss + prior penalty + AD backward pass + optimizer update — is compiled in a single `jit()` call. The result separates `deviance` (pure $-2\log L$) from `priorPenalty` for diagnostics.

**MCMC** (MATLAB DLM only) uses a normal prior on $\log W$ entries and produces a full posterior chain with credible intervals, yielding a posterior mean at much higher $-2\log L$ but visually smoother, better-regularised results.

| Point | dlm-js MLE | dlm-js MAP | MATLAB MCMC |
|-------|------------|------------|------------|
| Ozone $-2\log L$ at MATLAB posterior W | — | — | 435.6 |
| Ozone $-2\log L$ at MLE optimum | 203.8 | — | — |
| Regularisation | None | Prior penalty on $W$, optional on $s$ and $\phi$ | Full posterior via MCMC |
| Output | Point estimate | Point estimate + `priorPenalty` decomposition | Full chain + credible intervals |

See [MAP estimation](#map-estimation-custom-loss--priors) for API details and code examples.

#### Benchmark: same machine, same data

All timings measured on the same machine: <!-- computed:static("machine")-->Intel(R) Core(TM) Ultra 5 125H, 62 GB RAM<!-- /computed -->. The MATLAB DLM toolbox was run under Octave with `fminsearch` (Nelder-Mead, `maxfuneval=400` for Nile models, `maxfuneval=800` for Kaisaniemi). dlm-js uses `dlmMLE` with two optimizers: **Adam** (first-order, `maxIter=300`, `b2=0.9`, `checkpoint: false`) and **natural gradient** (second-order LM-damped Fisher scoring with FD Hessian, `maxIter=50`), both on `wasm` backend, `Float64`. Octave timings are median of 5 runs after 1 warmup (measured <!-- computed:static("_measured")-->2026-02<!-- /computed -->); dlm-js timings are single fresh-run wall-clock times (including first-call JIT overhead).

| Model | $n$ | $m$ | params | dlm-js Adam (wasm) | dlm-js natural (wasm) | Octave `fminsearch` | $-2\log L$ (Adam) | $-2\log L$ (natural) | $-2\log L$ (Octave) |
|-------|---|---|--------|--------------------|-----------------------|---------------------|-------------------|----------------------|---------------------|
| Nile, order=1, fit s+w | 100 | 2 | 3 | <!-- timing:mle-bench:nile-order1:elapsed -->3259 ms<!-- /timing --> | <!-- timing:nat-mle-bench:nile-order1:elapsed -->2117 ms<!-- /timing --> | <!-- computed:static("octave-nile-order1-elapsed-ms") + " ms" -->2814 ms<!-- /computed --> | <!-- timing:mle-bench:nile-order1:lik -->1104.9<!-- /timing --> | <!-- timing:nat-mle-bench:nile-order1:lik -->1104.9<!-- /timing --> | <!-- computed:static("octave-nile-order1-lik") -->1104.6<!-- /computed --> |
| Nile, order=1, fit w only | 100 | 2 | 2 | <!-- timing:mle-bench:nile-wonly:elapsed -->3088 ms<!-- /timing --> | <!-- timing:nat-mle-bench:nile-wonly:elapsed -->1733 ms<!-- /timing --> | <!-- computed:static("octave-nile-w-only-elapsed-ms") + " ms" -->1614 ms<!-- /computed --> | <!-- timing:mle-bench:nile-wonly:lik -->1104.9<!-- /timing --> | <!-- timing:nat-mle-bench:nile-wonly:lik -->1104.9<!-- /timing --> | <!-- computed:static("octave-nile-w-only-lik") -->1104.7<!-- /computed --> |
| Nile, order=0, fit s+w | 100 | 1 | 2 | <!-- timing:mle-bench:nile-order0:elapsed -->2277 ms<!-- /timing --> | <!-- timing:nat-mle-bench:nile-order0:elapsed -->1336 ms<!-- /timing --> | <!-- computed:static("octave-nile-order0-elapsed-ms") + " ms" -->607 ms<!-- /computed --> | <!-- timing:mle-bench:nile-order0:lik -->1095.8<!-- /timing --> | <!-- timing:nat-mle-bench:nile-order0:lik -->1095.8<!-- /timing --> | <!-- computed:static("octave-nile-order0-lik") -->1095.8<!-- /computed --> |
| Kaisaniemi, trig, fit s+w | 117 | 4 | 5 | <!-- timing:mle-bench:kaisaniemi:elapsed -->4964 ms<!-- /timing --> | <!-- timing:nat-mle-bench:kaisaniemi:elapsed -->4068 ms<!-- /timing --> | **failed** (NaN/Inf) | <!-- timing:mle-bench:kaisaniemi:lik -->341.3<!-- /timing --> | <!-- timing:nat-mle-bench:kaisaniemi:lik -->341.3<!-- /timing --> | — |
| Energy, trig+AR, fit s+w+φ | 120 | 5 | 7 | <!-- timing:energy-mle:elapsed-ms -->5718 ms<!-- /timing --> | <!-- timing:nat-mle-bench:energy:elapsed -->5614 ms<!-- /timing --> | — | <!-- timing:energy-mle:lik -->443.1<!-- /timing --> | <!-- timing:nat-mle-bench:energy:lik -->443.1<!-- /timing --> | — |

Octave timings are from Octave with `fminsearch`; Nile, Kaisaniemi, and Nile (w only) dlm-js timings are from `pnpm run bench:mle`; Energy Adam timings are from `collect-energy-mle-frames.ts` (includes frame extraction overhead), Energy natural gradient from `bench:mle`. All dlm-js timings are single fresh-run wall-clock times including JIT overhead.

**Key observations:**
- **Nile (n=100, m=2):** Octave `fminsearch` is <!-- computed:static("octave-nile-order1-elapsed-ms") < slot("mle-bench:nile-order1:elapsed") ? "faster" : "slower" -->faster<!-- /computed --> (see table). dlm-js includes one-time JIT compilation overhead in the reported time.
- **Likelihood values:** All optimizers converge to very similar $-2\log L$ values on Nile (Adam vs Octave difference ~<!-- computed:Math.abs(slot("mle-bench:nile-order1:lik") - static("octave-nile-order1-lik")).toFixed(1) -->0.3<!-- /computed -->).
- **Natural gradient:** Uses second-order curvature (FD Hessian + Levenberg-Marquardt damping) and converges in fewer iterations (≤50 vs 300 for Adam), but each iteration is more expensive due to per-parameter finite-difference Hessian evaluations.
- **Kaisaniemi (m=4, 5 params):** Octave `fminsearch` (`maxfuneval=800`) failed with NaN/Inf; Adam converged in <!-- timing:mle-bench:kaisaniemi:iterations -->300<!-- /timing --> iterations (~<!-- timing:mle-bench:kaisaniemi:elapsed-s -->5.0 s<!-- /timing -->), natural gradient in <!-- timing:nat-mle-bench:kaisaniemi:iterations -->20<!-- /timing --> iterations (~<!-- timing:nat-mle-bench:kaisaniemi:elapsed-s -->4.1 s<!-- /timing -->).
- **Joint $s+w$ fitting:** dlm-js can fit both $s$ and $w$ jointly, or fix $s$ via `obsStdFixed` (matching MATLAB DLM's `fitv=0`).

##### Gradient checkpointing

`lax.scan` supports gradient checkpointing via a `checkpoint` option: `true` (default, √n segments), `false` (store all carries), or an explicit segment size.

**Benchmark (WASM, Float64, 60 iterations):**

| Dataset | n | m | `checkpoint: false` ($n$, warm) | `checkpoint: true` ($\sqrt{n}$, warm) | speedup |
|---------|---|---|-------------------------------|---------------------------------------|---------|
| Nile, order=1 | 100 | 2 | <!-- timing:ckpt:nile:false-ms -->2302 ms<!-- /timing --> | <!-- timing:ckpt:nile:true-ms -->2294 ms<!-- /timing --> | <!-- timing:ckpt:nile:speedup -->0%<!-- /timing --> |
| Energy, order=1+trig1+ar1 | 120 | 5 | <!-- timing:ckpt:energy:false-ms -->2848 ms<!-- /timing --> | <!-- timing:ckpt:energy:true-ms -->2918 ms<!-- /timing --> | <!-- timing:ckpt:energy:speedup -->+2%<!-- /timing --> |


#### MCMC (MATLAB DLM only)

The MATLAB DLM toolbox supports MCMC via Adaptive Metropolis (`mcmcrun`): 5000 simulations, log-normal priors, full posterior chain with credible intervals, and disturbance smoother for Gibbs-style state sampling.

**dlm-js has no MCMC equivalent** — `dlmMLE` returns a point estimate (MLE or MAP), not a posterior chain. For regularisation without MCMC, use MAP estimation with `dlmPrior` or a custom `loss` callback — see [MAP estimation](#map-estimation-custom-loss--priors). Possible future directions:
- Hessian at the MLE optimum for approximate confidence intervals
- Stochastic gradient MCMC (e.g., SGLD) using the existing AD infrastructure

#### Feature comparison summary

| Capability | dlm-js `dlmMLE` | MATLAB DLM |
|-----------|-----------------|-----------|
| MLE point estimate | ✅ Adam + autodiff | ✅ `fminsearch` |
| Gradient-based optimization | ✅ | ❌ |
| Second-order optimizer | ✅ (`optimizer: 'natural'`; see [Optimizers](#optimizers)) | ❌ |
| JIT compilation of optimizer | ✅ | ❌ |
| Fit observation noise `obsStd` | ✅ (always) | ✅ (optional via `fitv`) |
| Fit process noise `processStd` | ✅ | ✅ |
| Fit AR coefficients `arCoefficients` | ✅ (`fitAr: true`) | ✅ |
| Tie W parameters (`winds`) | ❌ (each W entry independent) | ✅ |
| MAP estimation (priors) | ✅ (`dlmPrior` factory: IG on variances, Normal on AR; or custom `loss` callback) | ⚠️ (priors used in MCMC, not in `fminsearch` MLE) |
| Prior penalty decomposition | ✅ (`deviance` + `priorPenalty` in result) | ❌ |
| Custom cost function | ✅ (`loss` option: custom callback or `dlmPrior` factory) | ✅ (`options.fitfun`) |
| MCMC posterior sampling | ❌ | ✅ (Adaptive Metropolis via `mcmcrun`) |
| State sampling for Gibbs | ❌ | ✅ (disturbance smoother) |
| Posterior uncertainty | ❌ (point estimate only) | ✅ (full chain) |
| Convergence diagnostics | ⚠️ Limited (`devianceHistory`, no posterior chain) | ✅ (`chain`, `sschain` in MCMC mode) |
| Runs in browser | ✅ | ❌ |
| MEX/WASM acceleration | ✅ (`wasm` backend; see [benchmark](#benchmark-same-machine-same-data)) | ✅ (`dlmmex` optional) |
| Irregular timestamps | ✅ (`timestamps` in `dlmFit`; see [irregular timestamps](#irregular-timestamps)) | ❌ (unit spacing only) |

#### What dlm-js does differently

1. **Exact gradients** vs derivative-free simplex — for smooth likelihoods this often improves optimizer guidance, especially as parameter dimension grows (the Kaisaniemi benchmark is one example).
2. **JIT-wrapped optimization step** — forward filter + AD + parameter update are traced together in one optimization step function. JIT overhead currently dominates for small datasets (n=100); the advantage grows with larger n or more complex models.
3. **WASM backend** — runs in Node.js and the browser without native dependencies.
4. **Potentially more robust as dimension grows** — gradient-based optimization can remain practical in settings where derivative-free simplex methods become expensive or unstable.
5. **Joint AR coefficient estimation** — `fitAr: true` jointly estimates observation noise, state variances, and AR coefficients in a single autodiff pass. The AR coefficients enter the G matrix via AD-safe rank-1 updates (`buildG`), keeping the entire optimization `jit()`-compilable.
6. **Built-in MAP estimation** — `dlmPrior` factory provides MATLAB DLM-style Inverse-Gamma priors on variances and Normal priors on AR coefficients (matching `dlmGibbsDIG` conventions), compiled end-to-end in a single `jit()` call. Alternatively, a custom `loss` callback can implement arbitrary penalties. See [MAP estimation](#map-estimation-custom-loss--priors).

#### What MATLAB DLM does that dlm-js doesn't (yet)

1. **MCMC posterior sampling** — full posterior chain with credible intervals. dlm-js supports priors (via MAP), but returns a point estimate rather than a full Bayesian posterior.
2. **Parameter tying** (`winds`) — reduces optimization dimension for structured models.
3. **V factor fitting** (`options.fitv`) — fits a multiplicative factor on V rather than V directly (useful when V is partially known from instrument specification).

## Project structure

```text
├── .github/             # GitHub configuration
├── assets/              # Generated images and timing sidecars
├── dist/                # Compiled and bundled output (after build)
├── docs/                # Generated API documentation (after `pnpm run docs`, gitignored)
├── issues/              # Drafted GitHub issues for upstream jax-js-nonconsuming
├── scripts/             # SVG generators, frame collectors, benchmark runners, timing automation
├── src/                 # Library TypeScript sources
├── tests/               # Test suite (TypeScript tests, JSON fixtures, Octave reference generators)
├── tmp/                 # Scratch / temp directory for agents and debug (gitignored)
├── eslint.config.ts     # ESLint configuration (jax-js-nonconsuming memory rules)
├── LICENSE              # License (does not apply to tests/octave/dlm/)
├── package.json         # Node.js package information
├── README.md            # This readme
├── tsconfig.json        # Configuration file of the TypeScript project
├── typedoc.json         # TypeDoc API documentation configuration
└── vite.config.ts       # Configuration file of the Vite project
```

## Included MATLAB sources (`tests/octave/dlm/`)

The `dlm/` directory contains a curated subset of Marko Laine's [dlm](https://mjlaine.github.io/dlm/dlmtut.html) and [mcmcstat](https://mjlaine.github.io/mcmcstat/) MATLAB toolboxes — just enough to run the Kalman filter and RTS smoother without MCMC or optimization dependencies. Licensing for this included subset is documented in [`tests/octave/dlm/LICENSE.txt`](tests/octave/dlm/LICENSE.txt).

## Development

### Prerequisites

* **Node.js**: [Install Node.js](https://nodejs.org/en/download/) to run JavaScript locally.
* **pnpm**: Install globally via `npm install -g pnpm`.
* **Octave**: Version 10.3.0 is known to work. Install and add `octave-cli` to your system path.

### Install dependencies

```shell
pnpm install
```

### Building and bundling

This project is written in TypeScript. You need to build (compile) it before use:

```shell
pnpm run build
```
This does two things:
  - **Compiles TypeScript (`src/index.ts`) to ESM and CommonJS JavaScript (`dist/dlm-js.es.js`, `dist/dlm-js.cjs.js`) and type definitions (`dist/index.d.ts`).** TypeScript lets you write code with types, but Node.js and browsers only run JavaScript. The build step converts your code to JavaScript.
  - **Bundles the code with Vite for use as a library (outputs ESM and CommonJS formats in `dist/`).** Vite bundles your code so it can be used easily in other projects, in Node.js or browsers, and optimizes it for distribution.

### Testing

#### Generate reference output using Octave

```shell
pnpm run test:octave
```

This generates Octave reference outputs:
- `tests/niledemo-out-m.json` (from `niledemo.m` — pre-existing MATLAB DLM demo)
- `tests/{order0,order2,seasonal,trig,trigar,level,energy,ar2}-out-m.json` (from `gensys_tests.m` — generated for this project)
- `tests/kaisaniemi-out-m.json` (from `kaisaniemi_demo.m` — Kaisaniemi seasonal demo)
- `tests/ozone-out-m.json` (from `ozonedemo.m` — ozone seasonal demo)
- `tests/{gapped,gapped-order0}-out-m.json` (from `gappeddata_test.m` — NaN gapped-data tests)
- `tests/{multivariate,multivariate-order0,multivariate-gapped,multivariate-ar2}-out-m.json` (from `multivariate_demo.m` — p=2 vector observations)

It will also generate test input files unless they already exist.

#### Run tests

You can run all tests directly (no build step needed) with:

```shell
pnpm vitest run
```

or

```shell
pnpm run test:node
```

This runs 14 test suites (318 tests) — `niledemo.test.ts`, `gensys.test.ts`, `synthetic.test.ts`, `mle.test.ts`, `covariate.test.ts`, `ozone.test.ts`, `forecast.test.ts`, `gapped.test.ts`, `assocscan.test.ts`, `timestamps.test.ts`, `sqrtassoc.test.ts`, `multivariate.test.ts`, `ud.test.ts`, and `options.test.ts` — against all available device × dtype combinations. Vitest compiles TypeScript on the fly.

To run the full CI-local check (lint + Octave reference generation + tests):

```shell
pnpm run test
```

#### Synthetic ground-truth tests

In addition to the Octave reference tests above, `synthetic.test.ts` generates state-space data from a **known generating process** with known true hidden states (using a seeded PRNG with Box-Muller transform for reproducible Gaussian noise). The DLM smoother is then tested against mathematical ground truth rather than another implementation's rounding:

- **Finite outputs**: No NaN/Inf in any result field
- **Positive covariance**: Smoothed covariance diagonals `C[k][k][t] > 0` for all states and timesteps
- **Noise reduction**: Smoother RMSE < observation RMSE (the smoother actually reduces noise)
- **Calibrated uncertainty**: True states fall within the 95% posterior credible intervals at roughly the nominal rate

Models tested: local level (m=1) at moderate/high/low SNR, local linear trend (m=2), trigonometric seasonal (m=6), and full seasonal (m=13). All run across the full device × dtype matrix. Float32 is skipped for m > 2 (see [scan algorithm / Float32 stabilization](#scan-algorithm)).

## TODO

* Test the built library (in `dist/`)
* MCMC parameter estimation — depends on Marko Laine's `mcmcrun` toolbox; would require porting or replacing the MCMC engine
* State sampling (disturbance smoother) — blocked on MCMC
* Human review the AI-generated DLM port

## References

1. Särkkä, S. & García-Fernández, Á. F. (2020). [Temporal Parallelization of Bayesian Smoothers](https://arxiv.org/abs/1905.13002). *IEEE Transactions on Automatic Control*, 66(1), 299–306. doi:[10.1109/TAC.2020.2976316](https://doi.org/10.1109/TAC.2020.2976316). — Lemmas 1–2: exact parallel forward Kalman filter (5-tuple elements + associative composition); Lemmas 3–4 + Theorem 2: parallel backward smoother (linear/Gaussian specialization). dlm-js uses Lemmas 1–2 (forward filter) and Lemmas 3–4 (backward smoother).
2. Pyro contributors. [Forecasting II: state space models](https://pyro.ai/examples/forecasting_ii.html). — Parallel-scan Kalman filtering on 78,888-step BART ridership data.
3. NumPyro contributors. [Example: Enumerate Hidden Markov Model](https://num.pyro.ai/en/latest/examples/hmm_enum.html). — Parallel-scan HMM inference using [1].
4. Razavi, H., García-Fernández, Á. F. & Särkkä, S. (2025). Temporal Parallelisation of Continuous-Time MAP Trajectory Estimation. *Preprint*. — Extends [1] to continuous-time MAP estimation with exact associative elements for irregular timesteps; see [irregular timestamps](#irregular-timestamps).
5. Yaghoobi, F., Corenflos, A., Hassan, S. & Särkkä, S. (2021). [Parallel Iterated Extended and Sigma-Point Kalman Smoothers](https://arxiv.org/abs/2102.00514). *Proc. IEEE ICASSP*. [Code](https://github.com/EEA-sensors/parallel-non-linear-gaussian-smoothers) (JAX). — Extends [1] to nonlinear models via parallel EKF, CKF, and iterated variants. Uses the same 5-tuple / 3-tuple associative elements as dlm-js.
6. Yaghoobi, F., Corenflos, A., Hassan, S. & Särkkä, S. (2022). [Parallel Square-Root Statistical Linear Regression for Inference in Nonlinear State Space Models](https://arxiv.org/abs/2207.00426). *Preprint*. [Code](https://github.com/EEA-sensors/sqrt-parallel-smoothers) (JAX). — Reformulates [5] in square-root (Cholesky factor) space: replaces covariance matrices $C$, $J$, $L$ in the associative elements with their Cholesky factors, using QR-based composition that inherently maintains positive semi-definiteness. Includes gradient computation through the parallel scan and implicit differentiation for iterated smoothers. See [square-root parallel smoother](#square-root-parallel-smoother-sqrt-assoc).
7. Bernstein, J. & Newhouse, L. (2024). [Modular Duality in Deep Learning](https://arxiv.org/abs/2410.21265). *arXiv:2410.21265*. — Unifying framework for optimizer design via steepest descent under a norm. The duality map $\mathcal{D}_{\|\cdot\|}$ converts gradients to primal-space updates; for the Euclidean norm this recovers gradient descent, for the operator norm it yields Shampoo/µP, and for the Hessian quadratic form it gives Newton's method. See [Optimizers](#optimizers).

### Authors
* Marko Laine -- Original DLM and mcmcstat sources in `tests/octave/dlm/`, `tests/octave/niledemo.m`, `tests/octave/ozonedemo.m`, and `tests/octave/kaisaniemi.mat`
* Olli Niemitalo (Olli.Niemitalo@hamk.fi) -- Supervision of AI coding agents

### Copyright
* 2013-2017 Marko Laine -- Original DLM and mcmcstat sources in `tests/octave/dlm/`, `tests/octave/niledemo.m`, `tests/octave/ozonedemo.m`, and `tests/octave/kaisaniemi.mat`
* 2026 HAMK Häme University of Applied Sciences
  
### License
This project is MIT licensed (see [`LICENSE`](LICENSE)).

The included original DLM and mcmcstat MATLAB subset in [`tests/octave/dlm/`](tests/octave/dlm/) is covered by its own license text in [`tests/octave/dlm/LICENSE.txt`](tests/octave/dlm/LICENSE.txt).
