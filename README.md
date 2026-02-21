# dlm-js — TypeScript Kalman filter/smoother with autodiff MLE

<strong>
  <a href="https://hamk-uas.github.io/dlm-js/">API Reference</a> |
  <a href="https://github.com/hamk-uas/dlm-js">GitHub</a> |
  <a href="https://mjlaine.github.io/dlm/">Original DLM Docs</a> |
  <a href="https://github.com/mjlaine/dlm">Original DLM GitHub</a>
</strong>

A TypeScript Kalman filter + RTS smoother library using [jax-js-nonconsuming](https://github.com/hamk-uas/jax-js-nonconsuming), inspired by [dynamic linear model](https://mjlaine.github.io/dlm/dlmtut.html) (MATLAB). Extends the original with autodiff-based MLE via `jit(valueAndGrad + Adam)` and an exact O(log N) parallel filter+smoother via `lax.associativeScan` (Särkkä & García-Fernández 2020).

🤖 AI generated code & documentation with gentle human supervision.

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
import { DType } from "@hamk-uas/jax-js-nonconsuming";

// Nile river annual flow data (excerpt)
const y = [1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140];

// Fit a local linear trend model (order=1, state dim m=2)
const result = await dlmFit(y, 120, [40, 10], { order: 1 }, { dtype: DType.Float64 });

console.log(result.yhat);  // smoothed predictions [n]
console.log(result.x);     // smoothed states [m][n]
console.log(result.lik);   // -2·log-likelihood
// Also available: result.xstd [m][n], result.ystd [n], result.v [n],
//   result.resid2 [n], result.mse, result.mape, result.ssy, result.s2, result.nobs
```

For an order=1 model with `options.spline: true`, the W covariance is scaled to produce an integrated random walk (matches MATLAB `dlmfit` spline mode).

### CommonJS (Node.js)

```js
const { dlmFit } = require("dlm-js");
const { DType } = require("@hamk-uas/jax-js-nonconsuming");
```

### Generate system matrices only

```js
import { dlmGenSys } from "dlm-js";

const sys = dlmGenSys({ order: 1, trig: 2, ns: 12 });
console.log(sys.G);  // state transition matrix (m×m)
console.log(sys.F);  // observation vector (1×m)
console.log(sys.m);  // state dimension
```

### MLE parameter estimation

Estimate observation noise `s`, state noise `w`, and optionally AR coefficients by maximizing the Kalman filter log-likelihood via autodiff:

```js
import { dlmMLE } from "dlm-js";
import { DType, defaultDevice } from "@hamk-uas/jax-js-nonconsuming";

defaultDevice("wasm"); // recommended: ~30× faster than "cpu"

const y = [1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140 /* ... */];

// Basic: estimate s and w
const mle = await dlmMLE(
  y,
  { order: 1 },           // model: local linear trend (m=2)
  undefined,              // auto initial guess from data variance
  300,                    // max iterations
  0.05,                   // Adam learning rate
  1e-6,                   // convergence tolerance
  { dtype: DType.Float64 },
);

console.log(mle.s);           // estimated observation noise std dev
console.log(mle.w);           // estimated state noise std devs
console.log(mle.lik);         // -2·log-likelihood at optimum
console.log(mle.iterations);  // iterations to convergence
console.log(mle.elapsed);     // wall-clock ms
console.log(mle.fit);         // full DlmFitResult with optimized parameters

// With AR fitting: estimate s, w, and AR coefficients jointly
const mleAR = await dlmMLE(
  y,
  { order: 0, arphi: [0.5], fitar: true },  // initial arphi + fitar flag
  undefined,
  300, 0.02, 1e-6, { dtype: DType.Float64 },
);
console.log(mleAR.arphi);     // estimated AR coefficients (e.g. [0.81])
```

The entire optimization step is wrapped in a single `jit()` call: `valueAndGrad(loss)` (Kalman filter forward pass + AD backward pass) and optax Adam parameter update. Noise parameters are unconstrained via log-space: `s = exp(θ_s)`, `w[i] = exp(θ_{w,i})`. AR coefficients are optimized directly (unconstrained, not log-transformed — matching MATLAB DLM behavior).

**Two MLE loss paths:** The Kalman filter inside the loss function follows the `run` config: `algorithm: 'scan'` (the default) uses sequential `lax.scan` (O(n) depth); `algorithm: 'assoc'` uses `makeKalmanLossAssoc` with `lax.associativeScan` (O(log n) depth), using the exact 5-tuple forward filter from [1, Lemmas 1–2]. Both paths minimize the same prediction-error likelihood $-2\log L = \sum_t [v_t^2/C_p^{(t)} + \log C_p^{(t)}]$. See [makeKalmanLossAssoc — parallel MLE loss via associative scan](#makekalmanlossassoc--parallel-mle-loss-via-associative-scan) for the full derivation.

**Performance**: on the `wasm` backend, one Nile MLE run (100 observations, m = 2) converges in ~122 iterations (~2.6 s) with the default Adam b2=0.9. The `jit()` compilation happens on the first iteration; subsequent iterations run from compiled code.

For a detailed comparison of dlm-js MLE vs the original MATLAB DLM parameter estimation (Nelder-Mead, MCMC), see [Parameter estimation (maximum likelihood): MATLAB DLM vs dlm-js](#parameter-estimation-maximum-likelihood-matlab-dlm-vs-dlm-js).

### h-step-ahead forecasting

Propagate the last smoothed state h steps forward with no new observations:

```js
import { dlmFit, dlmForecast } from "dlm-js";
import { DType } from "@hamk-uas/jax-js-nonconsuming";

const y = [1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140];

// Fit a local linear trend model
const fit = await dlmFit(y, 120, [40, 10], { order: 1 }, { dtype: DType.Float64 });

// Forecast 12 steps ahead
const fc = await dlmForecast(fit, 120, 12, { dtype: DType.Float64 });

console.log(fc.yhat);  // predicted observation means [h] = F·x_pred
console.log(fc.ystd);  // observation prediction std devs [h] — grows monotonically
console.log(fc.x);     // state trajectories [m][h]
console.log(fc.h);     // 12
console.log(fc.m);     // 2 (state dimension)
```

`fc.yhat` is the full observation prediction `F·x_pred`. For pure trend models (no seasonality) this equals the level state and is appropriate to plot directly. For seasonal or AR models, `yhat` oscillates with the harmonics/AR dynamics in the forecast horizon — if you want a smooth trendline, use the level state `fc.x[0]` directly:

```js
// For seasonal/AR models: plot level state, not yhat
const trend = Array.from(fc.x[0]);        // smooth trend mean
const trendStd = fc.xstd.map(r => r[0]);  // level state std dev
```

With covariates, pass `X_forecast` rows for each forecast step:

```js
// Forecast 3 steps ahead with known future covariate values
const fc = await dlmForecast(fit, 120, 3, { dtype: DType.Float64 }, [
  [solarProxy[n], qbo1[n], qbo2[n]],    // step n+1
  [solarProxy[n+1], qbo1[n+1], qbo2[n+1]], // step n+2
  [solarProxy[n+2], qbo1[n+2], qbo2[n+2]], // step n+3
]);
```

Current behavior for unknown future covariates: if `X_forecast` is omitted (or does not provide a row/entry), dlm-js uses `0` for the missing covariate value in that step. Interpret this as a **baseline conditional forecast** (unknown driver effects set to zero), not a full unconditional forecast.

For a more neutral assumption in practice, center covariates before fitting so that `0` represents a typical/historical-average driver level. Then the default forecast corresponds to “no expected driver anomaly.”

For decision use, prefer scenario forecasting: provide multiple plausible `X_forecast` paths (e.g. low/base/high) and compare resulting forecast bands.

### Missing data (NaN observations)

Place `NaN` in the observation vector `y` wherever a measurement is absent. `dlmFit` automatically skips those timesteps in the Kalman gain and residual calculations (K and v are zeroed), so the smoother interpolates through the gaps without any extra configuration:

```js
import { dlmFit } from "dlm-js";
import { DType } from "@hamk-uas/jax-js-nonconsuming";

// Nile data with a gap in years 30–39 and every 7th observation missing
const y = [1120, 1160, 963, NaN, 1210, 1160, 1160, NaN, 813, /* ... */];

const s2_w = 120;
const s2_v = [40, 10];

const result = await dlmFit(y, s2_w, s2_v, { order: 1 }, { dtype: DType.Float64 });

// nobs: number of non-NaN observations actually used
console.log(result.nobs);   // e.g. 77 when 23 of 100 values are NaN

// yhat, x, xstd, ystd: fully interpolated — finite at every timestep
console.log(result.yhat);   // smoothed observation mean [n] — no NaN
console.log(result.x);      // smoothed state trajectories [m][n] — no NaN
console.log(result.xstd);   // smoothed state std devs [m][n] — no NaN
console.log(result.ystd);   // smoothed observation std devs [n] — no NaN

// v and resid2: NaN at missing positions (consistent with MATLAB dlmsmo)
console.log(result.v);      // innovations [n] — NaN at missing timesteps
console.log(result.resid2); // squared normalised residuals [n] — NaN at missing timesteps

// lik is the log-likelihood summed only over observed timesteps
console.log(result.lik);
```

Missing observations are handled identically to MATLAB's `dlmsmo` (`ig = not(isnan(y(i,:)))` logic): the filter propagates through the gap using only the prior, and the RTS smoother then distributes the information from surrounding observations backward and forward. `ystd` grows wider over the gap, reflecting higher uncertainty where no data was seen.

`dlmMLE` also supports missing data — the Kalman loss scan zeros K, v, and the log-likelihood contribution at NaN timesteps, so autodiff and Adam optimization work correctly through the gaps:

```js
const mle = await dlmMLE(y, { order: 1 }, undefined, 200, 0.05);
// mle.lik is the log-likelihood summed only over observed timesteps
// mle.fit.nobs reports the count of non-NaN observations used
```

## Fit

### Demos

All demos can be regenerated locally with `pnpm run gen:svg`. The `assoc` and `webgpu` variants use an exact O(log N) parallel filter+smoother (Särkkä & García-Fernández 2020) and match the sequential `scan` results to within numerical tolerance (validated by `assocscan.test.ts`).

#### Nile River Flow (Local Linear Trend)

<p align="center">
  <img alt="Nile demo (sequential scan)" src="assets/niledemo-scan.svg" width="100%" />
  <br/><br/>
  <img alt="Nile demo (associative scan)" src="assets/niledemo-assoc.svg" width="100%" />
</p>

*First smoothed state (level) `x[0]` from dlm-js (solid blue) vs MATLAB/Octave dlm (dashed red), with ± 2σ bands from `xstd[:,0]` (state uncertainty, not observation prediction intervals).*

#### Kaisaniemi Monthly Temperatures (Seasonal)

<p align="center">
  <img alt="Kaisaniemi demo (sequential scan)" src="assets/kaisaniemi-scan.svg" width="100%" />
  <br/><br/>
  <img alt="Kaisaniemi demo (associative scan)" src="assets/kaisaniemi-assoc.svg" width="100%" />
</p>

*Top panel: level state `x[0] ± 2σ`. Bottom panel: covariance-aware combined signal `x[0]+x[2] ± 2σ`, using `Var(x0+x2)=Var(x0)+Var(x2)+2Cov(x0,x2)`. Model settings: `order=1`, `trig=1`, `s=2`, `w=[0,0.005,0.4,0.4]`.*

#### Energy Demand (Seasonal + AR)

<p align="center">
  <img alt="Energy demand demo (sequential scan)" src="assets/trigar-scan.svg" width="100%" />
  <br/><br/>
  <img alt="Energy demand demo (associative scan)" src="assets/trigar-assoc.svg" width="100%" />
</p>

*Synthetic 10-year monthly data. Panels top to bottom: smoothed level `x[0] ± 2σ`, trigonometric seasonal `x[2] ± 2σ`, AR(1) state `x[4] ± 2σ`, and covariance-aware combined signal `F·x = x[0]+x[2]+x[4] ± 2σ`. True hidden states (green dashed) are overlaid. Model settings: `order=1`, `trig=1`, `ns=12`, `arphi=[0.85]`, `s=1.5`, `w=[0.3,0.02,0.02,0.02,2.5]`, m=5.*

#### Stratospheric Ozone Trend Analysis

<p align="center">
  <img alt="Stratospheric ozone demo (sequential scan)" src="assets/ozone-demo-scan.svg" width="100%" />
  <br/><br/>
  <img alt="Stratospheric ozone demo (associative scan)" src="assets/ozone-demo-assoc.svg" width="100%" />
</p>

*Top panel: O₃ density (SAGE II / GOMOS observations, 1984–2011) with smoothed level state ± 2σ and a 15-year `dlmForecast` trend extrapolation. Bottom panel: proxy covariate contributions — solar cycle (β̂·X_solar, amber) and QBO (β̂_qbo1·X₁ + β̂_qbo2·X₂, purple). Model: `order=1`, `trig=2`, `ns=12`, 3 static-β covariates, state dimension m=9.*

#### Missing Data (NaN observations)

<p align="center">
  <img alt="Missing-data demo (sequential scan)" src="assets/missing-demo-scan.svg" width="100%" />
  <br/><br/>
  <img alt="Missing-data demo (associative scan)" src="assets/missing-demo-assoc.svg" width="100%" />
</p>

*Nile flow (n=100) with 23 NaN observations. Gray bands mark missing timesteps. Outer light band: observation prediction interval `F·x_smooth ± 2·ystd`; inner opaque band: state uncertainty `x[0] ± 2·xstd[0]`. The smoother interpolates continuously through all gaps with no extra configuration.*

### Backend performance

`dlmFit` warm-run timings (jitted core, second of two sequential runs) and maximum errors vs. the Octave/MATLAB reference (worst case across all 4 models and all outputs: yhat, ystd, x, xstd) for each `DlmRunConfig` combination — backend × dtype × algorithm × stabilization. `assoc + joseph` is an invalid combination and throws. Regenerate with `pnpm run bench:full`. **†** marks the combination used when `run: {}` is passed: f64 → scan + none; f32 → scan + joseph; webgpu/f32 → assoc (no explicit stabilization).

Models: Nile order=0 (n=100, m=1) · Nile order=1 (n=100, m=2) · Kaisaniemi trig (n=117, m=4) · Energy trig+AR (n=120, m=5). Benchmarked on: <!-- computed:static("machine") -->Intel(R) Core(TM) Ultra 5 125H, 62 GB RAM<!-- /computed --> · GPU: <!-- computed:static("gpu") -->GeForce RTX 4070 Ti SUPER (WebGPU adapter)<!-- /computed -->.

| backend | dtype | algorithm | stab | Nile o=0 | Nile o=1 | Kaisaniemi | Energy | max \|Δ\| | max \|Δ\|% |
|---------|-------|-----------|------|----------|----------|------------|--------|----------|------------|
| cpu | f64 | scan | none | **166 ms †** | **358 ms †** | **435 ms †** | **478 ms †** | 3.78e-8 | 1.62e-4 |
| cpu | f64 | scan | joseph | 191 ms | 389 ms | 481 ms | 528 ms | 9.38e-11 | 3.56e-9 |
| cpu | f64 | assoc | none | 74 ms | 201 ms | 853 ms | 1534 ms | 5.12e-9 | 2.17e-5 |
| cpu | f32 | scan | none | 171 ms | 345 ms | 439 ms | 482 ms | ⚠️ 180 | ⚠️ 1.4e6 |
| cpu | f32 | scan | joseph | **184 ms †** | **384 ms †** | **484 ms †** | **533 ms †** | 1.32e-2 | 0.17 |
| cpu | f32 | assoc | none | 67 ms | 204 ms | 869 ms | 1552 ms | 4.93e-3 | 19.7 |
| wasm | f64 | scan | none | **16 ms †** | **20 ms †** | **22 ms †** | **23 ms †** | 3.78e-8 | 1.62e-4 |
| wasm | f64 | scan | joseph | 18 ms | 22 ms | 22 ms | 22 ms | 9.38e-11 | 3.56e-9 |
| wasm | f64 | assoc | none | 24 ms | 25 ms | 32 ms | 39 ms | 5.12e-9 | 2.17e-5 |
| wasm | f32 | scan | none | 15 ms | 20 ms | 21 ms | 19 ms | ⚠️ 7000 | ⚠️ 2e6 |
| wasm | f32 | scan | joseph | **17 ms †** | **20 ms †** | **24 ms †** | **21 ms †** | 3.99e-2 | 1.37 |
| wasm | f32 | assoc | none | 23 ms | 24 ms | 33 ms | 36 ms | 4.93e-3 | 21.9 |
| webgpu | f32 | scan | none | 549 ms | 913 ms | 1011 ms | 1141 ms | ⚠️ 110 | ⚠️ 6.7e4 |
| webgpu | f32 | scan | joseph | 712 ms | 888 ms | 1041 ms | 1169 ms | 2.49e-2 | 1.32 |
| webgpu | f32 | assoc | none | **325 ms †** | **353 ms †** | **356 ms †** | **372 ms †** | 4.93e-3 | 19.8 |

⚠️ = numerically unstable: f32 + scan + none without Joseph-form stabilization blows up for larger state dimensions (m ≥ 4). Both columns show worst case across all 4 benchmark models and all output variables (yhat, ystd, x, xstd). `max |Δ|%` uses the Octave reference value as denominator; percentages >1% in the `assoc` rows come from small xstd values (not from yhat/ystd).

**Key findings:**
- **WASM is ~10–20× faster than CPU** — the JS interpreter backend has significant overhead for small matrix operations.
- **`assoc` on CPU is faster for small m, slower for large m** — for m=1–2, the scan composition is cheap and reduces interpreter overhead; for m=4–5 the extra matrix operations dominate (~2× slower than `scan` on CPU).
- **`assoc` on WASM has no warm-run advantage over `scan`** — warm times are nearly identical (~20–40 ms) for all models; the first-run cost is ~5× higher due to extra JIT compilation paths, so prefer `scan` on WASM unless you need the parallel path explicitly.
- **`assoc + joseph` is an error** — `stabilization: 'joseph'` combined with an explicit `algorithm: 'assoc'` throws at runtime. The assoc path always applies its own numerically stable formulation; use `assoc` without setting `stabilization`.
- **f32 + scan + none is dangerous for large models** — covariance catastrophically cancels for m ≥ 4; `joseph` form (or `assoc`) is required for float32 stability. The `assoc` path is stable with float32 even without joseph, as shown by the reasonable 4.93e-3 max error vs the ⚠️ 7000 for scan+none.
- **Joseph form overhead is negligible on WASM** — f32+joseph vs f64+none differ by <5 ms across all models, well within JIT variance. The stabilization choice is numerically important but not a performance concern.
- **WebGPU `assoc` is ~4× faster than WebGPU `scan`** for larger models (m=4–5) — sequential scan on WebGPU dispatches O(n) kernels (no GPU parallelism); `assoc` uses O(log n) dispatches (Kogge-Stone), cutting ms from ~1800 to ~450 for Energy.
- **WebGPU `scan` is the worst option** — 1800 ms warm for Energy (m=5) vs 29 ms on WASM; every filter step is a separate GPU dispatch with no cross-workgroup sync.
- **WASM stays flat up to N≈3200 (fixed overhead), then scales linearly** — asymptotic per-step cost ~1.6 µs/step, giving ~<!-- timing:scale:wasm-f64:n102400 -->156 ms<!-- /timing --> at N=102400. WebGPU/f32 `assoc` scales **sub-linearly (O(log n))**: a 1024× increase from N=100 to N=102400 only doubles the runtime (<!-- timing:scale:webgpu-f32:n100 -->305 ms<!-- /timing --> → <!-- timing:scale:webgpu-f32:n102400 -->648 ms<!-- /timing -->). A crossover is plausible at N≈800k–1M.
- **WebGPU results may differ slightly** from sequential WASM/f64 due to Float32 precision and operation reordering in the parallel scan, not from any algorithmic approximation — both paths use exact per-timestep Kalman gains.

For background on the Nile and Kaisaniemi demos and the original model formulation, see [Marko Laine's DLM page](https://mjlaine.github.io/dlm/). The energy demand demo uses synthetic data generated for this project. The missing-data demo uses the same Nile dataset with 23 observations removed.

### Numerical precision

Since jax-js-nonconsuming v0.2.1, Float64 dot product reductions use Kahan compensated summation, reducing per-dot rounding from O(m·ε) to O(ε²). This improved the seasonal model (m=13) from ~3e-5 to ~1.8e-5 worst-case relative error.

### scan algorithm

`algorithm: 'scan'` uses sequential `lax.scan` for both the Kalman forward filter and RTS backward smoother. It is the default when `algorithm` is not set in `DlmRunConfig` and the `assoc` path is not auto-selected.

The dominant error source is **not** summation accuracy — it is catastrophic cancellation in the RTS backward smoother step `C_smooth = C - C·N·C`. When the smoothing correction nearly equals the prior covariance, the subtraction amplifies any rounding in the operands. Kahan summation cannot fix this because it only improves the individual dot products, not the outer subtraction. See detailed comments in `src/index.ts`.

**Float32 stabilization (Joseph form):** When `dtype: DType.Float32`, the scan path defaults to `stabilization: 'joseph'`, replacing the standard covariance update `C_filt = C_pred - K·F·C_pred` with:

$$C_{\text{filt}} = (I - K F) \, C_{\text{pred}} \, (I - K F)^\top + K \, V^2 \, K^\top$$

This is algebraically equivalent but numerically more stable — it guarantees a positive semi-definite result even with rounding. Combined with explicit symmetrization (`(C + C') / 2`), this prevents the covariance from going non-positive-definite for m ≤ 2. Without Joseph form (`stabilization: 'none'`), Float32 + scan is numerically unstable for m ≥ 4 (see ⚠️ entries in the benchmark table). Float32 is still skipped in tests for m > 2 even with Joseph form, due to accumulated rounding in the smoother.

### assoc algorithm

`algorithm: 'assoc'` uses `lax.associativeScan` to evaluate the **exact O(log N) parallel Kalman filter + smoother** from Särkkä & García-Fernández (2020) [1], Lemmas 1–6. Pass `{ algorithm: 'assoc' }` in `DlmRunConfig` to use it on any backend and any dtype. Combining `algorithm: 'assoc'` with `stabilization: 'joseph'` throws an error — the assoc path always applies its own numerically stable formulation.

Both passes dispatch ⌈log₂N⌉+1 kernel rounds (Kogge-Stone), giving O(log n) total depth. Results are numerically equivalent to `scan` to within floating-point reordering (validated by `assocscan.test.ts`).

- **Forward filter** (exact 5-tuple from [1, Lemmas 1–2]): Constructs per-timestep 5-tuple elements $(A_k, b_k, C_k, \eta_k, J_k)$ with exact Kalman gains per Lemma 1, composed via `lax.associativeScan` using Lemma 2 with regularized inverse and push-through identity. No approximation — produces the same filtered states as sequential `scan`, up to floating-point reordering.
- **Backward smoother** ([1], Lemmas 5–6 + Theorem 2): Exact per-timestep smoother gains $E_k = C_{filt,k} G^\top (G C_{filt,k} G^\top + W)^{-1}$ computed from the forward-filtered covariances via batched `np.linalg.inv`. Smoother elements $(E_k, g_k, L_k)$ with Joseph form $L_k$ composed via `lax.associativeScan(compose, elems, { reverse: true })` (suffix scan). No accuracy loss — the backward smoother is algebraically equivalent to sequential RTS.

Combined with a WebGPU backend, this provides two orthogonal dimensions of parallelism: across time steps (O(log n) depth via `associativeScan`) and within each step's matrix operations (GPU ALUs). The same technique is used in production by Pyro's `GaussianHMM` [2] and NumPyro's parallel HMM inference [3].

#### Impact on MLE

`dlmMLE` in `src/mle.ts` dispatches between two loss functions based on device and dtype:

- **CPU/WASM (any dtype):** `makeKalmanLoss` — sequential `lax.scan` forward filter (O(n) depth per iteration). For the energy demo (n=120, <!-- timing:energy-mle:iterations -->300<!-- /timing --> iters, ~<!-- timing:energy-mle:elapsed -->6.5 s<!-- /timing --> on WASM).
- **WebGPU + Float32:** `makeKalmanLossAssoc` — `lax.associativeScan` forward filter (O(log n) depth per iteration). Details below.

Both paths are wrapped in `jit(valueAndGrad(lossFn))` with optax Adam. The final refit after convergence calls `dlmFit` (which itself uses the parallel path on WebGPU).

#### `makeKalmanLossAssoc` — parallel MLE loss via associative scan

The parallel MLE loss function replaces the sequential Kalman forward pass inside `valueAndGrad` with the exact 5-tuple forward filter from [1, Lemmas 1–2]. Each timestep produces per-step Kalman gains directly. Gradients propagate through $\theta \to (W, V^2) \to$ scan elements $\to$ loss naturally because the element construction uses standard differentiable ops.

**Step-by-step derivation:**

1. **Parameter extraction (traced):** $\theta \xrightarrow{\exp} (s, w_0 \ldots w_{m-1}, \phi_1 \ldots \phi_p)$. Observation variance $V^2 = s^2$ (scalar); state noise $W = \text{diag}(w_i^2)$; $G$ updated with AR coefficients if `fitar: true`.

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

#### Design choices

| Aspect | Choice | Rationale |
|--------|--------|----------|
| Exact 5-tuple elements | Per-timestep $(A, b, C, \eta, J)$ from Lemma 1 | Each timestep has its own Kalman gain — exact, no approximation. |
| Regularized inverse in compose | $(I + C_i J_j + \epsilon I)^{-1}$ | Guards against near-singular matrices at degenerate (NaN/zero-J) compose steps. $\epsilon = 10^{-6}$ (Float32) or $10^{-12}$ (Float64). |
| Push-through identity | $N = I - J_j M C_i$ | Derives the second inverse from the first — only one `np.linalg.inv` call per compose step. |
| Float mask blending | $A = \text{mask} \cdot A_{\text{obs}} + (1 - \text{mask}) \cdot G$ | Avoids boolean-conditioned `np.where` which can create discontinuous gradients in some AD frameworks. Arithmetic blending is smooth and AD-safe. |
| Scan output disposal | Individual `scanned.*.dispose()` after `x_filt` and `C_filt` recovery | Safe because `np.add` produces new arrays — `x_filt` and `C_filt` are independent of the scan pytree. |

#### WebGPU vs WASM benchmark

`dlmFit` warm-run timings (jitted core, second of two runs):

| Model | $n$ | $m$ | wasm / f64 (scan) | webgpu / f32 (assocScan) |
|-------|-----|-----|-------------------|--------------------------|
| Nile, order=0 | 100 | 1 | <!-- timing:bb:nile-o0:wasm-f64 -->19 ms<!-- /timing --> | <!-- timing:bb:nile-o0:webgpu-f32 -->300 ms<!-- /timing --> |
| Nile, order=1 | 100 | 2 | <!-- timing:bb:nile-o1:wasm-f64 -->20 ms<!-- /timing --> | <!-- timing:bb:nile-o1:webgpu-f32 -->299 ms<!-- /timing --> |
| Kaisaniemi, trig | 117 | 4 | <!-- timing:bb:kaisaniemi:wasm-f64 -->20 ms<!-- /timing --> | <!-- timing:bb:kaisaniemi:webgpu-f32 -->339 ms<!-- /timing --> |
| Energy, trig+AR | 120 | 5 | <!-- timing:bb:trigar:wasm-f64 -->20 ms<!-- /timing --> | <!-- timing:bb:trigar:webgpu-f32 -->356 ms<!-- /timing --> |

**WebGPU scaling: O(log n) with high fixed overhead.**

A scaling benchmark (Nile order=1, m=2) measured `dlmFit` warm-run timings at exponentially increasing N (WASM: 2 warmup + 4 timed runs median; WebGPU: same). Both forward filter and backward smoother use `lax.associativeScan` on the WebGPU path:

| N | wasm/f64 | webgpu/f32 | ratio |
|---|--------------|-----------------|-------|
| 100 | <!-- timing:scale:wasm-f64:n100 -->24 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n100 -->305 ms<!-- /timing --> | 27× |
| 200 | <!-- timing:scale:wasm-f64:n200 -->23 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n200 -->328 ms<!-- /timing --> | 29× |
| 400 | <!-- timing:scale:wasm-f64:n400 -->21 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n400 -->352 ms<!-- /timing --> | 30× |
| 800 | <!-- timing:scale:wasm-f64:n800 -->20 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n800 -->376 ms<!-- /timing --> | 30× |
| 1600 | <!-- timing:scale:wasm-f64:n1600 -->21 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n1600 -->388 ms<!-- /timing --> | 31× |
| 3200 | <!-- timing:scale:wasm-f64:n3200 -->23 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n3200 -->409 ms<!-- /timing --> | 36× |
| 6400 | <!-- timing:scale:wasm-f64:n6400 -->29 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n6400 -->439 ms<!-- /timing --> | 33× |
| 12800 | <!-- timing:scale:wasm-f64:n12800 -->34 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n12800 -->460 ms<!-- /timing --> | 27× |
| 25600 | <!-- timing:scale:wasm-f64:n25600 -->54 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n25600 -->490 ms<!-- /timing --> | 19× |
| 51200 | <!-- timing:scale:wasm-f64:n51200 -->84 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n51200 -->511 ms<!-- /timing --> | 13× |
| 102400 | <!-- timing:scale:wasm-f64:n102400 -->156 ms<!-- /timing --> | <!-- timing:scale:webgpu-f32:n102400 -->648 ms<!-- /timing --> | 7× |

Three findings:

1. **WASM stays flat up to N≈3200**, then grows roughly linearly (O(n)). The per-step cost asymptotes around ~1.4 µs/step at N=102400 (~<!-- timing:scale:wasm-f64:n102400 -->156 ms<!-- /timing --> total). The flat region reflects fixed JIT/dispatch overhead, not compute.

2. **WebGPU scales sub-linearly (O(log n))** — both forward and backward passes use `lax.associativeScan`, so each dispatches ⌈log₂N⌉+1 Kogge-Stone rounds. A 1024× increase from N=100 to N=102400 only doubles the runtime (<!-- timing:scale:webgpu-f32:n100 -->305 ms<!-- /timing --> → <!-- timing:scale:webgpu-f32:n102400 -->648 ms<!-- /timing -->). However, the fixed per-dispatch overhead of WebGPU command submission is high (~500 ms base), so the constant factor dominates at practical series lengths.

3. **The WASM-to-WebGPU ratio converges as N grows**: ~27× at N=100, ~7× at N=102400. WASM is faster at all measured N, but the gap shrinks with series length because WASM's O(n) growth outpaces WebGPU's O(log n) growth. A crossover is plausible at N≈800k–1M where WASM's linear growth would overtake WebGPU's logarithmic growth.


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

*Parameter estimation via autodiff (`dlmMLE`). Orange dashed = initial variance-based guess, blue solid = MLE optimum. The entire optimization step is wrapped in a single `jit()` call. Estimated observation noise s = 121.1 (known: 122.9), -2·log-likelihood = 1105.0. WebGPU is slower than WASM at n=100 due to dispatch overhead, but pays off at large N.*

#### Energy MLE Optimization (with AR estimation)

<p align="center">
  <img alt="Energy MLE optimization (sequential scan)" src="assets/energy-mle-anim-scan.svg" width="100%" />
  <br/><br/>
  <img alt="Energy MLE optimization (associative scan)" src="assets/energy-mle-anim-assoc.svg" width="100%" />
  <br/><br/>
  <img alt="Energy MLE optimization (WebGPU)" src="assets/energy-mle-anim-webgpu.svg" width="100%" />
</p>

*Joint estimation of observation noise s, state variances w, and AR(1) coefficient φ via autodiff (`dlmMLE` with `fitar: true`). Shows the combined signal F·x ± 2σ converging. Two sparklines track convergence: −2·log-likelihood (amber) and AR coefficient φ (green, 0.50 → 0.68, true: 0.85).*

The Nile MLE demo estimates `s` and `w` on the classic Nile dataset; the energy MLE demo jointly estimates `s`, `w`, and AR coefficient `φ` on the synthetic energy model (`fitar: true`). See [Parameter estimation (maximum likelihood): MATLAB DLM vs dlm-js](#parameter-estimation-maximum-likelihood-matlab-dlm-vs-dlm-js) for details.

### Parameter estimation (maximum likelihood): MATLAB DLM vs dlm-js

This comparison focuses on the univariate MLE workflow ($p=1$). For the original MATLAB DLM, see the [tutorial](https://mjlaine.github.io/dlm/dlmtut.html) and [source](https://github.com/mjlaine/dlm).

#### Objective function

Both optimize the same scalar likelihood form (for $p=1$ observations) — $-2 \log L$ from the Kalman filter prediction error decomposition:

$$-2 \log L = \sum_{t=1}^{n} \left[ \frac{v_t^2}{C_p^{(t)}} + \log C_p^{(t)} \right]$$

where $v_t = y_t - F x_{t|t-1}$ is the innovation and $C_p^{(t)} = F C_{t|t-1} F' + V^2$ is the innovation covariance. The `dlm_costfun` function (inside `dlmfit.m`) calls `dlmsmo(...,0,0)` (filter only, no smoother, no sample) and returns `out.lik`; we ran this under Octave. The dlm-js `makeKalmanLoss` in `src/mle.ts` computes the same per-step terms via `lax.scan` over the forward filter steps.

In practice, exact numeric equality is not expected because initialization and optimization procedures differ (e.g., `dlmfit` uses a two-pass prefit for initial state/covariance before optional optimization, as run under Octave).

#### Parameterization

| Aspect | MATLAB DLM | dlm-js |
|--------|-----------|--------|
| Observation noise $s$ | Optionally fitted as a multiplicative factor $V \cdot e^{\theta_v}$ (controlled by `options.fitv`) | Always fitted: $s = e^{\theta_s}$ |
| State noise $w$ | $W_{ii} = (e^{\theta_{w,i}})^2$ | $W_{ii} = (e^{\theta_{w,i}})^2$ via `buildDiagW` |
| AR coefficients | Directly optimized (not log-transformed): $G(\text{arInds}) = \theta_\phi$ | Directly optimized (not log-transformed): $G(\text{arInds}) = \theta_\phi$ via `buildG` rank-1 update (AD-safe) |
| Parameter grouping | `options.winds` maps $\text{diag}(W)$ entries to shared parameters (e.g., `winds=[1,1,2,2]` ties states 1&2 and 3&4) | Each $W_{ii}$ is an independent parameter |

Both use the same positivity enforcement: log-space for variance parameters, then $e^{(\cdot)}$ to map back. The MATLAB version has an extra feature — `winds` — that lets you **tie** $\text{diag}(W)$ entries to shared parameters, reducing the optimization dimension when multiple states should share the same noise variance.

#### Optimizer

| Aspect | MATLAB DLM | dlm-js |
|--------|-----------|--------|
| **Algorithm** | `fminsearch` (Nelder-Mead simplex) | Adam (gradient-based, 1st-order momentum) |
| **Gradient computation** | **None** — derivative-free | **Autodiff** via `valueAndGrad()` + reverse-mode AD through `lax.scan` |
| **Convergence** | Simplex shrinkage heuristic (no guaranteed rate for non-convex objectives) | Adaptive first-order method with bias-corrected moments; practical convergence depends on learning rate and objective conditioning |
| **Cost per optimizer step** | Multiple likelihood evaluations per simplex update (depends on dimension and simplex operations) | One `valueAndGrad` evaluation (forward + reverse AD through the loss) plus Adam state update |
| **Typical run budget** | 400 function evaluations (`options.maxfuneval` default) | 200 optimizer iterations (`maxIter` default) |
| **Compilation** | None (interpreted; tested under Octave, or optional `dlmmex` C MEX) | Optimization step is wrapped in a single `jit()`-traced function (forward filter + AD + Adam update) |
| **Jittability** | N/A | Fully jittable — optax Adam (as of v0.4.0, `count.item()` fix) |
| **Adam defaults** | N/A | `b1=0.9, b2=0.9, eps=1e-8` — b2=0.9 converges ~3× faster than canonical 0.999 on DLM likelihoods (measured across Nile, Kaisaniemi, ozone benchmarks) |
| **WASM performance** | N/A | ~<!-- timing:ckpt:nile:false-s -->1.7 s<!-- /timing --> for 60 iterations (Nile, n=100, m=2, b2=0.9, `checkpoint: false`); see [checkpointing note](#gradient-checkpointing-always-use-checkpoint-false) |

**Key tradeoff**: Nelder-Mead needs only function evaluations (no gradients), making it simple to apply and often robust on noisy/non-smooth surfaces. But cost grows quickly with parameter dimension because simplex updates require repeated objective evaluations. Adam with autodiff has higher per-step compute cost, but uses gradient information and often needs fewer optimization steps on smooth likelihoods like DLM filtering objectives.

##### MLE vs MCMC: different objectives

Pure MLE minimises $-2 \log L$ without any prior on $W$. On real data such as satellite ozone measurements, this can produce degenerate solutions — e.g. most seasonal noise variances collapse to near-zero while one or two grow large — because the likelihood surface has a wide, nearly flat ridge. MATLAB MCMC uses a normal prior on $\log W$ entries that keeps them symmetric and away from zero, yielding a posterior mean at much higher $-2\log L$ but visually smoother, better-regularised results.

| Point | MATLAB MCMC | dlm-js MLE |
|-------|------------|------------|
| Ozone $-2\log L$ at MATLAB posterior W | 435.6 | — |
| Ozone $-2\log L$ at MLE optimum | — | 203.8 |
| Ozone trend shape | Smooth, symmetric seasonal noise | Same global trend, but seasonal W values degenerate |

If MCMC-like regularisation is needed, the recommended approach is MAP estimation: add a log-normal penalty on $W$ entries to the loss before differentiating. dlm-js `makeKalmanLoss` is a plain differentiable function and the penalty can be added outside of it before wrapping in `jit(valueAndGrad(...))`.

#### Benchmark: same machine, same data

All timings measured on the same machine. The MATLAB DLM toolbox was run under Octave with `fminsearch` (Nelder-Mead, `maxfuneval=400` for Nile models, `maxfuneval=800` for Kaisaniemi). dlm-js uses `dlmMLE` (Adam + autodiff, `maxIter=300`, `b2=0.9` default, `checkpoint: false`, `wasm` backend). Octave timings are median of 5 runs after 1 warmup; dlm-js timings are single fresh-run wall-clock times (including first-call JIT overhead).

| Model | $n$ | $m$ | params | Octave `fminsearch` | dlm-js `dlmMLE` (wasm) | $-2\log L$ (Octave) | $-2\log L$ (dlm-js) |
|-------|---|---|--------|---------------------|------------------------|-----------------|-----------------|
| Nile, order=1, fit s+w | 100 | 2 | 3 | 2827 ms | <!-- timing:nile-mle:elapsed -->2810 ms<!-- /timing --> | 1104.6 | <!-- timing:mle-bench:nile-order1:lik -->1104.9<!-- /timing --> |
| Nile, order=1, fit w only | 100 | 2 | 2 | 1623 ms | — | 1104.7 | — |
| Nile, order=0, fit s+w | 100 | 1 | 2 | 610 ms | <!-- timing:mle-bench:nile-order0:elapsed -->1804 ms<!-- /timing --> | 1095.8 | <!-- timing:mle-bench:nile-order0:lik -->1095.8<!-- /timing --> |
| Kaisaniemi, trig, fit s+w | 117 | 4 | 5 | **failed** (NaN/Inf) | <!-- timing:mle-bench:kaisaniemi:elapsed -->5785 ms<!-- /timing --> | — | <!-- timing:mle-bench:kaisaniemi:lik -->341.3<!-- /timing --> |
| Energy, trig+AR, fit s+w+φ | 120 | 5 | 7 | — | <!-- timing:energy-mle:elapsed-ms -->6462 ms<!-- /timing --> | — | <!-- timing:energy-mle:lik -->443.1<!-- /timing --> |

Octave timings are from Octave with `fminsearch`; dlm-js timings are single fresh-run wall-clock times (including JIT overhead) from `pnpm run bench:mle`.

**Key observations:**
- **Nile (n=100, m=2):** Octave `fminsearch` is <!-- computed:static("octave-nile-order1-elapsed-ms") < slot("nile-mle:elapsed") ? "faster" : "slower" -->slower<!-- /computed --> (see table). dlm-js includes one-time JIT compilation overhead in the reported time.
- **Likelihood values:** Both converge to very similar $-2\log L$ values on Nile (difference ~<!-- computed:Math.abs(slot("mle-bench:nile-order1:lik") - static("octave-nile-order1-lik")).toFixed(1) -->0.3<!-- /computed -->).
- **Kaisaniemi (m=4, 5 params):** Octave `fminsearch` (`maxfuneval=800`) failed with NaN/Inf; dlm-js converged in <!-- timing:mle-bench:kaisaniemi:iterations -->300<!-- /timing --> iterations (~<!-- timing:mle-bench:kaisaniemi:elapsed-s -->5.8 s<!-- /timing -->), reaching $-2\log L =$ <!-- timing:mle-bench:kaisaniemi:lik -->341.3<!-- /timing -->.
- **Joint $s+w$ fitting:** dlm-js always fits both $s$ and $w$; MATLAB DLM can fit $w$ only (`fitv=0`).

##### Gradient checkpointing

`lax.scan` supports gradient checkpointing via a `checkpoint` option: `true` (default, √N segments), `false` (store all carries), or an explicit segment size.

**Benchmark (WASM, Float64, 60 iterations):**

| Dataset | n | m | `checkpoint: false` | `checkpoint: true` (√N) | speedup |
|---------|---|---|--------------------|-----------------------|---------|
| Nile, order=1 | 100 | 2 | <!-- timing:ckpt:nile:false-ms -->1702 ms<!-- /timing --> | <!-- timing:ckpt:nile:true-ms -->230 ms<!-- /timing --> | <!-- timing:ckpt:nile:speedup -->-86%<!-- /timing --> |
| Energy, order=1+trig1+ar1 | 120 | 5 | <!-- timing:ckpt:energy:false-ms -->2136 ms<!-- /timing --> | <!-- timing:ckpt:energy:true-ms -->943 ms<!-- /timing --> | <!-- timing:ckpt:energy:speedup -->-56%<!-- /timing --> |


#### MCMC (MATLAB DLM only)

The MATLAB DLM toolbox supports MCMC via Adaptive Metropolis (`mcmcrun`): 5000 simulations, log-normal priors, full posterior chain with credible intervals, and disturbance smoother for Gibbs-style state sampling.

**dlm-js has no MCMC equivalent** — `dlmMLE` returns a point estimate only. Possible future directions:
- Hessian at the MLE optimum for approximate confidence intervals
- Stochastic gradient MCMC (e.g., SGLD) using the existing AD infrastructure

#### Feature comparison summary

| Capability | MATLAB DLM | dlm-js `dlmMLE` |
|-----------|-----------|-----------------|
| MLE point estimate | ✅ `fminsearch` | ✅ Adam + autodiff |
| Gradient-based optimization | ❌ | ✅ |
| JIT compilation of optimizer | ❌ | ✅ |
| Fit observation noise `s` | ✅ (optional via `fitv`) | ✅ (always) |
| Fit state noise `w` | ✅ | ✅ |
| Fit AR coefficients `arphi` | ✅ | ✅ (`fitar: true`) |
| Tie W parameters (`winds`) | ✅ | ❌ (each W entry independent) |
| Custom cost function | ✅ (`options.fitfun`) | ❌ |
| MCMC posterior sampling | ✅ (Adaptive Metropolis via `mcmcrun`) | ❌ |
| State sampling for Gibbs | ✅ (disturbance smoother) | ❌ |
| Posterior uncertainty | ✅ (full chain) | ❌ (point estimate only) |
| Convergence diagnostics | ✅ (`chain`, `sschain` in MCMC mode) | ⚠️ Limited (`likHistory`, no posterior chain) |
| Runs in browser | ❌ | ✅ |
| MEX/WASM acceleration | ✅ (`dlmmex` optional) | ✅ (`wasm` backend; see [benchmark](#benchmark-same-machine-same-data)) |

#### What dlm-js does differently

1. **Exact gradients** vs derivative-free simplex — for smooth likelihoods this often improves optimizer guidance, especially as parameter dimension grows (the Kaisaniemi benchmark is one example).
2. **JIT-wrapped optimization step** — forward filter + AD + parameter update are traced together in one optimization step function. JIT overhead currently dominates for small datasets (n=100); the advantage grows with larger n or more complex models.
3. **WASM backend** — runs in Node.js and the browser without native dependencies.
4. **Potentially more robust as dimension grows** — gradient-based optimization can remain practical in settings where derivative-free simplex methods become expensive or unstable.
5. **Joint AR coefficient estimation** — `fitar: true` jointly estimates observation noise, state variances, and AR coefficients in a single autodiff pass. The AR coefficients enter the G matrix via AD-safe rank-1 updates (`buildG`), keeping the entire optimization `jit()`-compilable.

#### What MATLAB DLM does that dlm-js doesn't (yet)

1. **MCMC posterior sampling** — full Bayesian uncertainty quantification with priors.
2. **Parameter tying** (`winds`) — reduces optimization dimension for structured models.
3. **Custom fit functions** (`options.fitfun`) — user-supplied cost functions.
4. **V factor fitting** (`options.fitv`) — fits a multiplicative factor on V rather than V directly (useful when V is partially known from instrument specification).

### Project structure

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

### Included MATLAB sources (`tests/octave/dlm/`)

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

This runs `niledemo.test.ts`, `gensys.test.ts`, `synthetic.test.ts`, `mle.test.ts`, `covariate.test.ts`, and `ozone.test.ts` against all available device × dtype combinations. Vitest compiles TypeScript on the fly.

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
* Multivariate observations (p > 1) — biggest remaining gap; affects all matrix dimensions throughout the filter/smoother (dlm-js currently assumes scalar observations, p = 1)
* MCMC parameter estimation — depends on Marko Laine's `mcmcrun` toolbox; would require porting or replacing the MCMC engine
* State sampling (disturbance smoother) — blocked on MCMC
* Human review the AI-generated DLM port

## References

1. Särkkä, S. & García-Fernández, Á. F. (2020). [Temporal Parallelization of Bayesian Smoothers](https://arxiv.org/abs/1905.13002). *IEEE Transactions on Automatic Control*, 66(1), 299–306. doi:[10.1109/TAC.2020.2976316](https://doi.org/10.1109/TAC.2020.2976316). — Lemmas 1–2: exact parallel forward Kalman filter (5-tuple elements + associative composition); Lemmas 5–6 + Theorem 2: parallel backward smoother. dlm-js uses Lemmas 1–2 (forward) and Lemmas 5–6 (backward).
2. Pyro contributors. [Forecasting II: state space models](https://pyro.ai/examples/forecasting_ii.html). — Parallel-scan Kalman filtering on 78,888-step BART ridership data.
3. NumPyro contributors. [Example: Enumerate Hidden Markov Model](https://num.pyro.ai/en/latest/examples/hmm_enum.html). — Parallel-scan HMM inference using [1].
4. Razavi, H., García-Fernández, Á. F. & Särkkä, S. (2025). Temporal Parallelisation of Continuous-Time MAP Trajectory Estimation. *Preprint*. — Extends the framework to continuous-time MAP estimation.

### Authors
* Marko Laine -- Original DLM and mcmcstat sources in `tests/octave/dlm/` and `tests/octave/niledemo.m`
* Olli Niemitalo (Olli.Niemitalo@hamk.fi) -- Supervision of AI coding agents

### Copyright
* 2013-2017 Marko Laine -- Original DLM and mcmcstat sources in `tests/octave/dlm/` and `tests/octave/niledemo.m`
* 2026 HAMK Häme University of Applied Sciences
  
### License
This project is MIT licensed (see [`LICENSE`](LICENSE)).

The included original DLM and mcmcstat MATLAB subset in [`tests/octave/dlm/`](tests/octave/dlm/) is covered by its own license text in [`tests/octave/dlm/LICENSE.txt`](tests/octave/dlm/LICENSE.txt).
