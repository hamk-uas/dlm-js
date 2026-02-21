# Migration Guide — dlm-js API overhaul

This guide covers the breaking changes introduced in the API overhaul. All changes are mechanical renames or signature restructuring — no numeric behavior has changed.

## Quick migration strategy

1. **If you're comparing against MATLAB DLM**: wrap results with `toMatlab(result)` and `toMatlabMle(mleResult)` — these return objects with the original MATLAB field names.
2. **If you're using the JS API directly**: follow the rename tables below.

## Signature changes

### `dlmFit`

```ts
// Before
import { DType } from "@hamk-uas/jax-js-nonconsuming";
const result = await dlmFit(y, s, w, { order: 1, trig: 1, ns: 12 }, { dtype: DType.Float64 });

// After
const result = await dlmFit(y, {
  obsStd: s,
  processStd: w,
  order: 1,
  harmonics: 1,
  seasonLength: 12,
  dtype: 'f64',
});
```

### `dlmMLE`

```ts
// Before
const mle = await dlmMLE(y, { order: 1 }, undefined, 300, 0.05, 1e-6, { dtype: DType.Float64 });

// After
const mle = await dlmMLE(y, { order: 1, maxIter: 300, lr: 0.05, tol: 1e-6, dtype: 'f64' });
```

### `dlmForecast`

```ts
// Before
const fc = await dlmForecast(fit, s, h, { dtype: DType.Float64 }, X_forecast);

// After
const fc = await dlmForecast(fit, obsStd, h, { dtype: 'f64', X: X_forecast });
```

### `dlmGenSys`

```ts
// Before
const sys = dlmGenSys({ order: 1, trig: 2, ns: 12, arphi: [0.7], fullseas: true });

// After
const sys = dlmGenSys({ order: 1, harmonics: 2, seasonLength: 12, arCoefficients: [0.7], fullSeasonal: true });
```

## Field rename tables

### `DlmFitResult`

| Old name | New name | Access pattern |
|----------|----------|----------------|
| `result.x` | `result.smoothed` | `StateMatrix`: `.series(i)`, `.get(t,i)`, `.at(t)` |
| `result.xf` | `result.filtered` | `StateMatrix` |
| `result.xstd` | `result.smoothedStd` | `StateMatrix` |
| `result.C` | `result.smoothedCov` | `CovMatrix`: `.get(t,i,j)`, `.variance(t,i)`, `.at(t)` |
| `result.Cf` | `result.filteredCov` | `CovMatrix` |
| `result.lik` | `result.deviance` | `number` |
| `result.v` | `result.innovations` | `FloatArray` |
| `result.Cp` | `result.innovationVar` | `FloatArray` |
| `result.resid0` | `result.rawResiduals` | `FloatArray` |
| `result.resid` | `result.scaledResiduals` | `FloatArray` |
| `result.resid2` | `result.standardizedResiduals` | `FloatArray` |
| `result.ssy` | `result.rss` | `number` |
| `result.s2` | `result.residualVariance` | `number` |
| `result.V` | `result.obsNoise` | `FloatArray` |
| `result.x0` | `result.initialState` | `number[]` |
| `result.C0` | `result.initialCov` | `number[][]` |
| `result.XX` | `result.covariates` | `number[][]` |

### `DlmMleResult`

| Old name | New name |
|----------|----------|
| `mle.s` | `mle.obsStd` |
| `mle.w` | `mle.processStd` |
| `mle.lik` | `mle.deviance` |
| `mle.likHistory` | `mle.devianceHistory` |
| `mle.jitMs` | `mle.compilationMs` |
| `mle.arphi` | `mle.arCoefficients` |

### `DlmForecastResult`

| Old name | New name | Access pattern |
|----------|----------|----------------|
| `fc.x` | `fc.predicted` | `StateMatrix` |
| `fc.xstd` | `fc.predictedStd` | `StateMatrix` |
| `fc.xcov` | `fc.predictedCov` | `CovMatrix` |

### `DlmOptions` (dlmGenSys / dlmFit / dlmMLE)

| Old name | New name |
|----------|----------|
| `trig` | `harmonics` |
| `ns` | `seasonLength` |
| `arphi` | `arCoefficients` |
| `fitar` | `fitAr` |
| `fullseas` | `fullSeasonal` |

## Removed types and parameters

| Removed | Replacement |
|---------|-------------|
| `DType` enum (`DType.Float32`, `DType.Float64`) | `'f32'` / `'f64'` strings in `dtype` option |
| `DlmRunConfig` | Fields (`dtype`, `algorithm`) inlined into `DlmFitOptions` / `DlmMleOptions` |
| `stabilization` parameter | Auto-selected internally (Joseph form for Float32 scan, own formulation for assoc) |
| `forceAssocScan` | `algorithm: 'assoc'` in options |

## MATLAB compatibility helpers

For code that compares against MATLAB DLM output:

```ts
import { dlmFit, toMatlab } from "dlm-js";

const result = await dlmFit(y, { obsStd: 120, processStd: [40, 10], order: 1, dtype: 'f64' });
const matlab = toMatlab(result);

// matlab.x, matlab.xf, matlab.C, matlab.Cf, matlab.lik, etc. — MATLAB field names
```

```ts
import { dlmMLE, toMatlabMle } from "dlm-js";

const mle = await dlmMLE(y, { order: 1, maxIter: 300, dtype: 'f64' });
const matlab = toMatlabMle(mle);

// matlab.s, matlab.w, matlab.lik, matlab.likHistory, matlab.jitMs, etc.
```

## Search-and-replace patterns

Common sed/find-and-replace patterns for migrating existing code:

```
# Imports
DType.Float64  →  'f64'
DType.Float32  →  'f32'
import { DType } from "@hamk-uas/jax-js-nonconsuming"  →  (delete line)

# DlmGenSys options
trig:          →  harmonics:
ns:            →  seasonLength:
arphi:         →  arCoefficients:
fitar:         →  fitAr:
fullseas:      →  fullSeasonal:

# DlmFitResult fields
.lik           →  .deviance
.v             →  .innovations
.resid2        →  .standardizedResiduals
.ssy           →  .rss
.s2            →  .residualVariance

# DlmMleResult fields
.s             →  .obsStd
.w             →  .processStd
.lik           →  .deviance
.likHistory    →  .devianceHistory
.jitMs         →  .compilationMs
.arphi         →  .arCoefficients
```
