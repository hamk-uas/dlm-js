# dlmFitTensor Future Plan

## Status

Optional future work.

This is no longer a prerequisite for the current public API, MLE, or MAP workflow.

Already solved elsewhere:
- JS-idiomatic materialized results are shipped via `StateMatrix` / `CovMatrix`
- MATLAB compatibility is shipped via `toMatlab()` / `toMatlabMle()`
- parameter-space MLE and MAP are already supported via `dlmMLE(..., { loss })`

Not currently implemented:
- a public `dlmFitTensor(...)` entry point that returns on-device `np.Array` outputs
- `y: np.Array` input support for `dlmFit` / `dlmMLE`

## Why This Is Narrower Now

The original API-overhaul plan assumed `dlmFitTensor` was needed to support autodifferentiated losses with respect to model parameters.

That is no longer true in practice. The current `dlmMLE` path differentiates an internal Kalman objective directly, and the public custom-loss hook operates in parameter space:
- `DlmLossFn` receives `(deviance, params, meta)`
- `dlmMLE` optimizes that objective entirely on-device
- the final `dlmFit(...)` call happens only after optimization to build the user-facing result

So `dlmFitTensor` should be treated as a power-user and future-research feature, not as unfinished core infrastructure.

## When dlmFitTensor Would Still Be Useful

### 1. On-device post-fit composition

Power users may want smoothed states, covariances, or residual paths as `np.Array` tensors without materializing them into JS TypedArrays first.

Examples:
- custom tensor algebra on smoothed states
- downstream JAX-style kernels that consume filter/smoother output directly
- minimizing JS readback when chaining multiple on-device computations

### 2. Output-space differentiable objectives

Current MAP support is parameter-space only. `DlmLossFn` sees the scalar Kalman deviance and parameter vector, not the full smoothed/filter trajectories.

`dlmFitTensor` becomes relevant if we want objectives like:
- penalties on smoothed state trajectories
- penalties on forecast paths or uncertainty bands
- residual-shape losses beyond the prediction-error likelihood
- trajectory-regularized estimation that depends on latent states, not only parameters

### 3. Fully on-device workflows

If callers already have observations on-device, `y: np.Array` plus `dlmFitTensor` would avoid the extra JS upload/materialization boundary.

This matters most for:
- repeated batched fitting
- larger GPU workflows
- future MCMC or simulation loops that do not need a JS-facing fit object each iteration

### 4. Research APIs

The tensor API is a reasonable substrate for experimental work that should not distort the main JS API:
- batched fits
- output-space MAP objectives
- alternative posterior approximations
- iterative or simulation-heavy methods that only need selected scalar readback

## What It Is Not Needed For

`dlmFitTensor` is not required to support:
- current `dlmMLE`
- current MAP estimation via `loss`
- the current migration/compatibility surface
- ordinary JS/TS usage of `dlmFit` and `dlmForecast`

Those are already adequately served by the implemented API.

## If We Implement It Later

Keep the scope tight.

### Proposed API surface

```ts
interface DlmTensorResult extends Disposable {
  smoothed: np.Array;
  filtered: np.Array;
  smoothedCov: np.Array;
  filteredCov: np.Array;
  smoothedStd: np.Array;
  yhat: np.Array;
  ystd: np.Array;
  innovations: np.Array;
  innovationVar: np.Array;
  rawResiduals: np.Array;
  scaledResiduals: np.Array;
  standardizedResiduals: np.Array;
  deviance: np.Array;
  residualVariance: np.Array;
  mse: np.Array;
  mape: np.Array;
  rss: np.Array;
  nobs: np.Array;
  n: number;
  m: number;
}

async function dlmFitTensor(
  y: ArrayLike<number> | np.Array,
  opts: DlmFitOptions,
): Promise<DlmTensorResult>
```

### Implementation sketch

- reuse the existing `dlmSmo` core
- return squeezed state tensors directly instead of calling `tree.consumeData()` for the public result path
- keep ownership/disposal explicit via `Disposable`
- avoid broad API changes elsewhere unless a concrete downstream use-case needs them

### Non-goals

- do not reopen the already-completed naming/migration overhaul
- do not make `dlmFitTensor` a required dependency of `dlmMLE`
- do not widen the main JS API around speculative batching or MCMC until there is an active implementation task

## Decision Rule

Implement `dlmFitTensor` only if at least one of these becomes concrete:
- a user needs on-device outputs for downstream tensor computation
- we add output-space losses that depend on smoothed/filter trajectories
- a performance bottleneck is measured at the materialization boundary rather than assumed

Until then, this should remain documented as optional future work rather than unfinished core API debt.