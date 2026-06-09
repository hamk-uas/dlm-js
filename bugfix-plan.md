# Bugfix Plan

## Goal

Fix confirmed chronology-related bugs around irregular timestamps, focusing on forecast behavior that is currently documented but not implemented.

## Findings

- `dlmForecast(..., { timestamps })` is documented, but the implementation ignores `opts.timestamps` and always uses static `fit.G` / `fit.W`.
- Timestamp-aware forecasting needs enough metadata from `dlmFit` to reconstruct per-step `G(Δt)` and `W(Δt)`.
- Timestamped fits already showed one chronology bug in initialization; similar bugs are most likely where compressed observed-only data is treated like unit-step data.

## Plan

- [x] Preserve the minimal fit metadata needed for timestamp-aware forecasting.
- [x] Implement real `dlmForecast` support for irregular timestamps using reconstructed per-step `G/W`.
- [x] Add forecast tests that compare timestamp-aware forecast output against an equivalent NaN-extended `dlmFit` run.
- [x] Update public docs/types for any new fit-result metadata and timestamp forecast behavior.
- [x] Run focused forecast validation.

## Validation target

- `tests/forecast.test.ts`
- Narrow timestamp-aware forecast equivalence tests first, then the full forecast suite.