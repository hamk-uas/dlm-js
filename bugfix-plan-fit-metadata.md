# Bugfix Plan — Timestamped Fit Metadata

## Goal

Fix the remaining irregular-timestamp API mismatch where `dlmFit` uses time-varying transition/noise matrices internally but only returns static base-step `G` and `W` in the public result.

## Findings

- Timestamped fits compute per-step `G_scan` / `W_scan` internally.
- The public `DlmFitResult` currently exposes only static `G` / `W`, which is misleading for irregular-time fits.
- Forecast no longer depends on those static fields, but users inspecting fit results still cannot recover the actual matrices used at each step.

## Plan

- [x] Add additive result fields for per-step transition matrices/covariances used during the fit.
- [x] Populate those fields for timestamped fits without changing existing `G` / `W` behavior.
- [x] Add timestamp tests that verify the exposed matrices match `dlmGenSysTV`.
- [x] Update docs/types/migration notes for the new additive fields.
- [x] Run focused timestamps validation.