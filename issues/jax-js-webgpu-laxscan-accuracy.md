# WebGPU `lax.scan` produces catastrophic accuracy errors

🔴 **Open**

## Summary

`lax.scan` on the WebGPU backend produces catastrophically wrong results — output arrays are either all-NaN (standalone invocations) or contain values with 10,000%–1,370,000% relative error vs Octave/f64 reference (when run after cpu/wasm warmup in bench-full).

Both `algorithm:'scan'` and `algorithm:'ud'` use `lax.scan` for the forward Kalman filter pass and are equally affected. `algorithm:'assoc'` uses `lax.associativeScan` and shows elevated but much smaller errors (61%–27,500% — separate issue from the catastrophic scan failure).

## Affected version

- **jax-js-nonconsuming:** block-map branch, commit `689d8a5` (and local `4cf7398` — identical source)
- **Last known good:** v0.7.10, commit `e3b88ab` (ultimate-architecture-plan branch)
- **First broken:** block-map branch commit `3e580f4` (dlm-js commit `632321a` switched to this)

## Bisection

| dlm-js commit | jax-js commit | branch | WebGPU scan maxRelErr (Nile o=1) |
|---|---|---|---|
| `90a9e77` | `e3b88ab` | ultimate-architecture-plan | 5.58e-3 (0.56%) ✓ |
| `07917b8` | v0.7.10 | ultimate-architecture-plan | ~same ✓ |
| `632321a` | `3e580f4` | **block-map** | **1.37e+6 (1,370,000%)** ✗ |
| `1d7e5a7` | `689d8a5` | block-map | **1.37e+6** ✗ |

The regression appeared when switching from ultimate-architecture-plan to the block-map branch.

## Two failure modes

### Mode 1: all-NaN readback (standalone)

When calling `dlmFit` on WebGPU without prior cpu/wasm calls in the same process, `tree.consumeData()` returns all-NaN for every output field:

```
scan:  yhat[0..4] = NaN, NaN, NaN, NaN, NaN
assoc: yhat[0..4] = NaN, NaN, NaN, NaN, NaN
ud:    yhat[0..4] = NaN, NaN, NaN, NaN, NaN
```

All algorithms affected in this mode (including assoc).

### Mode 2: wrong values (sequential bench-full)

When `scripts/bench-full.ts` runs all backend×dtype×algorithm combos sequentially (cpu/wasm first, then webgpu), lax.scan produces finite but drastically wrong values:

| Model | Good-era maxRelErr | Current maxRelErr | Regression factor |
|---|---|---|---|
| Nile, order=0 (m=1) | 1.06e-4 | 1.00e+4 (10,001%) | 100M× |
| Nile, order=1 (m=2) | 5.58e-3 | 1.37e+6 (1,370,209%) | 250M× |
| Kaisaniemi (m=4) | 0.985 | 1.27e+6 (1,268,584%) | 1.3M× |
| Energy (m=5) | 0.116 | 2.49e+5 (248,710%) | 2.1M× |
| Gapped (m=2) | 2.78e-3 | 1.64e+4 (16,428%) | 5.9M× |

Per-field breakdown (Nile o=1, Mode 2):
- **C0** (initial covariance): 100% error — catastrophic
- **x0** (initial state): 63.6% error — catastrophic
- **lik, s2, ssy** (scalar diagnostics): 0.4–0.8% error — degraded
- **F, G, W, n, nobs** (structural/config): 0% — correct

## Root cause hypothesis

The block-map branch likely changed the WebGPU `lax.scan` kernel dispatch or carry propagation. The sequential batched-submit path (`O(N/256)` dispatches) may have a carry-forwarding bug where intermediate state is not correctly transferred between batches.

Evidence:
- `lax.associativeScan` (different code path) is NOT catastrophically broken — its errors are elevated but orders of magnitude smaller
- cpu/wasm `lax.scan` is unaffected — identical results to good-era
- The error pattern (C0 and x0 catastrophically wrong, scalar sums less affected) is consistent with state accumulation corruption

## Reproduction

### Using dlm-js tests (recommended)

```bash
cd /path/to/dlm-js
pnpm install

# Node (skips WebGPU tests — no device available):
pnpm vitest run tests/webgpu-scan.test.ts
# → 2 skipped (expected)

# Deno (exercises WebGPU):
# tests/webgpu-scan.test.ts has 5 models × 2 algorithms = 10 tests
# Thresholds from good-era commit 90a9e77 with 5-10× headroom

# Full bench-full regeneration (shows the error table):
pnpm run bench:full
# → Check webgpu/f32/scan and webgpu/f32/ud rows
```

### Standalone repro

```bash
# From dlm-js root:
deno run --unstable-webgpu --allow-read --allow-write --allow-env --allow-run issues/repro-webgpu-laxscan.ts
```

## Impact on dlm-js

- WebGPU `algorithm:'scan'` and `algorithm:'ud'` are completely broken — unusable for production
- The default WebGPU path (`algorithm:'assoc'`) still works but has elevated errors
- All 5 benchmark models affected (Nile o=0, Nile o=1, Kaisaniemi, Energy, Gapped)
- `tests/webgpu-scan.test.ts` guards against this regression with good-era thresholds

## Good-era thresholds (for reference)

From dlm-js commit `90a9e77`, jax-js v0.7.10 (`e3b88ab`):

| Model | maxRelErr (scan) | maxRelErr (ud) |
|---|---|---|
| Nile, order=0 (m=1) | 1.06e-4 | ~same |
| Nile, order=1 (m=2) | 5.58e-3 | ~same |
| Kaisaniemi (m=4) | 0.985 | ~same |
| Energy (m=5) | 0.116 | ~same |
| Gapped (m=2) | 2.78e-3 | ~same |
