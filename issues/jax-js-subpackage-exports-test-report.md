# Test report: `feat/subpackage-exports` branch installation & optax sub-path export

**Branch**: `feat/subpackage-exports` (commit `c380284`)  
**Tested from**: dlm-js (downstream consumer project)  
**Date**: 2025-07-18  
**Node**: v22+  
**pnpm**: v10.28.2  

## Summary

Installation and sub-path exports work correctly. The new `@hamk-uas` scope and sub-path export pattern (`@hamk-uas/jax-js-nonconsuming/optax`) are a significant ergonomics improvement — no separate package install, no pnpm overrides, no `file:` dependencies.

**One existing issue confirmed**: `optax.adam` is still not jittable due to `count.item()` in the bias correction logic. This is not a regression — same behavior as the old separate-package optax.

## Installation

### What works

```json
{
  "dependencies": {
    "@hamk-uas/jax-js-nonconsuming": "github:hamk-uas/jax-js-nonconsuming#feat/subpackage-exports"
  },
  "pnpm": {
    "onlyBuiltDependencies": ["@hamk-uas/jax-js-nonconsuming"]
  }
}
```

- `pnpm install` completes in ~270ms (cached) or ~2s (fresh build)
- All sub-packages are built during install via the `prepare` script (tsdown)
- The `onlyBuiltDependencies` entry is **required** — without it pnpm refuses to run the build script

### Package structure after install

```
node_modules/@hamk-uas/jax-js-nonconsuming/
├── dist/           # main package (jax core)
├── packages/
│   ├── optax/dist/      # sub-path: /optax
│   ├── loaders/dist/    # sub-path: /loaders
│   ├── onnx/dist/       # sub-path: /onnx
│   └── eslint-plugin/dist/  # sub-path: /eslint-plugin
├── package.json    # has "exports" field with all sub-paths
└── README.md
```

### Exports map (from package.json)

```json
{
  ".":              "./dist/index.js",
  "./optax":        "./packages/optax/dist/index.js",
  "./loaders":      "./packages/loaders/dist/index.js",
  "./onnx":         "./packages/onnx/dist/index.js",
  "./eslint-plugin": "./packages/eslint-plugin/dist/index.js"
}
```

## Import tests

### ✅ Main package import

```js
import { numpy as np, defaultDevice, grad, valueAndGrad, jit } from "@hamk-uas/jax-js-nonconsuming";
// Works. Array creation, disposal, grad, all OK.
```

### ✅ Optax sub-path import

```js
import { adam, applyUpdates } from "@hamk-uas/jax-js-nonconsuming/optax";
// Works. adam() returns optimizer with .init() and .update().
```

### ✅ Optax + valueAndGrad (eager mode)

```js
const loss = (x) => np.sum(np.square(x));
const optimizer = adam(0.01);
let params = np.array([5.0, -3.0, 2.0]);
let state = optimizer.init(params);
// 5 iterations, loss decreases correctly: 38.00 → 37.20
```

### ❌ Optax + jit (jit wrapping the optimizer step)

```js
const step = jit((params, state) => {
  const [val, grads] = valueAndGrad(loss)(params);
  const [updates, newState] = optimizer.update(grads, state);
  const newParams = applyUpdates(params, updates);
  return [newParams, newState, val];
});
// Error: count.item is not a function
```

**Root cause**: `optimizer.update()` internally calls `count.item()` on the step counter to extract a JS number for bias correction (`1 - β^t`). Under `jit()`, `count` is a tracer (abstract value), not a concrete array, so `.item()` fails.

**Workaround used in dlm-js**: We replaced optax adam with a pure-array Adam implementation where all operations (including bias correction via `np.power(beta, step)`) use only `np.Array` ops — fully traceable by JIT.

## Feedback for upstream

1. **`onlyBuiltDependencies` requirement**: The README should mention that pnpm users need `"onlyBuiltDependencies": ["@hamk-uas/jax-js-nonconsuming"]` in their package.json. Without it, installation fails with `ERR_PNPM_GIT_DEP_PREPARE_NOT_ALLOWED`.

2. **Optax JIT compatibility**: `adam`'s `count.item()` prevents wrapping the optimizer step in `jit()`. This is a significant limitation for performance-sensitive optimization loops. Consider replacing `count.item()` with pure array operations for bias correction (e.g., `np.power(beta1, count)` instead of `Math.pow(beta1, count.item())`).

3. **ESLint plugin sub-path**: The eslint plugin export at `@hamk-uas/jax-js-nonconsuming/eslint-plugin` is great — this replaces the previous `github:hamk-uas/jax-js-nonconsuming#path:packages/eslint-plugin&eslint-plugin-v0.1.0` install syntax which was fragile.

## Impact on dlm-js

When dlm-js migrates to this branch, the dependency section simplifies from:

```json
{
  "dependencies": {
    "@jax-js-nonconsuming/jax": "github:hamk-uas/jax-js-nonconsuming",
    "@jax-js-nonconsuming/optax": "file:tmp/jax-js-mono/packages/optax"
  },
  "pnpm": {
    "onlyBuiltDependencies": ["@jax-js-nonconsuming/jax"],
    "overrides": {
      "@jax-js-nonconsuming/optax>@jax-js-nonconsuming/jax": "github:hamk-uas/jax-js-nonconsuming"
    }
  }
}
```

To:

```json
{
  "dependencies": {
    "@hamk-uas/jax-js-nonconsuming": "github:hamk-uas/jax-js-nonconsuming#feat/subpackage-exports"
  },
  "pnpm": {
    "onlyBuiltDependencies": ["@hamk-uas/jax-js-nonconsuming"]
  }
}
```

And imports change from `"@jax-js-nonconsuming/jax"` to `"@hamk-uas/jax-js-nonconsuming"`.

Note: dlm-js currently uses a **pure-array Adam** (no optax dependency) because optax isn't jittable. If the `count.item()` issue is fixed, we could switch back to optax for cleaner code.
