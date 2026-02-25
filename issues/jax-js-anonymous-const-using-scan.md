# `using` + `markAnonymousIfTracing()` inside `jit(valueAndGrad(...))` body crashes JIT

- **Status**: 🔴 Open
- **Affected version**: `a9db43d` (partially fixed from `2303627` by `41b0bda`)
- **Last known good**: `6e6d4fe`
- **Impact on dlm-js**: 21 of 200 tests fail (all `mle.test.ts`)

## Summary

Commit `41b0bda` fixed the `jit → scan` nesting level (our original repro passes), but the fix does not cover `jit(valueAndGrad(lossFn))` where `lossFn` uses `np.ones`/`np.zeros`/`np.eye` with `using` declarations. The `valueAndGrad` creates an additional abstract trace nesting level that the `inMakeJaxprBody()` guard doesn't cover.

### What's fixed (41b0bda)

- `jit → scan` body → `using` + `np.eye`/`np.ones`/`np.zeros` ✅ works
- `valueAndGrad(lossFn)` without `jit` ✅ works
- Nested jaxpr disposal (`repro-nested-jaxpr-disposal.ts`) ✅ zero leaks

### What still crashes

- `jit(valueAndGrad(lossFn))` where `lossFn` contains `using _V2_ones = np.ones(...)` ❌

## Root cause hypothesis

`41b0bda` guards `[Symbol.dispose]()` to skip if `inMakeJaxprBody() && anonymousConstArrays.has(this)`. But `jit(valueAndGrad(...))` creates a deeper nesting: jit trace → valueAndGrad trace → scan trace. The anonymous const created by `np.ones()` inside the loss function body gets disposed during the `valueAndGrad` trace before the outer `jit` trace can capture it.

## Reproduction

```bash
npx tsx issues/repro-anonymous-const-using-scan.ts
```

Standalone minimal repro (the `jit(valueAndGrad(...))` path):

```typescript
import { numpy as np, jit, lax, DType, valueAndGrad } from '@hamk-uas/jax-js-nonconsuming';

const lossFn = (theta: any) => {
  using s = np.exp(theta);
  using V2 = np.reshape(np.square(s), [1, 1]);
  using _V2_ones = np.ones([5, 1, 1], { dtype: DType.Float64 }); // ← crashes
  using V2_arr = np.multiply(_V2_ones, V2);

  const step = (carry: any, inp: any) => {
    using pred = np.matmul(np.eye(1, undefined, { dtype: DType.Float64 }), carry);
    using diff = np.subtract(inp, pred);
    using lik = np.divide(np.square(diff), V2_arr);
    return [carry, np.squeeze(lik)];
  };
  const [_, liks] = lax.scan(step, np.zeros([1, 1], { dtype: DType.Float64 }), V2_arr);
  _.dispose();
  return np.sum(liks);
};

const theta = np.array([0.5], { dtype: DType.Float64 });

// valueAndGrad alone: OK
const [val1, grad1] = valueAndGrad(lossFn)(theta);  // ✅ works

// jit(valueAndGrad): CRASHES
const [val2, grad2] = await jit(valueAndGrad(lossFn))(theta);
// Throws: "Referenced tracer Array:float64[1,1] has been disposed"
```

## Impact on dlm-js

- **21 of 200 tests fail** — all in `mle.test.ts`. `dlmMLE` uses `jit(valueAndGrad(lossFn))` where `lossFn` creates arrays with `np.ones([n,1,1])`, `np.eye(m)`, `np.zeros([1,1,1])` declared with `using`.
- 179 tests pass (all non-MLE tests) — the `jit → scan` fix in `41b0bda` resolved the `dlmSmo`/`dlmFit` crashes.

### Workaround locations

No workaround applied — waiting for upstream fix.

## Suggested fix

The `inMakeJaxprBody()` guard in `[Symbol.dispose]()` needs to account for the full nesting depth. When `jit(valueAndGrad(...))` traces, there are multiple makeJaxprBody levels on the stack. The anonymous const disposal guard should apply at ALL nesting depths, not just the immediate `makeJaxprBody` context.

## Related

- `jax-js-jit-nested-jaxpr-disposal.md` — the original issue. Now RESOLVED for `jit → scan` nesting (zero leaks in repro). This issue tracks the remaining `jit(valueAndGrad(...))` regression.
