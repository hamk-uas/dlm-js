# JIT nested ClosedJaxpr disposal is non-recursive

**Status:** 🔴 Open  
**Affected version:** `892e95dc71942217cf7589b758852a9532567169`  
**Reported:** 2026-02-28

## Summary

`ClosedJaxpr.dispose()` only disposes its direct `this.consts` array but does
**not** recurse into equations that contain nested sub-jaxprs (e.g. `lax.scan`,
`lax.associativeScan`, `jvp`, `transpose` sub-jaxprs). This causes arrays
captured by those nested sub-jaxprs to leak even after `_disposeAllJitCaches()`
runs (which is called by `checkLeaks.stop()`).

Additionally, `Array[Symbol.dispose]()` is a no-op during PE tracing (guarded by
`_peArrayCreationTracker`), so `using` declarations on concrete arrays inside
JIT-traced bodies never fire. Constants created inside the traced body become
nested sub-jaxpr `consts` entries with `.ref()` that survive `clearCaches()`.

## Root cause

```js
// dist/index.js:2257
dispose() {
  for (const c of this.consts) c.dispose();  // ← only top-level
}
```

This iterates `this.consts` but does NOT walk `this.jaxpr.eqns` to find
equations whose `params` contain nested `ClosedJaxpr` objects (e.g. scan
body jaxprs, JVP sub-jaxprs, transpose sub-jaxprs).

```js
// dist/index.js:5520
[Symbol.dispose]() {
  if (!_peArrayCreationTracker && this.#rc > 0) this.dispose();
  // ↑ no-op when inside PE (JIT/grad/scan tracing)
}
```

## Reproduction

Structure: `jit → valueAndGrad → lax.scan(step, ...)` where `step` captures
closure constants (F, x0, C0, y_arr). After the JIT function executes and the
`using` scope exits, `checkLeaks.stop()` reports the closure-captured arrays as
leaked with `rc=1`.

```ts
// issues/repro-nested-jaxpr-disposal.ts
import { numpy as np, jit, lax, valueAndGrad, checkLeaks, DType } from '@hamk-uas/jax-js-nonconsuming';

const dtype = DType.Float64;
const n = 10;

checkLeaks.start();

{
  // Constants created OUTSIDE the JIT body (closure-captured)
  using F  = np.array([[1]], { dtype });
  using Ft = np.transpose(F);
  using x0 = np.zeros([1, 1], { dtype });
  using C0 = np.eye(1, undefined, { dtype });
  using y_arr = np.ones([n, 1, 1], { dtype });

  const lossFn = (theta: np.Array): np.Array => {
    type Carry = { x: np.Array; C: np.Array };
    type Inp   = { y: np.Array };

    // Inline constant — leaks because `using` is no-op during PE
    using ones = np.ones([n, 1, 1], { dtype });

    const step = (carry: Carry, inp: Inp): [Carry, np.Array] => {
      // F, Ft captured from outer scope → become scan sub-jaxpr consts
      using v = np.subtract(inp.y, np.matmul(F, carry.x));
      using CFt = np.matmul(carry.C, Ft);
      const Cp = np.add(np.matmul(F, CFt), ones);
      const x_next = np.add(np.matmul(F, carry.x), np.divide(np.matmul(F, CFt), Cp));
      using lik = np.add(np.divide(np.square(v), Cp), np.log(Cp));
      return [{ x: x_next, C: carry.C }, np.squeeze(lik)];
    };

    const [fc, liks] = lax.scan(step, { x: x0, C: C0 }, { y: y_arr });
    fc.x.dispose(); fc.C.dispose();
    const total = np.sum(liks);
    liks.dispose();
    return total;
  };

  const fn = jit((theta: np.Array) => {
    const [val, grad] = valueAndGrad(lossFn)(theta);
    grad.dispose();
    return val;
  });

  using theta = np.zeros([1], { dtype });
  using result = await fn(theta);
}

const report = checkLeaks.stop();
console.log(`leaked=${report.leaked} userLeaked=${report.userLeaked} internalLeaked=${report.internalLeaked}`);
if (report.userLeaked > 0) {
  console.log(report.summary);
}

// Expected: userLeaked=0 (all constants properly `using`'d or closure-captured)
// Actual:   userLeaked>0 (F, Ft, x0, C0, y_arr, ones all leak with rc=1)
```

Run: `npx tsx issues/repro-nested-jaxpr-disposal.ts`

## Impact on dlm-js

`dlmMLE` uses nested `jit → lax.scan → valueAndGrad → lax.scan` (sequential
path) or `jit → valueAndGrad → lax.associativeScan` (assoc path). Every MLE
test leaks 11–29 `userLeaked` arrays — all upstream-caused. We work around this
with a per-test leak budget (`globalThis.__jaxUserLeakBudget = 30`), but this
masks potential real user-code leaks.

**Update (6e6d4fe):** After migrating from `np.dot(vec, one_hot_mask)` to
`lax.dynamicIndexInDim` (which uses `Shrink` primitive internally), leaks
increase to 33 per test — the reshaped traced graph captures 3 more constants
in nested sub-jaxprs. Budget NOT bumped — waiting for upstream fix.

Leak categories:
1. **Closure-captured constants** (G, F, Ft, x0, C0, y_arr) — properly `using`'d
   at outer scope, but `.ref()`'d into nested scan sub-jaxpr `consts` that
   `_disposeAllJitCaches()` cannot reach.
2. **Inline constants** (`np.ones`, `np.eye` inside `np.diag`, `np.array` masks)
   — created inside JIT-traced body where `[Symbol.dispose]()` is a no-op.
3. **`evalJaxprTransposed` onesLike seed** — `valueAndGrad` internally creates
   an `onesLike` seed that gets tagged as user code.

## Suggested fix

Make `ClosedJaxpr.dispose()` recursive: walk `this.jaxpr.eqns`, and for each
equation whose `params` contains a `ClosedJaxpr` (e.g. scan body, JVP/transpose
sub-jaxprs), call `.dispose()` on it recursively. Guard against cycles if needed.

```js
dispose() {
  for (const c of this.consts) c.dispose();
  for (const eqn of this.jaxpr.eqns) {
    for (const val of Object.values(eqn.params)) {
      if (val instanceof ClosedJaxpr) val.dispose();
    }
  }
}
```
