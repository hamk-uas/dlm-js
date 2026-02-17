# eslint-plugin-jax-js: `require-using` should support AD-traced contexts

**Package:** `@jax-js-nonconsuming/eslint-plugin-jax-js`

## Problem

The `jax-js/require-using` rule warns on every `const` binding of an `np.Array`.
Inside AD-traced functions (passed to `valueAndGrad`, `grad`, or used as a
`lax.scan` step), the tracer manages array lifetimes — calling `.dispose()` or
using `using` is unnecessary and potentially harmful (it would dispose a tracer
object).

Currently the only workaround is a broad `/* eslint-disable jax-js/require-using */`
block around traced code.

## Desired behaviour

Ideally the rule would recognise that arrays created inside a function passed to
`valueAndGrad`, `grad`, `jit`, or `lax.scan` (step fn) are tracer-managed, and
suppress the warning automatically. Possible approaches:

1. **Scope-based suppression** — if the enclosing function is an argument to a
   known tracing API (`grad`, `valueAndGrad`, `jit`, `lax.scan`), skip the
   warning for bindings inside it.
2. **Inline annotation** — a lighter-weight marker like
   `// jax-js-traced` on a function that opts out its body.
3. **Rule option** — e.g. `"jax-js/require-using": ["warn", { "ignoreInTracedFns": true }]`
   with a configurable list of tracing entry-points.

## Current workaround

```ts
/* eslint-disable jax-js/require-using */
const buildDiagW = (...) => { ... };

const makeKalmanLoss = (...) => {
  const step = (carry, inp) => { ... };  // scan step — traced
  return (theta) => { ... };             // loss fn — traced via valueAndGrad
};
/* eslint-enable jax-js/require-using */
```

## Context

This came up in `dlm-js` while implementing MLE via autodiff (`src/mle.ts`).
The Kalman filter loss function and its helpers are fully AD-traced — every
intermediate `np.Array` is a tracer, not a real tensor. Using `const` is
correct; `using` would attempt to dispose tracer objects.
