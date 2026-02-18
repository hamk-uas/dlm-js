# eslint-plugin-jax-js: `require-using` should support AD-traced contexts

**Package:** `@jax-js-nonconsuming/eslint-plugin-jax-js`

**Status:** Resolved — the premise was incorrect. See resolution below.

## Original Problem (incorrect)

The `jax-js/require-using` rule warns on every `const` binding of an `np.Array`.
Inside AD-traced functions (passed to `valueAndGrad`, `grad`, or used as a
`lax.scan` step), the tracer manages array lifetimes — calling `.dispose()` or
using `using` is unnecessary and potentially harmful (it would dispose a tracer
object).

## Resolution

**`using` is safe and correct inside `grad` / `jit` / `lax.scan` bodies.** The
original concern was a misconception. Per upstream commit `767bd26` (Feb 18, 2026),
the tracer dispose behaviour is:

| Context | Tracer type | `dispose()` behaviour |
|---|---|---|
| Inside `jit(...)` body | `JaxprTracer` | No-op. Harmless. |
| Inside `grad(...)`/`valueAndGrad(...)` body | `JVPTracer` | Decrements its own `#rc`. Correct. |
| Concrete `Array` (captured constant/input) | `Array` | Normal ref-counted disposal. Required. |

Ownership rules are identical in eager and traced contexts. `jit()`/`grad()` are
pure performance optimisations — they must not change ownership semantics.

The `/* eslint-disable jax-js/require-using */` blocks in `src/mle.ts` should be
removed; the rule now also includes a clarifying message in its error text.

## Context

This came up in `dlm-js` while implementing MLE via autodiff (`src/mle.ts`).
The original incorrect conclusion was that `using` would dispose tracer objects
harmfully. In fact `using` on a `JVPTracer` intermediate correctly decrements its
reference count, and on a `JaxprTracer` it is a no-op.
