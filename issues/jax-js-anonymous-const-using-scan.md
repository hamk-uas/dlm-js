# `using` + anonymous consts inside triple-nested `jit → lax.scan → valueAndGrad` crashes

- **Status**: 🔴 Open
- **Affected version**: `99ca222` (partially fixed from `2303627` by `41b0bda`; `99ca222` evalJaxpr refactor did not help)
- **Last known good**: `6e6d4fe`
- **Impact on dlm-js**: 21 of 200 tests fail (all `mle.test.ts`)

## Summary

The anonymous const disposal fix in `41b0bda` covers 2-level nesting (`jit → valueAndGrad`), but NOT 3-level nesting (`jit → lax.scan → valueAndGrad`). The `99ca222` evalJaxpr non-consuming refactor did not change this.

### What's fixed (41b0bda / 99ca222)

- `jit(valueAndGrad(lossFn))` where lossFn uses `np.ones` + `using` ✅ works (2 levels)
- `jit → scan` body → `using` + `np.eye`/`np.ones`/`np.zeros` ✅ works
- `valueAndGrad(lossFn)` without `jit` ✅ works
- Nested jaxpr disposal ✅ zero leaks

### What still crashes

- `jit(fn)` where `fn` calls `lax.scan(innerStep, ...)` and `innerStep` calls `valueAndGrad(lossFn)` where `lossFn` contains `using _ones = np.ones(...)` ❌ (3 levels: jit → scan → valueAndGrad)

This is the exact architecture of dlmMLE's Adam optimizer path:
```
jit((theta, optState, lastLik) => {
  lax.scan(innerStep, {theta, optState, lastLik}, null, {length: 10});
})
// where innerStep = (carry) => {
//   const [lik, grad] = valueAndGrad(lossFn)(carry.theta);  // lossFn has np.ones
//   ...
// }
```

## Root cause hypothesis

`41b0bda` guards `[Symbol.dispose]()` to skip if `inMakeJaxprBody() && anonymousConstArrays.has(this)`. This works for 2-level nesting (`jit → valueAndGrad`). But in 3-level nesting (`jit → lax.scan → valueAndGrad`), the `lax.scan` body creates its own makeJaxprBody context. When valueAndGrad traces inside that scan body, anonymous consts created by `np.ones()` in the innermost lossFn get disposed during the scan body tracing, before the outer jit can capture them.

## Reproduction

```bash
npx tsx issues/repro-anonymous-const-using-scan.ts
```

Standalone minimal repro — the triple-nesting pattern:

```typescript
// The REAL crash pattern (Case 5 in repro):
// jit → lax.scan(innerStep) → valueAndGrad(lossFn) → inner lax.scan
// This is dlmMLE's Adam optimizer architecture.

const lossFn = (theta) => {
  using _ones = np.ones([n, 1, 1], { dtype });  // ← disposed during scan body tracing
  // ... lax.scan(step, ...) inside ...
  return np.sum(liks);
};

const innerStep = (carry, _x) => {
  const [lik, grad] = valueAndGrad(lossFn)(carry.theta);
  // ...
  return [newCarry, null];
};

const optimBlock = jit((theta, optState, lastLik) => {
  const [finalCarry] = lax.scan(innerStep, carry, null, { length: 10 });
  return finalCarry;
});
// Throws: "Referenced tracer Array:float64[n,1,1] has been disposed"
```

## Impact on dlm-js

- **21 of 200 tests fail** — all in `mle.test.ts`. The Adam path uses `jit → lax.scan(innerStep, ..., {length: 10})` where `innerStep` calls `valueAndGrad(lossFn)` and `lossFn` creates arrays with `np.ones([n,1,1])`, `np.eye(m)`, `np.zeros([1,1,1])` declared with `using`.
- The natural gradient path (`jit(valueAndGrad(lossFn))` — 2 levels) now works on 99ca222 but is exercised in same test suite and fails due to earlier Adam tests contaminating state.
- 179 tests pass (all non-MLE tests).

### Workaround locations

No workaround applied — waiting for upstream fix.

## Suggested fix

The `inMakeJaxprBody()` guard in `[Symbol.dispose]()` needs to account for the full nesting depth. When `jit → lax.scan → valueAndGrad` traces, there are 3+ makeJaxprBody levels on the stack. An anonymous const array created by `np.ones()` in the innermost lossFn body gets disposed during the scan body tracing. The guard should suppress disposal at ALL nesting depths, not just 1-2 levels deep.

## Related

- `jax-js-jit-nested-jaxpr-disposal.md` — RESOLVED in `41b0bda`+`a9db43d`: zero leaks in repro. This issue tracks the remaining triple-nesting regression.
