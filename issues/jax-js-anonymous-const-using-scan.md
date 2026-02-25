# `using` + `markAnonymousIfTracing()` inside scan body crashes JIT

- **Status**: 🔴 Open
- **Affected version**: `2303627` (specifically commits `2af40bc` + `ec9acec`)
- **Last known good**: `6e6d4fe`
- **Impact on dlm-js**: 132 of 200 tests fail — complete regression

## Summary

After the anonymous const marking changes (`2af40bc`: mark anonymous consts at creation time; `ec9acec`: always ref anonymous consts for eye/arange/linspace), arrays created by `np.eye()`, `np.zeros()`, `np.arange()`, `np.linspace()` inside `lax.scan` bodies within `jit()` are disposed during tracing when declared with `using`, leaving the JIT with dangling constant references.

## Root cause hypothesis

`markAnonymousIfTracing()` on `np.eye()` (and other array-creation functions) changes the ownership semantics of the array. Inside a nested abstract trace (lax.scan body within jit), `[Symbol.dispose]()` was previously a no-op. With the new anonymous const system, the array is marked as owned by the outer JIT builder. The `using` keyword's `[Symbol.dispose]()` then actually disposes the array during tracing (because it's now "anonymous" and disposal is allowed), but the JIT retains a reference to it as a `ClosedJaxpr` constant. When the JIT executes, `_realizeSource` finds the disposed constant and throws:

```
ReferenceError: Referenced tracer Array:float64[2,2] has been disposed
```

## Reproduction

```bash
npx tsx issues/repro-anonymous-const-using-scan.ts
```

Inline minimal repro:

```typescript
import { numpy as np, jit, lax, DType } from '@hamk-uas/jax-js-nonconsuming';

const fn = (x) => {
  const step = (carry, inp) => {
    using e = np.eye(2, undefined, { dtype: DType.Float64 });  // ← crashes
    // const e = np.eye(2, undefined, { dtype: DType.Float64 });  // ← works
    const newCarry = np.add(np.matmul(e, carry), inp);
    return [newCarry, newCarry];
  };
  const [finalCarry, allSteps] = lax.scan(step, np.zeros([2, 1], { dtype: DType.Float64 }), x);
  finalCarry.dispose();
  return allSteps;
};

const result = await jit(fn)(np.ones([5, 2, 1], { dtype: DType.Float64 }));
// Throws: "Referenced tracer Array:float64[2,2] has been disposed"
```

Key observation: the same pattern works fine in `jit` WITHOUT `lax.scan` nesting, and works fine in `lax.scan` without `using`. The regression is specifically `jit` → `lax.scan` body → `using` + anonymous-const-marked creation function.

## Impact on dlm-js

- **132 of 200 tests fail** — `dlmSmo` uses `jit(core)` where `core` contains `lax.scan(forwardStep, ...)` and `lax.scan(backwardStep, ...)`. Both step functions use `using` declarations for arrays created by `np.eye()`, `np.zeros()`, `np.array(...)`, `np.full(...)`, etc.
- All 9 of 11 test files that call `dlmFit`/`dlmSmo` are affected.
- The `using` pattern inside scan bodies is extensively used per the library's memory management conventions (ESLint rule `jax-js/require-using` enforces it).

### Workaround locations

The workaround would be removing `using` from all array creation inside scan bodies, but this would:
1. Violate the project's own ESLint rules
2. Reintroduce the exact leaks that `using` was designed to prevent
3. Contradict the documented behavior: "`using` IS correct inside grad/jit/scan traced bodies"

**No workaround applied** — waiting for upstream fix.

## Suggested fix

Restore the `[Symbol.dispose]()` no-op behavior for arrays created inside nested abstract traces (lax.scan/associativeScan body within jit). The `markAnonymousIfTracing()` call should not change disposal semantics during tracing — it should only affect ownership tracking for `ClosedJaxpr.dispose()` after tracing completes.

Specifically: in `[Symbol.dispose]()`, if the array is inside a `makeJaxprBody` context (scan body tracing), disposal should remain a no-op regardless of anonymous marking.

## Related

- `jax-js-jit-nested-jaxpr-disposal.md` — the original issue that motivated the anonymous const fix. The nested jaxpr disposal fix (`2af40bc`) is the correct direction, but the implementation broke `using` inside scan bodies.
