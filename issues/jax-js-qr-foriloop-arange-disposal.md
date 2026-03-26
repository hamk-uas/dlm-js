# QR foriLoop path: UseAfterFreeError on arange constants inside JIT

- **Status**: 🔴 Open
- **Affected version**: v0.9.5 (`bbd1d2ae`) through v0.9.6 (`9c03f593`)
- **Last known good**: v0.9.3 (`ea264386`)
- **Severity**: Blocking — breaks all `np.linalg.qr` calls with matrix dim > 8 inside `jit()`

## Summary

`householderQR2D` in `src/library/lax-linalg.ts` creates `arange_m` and `arange_n` (int32 index arrays), uses them inside a `foriLoop` body (which captures them as jaxpr constants), then explicitly disposes them after `foriLoop` returns.

Under the v0.9.5 ownership restructuring (explicit creation-ref balancing, commit `1967dddc`), the explicit `.dispose()` calls on lines 595–596 free the backing data while the `ClosedJaxpr` still references them. The old phantom-ref model presumably kept them alive via the jaxpr's constant references.

## Reproduction

```
npx tsx issues/repro-jit-assocscan-qr-disposal.ts
```

Minimal inline repro:

```typescript
import { init, numpy as np, jit, defaultDevice } from '@hamk-uas/jax-js-nonconsuming';
await init('wasm');
defaultDevice('wasm');

const result = await jit(() => {
  using A = np.eye(9, { dtype: 'float64' }); // 9 > QR_UNROLL_LIMIT=8
  const [Q, R] = np.linalg.qr(A);
  Q.dispose();
  return R;
})();
// → ReferenceError: Referenced tracer Array:int32[9] has been disposed
```

- `m ≤ 8`: QR takes the **unrolled** path → no foriLoop → arange not captured by jaxpr → OK
- `m > 8`: QR takes the **foriLoop** path → arange captured as jaxpr const → disposed → **FAIL**

## Root cause

In `lax-linalg.ts`, `householderQR2D`:

```typescript
// Lines ~526-527: created WITHOUT `using` — comment says they're jaxpr consts
const arange_m = numpy.arange(m);  // int32[m]
const arange_n = numpy.arange(n);  // int32[n]

// ... used inside foriLoop body (captured by jaxpr) ...

// Lines ~595-596: explicitly disposed after foriLoop returns
arange_m.dispose();   // ← frees backing data; ClosedJaxpr still references it
arange_n.dispose();   // ← same
```

The existing comment ("Don't use `using` — under PE tracing, these concrete arrays become jaxpr consts referenced by the backward pass") correctly identified that these arrays need to outlive the function scope. But the explicit `.dispose()` at the end defeats that intention under the new ownership model.

## Suggested fix

Remove the explicit `.dispose()` calls on `arange_m` and `arange_n` in `householderQR2D`. Let the `ClosedJaxpr` manage their lifetime — they should be freed when the jaxpr that captured them is disposed.

Alternatively, if explicit disposal is needed for eager mode, gate the disposal on whether a jaxpr trace is active (similar to the existing `core.inMakeJaxprBody()` check on line ~583).

## Impact on dlm-js

- **21 test failures** in `sqrtassoc.test.ts` (4 genuine + 17 cascade from JIT state corruption)
- All sqrt-assoc models with state dimension m > 8 are broken (fullSeasonal m=13, trig seasonal m=12)
- Blocks v0.9.6 upgrade: `package.json` stays at v0.9.3 until resolved
- No workaround exists — QR is fundamental to the tria() operation in square-root Kalman filtering
