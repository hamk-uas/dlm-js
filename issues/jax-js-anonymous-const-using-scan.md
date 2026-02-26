# `using` + anonymous consts inside `valueAndGrad(wrapper)` where wrapper calls scan-containing function

- **Status**: 🟡 Partially fixed — `e034f26` resolved core triple-nesting crash (14/21 MLE tests fixed), but wrapping a scan-containing function inside another function traced by `valueAndGrad` still leaks/crashes (7/21 tests)
- **Affected version**: `e034f26` (PETracer #rc drain fix; previous: `99ca222`, `41b0bda`)
- **Impact on dlm-js**: 7 of 200 tests fail (MAP/prior tests in `mle.test.ts`; was 21 on `99ca222`)

## Summary

The `e034f26` fix (drain PETracer #rc in `partialEvalGraphToJaxpr` for multi-output scan equations) resolved the core triple-nesting crash. **14 of 21 MLE tests now pass** — the standard `dlmMLE` path (`jit → lax.scan → valueAndGrad(kalmanLoss)`) works correctly.

However, **7 MAP/prior tests still fail** due to a remaining leak vector: wrapping a function that contains `lax.scan` inside another function passed to `valueAndGrad`. This is the exact pattern dlm-js uses for MAP estimation.

### What works after e034f26

- `valueAndGrad(kalmanLoss)(theta)` where `kalmanLoss` internally calls `lax.scan` ✅
- `jit(lax.scan(step))` where step calls `valueAndGrad(kalmanLoss)` ✅ (14 tests)
- All non-MLE tests (179/179) ✅

### What still crashes/leaks

- `valueAndGrad(wrapper)(theta)` where `wrapper = (theta) => kalmanLoss(theta)` ❌
- Same crash with any custom wrapper, even the trivial identity `(theta) => kalmanLoss(theta)` ❌
- Adam path (jit → scan → valueAndGrad(wrapper)): 8 leaked arrays + UseAfterFreeError
- Natural gradient path (jit → valueAndGrad(wrapper)): 4 leaked arrays + UseAfterFreeError

## Root cause

When `valueAndGrad` traces `wrapper(theta)`, and `wrapper` calls `kalmanLoss(theta)` which internally uses `lax.scan`, the inner scan creates anonymous const PETracers. These tracers are created one level deeper in the call stack compared to when `kalmanLoss` is traced directly. The `e034f26` drain fix handles const tracers at the direct scan level but not when the scan is called through an intermediate wrapper function during partial evaluation.

The key invariant: `valueAndGrad(f)` where `f` directly IS the scan-containing function → 0 leaks. But `valueAndGrad(g)` where `g` calls `f` → leaks. The only difference is one level of function call indirection.

## Reproduction

```bash
npx tsx issues/repro-wrapped-valueandgrad-scan.ts
```

Or run the failing tests directly:
```bash
pnpm vitest run tests/mle.test.ts -t "MAP"
pnpm vitest run tests/mle.test.ts -t "dlmPrior"
```

### Expected output from repro

```
Case 1 (no loss):        userLeaked= 0
Case 2 (identity+Adam):  CRASH: Referenced tracer Array:float64[1] has been disposed
  leaked anyway: 8
Case 3 (identity+natrl): CRASH: Referenced tracer Array:float64[1] has been disposed
  leaked anyway: 4

=== Summary ===
  ✅ OK  No loss (direct kalmanLoss)
  💥 CRASH  Identity loss (Adam)
  💥 CRASH  Identity loss (natural)
```

### Minimal pattern

```typescript
// This works — kalmanLoss is traced directly:
const gradFn = jit(valueAndGrad(kalmanLoss));
gradFn(theta); // ✅ 0 leaks

// This crashes — kalmanLoss is called through a wrapper:
const wrapper = (theta: np.Array): np.Array => kalmanLoss(theta);
const gradFn2 = jit(valueAndGrad(wrapper));
gradFn2(theta); // ❌ UseAfterFreeError + leaks
```

Where `kalmanLoss` is built by `makeKalmanLoss` — it contains `lax.scan` with structured inputs, many `using` intermediates, and multi-field carry/output shapes (the full Kalman filter forward pass).

## Impact on dlm-js

- **7 of 200 tests fail** — all MAP/prior tests in `mle.test.ts`:
  - `MAP: adam returns priorPenalty` (crash + 8 leaks)
  - `MAP: natural gradient returns priorPenalty` (crash + 4 leaks)
  - `MAP: identity loss` (8 leaks)
  - `dlmPrior: IG on obsVar` (8 leaks)
  - `dlmPrior: IG on processVar` (8 leaks)
  - `dlmPrior: IG on both` (8 leaks)
  - `dlmPrior: IG with natural gradient` (4 leaks)
- 193 tests pass (all non-MAP tests).
- The MAP feature wraps `kalmanLoss` to add prior penalty terms. Even a trivial identity wrapper triggers the bug.

### Workaround locations

No workaround available — the wrapper is architecturally necessary for MAP/prior estimation.
Affected code: `src/mle.ts` lines 862–880 (the `lossFn` closure).

## Suggested fix

The `e034f26` drain logic in `partialEvalGraphToJaxpr` handles const PETracers for scan equations at the top level of the traced function. It needs to also handle const PETracers created by scan equations called through intermediate function closures. The closure call doesn't add a new tracing context — it's still the same `valueAndGrad` trace — but the scan's anonymous consts may be registered at a different depth in the equation list.

## Related

- `e034f26` — drain PETracer #rc in partialEvalGraphToJaxpr (PARTIAL FIX for this issue)
- `41b0bda` — anonymous const disposal guard for 2-level nesting (RESOLVED: 2-level works)
- `a9db43d` — nested jaxpr disposal (RESOLVED: zero leaks in repro)
- `99ca222` — evalJaxpr non-consuming refactor (did not fix triple-nesting)
