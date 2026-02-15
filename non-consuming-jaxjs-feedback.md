# Feedback: non-consuming jax-js & @jax-js/eslint-plugin

Migration report from **dlm-js** — a Kalman filter + RTS smoother library (~430 lines TypeScript) using `lax.scan` and `jit`.

Migrated from `feat/scan-v2` (move semantics, `.ref`) → `feat/non-consuming-ops` (`using` + `.dispose()`).

**Updated** after adopting jax-js commits `66aefd2`, `fbcadc2`, `2eecf95`, and `44b65ea`.

Standalone repro files in [`tmp/`](tmp/) — each can be linted directly by placing in a directory covered by the `@jax-js/eslint-plugin` recommended config.

---

## 1  Linter issues

### 1.1  ✅ `require-using` false positive: indirect return via intermediate variable — Fixed in `fbcadc2`

Single-hop indirect return detection via `isAssignedToReturnedVariable` now correctly suppresses the warning.

**Repro** ([`tmp/repro-1.1-require-using-indirect-return.ts`](tmp/repro-1.1-require-using-indirect-return.ts)):

```typescript
import { numpy as np } from "@jax-js/jax";

// ── Case 1: Direct return — OK ─────────────────────────────────────────
function directReturn(x: np.Array): { a: np.Array; b: np.Array } {
  const a = np.add(x, x);      // no warning ✓
  const b = np.add(x, x);      // no warning ✓
  return { a, b };
}

// ── Case 2: Indirect return via variable — was FALSE POSITIVE, now OK ───
function indirectReturn(x: np.Array): { a: np.Array; b: np.Array } {
  const a = np.add(x, x);      // no warning ✓ (was ⚠ before fbcadc2)
  const b = np.add(x, x);      // no warning ✓
  const output = { a, b };
  return output;
}

// ── Case 3: Indirect return in tuple (lax.scan callback shape) — now OK ─
function scanCallback(
  carry: np.Array,
  x: np.Array,
): [{ x: np.Array }, { pred: np.Array; gain: np.Array }] {
  const pred = np.add(x, carry);          // no warning ✓ (was ⚠ before)
  const gain = np.einsum('ij,jk->ik', x, carry); // no warning ✓
  const newCarry = { x: np.add(x, carry) };
  const output = { pred, gain };
  return [newCarry, output];
}
```

**Lint output**: 0 problems ✓

---

### 1.2  ✅ `no-use-after-dispose` false positive on `tree.dispose(arg)` — Fixed in `fbcadc2`

The rule now skips receiver marking when `.dispose()` has arguments: `tree.dispose(obj)` disposes `obj`, not `tree`.

**Repro** ([`tmp/repro-1.2-no-use-after-dispose-tree.ts`](tmp/repro-1.2-no-use-after-dispose-tree.ts)):

```typescript
import { numpy as np, tree } from "@jax-js/jax";

function disposeMultiple(a: np.Array, b: np.Array) {
  const objA = { x: a };
  const objB = { x: b };

  tree.dispose(objA);        // no warning ✓ (was marking `tree` as disposed)
  tree.dispose(objB);        // no warning ✓
  tree.makeDisposable(objB); // no warning ✓ (was ⚠ "tree used after dispose")
}

function disposeThenMake(a: np.Array) {
  const old = { x: a };
  tree.dispose(old);
  const result = { y: np.add(a, a) };
  return tree.makeDisposable(result);  // no warning ✓
}
```

**Lint output**: 0 problems ✓

**Note**: We reverted our destructuring workaround (`const { dispose: disposeTree } = tree`) back to direct `tree.dispose()` / `tree.makeDisposable()` calls. Cleaner.

---

### 1.3  ✅ `require-using` missing numpy long-form function names — Fixed in `fbcadc2`

~45 numpy long-form aliases added to `ARRAY_FACTORIES` (`subtract`, `multiply`, `divide`, `trueDivide`, `negative`, `reciprocal`, etc.).

**Repro** ([`tmp/repro-1.4-missing-function-names.ts`](tmp/repro-1.4-missing-function-names.ts)):

```typescript
import { numpy as np } from "@jax-js/jax";

function leakyFunction(x: np.Array, y: np.Array): number {
  const a = np.add(x, y);           // ⚠ FLAGGED ✓
  const b = np.einsum('ij->ji', x); // ⚠ FLAGGED ✓
  const c = np.subtract(x, y);      // ⚠ FLAGGED ✓ (was ✗ SILENT before)
  const d = np.multiply(x, y);      // ⚠ FLAGGED ✓ (was ✗ SILENT before)
  const e = np.divide(x, y);        // ⚠ FLAGGED ✓ (was ✗ SILENT before)
  const f = np.matmul(x, y);        // ⚠ FLAGGED ✓

  void a; void b; void c; void d; void e; void f;
  return 0;
}
```

**Lint output**: 6 of 6 flagged ✓ (was 3 of 6 before `fbcadc2`)

---

### 1.4  ✅ scan result disposal tracking — Fixed in `44b65ea` (+ `2eecf95`)

`lax.scan` result bindings are now checked by `jax-js/require-scan-result-dispose`. Leaky destructured scan outputs are flagged unless they are disposed or returned.

**Repro** ([`tmp/repro-1.3-object-property-arrays.ts`](tmp/repro-1.3-object-property-arrays.ts)):

```typescript
import { numpy as np, lax, tree } from "@jax-js/jax";

type Carry = { x: np.Array; C: np.Array };
type StepOut = { pred: np.Array; gain: np.Array };

declare function forwardStep(carry: Carry, inp: np.Array): [Carry, StepOut];
declare const x0: np.Array, C0: np.Array, xs: np.Array;

// ── Minimal leak repro (now flagged) ───────────────────────────────────
function leakyScanResult() {
  const [carry, ys] = lax.scan(forwardStep, { x: x0, C: C0 }, xs);
  const result = np.add(ys.pred, ys.pred);
  void carry;
  return result;
}

// ── Preferred non-consuming pattern: explicit ownership + finally ───────
function preferredScanPattern() {
  const [carry, ys] = lax.scan(forwardStep, { x: x0, C: C0 }, xs);
  tree.dispose(carry);
  try {
    return np.add(ys.pred, ys.pred);
  } finally {
    tree.dispose(ys);
  }
}
```

**Lint output**:
```
26:9  warning  `carry` comes from `lax.scan(...)` and should be disposed ...  jax-js/require-scan-result-dispose
26:9  warning  `ys` comes from `lax.scan(...)` and should be disposed ...     jax-js/require-scan-result-dispose
```

`leakyScanResult` is flagged; `preferredScanPattern` is clean.

This closes the most dangerous remaining leak gap from our migration.

`2eecf95` also improves transitive alias escape handling for pytrees, reducing false positives in alias-heavy code.

**Status**: Resolved ✅

---

## 2  API & library suggestions — Status after `66aefd2`

### 2.1  ✅ `tree.makeDisposable` — Shipped, works great

Replaced our local `makeDisposable` utility. The "block scope + using" pattern for temporary results works perfectly:

```typescript
let x0, x1;
{
  using out1 = await dlmSmo(y, F, V, x0_data, G, W, C0_data);
  x0 = (await out1.x_0.consumeData())[0];
  x1 = (await out1.x_1.consumeData())[0];
}
// all np.Array properties in out1 auto-disposed at block end
```

### 2.2  ✅ `Array.consumeData()` — Shipped, works great

Replaced manual "read + dispose" helpers:

```typescript
// Before: two steps
const result = new FA(await a.data() as ArrayLike<number>);
a.dispose();

// After: one call
const result = new FA(await a.consumeData() as ArrayLike<number>);
```

### 2.3  ✅ `tree.dispose()` — Shipped, works great

Replaced per-property disposal of scan carries and outputs:

```typescript
// Before
fwdCarry.x.dispose();
fwdCarry.C.dispose();

// After
tree.dispose(fwdCarry);
```

### 2.4  Still open: Disposable scan carries / scan results

`lax.scan` results still don't implement `Disposable`. If scan returned Disposable wrappers, we could write:

```typescript
using [carry, ys] = lax.scan(f, init, xs);
// carry and ys auto-disposed at scope end
```

**Priority**: Low — `tree.dispose()` is adequate.

---

## 3  Migration experience — what went well

| Metric | Before (move semantics) | After (non-consuming) |
|--------|------------------------|----------------------|
| `.ref` calls | 30+ | **0** |
| `disposeAll(...)` calls | 5 | **0** |
| Local helper utilities | `disposeAll`, `makeDisposable` | **0** (both in jax-js now) |
| Lint suppression comments | 1 | **0** |
| Workaround code | — | **0** (all reverted after `fbcadc2`) |
| Cognitive load per expression | "who owns this now?" | "is this a temp I should `using`?" |
| Test results | 2/2 pass | 2/2 pass (identical outputs) |

**The `.ref` elimination is transformative.** In matrix-heavy code, virtually every array participates in multiple expressions. `.ref` was needed on nearly every binding, making the code look like C++ `shared_ptr` bookkeeping. The non-consuming model removes that entire category of concern.

**`using` reads as a declaration of intent** — "this is a temporary" — much clearer than calling `disposeAll(a, b, c)` at block end where you mentally match disposal lists to creation sites.

**Inside `jit()`, `using` on tracers being a no-op is a good design choice.** It lets us write uniform code that's correct in both eager and JIT paths.

**`tree.dispose()` + `tree.makeDisposable()` + `consumeData()` eliminated all local workaround utilities.** The library now provides everything we need.

---

## 4  Migration experience — what was tricky

1. **`.data()` behaviour change** — In the old model, `.data()` auto-disposed. In non-consuming mode it doesn't. Easy to miss if migrating mechanically. `consumeData()` now provides the old semantics when desired.

2. **Returned values must NOT have `using`** — The `using` keyword disposes at scope end, which destroys returned values. Intuitive once understood, but not obvious to someone coming from `.ref`.

3. **Object property arrays were initially invisible to the linter** — now addressed via `require-scan-result-dispose` for destructured `lax.scan` outputs (§1.4).

---

## 5  Documentation suggestions

### 5.1  "Disposal patterns" guide — now partially covered in copilot-instructions

The commits added patterns and migration guide to `copilot-instructions.md`. Consider also adding to user-facing docs:

- **`tree.dispose()` for scan results** — the most common disposal site
- **Block scope + `using` for early disposal** — `{ using r = tree.makeDisposable(await f()); extract; }`
- **`consumeData()` for the JS boundary** — replaces the `.data()` + `.dispose()` dance

### 5.2  JIT output pytree aliasing guarantee

Document that when a JIT function returns the same tracer under multiple output keys (e.g., `{ xf_0, yhat: xf_0 }`), the materialised result contains independent `np.Array` instances (one per key). This is important for consumers that dispose arrays individually or use `tree.dispose()` on the result.

---

## 6  Updated summary — what's left

After `44b65ea`, all linter bugs from our feedback are resolved.

| # | Issue | Type | Priority | Repro |
|---|-------|------|----------|-------|
| 1 | Disposable scan results | Feature request | **Low** | — |
| 2 | Documentation | Docs | **Low** | — |

### Resolved in `44b65ea` + `2eecf95`

| Issue | Fix |
|-------|-----|
| `lax.scan` destructured outputs could leak silently | New `jax-js/require-scan-result-dispose` rule (wired into recommended config) |
| Pytree alias escape tracking too shallow | Transitive alias escape tracking in `require-using` |

### Resolved in `fbcadc2`

| Issue | Fix |
|-------|-----|
| `np.subtract`/`multiply`/`divide` invisible to linter | Added ~45 numpy long-form aliases to `ARRAY_FACTORIES` |
| `tree.dispose(arg)` marks `tree` as disposed | Skip receiver marking when `.dispose()` has arguments |
| Indirect returns not traced | Single-hop `isAssignedToReturnedVariable` detection |
| `tree` destructuring workaround in docs | Added to migration guide |

### Resolved in `66aefd2`

| Issue | Fix |
|-------|-----|
| No `makeDisposable` utility | `tree.makeDisposable()` shipped |
| No "read + dispose" helper | `Array.consumeData()` shipped |
| No pytree disposal utility | `tree.dispose()` shipped |
| Improved `require-using` coverage | Deep recursive return/yield search, expanded function lists |
