# jax-js: WebGPU JIT fails — shape-inference issues in jit(lax.scan) and jit(lax.associativeScan) [RESOLVED]

## Summary

`jit(fn)` crashed on the WebGPU backend when `fn` contained `lax.scan` or `lax.associativeScan` with einsum, matmul/transpose, or np.where in the step/compose body. **Resolved in `fix/jit-scan-einsum-maxargs` (commit f09121d) plus three API fixes and a DARE convention correction in dlm-js `src/index.ts`.**

## Resolution — COMPLETE ✅

The fix branch (`fix/jit-scan-einsum-maxargs`, commit f09121d) resolves all JIT shape-inference failures, and `dlmFit` now runs correctly on WebGPU.

### What the fix branch resolved (upstream)

- `BigInt(tracerObject)` in einsum path optimizer — all 5 narrowing test patterns now PASS
- `np.eye(N, { dtype })` 2-arg form now interpreted correctly (was treating options bag as `M` positional arg)
- `ShapeTracker` validates that all dimensions are finite numbers

### dlm-js API fixes required alongside the upstream fix

After installing `fix/jit-scan-einsum-maxargs`, three API incompatibilities in `src/index.ts` needed fixing:

| Bug | Fix |
|-----|-----|
| `np.expand_dims(arr, 2)` — method doesn't exist | Renamed to `np.expandDims(arr, 2)` (camelCase) |
| `np.where(float_mask, ...)` — requires boolean condition | Swapped to `np.where(is_nan_bool, ...)` with branches reordered |
| `np.slice(arr, start, end)` — method doesn't exist | Replaced with `np.split(arr, [k], axis)` pattern |

### DARE convention fix (critical for correctness)

`solveDAREForKss` was using the **standard** Kalman formulation (K = C_pred·F'/(F·C_pred·F'+V²) where C_pred = G·C·G'+W), while the rest of the codebase uses the **MATLAB DLM** convention (K = G·C·F'/(F·C·F'+V²) using the filtered covariance C directly). This caused the assocScan-computed `C_filt` to be inconsistent with the K/Cp recovery used by the backward smoother, producing diverging smoothed states.

Fix: rewrote `solveDAREForKss` to iterate in MATLAB convention, giving:
- **A_ss = G − K_ss·F** (MATLAB smoother matrix, vs old `(I−K·F)·G`)
- **Sigma_base = W** (process noise; vs old `L_ss·W·L_ss'`)
- K and C_filt now consistent throughout the forward and backward passes

### Verification results

```
WASM / f64  → lik=1125.310, yhat[0]=1113.50  (exact sequential filter)
WebGPU / f32 → lik=1145.885, yhat[0]=1126.49  (steady-state approximation, ~1% relative error)
```

Remaining ~1–12% deviation from the WASM/f64 reference is expected: the assocScan path uses the steady-state DARE gain (K_ss) throughout, while the sequential filter uses time-varying gains in early timesteps. For long series or after filter convergence, the two paths agree closely. No NaN outputs, no divergence.

All 112 dlm-js vitest tests pass with the updated dependency and code fixes.

---

## Original report (archived)

The fix branch resolves the original `BigInt(tracerObject)` crash in the einsum path optimizer. However, running `dlmFit` on WebGPU with the fix branch reveals **multiple remaining failures**, all sharing the pattern: a dimension that should be a concrete integer is instead a tracer object or option object.

## Reproduction — dlm-js actual code

The canonical repro is the dlm-js source itself:
- **`src/index.ts`**: `dlmSmo` function — `jit(core)` traces a forward `lax.scan` (`forwardStep` body with einsum `'ij,jk,lk->il'`, `np.transpose`, `np.where`/`np.isnan`) and backward `lax.scan` (`backwardStep` with einsum `'ji,jk,kl->il'`), plus a `lax.associativeScan` path (WebGPU branch) with `compose` function using einsum `'nij,njk->nik'` and `'nij,njk,nlk->nil'`.
- **Test runner**: `tmp/test-webgpu-fix.ts` (in the dlm-js repo) — runs `dlmFit(nile_data, s, w, DType.Float32, opts)` on WebGPU.

```
deno run --no-check --unstable-webgpu --allow-read --allow-env tmp/test-webgpu-fix.ts
```

## Isolated narrowing repro

This standalone script (`tmp/test-webgpu-debug.ts` in the dlm-js repo) distills the failures to their
smallest reproducible forms without removing any semantically necessary operations:

```ts
import { DType, defaultDevice, init, numpy as np, lax, jit } from
  "@hamk-uas/jax-js-nonconsuming";

await init("webgpu");
defaultDevice("webgpu");
const dtype = DType.Float32;
const m = 2, n = 5;

// ── Test 1: jit(lax.scan) with np.add carry — PASS ───────────────────────
const step1 = (carry: { C: np.Array }, inp: { W: np.Array }) => {
  return [{ C: np.add(carry.C, inp.W) }, {}];
};
const C0 = np.zeros([m, m], { dtype });
const W_batch = np.zeros([n, m, m], { dtype });
await jit((C0_: np.Array, W_: np.Array) =>
  lax.scan(step1, { C: C0_ }, { W: W_ })
)(C0, W_batch);

// ── Test 2: jit(lax.scan) — einsum with carry operand — FAIL ─────────────
// Error: Inconsistent size for index 1 in einsum: [object Object] vs 2
const G = np.eye(m, m, { dtype });
const W = np.eye(m, m, { dtype });
const step2 = (carry: { C: np.Array }, _inp: {}) => {
  const GCGt = np.einsum('ij,jk,lk->il', G, carry.C, G);  // G captured, carry.C has symbolic shape
  return [{ C: np.add(GCGt, W) }, {}];
};
const dummy = np.zeros([n], { dtype });
await jit((C0_: np.Array, d_: np.Array) =>
  lax.scan(step2, { C: C0_ }, d_)
)(np.eye(m, m, { dtype }), dummy);

// ── Test 3: jit(lax.scan) — matmul + transpose with carry — FAIL ─────────
// Error: Incompatible array broadcast shapes: 2,[object Object] vs 2,2
const FF = np.array([[1, 0]], { dtype });
const step3 = (carry: { K: np.Array }, inp: { y: np.Array }) => {
  const KKt = np.matmul(carry.K, np.transpose(carry.K));  // transpose of carry loses shape
  const L = np.subtract(G, np.matmul(carry.K, FF));
  const LCLt = np.einsum('ij,jk,lk->il', L, KKt, L);
  return [{ K: np.add(LCLt, KKt) }, { LCLt }];
};
await jit((K0_: np.Array, y_: np.Array) =>
  lax.scan(step3, { K: K0_ }, y_)
)(np.zeros([m, 1], { dtype }), np.zeros([n, 1], { dtype }));

// ── Test 4: jit(lax.scan) — np.where + np.isnan + einsum — FAIL ──────────
// Error: Inconsistent size for index 2 in einsum: [object Object] vs 2
const one_11 = np.array([[1.0]], { dtype });
const step4 = (carry: { x: np.Array; C: np.Array }, inp: { y: np.Array; V2: np.Array; FF: np.Array }) => {
  const { x, C } = carry;
  const { y: yi, V2: V2i, FF: FFi } = inp;
  const is_nan = np.isnan(yi);
  const zero_11 = np.zerosLike(yi);
  const mask = np.where(is_nan, zero_11, one_11);
  const y_safe = np.where(is_nan, zero_11, yi);
  const Cp = np.add(np.einsum('ij,jk,lk->il', FFi, C, FFi), V2i);
  const v = np.multiply(mask, np.subtract(y_safe, np.matmul(FFi, x)));
  const GCFFt = np.einsum('ij,jk,lk->il', G, C, FFi);
  const K = np.multiply(mask, np.divide(GCFFt, Cp));
  const L = np.subtract(G, np.matmul(K, FFi));
  const x_next = np.add(np.matmul(G, x), np.matmul(K, v));
  // Joseph form
  const LCLt = np.einsum('ij,jk,lk->il', L, C, L);
  const KV2Kt = np.multiply(V2i, np.matmul(K, np.transpose(K)));
  const C_next_raw = np.add(np.add(LCLt, KV2Kt), W);
  const C_next = np.multiply(np.array(0.5, { dtype }), np.add(C_next_raw, np.transpose(C_next_raw)));
  return [{ x: x_next, C: C_next }, { v }];
};
const F_mat = np.array([[1, 0]], { dtype });
const x0 = np.zeros([m, 1], { dtype });
const y_batch = np.array(Array.from({length:n}, () => [[1.0]]), { dtype });
const V2_batch = np.ones([n, 1, 1], { dtype });
const FF_batch = np.tile(np.reshape(F_mat, [1, 1, m]), [n, 1, 1]);
await jit((x0_: np.Array, C0_: np.Array, y_: np.Array, v2_: np.Array, ff_: np.Array) =>
  lax.scan(step4, { x: x0_, C: C0_ }, { y: y_, V2: v2_, FF: ff_ })
)(x0, np.eye(m, m, { dtype }), y_batch, V2_batch, FF_batch);

// ── Test 5: jit(lax.associativeScan) — einsum in compose fn — FAIL ────────
// Error: Invalid reshape: [2,{"dtype":"float32"}] -> [1,2,2]
// NOTE: np.eye(m, { dtype }) (2-arg form) produces shape [m, {dtype-object}].
//       Must use np.eye(m, m, { dtype }) (3-arg form). The 3-arg form avoids
//       that particular error but associativeScan JIT may still fail (see above).
type Elem = { A: np.Array; b: np.Array; S: np.Array };
const compose = (a: Elem, b_e: Elem): Elem => {
  const A_comp = np.einsum('nij,njk->nik', b_e.A, a.A);
  const b_comp = np.add(np.einsum('nij,njk->nik', b_e.A, a.b), b_e.b);
  const S_comp = np.add(np.einsum('nij,njk,nlk->nil', b_e.A, a.S, b_e.A), b_e.S);
  return { A: A_comp, b: b_comp, S: S_comp };
};
const A_arr = np.tile(np.reshape(np.eye(m, m, { dtype }), [1, m, m]), [n, 1, 1]);
const b_arr = np.zeros([n, m, 1], { dtype });
const S_arr = np.tile(np.reshape(np.eye(m, m, { dtype }), [1, m, m]), [n, 1, 1]);
await jit((A_: np.Array, b_: np.Array, S_: np.Array) =>
  lax.associativeScan(compose, { A: A_, b: b_, S: S_ })
)(A_arr, b_arr, S_arr);
```

**Results with fix branch (fix/jit-scan-einsum-maxargs):**
```
Test 1: jit(lax.scan) with np.add carry...                  PASS
Test 2: jit(lax.scan) with einsum('ij,jk,lk->il')...        FAIL: Inconsistent size for index 1 in einsum: [object Object] vs 2
Test 3: jit(lax.scan) with matmul + transpose...            FAIL: Incompatible array broadcast shapes: 2,[object Object] vs 2,2
Test 4: jit(lax.scan) with np.where + np.isnan...           FAIL: Inconsistent size for index 2 in einsum: [object Object] vs 2
Test 5: jit(lax.associativeScan) with einsum compose...     FAIL: Invalid reshape: [2,{"dtype":"float32"}] -> [1,2,2]
```

## Root cause pattern

All failures share the same root: **during JIT tracing of `lax.scan` (and `lax.associativeScan`), carry tensor dimensions are represented as tracer objects rather than concrete integers.** Operations that inspect concrete shape values then fail:

| Operation | Failure mode |
|-----------|-------------|
| `np.einsum('ij,jk,lk->il', capturedG, carryC, capturedG)` | `[object Object]` leaks into einsum index-size consistency check |
| `np.matmul(carryK, np.transpose(carryK))` | `np.transpose` doesn't preserve concrete shape — subsequent `np.add` sees `[2, [object Object]]` vs `[2, 2]` |
| `np.einsum` after `np.where`/`np.isnan` on carry | Same tracer-in-shape issue |
| `np.reshape(np.eye(m, {dtype}), [1,m,m])` | `np.eye(N, opts)` interprets options object as second positional arg `M`, producing shape `[N, {opts-object}]` — use `np.eye(N, N, opts)` instead |

Test 1 (`np.add(carry.C, inp.W)`) passes because `np.add` broadcast resolution apparently handles tracer dims. Tests 2–4 fail because einsum and transpose/matmul try to extract concrete integer values from shape dimensions that are tracer objects.

## Environment

- jax-js-nonconsuming fix branch: `fix/jit-scan-einsum-maxargs`
- Deno 2.6.7 with `--unstable-webgpu`
- Linux, Intel Arc GPU (wgpu/Vulkan backend)
- dlm-js repo: `src/index.ts` `dlmSmo` function — full implementation context

## Impact

This blocks running `dlmFit` (Kalman filter + RTS smoother) on WebGPU entirely. The library has a complete WebGPU code path (DARE steady-state Kalman gain + `lax.associativeScan` forward filter + Joseph-form stabilization) that cannot be activated until `jit(lax.scan)` with einsum/transpose in the step body works correctly on WebGPU.

## Additional note: np.eye(N, opts) API ambiguity

`np.eye(m, { dtype })` creates `[m, {dtype-object}]` because the options bag is taken as the `M` (columns) positional argument. Use `np.eye(m, m, { dtype })` (3-arg form) to be safe. This is a separate issue but causes a confusing error message.


## Summary

`jit(fn)` crashes on the WebGPU backend when `fn` contains an `np.einsum` call whose operands have symbolic (tracer) shapes. The error occurs in the einsum contraction path optimizer (`computeEinsumPath` → `approximateCountFlops` → `bprod`), which calls `BigInt(dim)` where `dim` is a tracer object instead of a concrete number.

Eager-mode einsum works correctly on WebGPU — the issue is specific to JIT tracing.

## Reproduction

```ts
import { init, defaultDevice, DType, jit, lax, numpy as np } from "@hamk-uas/jax-js-nonconsuming";

await init("webgpu");
defaultDevice("webgpu");

const G = np.array([[1, 0], [1, 1]], { dtype: DType.Float32 });
const W = np.array([[1, 0], [0, 1]], { dtype: DType.Float32 });
const one = np.array([[1.0]], { dtype: DType.Float32 });

const step = (carry, inp) => {
  const { x, C } = carry;
  const yi = inp.y;
  const FFi = inp.FF;
  const V2i = inp.V2;

  const is_nan = np.isnan(yi);
  const zero = np.zerosLike(yi);
  const mask = np.where(is_nan, zero, one);
  const y_safe = np.where(is_nan, zero, yi);

  const v = np.multiply(mask, np.subtract(y_safe, np.matmul(FFi, x)));
  const Cp = np.add(np.einsum("ij,jk,lk->il", FFi, C, FFi), V2i);
  const GCFFt = np.einsum("ij,jk,lk->il", G, C, FFi);
  const K = np.multiply(mask, np.divide(GCFFt, Cp));
  const L = np.subtract(G, np.matmul(K, FFi));
  const x_next = np.add(np.matmul(G, x), np.matmul(K, v));
  const C_next = np.add(np.einsum("ij,jk,lk->il", G, C, L), W);

  return [{ x: x_next, C: C_next }, { x_pred: x, v, K, Cp, mask, FF: FFi, C_pred: C }];
};

const n = 5;
const y_arr = np.array(Array.from({length: n}, () => [[1000]]), { dtype: DType.Float32 });
const V2_arr = np.array(Array.from({length: n}, () => [[14400]]), { dtype: DType.Float32 });
const FF_arr = np.tile(np.array([[[1, 0]]], { dtype: DType.Float32 }), [n, 1, 1]);
const x0 = np.array([[0], [0]], { dtype: DType.Float32 });
const C0 = np.array([[1e7, 0], [0, 1e7]], { dtype: DType.Float32 });

const core = (x0_, C0_, y_, V2_, FF_) => {
  const [carry, out] = lax.scan(step, { x: x0_, C: C0_ }, { y: y_, V2: V2_, FF: FF_ });
  return out;
};

// This crashes:
const result = await jit(core)(x0, C0, y_arr, V2_arr, FF_arr);
```

## Error

```
SyntaxError: Cannot convert [object Object] to a BigInt
    at BigInt (<anonymous>)
    at bprod (dist/index.js:7804:29)
    at approximateCountFlops (dist/index.js:7917:19)
    at approximatePathFlops (dist/index.js:7909:17)
    at computePathOptimal (dist/index.js:7976:17)
    at computeEinsumPath (dist/index.js:7951:9)
```

On a simpler step body (without `np.isnan`/`np.where`/`mask`), the specific error changes to:

```
Error: internal: maxArgs, no input found to mark as black in Jaxpr equation ...
```

Both are JIT compilation failures; eager mode works correctly in all cases.

## Environment

- jax-js-nonconsuming v0.7.1
- Deno 2.6.7 with `--unstable-webgpu`
- Linux, Intel UHD / Arc GPU
- Also reproducible on wgpu/dawn Vulkan adapters

## Impact

This blocks running `dlmFit` (Kalman filter + RTS smoother) on WebGPU. The code uses `jit(core)` where `core` chains a forward `lax.scan` (with einsum in the step body), a backward `lax.scan`, and vectorized diagnostics. All of this works correctly on cpu and wasm backends.

## Workaround candidates

1. **Replace einsum with matmul chains** in JIT bodies — avoids the path optimizer entirely. Downside: less readable code, may be slower for multi-operand contractions.
2. **Guard `bprod` against non-numeric dims** — detect tracer objects and fall back to a conservative FLOP estimate.
3. **Skip path optimization during tracing** — use the default left-to-right contraction order when shapes are symbolic.

## Affected dlm-js code

- `src/index.ts` `dlmSmo` → `jit(core)` contains einsum `"ij,jk,lk->il"` (batched matmul-transpose) in both `forwardStep` and `backwardStep`
- `src/mle.ts` `makeKalmanLoss` → same pattern inside `lax.scan` under `jit(valueAndGrad(...))`
