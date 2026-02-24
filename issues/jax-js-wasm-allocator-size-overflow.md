# jax-js-wasm-allocator-size-overflow

- Status: 🔴 Open
- Summary: WASM bump allocator overflows a signed 32-bit integer in its page-count calculation when a single JIT workload requires more than ~2 GiB of working memory. `WebAssembly.Memory.grow()` is called with a negative page count, producing the error "Argument 0 must be non-negative" instead of a clean OOM.
- Affected version: commit `297f93a4ebb0006fbaeecbe9929a915a0c47025f` (2 GB WASM memory limit, released post-65cb449)
- Root cause hypothesis: The allocator computes `pages = Math.ceil(bytes / 65536)` (or similar) using a JavaScript variable that is correct as a double-precision float but then passes the result to a WASM `Memory.grow()` binding that coerces to signed 32-bit int (i32). When `bytes` exceeds 2^31 (≈ 2.15 GiB), the i32 coercion wraps to a negative value, triggering the Deno/V8 argument validation.

## Reproduction

```
⚠ N=3,276,800 failed: WebAssembly.Memory.grow(): Argument 0 must be non-negative
```

Run the dlm-js scaling benchmark (requires Deno, commits through `297f93a`):

```ts
// issues/repro-wasm-allocator-size-overflow.ts
import { defaultDevice, init } from "../node_modules/@hamk-uas/jax-js-nonconsuming/dist/index.js";
import { dlmFit } from "../src/index.ts";

const N = 3_276_800;
const baseY = Array.from({ length: 100 }, (_, i) => i + 1);
const y = Array.from({ length: N }, (_, i) => baseY[i % 100]);

await init("wasm");
defaultDevice("wasm");

// Warmup at small N so allocator is in fresh state
const r0 = await dlmFit(y.slice(0, 100), { obsStd: 100, processStd: [50], order: 1, dtype: 'f64' });
r0[Symbol.dispose]?.();

// This call requires ~4–6 GiB of working memory → overflows signed 32-bit page count
try {
  const r1 = await dlmFit(y, { obsStd: 100, processStd: [50], order: 1, dtype: 'f64' });
  r1[Symbol.dispose]?.();
  console.log("Success (unexpected)");
} catch (e) {
  console.error("Error (expected):", (e as Error).message);
  // Expected: "WebAssembly.Memory.grow(): Argument 0 must be non-negative"
}
```

Run with:
```bash
/home/olli/.deno/bin/deno run --unstable-webgpu --allow-read --allow-write --allow-env issues/repro-wasm-allocator-size-overflow.ts
```

## Memory budget at N=3,276,800 (Nile order=1, m=2, f64)

Named arrays (output of two lax.scan passes):

| Array | Shape | Bytes |
|-------|-------|-------|
| y | 3.3M × 1 | 26 MB |
| x_filt, x_pred, x_smo | 3.3M × 2 (×3) | 157 MB |
| C_filt, C_pred, C_smo | 3.3M × 2×2 (×3) | 315 MB |
| innovations, S, K | 3.3M × … | ~200 MB |

**Named arrays ≈ 700 MB.** JIT intermediates (lax.scan unrolls + autodiff buffers) add 3–6× overhead, driving total to **2.5–4.5 GiB** — above the 2 GiB limit.

The issue manifests even on a cold call (first time at that shape), so it is not an accumulation issue like the prior OOM bugs. The error message ("Argument 0 must be non-negative") strongly suggests a signed integer overflow in the page-count path rather than a clean `Memory.grow()` returning -1.

## Impact on dlm-js

- `bench:scaling` fails at N=3,276,800. Maximum measurable N is 1,638,400 (~1.6 ms × 8 bytes/val × overhead).
- Workaround: cap `N_ALL` / `N_GPU` at 1,638,400. No in-process fix possible.

## Suggested fix

Replace the `Memory.grow(pages)` call site with an unsigned 32-bit saturating cast:
```js
const pages32 = pages >>> 0;  // coerce to uint32; if pages > 2^32 it clamps to 0 (detectable as error)
if (pages32 !== pages || pages32 === 0) throw new RangeError(`WASM allocation too large: ${bytes} bytes`);
const result = memory.grow(pages32);
```
Or, raise the internal byte-size tracking to a `BigInt` / `Number` check before the cast:
```js
if (bytes > 0x7FFF_FFFF) throw new RangeError(`WASM allocation exceeds 2 GiB: ${bytes} bytes`);
```

Either approach produces a clear, actionable error instead of the confusing "Argument 0 must be non-negative" message.
