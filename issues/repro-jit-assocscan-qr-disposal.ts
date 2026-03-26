/**
 * Repro: UseAfterFreeError on int32 arrays when `np.linalg.qr` is called
 * inside `jit()` on matrices larger than QR_UNROLL_LIMIT (8).
 *
 * Root cause: `householderQR2D` in lax-linalg.ts creates `arange_m` and
 * `arange_n` (int32 arrays), uses them inside a `foriLoop` body (which
 * captures them as jaxpr constants), then explicitly disposes them after
 * foriLoop returns. Under the v0.9.5 ownership restructuring (explicit
 * creation-ref balancing, commit 1967dddc), this disposal frees the backing
 * data while the ClosedJaxpr still references it.
 *
 * - m ≤ 8: QR takes the unrolled path → no foriLoop → no jaxpr capture → OK
 * - m > 8: QR takes the foriLoop path → arange captured as const → disposed → FAIL
 *
 * Passes on v0.9.3. Fails on v0.9.5+.
 *
 * Run: npx tsx issues/repro-jit-assocscan-qr-disposal.ts
 */
import { init, numpy as np, jit, defaultDevice, checkLeaks } from '@hamk-uas/jax-js-nonconsuming';

await init('wasm');
defaultDevice('wasm');
checkLeaks.start();

// ── Minimal repro: QR on m×m matrix inside jit, m > 8 ──
const m = 9; // > QR_UNROLL_LIMIT=8 → foriLoop path

const core = () => {
  using A = np.eye(m, undefined, { dtype: 'float64' });
  const [Q, R] = np.linalg.qr(A);
  Q.dispose();
  return R;
};

try {
  const result = await jit(core)();
  console.log('OK — QR inside jit, m=%d, shape:', m, result.shape);
  result.dispose();
} catch (e) {
  console.error('FAIL (m=%d):', m, (e as Error).message);
  process.exitCode = 1;
}

checkLeaks.stop();

checkLeaks.stop();
