/**
 * Repro: require-using numpy long-form function names — FIXED in fbcadc2
 *
 * Expected: ALL of a, b, c, d, e, f flagged (all are intermediates, not returned/disposed)
 * Actual (before fbcadc2): only `a`, `b`, `f` flagged; `c`, `d`, `e` missed
 * Actual (after fbcadc2):  all 6 flagged ✓
 *
 * Setup: place in a directory matched by the eslint-plugin config
 * Run:   npx eslint <this-file>
 */
import { numpy as np } from "@jax-js/jax";

function leakyFunction(x: np.Array, y: np.Array): number {
  // These ARE flagged (names are in the lists):
  const a = np.add(x, y);           // ⚠ FLAGGED  ← "add" is in ARRAY_METHODS ✓
  const b = np.einsum('ij->ji', x); // ⚠ FLAGGED  ← "einsum" is in ARRAY_FACTORIES ✓

  // These are NOT flagged (names missing from lists):
  const c = np.subtract(x, y);      // ✗ SILENT  ← "subtract" not in any list
  const d = np.multiply(x, y);      // ✗ SILENT  ← "multiply" not in any list
  const e = np.divide(x, y);        // ✗ SILENT  ← "divide" not in any list

  // This IS flagged (long form IS in list for matmul):
  const f = np.matmul(x, y);        // ⚠ FLAGGED  ← "matmul" is in ARRAY_METHODS ✓

  void a; void b; void c; void d; void e; void f;
  return 0;
}

/**
 * Suggested fix: add long-form names to ARRAY_METHODS or ARRAY_FACTORIES:
 *   "subtract", "multiply", "divide", "power",
 *   "trueDivide", "floorDivide", "remainder",
 *   "negative" (alias of "neg"),
 *   "positive", "logaddexp", "logaddexp2",
 *   "reciprocal", "sign", "floor", "ceil", "trunc", "round"
 *
 * Also verify all `np.*` exports are covered by cross-referencing
 * the numpy module's export list.
 */
void leakyFunction;
