/**
 * Repro: scan result disposal tracking with require-scan-result-dispose
 *
 * Expected: linter warns when `carry` / `ys` from `lax.scan(...)` are never disposed
 * Actual:   2 warnings on leakyScanResult (carry, ys) ✓
 *
 * This is the common leak shape in non-consuming mode:
 *   const [carry, ys] = lax.scan(...)
 * If neither `carry` nor `ys` is disposed, their array leaves leak.
 *
 * Setup: place in a directory matched by the eslint-plugin config
 * Run:   npx eslint <this-file>
 */
import { numpy as np, lax, tree } from "@jax-js/jax";

type Carry = { x: np.Array; C: np.Array };
type StepOut = { pred: np.Array; gain: np.Array };

declare function forwardStep(carry: Carry, inp: np.Array): [Carry, StepOut];
declare const x0: np.Array;
declare const C0: np.Array;
declare const xs: np.Array;

// ── Minimal leak repro (warns on both bindings) ─────────────────────────
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

// Lint result: leakyScanResult => 2 warnings (carry, ys); preferredScanPattern => 0 warnings.
void leakyScanResult;
void preferredScanPattern;
