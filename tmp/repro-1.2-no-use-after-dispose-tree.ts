/**
 * Repro: no-use-after-dispose on tree.dispose(arg) — FIXED in fbcadc2
 *
 * Expected: 0 warnings — tree.dispose(x) disposes `x`, not `tree`
 * Actual (before fbcadc2): tree.makeDisposable flagged as "tree used after dispose"
 * Actual (after fbcadc2):  0 warnings ✓
 *
 * Setup: place in a directory matched by the eslint-plugin config
 * Run:   npx eslint <this-file>
 */
import { numpy as np, tree } from "@jax-js/jax";

// ── Case 1: Back-to-back tree.dispose calls — FALSE POSITIVE ────────────
function disposeMultiple(a: np.Array, b: np.Array) {
  const objA = { x: a };
  const objB = { x: b };

  tree.dispose(objA);     // linter interprets this as: `tree` is now disposed
  tree.dispose(objB);     // OK — .dispose parent is exempted; but updates dispose line
  tree.makeDisposable(objB); // ⚠ FLAGGED: `tree` used after dispose on line 18
}

// ── Case 2: tree.dispose then tree.makeDisposable — FALSE POSITIVE ──────
function disposeThenMake(a: np.Array) {
  const old = { x: a };
  tree.dispose(old);               // marks `tree` as "disposed"

  const result = { y: np.add(a, a) };
  return tree.makeDisposable(result);  // ⚠ FLAGGED: `tree` used after dispose
}

void disposeMultiple; void disposeThenMake;
