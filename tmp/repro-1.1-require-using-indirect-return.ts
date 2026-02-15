/**
 * Repro: require-using indirect return — FIXED in fbcadc2
 *
 * Expected: 0 warnings — a, b, pred, gain all escape via return
 * Actual (before fbcadc2): `a`, `b`, `pred`, `gain` flagged in Cases 2 & 3
 * Actual (after fbcadc2):  0 warnings ✓
 *
 * Setup: place in a directory matched by the eslint-plugin config
 * Run:   npx eslint <this-file>
 */
import { numpy as np } from "@jax-js/jax";

// ── Case 1: Direct return — OK ─────────────────────────────────────────
function directReturn(x: np.Array): { a: np.Array; b: np.Array } {
  const a = np.add(x, x);      // no warning ✓ (linter sees `a` in return)
  const b = np.add(x, x);      // no warning ✓
  return { a, b };
}

// ── Case 2: Indirect return via variable — FALSE POSITIVE ───────────────
function indirectReturn(x: np.Array): { a: np.Array; b: np.Array } {
  const a = np.add(x, x);      // ⚠ FLAGGED — wants `using`
  const b = np.add(x, x);      // ⚠ FLAGGED — wants `using`
  const output = { a, b };     // a and b escape into output…
  return output;               // …which is returned, but linter doesn't trace this
}

// ── Case 3: Indirect return in tuple (lax.scan callback shape) ──────────
function scanCallback(
  carry: np.Array,
  x: np.Array,
): [{ x: np.Array }, { pred: np.Array; gain: np.Array }] {
  const pred = np.add(x, carry);          // ⚠ FLAGGED
  const gain = np.einsum('ij,jk->ik', x, carry); // ⚠ FLAGGED
  const newCarry = { x: np.add(x, carry) };
  const output = { pred, gain };
  return [newCarry, output];              // all escape via return
}

// ── Workaround: inline into return ──────────────────────────────────────
function inlinedReturn(x: np.Array): { a: np.Array; b: np.Array } {
  const a = np.add(x, x);      // no warning ✓ (linter sees `a` in return)
  const b = np.add(x, x);      // no warning ✓
  return { a, b };              // same as Case 1 — works fine
}

void directReturn; void indirectReturn; void scanCallback; void inlinedReturn;
