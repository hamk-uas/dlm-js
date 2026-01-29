/**
 * lax.scan implementation for jax-js
 *
 * Scans a function over leading axis of input arrays while carrying state.
 * This enables expressing recurrences (like Kalman filters) in a functional style.
 *
 * NOTE: This is a pure JS implementation for CPU. It is NOT yet JIT-compilable.
 * For JIT support, this would need to be implemented as a primitive in jax-js core.
 */

import { numpy as np, tree, type JsTree } from "@jax-js/jax";

// ============================================================================
// Scan implementation
// ============================================================================

/**
 * Scan a function over the leading axis of input arrays.
 *
 * @param f - Step function: (carry, x) => [newCarry, output]
 * @param init - Initial carry value (pytree)
 * @param xs - Input sequence (pytree, each leaf is an array to scan over axis 0)
 * @param length - Optional length override (for when xs is null)
 * @returns [finalCarry, stackedOutputs] - Final carry and outputs stacked along axis 0
 *
 * @example
 * ```ts
 * // Cumulative sum
 * const step = (carry, x) => {
 *   const sum = np.add(carry, x);
 *   return [sum, sum.ref];
 * };
 * const [final, sums] = await scan(step, np.array([0.0]), np.array([1, 2, 3, 4, 5]));
 * // final = [15.0], sums = [1, 3, 6, 10, 15]
 * ```
 */
export async function naiveScan<Carry extends JsTree<np.Array>, X extends JsTree<np.Array>, Y extends JsTree<np.Array>>(
  f: (carry: Carry, x: X) => [Carry, Y],
  init: Carry,
  xs: X,
  length?: number
): Promise<[Carry, Y]> {
  // Determine scan length from input
  const xsLeaves = tree.leaves(xs);
  const n = length ?? (xsLeaves.length > 0 ? xsLeaves[0].shape[0] : 0);

  if (n === 0) {
    throw new Error('scan: cannot determine length from empty inputs');
  }

  // Slice input pytree at index i (preserves original with tree.ref)
  const sliceAt = (t: X, i: number): X => {
    return tree.map((leaf: np.Array) => leaf.ref.slice(i), tree.ref(t)) as X;
  };

  // Accumulate outputs
  const outputsList: Y[] = [];
  let carry = init;

  for (let i = 0; i < n; i++) {
    const x = sliceAt(xs, i);
    const [newCarry, output] = f(carry, x);
    carry = newCarry;
    outputsList.push(output);
  }

  // Stack outputs along axis 0
  const stackedOutputs = stackPyTree(outputsList);

  return [carry, stackedOutputs];
}

/**
 * Stack a list of pytrees along a new leading axis.
 * Each pytree in the list must have the same structure.
 */
function stackPyTree<T extends JsTree<np.Array>>(trees: T[]): T {
  if (trees.length === 0) {
    throw new Error('stackPyTree: empty list');
  }

  const [firstLeaves, treedef] = tree.flatten(trees[0]);
  const allLeaves = trees.map((t) => tree.leaves(t));

  // Number of leaves per tree
  const numLeaves = firstLeaves.length;

  // Stack each leaf position across all trees
  const stackedLeaves: np.Array[] = [];
  for (let leafIdx = 0; leafIdx < numLeaves; leafIdx++) {
    const toStack = allLeaves.map((leaves) => leaves[leafIdx].ref);
    // Use np.stack to combine along new axis 0
    const stacked = np.stack(toStack, 0);
    stackedLeaves.push(stacked);
  }

  // Reconstruct pytree with stacked leaves
  return tree.unflatten(treedef, stackedLeaves) as T;
}

/**
 * Reverse scan - scan from end to beginning.
 * Useful for backward passes like RTS smoother.
 */
export async function naiveScanReverse<Carry extends JsTree<np.Array>, X extends JsTree<np.Array>, Y extends JsTree<np.Array>>(
  f: (carry: Carry, x: X) => [Carry, Y],
  init: Carry,
  xs: X,
  length?: number
): Promise<[Carry, Y]> {
  const xsLeaves = tree.leaves(xs);
  const n = length ?? (xsLeaves.length > 0 ? xsLeaves[0].shape[0] : 0);

  if (n === 0) {
    throw new Error('scanReverse: cannot determine length from empty inputs');
  }

  // Slice input pytree at index i (preserves original with tree.ref)
  const sliceAt = (t: X, i: number): X => {
    return tree.map((leaf: np.Array) => leaf.ref.slice(i), tree.ref(t)) as X;
  };

  // Accumulate outputs (in reverse order, then flip at end)
  const outputsList: Y[] = [];
  let carry = init;

  for (let i = n - 1; i >= 0; i--) {
    const x = sliceAt(xs, i);
    const [newCarry, output] = f(carry, x);
    carry = newCarry;
    outputsList.push(output);
  }

  // Reverse to get correct order
  outputsList.reverse();

  // Stack outputs along axis 0
  const stackedOutputs = stackPyTree(outputsList);

  return [carry, stackedOutputs];
}

// Export stackPyTree for testing
export { stackPyTree };
