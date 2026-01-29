/**
 * Feature Request: jax.lax.scan for sequential computations
 *
 * Use case: Kalman filter / state-space models where each timestep depends
 * on the previous state. Currently impossible to JIT because JS loops with
 * await break the computation graph.
 *
 * Requested API (matching JAX Python):
 *
 *   lax.scan(f, init, xs, length?) -> [finalCarry, stackedOutputs]
 *
 * Where:
 *   - f: (carry, x) => [newCarry, output]
 *   - init: initial carry (pytree)
 *   - xs: input sequence (pytree of arrays, scanned over leading axis)
 *   - Returns: [final carry, stacked outputs]
 */

import { DType, numpy as np, jit } from "@jax-js/jax";

// ============================================================================
// CURRENT APPROACH (cannot be JIT'd)
// ============================================================================

/**
 * Simple recurrence: x(t) = A * x(t-1) + b * y(t)
 * Must await each step, breaking JIT traceability.
 */
async function currentApproach(
  y: Float64Array,    // Observations [n]
  A: np.Array,        // Transition matrix [m, m]
  b: np.Array,        // Input gain [m, 1]
  x0: np.Array        // Initial state [m, 1]
): Promise<{ x: np.Array[]; v: Float64Array }> {
  const n = y.length;
  const x: np.Array[] = new Array(n);
  const v = new Float64Array(n);

  x[0] = x0;

  for (let i = 0; i < n; i++) {
    // Innovation
    const vi = y[i] - (await x[i].ref.data())[0];  // <-- await breaks JIT
    v[i] = vi;

    // State update
    if (i < n - 1) {
      x[i + 1] = np.add(
        np.matmul(A.ref, x[i].ref),
        np.multiply(b.ref, vi)
      );
    }
  }

  return { x, v };
}

// ============================================================================
// DESIRED APPROACH with lax.scan
// ============================================================================

/**
 * Same recurrence, but expressible as a pure function suitable for JIT.
 *
 * The step function takes:
 *   - carry: the state that propagates (pytree)
 *   - x: one slice of the input sequence (pytree)
 * And returns:
 *   - newCarry: updated state
 *   - output: values to accumulate (pytree)
 */
function desiredApproach() {
  const dtype = DType.Float64;

  // System matrices (would be closed over or passed as static args)
  const A = np.array([[0.9, 0.1], [0.0, 0.95]], { dtype });
  const b = np.array([[0.5], [0.1]], { dtype });
  const F = np.array([[1.0, 0.0]], { dtype });

  // Step function: (carry, input) => (newCarry, output)
  const step = (
    carry: { x: np.Array },
    input: { y: np.Array }
  ): [{ x: np.Array }, { v: np.Array; x_out: np.Array }] => {
    const { x } = carry;
    const { y } = input;

    // Innovation: v = y - F @ x
    const v = np.subtract(y, np.matmul(F.ref, x.ref));

    // State update: x_next = A @ x + b * v
    const x_next = np.add(
      np.matmul(A.ref, x.ref),
      np.multiply(b.ref, v.ref)
    );

    // Return new carry and outputs to stack
    return [
      { x: x_next },
      { v, x_out: x.ref }
    ];
  };

  // Initial state
  const x0 = np.array([[0.0], [0.0]], { dtype });

  // Observations (as np.Array for scan)
  const y_data = np.array([[1.0], [1.2], [0.9], [1.1], [1.0]], { dtype });

  // DESIRED: This would be JIT-compiled as a single fused operation
  // const [finalCarry, outputs] = lax.scan(step, { x: x0 }, { y: y_data });
  //
  // outputs.v would be [5, 1] array (stacked innovations)
  // outputs.x_out would be [5, 2, 1] array (stacked states)
  // finalCarry.x would be the final state [2, 1]
}

// ============================================================================
// Why this matters for Kalman filters
// ============================================================================

/**
 * In a full Kalman filter, the carry is a pytree with multiple components:
 *
 * Forward filter carry:
 *   { x: np.Array,    // state mean [m, 1]
 *     C: np.Array }   // state covariance [m, m]
 *
 * Forward filter outputs (accumulated):
 *   { v: np.Array,    // innovation [p, 1]
 *     Cp: np.Array,   // innovation covariance [p, p]
 *     K: np.Array,    // Kalman gain [m, p]
 *     x_pred: np.Array,
 *     C_pred: np.Array }
 *
 * The backward RTS smoother has similar structure, scanning in reverse.
 *
 * With lax.scan:
 * 1. The entire filter becomes a single JIT-compilable operation
 * 2. Memory can be optimized (no intermediate await allocations)
 * 3. Gradients flow through via autodiff (for parameter estimation)
 */

// ============================================================================
// Minimal test case for the feature request
// ============================================================================

/**
 * Cumulative sum - simplest possible scan example
 *
 * carry: running sum (scalar)
 * input: next value (scalar)
 * output: current sum (scalar)
 */
function cumulativeSumExample() {
  const step = (
    carry: np.Array,
    x: np.Array
  ): [np.Array, np.Array] => {
    const newCarry = np.add(carry, x);
    return [newCarry, newCarry.ref];
  };

  const init = np.array([0.0]);
  const xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0]);

  // DESIRED:
  // const [finalSum, runningSums] = lax.scan(step, init, xs);
  // finalSum = [15.0]
  // runningSums = [1.0, 3.0, 6.0, 10.0, 15.0]
}

/**
 * Pytree carry example - two values evolving together
 *
 * This demonstrates scanning over pytree leaves "simultaneously"
 */
function pytreeCarryExample() {
  // Carry is a pytree with two leaves
  type Carry = { a: np.Array; b: np.Array };
  type Output = { sum: np.Array; diff: np.Array };

  const step = (carry: Carry, x: np.Array): [Carry, Output] => {
    const { a, b } = carry;

    // Both components update based on input
    const newA = np.add(a, x);
    const newB = np.subtract(b, x);

    return [
      { a: newA, b: newB },
      { sum: np.add(newA.ref, newB.ref), diff: np.subtract(newA.ref, newB.ref) }
    ];
  };

  const init: Carry = {
    a: np.array([0.0]),
    b: np.array([10.0])
  };

  const xs = np.array([1.0, 2.0, 3.0]);

  // DESIRED:
  // const [finalCarry, outputs] = lax.scan(step, init, xs);
  //
  // Step 0: a=1, b=9  -> sum=10, diff=-8
  // Step 1: a=3, b=7  -> sum=10, diff=-4
  // Step 2: a=6, b=4  -> sum=10, diff=2
  //
  // finalCarry = { a: [6.0], b: [4.0] }
  // outputs.sum = [10.0, 10.0, 10.0]
  // outputs.diff = [-8.0, -4.0, 2.0]
}

export { currentApproach, desiredApproach, cumulativeSumExample, pytreeCarryExample };
