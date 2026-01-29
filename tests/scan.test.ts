/**
 * Tests for lax.scan implementation
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { init, defaultDevice, numpy as np, DType, tree, lax } from '@jax-js/jax';

const { scan, stackPyTree } = lax;

describe('scan', () => {
  beforeAll(async () => {
    const devices = await init();
    if (devices.includes('wasm')) {
      defaultDevice('wasm');
    } else if (devices.includes('cpu')) {
      defaultDevice('cpu');
    }
  });

  describe('pytree utilities', () => {
    it('tree.leaves extracts leaves from nested structure', () => {
      const a = np.array([1, 2, 3]);
      const b = np.array([4, 5, 6]);
      const c = np.array([7, 8, 9]);

      const t = { x: a, nested: { y: b, z: c } };
      const leaves = tree.leaves(t);

      expect(leaves).toHaveLength(3);
      expect(leaves[0]).toBe(a);
      expect(leaves[1]).toBe(b);
      expect(leaves[2]).toBe(c);
    });

    it('tree.map applies function to all leaves', () => {
      const a = np.array([1, 2, 3]);
      const b = np.array([4, 5, 6]);

      const t = { a, b };
      const mapped = tree.map((leaf: np.Array) => np.add(leaf.ref, np.array([10])), t);

      expect(tree.leaves(mapped)).toHaveLength(2);
    });

    it('stackPyTree stacks list of pytrees', async () => {
      const tree1 = { a: np.array([1.0]), b: np.array([2.0]) };
      const tree2 = { a: np.array([3.0]), b: np.array([4.0]) };
      const tree3 = { a: np.array([5.0]), b: np.array([6.0]) };

      const stacked = stackPyTree([tree1, tree2, tree3]);

      const aData = await (stacked as { a: np.Array; b: np.Array }).a.data();
      const bData = await (stacked as { a: np.Array; b: np.Array }).b.data();

      expect(Array.from(aData)).toEqual([1, 3, 5]);
      expect(Array.from(bData)).toEqual([2, 4, 6]);
    });
  });

  describe('scan basic', () => {
    it('computes cumulative sum', async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry.ref];
      };

      const init = np.array([0.0]);
      const xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

      const [finalCarry, outputs] = scan(step, init, xs);

      const finalData = await finalCarry.data();
      expect(finalData[0]).toBeCloseTo(15.0);

      const outputData = await outputs.data();
      expect(Array.from(outputData)).toEqual([1, 3, 6, 10, 15]);
    });

    it('computes factorial-like recurrence', async () => {
      // x(t) = x(t-1) * t
      const step = (carry: np.Array, t: np.Array): [np.Array, np.Array] => {
        const newCarry = np.multiply(carry, t);
        return [newCarry, newCarry.ref];
      };

      const init = np.array([1.0]);
      const ts = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

      const [final, outputs] = scan(step, init, ts);

      const finalData = await final.data();
      expect(finalData[0]).toBeCloseTo(120.0); // 5!

      const outputData = await outputs.data();
      expect(Array.from(outputData)).toEqual([1, 2, 6, 24, 120]);
    });
  });

  describe('scan with pytree carry', () => {
    it('tracks two values simultaneously', async () => {
      type Carry = { sum: np.Array; count: np.Array };

      const step = (carry: Carry, x: np.Array): [Carry, Carry] => {
        const newSum = np.add(carry.sum, x);
        const newCount = np.add(carry.count, np.array([1.0]));
        const newCarry = { sum: newSum, count: newCount };
        return [newCarry, { sum: newSum.ref, count: newCount.ref }];
      };

      const init: Carry = {
        sum: np.array([0.0]),
        count: np.array([0.0]),
      };
      const xs = np.array([[10.0], [20.0], [30.0]]);

      const [final, outputs] = scan(step, init, xs);

      const finalSum = await final.sum.data();
      const finalCount = await final.count.data();
      expect(finalSum[0]).toBeCloseTo(60.0);
      expect(finalCount[0]).toBeCloseTo(3.0);

      const outputSums = await (outputs as Carry).sum.data();
      expect(Array.from(outputSums)).toEqual([10, 30, 60]);
    });
  });

  describe('scan with pytree inputs', () => {
    it('processes paired inputs', async () => {
      type Input = { a: np.Array; b: np.Array };

      const step = (carry: np.Array, x: Input): [np.Array, np.Array] => {
        // weighted sum: carry + a - b
        const newCarry = np.subtract(np.add(carry, x.a), x.b);
        return [newCarry, newCarry.ref];
      };

      const init = np.array([0.0]);
      const xs: Input = {
        a: np.array([[5.0], [10.0], [15.0]]),
        b: np.array([[1.0], [2.0], [3.0]]),
      };

      const [final, outputs] = scan(step, init, xs);

      // Step 0: 0 + 5 - 1 = 4
      // Step 1: 4 + 10 - 2 = 12
      // Step 2: 12 + 15 - 3 = 24
      const finalData = await final.data();
      expect(finalData[0]).toBeCloseTo(24.0);

      const outputData = await outputs.data();
      expect(Array.from(outputData)).toEqual([4, 12, 24]);
    });
  });

  describe('Kalman-like recurrence', () => {
    it('runs simple state update', async () => {
      const dtype = DType.Float64;

      // Simple 1D state: x(t+1) = 0.9 * x(t) + 0.5 * y(t)
      const a = 0.9;
      const b = 0.5;

      const step = (carry: np.Array, y: np.Array): [np.Array, { x: np.Array; v: np.Array }] => {
        const x = carry;

        // Innovation: v = y - x (use x.ref since x is needed again below)
        const v = np.subtract(y, x.ref);

        // State update: x_new = a * x + b * v (use v.ref since v is returned)
        const x_new = np.add(
          np.multiply(np.array([a], { dtype }), x.ref),
          np.multiply(np.array([b], { dtype }), v.ref)
        );

        // Return x.ref for output, x is consumed by this point conceptually
        return [x_new, { x, v }];
      };

      const x0 = np.array([0.0], { dtype });
      const ys = np.array([[1.0], [1.0], [1.0], [1.0], [1.0]], { dtype });

      const [final, outputs] = scan(step, x0, ys);

      const finalData = await final.data();
      // Converges towards ~0.833 (fixed point of 0.9x + 0.5(1-x) = 0.4x + 0.5)
      expect(finalData[0]).toBeGreaterThan(0.8);

      const vData = await (outputs as { x: np.Array; v: np.Array }).v.data();
      // Innovations should decrease as state converges to observation
      expect(vData[0]).toBeCloseTo(1.0);  // First innovation: 1 - 0 = 1
      expect(vData[4]).toBeLessThan(vData[0]);  // Later innovations smaller
    });
  });
});
