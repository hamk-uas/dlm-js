import { expect, describe, test, beforeAll } from "vitest";
import { init, defaultDevice, numpy as np, jit, lax } from "@jax-js/jax";
import { arrayClose } from "./utils";

describe("scan with jit", () => {
  beforeAll(async () => {
    const devices = await init();
    if (devices.includes('wasm')) {
      defaultDevice('wasm');
    } else if (devices.includes('cpu')) {
      defaultDevice('cpu');
    }
  });

  test("jit(scan) computes cumulative sum", async () => {
    function cumulativeSum(xs: any) {
      const init = np.zeros([]);
      const [finalCarry, ys] = lax.scan(
        (carry: any, x: any) => {
          const newCarry = carry.add(x);
          return [newCarry, newCarry];
        },
        init,
        xs
      );
      return [finalCarry, ys];
    }

    const xs = np.array([1, 2, 3, 4, 5]);
    const jitCumsum = jit(cumulativeSum);
    const [carry, result] = jitCumsum(xs);

    const expected = [1, 3, 6, 10, 15];
    await arrayClose(result, expected, 1e-6);

    const carryData = await carry.data();
    expect(carryData[0]).toBeCloseTo(15, 6);
  });

  test("jit(scan) matches non-jit version", async () => {
    function recurrence(xs: any) {
      const init = np.ones([]);
      const [finalCarry, ys] = lax.scan(
        (carry: any, x: any) => {
          const newCarry = carry.mul(x);
          return [newCarry, newCarry];
        },
        init,
        xs
      );
      return [finalCarry, ys];
    }

    // Create separate arrays for each test to avoid ref issues
    const xs1 = np.array([1, 2, 3, 4, 5]);
    const xs2 = np.array([1, 2, 3, 4, 5]);

    // Without jit
    const [carry1, result1] = recurrence(xs1);
    const data1 = await result1.data();

    // With jit
    const jitRecurrence = jit(recurrence);
    const [carry2, result2] = jitRecurrence(xs2);
    const data2 = await result2.data();

    // Should be identical: [1, 2, 6, 24, 120]
    expect(Array.from(data2)).toEqual(Array.from(data1));

    const c1 = (await carry1.data())[0];
    const c2 = (await carry2.data())[0];
    expect(c2).toEqual(c1);
  });

  test("jit(scan) with pytree carry", async () => {
    function multiCarry(xs: any) {
      // Use .ref to take references since we access carry/output multiple times
      const init = { sum: np.zeros([]), prod: np.ones([]) };
      const [finalCarry, ys] = lax.scan(
        (carry: any, x: any) => {
          // Need .ref on carry since we use them multiple times (once for add/mul, once for newCarry)
          const newSum = carry.sum.ref.add(x.ref);
          const newProd = carry.prod.mul(x);
          const newCarry = { sum: newSum.ref, prod: newProd.ref };
          return [newCarry, { s: newSum, p: newProd }];
        },
        init,
        xs
      );
      return [finalCarry, ys];
    }

    const xs = np.array([1, 2, 3, 4, 5]);
    const jitMulti = jit(multiCarry);
    const [carry, result] = jitMulti(xs);

    // Sums: [1, 3, 6, 10, 15]
    // Products: [1, 2, 6, 24, 120]
    const sums = await result.s.data();
    const prods = await result.p.data();

    expect(Array.from(sums)).toEqual([1, 3, 6, 10, 15]);
    expect(Array.from(prods)).toEqual([1, 2, 6, 24, 120]);
  });
});
