/**
 * Repro: `using` + `markAnonymousIfTracing()` inside `lax.scan` body within `jit`
 *
 * After commits 2af40bc + ec9acec, np.eye/np.zeros/np.arange/np.linspace are
 * wrapped in markAnonymousIfTracing().  Inside a lax.scan body within jit(),
 * this causes [Symbol.dispose]() to ACTUALLY dispose the array during tracing
 * (instead of being a no-op as before), leaving the JIT with a dangling
 * reference to a disposed constant.
 *
 * Expected: using + np.eye inside scan body within jit is a no-op during
 *           tracing (as documented).  First jit call succeeds.
 * Actual:   First call throws "Referenced tracer Array:float64[2,2] has been
 *           disposed" at _realizeSource during JIT execution.
 *
 * Affected version: 2303627  (and any commit including 2af40bc or ec9acec)
 * Last known good:  6e6d4fe
 *
 * Run:  npx tsx issues/repro-anonymous-const-using-scan.ts
 */
import { numpy as np, jit, lax, DType } from '@hamk-uas/jax-js-nonconsuming';

async function main() {
  // ─── Case 1: using + np.eye inside scan body within jit → CRASHES ───
  const fn = (x: any) => {
    const step = (carry: any, inp: any) => {
      using e = np.eye(2, undefined, { dtype: DType.Float64 });
      const newCarry = np.add(np.matmul(e, carry), inp);
      return [newCarry, newCarry];
    };
    const [finalCarry, allSteps] = lax.scan(
      step,
      np.zeros([2, 1], { dtype: DType.Float64 }),
      x,
    );
    finalCarry.dispose();
    return allSteps;
  };

  const inp = np.ones([5, 2, 1], { dtype: DType.Float64 });

  try {
    const result = await jit(fn)(inp);
    console.log('Case 1 OK — shape:', result.shape);
    result.dispose();
  } catch (err: any) {
    console.error('Case 1 FAILED:', err.message);
    // Expected: "Referenced tracer Array:float64[2,2] has been disposed"
  }

  // ─── Case 2: same code WITHOUT using → works fine ───
  const fn2 = (x: any) => {
    const step = (carry: any, inp: any) => {
      const e = np.eye(2, undefined, { dtype: DType.Float64 });
      const newCarry = np.add(np.matmul(e, carry), inp);
      return [newCarry, newCarry];
    };
    const [finalCarry, allSteps] = lax.scan(
      step,
      np.zeros([2, 1], { dtype: DType.Float64 }),
      x,
    );
    finalCarry.dispose();
    return allSteps;
  };

  try {
    const result = await jit(fn2)(inp);
    console.log('Case 2 OK — shape:', result.shape);
    result.dispose();
  } catch (err: any) {
    console.error('Case 2 FAILED:', err.message);
  }

  // ─── Case 3: using + np.eye inside jit (no scan) → works ───
  const fn3 = (x: any) => {
    using e = np.eye(2, undefined, { dtype: DType.Float64 });
    return np.matmul(e, x);
  };

  try {
    const x3 = np.ones([2, 1], { dtype: DType.Float64 });
    const result = await jit(fn3)(x3);
    console.log('Case 3 OK — shape:', result.shape);
    result.dispose();
    x3.dispose();
  } catch (err: any) {
    console.error('Case 3 FAILED:', err.message);
  }

  inp.dispose();

  // Summary
  console.log('\nExpected: Case 1 OK, Case 2 OK, Case 3 OK');
  console.log('Actual:   Case 1 FAILED (regression), Case 2 OK, Case 3 OK');
  console.log('\nThe bug: markAnonymousIfTracing() on np.eye() inside a');
  console.log('lax.scan body trace causes [Symbol.dispose]() to actually');
  console.log('dispose the array during tracing, instead of being a no-op.');
  console.log('The outer JIT then holds a dangling reference to the disposed');
  console.log('constant and crashes at execution time.');
}
main();
