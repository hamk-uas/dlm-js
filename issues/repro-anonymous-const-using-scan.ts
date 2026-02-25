/**
 * Repro: `using` + `markAnonymousIfTracing()` inside `jit(valueAndGrad(...))`
 *
 * After 41b0bda, the jit→scan nesting level is fixed (Case 1-3 all pass).
 * But jit(valueAndGrad(lossFn)) with np.ones/np.zeros + using inside lossFn
 * still crashes — the valueAndGrad creates an additional abstract trace
 * nesting level that the inMakeJaxprBody() guard doesn't cover.
 *
 * Expected: All cases OK.
 * Actual:   Case 4 (jit(valueAndGrad)) FAILED on a9db43d.
 *
 * Affected version: a9db43d  (partially fixed from 2303627 by 41b0bda)
 * Last known good:  6e6d4fe
 *
 * Run:  npx tsx issues/repro-anonymous-const-using-scan.ts
 */
import { numpy as np, jit, lax, DType, valueAndGrad } from '@hamk-uas/jax-js-nonconsuming';

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

  // ─── Case 4: jit(valueAndGrad(lossFn)) with np.ones + using → CRASHES ───
  const lossFn = (theta: any) => {
    using s = np.exp(theta);
    using V2 = np.reshape(np.square(s), [1, 1]);
    using _V2_ones = np.ones([5, 1, 1], { dtype: DType.Float64 });
    using V2_arr = np.multiply(_V2_ones, V2);

    const step = (carry: any, inp2: any) => {
      using pred = np.matmul(np.eye(1, undefined, { dtype: DType.Float64 }), carry);
      using diff = np.subtract(inp2, pred);
      using lik = np.divide(np.square(diff), V2_arr);
      return [carry, np.squeeze(lik)];
    };
    const [c, liks] = lax.scan(step, np.zeros([1, 1], { dtype: DType.Float64 }), V2_arr);
    c.dispose();
    return np.sum(liks);
  };

  const theta = np.array([0.5], { dtype: DType.Float64 });

  // 4a: valueAndGrad alone → works
  try {
    const [val, grad] = valueAndGrad(lossFn)(theta) as any;
    console.log('Case 4a (valueAndGrad only) OK — val:', val.js());
    val.dispose(); grad.dispose();
  } catch (err: any) {
    console.error('Case 4a FAILED:', err.message);
  }

  // 4b: jit(valueAndGrad) → CRASHES
  try {
    const [val, grad] = await jit(valueAndGrad(lossFn))(theta) as any;
    console.log('Case 4b (jit(valueAndGrad)) OK — val:', val.js());
    val.dispose(); grad.dispose();
  } catch (err: any) {
    console.error('Case 4b FAILED:', err.message);
  }

  theta.dispose();

  // Summary
  console.log('\nExpected: All cases OK');
  console.log('Actual on a9db43d: Cases 1-3 OK (fixed by 41b0bda),');
  console.log('  Case 4a OK, Case 4b FAILED');
  console.log('\nThe remaining bug: jit(valueAndGrad(lossFn)) adds an extra');
  console.log('abstract trace nesting level. np.ones() inside lossFn gets');
  console.log('disposed during the valueAndGrad trace before the outer jit');
  console.log('can capture it.');
}
main();
