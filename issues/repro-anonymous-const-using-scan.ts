/**
 * Repro: `using` + anonymous consts in triple-nested jit → scan → valueAndGrad
 *
 * After 41b0bda + 99ca222, Cases 1-4b (2-level nesting) all pass.
 * But 3-level nesting (jit → lax.scan → valueAndGrad) still crashes.
 * This is the exact architecture of dlm-js's Adam optimizer in dlmMLE.
 *
 * Expected: All cases OK.
 * Actual on 99ca222: Cases 1-4b OK, Case 5 FAILED.
 *
 * Affected version: 99ca222
 * Last known good:  6e6d4fe (no anonymous-const-related crashes)
 *
 * Run:  npx tsx issues/repro-anonymous-const-using-scan.ts
 */
import { numpy as np, jit, lax, DType, valueAndGrad, tree } from '@hamk-uas/jax-js-nonconsuming';

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

  // ─── Case 5: jit → lax.scan → valueAndGrad (triple nesting, Adam optimBlock pattern) ───
  // This is the REAL bug: dlmMLE's Adam path uses:
  //   jit((theta, optState, lastLik) => {
  //     lax.scan(innerStep, {theta, optState, lastLik}, null, {length: 10})
  //   })
  // where innerStep calls valueAndGrad(lossFn)(carry.theta).
  // lossFn creates np.ones([n,1,1]) with `using` → disposed during scan body tracing.

  const lossFn5 = (theta5: any) => {
    using s5 = np.exp(theta5);
    using V2_5 = np.reshape(np.square(s5), [1, 1]);
    using _ones5 = np.ones([5, 1, 1], { dtype: DType.Float64 });  // ← the crash site
    using V2_arr5 = np.multiply(_ones5, V2_5);

    const step5 = (carry5: any, inp5: any) => {
      using pred5 = np.matmul(np.eye(1, undefined, { dtype: DType.Float64 }), carry5);
      using diff5 = np.subtract(inp5, pred5);
      using lik5 = np.divide(np.square(diff5), V2_arr5);
      return [carry5, np.squeeze(lik5)];
    };
    const [c5, liks5] = lax.scan(step5, np.zeros([1, 1], { dtype: DType.Float64 }), V2_arr5);
    c5.dispose();
    return np.sum(liks5);
  };

  const theta5 = np.array([0.5], { dtype: DType.Float64 });

  try {
    type OptCarry = { theta: np.Array; lastLik: np.Array };

    const innerStep5 = (carry: OptCarry, _x: null): [OptCarry, null] => {
      const [likVal, _grad] = valueAndGrad(lossFn5)(carry.theta) as [np.Array, np.Array];
      _grad.dispose();
      return [{ theta: carry.theta, lastLik: likVal }, null];
    };

    const lastLik5 = np.array(Infinity, { dtype: DType.Float64 });

    const optimBlock = jit((t: np.Array, lik: np.Array) => {
      const [finalCarry, _ys] = lax.scan(
        innerStep5,
        { theta: t, lastLik: lik } as OptCarry,
        null,
        { length: 2 },
      );
      return finalCarry;
    });

    const result = await optimBlock(theta5, lastLik5) as OptCarry;
    console.log('Case 5 (jit→scan→valueAndGrad, triple nesting) OK — lik:', result.lastLik.js());
    tree.dispose(result);
    lastLik5.dispose();
  } catch (err: any) {
    console.error('Case 5 (jit→scan→valueAndGrad, triple nesting) FAILED:', err.message);
  }

  theta5.dispose();

  // Summary
  console.log('\nExpected: All cases OK (including Case 5)');
  console.log('Actual on 99ca222: Cases 1-4b OK (fixed by 41b0bda + 99ca222),');
  console.log('  Case 5 FAILED (triple nesting: jit → scan → valueAndGrad)');
  console.log('\nThis is the architecture of dlmMLE Adam optimizer.');
  console.log('21 of 200 dlm-js tests fail because of this pattern.');
}
main();
