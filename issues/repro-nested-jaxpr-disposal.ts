/**
 * Repro: nested ClosedJaxpr disposal is non-recursive.
 *
 * Structure: jit → valueAndGrad → lax.scan(step, ...)
 * where `step` captures closure constants (F, Ft, x0, C0, y_arr).
 *
 * After the JIT function executes and the `using` scope exits,
 * checkLeaks.stop() reports the closure-captured arrays as leaked
 * with rc=1 because:
 *  1. ClosedJaxpr.dispose() only iterates this.consts (top-level),
 *     not nested scan/JVP sub-jaxpr consts.
 *  2. [Symbol.dispose]() is a no-op during PE tracing.
 *
 * Run: npx tsx issues/repro-nested-jaxpr-disposal.ts
 */

import { numpy as np, jit, lax, valueAndGrad, checkLeaks, DType } from '@hamk-uas/jax-js-nonconsuming';

const dtype = DType.Float64;
const n = 10;

checkLeaks.start();

{
  // Constants created OUTSIDE the JIT body (closure-captured)
  using F  = np.array([[1]], { dtype });
  using Ft = np.transpose(F);
  using x0 = np.zeros([1, 1], { dtype });
  using C0 = np.eye(1, undefined, { dtype });
  using y_arr = np.ones([n, 1, 1], { dtype });

  const lossFn = (theta: np.Array): np.Array => {
    type Carry = { x: np.Array; C: np.Array };
    type Inp   = { y: np.Array };

    // Inline constant — leaks because `using` is no-op during PE
    using ones = np.ones([n, 1, 1], { dtype });

    const step = (carry: Carry, inp: Inp): [Carry, np.Array] => {
      // F, Ft captured from outer scope → become scan sub-jaxpr consts
      using v = np.subtract(inp.y, np.matmul(F, carry.x));
      using CFt = np.matmul(carry.C, Ft);
      const Cp = np.add(np.matmul(F, CFt), ones);
      const x_next = np.add(
        np.matmul(F, carry.x),
        np.divide(np.matmul(F, CFt), Cp),
      );
      using lik = np.add(np.divide(np.square(v), Cp), np.log(Cp));
      return [{ x: x_next, C: carry.C }, np.squeeze(lik)];
    };

    const [fc, liks] = lax.scan(step, { x: x0, C: C0 }, { y: y_arr });
    fc.x.dispose(); fc.C.dispose();
    const total = np.sum(liks);
    liks.dispose();
    return total;
  };

  const fn = jit((theta: np.Array) => {
    const [val, grad] = valueAndGrad(lossFn)(theta);
    grad.dispose();
    return val;
  });

  using theta = np.zeros([1], { dtype });
  using result = await fn(theta);
}

const report = checkLeaks.stop();
console.log(`leaked=${report.leaked} userLeaked=${report.userLeaked} internalLeaked=${report.internalLeaked}`);

if (report.userLeaked > 0) {
  console.log('\n--- User-code leaks (all upstream-caused): ---');
  console.log(report.summary);
  console.log('\nExpected: userLeaked=0 (all constants properly using/closure-captured)');
  console.log('Actual:   userLeaked>0 (nested sub-jaxpr consts not freed by _disposeAllJitCaches)');
  process.exit(1);
} else {
  console.log('No user leaks — issue is fixed!');
}
