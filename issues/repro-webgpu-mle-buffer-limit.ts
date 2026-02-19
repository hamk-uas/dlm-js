/**
 * Minimal reproduction: jit(valueAndGrad) over lax.associativeScan exceeds
 * WebGPU 8-buffer bind-group limit.
 *
 * ─── BACKGROUND ──────────────────────────────────────────────────────────────
 *
 * WebGPU limits storage buffers to 8 per bind group.  When jax-js fuses the
 * backward pass of a differentiable lax.associativeScan into a single WebGPU
 * shader, the resulting bind group references inputs, outputs, and saved
 * activations from the compose body simultaneously — yielding > 8 buffers.
 *
 * The compose function has a 3-element tuple output (a, b, c), each a scalar
 * [N] tensor.  This directly mirrors the (A, x, C) / (A, b, S) triplet
 * structure used by makeKalmanLossAssoc in dlm-js.
 *
 * Even at this minimal complexity, the backward fuser produces > 8 buffer
 * references per bind group, crashing with:
 *
 *   Error: Too many buffers (12) for WebGPU pipeline (max: 8)
 *
 * There is no viable eager workaround: removing `jit` causes `using`-managed
 * intermediate tensors to be freed before the AD backward pass can read saved
 * activations:
 *
 *   Error: Referenced tracer Array:float32[N] has been disposed
 *
 * ─── HOW TO RUN ──────────────────────────────────────────────────────────────
 *
 *   deno run --no-check --unstable-webgpu --allow-read --allow-env \
 *     issues/repro-webgpu-mle-buffer-limit.ts
 *
 * ─── EXPECTED OUTPUT (after fix) ────────────────────────────────────────────
 *
 *   Test 1 [jit(valueAndGrad)]:              PASS  lik=<value>  grad=<value>
 *   Test 2 [valueAndGrad, no jit]:           PASS  lik=<value>  grad=<value>
 *   Test 3 [valueAndGrad(jit(lossFn))]:      PASS  lik=<value>  grad=<value>
 *
 * ─── ACTUAL OUTPUT (jax-js-nonconsuming v0.7.3) ──────────────────────────────
 *
 *   Test 1 [jit(valueAndGrad)]:              FAIL  Too many buffers (9) for WebGPU pipeline (max: 8)
 *   Test 2 [valueAndGrad, no jit]:           PASS  lik=284451.6875  grad=4813868.5000
 *                                            FAIL  Referenced tracer Array:float32[1] has been disposed
 *   Test 3 [valueAndGrad(jit(lossFn))]:      FAIL  Too many buffers (9) for WebGPU pipeline (max: 8)
 */

import {
  DType, defaultDevice, init,
  numpy as np, jit, valueAndGrad, lax,
} from "@hamk-uas/jax-js-nonconsuming";

await init("webgpu");
defaultDevice("webgpu");

const dtype = DType.Float32;
const N = 20;

// ── Loss function using lax.associativeScan with a 3-tuple compose ──────────
//
// theta: scalar [1] — log of a scale factor
// Compose: (a0,b0,c0) ⊕ (a1,b1,c1) → (a1·a0, a1·b0+b1, a1·c0+c1)
// (scalar multiply-add on three channels — mirrors (A, x, C) Kalman structure)
//
// This is the minimal differentiable 3-tuple associative scan that exercises
// the same backward-pass buffer structure as makeKalmanLossAssoc in dlm-js.

function makeAssocLoss(N: number) {
  return function assocLoss(theta: np.Array): np.Array {
    using scale = np.exp(theta);   // [1] — differentiable, broadcast over N

    // Build [N] element tensors from theta
    using ones_n = np.ones([N], { dtype });
    using a_elems = np.multiply(ones_n, scale);  // [N] = exp(theta)
    using b_elems = np.multiply(ones_n, scale);  // [N] = exp(theta)
    using c_elems = np.ones([N], { dtype });      // [N] = 1 (constant)

    // Compose: (lhs, rhs) → element-wise multiply-add on 3 channels
    function compose(
      lhs: [np.Array, np.Array, np.Array],
      rhs: [np.Array, np.Array, np.Array],
    ): [np.Array, np.Array, np.Array] {
      using a_new = np.multiply(rhs[0], lhs[0]);
      using rb    = np.multiply(rhs[0], lhs[1]);
      using b_new = np.add(rb, rhs[1]);
      using rc    = np.multiply(rhs[0], lhs[2]);
      using c_new = np.add(rc, rhs[2]);
      return [a_new, b_new, c_new];
    }

    const [a_scan, b_scan, c_scan] = lax.associativeScan(
      compose, [a_elems, b_elems, c_elems]);

    // Scalar loss: sum(a_scan + b_scan + c_scan)
    using ab  = np.add(a_scan, b_scan);
    using abc = np.add(ab, c_scan);
    using lik = np.sum(abc);

    a_scan.dispose(); b_scan.dispose(); c_scan.dispose();
    return lik;
  };
}

const lossFn = makeAssocLoss(N);
using theta0 = np.array([0.5], { dtype });  // single scalar param, no split needed

// ── Test 1: jit(valueAndGrad) ────────────────────────────────────────────────
console.log("Test 1 [jit(valueAndGrad(lossFn))]:  (expect: Too many buffers)");
try {
  const jvg = jit((theta: np.Array): [np.Array, np.Array] =>
    valueAndGrad(lossFn)(theta));
  const [lik, grad] = jvg(theta0);
  const likV = (await lik.consumeData())[0];
  const gradV = (await grad.consumeData())[0];
  console.log(`  PASS  lik=${likV.toFixed(4)}  grad=${gradV.toFixed(4)}`);
  grad.dispose();
} catch (e: any) {
  console.log(`  FAIL  ${e.message}`);
}

// ── Test 2: valueAndGrad without jit ─────────────────────────────────────────
console.log("Test 2 [valueAndGrad(lossFn), no jit]:  (expect: tracer disposed)");
try {
  const [lik, grad] = valueAndGrad(lossFn)(theta0);
  const likV = (await lik.consumeData())[0];
  const gradV = (await grad.consumeData())[0];
  console.log(`  PASS  lik=${likV.toFixed(4)}  grad=${gradV.toFixed(4)}`);
  grad.dispose();
} catch (e: any) {
  console.log(`  FAIL  ${e.message}`);
}

// ── Test 3: valueAndGrad(jit(fn)) — jit inside ───────────────────────────────
console.log("Test 3 [valueAndGrad(jit(lossFn))]:  (jit inside, AD outside)");
try {
  const jitLoss = jit(lossFn);
  const [lik, grad] = valueAndGrad(jitLoss)(theta0);
  const likV = (await lik.consumeData())[0];
  const gradV = (await grad.consumeData())[0];
  console.log(`  PASS  lik=${likV.toFixed(4)}  grad=${gradV.toFixed(4)}`);
  grad.dispose();
} catch (e: any) {
  console.log(`  FAIL  ${e.message}`);
}
