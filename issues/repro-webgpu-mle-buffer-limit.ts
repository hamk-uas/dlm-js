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
 * The compose function here mirrors the DLM Kalman loss in dlm-js
 * (makeKalmanLossAssoc), reduced to m=1 (scalar matrices) to minimise noise.
 * Even at m=1 the compose outputs a 3-tuple (A, b, S), and the backward fuser
 * produces 12 buffer references, crashing with:
 *
 *   Error: Too many buffers (12) for WebGPU pipeline (max: 8)
 *
 * There is no viable eager workaround: removing `jit` causes `using`-managed
 * intermediate tensors (disposals) to be freed before the AD backward pass can
 * read saved activations:
 *
 *   Error: Referenced tracer Array:float32[1,1] has been disposed
 *
 * ─── HOW TO RUN ──────────────────────────────────────────────────────────────
 *
 *   deno run --no-check --unstable-webgpu --allow-read --allow-env \
 *     issues/repro-webgpu-mle-buffer-limit.ts
 *
 * ─── EXPECTED OUTPUT (after fix) ────────────────────────────────────────────
 *
 *   Test 1 [jit(valueAndGrad)]:   PASS  lik=<value>  grad=<value>
 *   Test 2 [valueAndGrad, no jit]: PASS  lik=<value>  grad=<value>
 *
 * ─── ACTUAL OUTPUT (jax-js-nonconsuming v0.7.1) ─────────────────────────────
 *
 *   Test 1 [jit(valueAndGrad)]:   FAIL  Error: Too many buffers (12) for WebGPU pipeline (max: 8)
 *   Test 2 [valueAndGrad, no jit]: FAIL  Error: Referenced tracer Array:float32[1,1] has been disposed
 */

import {
  DType, defaultDevice, init,
  numpy as np, jit, valueAndGrad, lax, tree,
} from "@hamk-uas/jax-js-nonconsuming";

await init("webgpu");
defaultDevice("webgpu");

const dtype = DType.Float32;
const N = 20; // series length — small enough to be fast, big enough to show the issue

// ── Minimal differentiable loss using lax.associativeScan ──────────────────
//
// compose((A_a, b_a, S_a), (A_b, b_b, S_b)) → (A_c, b_c, S_c)
// mirrors the DLM Kalman forward-filter compose in makeKalmanLossAssoc (m=1).
//
// Using m=1 (scalars wrapped as [1,1] matrices) to minimise the example while
// still exercising the same code path as the real m=2 Nile model.

function makeMinimalLoss(N: number) {
  // Fake observations and a fixed G/F for simplicity
  using y_const = np.ones([N, 1, 1], { dtype });
  const y_snap = y_const; // captured by closure

  return function minimalAssocLoss(theta: np.Array): np.Array {
    // theta = [log_s, log_w]: [2]
    using log_s = np.slice(theta, [0], [1]);
    using log_w = np.slice(theta, [1], [2]);
    using s = np.exp(log_s);        // [1]
    using w = np.exp(log_w);        // [1]
    using V2 = np.einsum('i,i->i', s, s);  // s²
    using W  = np.reshape(w, [1, 1]);       // [1,1]
    using G  = np.eye(1, { dtype });        // [1,1]
    using F  = np.ones([1, 1], { dtype }); // [1,1]

    // Build scan elements: one per timestep [N, 1, 1], [N, 1, 1], [N, 1, 1]
    using v2_scalar = np.reshape(V2, []);
    using Ak = np.expand_dims(G, 0);                          // [1,1,1]
    using Ak_n = np.broadcast_to(Ak, [N, 1, 1]);             // [N,1,1]
    using bk = np.zeros([N, 1, 1], { dtype });
    using Sk_base = np.expand_dims(W, 0);                     // [1,1,1]
    using Sk_n = np.broadcast_to(Sk_base, [N, 1, 1]);        // [N,1,1]

    const elems: [np.Array, np.Array, np.Array] = [Ak_n, bk, Sk_n];

    // Compose: the 3-tuple associative operation
    function compose(
      a: [np.Array, np.Array, np.Array],
      b: [np.Array, np.Array, np.Array]
    ): [np.Array, np.Array, np.Array] {
      using A_c = np.einsum('nij,njk->nik', b[0], a[0]);
      using Ab  = np.einsum('nij,njk->nik', b[0], a[1]);
      using b_c = np.add(Ab, b[1]);
      using BSAt = np.einsum('nij,njk,nlk->nil', b[0], a[2], b[0]);
      using S_c  = np.add(BSAt, b[2]);
      return [A_c, b_c, S_c];
    }

    const [_A_scan, _b_scan, S_scan] = lax.associativeScan(compose, elems);

    // Simple scalar loss: sum of diagonal of final S (proxy for log-likelihood)
    using diag = np.einsum('nii->n', S_scan);
    using logdiag = np.log(np.abs(diag));
    using lik = np.sum(logdiag);

    _A_scan.dispose();
    _b_scan.dispose();
    S_scan.dispose();

    return lik;
  };
}

const lossFn = makeMinimalLoss(N);
using theta0 = np.array([Math.log(15.0), Math.log(1.0)], { dtype });

// ── Test 1: jit(valueAndGrad) ────────────────────────────────────────────────
console.log("Test 1 [jit(valueAndGrad(lossFn))]:  (expected: Too many buffers (12) for WebGPU pipeline (max: 8))");
try {
  const jittedVG = jit((theta: np.Array): [np.Array, np.Array] => {
    return valueAndGrad(lossFn)(theta);
  });
  const [lik, grad] = jittedVG(theta0);
  const likV = (await lik.consumeData())[0];
  const gradV = Array.from(await grad.consumeData());
  console.log(`  PASS  lik=${likV.toFixed(4)}  grad=[${gradV.map(v => v.toFixed(4)).join(', ')}]`);
  grad.dispose();
} catch (e: any) {
  console.log(`  FAIL  ${e.message}`);
}

// ── Test 2: valueAndGrad without jit ─────────────────────────────────────────
console.log("Test 2 [valueAndGrad(lossFn), no jit]:  (expected: Referenced tracer ... has been disposed)");
try {
  const vg = valueAndGrad(lossFn);
  const [lik, grad] = vg(theta0);
  const likV = (await lik.consumeData())[0];
  const gradV = Array.from(await grad.consumeData());
  console.log(`  PASS  lik=${likV.toFixed(4)}  grad=[${gradV.map(v => v.toFixed(4)).join(', ')}]`);
  grad.dispose();
} catch (e: any) {
  console.log(`  FAIL  ${e.message}`);
}
