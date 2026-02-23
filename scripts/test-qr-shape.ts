import { numpy as np, defaultDevice } from '@hamk-uas/jax-js-nonconsuming';
defaultDevice('wasm');
await np.ready;

// Test QR return shape with batched 3D input
const A = np.array([[[1,2],[3,4],[5,6]]], {dtype: 'float64'});  // [1, 3, 2]
console.log('A shape:', A.shape);

const result = np.linalg.qr(A);
console.log('qr result type:', typeof result);
console.log('qr result is array?', Array.isArray(result));
if (Array.isArray(result)) {
  console.log('Q shape:', result[0].shape);
  console.log('R shape:', result[1].shape);
} else {
  const r = result as any;
  console.log('result keys:', Object.keys(r));
  if (r.Q) console.log('Q shape:', r.Q.shape);
  if (r.q) console.log('q shape:', r.q.shape);
  console.log('result shape?', r.shape);
  console.log('result[0]?', r[0]);
}

process.exit(0);
