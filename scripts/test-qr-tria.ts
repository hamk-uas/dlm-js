import { numpy as np, defaultDevice } from '@hamk-uas/jax-js-nonconsuming';
defaultDevice('wasm');
await np.ready;

// Test tria with Psi-like shape: [n, 1+m, m+1] where p <= q
// Psi [1, 3, 4] for m=2: p=3, q=4
// A' is [1, 4, 3], qr(A') → Q [1, 4, 3], R [1, 3, 3] → L = R' [1, 3, 3] — square!
const C = np.array([[[1, 0.5, 0.2, 0.1], [0, 1, 0.3, 0.4], [0, 0, 1, 0.5]]], {dtype: 'float64'});  // [1, 3, 4]
console.log('C (Psi-like) shape:', C.shape);

const Ct = np.einsum('nij->nji', C);  // [1, 4, 3]
const [Q, R] = np.linalg.qr(Ct);
console.log('Q shape:', Q.shape);
console.log('R shape:', R.shape);

const L = np.einsum('nij->nji', R);
console.log('L shape:', L.shape, '— should be [1, 3, 3] (square)');

// Now verify: L·L' = C·C' ?
const LLT = np.einsum('nij,nkj->nik', L, L);
const CCT = np.einsum('nij,nkj->nik', C, C);
console.log('LLT data:', Array.from(LLT.consumeData()));
console.log('CCT data:', Array.from(CCT.consumeData()));

// Also test Xi-like shape: [n, 2m, 2m] — square
const Xi = np.array([[[1, 0.5, 0.2, 0.1], [0, 1, 0.3, 0.4], [0, 0, 1, 0.5], [0.1, 0, 0, 1]]], {dtype: 'float64'});  // [1, 4, 4]
console.log('\nXi (square) shape:', Xi.shape);
const Xit = np.einsum('nij->nji', Xi);
const [Q2, R2] = np.linalg.qr(Xit);
console.log('Q2 shape:', Q2.shape);
console.log('R2 shape:', R2.shape);
const L2 = np.einsum('nij->nji', R2);
console.log('L2 shape:', L2.shape, '— should be [1, 4, 4] (square)');

// And U/Z-like shape: [n, m, 2m] — "fat" p<q
const U = np.array([[[1, 0.5, 0.2, 0.1], [0, 1, 0.3, 0.4]]], {dtype: 'float64'});  // [1, 2, 4]
console.log('\nU (fat m×2m) shape:', U.shape);
const Ut = np.einsum('nij->nji', U);
const [Q3, R3] = np.linalg.qr(Ut);
console.log('Q3 shape:', Q3.shape);
console.log('R3 shape:', R3.shape);
const L3 = np.einsum('nij->nji', R3);
console.log('L3 shape:', L3.shape, '— should be [1, 2, 2] (square)');

process.exit(0);
