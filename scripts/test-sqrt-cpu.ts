import { numpy as np, defaultDevice, tree } from '@hamk-uas/jax-js-nonconsuming';
import { dlmGenSys, dlmFit } from '../src/index.ts';

defaultDevice('cpu');
await np.ready;
console.log('Testing sqrt-assoc on cpu backend');

// Linear trend m=2 on cpu
const y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
try {
  const res = await dlmFit(y, { obsStd: 1, processStd: [1, 1], dtype: 'f64', algorithm: 'sqrt-assoc' });
  const yhat = Array.from(res.yhat);
  const hasNaN = yhat.some(v => isNaN(v));
  console.log(`cpu f64 m=2: hasNaN=${hasNaN}, deviance=${res.deviance}`);
  tree.dispose(res);
} catch (e: any) {
  console.error('cpu f64 m=2 FAILED:', e.message);
}

process.exit(0);
