import { numpy as np, defaultDevice, tree } from '@hamk-uas/jax-js-nonconsuming';
import { dlmGenSys, dlmFit } from '../src/index.ts';

defaultDevice('wasm');
await np.ready;

// Local level (m=1)
const sys1 = dlmGenSys({ order: 0 });
console.log('local level m =', sys1.m);
const y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const res1 = await dlmFit(y, { obsStd: 1, processStd: [1], dtype: 'f64', algorithm: 'sqrt-assoc' });
console.log('local level deviance:', res1.deviance);
tree.dispose(res1);

// FullSeasonal (m=13)
const sys3 = dlmGenSys({ order: 1, fullSeasonal: true, seasonLength: 12 });
console.log('\nfullSeasonal m =', sys3.m);
const y3 = Array.from({length: 100}, (_, i) => Math.sin(2*Math.PI*i/12) + Math.random()*0.1);
const res3 = await dlmFit(y3, { obsStd: 0.1, processStd: new Array(sys3.m).fill(0.03), 
  fullSeasonal: true, seasonLength: 12, dtype: 'f64', algorithm: 'sqrt-assoc' });
console.log('fullSeasonal deviance:', res3.deviance);
const hasNaN = Array.from(res3.yhat).some(v => isNaN(v));
console.log(`fullSeasonal yhat: hasNaN=${hasNaN}, first 5:`, Array.from(res3.yhat).slice(0, 5));
tree.dispose(res3);

// f32 trig (m=6)
console.log('\ntrig m=6 f32');
const res4 = await dlmFit(y3, { obsStd: 0.1, processStd: new Array(6).fill(0.03),
  order: 1, harmonics: 2, seasonLength: 12, dtype: 'f32', algorithm: 'sqrt-assoc' });
console.log('f32 trig deviance:', res4.deviance);
const hasNaN4 = Array.from(res4.yhat).some(v => isNaN(v));
console.log(`f32 trig yhat: hasNaN=${hasNaN4}`);
tree.dispose(res4);

// f32 fullSeasonal (m=13)  
console.log('\nf32 fullSeasonal m=13');
const res5 = await dlmFit(y3, { obsStd: 0.1, processStd: new Array(sys3.m).fill(0.03),
  fullSeasonal: true, seasonLength: 12, dtype: 'f32', algorithm: 'sqrt-assoc' });
console.log('f32 fullSeasonal deviance:', res5.deviance);
const hasNaN5 = Array.from(res5.yhat).some(v => isNaN(v));
console.log(`f32 fullSeasonal yhat: hasNaN=${hasNaN5}`);
tree.dispose(res5);

console.log('\nAll tests passed!');
process.exit(0);
