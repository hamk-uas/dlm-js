/**
 * Repro: wrapping a function that contains lax.scan inside another function
 * passed to valueAndGrad causes PETracer leaks.
 *
 * Wrapping makeKalmanLoss (which contains lax.scan + structured scan inputs
 * + many `using` intermediates) inside valueAndGrad leaks arrays.
 *
 * Direct call:   valueAndGrad(kalmanLoss)(theta)    → 0 leaks  ✅
 * Wrapped call:  valueAndGrad(theta => kalmanLoss(theta))(theta)  → leaks  ❌
 *
 * Run:  npx tsx issues/repro-wrapped-valueandgrad-scan.ts
 */
import {
  numpy as np,
  jit,
  lax,
  valueAndGrad,
  tree,
  checkLeaks,
  defaultDevice,
} from '@hamk-uas/jax-js-nonconsuming';
import { adam, applyUpdates } from '@hamk-uas/jax-js-nonconsuming/optax';
import { dlmMLE, dlmGenSys } from '../src/index';
import type { DlmLossFn } from '../src/index';

const DType = np.DType;

function mulberry32(seed: number): () => number {
  let a = seed | 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gaussianRng(uniform: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u1: number;
    do { u1 = uniform(); } while (u1 === 0);
    const u2 = uniform();
    const mag = Math.sqrt(-2 * Math.log(u1));
    spare = mag * Math.sin(2 * Math.PI * u2);
    return mag * Math.cos(2 * Math.PI * u2);
  };
}
function generateData(
  G: number[][], F: number[], s: number, w: number[], n: number, seed: number,
): number[] {
  const m = G.length;
  const randn = gaussianRng(mulberry32(seed));
  const x = new Array(m).fill(0);
  x[0] = randn() * 10;
  for (let k = 1; k < m; k++) x[k] = randn();
  const y: number[] = [];
  for (let t = 0; t < n; t++) {
    const xNew = new Array(m).fill(0);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) xNew[i] += G[i][j] * x[j];
      if (i < w.length) xNew[i] += w[i] * randn();
    }
    let obs = 0;
    for (let k = 0; k < m; k++) obs += F[k] * xNew[k];
    y.push(obs + s * randn());
    for (let k = 0; k < m; k++) x[k] = xNew[k];
  }
  return y;
}

async function main() {
  defaultDevice('wasm');
  const options = { order: 0 };
  const sys = dlmGenSys(options);
  const y = generateData(sys.G, sys.F, 10, [3], 50, 42);
  const base = { ...options, init: { obsStd: 10, processStd: [3] },
    maxIter: 20, lr: 0.05, tol: 1e-6, dtype: 'f64' as const };
  const identity: DlmLossFn = (deviance, _params, _meta) => deviance;

  type CaseResult = { label: string; leaked: number; error?: string };
  const results: CaseResult[] = [];

  // ── Case 1: no custom loss → 0 leaks ──
  {
    const g = checkLeaks.start();
    try {
      await dlmMLE(y, base);
      const r = checkLeaks.stop(g as any);
      console.log('Case 1 (no loss):        userLeaked=', r.userLeaked);
      results.push({ label: 'No loss (direct kalmanLoss)', leaked: r.userLeaked });
    } catch (e: any) {
      checkLeaks.stop(g as any);
      console.log('Case 1 (no loss):        CRASH:', e.message);
      results.push({ label: 'No loss (direct kalmanLoss)', leaked: -1, error: e.message });
    }
  }

  // ── Case 2: identity loss callback via Adam → leaks ──
  {
    const g = checkLeaks.start();
    try {
      await dlmMLE(y, { ...base, loss: identity });
      const r = checkLeaks.stop(g as any);
      console.log('Case 2 (identity+Adam):  userLeaked=', r.userLeaked);
      if (r.userLeaked > 0) console.log('  ', r.summary);
      results.push({ label: 'Identity loss (Adam)', leaked: r.userLeaked });
    } catch (e: any) {
      const r = checkLeaks.stop(g as any);
      console.log('Case 2 (identity+Adam):  CRASH:', e.message);
      console.log('  leaked anyway:', r.userLeaked);
      results.push({ label: 'Identity loss (Adam)', leaked: r.userLeaked, error: e.message });
    }
  }

  // ── Case 3: identity loss + natural gradient → leaks ──
  {
    const g = checkLeaks.start();
    try {
      await dlmMLE(y, { ...base, loss: identity, optimizer: 'natural' });
      const r = checkLeaks.stop(g as any);
      console.log('Case 3 (identity+natrl): userLeaked=', r.userLeaked);
      if (r.userLeaked > 0) console.log('  ', r.summary);
      results.push({ label: 'Identity loss (natural)', leaked: r.userLeaked });
    } catch (e: any) {
      const r = checkLeaks.stop(g as any);
      console.log('Case 3 (identity+natrl): CRASH:', e.message);
      console.log('  leaked anyway:', r.userLeaked);
      results.push({ label: 'Identity loss (natural)', leaked: r.userLeaked, error: e.message });
    }
  }

  // ── Summary ──
  console.log('\n=== Summary ===');
  for (const r of results) {
    const status = r.error ? '💥 CRASH' : r.leaked === 0 ? '✅ OK' : `❌ ${r.leaked} leaks`;
    console.log(`  ${status}  ${r.label}${r.error ? ` (${r.error.slice(0, 60)})` : ''}`);
  }

  const anyBad = results.some(r => r.leaked > 0 || r.error);
  if (anyBad) {
    console.log('\nBug: dlmMLE leaks/crashes only when a custom loss callback is provided.');
    console.log('The callback is identity: (deviance) => deviance — adds zero ops.');
    console.log('Internally, dlmMLE wraps makeKalmanLoss (which uses lax.scan) in');
    console.log('a closure: (theta) => { const kl = kalmanLoss(theta); ... return kl; }');
    console.log('This wrapping alone triggers PETracer leaks/use-after-free.');
    process.exit(1);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
