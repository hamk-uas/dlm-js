/**
 * Collect per-frame data for the animated energy MLE + AR fitting SVG.
 *
 * Runs MLE optimization on the synthetic energy data with fitar=true,
 * capturing smoothed states at sampled iterations for animation frames.
 * The animated signal is the combined F·x = level + seasonal + AR.
 *
 * Model: order=1, trig=1, ns=12, arphi=[0.85], fitar=true, m=5
 * True: s=1.5, w=[0.3, 0.02, 0.02, 0.02, 2.5], arphi=[0.85]
 *
 * Output: tmp/energy-mle-frames.json
 */

import { DType, numpy as np, lax, jit, valueAndGrad, tree, defaultDevice } from "@hamk-uas/jax-js-nonconsuming";
import { adam, applyUpdates } from "@hamk-uas/jax-js-nonconsuming/optax";
import { dlmFit } from "../src/index.ts";
import { dlmGenSys, findArInds } from "../src/dlmgensys.ts";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { performance } from "node:perf_hooks";

defaultDevice("wasm");

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const input = JSON.parse(readFileSync(resolve(root, "tests/energy-in.json"), "utf8"));
const y: number[] = input.y;
const n = y.length;
const t: number[] = Array.from({ length: n }, (_, i) => i + 1); // month index 1..120
const dtype = DType.Float64;
const options = { order: 1, trig: 1, ns: 12, arphi: [0.5], fitar: true };
const maxIter = 300;
const lr = 0.02;
const tol = 1e-6;
const TARGET_FPS = 10;
const HOLD_SECONDS = 2;

// ── System setup ───────────────────────────────────────────────────────────

const sys = dlmGenSys(options);
const m = sys.m; // 5

const arInds = findArInds(options);
const nar = arInds.length; // 1
const nSwParams = 1 + m; // 6
const nTheta = nSwParams + nar; // 7

// Zero AR column in G for the loss function
const G_data = sys.G.map((row: number[]) => [...row]);
const arCol = arInds[0];
for (const idx of arInds) G_data[idx][arCol] = 0;

const G = np.array(G_data, { dtype });
const F = np.array([sys.F], { dtype }); // [1, m]
const Ft = np.transpose(F);

const yArr = Float64Array.from(y);
const y_arr = np.array(Array.from(yArr).map(yi => [[yi]]), { dtype }); // [n, 1, 1]

// Initial state
const ns = 12;
const count = Math.min(ns, n);
let initSum = 0;
for (let i = 0; i < count; i++) initSum += y[i];
const mean_y = initSum / count;
const c0_val = (Math.abs(mean_y) * 0.5) ** 2;
const c0 = c0_val === 0 ? 1e7 : c0_val;
const x0 = np.array(
  Array.from({ length: m }, (_, i) => [i === 0 ? mean_y : 0.0]),
  { dtype },
);
const C0 = np.array(
  Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => (i === j ? c0 : 0.0)),
  ),
  { dtype },
);

// Initial parameter guess
const variance =
  y.reduce((s, v) => s + v * v, 0) / n -
  (y.reduce((s, v) => s + v, 0) / n) ** 2;
const s_init = Math.sqrt(Math.abs(variance)) || 1.0;
const w_init = new Array(m).fill(s_init * 0.1);
const arphi_init = [0.5]; // initial guess for AR coefficient
const theta_init = [
  Math.log(s_init),
  ...w_init.map(wi => Math.log(Math.abs(wi) || 0.01)),
  ...arphi_init, // unconstrained
];

console.log(`Model: m=${m}, nTheta=${nTheta}, arInds=${arInds}`);
console.log(`Init: s=${s_init.toFixed(2)}, w=[${w_init.map(v => v.toFixed(2)).join(",")}], arphi=[${arphi_init}]`);

// ── Loss function (duplicated from mle.ts internals for AR) ────────────────

// eslint-disable-next-line -- AD-traced, no using
const buildDiagW_local = (expTheta: np.Array, m_: number, dt: DType, nTh: number): np.Array => {
  let W = np.zeros([m_, m_], { dtype: dt });
  for (let i = 0; i < m_; i++) {
    const maskData = new Array(nTh).fill(0);
    maskData[1 + i] = 1;
    const mask = np.array(maskData, { dtype: dt });
    const wi = np.dot(expTheta, mask);
    const wi2 = np.square(wi);
    const eiData = Array.from({ length: m_ }, (_, j) => (j === i ? [1] : [0]));
    const ei = np.array(eiData, { dtype: dt });
    const eit = np.transpose(ei);
    const outer = np.matmul(ei, eit);
    const scaled = np.multiply(np.reshape(wi2, [1, 1]), outer);
    const W_new = np.add(W, scaled);
    W.dispose();
    W = W_new;
  }
  return W;
};

const buildG_local = (
  G_base: np.Array, theta: np.Array,
  arInds_: number[], m_: number, nSw: number, nTh: number, dt: DType,
): np.Array => {
  const arCol_ = arInds_[0];
  const nar_ = arInds_.length;
  let arContrib = np.zeros([m_, m_], { dtype: dt });
  for (let i = 0; i < nar_; i++) {
    const maskData = new Array(nTh).fill(0);
    maskData[nSw + i] = 1;
    const mask = np.array(maskData, { dtype: dt });
    const phi_i = np.dot(theta, mask);
    const eiData = Array.from({ length: m_ }, (_, j) => j === arInds_[i] ? [1] : [0]);
    const ejData = Array.from({ length: m_ }, (_, j) => [j === arCol_ ? 1 : 0]);
    const ei = np.array(eiData, { dtype: dt });
    const ejt = np.transpose(np.array(ejData, { dtype: dt }));
    const outer = np.matmul(ei, ejt);
    const scaled = np.multiply(np.reshape(phi_i, [1, 1]), outer);
    const newContrib = np.add(arContrib, scaled);
    arContrib.dispose();
    arContrib = newContrib;
  }
  return np.add(G_base, arContrib);
};

type Carry = { x: np.Array; C: np.Array };
type ScanInp = { y: np.Array; V2: np.Array; W: np.Array };

const lossFn = (theta: np.Array): np.Array => {
  const G_eff = buildG_local(G, theta, arInds, m, nSwParams, nTheta, dtype);

  const step = (carry: Carry, inp: ScanInp): [Carry, np.Array] => {
    const { x, C } = carry;
    const { y: yi, V2: V2i, W } = inp;
    const v = np.subtract(yi, np.matmul(F, x));
    const CFt = np.matmul(C, Ft);
    const Cp = np.add(np.matmul(F, CFt), V2i);
    const K = np.divide(np.matmul(G_eff, CFt), Cp);
    const x_next = np.add(np.matmul(G_eff, x), np.matmul(K, v));
    const L = np.subtract(G_eff, np.matmul(K, F));
    const Lt = np.transpose(L);
    const CLt = np.matmul(C, Lt);
    const C_next = np.add(np.matmul(G_eff, CLt), W);
    const lik_t = np.add(np.divide(np.square(v), Cp), np.log(Cp));
    return [{ x: x_next, C: C_next }, np.squeeze(lik_t)];
  };

  const expTheta = np.exp(theta);
  const mask_s = np.array([1, ...new Array(nTheta - 1).fill(0)], { dtype });
  const sVal = np.dot(expTheta, mask_s);
  const V2 = np.reshape(np.square(sVal), [1, 1]);
  const W_mat = buildDiagW_local(expTheta, m, dtype, nTheta);
  const V2_arr = np.multiply(np.ones([n, 1, 1], { dtype }), np.reshape(V2, [1, 1, 1]));
  const W_arr = np.multiply(np.ones([n, 1, 1], { dtype }), np.reshape(W_mat, [1, m, m]));
  const [fc, likTerms] = lax.scan(step, { x: x0, C: C0 }, { y: y_arr, V2: V2_arr, W: W_arr });
  tree.dispose(fc);
  const total = np.sum(likTerms);
  likTerms.dispose();
  return total;
};

// ── Phase 1: Run optimization, capture theta at every iteration ────────────

console.log("\nPhase 1: Full optimization (capturing theta at every iteration)...");

const optimizer = adam(lr);
const optimStep = jit(
  (theta: np.Array, optState: any): [np.Array, any, np.Array] => {
    const [likVal, grad] = valueAndGrad(lossFn)(theta);
    const [updates, newOptState] = optimizer.update(grad, optState);
    const newTheta = applyUpdates(theta, updates);
    return [newTheta, newOptState, likVal];
  },
);

let theta = np.array(theta_init, { dtype });
let optState: any = optimizer.init(theta);

const thetaHistory: number[][] = [[...theta_init]]; // index 0 = before optimization
const likHistory: number[] = [];

let prevLik = Infinity;
let finalIter = 0;
const t0 = performance.now();

for (let iter = 0; iter < maxIter; iter++) {
  const [newTheta, newOptState, likVal] = optimStep(theta, optState);
  const likNum = (await likVal.consumeData() as Float64Array)[0];
  likHistory.push(likNum);

  const td = await newTheta.data() as Float64Array;
  thetaHistory.push(Array.from(td));

  theta.dispose();
  tree.dispose(optState);
  theta = newTheta;
  optState = newOptState;

  const relChange = Math.abs((likNum - prevLik) / (Math.abs(prevLik) + 1e-30));
  if (iter > 0 && relChange < tol) {
    prevLik = likNum;
    finalIter = iter;
    break;
  }
  prevLik = likNum;
  finalIter = iter;

  if ((iter + 1) % 50 === 0) {
    const arphiNow = td[nSwParams];
    console.log(`  iter ${iter + 1}: lik=${likNum.toFixed(2)}, arphi=${arphiNow.toFixed(4)}`);
  }
}

const elapsed = performance.now() - t0;
theta.dispose();
tree.dispose(optState);

const totalIters = finalIter + 1;
console.log(`  Done: ${totalIters} iterations in ${elapsed.toFixed(0)} ms`);

// ── Phase 2: Compute frame sampling ────────────────────────────────────────

const animDuration = elapsed / 1000;
const totalFrames = Math.max(2, Math.round(animDuration * TARGET_FPS));
const stepSize = Math.max(1, Math.round(totalIters / totalFrames));

const sampleIndices: number[] = [0];
for (let i = stepSize; i < totalIters; i += stepSize) sampleIndices.push(i);
if (sampleIndices[sampleIndices.length - 1] !== totalIters) sampleIndices.push(totalIters);

console.log(
  `\nPhase 2: ${animDuration.toFixed(2)}s at ${TARGET_FPS}fps → ` +
    `${sampleIndices.length} frames (step=${stepSize})`,
);

// ── Phase 3: Run dlmFit at sampled iterations ──────────────────────────────

console.log("\nPhase 3: Computing smoothed states at each frame...");

interface Frame {
  iter: number;
  s: number;
  w: number[];
  arphi: number[];
  lik: number | null;
  /** Combined signal F·x for each timestep */
  combined: number[];
  /** Combined signal std: sqrt(F·C·Fᵀ) for each timestep */
  combinedStd: number[];
}

const frames: Frame[] = [];

for (const idx of sampleIndices) {
  const td = thetaHistory[idx];
  const s = Math.exp(td[0]);
  const w = Array.from({ length: m }, (_, i) => Math.exp(td[1 + i]));
  const arphi = [td[nSwParams]]; // AR coefficient (unconstrained)
  const lik = idx === 0 ? null : likHistory[idx - 1];

  // Run dlmFit with fitted arphi
  const fitOptions = { ...options, arphi, fitar: false };
  const fit = await dlmFit(yArr, s, w, dtype, fitOptions);

  // Compute combined signal F·x = x[0] + x[2] + x[4] (for order=1,trig=1,arphi)
  // F = [1, 0, 1, 0, 1] → sum of level + cos + ar
  const combined: number[] = [];
  const combinedStd: number[] = [];
  const F_vec = sys.F; // [1, 0, 1, 0, 1]
  for (let ti = 0; ti < n; ti++) {
    let sig = 0;
    for (let k = 0; k < m; k++) sig += F_vec[k] * fit.x[k][ti];
    combined.push(sig);

    // Variance: F·C·Fᵀ = sum of F[i]*F[j]*C[i][j]
    let var_ = 0;
    for (let k1 = 0; k1 < m; k1++) {
      for (let k2 = 0; k2 < m; k2++) {
        var_ += F_vec[k1] * F_vec[k2] * fit.C[k1][k2][ti];
      }
    }
    combinedStd.push(Math.sqrt(Math.max(0, var_)));
  }

  frames.push({ iter: idx, s, w, arphi, lik, combined, combinedStd });

  const likStr = lik !== null ? lik.toFixed(2) : "—";
  const phiStr = arphi[0].toFixed(4);
  console.log(
    `  Frame ${frames.length}/${sampleIndices.length}: ` +
      `iter=${idx}, s=${s.toFixed(2)}, arphi=${phiStr}, lik=${likStr}`,
  );
}

// Also collect the arphi trajectory for a secondary sparkline
const arphiHistory = thetaHistory.slice(1).map(td => td[nSwParams]);

// ── Save output ────────────────────────────────────────────────────────────

const output = {
  t,
  y,
  n,
  m,
  s_init,
  w_init,
  arphi_init,
  elapsed: Math.round(elapsed),
  iterations: totalIters,
  targetFps: TARGET_FPS,
  holdSeconds: HOLD_SECONDS,
  stepSize,
  likHistory,
  arphiHistory,
  frames,
};

const outPath = resolve(root, "tmp/energy-mle-frames.json");
writeFileSync(outPath, JSON.stringify(output, null, 2));
console.log(`\nSaved ${frames.length} frames to ${outPath}`);
console.log(`  Animation: ${animDuration.toFixed(2)}s play + ${HOLD_SECONDS}s hold = ${(animDuration + HOLD_SECONDS).toFixed(2)}s total cycle`);
