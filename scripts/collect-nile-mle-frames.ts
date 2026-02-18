/**
 * Collect per-frame data for the animated Nile MLE SVG.
 *
 * Runs the MLE optimization manually (same algo as dlmMLE) and captures
 * smoothed states at every Nth iteration for animation frames.
 *
 * Phase 1: Full optimization loop, store theta (3 floats) at every iteration.
 * Phase 2: Compute frame sampling from elapsed time × 10 fps.
 * Phase 3: Run dlmFit only at sampled iterations.
 *
 * Output: tmp/mle-frames.json
 */

import { DType, numpy as np, lax, jit, valueAndGrad, tree, defaultDevice } from "@hamk-uas/jax-js-nonconsuming";
import { adam, applyUpdates } from "@hamk-uas/jax-js-nonconsuming/optax";
import { dlmFit } from "../src/index.ts";
import { dlmGenSys } from "../src/dlmgensys.ts";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { performance } from "node:perf_hooks";

defaultDevice("wasm");

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const input = JSON.parse(readFileSync(resolve(root, "tests/niledemo-in.json"), "utf8"));
const y: number[] = input.y;
const t: number[] = input.t;
const n = y.length;
const dtype = DType.Float64;
const options = { order: 1 };
const maxIter = 300;
const lr = 0.05;
const tol = 1e-6;
const TARGET_FPS = 10;
const HOLD_SECONDS = 2;

// ── System setup (same as dlmMLE) ──────────────────────────────────────────

const sys = dlmGenSys(options);
const m = sys.m; // 2 for order=1

const G = np.array(sys.G, { dtype });
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
const w_init = [s_init * 0.1, s_init * 0.1];
const theta_init = [
  Math.log(s_init),
  ...w_init.map(wi => Math.log(Math.abs(wi) || 0.01)),
];

console.log(`Init: s=${s_init.toFixed(2)}, w=[${w_init.map(v => v.toFixed(2)).join(",")}]`);

// ── Loss function (duplicated from mle.ts internals) ───────────────────────

// eslint-disable-next-line -- AD-traced, no using
const buildDiagW = (expTheta: np.Array, m_: number, dt: DType): np.Array => {
  let W = np.zeros([m_, m_], { dtype: dt });
  for (let i = 0; i < m_; i++) {
    const maskData = new Array(1 + m_).fill(0);
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

type Carry = { x: np.Array; C: np.Array };
type ScanInp = { y: np.Array; V2: np.Array; W: np.Array };

const step = (carry: Carry, inp: ScanInp): [Carry, np.Array] => {
  const { x, C } = carry;
  const { y: yi, V2: V2i, W } = inp;
  const v = np.subtract(yi, np.matmul(F, x));
  const CFt = np.matmul(C, Ft);
  const Cp = np.add(np.matmul(F, CFt), V2i);
  const K = np.divide(np.matmul(G, CFt), Cp);
  const x_next = np.add(np.matmul(G, x), np.matmul(K, v));
  const L = np.subtract(G, np.matmul(K, F));
  const Lt = np.transpose(L);
  const CLt = np.matmul(C, Lt);
  const C_next = np.add(np.matmul(G, CLt), W);
  const lik_t = np.add(np.divide(np.square(v), Cp), np.log(Cp));
  return [{ x: x_next, C: C_next }, np.squeeze(lik_t)];
};

const lossFn = (theta: np.Array): np.Array => {
  const expTheta = np.exp(theta);
  const mask_s = np.array([1, ...new Array(m).fill(0)], { dtype });
  const sVal = np.dot(expTheta, mask_s);
  const V2 = np.reshape(np.square(sVal), [1, 1]);
  const W = buildDiagW(expTheta, m, dtype);
  const V2_arr = np.multiply(
    np.ones([n, 1, 1], { dtype }),
    np.reshape(V2, [1, 1, 1]),
  );
  const W_arr = np.multiply(
    np.ones([n, 1, 1], { dtype }),
    np.reshape(W, [1, m, m]),
  );
  const [fc, likTerms] = lax.scan(
    step,
    { x: x0, C: C0 },
    { y: y_arr, V2: V2_arr, W: W_arr },
  );
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

  // Capture theta (3 floats — negligible overhead)
  const td = await newTheta.data() as Float64Array;
  thetaHistory.push(Array.from(td));

  theta.dispose();
  tree.dispose(optState);
  theta = newTheta;
  optState = newOptState;

  const relChange = Math.abs(
    (likNum - prevLik) / (Math.abs(prevLik) + 1e-30),
  );
  if (iter > 0 && relChange < tol) {
    prevLik = likNum;
    finalIter = iter;
    break;
  }
  prevLik = likNum;
  finalIter = iter;
}

const elapsed = performance.now() - t0;
theta.dispose();
tree.dispose(optState);

const totalIters = finalIter + 1;
console.log(`  Done: ${totalIters} iterations in ${elapsed.toFixed(0)} ms`);

// ── Phase 2: Compute frame sampling ────────────────────────────────────────

const animDuration = elapsed / 1000; // seconds
const totalFrames = Math.max(2, Math.round(animDuration * TARGET_FPS));
const stepSize = Math.max(1, Math.round(totalIters / totalFrames));

// Sample indices: always include 0 (initial) and totalIters (final)
const sampleIndices: number[] = [0];
for (let i = stepSize; i < totalIters; i += stepSize) {
  sampleIndices.push(i);
}
if (sampleIndices[sampleIndices.length - 1] !== totalIters) {
  sampleIndices.push(totalIters);
}

console.log(
  `\nPhase 2: ${animDuration.toFixed(2)}s at ${TARGET_FPS}fps → ` +
    `${sampleIndices.length} frames (step=${stepSize}, indices: [${sampleIndices[0]}..${sampleIndices[sampleIndices.length - 1]}])`,
);

// ── Phase 3: Run dlmFit at sampled iterations ──────────────────────────────

console.log("\nPhase 3: Computing smoothed states at each frame...");

interface Frame {
  iter: number;
  s: number;
  w: number[];
  lik: number | null;
  level: number[];
  std: number[];
}

const frames: Frame[] = [];

for (const idx of sampleIndices) {
  const td = thetaHistory[idx];
  const s = Math.exp(td[0]);
  const w = Array.from({ length: m }, (_, i) => Math.exp(td[1 + i]));
  // likHistory[i] = lik after iteration i (0-indexed)
  // thetaHistory[0] = initial, thetaHistory[i] = after iteration i-1
  // So thetaHistory[idx] corresponds to likHistory[idx-1]
  const lik = idx === 0 ? null : likHistory[idx - 1];

  const fit = await dlmFit(yArr, s, w, dtype, options);
  const level = Array.from(fit.x[0]);
  const std = fit.xstd.map((row: any) => row[0] as number);
  frames.push({ iter: idx, s, w, lik, level, std });

  const likStr = lik !== null ? lik.toFixed(2) : "—";
  console.log(
    `  Frame ${frames.length}/${sampleIndices.length}: ` +
      `iter=${idx}, s=${s.toFixed(2)}, w=[${w.map(v => v.toFixed(2)).join(",")}], lik=${likStr}`,
  );
}

// ── Save output ────────────────────────────────────────────────────────────

const output = {
  t,
  y,
  n,
  m,
  s_init,
  w_init,
  elapsed: Math.round(elapsed),
  iterations: totalIters,
  targetFps: TARGET_FPS,
  holdSeconds: HOLD_SECONDS,
  stepSize,
  likHistory,
  frames,
};

const outPath = resolve(root, "tmp/mle-frames.json");
writeFileSync(outPath, JSON.stringify(output, null, 2));
console.log(`\nSaved ${frames.length} frames to ${outPath}`);
console.log(`  Animation: ${animDuration.toFixed(2)}s play + ${HOLD_SECONDS}s hold = ${(animDuration + HOLD_SECONDS).toFixed(2)}s total cycle`);
