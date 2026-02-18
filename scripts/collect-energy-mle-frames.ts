/**
 * Collect per-frame data for the animated energy MLE SVG (with AR estimation).
 *
 * Uses `dlmMLE` with `onIteration` callback to capture theta at every
 * iteration, then samples frames and runs `dlmFit` at each for smoothed states.
 *
 * Output: tmp/energy-mle-frames.json
 */

import { DType, defaultDevice } from "@hamk-uas/jax-js-nonconsuming";
import { dlmFit } from "../src/index.ts";
import { dlmMLE } from "../src/mle.ts";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";

defaultDevice("wasm");

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const input = JSON.parse(readFileSync(resolve(root, "tests/energy-in.json"), "utf8"));
const y: number[] = input.y;
const n = y.length;
const t: number[] = Array.from({ length: n }, (_, i) => i + 1);
const dtype = DType.Float64;

// Model: trend + seasonal + AR(1), with AR coefficient estimation
const options = { order: 1, trig: 1, ns: 12, arphi: [0.5], fitar: true };
const m = 5; // 2 (poly order=1) + 2 (trig k=1) + 1 (AR)
const nSwParams = 1 + m; // theta[0]=log(s), theta[1..5]=log(w[i])
const maxIter = 300;
const lr = 0.02;
const tol = 1e-6;
const TARGET_FPS = 10;
const HOLD_SECONDS = 2;

// ── Phase 1: Run optimization via dlmMLE, capture theta at every iteration ─

console.log("Phase 1: Full optimization (capturing theta at every iteration)...");

const thetaHistory: number[][] = [];

const mle = await dlmMLE(y, options, undefined, maxIter, lr, tol, dtype, {
  onInit: (theta) => {
    thetaHistory.push(Array.from(theta));
  },
  onIteration: (_iter, theta, _lik) => {
    thetaHistory.push(Array.from(theta));
  },
});

const elapsed = mle.elapsed;
const totalIters = mle.iterations;
const likHistory = mle.likHistory;

// Extract arphi history from theta (AR coeff is at index nSwParams, unconstrained)
const arphiHistory = thetaHistory.slice(1).map((td) => td[nSwParams]);

console.log(`  Done: ${totalIters} iterations in ${elapsed.toFixed(0)} ms`);
console.log(`  Final: s=${mle.s.toFixed(4)}, arphi=${mle.arphi?.[0]?.toFixed(4)}`);

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
  combined: number[];
  combinedStd: number[];
}

const yArr = Float64Array.from(y);
const frames: Frame[] = [];

// F for this model: indices where F[j]=1 are the "observable" states
// For order=1, trig=1, ns=12, arphi: F = [1, 0, 1, 0, 1]
const fInds = [0, 2, 4];

for (const idx of sampleIndices) {
  const td = thetaHistory[idx];
  const s = Math.exp(td[0]);
  const w = Array.from({ length: m }, (_, i) => Math.exp(td[1 + i]));
  const arphi = [td[nSwParams]]; // unconstrained AR coeff
  const lik = idx === 0 ? null : likHistory[idx - 1];

  // Run dlmFit with the AR coefficient at this iteration
  const fitOpts = { ...options, arphi, fitar: false };
  const fit = await dlmFit(yArr, s, w, dtype, fitOpts);

  // Combined signal: F·x = x[0] + x[2] + x[4]
  const combined = Array.from({ length: n }, (_, i) =>
    fInds.reduce((sum, fi) => sum + fit.x[fi][i], 0),
  );

  // Combined std: sqrt(sum of Var + 2*sum of Cov for all fInds pairs)
  const combinedStd = Array.from({ length: n }, (_, i) => {
    let variance = 0;
    for (const fi of fInds) variance += fit.C[fi][fi][i];
    for (let a = 0; a < fInds.length; a++) {
      for (let b = a + 1; b < fInds.length; b++) {
        variance += 2 * fit.C[fInds[a]][fInds[b]][i];
      }
    }
    return Math.sqrt(Math.max(0, variance));
  });

  frames.push({ iter: idx, s, w, arphi, lik, combined, combinedStd });

  const likStr = lik !== null ? lik.toFixed(2) : "—";
  console.log(
    `  Frame ${frames.length}/${sampleIndices.length}: ` +
      `iter=${idx}, s=${s.toFixed(2)}, φ=${arphi[0].toFixed(3)}, lik=${likStr}`,
  );
}

// ── Save output ────────────────────────────────────────────────────────────

const output = {
  t, y, n, m,
  s_init: Math.exp(thetaHistory[0][0]),
  w_init: Array.from({ length: m }, (_, i) => Math.exp(thetaHistory[0][1 + i])),
  arphi_init: [thetaHistory[0][nSwParams]],
  elapsed: Math.round(elapsed),
  iterations: totalIters,
  targetFps: TARGET_FPS,
  holdSeconds: HOLD_SECONDS,
  stepSize,
  likHistory,
  arphiHistory,
  frames,
};

const outDir = resolve(root, "tmp");
mkdirSync(outDir, { recursive: true });
const outPath = resolve(outDir, "energy-mle-frames.json");
writeFileSync(outPath, JSON.stringify(output, null, 2));
console.log(`\nSaved ${frames.length} frames to ${outPath}`);
console.log(`  Animation: ${animDuration.toFixed(2)}s play + ${HOLD_SECONDS}s hold = ${(animDuration + HOLD_SECONDS).toFixed(2)}s total cycle`);
