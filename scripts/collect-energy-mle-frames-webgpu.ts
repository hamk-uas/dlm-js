/**
 * Collect per-frame data for the animated energy MLE SVG — WebGPU variant.
 *
 * Uses WebGPU + Float32. dlmMLE auto-dispatches to makeKalmanLossAssoc
 * (DARE steady-state + lax.associativeScan) when device=webgpu + dtype=Float32.
 *
 * Must be run with Deno (WebGPU requires --unstable-webgpu):
 *   deno run --unstable-webgpu --allow-read --allow-write --allow-env --allow-run \
 *     scripts/collect-energy-mle-frames-webgpu.ts
 *
 * Output: tmp/mle-frames-energy-webgpu.json
 */

import { DType, defaultDevice, init } from "../node_modules/@hamk-uas/jax-js-nonconsuming/dist/index.js";
import { dlmFit } from "../src/index.ts";
import { dlmMLE } from "../src/mle.ts";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const sidecarDir = resolve(root, "assets/timings");

await init("webgpu");
defaultDevice("webgpu");

const input = JSON.parse(readFileSync(resolve(root, "tests/energy-in.json"), "utf8"));
const y: number[] = input.y;
const n = y.length;
const t: number[] = Array.from({ length: n }, (_, i) => i + 1);
const dtype = DType.Float32;

// Model: trend + seasonal + AR(1), with AR coefficient estimation
const options = { order: 1, trig: 1, ns: 12, arphi: [0.5], fitar: true };
const m = 5; // 2 (poly order=1) + 2 (trig k=1) + 1 (AR)
const nSwParams = 1 + m; // theta[0]=log(s), theta[1..5]=log(w[i])
const maxIter = 300;
const lr = 0.02;
const tol = 1e-6;
const TARGET_FPS = 10;
const HOLD_SECONDS = 2;

// F for this model: indices where F[j]=1 are the "observable" states
// For order=1, trig=1, ns=12, arphi: F = [1, 0, 1, 0, 1]
const fInds = [0, 2, 4];

interface Frame {
  iter: number;
  s: number;
  w: number[];
  arphi: number[];
  lik: number | null;
  combined: number[];
  combinedStd: number[];
}

// ── Phase 1: Full optimization ─────────────────────────────────────────────

console.log("═══ WebGPU Energy MLE collector ═══");
console.log("Phase 1: Full optimization...");

const thetaHistory: number[][] = [];

const mle = await dlmMLE(y, options, undefined, maxIter, lr, tol, dtype, {
  onInit: (theta) => { thetaHistory.push(Array.from(theta)); },
  onIteration: (_iter, theta, _lik) => { thetaHistory.push(Array.from(theta)); },
});

const elapsed = mle.elapsed;
const totalIters = mle.iterations;
const likHistory = mle.likHistory;

const arphiHistory = thetaHistory.slice(1).map((td) => td[nSwParams]);

console.log(`  Done: ${totalIters} iterations in ${elapsed.toFixed(0)} ms`);
console.log(`  Final: s=${mle.s.toFixed(4)}, arphi=${mle.arphi?.[0]?.toFixed(4)}`);

// ── Phase 2: Frame sampling ────────────────────────────────────────────────

const animDuration = elapsed / 1000;
const totalFrames = Math.max(2, Math.round(animDuration * TARGET_FPS));
const stepSize = Math.max(1, Math.round(totalIters / totalFrames));

const sampleIndices: number[] = [0];
for (let i = stepSize; i < totalIters; i += stepSize) sampleIndices.push(i);
if (sampleIndices[sampleIndices.length - 1] !== totalIters) sampleIndices.push(totalIters);

console.log(
  `Phase 2: ${animDuration.toFixed(2)}s at ${TARGET_FPS}fps → ` +
    `${sampleIndices.length} frames (step=${stepSize})`,
);

// ── Phase 3: dlmFit at each sampled iteration ─────────────────────────────

console.log("Phase 3: Computing smoothed states at each frame...");

const yArr = Float32Array.from(y);
const frames: Frame[] = [];

for (const idx of sampleIndices) {
  const td = thetaHistory[idx];
  const s = Math.exp(td[0]);
  const w = Array.from({ length: m }, (_, i) => Math.exp(td[1 + i]));
  const arphi = [td[nSwParams]];
  const lik = idx === 0 ? null : (likHistory[idx - 1] as number);

  const fitOpts = { ...options, arphi, fitar: false };
  const fit = await dlmFit(yArr, s, w, dtype, fitOpts);

  const combined = Array.from({ length: n }, (_, i) =>
    fInds.reduce((sum, fi) => sum + (fit.x[fi] as ArrayLike<number>)[i], 0),
  );

  const combinedStd = Array.from({ length: n }, (_, i) => {
    let variance = 0;
    for (const fi of fInds) variance += (fit.C as any)[fi][fi][i];
    for (let a = 0; a < fInds.length; a++) {
      for (let b = a + 1; b < fInds.length; b++) {
        variance += 2 * (fit.C as any)[fInds[a]][fInds[b]][i];
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
  variant: "webgpu",
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

mkdirSync(resolve(root, "tmp"), { recursive: true });
const outPath = resolve(root, "tmp/mle-frames-energy-webgpu.json");
writeFileSync(outPath, JSON.stringify(output, null, 2));
console.log(`Saved ${frames.length} frames to ${outPath}`);
console.log(`  Animation: ${animDuration.toFixed(2)}s play + ${HOLD_SECONDS}s hold = ${(animDuration + HOLD_SECONDS).toFixed(2)}s total cycle`);

// ── Write timing sidecar ───────────────────────────────────────────────────

mkdirSync(sidecarDir, { recursive: true });
writeFileSync(
  resolve(sidecarDir, "collect-energy-mle-frames-webgpu.json"),
  JSON.stringify({ elapsed: Math.round(elapsed), iterations: totalIters, lik: mle.lik }, null, 2) + "\n",
);
console.log(`Timing sidecar written.`);
console.log(`\nSummary: ${totalIters} iters, ${elapsed.toFixed(0)} ms, lik=${mle.lik.toFixed(2)}`);
