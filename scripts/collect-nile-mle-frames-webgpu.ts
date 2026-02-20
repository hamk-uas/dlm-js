/**
 * Collect per-frame data for the animated Nile MLE SVG — WebGPU variant.
 *
 * Uses WebGPU + Float32. dlmMLE auto-dispatches to makeKalmanLossAssoc
 * (exact 5-tuple + lax.associativeScan) when device=webgpu + dtype=Float32.
 *
 * Must be run with Deno (WebGPU requires --unstable-webgpu):
 *   deno run --unstable-webgpu --allow-read --allow-write --allow-env --allow-run \
 *     scripts/collect-nile-mle-frames-webgpu.ts
 *
 * Output: tmp/mle-frames-nile-webgpu.json
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

const input = JSON.parse(readFileSync(resolve(root, "tests/niledemo-in.json"), "utf8"));
const y: number[] = input.y;
const t: number[] = input.t;
const n = y.length;
const dtype = DType.Float32;
const options = { order: 1 };
const m = 2; // order=1 → m=2
const maxIter = 300;
const lr = 0.05;
const tol = 1e-6;
const TARGET_FPS = 10;
const HOLD_SECONDS = 2;

interface Frame {
  iter: number;
  s: number;
  w: number[];
  lik: number | null;
  level: number[];
  std: number[];
  ystd: number[];
}

// ── Phase 1: Full optimization ─────────────────────────────────────────────

console.log("═══ WebGPU Nile MLE collector ═══");
console.log("Phase 1: Full optimization...");

const thetaHistory: number[][] = [];

const mle = await dlmMLE(y, options, undefined, maxIter, lr, tol, dtype, {
  onInit: (theta) => { thetaHistory.push(Array.from(theta)); },
  onIteration: (_iter, theta, _lik) => { thetaHistory.push(Array.from(theta)); },
});

const elapsed = mle.elapsed;
const totalIters = mle.iterations;
const likHistory = mle.likHistory;

console.log(`  Done: ${totalIters} iterations in ${elapsed.toFixed(0)} ms`);

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
  const lik = idx === 0 ? null : (likHistory[idx - 1] as number);

  const fit = await dlmFit(yArr, s, w, dtype, options);
  const level = Array.from(fit.x[0] as ArrayLike<number>);
  const std = (fit.xstd as any[]).map((row: any) => row[0] as number);
  const ystd = Array.from(fit.ystd as ArrayLike<number>);
  frames.push({ iter: idx, s, w, lik, level, std, ystd });

  const likStr = lik !== null ? lik.toFixed(2) : "—";
  console.log(
    `  Frame ${frames.length}/${sampleIndices.length}: ` +
      `iter=${idx}, s=${s.toFixed(2)}, w=[${w.map(v => v.toFixed(2)).join(",")}], lik=${likStr}`,
  );
}

// ── Save output ────────────────────────────────────────────────────────────

const output = {
  variant: "webgpu",
  t, y, n, m,
  s_init: Math.exp(thetaHistory[0][0]),
  w_init: Array.from({ length: m }, (_, i) => Math.exp(thetaHistory[0][1 + i])),
  elapsed: Math.round(elapsed),
  jitMs: mle.jitMs,
  iterations: totalIters,
  targetFps: TARGET_FPS,
  holdSeconds: HOLD_SECONDS,
  stepSize,
  likHistory,
  frames,
};

mkdirSync(resolve(root, "tmp"), { recursive: true });
const outPath = resolve(root, "tmp/mle-frames-nile-webgpu.json");
writeFileSync(outPath, JSON.stringify(output, null, 2));
console.log(`Saved ${frames.length} frames to ${outPath}`);
console.log(`  Animation: ${animDuration.toFixed(2)}s play + ${HOLD_SECONDS}s hold = ${(animDuration + HOLD_SECONDS).toFixed(2)}s total cycle`);

// ── Write timing sidecar ───────────────────────────────────────────────────

mkdirSync(sidecarDir, { recursive: true });
writeFileSync(
  resolve(sidecarDir, "collect-nile-mle-frames-webgpu.json"),
  JSON.stringify({ elapsed: Math.round(elapsed), iterations: totalIters, lik: mle.lik }, null, 2) + "\n",
);
console.log(`Timing sidecar written.`);
console.log(`\nSummary: ${totalIters} iters, ${elapsed.toFixed(0)} ms, lik=${mle.lik.toFixed(2)}`);
