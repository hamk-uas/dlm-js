/**
 * Collect per-frame data for the animated Nile MLE SVG.
 *
 * Runs `dlmMLE` twice — once with sequential `lax.scan` (default) and once
 * with `forceAssocScan` (DARE + `lax.associativeScan`) — capturing theta at
 * every iteration, sampling frames, and running `dlmFit` at each.
 *
 * Output:
 *   tmp/mle-frames-nile-scan.json   (sequential scan variant)
 *   tmp/mle-frames-nile-assoc.json  (associativeScan variant)
 */

import { DType, defaultDevice } from "@hamk-uas/jax-js-nonconsuming";
import { dlmFit } from "../src/index.ts";
import { dlmMLE } from "../src/mle.ts";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { withLeakCheck } from "./lib/leak-utils.ts";
import { writeTimingsSidecar } from "./lib/timing-sidecar.ts";

defaultDevice("wasm");

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const input = JSON.parse(readFileSync(resolve(root, "tests/niledemo-in.json"), "utf8"));
const y: number[] = input.y;
const t: number[] = input.t;
const n = y.length;
const dtype = DType.Float64;
const options = { order: 1 };
const m = 2; // order=1 → m=2
const maxIter = 300;
const lr = 0.05;
const tol = 1e-6;
const TARGET_FPS = 10;
const HOLD_SECONDS = 2;

// ── Shared types ───────────────────────────────────────────────────────────

interface Frame {
  iter: number;
  s: number;
  w: number[];
  lik: number | null;
  level: number[];
  std: number[];    // xstd[:,0]: state (level) uncertainty std
  ystd: number[];   // observation prediction std = sqrt(C_00 + s^2)
}

// ── Core collection function ───────────────────────────────────────────────

async function collectVariant(variantName: string, forceAssocScan: boolean) {
  console.log(`\n═══ Variant: ${variantName} (forceAssocScan=${forceAssocScan}) ═══`);

  // Phase 1: Run optimization, capture theta at every iteration
  console.log("Phase 1: Full optimization (capturing theta at every iteration)...");

  const thetaHistory: number[][] = [];

  const mle = await withLeakCheck(() =>
    dlmMLE(y, options, undefined, maxIter, lr, tol, dtype, {
      onInit: (theta) => {
        thetaHistory.push(Array.from(theta));
      },
      onIteration: (_iter, theta, _lik) => {
        thetaHistory.push(Array.from(theta));
      },
    },
    undefined, undefined, undefined,
    forceAssocScan,
    )
  );

  const elapsed = mle.elapsed;
  const totalIters = mle.iterations;
  const likHistory = mle.likHistory;

  console.log(`  Done: ${totalIters} iterations in ${elapsed.toFixed(0)} ms`);

  // Phase 2: Compute frame sampling
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

  // Phase 3: Run dlmFit at sampled iterations
  console.log("Phase 3: Computing smoothed states at each frame...");

  const yArr = Float64Array.from(y);
  const frames: Frame[] = [];

  for (const idx of sampleIndices) {
    const td = thetaHistory[idx];
    const s = Math.exp(td[0]);
    const w = Array.from({ length: m }, (_, i) => Math.exp(td[1 + i]));
    const lik = idx === 0 ? null : likHistory[idx - 1];

    const fit = await withLeakCheck(() => dlmFit(yArr, s, w, dtype, options));
    const level = Array.from(fit.x[0]);
    const std = fit.xstd.map((row: any) => row[0] as number);
    const ystd = Array.from(fit.ystd as ArrayLike<number>);
    frames.push({ iter: idx, s, w, lik, level, std, ystd });

    const likStr = lik !== null ? lik.toFixed(2) : "—";
    console.log(
      `  Frame ${frames.length}/${sampleIndices.length}: ` +
        `iter=${idx}, s=${s.toFixed(2)}, w=[${w.map(v => v.toFixed(2)).join(",")}], lik=${likStr}`,
    );
  }

  // Save output
  const output = {
    variant: variantName,
    t, y, n, m,
    s_init: Math.exp(thetaHistory[0][0]),
    w_init: Array.from({ length: m }, (_, i) => Math.exp(thetaHistory[0][1 + i])),
    elapsed: Math.round(elapsed),
    iterations: totalIters,
    targetFps: TARGET_FPS,
    holdSeconds: HOLD_SECONDS,
    stepSize,
    likHistory,
    frames,
  };

  const outDir = resolve(root, "tmp");
  mkdirSync(outDir, { recursive: true });
  const suffix = variantName;
  const outPath = resolve(outDir, `mle-frames-nile-${suffix}.json`);
  writeFileSync(outPath, JSON.stringify(output, null, 2));
  console.log(`Saved ${frames.length} frames to ${outPath}`);
  console.log(`  Animation: ${animDuration.toFixed(2)}s play + ${HOLD_SECONDS}s hold = ${(animDuration + HOLD_SECONDS).toFixed(2)}s total cycle`);

  // Write timing sidecar
  const sidecar = suffix === "scan" ? "collect-nile-mle-frames" : `collect-nile-mle-frames-${suffix}`;
  writeTimingsSidecar(sidecar, { elapsed: Math.round(elapsed), iterations: totalIters, lik: mle.lik });

  return { elapsed, totalIters, lik: mle.lik };
}

// ── Run both variants ──────────────────────────────────────────────────────

const scanResult = await collectVariant("scan", false);
const assocResult = await collectVariant("assoc", true);

console.log("\n═══ Summary ═══");
console.log(`  scan:  ${scanResult.totalIters} iters, ${scanResult.elapsed.toFixed(0)} ms, lik=${scanResult.lik.toFixed(2)}`);
console.log(`  assoc: ${assocResult.totalIters} iters, ${assocResult.elapsed.toFixed(0)} ms, lik=${assocResult.lik.toFixed(2)}`);
