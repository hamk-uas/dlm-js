/**
 * Collect per-frame data for the animated energy MLE SVG (with AR estimation).
 *
 * Runs `dlmMLE` twice — once with sequential `lax.scan` (default) and once
 * with `forceAssocScan` (exact 5-tuple + `lax.associativeScan`) — capturing theta at
 * every iteration, sampling frames, and running `dlmFit` at each.
 *
 * Output:
 *   tmp/mle-frames-energy-scan.json   (sequential scan variant)
 *   tmp/mle-frames-energy-assoc.json  (associativeScan variant)
 */

import { defaultDevice } from "@hamk-uas/jax-js-nonconsuming";
import { dlmFit } from "../src/index.ts";
import { dlmMLE } from "../src/mle.ts";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { withLeakCheck } from "./lib/leak-utils.ts";
import { writeTimingsSidecar } from "./lib/timing-sidecar.ts";

defaultDevice("wasm");

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const input = JSON.parse(readFileSync(resolve(root, "tests/energy-in.json"), "utf8"));
const y: number[] = input.y;
const n = y.length;
const t: number[] = Array.from({ length: n }, (_, i) => i + 1);

// Model: trend + seasonal + AR(1), with AR coefficient estimation
const options = { order: 1, harmonics: 1, seasonLength: 12, arCoefficients: [0.5], fitAr: true };
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

// ── Shared types ───────────────────────────────────────────────────────────

interface Frame {
  iter: number;
  s: number;
  w: number[];
  arphi: number[];
  lik: number | null;
  combined: number[];
  combinedStd: number[];
}

// ── Core collection function ───────────────────────────────────────────────

async function collectVariant(variantName: string, forceAssocScan: boolean) {
  console.log(`\n═══ Variant: ${variantName} (forceAssocScan=${forceAssocScan}) ═══`);

  // Phase 1: Run optimization, capture theta at every iteration
  console.log("Phase 1: Full optimization (capturing theta at every iteration)...");

  const thetaHistory: number[][] = [];

  const mle = await withLeakCheck(() =>
    dlmMLE(y, {
      ...options, maxIter, lr, tol, dtype: 'f64',
      callbacks: {
        onInit: (theta) => {
          thetaHistory.push(Array.from(theta));
        },
        onIteration: (_iter, theta, _lik) => {
          thetaHistory.push(Array.from(theta));
        },
      },
      algorithm: forceAssocScan ? 'assoc' : undefined,
    })
  );

  const elapsed = mle.elapsed;
  const jitMs = mle.compilationMs;
  const totalIters = mle.iterations;
  const likHistory = mle.devianceHistory;

  // Extract arphi history from theta (AR coeff is at index nSwParams, unconstrained)
  const arphiHistory = thetaHistory.slice(1).map((td) => td[nSwParams]);

  console.log(`  Done: ${totalIters} iterations in ${elapsed.toFixed(0)} ms`);
console.log(`  Final: s=${mle.obsStd.toFixed(4)}, arphi=${mle.arCoefficients?.[0]?.toFixed(4)}`);

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
    const arphi = [td[nSwParams]]; // unconstrained AR coeff
    const lik = idx === 0 ? null : likHistory[idx - 1];

    // Run dlmFit with the AR coefficient at this iteration
    const fitOpts = { ...options, arCoefficients: arphi, fitAr: false };
    const fit = await withLeakCheck(() => dlmFit(yArr, { obsStd: s, processStd: w, dtype: 'f64', ...fitOpts }));

    // Combined signal: F·x = x[0] + x[2] + x[4]
    const combined = Array.from({ length: n }, (_, i) =>
      fInds.reduce((sum, fi) => sum + fit.smoothed.get(i, fi), 0),
    );

    // Combined std: sqrt(sum of Var + 2*sum of Cov for all fInds pairs)
    const combinedStd = Array.from({ length: n }, (_, i) => {
      let variance = 0;
      for (const fi of fInds) variance += fit.smoothedCov.get(i, fi, fi);
      for (let a = 0; a < fInds.length; a++) {
        for (let b = a + 1; b < fInds.length; b++) {
          variance += 2 * fit.smoothedCov.get(i, fInds[a], fInds[b]);
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

  // Save output
  const output = {
    variant: variantName,
    t, y, n, m,
    s_init: Math.exp(thetaHistory[0][0]),
    w_init: Array.from({ length: m }, (_, i) => Math.exp(thetaHistory[0][1 + i])),
    arphi_init: [thetaHistory[0][nSwParams]],
    elapsed: Math.round(elapsed),
    jitMs,
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
  const suffix = variantName;
  const outPath = resolve(outDir, `mle-frames-energy-${suffix}.json`);
  writeFileSync(outPath, JSON.stringify(output, null, 2));
  console.log(`Saved ${frames.length} frames to ${outPath}`);
  console.log(`  Animation: ${animDuration.toFixed(2)}s play + ${HOLD_SECONDS}s hold = ${(animDuration + HOLD_SECONDS).toFixed(2)}s total cycle`);

  // Write timing sidecar
  const sidecar = suffix === "scan" ? "collect-energy-mle-frames" : `collect-energy-mle-frames-${suffix}`;
  writeTimingsSidecar(sidecar, { elapsed: Math.round(elapsed), iterations: totalIters, lik: mle.deviance });

  return { elapsed, totalIters, lik: mle.deviance };
}

// ── Run both variants ──────────────────────────────────────────────────────

const scanResult = await collectVariant("scan", false);
const assocResult = await collectVariant("assoc", true);

console.log("\n═══ Summary ═══");
console.log(`  scan:  ${scanResult.totalIters} iters, ${scanResult.elapsed.toFixed(0)} ms, lik=${scanResult.lik.toFixed(2)}`);
console.log(`  assoc: ${assocResult.totalIters} iters, ${assocResult.elapsed.toFixed(0)} ms, lik=${assocResult.lik.toFixed(2)}`);
