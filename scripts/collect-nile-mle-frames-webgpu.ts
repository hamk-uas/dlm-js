/**
 * Collect per-frame data for the animated Nile MLE SVG — WebGPU variant.
 *
 * Uses WebGPU + Float32. dlmMLE auto-dispatches to makeKalmanLossAssoc
 * (exact 5-tuple + lax.associativeScan) when device=webgpu + dtype=Float32.
 *
 * Runs in Chromium browser mode via @vitest/browser-playwright:
 *   GPU=nvidia bash scripts/gpu-test.sh run scripts/collect-nile-mle-frames-webgpu.ts
 *
 * Output: tmp/mle-frames-nile-webgpu.json
 */

import { describe, it } from 'vitest';
import { commands } from 'vitest/browser';
import { defaultDevice, init } from "@hamk-uas/jax-js-nonconsuming";
import { dlmFit } from "../src/index.ts";
import { dlmMLE } from "../src/mle.ts";

const TIMEOUT_MS = 120_000;

const timedOut = Symbol("timedOut");
function withTimeout<T>(p: Promise<T>, ms: number): Promise<T | typeof timedOut> {
  return Promise.race([p, new Promise<typeof timedOut>((r) => setTimeout(() => r(timedOut), ms))]);
}

interface Frame {
  iter: number;
  s: number;
  w: number[];
  lik: number | null;
  level: number[];
  std: number[];
  ystd: number[];
}

describe('collect-nile-mle-frames-webgpu', () => {
  it('collects MLE animation frames on WebGPU', async () => {
    await init("webgpu");
    defaultDevice("webgpu");

    const input = JSON.parse(await commands.readFile("tests/niledemo-in.json"));
    const y: number[] = input.y;
    const t: number[] = input.t;
    const n = y.length;
    const options = { order: 1 };
    const m = 2;
    const maxIter = 50;
    const tol = 1e-6;
    const TARGET_FPS = 10;
    const HOLD_SECONDS = 2;

    async function writeSidecar(data: { elapsed: number | null; iterations: number | null; lik: number | null }) {
      await commands.writeFile(
        "assets/timings/collect-nile-mle-frames-webgpu.json",
        JSON.stringify(data, null, 2) + "\n",
      );
    }

    // ── Phase 1: Full optimization (with timeout) ─────────────────────────────

    console.log("═══ WebGPU Nile MLE collector ═══");
    console.log(`Phase 1: Full optimization (timeout ${TIMEOUT_MS / 1000}s)...`);

    const thetaHistory: number[][] = [];

    const mleResult = await withTimeout(
      dlmMLE(y, {
        ...options, maxIter, tol, dtype: 'f32', optimizer: 'natural' as const,
        callbacks: {
          onInit: (theta) => { thetaHistory.push(Array.from(theta)); },
          onIteration: (_iter, theta, _lik) => { thetaHistory.push(Array.from(theta)); },
        },
      }),
      TIMEOUT_MS,
    );

    if (mleResult === timedOut) {
      console.log(`  TIMEOUT after ${TIMEOUT_MS / 1000}s — writing null sidecar`);
      await writeSidecar({ elapsed: null, iterations: null, lik: null });
      console.log("Timing sidecar written (null — timed out).");
      return;
    }

    const mle = mleResult;
    const elapsed = mle.elapsed;
    const totalIters = mle.iterations;
    const likHistory = mle.devianceHistory;

    console.log(`  Done: ${totalIters} iterations in ${elapsed.toFixed(0)} ms`);

    await writeSidecar({ elapsed: Math.round(elapsed), iterations: totalIters, lik: mle.deviance });
    console.log("Timing sidecar written (Phase 1 complete).");

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

    // ── Phase 3: dlmFit at each sampled iteration (budget-capped) ───────────

    const PHASE3_BUDGET_MS = 120_000;
    const phase3Start = performance.now();
    console.log(`Phase 3: Computing smoothed states at each frame (budget ${PHASE3_BUDGET_MS / 1000}s)...`);

    const yArr = Float32Array.from(y);
    const frames: Frame[] = [];

    for (const idx of sampleIndices) {
      if (performance.now() - phase3Start > PHASE3_BUDGET_MS) {
        console.log(`  Phase 3 budget exceeded after ${frames.length} frames — stopping`);
        break;
      }

      const td = thetaHistory[idx];
      const s = Math.exp(td[0]);
      const w = Array.from({ length: m }, (_, i) => Math.exp(td[1 + i]));
      const lik = idx === 0 ? null : (likHistory[idx - 1] as number);

      const fit = await dlmFit(yArr, { obsStd: s, processStd: w, dtype: 'f32', ...options });
      const level = Array.from(fit.smoothed.series(0));
      const std = Array.from({ length: n }, (_, t) => fit.smoothedStd.get(t, 0));
      const ystd = Array.from(fit.ystd);
      frames.push({ iter: idx, s, w, lik, level, std, ystd });

      const likStr = lik !== null ? lik.toFixed(2) : "—";
      console.log(
        `  Frame ${frames.length}/${sampleIndices.length}: ` +
          `iter=${idx}, s=${s.toFixed(2)}, w=[${w.map(v => v.toFixed(2)).join(",")}], lik=${likStr}`,
      );
    }

    // ── Save output ──────────────────────────────────────────────────────────

    const output = {
      variant: "webgpu",
      optimizer: "natural",
      t, y, n, m,
      s_init: Math.exp(thetaHistory[0][0]),
      w_init: Array.from({ length: m }, (_, i) => Math.exp(thetaHistory[0][1 + i])),
      elapsed: Math.round(elapsed),
      jitMs: mle.compilationMs,
      iterations: totalIters,
      targetFps: TARGET_FPS,
      holdSeconds: HOLD_SECONDS,
      stepSize,
      likHistory,
      frames,
    };

    await commands.writeFile("tmp/mle-frames-nile-webgpu.json", JSON.stringify(output, null, 2));
    console.log(`Saved ${frames.length} frames to tmp/mle-frames-nile-webgpu.json`);
    console.log(`  Animation: ${animDuration.toFixed(2)}s play + ${HOLD_SECONDS}s hold = ${(animDuration + HOLD_SECONDS).toFixed(2)}s total cycle`);
    console.log(`\nSummary: ${totalIters} iters, ${elapsed.toFixed(0)} ms, lik=${mle.deviance.toFixed(2)}`);
  });
});
