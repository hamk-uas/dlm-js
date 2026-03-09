/**
 * Repro: reverse lax.associativeScan produces a step-change at index N−32
 * on WebGPU/Float32.
 *
 * The backward RTS smoother composes (E, g, L) tuples via a reverse
 * associativeScan. On WebGPU, the result has a covariance discontinuity
 * at exactly index N−32 (stride-32 boundary in the Kogge-Stone prefix
 * tree). The same code on WASM produces smooth output.
 *
 * Run:
 *   deno run --unstable-webgpu --allow-read --allow-env \
 *     issues/repro-webgpu-reverse-assocscan-boundary.ts
 *
 * Expected: ystd jump at index 68 ≈ 0.0 (matches WASM)
 * Actual:   ystd jump at index 68 ≈ 8.6 (WebGPU)
 */

import { defaultDevice, init } from "../node_modules/@hamk-uas/jax-js-nonconsuming/dist/index.js";
import { dlmFit } from "../src/index.ts";
import { readFileSync } from "node:fs";

const root = new URL("../", import.meta.url).pathname.replace(/\/$/, "");

async function runTest(backend: string) {
  if (backend === "webgpu") {
    await init("webgpu");
    defaultDevice("webgpu");
  } else {
    defaultDevice("wasm");
  }

  const input = JSON.parse(readFileSync(`${root}/tests/niledemo-in.json`, "utf8"));
  const y: number[] = input.y;
  const n = y.length;    // 100

  // MLE-converged parameters (order=1 trend model, m=2 states)
  const s = 121.1255;
  const w = [41.8698, 0.0426];

  const yArr = Float32Array.from(y);
  const fit = await dlmFit(yArr, {
    obsStd: s,
    processStd: w,
    dtype: "f32" as const,
    order: 1,
  });
  const ystd = Array.from(fit.ystd);

  // The jump should be at index N-32 = 68 (year 1939)
  const jumpIdx = n - 32;
  const jump = ystd[jumpIdx] - ystd[jumpIdx - 1];

  console.log(`\n${backend}/f32:`);
  console.log(`  ystd[${jumpIdx - 1}] = ${ystd[jumpIdx - 1].toFixed(4)}`);
  console.log(`  ystd[${jumpIdx}]   = ${ystd[jumpIdx].toFixed(4)}`);
  console.log(`  jump = ${jump.toFixed(4)}`);

  // Show smoothed covariance C[0,0] around the boundary
  console.log(`  C_smooth[0,0] around boundary:`);
  for (let i = jumpIdx - 2; i <= jumpIdx + 2; i++) {
    console.log(`    [${i}] = ${fit.smoothedCov.get(i, 0, 0).toFixed(2)}`);
  }

  const pass = Math.abs(jump) < 0.1;
  console.log(`  ${pass ? "PASS" : "FAIL"}: jump ${pass ? "<" : "≥"} 0.1`);
  return pass;
}

// Run on both backends
const wasmOk = await runTest("wasm");
const gpuOk = await runTest("webgpu");

console.log(`\nSummary: WASM=${wasmOk ? "PASS" : "FAIL"} WebGPU=${gpuOk ? "PASS" : "FAIL"}`);
if (!gpuOk) {
  console.log("BUG CONFIRMED: reverse associativeScan has stride-32 boundary artifact on WebGPU/f32");
}
