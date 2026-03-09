/**
 * Standalone repro: WebGPU lax.scan accuracy regression.
 *
 * Run from dlm-js root:
 *   deno run --unstable-webgpu --allow-read --allow-write --allow-env --allow-run issues/repro-webgpu-laxscan.ts
 *
 * Expected: all models show maxRelErr within threshold.
 * Actual (block-map branch): catastrophic errors or all-NaN.
 */
import { defaultDevice, init } from "../node_modules/@hamk-uas/jax-js-nonconsuming/dist/index.js";
import { dlmFit, toMatlab } from "../src/index.ts";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const testsDir = resolve(root, "tests");

await init("webgpu");

const models = [
  { name: "Nile o=0 (m=1)", input: "order0-in.json",       ref: "order0-out-m.json",       opts: { order: 0 },                                                        threshold: 0.001 },
  { name: "Nile o=1 (m=2)", input: "niledemo-in.json",      ref: "niledemo-out-m.json",      opts: { order: 1 },                                                        threshold: 0.05  },
  { name: "Kaisaniemi (m=4)", input: "kaisaniemi-in.json",  ref: "kaisaniemi-out-m.json",   opts: { order: 1, harmonics: 1 },                                          threshold: 5.0   },
  { name: "Energy (m=5)", input: "energy-in.json",          ref: "energy-out-m.json",       opts: { order: 1, harmonics: 1, seasonLength: 12, arCoefficients: [0.85] }, threshold: 1.0   },
  { name: "Gapped (m=2)", input: "gapped-in.json",          ref: "gapped-out-m.json",       opts: { order: 1 },                                                        threshold: 0.05  },
];

function maxRelErr(a: unknown, b: unknown, absTol: number): { err: number; nanCount: number } {
  let maxErr = 0;
  let nanCount = 0;

  function walk(x: unknown, y: unknown) {
    if (Array.isArray(x) && Array.isArray(y)) {
      for (let i = 0; i < Math.min(x.length, y.length); i++) walk(x[i], y[i]);
    } else if (typeof x === "number" && typeof y === "number") {
      if (!isFinite(x)) { nanCount++; return; }
      if (Math.abs(y) < absTol && Math.abs(x) < absTol) return;
      const e = Math.abs(x - y) / Math.max(Math.abs(y), absTol);
      if (e > maxErr) maxErr = e;
    }
  }

  walk(a, b);
  return { err: maxErr, nanCount };
}

let passed = 0, failed = 0;

for (const algo of ["scan", "ud"] as const) {
  for (const m of models) {
    const input = JSON.parse(readFileSync(resolve(testsDir, m.input), "utf-8"));
    const ref = JSON.parse(readFileSync(resolve(testsDir, m.ref), "utf-8"));
    const w: number[] = Array.isArray(input.w) ? input.w : [input.w];
    const y: number[] = (input.y as (number | null)[]).map((v: number | null) => v === null ? NaN : v);

    defaultDevice("webgpu");
    const result = await dlmFit(y, {
      obsStd: input.s, processStd: w, dtype: "f32",
      algorithm: algo, ...m.opts,
    });
    const matlab = toMatlab(result) as Record<string, unknown>;
    result[Symbol.dispose]?.();

    // Check key fields
    let worstKey = "", worstErr = 0, totalNaN = 0;
    for (const k of ["yhat", "ystd", "x", "xstd"]) {
      if (!(k in matlab) || !(k in ref)) continue;
      const { err, nanCount } = maxRelErr(matlab[k], ref[k], 1e-4);
      totalNaN += nanCount;
      if (err > worstErr) { worstErr = err; worstKey = k; }
    }

    const status = totalNaN > 0 ? "NaN" : worstErr > m.threshold ? "FAIL" : "PASS";
    const detail = totalNaN > 0
      ? `${totalNaN} NaN values`
      : `relErr=${worstErr.toExponential(3)} at '${worstKey}' (threshold: ${m.threshold})`;

    console.log(`${status.padEnd(4)}  webgpu/f32/${algo}  ${m.name}: ${detail}`);
    if (status !== "PASS") failed++; else passed++;
  }
}

console.log(`\n${passed} passed, ${failed} failed out of ${passed + failed}`);
if (failed > 0) {
  console.log("\n>>> WebGPU lax.scan accuracy regression CONFIRMED <<<");
  Deno.exit(1);
}
