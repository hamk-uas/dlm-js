/**
 * WebGPU `dlmFit` benchmark — float32 with associativeScan forward filter.
 *
 * Runs the same models as bench-backends.ts but on the WebGPU backend,
 * which triggers the exact 5-tuple parallel forward filter + assoc path.
 *
 * Runs in Chromium browser mode via @vitest/browser-playwright:
 *   GPU=nvidia bash scripts/gpu-test.sh run scripts/bench-gpu.ts
 *
 * Writes timing data to assets/timings/bench-gpu.json.
 *
 * Output: assets/timings/bench-gpu.json
 */

import { describe, it } from 'vitest';
import { commands } from 'vitest/browser';
import { defaultDevice, init } from "@hamk-uas/jax-js-nonconsuming";
import { dlmFit } from "../src/index.ts";

async function readJSON(path: string): Promise<any> {
  return JSON.parse(await commands.readFile(path));
}

// ── Types ──────────────────────────────────────────────────────────────────

interface Model {
  label: string;
  key: string;
  y: number[];
  s: number | number[];
  w: number | number[];
  options: Record<string, unknown>;
  n: number;
  m: number;
}

const toW = (w: number | number[]): number[] => (Array.isArray(w) ? w : [w]);

// ── Timing helper ──────────────────────────────────────────────────────────

async function timedFit(model: Model): Promise<{ firstMs: number; warmMs: number }> {
  const { y, s, w, options } = model;

  const t0 = performance.now();
  const r1 = await dlmFit(y, { obsStd: s, processStd: w, dtype: 'f32', ...options });
  const t1 = performance.now();
  r1[Symbol.dispose]?.();

  const t2 = performance.now();
  const r2 = await dlmFit(y, { obsStd: s, processStd: w, dtype: 'f32', ...options });
  const t3 = performance.now();
  r2[Symbol.dispose]?.();

  return { firstMs: t1 - t0, warmMs: t3 - t2 };
}

// ── Benchmark ──────────────────────────────────────────────────────────────

describe('bench-gpu', () => {
  it('WebGPU dlmFit benchmark (float32, assoc)', async () => {
    await init("webgpu");
    defaultDevice("webgpu");

    const nileIn       = await readJSON("tests/niledemo-in.json");
    const kaisaniemiIn = await readJSON("tests/kaisaniemi-in.json");
    const trigarIn     = await readJSON("tests/trigar-in.json");
    const order0In     = await readJSON("tests/order0-in.json");

    const models: Model[] = [
      { label: "Nile, order=0", key: "nile_o0", y: order0In.y, s: order0In.s, w: toW(order0In.w), options: { order: 0 }, n: 100, m: 1 },
      { label: "Nile, order=1", key: "nile_o1", y: nileIn.y, s: nileIn.s, w: nileIn.w, options: { order: 1 }, n: 100, m: 2 },
      { label: "Kaisaniemi, trig", key: "kaisaniemi", y: kaisaniemiIn.y, s: kaisaniemiIn.s, w: kaisaniemiIn.w, options: { order: 1, harmonics: 1, seasonLength: 12 }, n: 117, m: 4 },
      { label: "Energy, trig+AR", key: "trigar", y: trigarIn.y, s: trigarIn.s, w: trigarIn.w, options: { order: 1, harmonics: 1, seasonLength: 12, arCoefficients: trigarIn.arphi }, n: 120, m: 5 },
    ];

    console.log("=== dlmFit WebGPU benchmark (float32, assoc) ===\n");

    const colW = [32, 8, 6, 14, 14];
    const hdr = [
      "Model".padEnd(colW[0]), "n".padStart(colW[1]), "m".padStart(colW[2]),
      "first (ms)".padStart(colW[3]), "warm (ms)".padStart(colW[4]),
    ].join("  ");
    console.log(hdr);
    console.log("─".repeat(hdr.length));

    const sidecar: Record<string, number> = {};

    for (const model of models) {
      const { firstMs, warmMs } = await timedFit(model);
      const cells = [
        model.label.padEnd(colW[0]), String(model.n).padStart(colW[1]),
        String(model.m).padStart(colW[2]),
        `${firstMs.toFixed(1)}`.padStart(colW[3]), `${warmMs.toFixed(1)}`.padStart(colW[4]),
      ];
      console.log(cells.join("  "));
      sidecar[`${model.key}__webgpu_f32__first`] = firstMs;
      sidecar[`${model.key}__webgpu_f32__warm`]  = warmMs;
    }

    console.log("\nDone.");
    await commands.writeFile("assets/timings/bench-gpu.json", JSON.stringify(sidecar, null, 2) + "\n");
    console.log('Wrote assets/timings/bench-gpu.json');
  });
});
