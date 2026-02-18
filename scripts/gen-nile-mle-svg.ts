/**
 * Generate a before/after MLE optimization SVG for the Nile demo.
 *
 * Shows:
 *   - Observations (grey dots)
 *   - "Before" smoothed level ± 2σ from initial guess (orange)
 *   - "After"  smoothed level ± 2σ from MLE-optimized params (blue)
 *
 * Usage:  npx tsx scripts/gen-nile-mle-svg.ts
 * Output: assets/nile-mle.svg
 */

import { defaultDevice, DType } from "@hamk-uas/jax-js-nonconsuming";
import { dlmFit } from "../src/index.ts";
import { dlmMLE } from "../src/mle.ts";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { performance } from "node:perf_hooks";
import {
  r, makeLinearScale, polylinePoints, bandPathD,
  renderGridLines, renderYAxis, renderXAxis, renderAxesBorder, writeSvg,
} from "./lib/svg-helpers.ts";

defaultDevice("wasm");

// ── Load data ──────────────────────────────────────────────────────────────

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const input = JSON.parse(readFileSync(resolve(root, "tests/niledemo-in.json"), "utf8"));

const t: number[] = input.t;
const y: number[] = input.y;
const n = t.length;
const dtype = DType.Float64;
const options = { order: 1 };  // local level model (m=2)

// ── "Before" fit: rough initial guess ──────────────────────────────────────

console.log("Running 'before' fit with initial guess...");
const s_init = Math.sqrt(
  y.reduce((s, v) => s + v * v, 0) / n - (y.reduce((s, v) => s + v, 0) / n) ** 2
);
const w_init = [s_init * 0.1, s_init * 0.1];
console.log(`  init: s=${s_init.toFixed(2)}, w=[${w_init.map(v => v.toFixed(2)).join(", ")}]`);

const beforeFit = await dlmFit(y, s_init, w_init, dtype, options);
const beforeLevel = Array.from(beforeFit.x[0]);
const beforeStd = beforeFit.xstd.map((row: any) => row[0] as number);

// ── MLE optimization ───────────────────────────────────────────────────────

console.log("Running MLE optimization (WASM backend)...");
const t0 = performance.now();
const mle = await dlmMLE(y, options, undefined, 300, 0.05, 1e-6, dtype);
const elapsed = performance.now() - t0;

console.log(`  MLE: s=${mle.s.toFixed(4)}, w=[${mle.w.map(v => v.toFixed(4)).join(", ")}]`);
console.log(`  lik=${mle.lik.toFixed(4)}, iters=${mle.iterations}, time=${elapsed.toFixed(0)}ms`);

const afterLevel = Array.from(mle.fit.x[0]);
const afterStd = mle.fit.xstd.map((row: any) => row[0] as number);

// Also show the known Octave params for reference
console.log(`  known: s=${input.s.toFixed(4)}, w=[${input.w.map((v: number) => v.toFixed(4)).join(", ")}]`);

// ── SVG generation ─────────────────────────────────────────────────────────

const margin = { top: 30, right: 20, bottom: 50, left: 65 };
const W = 800;
const H = 380;
const plotW = W - margin.left - margin.right;
const plotH = H - margin.top - margin.bottom;

// Scales
const tMin = t[0];
const tMax = t[n - 1];
const allVals = [
  ...y,
  ...beforeLevel.map((v, i) => v + 2 * beforeStd[i]),
  ...beforeLevel.map((v, i) => v - 2 * beforeStd[i]),
  ...afterLevel.map((v, i) => v + 2 * afterStd[i]),
  ...afterLevel.map((v, i) => v - 2 * afterStd[i]),
];
const yMin = Math.floor(Math.min(...allVals) / 50) * 50;
const yMax = Math.ceil(Math.max(...allVals) / 50) * 50;

const sx = makeLinearScale(tMin, tMax, margin.left, margin.left + plotW);
const sy = makeLinearScale(yMin, yMax, margin.top + plotH, margin.top);

// Bands
const beforeUpper = beforeLevel.map((v, i) => v + 2 * beforeStd[i]);
const beforeLower = beforeLevel.map((v, i) => v - 2 * beforeStd[i]);
const afterUpper = afterLevel.map((v, i) => v + 2 * afterStd[i]);
const afterLower = afterLevel.map((v, i) => v - 2 * afterStd[i]);

// Ticks
const yTicks: number[] = [];
for (let v = yMin; v <= yMax; v += 200) yTicks.push(v);
const tTicks: number[] = [];
for (let v = Math.ceil(tMin / 20) * 20; v <= tMax; v += 20) tTicks.push(v);

// Colors
const obsColor = "#555";
const beforeColor = "#f59e0b";     // amber/orange
const afterColor = "#2563eb";      // blue
const beforeBand = "rgba(245,158,11,0.12)";
const afterBand = "rgba(37,99,235,0.12)";

// Build SVG
const lines: string[] = [];
const push = (s: string) => lines.push(s);

push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="system-ui,-apple-system,sans-serif" font-size="12">`);
push(`<rect width="${W}" height="${H}" fill="white"/>`);

// Grid
lines.push(...renderGridLines(yTicks, sy, margin.left, W - margin.right));

// Before band + line
push(`<path d="${bandPathD(t, beforeUpper, beforeLower, sx, sy)}" fill="${beforeBand}" stroke="none"/>`);
push(`<polyline points="${polylinePoints(t, beforeLevel, sx, sy)}" fill="none" stroke="${beforeColor}" stroke-width="2" stroke-dasharray="6,3"/>`);

// After band + line
push(`<path d="${bandPathD(t, afterUpper, afterLower, sx, sy)}" fill="${afterBand}" stroke="none"/>`);
push(`<polyline points="${polylinePoints(t, afterLevel, sx, sy)}" fill="none" stroke="${afterColor}" stroke-width="2"/>`);

// Observations
for (let i = 0; i < n; i++) {
  push(`<circle cx="${r(sx(t[i]))}" cy="${r(sy(y[i]))}" r="2.5" fill="${obsColor}" opacity="0.6"/>`);
}

// Axes
lines.push(...renderAxesBorder(margin.left, margin.top, W - margin.right, H - margin.bottom));
lines.push(...renderYAxis(yTicks, sy, margin.left));
lines.push(...renderXAxis(tTicks.map(v => ({ val: v, label: String(v) })), sx, H - margin.bottom));

// Axis labels
push(`<text x="${margin.left + plotW / 2}" y="${H - 5}" text-anchor="middle" fill="#333" font-size="13">Year</text>`);
push(`<text x="${14}" y="${margin.top + plotH / 2}" text-anchor="middle" fill="#333" font-size="13" transform="rotate(-90,14,${margin.top + plotH / 2})">Annual flow</text>`);

// Title
push(`<text x="${margin.left + plotW / 2}" y="${16}" text-anchor="middle" fill="#333" font-size="14" font-weight="600">Nile — MLE optimization: before vs after (${mle.iterations} iters, ${elapsed.toFixed(0)} ms WASM)</text>`);

// Legend
const legX = W - margin.right - 290;
const legY = margin.top + 8;
push(`<rect x="${legX}" y="${legY}" width="285" height="80" rx="4" fill="white" stroke="#e5e7eb" stroke-width="1"/>`);

// Observations
push(`<circle cx="${legX + 14}" cy="${legY + 14}" r="3" fill="${obsColor}" opacity="0.6"/>`);
push(`<text x="${legX + 24}" y="${legY + 14}" dominant-baseline="middle" fill="#333" font-size="11">Observations</text>`);

// Before
push(`<line x1="${legX + 8}" y1="${legY + 32}" x2="${legX + 20}" y2="${legY + 32}" stroke="${beforeColor}" stroke-width="2" stroke-dasharray="6,3"/>`);
push(`<rect x="${legX + 8}" y="${legY + 27}" width="12" height="10" fill="${beforeBand}" stroke="none"/>`);
push(`<text x="${legX + 24}" y="${legY + 32}" dominant-baseline="middle" fill="#333" font-size="11">Before: s=${s_init.toFixed(1)}, w=[${w_init.map(v => v.toFixed(1)).join(",")}]</text>`);

// After
push(`<line x1="${legX + 8}" y1="${legY + 50}" x2="${legX + 20}" y2="${legY + 50}" stroke="${afterColor}" stroke-width="2"/>`);
push(`<rect x="${legX + 8}" y="${legY + 45}" width="12" height="10" fill="${afterBand}" stroke="none"/>`);
push(`<text x="${legX + 24}" y="${legY + 50}" dominant-baseline="middle" fill="#333" font-size="11">MLE: s=${mle.s.toFixed(1)}, w=[${mle.w.map(v => v.toFixed(1)).join(",")}]</text>`);

// Lik
push(`<text x="${legX + 24}" y="${legY + 68}" dominant-baseline="middle" fill="#666" font-size="10">−2·logL: ${mle.lik.toFixed(2)} (converged in ${mle.iterations} iters)</text>`);

push(`</svg>`);

// ── Write output ───────────────────────────────────────────────────────────

const outPath = resolve(root, "assets", "nile-mle.svg");
writeSvg(lines, outPath);
