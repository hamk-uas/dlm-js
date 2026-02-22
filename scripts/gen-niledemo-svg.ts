/**
 * Generate an SVG plot of the Nile demo: observations, smoothed level, and
 * ±2σ confidence bands from both dlm-js and MATLAB/Octave dlm.
 *
 * Usage:  npx tsx scripts/gen-niledemo-svg.ts
 * Output: assets/niledemo.svg
 */

import { dlmFit } from "../src/index.ts";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { performance } from "node:perf_hooks";
import {
  r, makeLinearScale, polylinePoints, bandPathD,
  renderGridLines, renderYAxis, renderXAxis, renderAxesBorder, writeSvg,
} from "./lib/svg-helpers.ts";
import { withLeakCheck } from "./lib/leak-utils.ts";
import { writeTimingsSidecar } from "./lib/timing-sidecar.ts";

// ── Load data ──────────────────────────────────────────────────────────────

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const input = JSON.parse(readFileSync(resolve(root, "tests/niledemo-in.json"), "utf8"));
const octave = JSON.parse(readFileSync(resolve(root, "tests/niledemo-out-m.json"), "utf8"));

const t: number[] = input.t;        // years 1871–1970
const y: number[] = input.y;        // observations
const s: number = input.s;          // observation noise std
const w: number[] = input.w;        // state noise stds

const variant = process.argv[2] === 'assoc' ? 'assoc'
  : process.argv[2] === 'sqrt-assoc' ? 'sqrt-assoc'
  : process.argv[2] === 'sqrt-assoc-f32' ? 'sqrt-assoc-f32'
  : 'scan';
const isF32 = variant === 'sqrt-assoc-f32';
const dtype: 'f64' | 'f32' = isF32 ? 'f32' : 'f64';
const algorithm: 'scan' | 'assoc' | 'sqrt-assoc' | undefined =
  variant === 'assoc' ? 'assoc'
  : (variant === 'sqrt-assoc' || variant === 'sqrt-assoc-f32') ? 'sqrt-assoc'
  : undefined;
const scanLabel =
  variant === 'assoc' ? 'associativeScan/WASM/f64' :
  variant === 'sqrt-assoc' ? 'sqrt-assoc/WASM/f64' :
  variant === 'sqrt-assoc-f32' ? 'sqrt-assoc/WASM/f32' :
  'scan/WASM/f64';

// ── Run dlm-js ─────────────────────────────────────────────────────────────

const timedFit = async () => {
  const t0 = performance.now();
  await withLeakCheck(() => dlmFit(y, { obsStd: s, processStd: w, dtype, order: 1, algorithm }));
  const t1 = performance.now();

  const warmStart = performance.now();
  const result = await withLeakCheck(() => dlmFit(y, { obsStd: s, processStd: w, dtype, order: 1, algorithm }));
  const warmEnd = performance.now();

  return {
    result,
    firstRunMs: t1 - t0,
    warmRunMs: warmEnd - warmStart,
  };
};

const timed = await timedFit();
const jsResult = timed.result;
const jsLevel = Array.from(jsResult.smoothed.series(0));              // smoothed state level
const n_ = jsResult.n;
const jsLevelStd = Array.from({ length: n_ }, (_, t) => jsResult.smoothedStd.get(t, 0)); // xstd[:,0]

// ── Octave results ─────────────────────────────────────────────────────────

const octLevel: number[] = octave.x[0];                  // smoothed state level
const octLevelStd: number[] = octave.xstd.map((row: number[]) => row[0]); // xstd[:,0]

// ── SVG generation ─────────────────────────────────────────────────────────

const n = t.length;

// Layout
const margin = { top: 30, right: 20, bottom: 50, left: 65 };
const W = 800;
const H = 360;
const plotW = W - margin.left - margin.right;
const plotH = H - margin.top - margin.bottom;

// Scales
const tMin = t[0];
const tMax = t[n - 1];
const allVals = [
  ...y,
  ...jsLevel.map((v, i) => v + 2 * jsLevelStd[i]),
  ...jsLevel.map((v, i) => v - 2 * jsLevelStd[i]),
];
const yMin = Math.floor(Math.min(...allVals) / 50) * 50;
const yMax = Math.ceil(Math.max(...allVals) / 50) * 50;

const sx = makeLinearScale(tMin, tMax, margin.left, margin.left + plotW);
const sy = makeLinearScale(yMin, yMax, margin.top + plotH, margin.top); // inverted: high values at top

// Confidence bands (smoothed level ± 2σ)
const jsUpper = jsLevel.map((v, i) => v + 2 * jsLevelStd[i]);
const jsLower = jsLevel.map((v, i) => v - 2 * jsLevelStd[i]);
const octUpper = octLevel.map((v, i) => v + 2 * octLevelStd[i]);
const octLower = octLevel.map((v, i) => v - 2 * octLevelStd[i]);

// Tick marks
const yTicks: number[] = [];
for (let v = yMin; v <= yMax; v += 200) yTicks.push(v);
const tTicks: number[] = [];
for (let v = Math.ceil(tMin / 20) * 20; v <= tMax; v += 20) tTicks.push(v);

// Colors
const obsColor = "#555";
const jsColor = "#2563eb";     // blue
const octColor = "#ef4444";    // red
const jsBandColor = "rgba(37,99,235,0.12)";
const octBandColor = "rgba(239,68,68,0.12)";

// Build SVG
const lines: string[] = [];
const push = (s: string) => lines.push(s);

push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="system-ui,-apple-system,sans-serif" font-size="12">`);

// Background
push(`<rect width="${W}" height="${H}" fill="white"/>`);

// Grid lines
lines.push(...renderGridLines(yTicks, sy, margin.left, W - margin.right));

// 1. dlm-js confidence band (blue, low alpha)
push(`<path d="${bandPathD(t, jsUpper, jsLower, sx, sy)}" fill="${jsBandColor}" stroke="none"/>`);
// 2. MATLAB/Octave confidence band (red, low alpha)
push(`<path d="${bandPathD(t, octUpper, octLower, sx, sy)}" fill="${octBandColor}" stroke="none"/>`);
// 3. dlm-js smoothed level (blue, full opacity)
push(`<polyline points="${polylinePoints(t, jsLevel, sx, sy)}" fill="none" stroke="${jsColor}" stroke-width="2"/>`);
// 4. MATLAB/Octave smoothed level (red dashed, full opacity)
push(`<polyline points="${polylinePoints(t, octLevel, sx, sy)}" fill="none" stroke="${octColor}" stroke-width="2" stroke-dasharray="6,3"/>`);

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
push(`<text x="${W / 2}" y="${16}" text-anchor="middle" fill="#333" font-size="12" font-weight="600">Nile demo — fit (order=1, trend), cold ${timed.firstRunMs.toFixed(0)} ms, warm ${timed.warmRunMs.toFixed(0)} ms, ${scanLabel}</text>`);

// Legend
const legX = W - margin.right - 255;
const legY = margin.top + 8;
push(`<rect x="${legX}" y="${legY}" width="250" height="62" rx="4" fill="white" stroke="#e5e7eb" stroke-width="1"/>`);
// Observations
push(`<circle cx="${legX + 14}" cy="${legY + 14}" r="3" fill="${obsColor}" opacity="0.6"/>`);
push(`<text x="${legX + 24}" y="${legY + 14}" dominant-baseline="middle" fill="#333" font-size="11">Observations</text>`);
// dlm-js
push(`<line x1="${legX + 8}" y1="${legY + 30}" x2="${legX + 20}" y2="${legY + 30}" stroke="${jsColor}" stroke-width="2"/>`);
push(`<rect x="${legX + 8}" y="${legY + 25}" width="12" height="10" fill="${jsBandColor}" stroke="none"/>`);
push(`<text x="${legX + 24}" y="${legY + 30}" dominant-baseline="middle" fill="#333" font-size="11">dlm-js smoothed level ± 2σ</text>`);
// Octave
push(`<line x1="${legX + 8}" y1="${legY + 46}" x2="${legX + 20}" y2="${legY + 46}" stroke="${octColor}" stroke-width="2" stroke-dasharray="6,3"/>`);
push(`<rect x="${legX + 8}" y="${legY + 41}" width="12" height="10" fill="${octBandColor}" stroke="none"/>`);
push(`<text x="${legX + 24}" y="${legY + 46}" dominant-baseline="middle" fill="#333" font-size="11">MATLAB/Octave dlm smoothed level ± 2σ</text>`);

push(`</svg>`);

// ── Write output ───────────────────────────────────────────────────────────

const outPath = resolve(root, "assets", `niledemo-${variant}.svg`);
writeSvg(lines, outPath);
const sidecarKey = variant === 'assoc' ? 'gen-niledemo-svg-assoc'
  : variant === 'sqrt-assoc' ? 'gen-niledemo-svg-sqrt-assoc'
  : variant === 'sqrt-assoc-f32' ? 'gen-niledemo-svg-sqrt-assoc-f32'
  : 'gen-niledemo-svg';
writeTimingsSidecar(sidecarKey, { firstRunMs: timed.firstRunMs, warmRunMs: timed.warmRunMs });
console.log(
  `Timing (dlmFit with jitted core): first-run ${timed.firstRunMs.toFixed(2)} ms, warm-run ${timed.warmRunMs.toFixed(2)} ms`
);
