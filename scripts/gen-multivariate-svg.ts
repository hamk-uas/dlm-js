/**
 * Generate an SVG plot for the multivariate (p=2) demo — two vertically
 * stacked panels sharing an x-axis, one per sensor.
 *
 * Each panel shows:
 *   - Outer light band: yhat_k ± 2·ystd_k (observation prediction interval)
 *   - Inner opaque band: F_k·x ± 2·σ_state (state uncertainty projected to obs)
 *   - Sensor observations as dots
 *   - Smoothed level line (F_k · x_smooth)
 *   - Octave reference (dashed red)
 *
 * Usage:  npx tsx scripts/gen-multivariate-svg.ts [scan|assoc]
 * Output: assets/multivariate-demo-{scan|assoc}.svg
 */

import { dlmFit, toMatlab } from "../src/index.ts";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { performance } from "node:perf_hooks";
import {
  r, makeLinearScale, polylinePoints, bandPathD,
  renderGridLines, renderYAxis, renderXAxis, renderAxesBorder, writeSvg,
  yTicksFromRange,
} from "./lib/svg-helpers.ts";
import { withLeakCheck } from "./lib/leak-utils.ts";
import { writeTimingsSidecar } from "./lib/timing-sidecar.ts";

// ── Load data ──────────────────────────────────────────────────────────────

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const input  = JSON.parse(readFileSync(resolve(root, "tests/multivariate-in.json"), "utf8"));
const octave = JSON.parse(readFileSync(resolve(root, "tests/multivariate-out-m.json"), "utf8"));

const y: number[][] = input.y;            // [n, p]
const F: number[][] = input.F;            // [p, m]  → [[1,0],[1,0]]
const s: number[]   = input.s;            // [p]     → [4, 7]
const w: number[]   = input.w;            // [m]
const n: number     = input.n;
const p: number     = input.p;
const m: number     = input.m;

const variant = process.argv[2] === 'assoc' ? 'assoc' : 'scan';
const isAssoc = variant === 'assoc';
const scanLabel = isAssoc ? 'assoc/WASM/f64' : 'scan/WASM/f64';

const order = m === 1 ? 0 : m - 1;
const t: number[] = Array.from({ length: n }, (_, i) => i + 1);  // 1-based timestep

// ── Run dlm-js ─────────────────────────────────────────────────────────────

const t0 = performance.now();
await withLeakCheck(() => dlmFit(y, { obsStd: s, processStd: w, F, order, dtype: 'f64', algorithm: isAssoc ? 'assoc' : undefined }));
const t1 = performance.now();

const warmStart = performance.now();
const jsResult = await withLeakCheck(() => dlmFit(y, { obsStd: s, processStd: w, F, order, dtype: 'f64', algorithm: isAssoc ? 'assoc' : undefined }));
const warmEnd = performance.now();

const firstRunMs = t1 - t0;
const warmRunMs  = warmEnd - warmStart;

const matlab = toMatlab(jsResult);

// ── Extract per-sensor data ────────────────────────────────────────────────

// Smoothed states: matlab.x is [m][n], matlab.xstd is [m][n]
const jsLevel: number[]    = Array.from(matlab.x[0]);       // level state
const jsLevelStd: number[] = Array.from(matlab.xstd[0]);    // std of level

// yhat and ystd are flat [n*p] — reshape to per-sensor
const yhatFlat = Array.from(matlab.yhat);
const ystdFlat = Array.from(matlab.ystd);

// Per-sensor arrays
const sensorObs: number[][]  = [];  // sensorObs[k][t]
const sensorYhat: number[][] = [];
const sensorYstd: number[][] = [];
for (let k = 0; k < p; k++) {
  sensorObs.push(y.map(row => row[k]));
  sensorYhat.push([]);
  sensorYstd.push([]);
}
for (let i = 0; i < n; i++) {
  for (let k = 0; k < p; k++) {
    sensorYhat[k].push(yhatFlat[i * p + k]);
    sensorYstd[k].push(ystdFlat[i * p + k]);
  }
}

// Octave reference: x[m][n]
const octLevel: number[] = octave.x[0];
const octYhat: number[][] = [];  // [p][n]
const octYstd: number[][] = [];
for (let k = 0; k < p; k++) {
  octYhat.push(octave.yhat.map((row: number[]) => row[k]));
  octYstd.push(octave.ystd.map((row: number[]) => row[k]));
}

// ── Layout: two panels stacked ─────────────────────────────────────────────

const margin = { top: 28, right: 20, bottom: 44, left: 65 };
const panelGap = 30;  // vertical gap between panels
const W = 800;
const panelH = 200;
const H = margin.top + panelH * 2 + panelGap + margin.bottom;
const plotW = W - margin.left - margin.right;

// Panel y-offsets
const panel1Top = margin.top;
const panel2Top = margin.top + panelH + panelGap;

// ── Shared x-scale ─────────────────────────────────────────────────────────

const sx = makeLinearScale(t[0], t[n - 1], margin.left, margin.left + plotW);

// Per-panel y-scales (each panel auto-ranged to its sensor data)
function panelYRange(k: number): [number, number] {
  const obs = sensorObs[k];
  const upper = sensorYhat[k].map((v, i) => v + 2 * sensorYstd[k][i]);
  const lower = sensorYhat[k].map((v, i) => v - 2 * sensorYstd[k][i]);
  const allVals = [...obs, ...upper, ...lower];
  const pad = 2;
  return [Math.floor(Math.min(...allVals) - pad), Math.ceil(Math.max(...allVals) + pad)];
}

const [y1Min, y1Max] = panelYRange(0);
const [y2Min, y2Max] = panelYRange(1);
const sy1 = makeLinearScale(y1Min, y1Max, panel1Top + panelH, panel1Top);
const sy2 = makeLinearScale(y2Min, y2Max, panel2Top + panelH, panel2Top);

// ── Ticks ──────────────────────────────────────────────────────────────────

const y1Ticks = yTicksFromRange(y1Min, y1Max);
const y2Ticks = yTicksFromRange(y2Min, y2Max);
const tTicks: { val: number; label: string }[] = [];
for (let v = 10; v <= n; v += 10) tTicks.push({ val: v, label: String(v) });

// ── Colors ─────────────────────────────────────────────────────────────────

const sensorColors = ["#2563eb", "#7c3aed"];       // blue, violet
const sensorObsBandColors = ["rgba(37,99,235,0.07)", "rgba(124,58,237,0.07)"];
const sensorStateBandColors = ["rgba(37,99,235,0.22)", "rgba(124,58,237,0.22)"];
const obsColor = "#555";
const octColor = "#ef4444";
const octBandColor = "rgba(239,68,68,0.10)";

// ── Build SVG ─────────────────────────────────────────────────────────────

const lines: string[] = [];
const push = (s: string) => lines.push(s);

push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="system-ui,-apple-system,sans-serif" font-size="12">`);
push(`<rect width="${W}" height="${H}" fill="white"/>`);

// ── Render one panel ────────────────────────────────────────────────────────

function renderPanel(
  k: number,
  panelTop: number,
  sy: (v: number) => number,
  yTicks: number[],
  sensorLabel: string,
) {
  push(`<defs><clipPath id="clip-panel${k}"><rect x="${margin.left}" y="${panelTop}" width="${plotW}" height="${panelH}"/></clipPath></defs>`);

  // Grid
  lines.push(...renderGridLines(yTicks, sy, margin.left, W - margin.right));

  push(`<g clip-path="url(#clip-panel${k})">`);

  // JS observation prediction band (outer, light)
  const jsObsUpper = sensorYhat[k].map((v, i) => v + 2 * sensorYstd[k][i]);
  const jsObsLower = sensorYhat[k].map((v, i) => v - 2 * sensorYstd[k][i]);
  push(`<path d="${bandPathD(t, jsObsUpper, jsObsLower, sx, sy)}" fill="${sensorObsBandColors[k]}" stroke="none"/>`);

  // Octave observation band (red, behind JS state band)
  const octObsUpper = octYhat[k].map((v, i) => v + 2 * octYstd[k][i]);
  const octObsLower = octYhat[k].map((v, i) => v - 2 * octYstd[k][i]);
  push(`<path d="${bandPathD(t, octObsUpper, octObsLower, sx, sy)}" fill="${octBandColor}" stroke="none"/>`);

  // JS state uncertainty band (inner, more opaque) — project level state to obs
  // Since F[k] = [1, 0], the level state maps directly to each sensor
  const jsStateUpper = jsLevel.map((v, i) => v + 2 * jsLevelStd[i]);
  const jsStateLower = jsLevel.map((v, i) => v - 2 * jsLevelStd[i]);
  push(`<path d="${bandPathD(t, jsStateUpper, jsStateLower, sx, sy)}" fill="${sensorStateBandColors[k]}" stroke="none"/>`);

  // JS smoothed prediction line (yhat)
  push(`<polyline points="${polylinePoints(t, sensorYhat[k], sx, sy)}" fill="none" stroke="${sensorColors[k]}" stroke-width="2"/>`);

  // Octave smoothed prediction (dashed red)
  push(`<polyline points="${polylinePoints(t, octYhat[k], sx, sy)}" fill="none" stroke="${octColor}" stroke-width="2" stroke-dasharray="6,3"/>`);

  push(`</g>`);

  // Observed data points
  for (let i = 0; i < n; i++) {
    push(`<circle cx="${r(sx(t[i]))}" cy="${r(sy(sensorObs[k][i]))}" r="2" fill="${obsColor}" opacity="0.5"/>`);
  }

  // Axes
  lines.push(...renderAxesBorder(margin.left, panelTop, W - margin.right, panelTop + panelH));
  lines.push(...renderYAxis(yTicks, sy, margin.left));

  // Panel label
  push(`<text x="${margin.left + 6}" y="${panelTop + 16}" fill="${sensorColors[k]}" font-size="12" font-weight="600">${sensorLabel} (σ=${s[k]})</text>`);
}

renderPanel(0, panel1Top, sy1, y1Ticks, "Sensor 1");
renderPanel(1, panel2Top, sy2, y2Ticks, "Sensor 2");

// Shared x-axis (below panel 2)
const xAxisY = panel2Top + panelH;
lines.push(...renderXAxis(tTicks, sx, xAxisY));
push(`<text x="${r(margin.left + plotW / 2)}" y="${H - 5}" text-anchor="middle" fill="#333" font-size="13">Timestep</text>`);

// Y-axis label (centred between both panels)
const yAxisCentre = margin.top + panelH + panelGap / 2;
push(`<text x="14" y="${r(yAxisCentre)}" text-anchor="middle" fill="#333" font-size="13" transform="rotate(-90,14,${r(yAxisCentre)})">Observed value</text>`);

// Title
push(`<text x="${W / 2}" y="18" text-anchor="middle" fill="#333" font-size="14" font-weight="600">Multivariate demo (p=2, order=${order}, m=${m}) — cold ${firstRunMs.toFixed(0)} ms, warm ${warmRunMs.toFixed(0)} ms, ${scanLabel}</text>`);

// Legend — top-right
const legW = 220;
const legH = 68;
const legX = W - margin.right - legW - 4;
const legY = panel1Top + 4;
push(`<rect x="${legX}" y="${legY}" width="${legW}" height="${legH}" rx="4" fill="rgba(255,255,255,0.92)" stroke="#e5e7eb" stroke-width="1"/>`);
// Observed
push(`<circle cx="${legX + 14}" cy="${legY + 14}" r="2.5" fill="${obsColor}" opacity="0.6"/>`);
push(`<text x="${legX + 24}" y="${legY + 14}" dominant-baseline="middle" fill="#333" font-size="11">Observations</text>`);
// dlm-js
push(`<rect x="${legX + 8}" y="${legY + 27}" width="14" height="14" fill="${sensorObsBandColors[0]}" stroke="none"/>`);
push(`<rect x="${legX + 8}" y="${legY + 31}" width="14" height="6" fill="${sensorStateBandColors[0]}" stroke="none"/>`);
push(`<line x1="${legX + 8}" y1="${legY + 34}" x2="${legX + 22}" y2="${legY + 34}" stroke="${sensorColors[0]}" stroke-width="2"/>`);
push(`<text x="${legX + 26}" y="${legY + 34}" dominant-baseline="middle" fill="#333" font-size="11">dlm-js ŷ ±2σ state / ±2σ obs</text>`);
// Octave
push(`<rect x="${legX + 8}" y="${legY + 48}" width="14" height="14" fill="${octBandColor}" stroke="none"/>`);
push(`<line x1="${legX + 8}" y1="${legY + 55}" x2="${legX + 22}" y2="${legY + 55}" stroke="${octColor}" stroke-width="2" stroke-dasharray="5,2"/>`);
push(`<text x="${legX + 26}" y="${legY + 55}" dominant-baseline="middle" fill="#333" font-size="11">MATLAB/Octave (reference)</text>`);

push("</svg>");

// ── Write ─────────────────────────────────────────────────────────────────

const outPath = resolve(root, `assets/multivariate-demo-${variant}.svg`);
writeSvg(lines, outPath);
writeTimingsSidecar(isAssoc ? "gen-multivariate-svg-assoc" : "gen-multivariate-svg", { firstRunMs, warmRunMs });
console.log(`p=${p}  m=${m}  nobs=${n}  firstRun=${firstRunMs.toFixed(2)} ms  warmRun=${warmRunMs.toFixed(2)} ms`);
