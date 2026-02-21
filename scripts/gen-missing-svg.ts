/**
 * Generate an SVG plot for the missing-data demo — single panel, dual-fill.
 *
 * Dual-fill layout (matching nile-mle-anim style):
 *   - Outer light band:  yhat ± 2·ystd  (observation prediction interval)
 *   - Inner opaque band: x[0] ± 2·xstd (state uncertainty)
 *   - Gray shading marks contiguous and isolated missing-data regions.
 *
 * Usage:  npx tsx scripts/gen-missing-svg.ts
 * Output: assets/missing-demo.svg
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
const input  = JSON.parse(readFileSync(resolve(root, "tests/missing-in.json"), "utf8"));
const octave = JSON.parse(readFileSync(resolve(root, "tests/missing-out-m.json"), "utf8"));

// y has nulls (JSON encoding of NaN); y_full is the complete Nile series
const yRaw: (number | null)[] = input.y;
const y: number[] = yRaw.map(v => (v === null ? NaN : v));  // null → NaN for dlmFit
const yFull: number[] = input.y_full;
const nanMask: number[] = input.nan_mask;              // 1 = missing
const s: number  = input.s;
const w: number[] = input.w;
const opts = input.options;

const variant = process.argv[2] === 'assoc' ? 'assoc' : 'scan';
const isAssoc = variant === 'assoc';
const scanLabel = isAssoc ? 'associativeScan/WASM/f64' : 'scan/WASM/f64';

const n = yRaw.length;
const t: number[] = Array.from({ length: n }, (_, i) => 1871 + i);  // 1871–1970

// ── Run dlm-js ─────────────────────────────────────────────────────────────

const t0 = performance.now();
await withLeakCheck(() => dlmFit(y, { obsStd: s, processStd: w, dtype: 'f64', ...opts, algorithm: isAssoc ? 'assoc' : undefined }));
const t1 = performance.now();

const warmStart = performance.now();
const jsResult = await withLeakCheck(() => dlmFit(y, { obsStd: s, processStd: w, dtype: 'f64', ...opts, algorithm: isAssoc ? 'assoc' : undefined }));
const warmEnd = performance.now();

const firstRunMs = t1 - t0;
const warmRunMs  = warmEnd - warmStart;

const jsLevel    = Array.from(jsResult.smoothed.series(0)) as number[];
const jsLevelStd = Array.from({ length: n }, (_, t) => jsResult.smoothedStd.get(t, 0));
const jsYhat     = Array.from(jsResult.yhat) as number[];
const jsYstd     = Array.from(jsResult.ystd) as number[];

// ── Octave reference ───────────────────────────────────────────────────────

const octLevel:    number[] = octave.x[0];
const octLevelStd: number[] = (octave.xstd as number[][]).map((row: number[]) => row[0]);

// ── Layout ─────────────────────────────────────────────────────────────────

const margin = { top: 28, right: 20, bottom: 50, left: 65 };
const W = 800;
const H = 380;
const plotW = W - margin.left - margin.right;
const plotH = H - margin.top - margin.bottom;

// ── Scales ─────────────────────────────────────────────────────────────────

// Y-range driven by the outer observation prediction band (widest)
// We use jsLevel (=F·x_smooth) as band centre alongside ystd — both are
// smoothed quantities and properly reflect the backward pass over gaps.
const allVals: number[] = [...yFull];
for (let i = 0; i < n; i++) {
  allVals.push(jsLevel[i] + 2 * jsYstd[i]);
  allVals.push(jsLevel[i] - 2 * jsYstd[i]);
}
const yMin = Math.floor(Math.min(...allVals) / 50) * 50;
const yMax = Math.ceil(Math.max(...allVals) / 50) * 50;

const sx = makeLinearScale(t[0], t[n - 1], margin.left, margin.left + plotW);
const sy = makeLinearScale(yMin, yMax, margin.top + plotH, margin.top);

// ── Bands ──────────────────────────────────────────────────────────────────

// Outer: obs prediction ± 2σ — centred on F·x_smooth (= jsLevel for F=[1,0])
// Both centre and width are posterior-smoothed quantities, so the band
// correctly widens over gaps and re-narrows as observations resume.
// (jsYhat = F·x_{t|t-1} is the one-step-ahead forward prediction and does
// *not* incorporate the backward pass; use jsLevel here instead.)
const jsObsUpper = jsLevel.map((v, i) => v + 2 * jsYstd[i]);
const jsObsLower = jsLevel.map((v, i) => v - 2 * jsYstd[i]);

// Inner: state uncertainty ± 2σ (x[0] ± 2·xstd[0])
const jsStateUpper = jsLevel.map((v, i) => v + 2 * jsLevelStd[i]);
const jsStateLower = jsLevel.map((v, i) => v - 2 * jsLevelStd[i]);

// Octave state band
const octStateUpper = octLevel.map((v, i) => v + 2 * octLevelStd[i]);
const octStateLower = octLevel.map((v, i) => v - 2 * octLevelStd[i]);

// ── Ticks ──────────────────────────────────────────────────────────────────

const yTicks: number[] = [];
for (let v = yMin; v <= yMax; v += 200) yTicks.push(v);
const tTicks: number[] = [];
for (let v = Math.ceil(t[0] / 20) * 20; v <= t[n - 1]; v += 20) tTicks.push(v);

// ── Missing-data shading ────────────────────────────────────────────────────

type GapBand = { t0: number; t1: number };
const gapBands: GapBand[] = [];
let inBand = false;
let bStart = 0;
for (let i = 0; i < n; i++) {
  if (nanMask[i] === 1 && !inBand) { bStart = i; inBand = true; }
  else if (nanMask[i] === 0 && inBand) { gapBands.push({ t0: t[bStart], t1: t[i - 1] }); inBand = false; }
}
if (inBand) gapBands.push({ t0: t[bStart], t1: t[n - 1] });

const isolatedIdx: number[] = [];
for (let i = 0; i < n; i++) {
  if (nanMask[i] === 1 && !gapBands.some(b => t[i] >= b.t0 && t[i] <= b.t1)) {
    isolatedIdx.push(i);
  }
}

// ── Colors (matching nile-mle-anim style) ─────────────────────────────────

const obsColor         = "#555";
const jsColor          = "#2563eb";
const octColor         = "#ef4444";
const jsObsBandColor   = "rgba(37,99,235,0.07)";   // outer, same as nile-mle-anim obsBandColor
const jsStateBandColor = "rgba(37,99,235,0.22)";   // inner, same as nile-mle-anim stateBandColor
const octStateBandColor = "rgba(239,68,68,0.12)";
const gapColor         = "rgba(180,180,180,0.18)";

// ── Build SVG ─────────────────────────────────────────────────────────────

const lines: string[] = [];
const push = (s: string) => lines.push(s);

push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="system-ui,-apple-system,sans-serif" font-size="12">`);
push(`<rect width="${W}" height="${H}" fill="white"/>`);
push(`<defs><clipPath id="plot-clip"><rect x="${margin.left}" y="${margin.top}" width="${plotW}" height="${plotH}"/></clipPath></defs>`);

// Grid
lines.push(...renderGridLines(yTicks, sy, margin.left, W - margin.right));

push(`<g clip-path="url(#plot-clip)">`);

// Gray shading for missing regions
const stepHalfW = (sx(t[1]) - sx(t[0])) * 0.5;
for (const b of gapBands) {
  const x1 = sx(b.t0 - 0.5); const x2 = sx(b.t1 + 0.5);
  push(`<rect x="${r(x1)}" y="${r(margin.top)}" width="${r(x2 - x1)}" height="${r(plotH)}" fill="${gapColor}"/>`);
}
for (const i of isolatedIdx) {
  const cx = sx(t[i]);
  push(`<rect x="${r(cx - stepHalfW)}" y="${r(margin.top)}" width="${r(stepHalfW * 2)}" height="${r(plotH)}" fill="${gapColor}"/>`);
}

// Outer observation prediction band (light, behind everything)
push(`<path d="${bandPathD(t, jsObsUpper, jsObsLower, sx, sy)}" fill="${jsObsBandColor}" stroke="none"/>`);

// Octave state band (red, behind dlm-js)
push(`<path d="${bandPathD(t, octStateUpper, octStateLower, sx, sy)}" fill="${octStateBandColor}" stroke="none"/>`);

// Inner state uncertainty band (more opaque)
push(`<path d="${bandPathD(t, jsStateUpper, jsStateLower, sx, sy)}" fill="${jsStateBandColor}" stroke="none"/>`);

// dlm-js smoothed level (solid blue)
push(`<polyline points="${polylinePoints(t, jsLevel, sx, sy)}" fill="none" stroke="${jsColor}" stroke-width="2"/>`);

// Octave smoothed level (dashed red — drawn on top)
push(`<polyline points="${polylinePoints(t, octLevel, sx, sy)}" fill="none" stroke="${octColor}" stroke-width="2" stroke-dasharray="6,3"/>`);

push(`</g>`);

// Observed data points (skip missing)
for (let i = 0; i < n; i++) {
  if (nanMask[i] === 0) {
    push(`<circle cx="${r(sx(t[i]))}" cy="${r(sy(yFull[i]))}" r="2.5" fill="${obsColor}" opacity="0.6"/>`);
  }
}

// Axes
lines.push(...renderAxesBorder(margin.left, margin.top, W - margin.right, margin.top + plotH));
lines.push(...renderYAxis(yTicks, sy, margin.left));
lines.push(...renderXAxis(tTicks.map(v => ({ val: v, label: String(v) })), sx, margin.top + plotH));

// Axis labels
push(`<text x="${r(margin.left + plotW / 2)}" y="${H - 5}" text-anchor="middle" fill="#333" font-size="13">Year</text>`);
push(`<text x="14" y="${r(margin.top + plotH / 2)}" text-anchor="middle" fill="#333" font-size="13" transform="rotate(-90,14,${r(margin.top + plotH / 2)})">Annual flow</text>`);

// Title
push(`<text x="${r(margin.left + plotW / 2)}" y="18" text-anchor="middle" fill="#333" font-size="14" font-weight="600">Nile demo (missing data, ${jsResult.nobs}/${y.length} observed) — fit (order=1, trend), cold ${firstRunMs.toFixed(0)} ms, warm ${warmRunMs.toFixed(0)} ms, ${scanLabel}</text>`);

// Legend — top centre
const legW = 255;
const legH = 82;
const legX = Math.round(margin.left + (plotW - legW) / 2 + legW * 0.5);
const legY = margin.top + 4;
push(`<rect x="${legX}" y="${legY}" width="${legW}" height="${legH}" rx="4" fill="rgba(255,255,255,0.92)" stroke="#e5e7eb" stroke-width="1"/>`);
// Observed
push(`<circle cx="${legX + 14}" cy="${legY + 14}" r="3" fill="${obsColor}" opacity="0.6"/>`);
push(`<text x="${legX + 25}" y="${legY + 14}" dominant-baseline="middle" fill="#333" font-size="11">Observed</text>`);
// Missing
push(`<rect x="${legX + 83}" y="${legY + 7}" width="14" height="14" fill="${gapColor}" stroke="#ccc" stroke-width="0.5"/>`);
push(`<text x="${legX + 103}" y="${legY + 14}" dominant-baseline="middle" fill="#333" font-size="11">Missing (NaN)</text>`);
// dlm-js dual-band
push(`<rect x="${legX + 8}" y="${legY + 30}" width="14" height="14" fill="${jsObsBandColor}" stroke="none"/>`);
push(`<rect x="${legX + 8}" y="${legY + 34}" width="14" height="6" fill="${jsStateBandColor}" stroke="none"/>`);
push(`<line x1="${legX + 8}" y1="${legY + 37}" x2="${legX + 22}" y2="${legY + 37}" stroke="${jsColor}" stroke-width="2"/>`);
push(`<text x="${legX + 26}" y="${legY + 37}" dominant-baseline="middle" fill="#333" font-size="11">dlm-js F·x_smooth ±2σ state / ±2ystd obs</text>`);
// Octave
push(`<rect x="${legX + 8}" y="${legY + 54}" width="14" height="14" fill="${octStateBandColor}" stroke="none"/>`);
push(`<line x1="${legX + 8}" y1="${legY + 61}" x2="${legX + 22}" y2="${legY + 61}" stroke="${octColor}" stroke-width="2" stroke-dasharray="5,2"/>`);
push(`<text x="${legX + 26}" y="${legY + 61}" dominant-baseline="middle" fill="#333" font-size="11">MATLAB/Octave x[0] ±2σ (reference)</text>`);

push("</svg>");

// ── Write ─────────────────────────────────────────────────────────────────

const outPath = resolve(root, `assets/missing-demo-${variant}.svg`);
writeSvg(lines, outPath);
writeTimingsSidecar(isAssoc ? "gen-missing-svg-assoc" : "gen-missing-svg", { firstRunMs, warmRunMs });
console.log(`nobs=${jsResult.nobs}  firstRun=${firstRunMs.toFixed(2)} ms  warmRun=${warmRunMs.toFixed(2)} ms`);

