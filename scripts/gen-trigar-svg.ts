/**
 * Generate an SVG plot of the synthetic energy demand demo
 * (trend + seasonal + strong AR).
 *
 * Four panels:
 * 1) Smoothed level state x[0] ± 2σ, with true level
 * 2) Seasonal component x[2] ± 2σ, with true seasonal
 * 3) AR(1) state x[4] ± 2σ, with true AR state
 * 4) Combined signal F·x = x[0]+x[2]+x[4] ± 2σ (covariance-aware), with obs
 *
 * The true hidden states (from the seeded generating process) are overlaid
 * in green, showing how well the smoother recovers the ground truth.
 *
 * Usage:  npx tsx scripts/gen-trigar-svg.ts
 * Output: assets/trigar.svg
 */

import { dlmFit } from "../src/index.ts";
import { DType } from "@hamk-uas/jax-js-nonconsuming";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { performance } from "node:perf_hooks";
import {
  r, polylinePoints, bandPathD, yTicksFromRange,
  renderGridLines, renderYAxis, renderXAxis, renderAxesBorder, writeSvg,
} from "./lib/svg-helpers.ts";
import { withLeakCheck } from "./lib/leak-utils.ts";
import { writeTimingsSidecar } from "./lib/timing-sidecar.ts";

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const input = JSON.parse(readFileSync(resolve(root, "tests/energy-in.json"), "utf8"));
const octave = JSON.parse(readFileSync(resolve(root, "tests/energy-out-m.json"), "utf8"));

const y: number[] = input.y;
const s: number = input.s;
const w: number[] = input.w;
const n = y.length;
const options = { order: 1, trig: 1, ns: 12, arphi: [0.85] };

// Time axis: months 1..n (synthetic monthly data)
const t: number[] = Array.from({ length: n }, (_, i) => i + 1);

// True hidden states from the generating process (saved by Octave)
const trueLevel: number[] = octave.x_true[0];
const trueSeasonal: number[] = octave.x_true[2];
const trueAR: number[] = octave.x_true[4];
const trueCombined = trueLevel.map((v, i) => v + trueSeasonal[i] + trueAR[i]);

const timedFit = async () => {
  const t0 = performance.now();
  await withLeakCheck(() => dlmFit(y, s, w, DType.Float64, options));
  const t1 = performance.now();

  const warmStart = performance.now();
  const result = await withLeakCheck(() => dlmFit(y, s, w, DType.Float64, options));
  const warmEnd = performance.now();

  return { result, firstRunMs: t1 - t0, warmRunMs: warmEnd - warmStart };
};

const timed = await timedFit();
const js = timed.result;

// Extract JS results
const jsLevel = Array.from(js.x[0]);
const jsLevelStd = js.xstd.map((row: any) => row[0] as number);
const jsSeasonal = Array.from(js.x[2]);
const jsSeasonalStd = js.xstd.map((row: any) => row[2] as number);
const jsAR = Array.from(js.x[4]);
const jsARStd = js.xstd.map((row: any) => row[4] as number);

// Combined F·x = x[0] + x[2] + x[4] with full covariance
function varianceCombinedJs(i: number): number {
  const c00 = js.C[0][0][i];
  const c22 = js.C[2][2][i];
  const c44 = js.C[4][4][i];
  const c02 = js.C[0][2][i];
  const c04 = js.C[0][4][i];
  const c24 = js.C[2][4][i];
  return Math.max(0, c00 + c22 + c44 + 2 * c02 + 2 * c04 + 2 * c24);
}
const jsCombined = jsLevel.map((v, i) => v + jsSeasonal[i] + (jsAR[i] as number));
const jsCombinedStd = Array.from({ length: n }, (_, i) => Math.sqrt(varianceCombinedJs(i)));

// Extract Octave results
const octLevel: number[] = octave.x[0];
const octLevelStd: number[] = octave.xstd.map((row: number[]) => row[0]);
const octSeasonal: number[] = octave.x[2];
const octSeasonalStd: number[] = octave.xstd.map((row: number[]) => row[2]);
const octAR: number[] = octave.x[4];
const octARStd: number[] = octave.xstd.map((row: number[]) => row[4]);

function varianceCombinedOct(i: number): number {
  const c00 = octave.C[0][0][i];
  const c22 = octave.C[2][2][i];
  const c44 = octave.C[4][4][i];
  const c02 = octave.C[0][2][i];
  const c04 = octave.C[0][4][i];
  const c24 = octave.C[2][4][i];
  return Math.max(0, c00 + c22 + c44 + 2 * c02 + 2 * c04 + 2 * c24);
}
const octCombined = octLevel.map((v, i) => v + octSeasonal[i] + octAR[i]);
const octCombinedStd = Array.from({ length: n }, (_, i) => Math.sqrt(varianceCombinedOct(i)));

// ── SVG layout ─────────────────────────────────────────────────────────────

const W = 900;
const H = 960;
const outer = { left: 66, right: 22, top: 34, bottom: 52, gap: 28 };
const plotW = W - outer.left - outer.right;
const numPanels = 4;
const panelH = (H - outer.top - outer.bottom - outer.gap * (numPanels - 1)) / numPanels;

const panelTops = Array.from({ length: numPanels }, (_, i) =>
  outer.top + i * (panelH + outer.gap),
);

const tMin = t[0];
const tMax = t[n - 1];

// Colors
const obsColor = "#555";
const jsColor = "#2563eb";       // blue
const octColor = "#ef4444";      // red
const trueColor = "#16a34a";     // green
const jsBandColor = "rgba(37,99,235,0.12)";
const octBandColor = "rgba(239,68,68,0.12)";

// X-axis ticks: every 12 months (years)
const xTicks: { val: number; label: string }[] = [];
for (let m = 12; m <= n; m += 12) {
  xTicks.push({ val: m, label: `${Math.floor(m / 12)}y` });
}

function sx(val: number): number {
  return outer.left + ((val - tMin) / (tMax - tMin)) * plotW;
}

function makeSy(yMin: number, yMax: number, panelTop: number) {
  return (val: number): number => panelTop + ((yMax - val) / (yMax - yMin)) * panelH;
}

// ── Build SVG ──────────────────────────────────────────────────────────────

const lines: string[] = [];
const push = (s: string) => lines.push(s);

push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="system-ui,-apple-system,sans-serif" font-size="12">`);
push(`<rect width="${W}" height="${H}" fill="white"/>`);
push(`<text x="${outer.left + plotW / 2}" y="18" text-anchor="middle" fill="#333" font-size="14" font-weight="600">Synthetic energy demand — trend + seasonal + strong AR(1)</text>`);

interface PanelSpec {
  title: string;
  jsMean: number[];
  jsStd: number[];
  octMean: number[];
  octStd: number[];
  trueMean: number[];
  showObs: boolean;
  yLabel: string;
}

const panels: PanelSpec[] = [
  {
    title: "Smoothed level x[0] ± 2σ",
    jsMean: jsLevel,
    jsStd: jsLevelStd,
    octMean: octLevel,
    octStd: octLevelStd,
    trueMean: trueLevel,
    showObs: false,
    yLabel: "Level",
  },
  {
    title: "Seasonal x[2] ± 2σ  (trig harmonic k=1)",
    jsMean: jsSeasonal,
    jsStd: jsSeasonalStd,
    octMean: octSeasonal,
    octStd: octSeasonalStd,
    trueMean: trueSeasonal,
    showObs: false,
    yLabel: "Seasonal",
  },
  {
    title: "AR(1) state x[4] ± 2σ  (φ = 0.85)",
    jsMean: jsAR as number[],
    jsStd: jsARStd as number[],
    octMean: octAR,
    octStd: octARStd,
    trueMean: trueAR,
    showObs: false,
    yLabel: "AR state",
  },
  {
    title: "Combined F·x = x[0]+x[2]+x[4] ± 2σ (covariance-aware)",
    jsMean: jsCombined,
    jsStd: jsCombinedStd,
    octMean: octCombined,
    octStd: octCombinedStd,
    trueMean: trueCombined,
    showObs: true,
    yLabel: "Demand",
  },
];

for (let p = 0; p < panels.length; p++) {
  const panelTop = panelTops[p];
  const panel = panels[p];
  const showXAxis = p === panels.length - 1;

  const jsUpper = panel.jsMean.map((v, i) => v + 2 * panel.jsStd[i]);
  const jsLower = panel.jsMean.map((v, i) => v - 2 * panel.jsStd[i]);
  const octUpper = panel.octMean.map((v, i) => v + 2 * panel.octStd[i]);
  const octLower = panel.octMean.map((v, i) => v - 2 * panel.octStd[i]);

  const allVals = [
    ...jsUpper, ...jsLower, ...octUpper, ...octLower,
    ...panel.trueMean,
    ...(panel.showObs ? y : []),
  ];
  const rawMin = Math.min(...allVals);
  const rawMax = Math.max(...allVals);
  const pad = (rawMax - rawMin) * 0.05 + 1;
  const yMin = rawMin - pad;
  const yMax = rawMax + pad;
  const sy = makeSy(yMin, yMax, panelTop);

  const yTicks = yTicksFromRange(yMin, yMax);

  // Grid
  lines.push(...renderGridLines(yTicks, sy, outer.left, W - outer.right));

  // Bands
  push(`<path d="${bandPathD(t, jsUpper, jsLower, sx, sy)}" fill="${jsBandColor}" stroke="none"/>`);
  push(`<path d="${bandPathD(t, octUpper, octLower, sx, sy)}" fill="${octBandColor}" stroke="none"/>`);

  // True state (green, thin dashed)
  push(`<polyline points="${polylinePoints(t, panel.trueMean, sx, sy)}" fill="none" stroke="${trueColor}" stroke-width="1.5" stroke-dasharray="3,3"/>`);

  // Smoothed lines
  push(`<polyline points="${polylinePoints(t, panel.jsMean, sx, sy)}" fill="none" stroke="${jsColor}" stroke-width="2"/>`);
  push(`<polyline points="${polylinePoints(t, panel.octMean, sx, sy)}" fill="none" stroke="${octColor}" stroke-width="2" stroke-dasharray="6,3"/>`);

  // Observations
  if (panel.showObs) {
    for (let i = 0; i < n; i++) {
      push(`<circle cx="${r(sx(t[i]))}" cy="${r(sy(y[i]))}" r="2.0" fill="${obsColor}" opacity="0.50"/>`);
    }
  }

  // Axes
  lines.push(...renderAxesBorder(outer.left, panelTop, W - outer.right, panelTop + panelH));
  lines.push(...renderYAxis(yTicks, sy, outer.left));

  // X-axis ticks (bottom panel only)
  if (showXAxis) {
    lines.push(...renderXAxis(xTicks.map(tick => ({ val: tick.val, label: tick.label })), sx, panelTop + panelH));
  }

  // Panel title
  push(`<text x="${outer.left + 6}" y="${panelTop + 14}" text-anchor="start" fill="#333" font-size="12" font-weight="600">${panel.title}</text>`);

  // Y-axis label
  push(`<text x="14" y="${panelTop + panelH / 2}" text-anchor="middle" fill="#333" font-size="13" transform="rotate(-90,14,${panelTop + panelH / 2})">${panel.yLabel}</text>`);
}

// X-axis label
push(`<text x="${W / 2}" y="${H - 8}" text-anchor="middle" fill="#333" font-size="13">Month</text>`);

// Legend (in first panel)
const legW = 308;
const legX = outer.left + (plotW - legW) / 2;
const legY = panelTops[0] + 20;
push(`<rect x="${legX}" y="${legY}" width="${legW}" height="78" rx="4" fill="white" stroke="#e5e7eb" stroke-width="1"/>`);
push(`<circle cx="${legX + 14}" cy="${legY + 14}" r="3" fill="${obsColor}" opacity="0.6"/>`);
push(`<text x="${legX + 24}" y="${legY + 14}" dominant-baseline="middle" fill="#333" font-size="11">Observations (synthetic energy demand)</text>`);
push(`<line x1="${legX + 8}" y1="${legY + 30}" x2="${legX + 20}" y2="${legY + 30}" stroke="${trueColor}" stroke-width="1.5" stroke-dasharray="3,3"/>`);
push(`<text x="${legX + 24}" y="${legY + 30}" dominant-baseline="middle" fill="#333" font-size="11">True hidden state (generating process)</text>`);
push(`<line x1="${legX + 8}" y1="${legY + 46}" x2="${legX + 20}" y2="${legY + 46}" stroke="${jsColor}" stroke-width="2"/>`);
push(`<rect x="${legX + 8}" y="${legY + 41}" width="12" height="10" fill="${jsBandColor}" stroke="none"/>`);
push(`<text x="${legX + 24}" y="${legY + 46}" dominant-baseline="middle" fill="#333" font-size="11">dlm-js mean ± 2σ</text>`);
push(`<line x1="${legX + 8}" y1="${legY + 62}" x2="${legX + 20}" y2="${legY + 62}" stroke="${octColor}" stroke-width="2" stroke-dasharray="6,3"/>`);
push(`<rect x="${legX + 8}" y="${legY + 57}" width="12" height="10" fill="${octBandColor}" stroke="none"/>`);
push(`<text x="${legX + 24}" y="${legY + 62}" dominant-baseline="middle" fill="#333" font-size="11">MATLAB/Octave mean ± 2σ</text>`);

push(`</svg>`);

// ── Write output ───────────────────────────────────────────────────────────

const outPath = resolve(root, "assets", "trigar.svg");
writeSvg(lines, outPath);
writeTimingsSidecar("gen-trigar-svg", { firstRunMs: timed.firstRunMs, warmRunMs: timed.warmRunMs });
console.log(
  `Timing (dlmFit with jitted core): first-run ${timed.firstRunMs.toFixed(2)} ms, warm-run ${timed.warmRunMs.toFixed(2)} ms`,
);
