/**
 * Generate an SVG plot of the Kaisaniemi seasonal demo from mjlaine/dlm.
 *
 * Two panels:
 * 1) Smoothed level state x[0] ± 2σ
 * 2) Combined signal x[0] + x[2] ± 2σ, where
 *      Var(x0+x2) = Var(x0) + Var(x2) + 2*Cov(x0,x2)
 */

import { dlmFit } from "../src/index.ts";
import { DType } from "@hamk-uas/jax-js-nonconsuming";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { performance } from "node:perf_hooks";
import {
  r, makeLinearScale, polylinePoints, bandPathD, yTicksFromRange,
  renderGridLines, renderYAxis, renderXAxis, renderAxesBorder, writeSvg,
} from "./lib/svg-helpers.ts";
import { withLeakCheck } from "./lib/leak-utils.ts";
import { writeTimingsSidecar } from "./lib/timing-sidecar.ts";

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const input = JSON.parse(readFileSync(resolve(root, "tests/kaisaniemi-in.json"), "utf8"));
const octave = JSON.parse(readFileSync(resolve(root, "tests/kaisaniemi-out-m.json"), "utf8"));

const tDatenum: number[] = input.t;
const y: number[] = input.y;
const s: number = input.s;
const w: number[] = input.w;
const options = input.options;

const variant = process.argv[2] === 'assoc' ? 'assoc' : 'scan';
const isAssoc = variant === 'assoc';
const scanLabel = isAssoc ? 'associativeScan/WASM/f64' : 'scan/WASM/f64';

const timedFit = async () => {
  const t0 = performance.now();
  await withLeakCheck(() => dlmFit(y, s, w, DType.Float64, options, undefined, isAssoc));
  const t1 = performance.now();

  const warmStart = performance.now();
  const result = await withLeakCheck(() => dlmFit(y, s, w, DType.Float64, options, undefined, isAssoc));
  const warmEnd = performance.now();

  return {
    result,
    firstRunMs: t1 - t0,
    warmRunMs: warmEnd - warmStart,
  };
};

const timed = await timedFit();
const jsResult = timed.result;

const jsLevel = Array.from(jsResult.x[0]);
const jsLevelStd = jsResult.xstd.map((row: any) => row[0] as number);
const jsSeasObs = Array.from(jsResult.x[2]);

const octLevel: number[] = octave.x[0];
const octLevelStd: number[] = octave.xstd.map((row: number[]) => row[0]);
const octSeasObs: number[] = octave.x[2];

const n = tDatenum.length;

function varianceCombinedFromJs(t: number): number {
  const c00 = jsResult.C[0][0][t];
  const c22 = jsResult.C[2][2][t];
  const c02 = jsResult.C[0][2][t];
  return Math.max(0, c00 + c22 + 2 * c02);
}

function varianceCombinedFromOct(t: number): number {
  const c00 = octave.C[0][0][t];
  const c22 = octave.C[2][2][t];
  const c02 = octave.C[0][2][t];
  return Math.max(0, c00 + c22 + 2 * c02);
}

const jsCombined = jsLevel.map((v, i) => v + jsSeasObs[i]);
const jsCombinedStd = Array.from({ length: n }, (_, i) => Math.sqrt(varianceCombinedFromJs(i)));

const octCombined = octLevel.map((v, i) => v + octSeasObs[i]);
const octCombinedStd = Array.from({ length: n }, (_, i) => Math.sqrt(varianceCombinedFromOct(i)));

const panel1 = {
  title: "Smoothed level state x[0] ± 2σ",
  jsMean: jsLevel,
  jsStd: jsLevelStd,
  octMean: octLevel,
  octStd: octLevelStd,
};

const panel2 = {
  title: "Combined x[0] + seasonal x[2] ± 2σ (covariance-aware)",
  jsMean: jsCombined,
  jsStd: jsCombinedStd,
  octMean: octCombined,
  octStd: octCombinedStd,
};

const matlabDatenumToYear = (dn: number): number => {
  const unixMs = (dn - 719529) * 86400000;
  const d = new Date(unixMs);
  return d.getUTCFullYear() + (d.getUTCMonth() + 0.5) / 12;
};

const tMin = tDatenum[0];
const tMax = tDatenum[n - 1];

const W = 900;
const H = 640;
const outer = { left: 66, right: 22, top: 34, bottom: 52, gap: 48 };
const plotW = W - outer.left - outer.right;
const panelH = (H - outer.top - outer.bottom - outer.gap) / 2;

const panel1Top = outer.top;
const panel2Top = outer.top + panelH + outer.gap;

const obsColor = "#555";
const jsColor = "#2563eb";
const octColor = "#ef4444";
const jsBandColor = "rgba(37,99,235,0.12)";
const octBandColor = "rgba(239,68,68,0.12)";

const startYear = Math.floor(matlabDatenumToYear(tMin));
const endYear = Math.ceil(matlabDatenumToYear(tMax));
const yearTicks: { dn: number; label: string }[] = [];
for (let year = startYear; year <= endYear; year += 1) {
  const jan1 = Date.UTC(year, 0, 1);
  const dn = jan1 / 86400000 + 719529;
  if (dn >= tMin && dn <= tMax) yearTicks.push({ dn, label: `${year}` });
}

function sx(val: number): number {
  return outer.left + ((val - tMin) / (tMax - tMin)) * plotW;
}

function makeSy(yMin: number, yMax: number, panelTop: number) {
  return makeLinearScale(yMin, yMax, panelTop + panelH, panelTop);
}

const lines: string[] = [];
const push = (s: string) => lines.push(s);

push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="system-ui,-apple-system,sans-serif" font-size="12">`);
push(`<rect width="${W}" height="${H}" fill="white"/>`);
push(`<text x="${outer.left + plotW / 2}" y="18" text-anchor="middle" fill="#333" font-size="14" font-weight="600">Kaisaniemi demo — fit (order=1, trig, s=2), ${timed.warmRunMs.toFixed(0)} ms, ${scanLabel}</text>`);

function drawPanel(
  panelTop: number,
  panel: { title: string; jsMean: number[]; jsStd: number[]; octMean: number[]; octStd: number[] },
  showXAxis: boolean,
) {
  const jsUpper = panel.jsMean.map((v, i) => v + 2 * panel.jsStd[i]);
  const jsLower = panel.jsMean.map((v, i) => v - 2 * panel.jsStd[i]);
  const octUpper = panel.octMean.map((v, i) => v + 2 * panel.octStd[i]);
  const octLower = panel.octMean.map((v, i) => v - 2 * panel.octStd[i]);

  const allVals = [...y, ...jsUpper, ...jsLower, ...octUpper, ...octLower];
  const yMin = Math.floor((Math.min(...allVals) - 1) / 2) * 2;
  const yMax = Math.ceil((Math.max(...allVals) + 1) / 2) * 2;
  const sy = makeSy(yMin, yMax, panelTop);

  const yTicks = yTicksFromRange(yMin, yMax);

  lines.push(...renderGridLines(yTicks, sy, outer.left, W - outer.right));

  push(`<path d="${bandPathD(tDatenum, jsUpper, jsLower, sx, sy)}" fill="${jsBandColor}" stroke="none"/>`);
  push(`<path d="${bandPathD(tDatenum, octUpper, octLower, sx, sy)}" fill="${octBandColor}" stroke="none"/>`);
  push(`<polyline points="${polylinePoints(tDatenum, panel.jsMean, sx, sy)}" fill="none" stroke="${jsColor}" stroke-width="2"/>`);
  push(`<polyline points="${polylinePoints(tDatenum, panel.octMean, sx, sy)}" fill="none" stroke="${octColor}" stroke-width="2" stroke-dasharray="6,3"/>`);

  for (let i = 0; i < n; i++) {
    push(`<circle cx="${r(sx(tDatenum[i]))}" cy="${r(sy(y[i]))}" r="2.0" fill="${obsColor}" opacity="0.50"/>`);
  }

  lines.push(...renderAxesBorder(outer.left, panelTop, W - outer.right, panelTop + panelH));
  lines.push(...renderYAxis(yTicks, sy, outer.left));

  if (showXAxis) {
    lines.push(...renderXAxis(yearTicks.map(tick => ({ val: tick.dn, label: tick.label })), sx, panelTop + panelH));
  }

  push(`<text x="${outer.left + 6}" y="${panelTop + 14}" text-anchor="start" fill="#333" font-size="12" font-weight="600">${panel.title}</text>`);
}

drawPanel(panel1Top, panel1, false);
drawPanel(panel2Top, panel2, true);

push(`<text x="${W / 2}" y="${H - 8}" text-anchor="middle" fill="#333" font-size="13">Year</text>`);
push(`<text x="14" y="${panel1Top + panelH / 2}" text-anchor="middle" fill="#333" font-size="13" transform="rotate(-90,14,${panel1Top + panelH / 2})">Temperature (°C)</text>`);
push(`<text x="14" y="${panel2Top + panelH / 2}" text-anchor="middle" fill="#333" font-size="13" transform="rotate(-90,14,${panel2Top + panelH / 2})">Temperature (°C)</text>`);

const legX = W - outer.right - 316;
const legY = panel1Top + 20;
push(`<rect x="${legX}" y="${legY}" width="308" height="62" rx="4" fill="white" stroke="#e5e7eb" stroke-width="1"/>`);
push(`<circle cx="${legX + 14}" cy="${legY + 14}" r="3" fill="${obsColor}" opacity="0.6"/>`);
push(`<text x="${legX + 24}" y="${legY + 14}" dominant-baseline="middle" fill="#333" font-size="11">Observed monthly temperature</text>`);
push(`<line x1="${legX + 8}" y1="${legY + 30}" x2="${legX + 20}" y2="${legY + 30}" stroke="${jsColor}" stroke-width="2"/>`);
push(`<rect x="${legX + 8}" y="${legY + 25}" width="12" height="10" fill="${jsBandColor}" stroke="none"/>`);
push(`<text x="${legX + 24}" y="${legY + 30}" dominant-baseline="middle" fill="#333" font-size="11">dlm-js mean ± 2σ</text>`);
push(`<line x1="${legX + 8}" y1="${legY + 46}" x2="${legX + 20}" y2="${legY + 46}" stroke="${octColor}" stroke-width="2" stroke-dasharray="6,3"/>`);
push(`<rect x="${legX + 8}" y="${legY + 41}" width="12" height="10" fill="${octBandColor}" stroke="none"/>`);
push(`<text x="${legX + 24}" y="${legY + 46}" dominant-baseline="middle" fill="#333" font-size="11">MATLAB/Octave mean ± 2σ</text>`);

push(`</svg>`);

const outPath = resolve(root, "assets", `kaisaniemi-${variant}.svg`);
writeSvg(lines, outPath);
writeTimingsSidecar(isAssoc ? "gen-kaisaniemi-svg-assoc" : "gen-kaisaniemi-svg", { firstRunMs: timed.firstRunMs, warmRunMs: timed.warmRunMs });
console.log(
  `Timing (dlmFit with jitted core): first-run ${timed.firstRunMs.toFixed(2)} ms, warm-run ${timed.warmRunMs.toFixed(2)} ms`
);
