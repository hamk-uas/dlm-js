/**
 * Generate an SVG plot of the Kaisaniemi seasonal demo from mjlaine/dlm.
 *
 * Two panels:
 * 1) Smoothed level state x[0] ± 2σ
 * 2) Combined signal x[0] + x[2] ± 2σ, where
 *      Var(x0+x2) = Var(x0) + Var(x2) + 2*Cov(x0,x2)
 */

import { dlmFit } from "../src/index.ts";
import { checkLeaks, DType } from "@jax-js-nonconsuming/jax";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { performance } from "node:perf_hooks";

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const input = JSON.parse(readFileSync(resolve(root, "tests/kaisaniemi-in.json"), "utf8"));
const octave = JSON.parse(readFileSync(resolve(root, "tests/kaisaniemi-out-m.json"), "utf8"));

const tDatenum: number[] = input.t;
const y: number[] = input.y;
const s: number = input.s;
const w: number[] = input.w;
const options = input.options;

const withLeakCheck = async <T>(fn: () => Promise<T>): Promise<T> => {
  checkLeaks.start();
  try {
    return await fn();
  } finally {
    checkLeaks.stop();
  }
};

const timedFit = async () => {
  const t0 = performance.now();
  await withLeakCheck(() => dlmFit(y, s, w, DType.Float64, options));
  const t1 = performance.now();

  const warmStart = performance.now();
  const result = await withLeakCheck(() => dlmFit(y, s, w, DType.Float64, options));
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
  return (val: number): number => panelTop + ((yMax - val) / (yMax - yMin)) * panelH;
}

function polyline(xs: number[], ys: number[], sy: (val: number) => number): string {
  return xs.map((x, i) => `${sx(x).toFixed(1)},${sy(ys[i]).toFixed(1)}`).join(" ");
}

function bandPath(
  xs: number[],
  upper: number[],
  lower: number[],
  sy: (val: number) => number,
): string {
  const fwd = xs.map((x, i) => `${sx(x).toFixed(1)},${sy(upper[i]).toFixed(1)}`);
  const bwd = xs.map((x, i) => `${sx(x).toFixed(1)},${sy(lower[i]).toFixed(1)}`).reverse();
  return `M${fwd.join("L")}L${bwd.join("L")}Z`;
}

function yTicksFromRange(min: number, max: number): number[] {
  const span = max - min;
  const step = span > 20 ? 4 : span > 8 ? 2 : 1;
  const ticks: number[] = [];
  const start = Math.floor(min / step) * step;
  const end = Math.ceil(max / step) * step;
  for (let v = start; v <= end; v += step) ticks.push(v);
  return ticks;
}

const lines: string[] = [];
const push = (s: string) => lines.push(s);

push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="system-ui,-apple-system,sans-serif" font-size="12">`);
push(`<rect width="${W}" height="${H}" fill="white"/>`);
push(`<text x="${outer.left + plotW / 2}" y="18" text-anchor="middle" fill="#333" font-size="14" font-weight="600">Kaisaniemi demo — level and covariance-aware combined signal</text>`);

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

  for (const v of yTicks) {
    push(`<line x1="${outer.left}" y1="${sy(v).toFixed(1)}" x2="${W - outer.right}" y2="${sy(v).toFixed(1)}" stroke="#e5e7eb" stroke-width="1"/>`);
  }

  push(`<path d="${bandPath(tDatenum, jsUpper, jsLower, sy)}" fill="${jsBandColor}" stroke="none"/>`);
  push(`<path d="${bandPath(tDatenum, octUpper, octLower, sy)}" fill="${octBandColor}" stroke="none"/>`);
  push(`<polyline points="${polyline(tDatenum, panel.jsMean, sy)}" fill="none" stroke="${jsColor}" stroke-width="2"/>`);
  push(`<polyline points="${polyline(tDatenum, panel.octMean, sy)}" fill="none" stroke="${octColor}" stroke-width="2" stroke-dasharray="6,3"/>`);

  for (let i = 0; i < n; i++) {
    push(`<circle cx="${sx(tDatenum[i]).toFixed(1)}" cy="${sy(y[i]).toFixed(1)}" r="2.0" fill="${obsColor}" opacity="0.50"/>`);
  }

  push(`<line x1="${outer.left}" y1="${panelTop}" x2="${outer.left}" y2="${panelTop + panelH}" stroke="#333" stroke-width="1.5"/>`);
  push(`<line x1="${outer.left}" y1="${panelTop + panelH}" x2="${W - outer.right}" y2="${panelTop + panelH}" stroke="#333" stroke-width="1.5"/>`);

  for (const v of yTicks) {
    const yy = sy(v).toFixed(1);
    push(`<line x1="${outer.left - 5}" y1="${yy}" x2="${outer.left}" y2="${yy}" stroke="#333" stroke-width="1.5"/>`);
    push(`<text x="${outer.left - 8}" y="${yy}" text-anchor="end" dominant-baseline="middle" fill="#333">${v}</text>`);
  }

  if (showXAxis) {
    for (const tick of yearTicks) {
      const xx = sx(tick.dn).toFixed(1);
      push(`<line x1="${xx}" y1="${panelTop + panelH}" x2="${xx}" y2="${panelTop + panelH + 5}" stroke="#333" stroke-width="1.5"/>`);
      push(`<text x="${xx}" y="${panelTop + panelH + 18}" text-anchor="middle" fill="#333">${tick.label}</text>`);
    }
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

const outDir = resolve(root, "assets");
mkdirSync(outDir, { recursive: true });
const outPath = resolve(outDir, "kaisaniemi.svg");
writeFileSync(outPath, lines.join("\n"), "utf8");

console.log(`Written: ${outPath}`);
console.log(
  `Timing (dlmFit with jitted core): first-run ${timed.firstRunMs.toFixed(2)} ms, warm-run ${timed.warmRunMs.toFixed(2)} ms`
);
