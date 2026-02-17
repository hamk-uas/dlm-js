/**
 * Generate an SVG plot of the Kaisaniemi seasonal demo from mjlaine/dlm.
 *
 * Uses:
 * - Input data exported by tests/octave/kaisaniemi_demo.m
 * - Octave reference output from tests/kaisaniemi-out-m.json
 * - dlm-js output from src/index.ts dlmFit
 *
 * Plot: smoothed level state x[0] ± 2σ bands for dlm-js and MATLAB/Octave.
 */

import { dlmFit } from "../src/index.ts";
import { DType } from "@jax-js-nonconsuming/jax";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const input = JSON.parse(readFileSync(resolve(root, "tests/kaisaniemi-in.json"), "utf8"));
const octave = JSON.parse(readFileSync(resolve(root, "tests/kaisaniemi-out-m.json"), "utf8"));

const tDatenum: number[] = input.t;
const y: number[] = input.y;
const s: number = input.s;
const w: number[] = input.w;
const options = input.options;

const jsResult = await dlmFit(y, s, w, DType.Float64, options);
const jsLevel = Array.from(jsResult.x[0]);
const jsLevelStd = jsResult.xstd.map((row: any) => row[0] as number);

const octLevel: number[] = octave.x[0];
const octLevelStd: number[] = octave.xstd.map((row: number[]) => row[0]);

const n = tDatenum.length;

const margin = { top: 34, right: 20, bottom: 52, left: 64 };
const W = 860;
const H = 360;
const plotW = W - margin.left - margin.right;
const plotH = H - margin.top - margin.bottom;

const tMin = tDatenum[0];
const tMax = tDatenum[n - 1];

const allVals = [
  ...y,
  ...jsLevel.map((v, i) => v + 2 * jsLevelStd[i]),
  ...jsLevel.map((v, i) => v - 2 * jsLevelStd[i]),
  ...octLevel.map((v, i) => v + 2 * octLevelStd[i]),
  ...octLevel.map((v, i) => v - 2 * octLevelStd[i]),
];

const yMin = Math.floor((Math.min(...allVals) - 1) / 2) * 2;
const yMax = Math.ceil((Math.max(...allVals) + 1) / 2) * 2;

const matlabDatenumToYear = (dn: number): number => {
  const unixMs = (dn - 719529) * 86400000;
  const d = new Date(unixMs);
  return d.getUTCFullYear() + (d.getUTCMonth() + 0.5) / 12;
};

function sx(val: number): number {
  return margin.left + ((val - tMin) / (tMax - tMin)) * plotW;
}
function sy(val: number): number {
  return margin.top + ((yMax - val) / (yMax - yMin)) * plotH;
}

function polyline(xs: number[], ys: number[]): string {
  return xs.map((x, i) => `${sx(x).toFixed(1)},${sy(ys[i]).toFixed(1)}`).join(" ");
}

function bandPath(xs: number[], upper: number[], lower: number[]): string {
  const fwd = xs.map((x, i) => `${sx(x).toFixed(1)},${sy(upper[i]).toFixed(1)}`);
  const bwd = xs.map((x, i) => `${sx(x).toFixed(1)},${sy(lower[i]).toFixed(1)}`).reverse();
  return `M${fwd.join("L")}L${bwd.join("L")}Z`;
}

const jsUpper = jsLevel.map((v, i) => v + 2 * jsLevelStd[i]);
const jsLower = jsLevel.map((v, i) => v - 2 * jsLevelStd[i]);
const octUpper = octLevel.map((v, i) => v + 2 * octLevelStd[i]);
const octLower = octLevel.map((v, i) => v - 2 * octLevelStd[i]);

const yTicks: number[] = [];
for (let v = yMin; v <= yMax; v += 4) yTicks.push(v);

const startYear = Math.floor(matlabDatenumToYear(tMin));
const endYear = Math.ceil(matlabDatenumToYear(tMax));
const yearTicks: { dn: number; label: string }[] = [];
for (let year = startYear; year <= endYear; year += 1) {
  const jan1 = Date.UTC(year, 0, 1);
  const dn = jan1 / 86400000 + 719529;
  if (dn >= tMin && dn <= tMax) yearTicks.push({ dn, label: `${year}` });
}

const obsColor = "#555";
const jsColor = "#2563eb";
const octColor = "#ef4444";
const jsBandColor = "rgba(37,99,235,0.12)";
const octBandColor = "rgba(239,68,68,0.12)";

const lines: string[] = [];
const push = (s: string) => lines.push(s);

push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="system-ui,-apple-system,sans-serif" font-size="12">`);
push(`<rect width="${W}" height="${H}" fill="white"/>`);

for (const v of yTicks) {
  push(`<line x1="${margin.left}" y1="${sy(v).toFixed(1)}" x2="${W - margin.right}" y2="${sy(v).toFixed(1)}" stroke="#e5e7eb" stroke-width="1"/>`);
}

push(`<path d="${bandPath(tDatenum, jsUpper, jsLower)}" fill="${jsBandColor}" stroke="none"/>`);
push(`<path d="${bandPath(tDatenum, octUpper, octLower)}" fill="${octBandColor}" stroke="none"/>`);
push(`<polyline points="${polyline(tDatenum, jsLevel)}" fill="none" stroke="${jsColor}" stroke-width="2"/>`);
push(`<polyline points="${polyline(tDatenum, octLevel)}" fill="none" stroke="${octColor}" stroke-width="2" stroke-dasharray="6,3"/>`);

for (let i = 0; i < n; i++) {
  push(`<circle cx="${sx(tDatenum[i]).toFixed(1)}" cy="${sy(y[i]).toFixed(1)}" r="2.2" fill="${obsColor}" opacity="0.55"/>`);
}

push(`<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${H - margin.bottom}" stroke="#333" stroke-width="1.5"/>`);
push(`<line x1="${margin.left}" y1="${H - margin.bottom}" x2="${W - margin.right}" y2="${H - margin.bottom}" stroke="#333" stroke-width="1.5"/>`);

for (const v of yTicks) {
  const yy = sy(v).toFixed(1);
  push(`<line x1="${margin.left - 5}" y1="${yy}" x2="${margin.left}" y2="${yy}" stroke="#333" stroke-width="1.5"/>`);
  push(`<text x="${margin.left - 8}" y="${yy}" text-anchor="end" dominant-baseline="middle" fill="#333">${v}</text>`);
}

for (const tick of yearTicks) {
  const xx = sx(tick.dn).toFixed(1);
  push(`<line x1="${xx}" y1="${H - margin.bottom}" x2="${xx}" y2="${H - margin.bottom + 5}" stroke="#333" stroke-width="1.5"/>`);
  push(`<text x="${xx}" y="${H - margin.bottom + 18}" text-anchor="middle" fill="#333">${tick.label}</text>`);
}

push(`<text x="${margin.left + plotW / 2}" y="${H - 6}" text-anchor="middle" fill="#333" font-size="13">Year</text>`);
push(`<text x="${14}" y="${margin.top + plotH / 2}" text-anchor="middle" fill="#333" font-size="13" transform="rotate(-90,14,${margin.top + plotH / 2})">Temperature (°C)</text>`);
push(`<text x="${margin.left + plotW / 2}" y="${17}" text-anchor="middle" fill="#333" font-size="14" font-weight="600">Kaisaniemi seasonal demo — smoothed level state x[0] ± 2σ</text>`);

const legX = W - margin.right - 318;
const legY = margin.top + 8;
push(`<rect x="${legX}" y="${legY}" width="312" height="62" rx="4" fill="white" stroke="#e5e7eb" stroke-width="1"/>`);
push(`<circle cx="${legX + 14}" cy="${legY + 14}" r="3" fill="${obsColor}" opacity="0.6"/>`);
push(`<text x="${legX + 24}" y="${legY + 14}" dominant-baseline="middle" fill="#333" font-size="11">Observed monthly temperature</text>`);
push(`<line x1="${legX + 8}" y1="${legY + 30}" x2="${legX + 20}" y2="${legY + 30}" stroke="${jsColor}" stroke-width="2"/>`);
push(`<rect x="${legX + 8}" y="${legY + 25}" width="12" height="10" fill="${jsBandColor}" stroke="none"/>`);
push(`<text x="${legX + 24}" y="${legY + 30}" dominant-baseline="middle" fill="#333" font-size="11">dlm-js level x[0] ± 2σ</text>`);
push(`<line x1="${legX + 8}" y1="${legY + 46}" x2="${legX + 20}" y2="${legY + 46}" stroke="${octColor}" stroke-width="2" stroke-dasharray="6,3"/>`);
push(`<rect x="${legX + 8}" y="${legY + 41}" width="12" height="10" fill="${octBandColor}" stroke="none"/>`);
push(`<text x="${legX + 24}" y="${legY + 46}" dominant-baseline="middle" fill="#333" font-size="11">MATLAB/Octave level x[0] ± 2σ</text>`);

push(`</svg>`);

const outDir = resolve(root, "assets");
mkdirSync(outDir, { recursive: true });
const outPath = resolve(outDir, "kaisaniemi.svg");
writeFileSync(outPath, lines.join("\n"), "utf8");

console.log(`Written: ${outPath}`);
