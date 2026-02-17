/**
 * Generate an SVG plot of the Nile demo: observations, smoothed level, and
 * ±2σ confidence bands from both dlm-js and MATLAB/Octave dlm.
 *
 * Usage:  npx tsx scripts/gen-niledemo-svg.ts
 * Output: assets/niledemo.svg
 */

import { dlmFit } from "../src/index.ts";
import { DType } from "@jax-js-nonconsuming/jax";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";

// ── Load data ──────────────────────────────────────────────────────────────

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const input = JSON.parse(readFileSync(resolve(root, "tests/niledemo-in.json"), "utf8"));
const octave = JSON.parse(readFileSync(resolve(root, "tests/niledemo-out-m.json"), "utf8"));

const t: number[] = input.t;        // years 1871–1970
const y: number[] = input.y;        // observations
const s: number = input.s;          // observation noise std
const w: number[] = input.w;        // state noise stds

// ── Run dlm-js ─────────────────────────────────────────────────────────────

const jsResult = await dlmFit(y, s, w, DType.Float64, { order: 1 });
const jsYhat = Array.from(jsResult.yhat);
const jsYstd = Array.from(jsResult.ystd);

// ── Octave results ─────────────────────────────────────────────────────────

const octYhat: number[] = octave.yhat;
const octYstd: number[] = octave.ystd;

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
  ...jsYhat.map((v, i) => v + 2 * jsYstd[i]),
  ...jsYhat.map((v, i) => v - 2 * jsYstd[i]),
];
const yMin = Math.floor(Math.min(...allVals) / 50) * 50;
const yMax = Math.ceil(Math.max(...allVals) / 50) * 50;

function sx(val: number): number {
  return margin.left + ((val - tMin) / (tMax - tMin)) * plotW;
}
function sy(val: number): number {
  return margin.top + ((yMax - val) / (yMax - yMin)) * plotH;
}

// Helpers
function polyline(xs: number[], ys: number[]): string {
  return xs.map((x, i) => `${sx(x).toFixed(1)},${sy(ys[i]).toFixed(1)}`).join(" ");
}

function bandPath(xs: number[], upper: number[], lower: number[]): string {
  const fwd = xs.map((x, i) => `${sx(x).toFixed(1)},${sy(upper[i]).toFixed(1)}`);
  const bwd = xs.map((x, i) => `${sx(x).toFixed(1)},${sy(lower[i]).toFixed(1)}`).reverse();
  return `M${fwd.join("L")}L${bwd.join("L")}Z`;
}

// Confidence bands
const jsUpper = jsYhat.map((v, i) => v + 2 * jsYstd[i]);
const jsLower = jsYhat.map((v, i) => v - 2 * jsYstd[i]);
const octUpper = octYhat.map((v, i) => v + 2 * octYstd[i]);
const octLower = octYhat.map((v, i) => v - 2 * octYstd[i]);

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
for (const v of yTicks) {
  push(`<line x1="${margin.left}" y1="${sy(v).toFixed(1)}" x2="${W - margin.right}" y2="${sy(v).toFixed(1)}" stroke="#e5e7eb" stroke-width="1"/>`);
}

// Confidence bands (dlm-js first / behind)
push(`<path d="${bandPath(t, jsUpper, jsLower)}" fill="${jsBandColor}" stroke="none"/>`);
push(`<path d="${bandPath(t, octUpper, octLower)}" fill="${octBandColor}" stroke="none"/>`);

// Smoothed level lines (Octave dashed on top)
push(`<polyline points="${polyline(t, jsYhat)}" fill="none" stroke="${jsColor}" stroke-width="2"/>`);
push(`<polyline points="${polyline(t, octYhat)}" fill="none" stroke="${octColor}" stroke-width="2" stroke-dasharray="6,3"/>`);

// Observations
for (let i = 0; i < n; i++) {
  push(`<circle cx="${sx(t[i]).toFixed(1)}" cy="${sy(y[i]).toFixed(1)}" r="2.5" fill="${obsColor}" opacity="0.6"/>`);
}

// Axes
push(`<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${H - margin.bottom}" stroke="#333" stroke-width="1.5"/>`);
push(`<line x1="${margin.left}" y1="${H - margin.bottom}" x2="${W - margin.right}" y2="${H - margin.bottom}" stroke="#333" stroke-width="1.5"/>`);

// Y-axis ticks + labels
for (const v of yTicks) {
  const yy = sy(v).toFixed(1);
  push(`<line x1="${margin.left - 5}" y1="${yy}" x2="${margin.left}" y2="${yy}" stroke="#333" stroke-width="1.5"/>`);
  push(`<text x="${margin.left - 8}" y="${yy}" text-anchor="end" dominant-baseline="middle" fill="#333">${v}</text>`);
}

// X-axis ticks + labels
for (const v of tTicks) {
  const xx = sx(v).toFixed(1);
  push(`<line x1="${xx}" y1="${H - margin.bottom}" x2="${xx}" y2="${H - margin.bottom + 5}" stroke="#333" stroke-width="1.5"/>`);
  push(`<text x="${xx}" y="${H - margin.bottom + 18}" text-anchor="middle" fill="#333">${v}</text>`);
}

// Axis labels
push(`<text x="${margin.left + plotW / 2}" y="${H - 5}" text-anchor="middle" fill="#333" font-size="13">Year</text>`);
push(`<text x="${14}" y="${margin.top + plotH / 2}" text-anchor="middle" fill="#333" font-size="13" transform="rotate(-90,14,${margin.top + plotH / 2})">Annual flow</text>`);

// Title
push(`<text x="${margin.left + plotW / 2}" y="${16}" text-anchor="middle" fill="#333" font-size="14" font-weight="600">Nile river annual flow — Kalman smoother ± 2σ</text>`);

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

const outDir = resolve(root, "assets");
mkdirSync(outDir, { recursive: true });
const outPath = resolve(outDir, "niledemo.svg");
writeFileSync(outPath, lines.join("\n"), "utf8");
console.log(`Written: ${outPath}`);
