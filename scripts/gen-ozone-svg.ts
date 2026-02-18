/**
 * Stratospheric ozone trend analysis — replication of the MATLAB DLM demo.
 *
 * Reference: Laine, Latva-Pukkila & Kyrölä (2014), "Analyzing time-varying trends
 * in stratospheric ozone time series using state space approach",
 * Atmospheric Chemistry and Physics 14(18), doi:10.5194/acp-14-9707-2014.
 *
 * Data: ozonedata.dat (from https://github.com/mjlaine/dlm/tree/master/examples)
 *   col 1  decimal year (1984–2011)
 *   col 2  ozone density [1/cm³], NaN where missing
 *   col 3  uncertainty σ [1/cm³], NaN where missing
 *   col 4  solar proxy  (normalised F10.7-like index)
 *   col 5  QBO component 1
 *   col 6  QBO component 2
 *
 * Model (matches MATLAB options = struct('trig',2,'order',1) + 3 proxy covariates):
 *   y_t = μ_t + γ_t + β₁·solar + β₂·qbo1 + β₃·qbo2 + ε
 *   μ_t = μ_{t-1} + α_{t-1} + η_μ,   α_t = α_{t-1} + η_α   (local linear trend)
 *   γ_t = 2 trig harmonics for monthly seasonality (ns=12)
 *
 * Usage:  npx tsx scripts/gen-ozone-svg.ts
 * Output: assets/ozone-demo.svg
 */

import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { dlmFit } from "../src/index.ts";
import { DType } from "@hamk-uas/jax-js-nonconsuming";
import {
  r, makeLinearScale, polylinePoints, bandPathD,
  renderGridLines, renderYAxis, renderXAxis, renderAxesBorder, writeSvg,
  yTicksFromRange,
} from "./lib/svg-helpers.ts";

// ── Load data ─────────────────────────────────────────────────────────────
const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const raw = readFileSync(resolve(root, "tests/ozonedata.dat"), "utf8");

const rows = raw.trim().split("\n").map(line =>
  line.trim().split(/\s+/).map(v => (v === "NaN" ? NaN : parseFloat(v)))
);

const time   = rows.map(r => r[0]);  // decimal year
const y_raw  = rows.map(r => r[1]);  // ozone [1/cm³]
const s_raw  = rows.map(r => r[2]);  // uncertainty σ
const solar  = rows.map(r => r[3]);  // solar proxy
const qbo1   = rows.map(r => r[4]);  // QBO component 1
const qbo2   = rows.map(r => r[5]);  // QBO component 2
const N      = time.length;

// Scale y for numerical stability (MATLAB: yy = y./ys, ss = s./ys)
const y_valid = y_raw.filter(v => isFinite(v));
const ys = Math.sqrt(y_valid.reduce((s, v) => s + v * v, 0) / y_valid.length -
           Math.pow(y_valid.reduce((s, v) => s + v, 0) / y_valid.length, 2));
const yy = y_raw.map(v => v / ys);
const ss = s_raw.map(v => v / ys);

// Covariates matrix: X[t] = [solar(t), qbo1(t), qbo2(t)] — proxies always observed
const X: number[][] = time.map((_, i) => [solar[i], qbo1[i], qbo2[i]]);

// ── Fit DLM ───────────────────────────────────────────────────────────────
// Match MATLAB: options = struct('trig',2,'order',1)
// order=1 → local linear trend (2 states: level + slope)
// trig=2  → 2 harmonics for ns=12 monthly seasons (4 states: 2×[cos,sin])
// 3 proxy covariates → 3 static β states
// Total state dimension: 2 + 4 + 3 = 9

// wtrend = |ym|*0.00005, wseas = |ym|*0.015  (MATLAB ozonedemo defaults)
const ym = yy.filter(v => isFinite(v)).reduce((a, v) => a + v, 0) /
           yy.filter(v => isFinite(v)).length;
const wtrend = Math.abs(ym) * 0.00005;
const wseas  = Math.abs(ym) * 0.015;

// w = [level_std=0, trend_std, seas1_std, seas1*_std, seas2_std, seas2*_std]
// (no evolution noise on the level itself, trend and seasonal states have noise)
const w = [0, wtrend, wseas, wseas, wseas, wseas];

// Per-observation standard deviations — use ss[t], fall back to mean ss for NaN obs
const ss_mean = ss.filter(v => isFinite(v)).reduce((a, v) => a + v, 0) /
                ss.filter(v => isFinite(v)).length;
const s_vec = ss.map(v => (isFinite(v) ? v : ss_mean));

// Replace NaN observations with 0 for the solver — they are masked by large V
// dlm-js handles per-time V via the s parameter array
const y_filled = yy.map(v => (isFinite(v) ? v : 0));
const s_filled = yy.map((v, i) => (isFinite(v) ? s_vec[i] : 1e6));  // huge σ for missing

console.log(`Fitting DLM: N=${N} months (${time[0].toFixed(2)}–${time[N-1].toFixed(2)})`);
console.log(`  ym=${ym.toFixed(4)}, wtrend=${wtrend.toExponential(2)}, wseas=${wseas.toFixed(4)}`);

const fit = await dlmFit(
  y_filled, s_filled, w, DType.Float64,
  { order: 1, trig: 2 },
  X,
);

// State layout (order=1, trig=2, q=3):
//  x[0] = μ (level)
//  x[1] = α (trend slope)
//  x[2], x[3] = harmonic 1 (annual, cos/sin pair)
//  x[4], x[5] = harmonic 2 (semi-annual)
//  x[6] = β_solar
//  x[7] = β_qbo1
//  x[8] = β_qbo2

const mu_hat   = Array.from(fit.x[0]).map(v => v * ys);     // level [1/cm³]
const mu_std   = fit.xstd.map(row => row[0] * ys);
const beta_solar = Array.from(fit.x[6]);
const beta_qbo1  = Array.from(fit.x[7]);
const beta_qbo2  = Array.from(fit.x[8]);

const beta_solar_f = beta_solar[N - 1];
const beta_qbo1_f  = beta_qbo1[N - 1];
const beta_qbo2_f  = beta_qbo2[N - 1];
console.log(`  β_solar=${(beta_solar_f * ys).toExponential(3)}`);
console.log(`  β_qbo1 =${(beta_qbo1_f  * ys).toExponential(3)}`);
console.log(`  β_qbo2 =${(beta_qbo2_f  * ys).toExponential(3)}`);

// Back-scale level and band
const mu_upper = mu_hat.map((v, i) => v + 2 * mu_std[i]);
const mu_lower = mu_hat.map((v, i) => v - 2 * mu_std[i]);

// Proxy contributions [1/cm³]
const contrib_solar: number[] = time.map((_, i) => beta_solar[i] * solar[i] * ys);
const contrib_qbo1:  number[] = time.map((_, i) => beta_qbo1[i]  * qbo1[i]  * ys);
const contrib_qbo2:  number[] = time.map((_, i) => beta_qbo2[i]  * qbo2[i]  * ys);
const contrib_qbo:   number[] = contrib_qbo1.map((v, i) => v + contrib_qbo2[i]);

// Observations (back-scaled, NaN where missing)
const y_sc = yy.map((v, i) => (isFinite(v) ? v * ys : NaN));

// ── SVG layout ────────────────────────────────────────────────────────────
const margin = { top: 30, right: 22, bottom: 50, left: 68 };
const panelGap = 38;
const W = 860;
const panelH = 230;
const H = margin.top + panelH + panelGap + panelH + margin.bottom;
const plotW = W - margin.left - margin.right;

const sx = makeLinearScale(time[0], time[N - 1], margin.left, margin.left + plotW);

// x-axis ticks: every 2 years, show year as integer
const xTickYears: number[] = [];
for (let yr = Math.ceil(time[0]); yr <= Math.floor(time[N - 1]); yr++) {
  if ((yr - 1984) % 2 === 0) xTickYears.push(yr);
}
// Convert integer years to decimal (approx mid-year) but tick at Jan of that year
const xTickObjs = xTickYears.map(yr => ({ val: yr, label: `${yr}` }));

// ── Panel 1: observations + smoothed level ────────────────────────────────
const p1Top = margin.top;
const p1Bot = margin.top + panelH;

// Build finite observation lists for polyline rendering (skip NaN spans)
function finiteSegments(xs: number[], ys_: number[]): string {
  const pts: string[] = [];
  for (let i = 0; i < xs.length; i++) {
    if (isFinite(ys_[i])) pts.push(`${r(sx(xs[i]))},${r(sy1(ys_[i]))}`);
  }
  return pts.join(" ");
}

// Y range from observations + band
const allP1 = [...y_sc.filter(isFinite), ...mu_upper, ...mu_lower];
const y1Min = Math.floor(Math.min(...allP1) / 2e9) * 2e9;
const y1Max = Math.ceil(Math.max(...allP1)  / 2e9) * 2e9;
const sy1 = makeLinearScale(y1Min, y1Max, p1Bot, p1Top);

// Y ticks in units of 1e10
const y1Step = (y1Max - y1Min) > 2e10 ? 1e10 : 5e9;
const yTicks1Raw: number[] = [];
for (let v = Math.ceil((y1Min) / y1Step) * y1Step; v <= y1Max; v += y1Step) {
  yTicks1Raw.push(v);
}

// ── Panel 2: proxy contributions ─────────────────────────────────────────
const p2Top = margin.top + panelH + panelGap;
const p2Bot = margin.top + panelH + panelGap + panelH;

const allP2 = [...contrib_solar, ...contrib_qbo];
const p2Abs = Math.ceil(Math.max(...allP2.map(Math.abs)) / 1e9) * 1e9;
const y2Min = -p2Abs, y2Max = p2Abs;
const sy2 = makeLinearScale(y2Min, y2Max, p2Bot, p2Top);
const y2Step = p2Abs > 4e9 ? 2e9 : 1e9;
const yTicks2Raw: number[] = [];
for (let v = Math.ceil(y2Min / y2Step) * y2Step; v <= y2Max; v += y2Step) {
  yTicks2Raw.push(v);
}

// Format ticks as ×10¹⁰
const fmt1 = (v: number): string => (v / 1e10).toFixed(1);
const fmt2 = (v: number): string => (v / 1e9).toFixed(0);

// ── Build SVG ─────────────────────────────────────────────────────────────
const svg: string[] = [
  `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif" font-size="12">`,
  `<rect width="${W}" height="${H}" fill="white"/>`,
];

// ── Panel 1 ──────────────────────────────────────────────────────────────
svg.push(...renderGridLines(yTicks1Raw, sy1, margin.left, margin.left + plotW));

// ±2σ band
svg.push(`<path d="${bandPathD(time, mu_upper, mu_lower, sx, sy1)}" fill="#3b82f6" fill-opacity="0.15" stroke="none"/>`);

// Observations as dots for sparse/gappy data, plus connecting line
{
  const pts: string[] = [];
  for (let i = 0; i < N; i++) {
    if (isFinite(y_sc[i])) pts.push(`${r(sx(time[i]))},${r(sy1(y_sc[i]))}`);
  }
  svg.push(`<polyline points="${pts.join(" ")}" fill="none" stroke="#9ca3af" stroke-width="1" opacity="0.6"/>`);
}

// Smoothed level
svg.push(`<polyline points="${polylinePoints(time, mu_hat, sx, sy1)}" fill="none" stroke="#2563eb" stroke-width="2.2"/>`);

// Axes
svg.push(...renderYAxis(yTicks1Raw, sy1, margin.left, fmt1));
svg.push(...renderXAxis(xTickObjs, sx, p1Bot));
svg.push(...renderAxesBorder(margin.left, p1Top, margin.left + plotW, p1Bot));

// Y-axis title (×10¹⁰ cm⁻³)
svg.push(`<text transform="rotate(-90)" x="${-(p1Top + panelH / 2)}" y="${margin.left - 52}" text-anchor="middle" fill="#374151" font-size="11">O₃ density [×10¹⁰ cm⁻³]</text>`);

// Legend
const l1x = margin.left + plotW - 255, l1y = p1Top + 14;
svg.push(`<line x1="${l1x}" y1="${l1y}" x2="${l1x+22}" y2="${l1y}" stroke="#9ca3af" stroke-width="1"/>`);
svg.push(`<text x="${l1x+27}" y="${l1y+4}" fill="#374151" font-size="11">Observations (SAGE II / GOMOS)</text>`);
svg.push(`<line x1="${l1x}" y1="${l1y+15}" x2="${l1x+22}" y2="${l1y+15}" stroke="#2563eb" stroke-width="2.2"/>`);
svg.push(`<text x="${l1x+27}" y="${l1y+19}" fill="#374151" font-size="11">Smoothed level ±2σ</text>`);

// Title
svg.push(`<text x="${margin.left + plotW / 2}" y="${p1Top - 10}" text-anchor="middle" fill="#374151" font-size="13" font-weight="bold">Stratospheric ozone (45–55 km, 40°N–50°N): trend extraction</text>`);

// ── Panel 2 ──────────────────────────────────────────────────────────────
svg.push(...renderGridLines(yTicks2Raw, sy2, margin.left, margin.left + plotW));
svg.push(`<line x1="${margin.left}" y1="${r(sy2(0))}" x2="${margin.left+plotW}" y2="${r(sy2(0))}" stroke="#6b7280" stroke-width="0.8" stroke-dasharray="2 2"/>`);

// Solar contribution
svg.push(`<polyline points="${polylinePoints(time, contrib_solar, sx, sy2)}" fill="none" stroke="#f59e0b" stroke-width="1.8"/>`);
// QBO total (qbo1+qbo2)
svg.push(`<polyline points="${polylinePoints(time, contrib_qbo, sx, sy2)}" fill="none" stroke="#8b5cf6" stroke-width="1.8"/>`);

// Axes
svg.push(...renderYAxis(yTicks2Raw, sy2, margin.left, fmt2));
svg.push(...renderXAxis(xTickObjs, sx, p2Bot));
svg.push(...renderAxesBorder(margin.left, p2Top, margin.left + plotW, p2Bot));

// Y-axis title
svg.push(`<text transform="rotate(-90)" x="${-(p2Top + panelH / 2)}" y="${margin.left - 52}" text-anchor="middle" fill="#374151" font-size="11">Contribution [×10⁹ cm⁻³]</text>`);

// x-axis label
svg.push(`<text x="${margin.left + plotW / 2}" y="${p2Bot + 38}" text-anchor="middle" fill="#4b5563" font-size="11">Year</text>`);

// Legend
const l2x = margin.left + 10, l2y = p2Top + 14;
svg.push(`<line x1="${l2x}" y1="${l2y}" x2="${l2x+22}" y2="${l2y}" stroke="#f59e0b" stroke-width="1.8"/>`);
svg.push(`<text x="${l2x+27}" y="${l2y+4}" fill="#374151" font-size="11">Solar proxy contribution β̂_solar · X_solar</text>`);
svg.push(`<line x1="${l2x}" y1="${l2y+15}" x2="${l2x+22}" y2="${l2y+15}" stroke="#8b5cf6" stroke-width="1.8"/>`);
svg.push(`<text x="${l2x+27}" y="${l2y+19}" fill="#374151" font-size="11">QBO contribution β̂_qbo1·X_qbo1 + β̂_qbo2·X_qbo2</text>`);

// Panel 2 title
svg.push(`<text x="${margin.left + plotW / 2}" y="${p2Top - 10}" text-anchor="middle" fill="#374151" font-size="13" font-weight="bold">Proxy covariate contributions (solar + QBO)</text>`);

svg.push("</svg>");

const outPath = resolve(root, "assets/ozone-demo.svg");
writeSvg(svg, outPath);
