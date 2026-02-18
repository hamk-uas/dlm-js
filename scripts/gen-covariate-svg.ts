/**
 * Generate a covariate demo SVG.
 *
 * Dataset: synthetic "stratospheric ozone proxy" — a slow trend plus two
 * periodic proxy signals with known regression coefficients β.
 *
 *   y(t) = μ(t) + β₁·X₁(t) + β₂·X₂(t) + ε,   ε ~ N(0, s²)
 *   μ(t) = μ(t-1) + η,                            η ~ N(0, w²)
 *
 * X₁ = solar proxy (sine, T=11 yr ≈ 132 months)
 * X₂ = QBO proxy   (cosine, T=28 months)
 *
 * True parameters: β₁ = 3.0, β₂ = -2.0, s = 1.5, w = 0.08.
 *
 * The DLM smoother recovers:
 *  - the underlying trend μ(t)
 *  - the regression coefficients β₁, β₂ as static states
 *  - the covariate contributions β̂₁·X₁(t) and β̂₂·X₂(t)
 *
 * Usage:  npx tsx scripts/gen-covariate-svg.ts
 * Output: assets/covariate-demo.svg
 */

import { dlmFit } from "../src/index.ts";
import { DType } from "@hamk-uas/jax-js-nonconsuming";
import { resolve, dirname } from "node:path";
import {
  r, makeLinearScale, polylinePoints, bandPathD,
  renderGridLines, renderYAxis, renderXAxis, renderAxesBorder, writeSvg,
  yTicksFromRange,
} from "./lib/svg-helpers.ts";

// ── Deterministic PRNG ────────────────────────────────────────────────────
function mulberry32(seed: number): () => number {
  let a = seed | 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gaussianRng(uniform: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u1: number; do { u1 = uniform(); } while (u1 === 0);
    const u2 = uniform();
    const mag = Math.sqrt(-2 * Math.log(u1));
    spare = mag * Math.sin(2 * Math.PI * u2);
    return mag * Math.cos(2 * Math.PI * u2);
  };
}

// ── Synthetic data ────────────────────────────────────────────────────────
const N = 240;             // 20 years of monthly data
const TRUE_BETA = [3.0, -2.0];
const TRUE_S = 1.5;
const TRUE_W = 0.08;
const T_SOLAR = 132;       // ~11-year solar cycle in months
const T_QBO = 28;          // ~28-month QBO cycle

const randn = gaussianRng(mulberry32(7));

const t_months = Array.from({ length: N }, (_, i) => i + 1);

// Covariate matrix: X[t] = [solar_proxy(t), qbo_proxy(t)]
const X: number[][] = t_months.map(t => [
  Math.sin(2 * Math.PI * t / T_SOLAR),
  Math.cos(2 * Math.PI * t / T_QBO),
]);

// Slow random walk trend (μ)
const mu_true: number[] = new Array(N);
mu_true[0] = 300 + randn() * 2;
for (let i = 1; i < N; i++) mu_true[i] = mu_true[i - 1] + randn() * TRUE_W;

// Observations
const y: number[] = Array.from({ length: N }, (_, i) =>
  mu_true[i] + TRUE_BETA[0] * X[i][0] + TRUE_BETA[1] * X[i][1] + randn() * TRUE_S
);

// ── Fit DLM with covariates ───────────────────────────────────────────────
console.log("Fitting DLM with 2 covariates...");
const fit = await dlmFit(y, TRUE_S, [TRUE_W], DType.Float64, { order: 0 }, X);
// order=0: local level (m_base=1); x[0]=μ, x[1]=β₁, x[2]=β₂

const mu_hat    = Array.from(fit.x[0]);      // smoothed level
const beta1_hat = Array.from(fit.x[1]);      // smoothed β₁
const beta2_hat = Array.from(fit.x[2]);      // smoothed β₂
const mu_std    = fit.xstd.map(row => row[0]);  // std of level state

const beta1_final = beta1_hat[N - 1];
const beta2_final = beta2_hat[N - 1];
console.log(`  β₁ estimated: ${beta1_final.toFixed(3)}  (true: ${TRUE_BETA[0]})`);
console.log(`  β₂ estimated: ${beta2_final.toFixed(3)}  (true: ${TRUE_BETA[1]})`);

// Covariate contributions using estimated β (final smoothed values)
const contrib1: number[] = t_months.map((_, i) => beta1_final * X[i][0]);
const contrib2: number[] = t_months.map((_, i) => beta2_final * X[i][1]);
const total_contrib: number[] = contrib1.map((v, i) => v + contrib2[i]);

// True covariate contributions for comparison
const true_contrib1: number[] = t_months.map((_, i) => TRUE_BETA[0] * X[i][0]);
const true_contrib2: number[] = t_months.map((_, i) => TRUE_BETA[1] * X[i][1]);
const true_total: number[] = true_contrib1.map((v, i) => v + true_contrib2[i]);

// Level ± 2σ
const mu_upper = mu_hat.map((v, i) => v + 2 * mu_std[i]);
const mu_lower = mu_hat.map((v, i) => v - 2 * mu_std[i]);

// ── SVG layout ────────────────────────────────────────────────────────────
const root = resolve(dirname(new URL(import.meta.url).pathname), "..");

// Two panels stacked: [top] observations + level+band, [bottom] covariate contributions
const margin = { top: 28, right: 20, bottom: 48, left: 58 };
const panelGap = 36;
const W = 820;
const panelH = 220;
const H = margin.top + panelH + panelGap + panelH + margin.bottom;
const plotW = W - margin.left - margin.right;

// x scale shared across panels
const sx = makeLinearScale(t_months[0], t_months[N - 1], margin.left, margin.left + plotW);

// ── Panel 1: observations + smoothed level with band ─────────────────────
const allVals1 = [...y, ...mu_upper, ...mu_lower];
const y1Min = Math.floor((Math.min(...allVals1) - 1) / 5) * 5;
const y1Max = Math.ceil((Math.max(...allVals1) + 1) / 5) * 5;
const p1Top = margin.top;
const p1Bot = margin.top + panelH;
const sy1 = makeLinearScale(y1Min, y1Max, p1Bot, p1Top);
const yTicks1 = yTicksFromRange(y1Min, y1Max, 5);

// ── Panel 2: covariate contributions ─────────────────────────────────────
const allVals2 = [...contrib1, ...contrib2, ...total_contrib,
                  ...true_contrib1, ...true_contrib2, ...true_total];
const y2Max_abs = Math.ceil(Math.max(...allVals2.map(Math.abs)) * 1.2 / 0.5) * 0.5;
const y2Min = -y2Max_abs;
const y2Max = y2Max_abs;
const p2Top = margin.top + panelH + panelGap;
const p2Bot = margin.top + panelH + panelGap + panelH;
const sy2 = makeLinearScale(y2Min, y2Max, p2Bot, p2Top);
const yTicks2 = yTicksFromRange(y2Min, y2Max);

// x-axis ticks every 24 months, starting from 24 so no tick crowds origin
const xTicksRaw = Array.from({ length: Math.floor(N / 24) }, (_, i) => (i + 1) * 24)
  .filter(v => v <= N);
// Build { val, label } objects (renderXAxis expects this format)
const xTickObjs1 = xTicksRaw.map(v => ({ val: v, label: `Y${v / 12}` }));
const xTickObjs2 = xTicksRaw.map(v => ({ val: v, label: `Y${v / 12}` }));

// ── Render ────────────────────────────────────────────────────────────────
const svg: string[] = [
  `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif" font-size="12">`,
  `<rect width="${W}" height="${H}" fill="white"/>`,
];

// ── Panel 1 ──────────────────────────────────────────────────────────────
svg.push(...renderGridLines(yTicks1, sy1, margin.left, margin.left + plotW));
svg.push(`<path d="${bandPathD(t_months, mu_upper, mu_lower, sx, sy1)}" fill="#3b82f6" fill-opacity="0.15" stroke="none"/>`);
svg.push(`<polyline points="${polylinePoints(t_months, y, sx, sy1)}" fill="none" stroke="#9ca3af" stroke-width="1" opacity="0.7"/>`);
svg.push(`<polyline points="${polylinePoints(t_months, mu_hat, sx, sy1)}" fill="none" stroke="#2563eb" stroke-width="2"/>`);
svg.push(`<polyline points="${polylinePoints(t_months, mu_true, sx, sy1)}" fill="none" stroke="#16a34a" stroke-width="1.2" stroke-dasharray="4 3" opacity="0.6"/>`);

svg.push(...renderYAxis(yTicks1, sy1, margin.left, v => v.toFixed(0)));
svg.push(...renderXAxis(xTickObjs1, sx, p1Bot));
svg.push(...renderAxesBorder(margin.left, p1Top, margin.left + plotW, p1Bot));

// Y-axis title (rotated)
svg.push(`<text transform="rotate(-90)" x="${-(p1Top + panelH / 2)}" y="${margin.left - 42}" text-anchor="middle" fill="#374151" font-size="11">Ozone proxy</text>`);

// Legend
const leg1X = margin.left + plotW - 235;
const leg1Y = p1Top + 14;
const leg1 = [
  [`<line x1="${leg1X}" y1="${leg1Y}" x2="${leg1X + 24}" y2="${leg1Y}" stroke="#9ca3af" stroke-width="1"/>`, "Observations"],
  [`<line x1="${leg1X}" y1="${leg1Y + 16}" x2="${leg1X + 24}" y2="${leg1Y + 16}" stroke="#2563eb" stroke-width="2"/>`, "Smoothed level μ̂(t)"],
  [`<line x1="${leg1X}" y1="${leg1Y + 32}" x2="${leg1X + 24}" y2="${leg1Y + 32}" stroke="#16a34a" stroke-width="1.2" stroke-dasharray="4 3"/>`, "True level μ(t)"],
] as [string, string][];
leg1.forEach(([lineSvg, label], i) => {
  svg.push(lineSvg);
  svg.push(`<text x="${leg1X + 30}" y="${leg1Y + i * 16 + 4}" fill="#374151" font-size="11">${label}</text>`);
});
svg.push(`<text x="${margin.left + plotW / 2}" y="${p1Top - 8}" text-anchor="middle" fill="#374151" font-size="13" font-weight="bold">Synthetic ozone proxy: observations and smoothed trend</text>`);

// ── Panel 2 ──────────────────────────────────────────────────────────────
svg.push(...renderGridLines(yTicks2, sy2, margin.left, margin.left + plotW));
svg.push(`<line x1="${margin.left}" y1="${r(sy2(0))}" x2="${margin.left + plotW}" y2="${r(sy2(0))}" stroke="#6b7280" stroke-width="0.8" stroke-dasharray="2 2"/>`);

// True contributions (dashed, faint)
svg.push(`<polyline points="${polylinePoints(t_months, true_contrib1, sx, sy2)}" fill="none" stroke="#d97706" stroke-width="1" stroke-dasharray="4 3" opacity="0.65"/>`);
svg.push(`<polyline points="${polylinePoints(t_months, true_contrib2, sx, sy2)}" fill="none" stroke="#7c3aed" stroke-width="1" stroke-dasharray="4 3" opacity="0.65"/>`);
svg.push(`<polyline points="${polylinePoints(t_months, true_total, sx, sy2)}" fill="none" stroke="#6b7280" stroke-width="1.2" stroke-dasharray="4 3" opacity="0.65"/>`);

// Estimated contributions (solid)
svg.push(`<polyline points="${polylinePoints(t_months, contrib1, sx, sy2)}" fill="none" stroke="#f59e0b" stroke-width="1.8"/>`);
svg.push(`<polyline points="${polylinePoints(t_months, contrib2, sx, sy2)}" fill="none" stroke="#8b5cf6" stroke-width="1.8"/>`);
svg.push(`<polyline points="${polylinePoints(t_months, total_contrib, sx, sy2)}" fill="none" stroke="#374151" stroke-width="2.2" stroke-dasharray="5 3"/>`);

svg.push(...renderYAxis(yTicks2, sy2, margin.left));
svg.push(...renderXAxis(xTickObjs2, sx, p2Bot));
svg.push(...renderAxesBorder(margin.left, p2Top, margin.left + plotW, p2Bot));

// Y-axis title
svg.push(`<text transform="rotate(-90)" x="${-(p2Top + panelH / 2)}" y="${margin.left - 42}" text-anchor="middle" fill="#374151" font-size="11">Contribution</text>`);

// x-axis label under bottom panel
svg.push(`<text x="${margin.left + plotW / 2}" y="${p2Bot + 36}" text-anchor="middle" fill="#4b5563" font-size="11">Year (month 24 = Y2, 48 = Y4, …)</text>`);

// Panel 2 legend — 2-column: estimated (left) | true (right)
const leg2W = 360;
const leg2X = margin.left + plotW - leg2W;
const leg2Y = p2Top + 12;
const leg2Col0 = leg2X;
const leg2Col1 = leg2X + leg2W / 2;
const leg2Entries = [
  // col 0 — estimated
  { x: leg2Col0, y: leg2Y,      line: `<line x1="${leg2Col0}" y1="${leg2Y}" x2="${leg2Col0+22}" y2="${leg2Y}" stroke="#f59e0b" stroke-width="1.8"/>`,      label: `est. β̂₁·X₁ solar (β̂₁=${beta1_final.toFixed(2)})` },
  { x: leg2Col0, y: leg2Y + 15, line: `<line x1="${leg2Col0}" y1="${leg2Y+15}" x2="${leg2Col0+22}" y2="${leg2Y+15}" stroke="#8b5cf6" stroke-width="1.8"/>`,  label: `est. β̂₂·X₂ QBO (β̂₂=${beta2_final.toFixed(2)})` },
  { x: leg2Col0, y: leg2Y + 30, line: `<line x1="${leg2Col0}" y1="${leg2Y+30}" x2="${leg2Col0+22}" y2="${leg2Y+30}" stroke="#374151" stroke-width="2.2" stroke-dasharray="5 3"/>`, label: "est. total β̂·X" },
  // col 1 — true
  { x: leg2Col1, y: leg2Y,      line: `<line x1="${leg2Col1}" y1="${leg2Y}" x2="${leg2Col1+22}" y2="${leg2Y}" stroke="#d97706" stroke-width="1" stroke-dasharray="4 3" opacity="0.65"/>`,      label: `true β₁·X₁ (β₁=${TRUE_BETA[0]})` },
  { x: leg2Col1, y: leg2Y + 15, line: `<line x1="${leg2Col1}" y1="${leg2Y+15}" x2="${leg2Col1+22}" y2="${leg2Y+15}" stroke="#7c3aed" stroke-width="1" stroke-dasharray="4 3" opacity="0.65"/>`, label: `true β₂·X₂ (β₂=${TRUE_BETA[1]})` },
  { x: leg2Col1, y: leg2Y + 30, line: `<line x1="${leg2Col1}" y1="${leg2Y+30}" x2="${leg2Col1+22}" y2="${leg2Y+30}" stroke="#6b7280" stroke-width="1.2" stroke-dasharray="4 3" opacity="0.65"/>`, label: "true total β·X" },
];
for (const e of leg2Entries) {
  svg.push(e.line);
  svg.push(`<text x="${e.x + 27}" y="${e.y + 4}" fill="#374151" font-size="11">${e.label}</text>`);
}
svg.push(`<text x="${margin.left + plotW / 2}" y="${p2Top - 8}" text-anchor="middle" fill="#374151" font-size="13" font-weight="bold">Covariate contributions β·X(t): estimated (solid) vs true (dashed)</text>`);

svg.push("</svg>");

const outPath = resolve(root, "assets/covariate-demo.svg");
writeSvg(svg, outPath);
