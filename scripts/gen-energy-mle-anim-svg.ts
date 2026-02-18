/**
 * Generate an animated MLE + AR fitting SVG from pre-collected energy frames.
 *
 * Reads:   tmp/energy-mle-frames.json   (produced by scripts/collect-energy-mle-frames.ts)
 * Writes:  assets/energy-mle-anim.svg
 *
 * Shows the combined signal F·x converging as s, w and arphi are jointly
 * optimized. Includes a −2·logL sparkline and an arphi convergence trace.
 *
 * Usage:  npx tsx scripts/gen-energy-mle-anim-svg.ts
 */

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const data = JSON.parse(readFileSync(resolve(root, "tmp/energy-mle-frames.json"), "utf8"));

const {
  t, y, n,
  elapsed: elapsedMs,
  iterations,
  holdSeconds,
  likHistory,
  arphiHistory,
  frames,
} = data as {
  t: number[];
  y: number[];
  n: number;
  elapsed: number;
  iterations: number;
  holdSeconds: number;
  likHistory: number[];
  arphiHistory: number[];
  frames: {
    iter: number;
    s: number;
    w: number[];
    arphi: number[];
    lik: number | null;
    combined: number[];
    combinedStd: number[];
  }[];
};

// ── Layout ─────────────────────────────────────────────────────────────────

const margin = { top: 30, right: 20, bottom: 50, left: 65 };
const W = 800;
const H = 380;
const plotW = W - margin.left - margin.right;
const plotH = H - margin.top - margin.bottom;

// ── Scales ─────────────────────────────────────────────────────────────────

const tMin = t[0];
const tMax = t[n - 1];

// Compute y-range across ALL frames
const allVals: number[] = [...y];
for (const f of frames) {
  for (let i = 0; i < n; i++) {
    allVals.push(f.combined[i] + 2 * f.combinedStd[i]);
    allVals.push(f.combined[i] - 2 * f.combinedStd[i]);
  }
}
const yMin = Math.floor(Math.min(...allVals) / 10) * 10;
const yMax = Math.ceil(Math.max(...allVals) / 10) * 10;

function sx(val: number): number {
  return margin.left + ((val - tMin) / (tMax - tMin)) * plotW;
}
function sy(val: number): number {
  return margin.top + ((yMax - val) / (yMax - yMin)) * plotH;
}

function r(v: number): string {
  return v.toFixed(1);
}

// ── Animation timing ───────────────────────────────────────────────────────

const animDuration = elapsedMs / 1000;
const totalDuration = animDuration + holdSeconds;
const numFrames = frames.length;

const keyTimes: number[] = [];
for (let i = 0; i < numFrames; i++) {
  keyTimes.push((i / (numFrames - 1)) * animDuration / totalDuration);
}
keyTimes.push(1.0);

const keyTimesStr = keyTimes.map(kt => kt.toFixed(4)).join(";");

// ── Pre-compute polyline points and band paths per frame ───────────────────

function makePolyline(vals: number[]): string {
  return t.map((tv, i) => `${r(sx(tv))},${r(sy(vals[i]))}`).join(" ");
}

function makeBandPath(vals: number[], stds: number[]): string {
  const fwd = t.map((tv, i) => `${r(sx(tv))},${r(sy(vals[i] + 2 * stds[i]))}`);
  const bwd = t
    .map((tv, i) => `${r(sx(tv))},${r(sy(vals[i] - 2 * stds[i]))}`)
    .reverse();
  return `M${fwd.join("L")}L${bwd.join("L")}Z`;
}

const polylineValues: string[] = frames.map(f => makePolyline(f.combined));
polylineValues.push(polylineValues[polylineValues.length - 1]);

const bandValues: string[] = frames.map(f => makeBandPath(f.combined, f.combinedStd));
bandValues.push(bandValues[bandValues.length - 1]);

// ── Legend layout ──────────────────────────────────────────────────────────

const legW = 400;
const legH = 72;
const legX = W - margin.right - legW - 5;
const legY = margin.top + 8;

// Right half: two mini sparklines stacked
const sparkW = 140;
const sparkH1 = 20; // −2·logL sparkline
const sparkH2 = 20; // arphi sparkline
const sparkGap = 6;
const sparkMarginRight = 10;
const sparkMarginTop = 10;

const sparkX = legX + legW - sparkMarginRight - sparkW;
const sparkY1 = legY + sparkMarginTop;
const sparkY2 = sparkY1 + sparkH1 + sparkGap;

// −2·logL sparkline
const likMin = Math.min(...likHistory);
const likMax = Math.max(...likHistory);
const likRange = likMax - likMin || 1;

function sparkPt(i: number, val: number, sy_: number, sh: number, vmin: number, vrange: number): string {
  const px = sparkX + (i / (likHistory.length - 1)) * sparkW;
  const py = sy_ + sh - ((val - vmin) / vrange) * sh;
  return `${r(px)},${r(py)}`;
}

const sparkLikPoints = likHistory.map((v, i) => sparkPt(i, v, sparkY1, sparkH1, likMin, likRange)).join(" ");

// arphi sparkline
const arphiMin = Math.min(...arphiHistory) * 0.95;
const arphiMax = Math.max(...arphiHistory) * 1.05;
const arphiRange = arphiMax - arphiMin || 1;

const sparkArphiPoints = arphiHistory.map((v, i) => sparkPt(i, v, sparkY2, sparkH2, arphiMin, arphiRange)).join(" ");

// ── Ticks ──────────────────────────────────────────────────────────────────

// Y ticks: every 20 units
const yTicks: number[] = [];
for (let v = yMin; v <= yMax; v += 20) yTicks.push(v);

// X ticks: every 12 months
const tTicks: number[] = [];
for (let v = 12; v <= tMax; v += 12) tTicks.push(v);

// ── Colors ─────────────────────────────────────────────────────────────────

const obsColor = "#555";
const lineColor = "#2563eb";
const bandColor = "rgba(37,99,235,0.15)";
const sparkLikColor = "#f59e0b";
const sparkArColor = "#10b981"; // green for arphi

// ── Final frame values ─────────────────────────────────────────────────────

const finalFrame = frames[frames.length - 1];
const finalLik = finalFrame.lik!;
const finalArphi = finalFrame.arphi[0];

// ── Build SVG ──────────────────────────────────────────────────────────────

const lines: string[] = [];
const push = (s: string) => lines.push(s);

push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="system-ui,-apple-system,sans-serif" font-size="12">`);
push(`<rect width="${W}" height="${H}" fill="white"/>`);

// Clip paths
push(`<defs>`);
push(`  <clipPath id="plot-clip"><rect x="${margin.left}" y="${margin.top}" width="${plotW}" height="${plotH}"/></clipPath>`);
push(`  <clipPath id="spark-clip">`);
push(`    <rect x="${sparkX}" y="${sparkY1 - 2}" width="0" height="${sparkH1 + sparkGap + sparkH2 + 4}">`);
push(`      <animate attributeName="width" values="0;${sparkW};${sparkW}" keyTimes="0;${(animDuration / totalDuration).toFixed(4)};1" dur="${r(totalDuration)}s" repeatCount="indefinite"/>`);
push(`    </rect>`);
push(`  </clipPath>`);
push(`</defs>`);

// Grid lines
for (const v of yTicks) {
  push(`<line x1="${margin.left}" y1="${r(sy(v))}" x2="${W - margin.right}" y2="${r(sy(v))}" stroke="#e5e7eb" stroke-width="1"/>`);
}

// ── Animated elements ──────────────────────────────────────────────────────

push(`<g clip-path="url(#plot-clip)">`);

// Confidence band
push(`<path fill="${bandColor}" stroke="none">`);
push(`  <animate attributeName="d" values="${bandValues.join(";")}" keyTimes="${keyTimesStr}" dur="${r(totalDuration)}s" repeatCount="indefinite" calcMode="discrete"/>`);
push(`</path>`);

// Combined signal line
push(`<polyline fill="none" stroke="${lineColor}" stroke-width="2">`);
push(`  <animate attributeName="points" values="${polylineValues.join(";")}" keyTimes="${keyTimesStr}" dur="${r(totalDuration)}s" repeatCount="indefinite" calcMode="discrete"/>`);
push(`</polyline>`);

push(`</g>`);

// Observations (static)
for (let i = 0; i < n; i++) {
  push(`<circle cx="${r(sx(t[i]))}" cy="${r(sy(y[i]))}" r="2" fill="${obsColor}" opacity="0.5"/>`);
}

// ── Axes ───────────────────────────────────────────────────────────────────

push(`<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${H - margin.bottom}" stroke="#333" stroke-width="1.5"/>`);
push(`<line x1="${margin.left}" y1="${H - margin.bottom}" x2="${W - margin.right}" y2="${H - margin.bottom}" stroke="#333" stroke-width="1.5"/>`);

for (const v of yTicks) {
  const yy = r(sy(v));
  push(`<line x1="${margin.left - 5}" y1="${yy}" x2="${margin.left}" y2="${yy}" stroke="#333" stroke-width="1.5"/>`);
  push(`<text x="${margin.left - 8}" y="${yy}" text-anchor="end" dominant-baseline="middle" fill="#333">${v}</text>`);
}
for (const v of tTicks) {
  const xx = r(sx(v));
  push(`<line x1="${xx}" y1="${H - margin.bottom}" x2="${xx}" y2="${H - margin.bottom + 5}" stroke="#333" stroke-width="1.5"/>`);
  push(`<text x="${xx}" y="${H - margin.bottom + 18}" text-anchor="middle" fill="#333">${v === 12 ? "1y" : v === 24 ? "2y" : v === 36 ? "3y" : v === 48 ? "4y" : v === 60 ? "5y" : v === 72 ? "6y" : v === 84 ? "7y" : v === 96 ? "8y" : v === 108 ? "9y" : v === 120 ? "10y" : String(v)}</text>`);
}

push(`<text x="${margin.left + plotW / 2}" y="${H - 5}" text-anchor="middle" fill="#333" font-size="13">Month</text>`);
push(`<text x="14" y="${margin.top + plotH / 2}" text-anchor="middle" fill="#333" font-size="13" transform="rotate(-90,14,${margin.top + plotH / 2})">Energy demand</text>`);

// ── Title ──────────────────────────────────────────────────────────────────

push(`<text x="${margin.left + plotW / 2}" y="16" text-anchor="middle" fill="#333" font-size="14" font-weight="600">Energy demand — MLE + AR fitting (${iterations} iters, ${(elapsedMs / 1000).toFixed(1)} s WASM)</text>`);

// ── Legend ──────────────────────────────────────────────────────────────────

push(`<rect x="${legX}" y="${legY}" width="${legW}" height="${legH}" rx="4" fill="white" stroke="#e5e7eb" stroke-width="1"/>`);

// Observations
push(`<circle cx="${legX + 14}" cy="${legY + 14}" r="2.5" fill="${obsColor}" opacity="0.5"/>`);
push(`<text x="${legX + 24}" y="${legY + 14}" dominant-baseline="middle" fill="#333" font-size="11">Observations</text>`);

// MLE fit line
push(`<line x1="${legX + 8}" y1="${legY + 30}" x2="${legX + 20}" y2="${legY + 30}" stroke="${lineColor}" stroke-width="2"/>`);
push(`<rect x="${legX + 8}" y="${legY + 25}" width="12" height="10" fill="${bandColor}" stroke="none"/>`);
push(`<text x="${legX + 24}" y="${legY + 30}" dominant-baseline="middle" fill="#333" font-size="11">MLE: s=${finalFrame.s.toFixed(1)}, \u03c6=${finalArphi.toFixed(2)}</text>`);

// Lik + arphi values
push(`<text x="${legX + 24}" y="${legY + 46}" dominant-baseline="middle" fill="#666" font-size="10">−2·logL = ${finalLik.toFixed(1)}</text>`);

// Model description
push(`<text x="${legX + 24}" y="${legY + 60}" dominant-baseline="middle" fill="#999" font-size="9">order=1, trig=1, ns=12, fitar=true</text>`);

// ── Convergence miniplots (right half) ─────────────────────────────────────

// −2·logL sparkline
push(`<text x="${sparkX}" y="${sparkY1 - 2}" fill="#666" font-size="8">−2·logL</text>`);
push(`<g clip-path="url(#spark-clip)">`);
push(`  <polyline points="${sparkLikPoints}" fill="none" stroke="${sparkLikColor}" stroke-width="1.5" stroke-linejoin="round"/>`);
push(`</g>`);
push(`<line x1="${sparkX}" y1="${sparkY1 + sparkH1}" x2="${sparkX + sparkW}" y2="${sparkY1 + sparkH1}" stroke="#eee" stroke-width="0.5"/>`);
// y labels
push(`<text x="${sparkX - 2}" y="${sparkY1 + 3}" text-anchor="end" fill="#999" font-size="7">${likMax.toFixed(0)}</text>`);
push(`<text x="${sparkX - 2}" y="${sparkY1 + sparkH1}" text-anchor="end" fill="#999" font-size="7">${likMin.toFixed(0)}</text>`);

// arphi sparkline
push(`<text x="${sparkX}" y="${sparkY2 - 2}" fill="#666" font-size="8">\u03c6 (AR)</text>`);
push(`<g clip-path="url(#spark-clip)">`);
push(`  <polyline points="${sparkArphiPoints}" fill="none" stroke="${sparkArColor}" stroke-width="1.5" stroke-linejoin="round"/>`);
push(`</g>`);
push(`<line x1="${sparkX}" y1="${sparkY2 + sparkH2}" x2="${sparkX + sparkW}" y2="${sparkY2 + sparkH2}" stroke="#eee" stroke-width="0.5"/>`);
// y labels
push(`<text x="${sparkX - 2}" y="${sparkY2 + 3}" text-anchor="end" fill="#999" font-size="7">${arphiMax.toFixed(2)}</text>`);
push(`<text x="${sparkX - 2}" y="${sparkY2 + sparkH2}" text-anchor="end" fill="#999" font-size="7">${arphiMin.toFixed(2)}</text>`);

// Shared x-axis labels
const sparkAxisY = sparkY2 + sparkH2 + 8;
push(`<text x="${sparkX}" y="${sparkAxisY}" text-anchor="middle" fill="#999" font-size="7">0</text>`);
push(`<text x="${sparkX + sparkW / 2}" y="${sparkAxisY}" text-anchor="middle" fill="#999" font-size="7">iters</text>`);
push(`<text x="${sparkX + sparkW}" y="${sparkAxisY}" text-anchor="middle" fill="#999" font-size="7">${iterations}</text>`);

push(`</svg>`);

// ── Write output ───────────────────────────────────────────────────────────

const outDir = resolve(root, "assets");
mkdirSync(outDir, { recursive: true });
const outPath = resolve(outDir, "energy-mle-anim.svg");
writeFileSync(outPath, lines.join("\n"), "utf8");

const sizeKb = (Buffer.byteLength(lines.join("\n")) / 1024).toFixed(0);
console.log(`Written: ${outPath} (${sizeKb} KB)`);
console.log(`  ${numFrames} frames, ${r(totalDuration)}s cycle (${r(animDuration)}s play + ${holdSeconds}s hold)`);
