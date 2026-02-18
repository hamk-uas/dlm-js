/**
 * Generate an animated MLE optimization SVG from pre-collected frame data.
 *
 * Reads:   tmp/mle-frames.json   (produced by tmp/collect-mle-frames.ts)
 * Writes:  assets/nile-mle-anim.svg
 *
 * Animation: SMIL <animate> on the level polyline + confidence band path.
 * Timing derived from the actual measured runtime at 10 fps, then holds 2 s.
 * Text is static (final converged values). A loss sparkline reveals
 * progressively as a graphical convergence indicator.
 *
 * Usage:  npx tsx scripts/gen-nile-mle-anim-svg.ts
 */

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const data = JSON.parse(readFileSync(resolve(root, "tmp/mle-frames.json"), "utf8"));

const {
  t, y, n,
  elapsed: elapsedMs,
  iterations,
  holdSeconds,
  likHistory,
  frames,
} = data as {
  t: number[];
  y: number[];
  n: number;
  elapsed: number;
  iterations: number;
  holdSeconds: number;
  likHistory: number[];
  frames: {
    iter: number;
    s: number;
    w: number[];
    lik: number | null;
    level: number[];
    std: number[];
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

// Compute y-range across ALL frames (not just final)
const allVals: number[] = [...y];
for (const f of frames) {
  for (let i = 0; i < n; i++) {
    allVals.push(f.level[i] + 2 * f.std[i]);
    allVals.push(f.level[i] - 2 * f.std[i]);
  }
}
const yMin = Math.floor(Math.min(...allVals) / 50) * 50;
const yMax = Math.ceil(Math.max(...allVals) / 50) * 50;

function sx(val: number): number {
  return margin.left + ((val - tMin) / (tMax - tMin)) * plotW;
}
function sy(val: number): number {
  return margin.top + ((yMax - val) / (yMax - yMin)) * plotH;
}

// Round to 1 decimal for compact SVG
function r(v: number): string {
  return v.toFixed(1);
}

// ── Animation timing ───────────────────────────────────────────────────────

const animDuration = elapsedMs / 1000;
const totalDuration = animDuration + holdSeconds;
const numFrames = frames.length;

// keyTimes: frames spread across animDuration, then hold at 1.0
// numFrames values + 1 duplicate for hold = numFrames+1 entries
const keyTimes: number[] = [];
for (let i = 0; i < numFrames; i++) {
  keyTimes.push((i / (numFrames - 1)) * animDuration / totalDuration);
}
keyTimes.push(1.0);

const keyTimesStr = keyTimes.map(t => t.toFixed(4)).join(";");

// ── Pre-compute polyline points and band paths per frame ───────────────────

function makePolyline(level: number[]): string {
  return t.map((tv, i) => `${r(sx(tv))},${r(sy(level[i]))}`).join(" ");
}

function makeBandPath(level: number[], std: number[]): string {
  const fwd = t.map((tv, i) => `${r(sx(tv))},${r(sy(level[i] + 2 * std[i]))}`);
  const bwd = t
    .map((tv, i) => `${r(sx(tv))},${r(sy(level[i] - 2 * std[i]))}`)
    .reverse();
  return `M${fwd.join("L")}L${bwd.join("L")}Z`;
}

// 41 frame values + 1 duplicate of last for hold
const polylineValues: string[] = frames.map(f => makePolyline(f.level));
polylineValues.push(polylineValues[polylineValues.length - 1]);

const bandValues: string[] = frames.map(f => makeBandPath(f.level, f.std));
bandValues.push(bandValues[bandValues.length - 1]);

// ── Loss sparkline (drawn in legend area) ──────────────────────────────────

const legW = 356;
const legH = 62;

// Sparkline occupies right half of legend
const sparkW = 155;
const sparkH = 30;
const sparkMarginRight = 10;
const sparkMarginTop = 12;
const legX = W - margin.right - legW - 5;
const legY = margin.top + 8;

const sparkX = legX + legW - sparkMarginRight - sparkW;
const sparkY = legY + sparkMarginTop;

const likMin = Math.min(...likHistory);
const likMax = Math.max(...likHistory);
const likRange = likMax - likMin || 1;

function sparkPt(i: number, lik: number): string {
  const px = sparkX + (i / (likHistory.length - 1)) * sparkW;
  const py = sparkY + sparkH - ((lik - likMin) / likRange) * sparkH;
  return `${r(px)},${r(py)}`;
}
const sparkPoints = likHistory.map((lik, i) => sparkPt(i, lik)).join(" ");

// ── Ticks ──────────────────────────────────────────────────────────────────

const yTicks: number[] = [];
for (let v = yMin; v <= yMax; v += 200) yTicks.push(v);
const tTicks: number[] = [];
for (let v = Math.ceil(tMin / 20) * 20; v <= tMax; v += 20) tTicks.push(v);

// ── Colors ─────────────────────────────────────────────────────────────────

const obsColor = "#555";
const lineColor = "#2563eb";
const bandColor = "rgba(37,99,235,0.15)";
const sparkColor = "#f59e0b";

// ── Final frame values for static text ─────────────────────────────────────

const finalFrame = frames[frames.length - 1];
const finalLik = finalFrame.lik!;

// ── Build SVG ──────────────────────────────────────────────────────────────

const lines: string[] = [];
const push = (s: string) => lines.push(s);

push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="system-ui,-apple-system,sans-serif" font-size="12">`);
push(`<rect width="${W}" height="${H}" fill="white"/>`);

// Clip path for main plot area (prevents band overflow)
push(`<defs>`);
push(`  <clipPath id="plot-clip"><rect x="${margin.left}" y="${margin.top}" width="${plotW}" height="${plotH}"/></clipPath>`);
// Clip for sparkline progressive reveal
push(`  <clipPath id="spark-clip">`);
push(`    <rect x="${sparkX}" y="${sparkY - 2}" width="0" height="${sparkH + 4}">`);
push(`      <animate attributeName="width" values="0;${sparkW};${sparkW}" keyTimes="0;${(animDuration / totalDuration).toFixed(4)};1" dur="${r(totalDuration)}s" repeatCount="indefinite"/>`);
push(`    </rect>`);
push(`  </clipPath>`);
push(`</defs>`);

// Grid lines
for (const v of yTicks) {
  push(`<line x1="${margin.left}" y1="${r(sy(v))}" x2="${W - margin.right}" y2="${r(sy(v))}" stroke="#e5e7eb" stroke-width="1"/>`);
}

// ── Animated elements (clipped to plot area) ───────────────────────────────

push(`<g clip-path="url(#plot-clip)">`);

// Confidence band
push(`<path fill="${bandColor}" stroke="none">`);
push(`  <animate attributeName="d" values="${bandValues.join(";")}" keyTimes="${keyTimesStr}" dur="${r(totalDuration)}s" repeatCount="indefinite" calcMode="discrete"/>`);
push(`</path>`);

// Level line
push(`<polyline fill="none" stroke="${lineColor}" stroke-width="2">`);
push(`  <animate attributeName="points" values="${polylineValues.join(";")}" keyTimes="${keyTimesStr}" dur="${r(totalDuration)}s" repeatCount="indefinite" calcMode="discrete"/>`);
push(`</polyline>`);

push(`</g>`);

// Observations (static, on top of line)
for (let i = 0; i < n; i++) {
  push(`<circle cx="${r(sx(t[i]))}" cy="${r(sy(y[i]))}" r="2.5" fill="${obsColor}" opacity="0.6"/>`);
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
  push(`<text x="${xx}" y="${H - margin.bottom + 18}" text-anchor="middle" fill="#333">${v}</text>`);
}

push(`<text x="${margin.left + plotW / 2}" y="${H - 5}" text-anchor="middle" fill="#333" font-size="13">Year</text>`);
push(`<text x="14" y="${margin.top + plotH / 2}" text-anchor="middle" fill="#333" font-size="13" transform="rotate(-90,14,${margin.top + plotH / 2})">Annual flow</text>`);

// ── Title ──────────────────────────────────────────────────────────────────

push(`<text x="${margin.left + plotW / 2}" y="16" text-anchor="middle" fill="#333" font-size="14" font-weight="600">Nile — MLE optimization (${iterations} iters, ${elapsedMs} ms WASM)</text>`);

// ── Legend ──────────────────────────────────────────────────────────────────

push(`<rect x="${legX}" y="${legY}" width="${legW}" height="${legH}" rx="4" fill="white" stroke="#e5e7eb" stroke-width="1"/>`);

// Observations
push(`<circle cx="${legX + 14}" cy="${legY + 14}" r="3" fill="${obsColor}" opacity="0.6"/>`);
push(`<text x="${legX + 24}" y="${legY + 14}" dominant-baseline="middle" fill="#333" font-size="11">Observations</text>`);

// MLE fit
push(`<line x1="${legX + 8}" y1="${legY + 32}" x2="${legX + 20}" y2="${legY + 32}" stroke="${lineColor}" stroke-width="2"/>`);
push(`<rect x="${legX + 8}" y="${legY + 27}" width="12" height="10" fill="${bandColor}" stroke="none"/>`);
push(`<text x="${legX + 24}" y="${legY + 32}" dominant-baseline="middle" fill="#333" font-size="11">MLE: s=${finalFrame.s.toFixed(1)}, w=[${finalFrame.w.map((v: number) => v.toFixed(1)).join(",")}]</text>`);

// Lik value
push(`<text x="${legX + 24}" y="${legY + 48}" dominant-baseline="middle" fill="#666" font-size="10">final −2·logL: ${finalLik.toFixed(2)}</text>`);

// ── Convergence miniplot (right half of legend) ────────────────────────────

// Label
push(`<text x="${sparkX}" y="${sparkY - 3}" fill="#666" font-size="9">\u22122\u00b7logL</text>`);

// Loss sparkline (with progressive reveal clip)
push(`<g clip-path="url(#spark-clip)">`);
push(`  <polyline points="${sparkPoints}" fill="none" stroke="${sparkColor}" stroke-width="1.5" stroke-linejoin="round"/>`);
push(`</g>`);

// Sparkline axis line (full, thin)
push(`<line x1="${sparkX}" y1="${sparkY + sparkH}" x2="${sparkX + sparkW}" y2="${sparkY + sparkH}" stroke="#ddd" stroke-width="0.5"/>`);

// Sparkline y markers
push(`<text x="${sparkX - 2}" y="${sparkY + 4}" text-anchor="end" fill="#999" font-size="7">${likMax.toFixed(0)}</text>`);
push(`<text x="${sparkX - 2}" y="${sparkY + sparkH}" text-anchor="end" fill="#999" font-size="7">${likMin.toFixed(0)}</text>`);

// Sparkline iteration axis labels: "0", "iters", "199"
const sparkAxisY = sparkY + sparkH + 9;
push(`<text x="${sparkX}" y="${sparkAxisY}" text-anchor="middle" fill="#999" font-size="7">0</text>`);
push(`<text x="${sparkX + sparkW / 2}" y="${sparkAxisY}" text-anchor="middle" fill="#999" font-size="7">iters</text>`);
push(`<text x="${sparkX + sparkW}" y="${sparkAxisY}" text-anchor="middle" fill="#999" font-size="7">${iterations}</text>`);

push(`</svg>`);

// ── Write output ───────────────────────────────────────────────────────────

const outDir = resolve(root, "assets");
mkdirSync(outDir, { recursive: true });
const outPath = resolve(outDir, "nile-mle-anim.svg");
writeFileSync(outPath, lines.join("\n"), "utf8");

const sizeKb = (Buffer.byteLength(lines.join("\n")) / 1024).toFixed(0);
console.log(`Written: ${outPath} (${sizeKb} KB)`);
console.log(`  ${numFrames} frames, ${r(totalDuration)}s cycle (${r(animDuration)}s play + ${holdSeconds}s hold)`);
