/**
 * Generate an animated MLE SVG (with AR coefficient estimation) from pre-collected energy frames.
 *
 * Accepts a variant argument:
 *   scan   → reads tmp/mle-frames-energy-scan.json,  writes assets/energy-mle-anim-scan.svg
 *   assoc  → reads tmp/mle-frames-energy-assoc.json, writes assets/energy-mle-anim-assoc.svg
 *   webgpu → reads tmp/mle-frames-energy-webgpu.json, writes assets/energy-mle-anim-webgpu.svg
 *
 * Usage:  npx tsx scripts/gen-energy-mle-anim-svg.ts scan
 *         npx tsx scripts/gen-energy-mle-anim-svg.ts assoc
 *         npx tsx scripts/gen-energy-mle-anim-svg.ts webgpu
 */

import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import {
  r, makeLinearScale, renderGridLines, renderYAxis, renderXAxis, renderAxesBorder, writeSvg,
} from "./lib/svg-helpers.ts";
import {
  computeKeyTimes, buildAnimPolylineValues, buildAnimBandValues,
  sparklinePoints, renderSparkline, renderSparklineLabels,
} from "./lib/svg-anim-helpers.ts";

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const variant = (process.argv[2] || "scan") as "scan" | "assoc" | "webgpu";

const backendLabel: Record<string, string> = { scan: "scan/WASM/f64", assoc: "assoc/WASM/f64", webgpu: "assoc/WebGPU/f32" }[variant] ?? "scan/WASM/f64";
const inputPath = resolve(root, `tmp/mle-frames-energy-${variant}.json`);
const data = JSON.parse(readFileSync(inputPath, "utf8"));

const {
  t, y, n,
  elapsed: elapsedMs,
  jitMs = 0,
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
  jitMs?: number;
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

const sx = makeLinearScale(tMin, tMax, margin.left, margin.left + plotW);
const sy = makeLinearScale(yMin, yMax, margin.top + plotH, margin.top);

// ── Animation timing ───────────────────────────────────────────────────────

const animDuration = elapsedMs / 1000;       // total play duration (jit + training)
const jitDuration = jitMs / 1000;            // JIT phase duration
const trainDuration = animDuration - jitDuration; // actual training duration
const totalDuration = animDuration + holdSeconds;
const numFrames = frames.length;

// Fractional positions in the [0,1] normalized timeline
const jitEndFrac = jitDuration / totalDuration;
const trainEndFrac = animDuration / totalDuration;

// Main plot: hold frame0 during JIT phase, then animate N frames, then hold.
// keyTimes: length numFrames+2.  values: length numFrames+2.
const mainKeyTimes: string = [
  (0).toFixed(4),
  ...Array.from({ length: numFrames }, (_, i) => {
    const kt = numFrames === 1
      ? jitEndFrac
      : jitEndFrac + (i / (numFrames - 1)) * (trainEndFrac - jitEndFrac);
    return kt.toFixed(4);
  }),
  (1).toFixed(4),
].join(";");

// ── Pre-compute polyline points and band paths per frame ───────────────────
// Values arrays extended with JIT-phase copy: len numFrames+2.

const _trainingPolyValues = buildAnimPolylineValues(
  frames.map((f: any) => f.combined as number[]), t, sx, sy,
);
const polylineValues = [_trainingPolyValues[0], ..._trainingPolyValues];

const _trainingBandValues = buildAnimBandValues(
  frames.map((f: any) => ({
    upper: f.combined.map((v: number, i: number) => v + 2 * f.combinedStd[i]),
    lower: f.combined.map((v: number, i: number) => v - 2 * f.combinedStd[i]),
  })),
  t, sx, sy,
);
const bandValues = [_trainingBandValues[0], ..._trainingBandValues];

// ── Legend layout ──────────────────────────────────────────────────────────

const legW = 400;
const legH = 72;
const legX = margin.left + 5;
const legY = margin.top + 8;

// Right half: two mini sparklines stacked
const sparkW = 140;
const sparkH1 = 20; // −2·logL sparkline
const sparkH2 = 20; // arphi sparkline
const sparkGap = 6;
const sparkMarginRight = 10;
const sparkMarginTop = 14; // top margin: enough for label clearance from legend border

const sparkX = legX + legW - sparkMarginRight - sparkW;
const sparkY1 = legY + sparkMarginTop;
const sparkY2 = sparkY1 + sparkH1 + sparkGap;

// JIT/training split within sparkline area
const jitFrac = Math.min(0.95, jitMs / elapsedMs);
const jitBarW = Math.round(jitFrac * sparkW);
const trainSparkW = sparkW - jitBarW;
const sparkX_train = sparkX + jitBarW;

// −2·logL sparkline (training portion only)
const likMin = Math.min(...likHistory);
const likMax = Math.max(...likHistory);

const sparkLikPoints = sparklinePoints(likHistory, sparkX_train, sparkY1, trainSparkW, sparkH1, likMin, likMax);

// arphi sparkline (training portion only)
const arphiMin = Math.min(...arphiHistory) * 0.95;
const arphiMax = Math.max(...arphiHistory) * 1.05;

const sparkArphiPoints = sparklinePoints(arphiHistory, sparkX_train, sparkY2, trainSparkW, sparkH2, arphiMin, arphiMax);

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
// Clip covers only the training section; starts revealing at jitEndFrac.
push(`  <clipPath id="spark-clip">`);
push(`    <rect x="${r(sparkX_train)}" y="${sparkY1 - 12}" width="0" height="${sparkH1 + sparkGap + sparkH2 + 14}">`);
push(`      <animate attributeName="width" values="0;0;${r(trainSparkW)};${r(trainSparkW)}" keyTimes="0;${jitEndFrac.toFixed(4)};${trainEndFrac.toFixed(4)};1" dur="${r(totalDuration)}s" repeatCount="indefinite"/>`);
push(`    </rect>`);
push(`  </clipPath>`);
push(`</defs>`);

// Grid lines
lines.push(...renderGridLines(yTicks, sy, margin.left, W - margin.right));

// ── Animated elements ──────────────────────────────────────────────────────

push(`<g clip-path="url(#plot-clip)">`);

// Confidence band
push(`<path fill="${bandColor}" stroke="none">`);
push(`  <animate attributeName="d" values="${bandValues.join(";") }" keyTimes="${mainKeyTimes}" dur="${r(totalDuration)}s" repeatCount="indefinite" calcMode="discrete"/>`);
push(`</path>`);

// Combined signal line
push(`<polyline fill="none" stroke="${lineColor}" stroke-width="2">`);
push(`  <animate attributeName="points" values="${polylineValues.join(";") }" keyTimes="${mainKeyTimes}" dur="${r(totalDuration)}s" repeatCount="indefinite" calcMode="discrete"/>`);
push(`</polyline>`);

push(`</g>`);

// Observations (static)
for (let i = 0; i < n; i++) {
  push(`<circle cx="${r(sx(t[i]))}" cy="${r(sy(y[i]))}" r="2" fill="${obsColor}" opacity="0.5"/>`);
}

// ── Axes ───────────────────────────────────────────────────────────────────

lines.push(...renderAxesBorder(margin.left, margin.top, W - margin.right, H - margin.bottom));
lines.push(...renderYAxis(yTicks, sy, margin.left));

const xTickLabels = tTicks.map(v => {
  const yrs = v / 12;
  return { val: v, label: Number.isInteger(yrs) ? `${yrs}y` : String(v) };
});
lines.push(...renderXAxis(xTickLabels, sx, H - margin.bottom));

push(`<text x="${margin.left + plotW / 2}" y="${H - 5}" text-anchor="middle" fill="#333" font-size="13">Month</text>`);
push(`<text x="14" y="${margin.top + plotH / 2}" text-anchor="middle" fill="#333" font-size="13" transform="rotate(-90,14,${margin.top + plotH / 2})">Energy demand</text>`);

// ── Title ──────────────────────────────────────────────────────────────────

push(`<text x="${W / 2}" y="16" text-anchor="middle" fill="#333" font-size="14" font-weight="600">Synthetic energy demand — MLE (order=1, harmonics=1, seasonLength=12, AR(1)), ${iterations} iters, ${(elapsedMs / 1000).toFixed(1)} s, ${backendLabel}</text>`);

// ── Legend ──────────────────────────────────────────────────────────────────

push(`<rect x="${legX}" y="${legY}" width="${legW}" height="${legH}" rx="4" fill="white" stroke="#e5e7eb" stroke-width="1"/>`);

// JIT fill rect — placed immediately after legend background so it renders behind all text/lines
if (jitBarW > 0) {
  const jitBarH = 11;
  const jitBarY = Math.round(sparkY1 + (sparkH1 - jitBarH) / 2);
  push(`<rect x="${r(sparkX)}" y="${jitBarY}" width="0" height="${jitBarH}" fill="#f3f4f6">`);
  push(`  <animate attributeName="width" values="0;${r(jitBarW)};${r(jitBarW)};${r(jitBarW)}" keyTimes="0;${jitEndFrac.toFixed(4)};${trainEndFrac.toFixed(4)};1" dur="${r(totalDuration)}s" repeatCount="indefinite"/>`);
  push(`</rect>`);
}

// Observations
push(`<circle cx="${legX + 14}" cy="${legY + 14}" r="2.5" fill="${obsColor}" opacity="0.5"/>`);
push(`<text x="${legX + 24}" y="${legY + 14}" dominant-baseline="middle" fill="#333" font-size="11">Observations</text>`);

// MLE fit line
push(`<line x1="${legX + 8}" y1="${legY + 30}" x2="${legX + 20}" y2="${legY + 30}" stroke="${lineColor}" stroke-width="2"/>`);
push(`<rect x="${legX + 8}" y="${legY + 25}" width="12" height="10" fill="${bandColor}" stroke="none"/>`);
push(`<text x="${legX + 24}" y="${legY + 30}" dominant-baseline="middle" fill="#333" font-size="11">MLE: final s=${finalFrame.s.toFixed(1)}, \u03c6=${finalArphi.toFixed(2)}</text>`);

// Lik + arphi values
push(`<text x="${legX + 24}" y="${legY + 46}" dominant-baseline="middle" fill="#666" font-size="10">Final \u22122\u00b7logL = ${finalLik.toFixed(1)}</text>`);

// Model description
push(`<text x="${legX + 24}" y="${legY + 60}" dominant-baseline="middle" fill="#999" font-size="9">order=1, harmonics=1, seasonLength=12, fitAr=true</text>`);

// ── Convergence miniplots (right half) ─────────────────────────────────────

// JIT label + separator (rendered on top of jit fill rect)
if (jitBarW > 0) {
  // "jit" text: fades in during JIT phase, stays visible
  push(`<text x="${r(sparkX + jitBarW / 2)}" y="${r(sparkY1 + sparkH1 / 2)}" text-anchor="middle" dominant-baseline="middle" fill="#9ca3af" font-size="7" opacity="0">`);
  push(`  <animate attributeName="opacity" values="0;0;1;1" keyTimes="0;${(jitEndFrac * 0.5).toFixed(4)};${jitEndFrac.toFixed(4)};1" dur="${r(totalDuration)}s" repeatCount="indefinite"/>`);
  push(`  jit`);
  push(`</text>`);
  push(`<line x1="${r(sparkX_train)}" y1="${sparkY1}" x2="${r(sparkX_train)}" y2="${sparkY2 + sparkH2}" stroke="#d1d5db" stroke-width="0.5" stroke-dasharray="2,2"/>`);
}

// Polylines inside the clip group (progressively revealed during training)
push(`<g clip-path="url(#spark-clip)">`);

// −2·logL sparkline
lines.push(...renderSparkline({
  points: sparkLikPoints,
  color: sparkLikColor,
  x0: sparkX_train, y0: sparkY1, w: trainSparkW, h: sparkH1,
  label: "\u22122\u00b7logL",
  vmin: likMin, vmax: likMax,
  noLabels: true,
  noBaseline: true,
}));

// arphi sparkline
lines.push(...renderSparkline({
  points: sparkArphiPoints,
  color: sparkArColor,
  x0: sparkX_train, y0: sparkY2, w: trainSparkW, h: sparkH2,
  label: "\u03c6 (AR)",
  vmin: arphiMin, vmax: arphiMax,
  vminFmt: arphiMin.toFixed(2),
  vmaxFmt: arphiMax.toFixed(2),
  noLabels: true,
  noBaseline: true,
}));

push(`</g>`);

// Baseline lines outside clip group (always visible, not affected by reveal)
push(`<line x1="${r(sparkX_train)}" y1="${sparkY1 + sparkH1}" x2="${r(sparkX_train + trainSparkW)}" y2="${sparkY1 + sparkH1}" stroke="#eee" stroke-width="0.5"/>`);
push(`<line x1="${r(sparkX_train)}" y1="${sparkY2 + sparkH2}" x2="${r(sparkX_train + trainSparkW)}" y2="${sparkY2 + sparkH2}" stroke="#eee" stroke-width="0.5"/>`);

// Sparkline labels on top of baselines
lines.push(...renderSparklineLabels({
  x0: sparkX_train, y0: sparkY1, h: sparkH1,
  label: "\u22122\u00b7logL",
  vmin: likMin, vmax: likMax,
}));
lines.push(...renderSparklineLabels({
  x0: sparkX_train, y0: sparkY2, h: sparkH2,
  label: "\u03c6 (AR)",
  vmin: arphiMin, vmax: arphiMax,
  vminFmt: arphiMin.toFixed(2),
  vmaxFmt: arphiMax.toFixed(2),
}));

// Shared x-axis labels (training portion)
const sparkAxisY = sparkY2 + sparkH2 + 8;
const trainMs = Math.round(trainDuration * 1000);
push(`<text x="${r(sparkX_train)}" y="${sparkAxisY}" text-anchor="middle" fill="#999" font-size="7">0</text>`);
push(`<text x="${r(sparkX_train + trainSparkW / 2)}" y="${sparkAxisY}" text-anchor="middle" fill="#999" font-size="7">iters</text>`);
push(`<text x="${r(sparkX_train + trainSparkW)}" y="${sparkAxisY}" text-anchor="middle" fill="#999" font-size="7">${iterations}</text>`);
// "train Xms" label overlaid on center of loss sparkline, fades in at end of JIT phase
push(`<text x="${r(sparkX_train + trainSparkW / 2)}" y="${r(sparkY1 + sparkH1 / 2)}" text-anchor="middle" dominant-baseline="middle" fill="#9ca3af" font-size="7" opacity="0">`);
push(`  <animate attributeName="opacity" values="0;0;1;1" keyTimes="0;${jitEndFrac.toFixed(4)};${trainEndFrac.toFixed(4)};1" dur="${r(totalDuration)}s" repeatCount="indefinite"/>`);
push(`  train ${trainMs}ms`);
push(`</text>`);

push(`</svg>`);

// ── Write output ───────────────────────────────────────────────────────────

const outPath = resolve(root, "assets", `energy-mle-anim-${variant}.svg`);
writeSvg(lines, outPath);
console.log(`  [${variant}] ${numFrames} frames, ${r(totalDuration)}s cycle (${r(animDuration)}s play + ${holdSeconds}s hold)`);
