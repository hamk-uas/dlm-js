/**
 * Generate an animated MLE optimization SVG from pre-collected frame data.
 *
 * Accepts a variant argument:
 *   scan   → reads tmp/mle-frames-nile-scan.json,  writes assets/nile-mle-anim-scan.svg
 *   assoc  → reads tmp/mle-frames-nile-assoc.json, writes assets/nile-mle-anim-assoc.svg
 *   webgpu → reads tmp/mle-frames-nile-webgpu.json, writes assets/nile-mle-anim-webgpu.svg
 *
 * Usage:  npx tsx scripts/gen-nile-mle-anim-svg.ts scan
 *         npx tsx scripts/gen-nile-mle-anim-svg.ts assoc
 *         npx tsx scripts/gen-nile-mle-anim-svg.ts webgpu
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
const inputPath = resolve(root, `tmp/mle-frames-nile-${variant}.json`);
const data = JSON.parse(readFileSync(inputPath, "utf8"));

const {
  t, y, n,
  elapsed: elapsedMs,
  jitMs = 0,
  iterations,
  holdSeconds,
  likHistory,
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
  frames: {
    iter: number;
    s: number;
    w: number[];
    lik: number | null;
    level: number[];
    std: number[];    // state (level) uncertainty std
    ystd: number[];   // observation prediction std
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

// Compute y-range across ALL frames using obs-prediction band (wider than state band)
const allVals: number[] = [...y];
for (const f of frames) {
  for (let i = 0; i < n; i++) {
    allVals.push(f.level[i] + 2 * f.ystd[i]);
    allVals.push(f.level[i] - 2 * f.ystd[i]);
  }
}
const yMin = Math.floor(Math.min(...allVals) / 50) * 50;
const yMax = Math.ceil(Math.max(...allVals) / 50) * 50;

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

// Values arrays are extended with a JIT-phase entry: [frame0(JIT hold), frame0..frameN-1(training), frameN-1(hold)]
// buildAnimPolylineValues returns N+1 items; prepend frame0 → N+2 to match mainKeyTimes.

const _trainingPolyValues = buildAnimPolylineValues(
  frames.map((f: any) => f.level as number[]), t, sx, sy,
);
const polylineValues = [_trainingPolyValues[0], ..._trainingPolyValues];

// State uncertainty band (narrow, more opaque)
const _trainingStateBandValues = buildAnimBandValues(
  frames.map((f: any) => ({
    upper: f.level.map((v: number, i: number) => v + 2 * f.std[i]),
    lower: f.level.map((v: number, i: number) => v - 2 * f.std[i]),
  })),
  t, sx, sy,
);
const stateBandValues = [_trainingStateBandValues[0], ..._trainingStateBandValues];

// Observation prediction band (wider = includes obs noise, lighter fill)
const _trainingObsBandValues = buildAnimBandValues(
  frames.map((f: any) => ({
    upper: f.level.map((v: number, i: number) => v + 2 * f.ystd[i]),
    lower: f.level.map((v: number, i: number) => v - 2 * f.ystd[i]),
  })),
  t, sx, sy,
);
const obsBandValues = [_trainingObsBandValues[0], ..._trainingObsBandValues];

// ── Loss sparkline (drawn in legend area) ──────────────────────────────────

const legW = 445;
const legH = 78;

// Sparkline occupies right half of legend
const sparkW = 155;
const sparkH = 30;
const sparkMarginRight = 10;
const sparkMarginTop = 22; // extra top margin so vmax label doesn't clip
const legX = W - margin.right - legW - 5;
const legY = margin.top + 8;

const sparkX = legX + legW - sparkMarginRight - sparkW;
const sparkY = legY + sparkMarginTop;

// JIT/training split within sparkline area
const jitFrac = Math.min(0.95, jitMs / elapsedMs);   // fraction of total time used by JIT
const jitBarW = Math.round(jitFrac * sparkW);          // pixel width of JIT box
const trainSparkW = sparkW - jitBarW;                   // pixel width of training sparkline
const sparkX_train = sparkX + jitBarW;                  // left edge of training sparkline

const likMin = Math.min(...likHistory);
const likMax = Math.max(...likHistory);

// Sparkline points are computed over the training section only
const sparkPoints = sparklinePoints(likHistory, sparkX_train, sparkY, trainSparkW, sparkH, likMin, likMax);

// ── Ticks ──────────────────────────────────────────────────────────────────

const yTicks: number[] = [];
for (let v = yMin; v <= yMax; v += 200) yTicks.push(v);
const tTicks: number[] = [];
for (let v = Math.ceil(tMin / 20) * 20; v <= tMax; v += 20) tTicks.push(v);

// ── Colors ─────────────────────────────────────────────────────────────────

const obsColor = "#555";
const lineColor = "#2563eb";
const stateBandColor = "rgba(37,99,235,0.22)";
const obsBandColor = "rgba(37,99,235,0.07)";
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
// Clip for sparkline progressive reveal — covers only the training section, starts revealing at jitEndFrac.
push(`  <clipPath id="spark-clip">`);
push(`    <rect x="${r(sparkX_train)}" y="${sparkY - 12}" width="0" height="${sparkH + 14}">`);
push(`      <animate attributeName="width" values="0;0;${r(trainSparkW)};${r(trainSparkW)}" keyTimes="0;${jitEndFrac.toFixed(4)};${trainEndFrac.toFixed(4)};1" dur="${r(totalDuration)}s" repeatCount="indefinite"/>`);
push(`    </rect>`);
push(`  </clipPath>`);
push(`</defs>`);

// Grid lines
lines.push(...renderGridLines(yTicks, sy, margin.left, W - margin.right));

// ── Animated elements (clipped to plot area) ───────────────────────────────

push(`<g clip-path="url(#plot-clip)">`);

// Observation prediction band (wider, behind state band)
push(`<path fill="${obsBandColor}" stroke="none">`);
push(`  <animate attributeName="d" values="${obsBandValues.join(";") }" keyTimes="${mainKeyTimes}" dur="${r(totalDuration)}s" repeatCount="indefinite" calcMode="discrete"/>`);
push(`</path>`);

// State uncertainty band (narrower, more opaque)
push(`<path fill="${stateBandColor}" stroke="none">`);
push(`  <animate attributeName="d" values="${stateBandValues.join(";") }" keyTimes="${mainKeyTimes}" dur="${r(totalDuration)}s" repeatCount="indefinite" calcMode="discrete"/>`);
push(`</path>`);

// Level line
push(`<polyline fill="none" stroke="${lineColor}" stroke-width="2">`);
push(`  <animate attributeName="points" values="${polylineValues.join(";") }" keyTimes="${mainKeyTimes}" dur="${r(totalDuration)}s" repeatCount="indefinite" calcMode="discrete"/>`);
push(`</polyline>`);

push(`</g>`);

// Observations (static, on top of line)
for (let i = 0; i < n; i++) {
  push(`<circle cx="${r(sx(t[i]))}" cy="${r(sy(y[i]))}" r="2.5" fill="${obsColor}" opacity="0.6"/>`);
}

// ── Axes ───────────────────────────────────────────────────────────────────

lines.push(...renderAxesBorder(margin.left, margin.top, W - margin.right, H - margin.bottom));
lines.push(...renderYAxis(yTicks, sy, margin.left));
lines.push(...renderXAxis(tTicks.map(v => ({ val: v, label: String(v) })), sx, H - margin.bottom));

push(`<text x="${margin.left + plotW / 2}" y="${H - 5}" text-anchor="middle" fill="#333" font-size="13">Year</text>`);
push(`<text x="14" y="${margin.top + plotH / 2}" text-anchor="middle" fill="#333" font-size="13" transform="rotate(-90,14,${margin.top + plotH / 2})">Annual flow</text>`);

// ── Title ──────────────────────────────────────────────────────────────────

push(`<text x="${W / 2}" y="16" text-anchor="middle" fill="#333" font-size="14" font-weight="600">Nile demo — MLE (order=1, trend), ${iterations} iters, ${elapsedMs} ms, ${backendLabel}</text>`);

// ── Legend ──────────────────────────────────────────────────────────────────

push(`<rect x="${legX}" y="${legY}" width="${legW}" height="${legH}" rx="4" fill="white" stroke="#e5e7eb" stroke-width="1"/>`);

// JIT fill rect — placed immediately after legend background so it renders behind all text/lines
if (jitBarW > 0) {
  const jitBarH = 11;
  const jitBarY = Math.round(sparkY + (sparkH - jitBarH) / 2);
  push(`<rect x="${r(sparkX)}" y="${jitBarY}" width="0" height="${jitBarH}" fill="#f3f4f6">`);
  push(`  <animate attributeName="width" values="0;${r(jitBarW)};${r(jitBarW)};${r(jitBarW)}" keyTimes="0;${jitEndFrac.toFixed(4)};${trainEndFrac.toFixed(4)};1" dur="${r(totalDuration)}s" repeatCount="indefinite"/>`);
  push(`</rect>`);
}

// Observations
push(`<circle cx="${legX + 14}" cy="${legY + 14}" r="3" fill="${obsColor}" opacity="0.6"/>`);
push(`<text x="${legX + 24}" y="${legY + 14}" dominant-baseline="middle" fill="#333" font-size="11">Observations</text>`);

// MLE fit — state band swatch
push(`<line x1="${legX + 8}" y1="${legY + 32}" x2="${legX + 20}" y2="${legY + 32}" stroke="${lineColor}" stroke-width="2"/>`);
push(`<rect x="${legX + 8}" y="${legY + 27}" width="12" height="10" fill="${stateBandColor}" stroke="none"/>`);
push(`<text x="${legX + 24}" y="${legY + 32}" dominant-baseline="middle" fill="#333" font-size="11">MLE level \u00b12\u03c3 state, final s=${finalFrame.s.toFixed(1)}, w=[${finalFrame.w.map((v: number) => v.toFixed(1)).join(",")}]</text>`);

// Obs prediction band swatch
push(`<rect x="${legX + 8}" y="${legY + 46}" width="12" height="10" fill="${obsBandColor}" stroke="#2563eb" stroke-width="0.5"/>`);
push(`<text x="${legX + 24}" y="${legY + 51}" dominant-baseline="middle" fill="#666" font-size="11">\u00b12\u03c3 obs prediction (incl. noise)</text>`);

// Lik value
push(`<text x="${legX + 24}" y="${legY + 65}" dominant-baseline="middle" fill="#666" font-size="10">Final \u22122\u00b7logL: ${finalLik.toFixed(2)}</text>`);

// ── Convergence miniplot (right half of legend) ────────────────────────────

// JIT label + separator (rendered on top of jit fill rect)
if (jitBarW > 0) {
  // "jit" text: fades in during JIT phase, stays visible
  push(`<text x="${r(sparkX + jitBarW / 2)}" y="${r(sparkY + sparkH / 2)}" text-anchor="middle" dominant-baseline="middle" fill="#9ca3af" font-size="7" opacity="0">`);
  push(`  <animate attributeName="opacity" values="0;0;1;1" keyTimes="0;${(jitEndFrac * 0.5).toFixed(4)};${jitEndFrac.toFixed(4)};1" dur="${r(totalDuration)}s" repeatCount="indefinite"/>`);
  push(`  jit`);
  push(`</text>`);
  // Separator between jit and training sections
  push(`<line x1="${r(sparkX_train)}" y1="${sparkY}" x2="${r(sparkX_train)}" y2="${sparkY + sparkH}" stroke="#d1d5db" stroke-width="0.5" stroke-dasharray="2,2"/>`);
}

// Sparkline polyline inside clip group (progressively revealed during training)
push(`<g clip-path="url(#spark-clip)">`);
lines.push(...renderSparkline({
  points: sparkPoints,
  color: sparkColor,
  x0: sparkX_train, y0: sparkY, w: trainSparkW, h: sparkH,
  label: "\u22122\u00b7logL",
  vmin: likMin, vmax: likMax,
  noLabels: true,
  noBaseline: true,
}));
push(`</g>`);

// Baseline outside clip group (always visible, not affected by reveal)
push(`<line x1="${r(sparkX_train)}" y1="${sparkY + sparkH}" x2="${r(sparkX_train + trainSparkW)}" y2="${sparkY + sparkH}" stroke="#eee" stroke-width="0.5"/>`);

// Sparkline labels on top of baseline
lines.push(...renderSparklineLabels({
  x0: sparkX_train, y0: sparkY, h: sparkH,
  label: "\u22122\u00b7logL",
  vmin: likMin, vmax: likMax,
}));

// Sparkline x-axis labels
const sparkAxisY = sparkY + sparkH + 9;
const trainMs = Math.round(trainDuration * 1000);
push(`<text x="${r(sparkX_train)}" y="${sparkAxisY}" text-anchor="middle" fill="#999" font-size="7">0</text>`);
push(`<text x="${r(sparkX_train + trainSparkW / 2)}" y="${sparkAxisY}" text-anchor="middle" fill="#999" font-size="7">iters</text>`);
push(`<text x="${r(sparkX_train + trainSparkW)}" y="${sparkAxisY}" text-anchor="middle" fill="#999" font-size="7">${iterations}</text>`);
// "train Xms" label overlaid on center of loss sparkline, fades in at end of JIT phase
push(`<text x="${r(sparkX_train + trainSparkW / 2)}" y="${r(sparkY + sparkH / 2)}" text-anchor="middle" dominant-baseline="middle" fill="#9ca3af" font-size="7" opacity="0">`);
push(`  <animate attributeName="opacity" values="0;0;1;1" keyTimes="0;${jitEndFrac.toFixed(4)};${trainEndFrac.toFixed(4)};1" dur="${r(totalDuration)}s" repeatCount="indefinite"/>`);
push(`  train ${trainMs}ms`);
push(`</text>`);

push(`</svg>`);

// ── Write output ───────────────────────────────────────────────────────────

const outPath = resolve(root, "assets", `nile-mle-anim-${variant}.svg`);
writeSvg(lines, outPath);
console.log(`  [${variant}] ${numFrames} frames, ${r(totalDuration)}s cycle (${r(animDuration)}s play + ${holdSeconds}s hold)`);
