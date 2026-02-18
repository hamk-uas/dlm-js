/**
 * Shared SVG plotting helpers used by all SVG generators.
 *
 * Extracted to eliminate duplicate polyline/bandPath/axis/grid code
 * across gen-niledemo-svg.ts, gen-nile-mle-svg.ts, gen-kaisaniemi-svg.ts,
 * gen-trigar-svg.ts, gen-nile-mle-anim-svg.ts, gen-energy-mle-anim-svg.ts.
 */

import { writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

// ── Rounding ───────────────────────────────────────────────────────────────

/** Round to 1 decimal for compact SVG output. */
export const r = (v: number): string => v.toFixed(1);

// ── Scale factories ────────────────────────────────────────────────────────

/** Create a linear scale function mapping [domainMin, domainMax] → [rangeMin, rangeMax]. */
export function makeLinearScale(
  domainMin: number, domainMax: number,
  rangeMin: number, rangeMax: number,
): (v: number) => number {
  const domainSpan = domainMax - domainMin;
  const rangeSpan = rangeMax - rangeMin;
  return (v: number) => rangeMin + ((v - domainMin) / domainSpan) * rangeSpan;
}

// ── Polyline / band path ───────────────────────────────────────────────────

/** Build an SVG polyline `points` string from parallel x/y arrays. */
export function polylinePoints(
  xs: number[], ys: number[],
  sx: (v: number) => number, sy: (v: number) => number,
): string {
  return xs.map((x, i) => `${r(sx(x))},${r(sy(ys[i]))}`).join(" ");
}

/**
 * Build an SVG `<path d="...">` string for a closed confidence band
 * (upper edge forward, lower edge reversed).
 */
export function bandPathD(
  xs: number[], upper: number[], lower: number[],
  sx: (v: number) => number, sy: (v: number) => number,
): string {
  const fwd = xs.map((x, i) => `${r(sx(x))},${r(sy(upper[i]))}`);
  const bwd = xs.map((x, i) => `${r(sx(x))},${r(sy(lower[i]))}`).reverse();
  return `M${fwd.join("L")}L${bwd.join("L")}Z`;
}

// ── Tick generation ────────────────────────────────────────────────────────

/** Generate evenly spaced y-axis ticks from a range, with auto step. */
export function yTicksFromRange(min: number, max: number, step?: number): number[] {
  const span = max - min;
  const s = step ?? (span > 40 ? 10 : span > 15 ? 5 : span > 6 ? 2 : 1);
  const ticks: number[] = [];
  const start = Math.ceil(min / s) * s;
  const end = Math.floor(max / s) * s;
  for (let v = start; v <= end; v += s) ticks.push(v);
  return ticks;
}

// ── SVG element helpers ────────────────────────────────────────────────────

/** Render horizontal grid lines for given y-tick values. */
export function renderGridLines(
  yTicks: number[], sy: (v: number) => number,
  leftX: number, rightX: number,
): string[] {
  return yTicks.map((v) =>
    `<line x1="${leftX}" y1="${r(sy(v))}" x2="${rightX}" y2="${r(sy(v))}" stroke="#e5e7eb" stroke-width="1"/>`,
  );
}

/** Render y-axis tick marks and labels. */
export function renderYAxis(
  yTicks: number[], sy: (v: number) => number,
  leftX: number, fmt?: (v: number) => string,
): string[] {
  const format = fmt ?? ((v) => String(v));
  return yTicks.flatMap((v) => {
    const yy = r(sy(v));
    return [
      `<line x1="${leftX - 5}" y1="${yy}" x2="${leftX}" y2="${yy}" stroke="#333" stroke-width="1.5"/>`,
      `<text x="${leftX - 8}" y="${yy}" text-anchor="end" dominant-baseline="middle" fill="#333">${format(v)}</text>`,
    ];
  });
}

/** Render x-axis tick marks and labels. */
export function renderXAxis(
  ticks: { val: number; label: string }[],
  sx: (v: number) => number, bottomY: number,
): string[] {
  return ticks.flatMap(({ val, label }) => {
    const xx = r(sx(val));
    return [
      `<line x1="${xx}" y1="${bottomY}" x2="${xx}" y2="${bottomY + 5}" stroke="#333" stroke-width="1.5"/>`,
      `<text x="${xx}" y="${bottomY + 18}" text-anchor="middle" fill="#333">${label}</text>`,
    ];
  });
}

/** Render y-axis and x-axis border lines. */
export function renderAxesBorder(
  leftX: number, topY: number, rightX: number, bottomY: number,
): string[] {
  return [
    `<line x1="${leftX}" y1="${topY}" x2="${leftX}" y2="${bottomY}" stroke="#333" stroke-width="1.5"/>`,
    `<line x1="${leftX}" y1="${bottomY}" x2="${rightX}" y2="${bottomY}" stroke="#333" stroke-width="1.5"/>`,
  ];
}

// ── File I/O ───────────────────────────────────────────────────────────────

/** Write SVG lines to a file, creating directories as needed. Logs path and size. */
export function writeSvg(lines: string[], outPath: string): void {
  const content = lines.join("\n");
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, content, "utf8");
  const sizeKb = (Buffer.byteLength(content) / 1024).toFixed(0);
  console.log(`Written: ${outPath} (${sizeKb} KB)`);
}
