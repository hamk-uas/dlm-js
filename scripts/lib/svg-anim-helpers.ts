/**
 * Shared SMIL animation helpers for the animated MLE SVG generators.
 *
 * Extracted from gen-nile-mle-anim-svg.ts and gen-energy-mle-anim-svg.ts.
 */

import { r, polylinePoints, bandPathD } from "./svg-helpers.ts";

// ── Key times ──────────────────────────────────────────────────────────────

/**
 * Compute SMIL `keyTimes` for discrete frame animation with a hold period.
 *
 * Frames are evenly spread across `animDuration`, then one extra duplicate
 * frame at keyTime=1.0 creates the hold effect.
 *
 * @returns Array of length `numFrames + 1`
 */
export function computeKeyTimes(
  numFrames: number, animDuration: number, totalDuration: number,
): number[] {
  const keyTimes: number[] = [];
  for (let i = 0; i < numFrames; i++) {
    keyTimes.push((i / (numFrames - 1)) * animDuration / totalDuration);
  }
  keyTimes.push(1.0);
  return keyTimes;
}

// ── Animated polyline / band ───────────────────────────────────────────────

/**
 * Build per-frame polyline `points` strings + a duplicate for hold.
 *
 * @param frames - Array of per-frame y-values
 * @param xs - Shared x-axis values
 * @param sx - x scale function
 * @param sy - y scale function
 * @returns Array of length `frames.length + 1` (last duplicated for hold)
 */
export function buildAnimPolylineValues(
  frames: number[][], xs: number[],
  sx: (v: number) => number, sy: (v: number) => number,
): string[] {
  const values = frames.map((ys) => polylinePoints(xs, ys, sx, sy));
  values.push(values[values.length - 1]);
  return values;
}

/**
 * Build per-frame band `<path d>` strings + a duplicate for hold.
 *
 * @param frames - Array of { upper, lower } arrays per frame
 * @param xs - Shared x-axis values
 * @param sx - x scale function
 * @param sy - y scale function
 * @returns Array of length `frames.length + 1`
 */
export function buildAnimBandValues(
  frames: { upper: number[]; lower: number[] }[],
  xs: number[],
  sx: (v: number) => number, sy: (v: number) => number,
): string[] {
  const values = frames.map(({ upper, lower }) => bandPathD(xs, upper, lower, sx, sy));
  values.push(values[values.length - 1]);
  return values;
}

// ── Sparkline ──────────────────────────────────────────────────────────────

/** Build a polyline `points` string for a sparkline (mini time-series plot). */
export function sparklinePoints(
  values: number[],
  x0: number, y0: number, w: number, h: number,
  vmin: number, vmax: number,
): string {
  const vrange = vmax - vmin || 1;
  return values.map((v, i) => {
    const px = x0 + (i / (values.length - 1)) * w;
    const py = y0 + h - ((v - vmin) / vrange) * h;
    return `${r(px)},${r(py)}`;
  }).join(" ");
}

/**
 * Render a sparkline with a progressive-reveal clip animation.
 *
 * @param id - Unique clip-path id
 * @param points - Pre-computed polyline points
 * @param color - Stroke color
 * @param x0 - Left edge
 * @param y0 - Top edge
 * @param w - Width
 * @param h - Height
 * @param label - Label text shown above
 * @param vmin/vmax - Value range for y-axis labels
 * @param animDuration - Play duration in seconds
 * @param totalDuration - Total cycle duration in seconds
 */
export function renderSparkline(opts: {
  points: string;
  color: string;
  x0: number;
  y0: number;
  w: number;
  h: number;
  label: string;
  vmin: number;
  vmax: number;
  vminFmt?: string;
  vmaxFmt?: string;
  /** When true, omit the static label/vmin/vmax texts so they can be placed outside a clip group. */
  noLabels?: boolean;
  /** When true, omit the horizontal baseline rule so it can be placed outside a reveal clip group. */
  noBaseline?: boolean;
}): string[] {
  const { points, color, x0, y0, w, h, label, vmin, vmax } = opts;
  const vmaxFmt = opts.vmaxFmt ?? vmax.toFixed(0);
  const vminFmt = opts.vminFmt ?? vmin.toFixed(0);
  const poly = `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linejoin="round"/>`;
  const rule = `<line x1="${x0}" y1="${y0 + h}" x2="${x0 + w}" y2="${y0 + h}" stroke="#eee" stroke-width="0.5"/>`;
  const polyAndRule = opts.noBaseline ? [poly] : [poly, rule];
  if (opts.noLabels) return polyAndRule;
  return [
    `<text x="${x0}" y="${y0 - 2}" fill="#666" font-size="8">${label}</text>`,
    ...polyAndRule,
    `<text x="${x0 - 2}" y="${y0 + 3}" text-anchor="end" fill="#999" font-size="7">${vmaxFmt}</text>`,
    `<text x="${x0 - 2}" y="${y0 + h}" text-anchor="end" fill="#999" font-size="7">${vminFmt}</text>`,
  ];
}

/** Emit the static label/vmin/vmax texts for a sparkline outside a clip group. */
export function renderSparklineLabels(opts: {
  x0: number; y0: number; h: number;
  label: string; vmin: number; vmax: number;
  vminFmt?: string; vmaxFmt?: string;
}): string[] {
  const { x0, y0, h, label, vmin, vmax } = opts;
  const vmaxFmt = opts.vmaxFmt ?? vmax.toFixed(0);
  const vminFmt = opts.vminFmt ?? vmin.toFixed(0);
  return [
    `<text x="${x0}" y="${y0 - 2}" fill="#666" font-size="8">${label}</text>`,
    `<text x="${x0 - 2}" y="${y0 + 3}" text-anchor="end" fill="#999" font-size="7">${vmaxFmt}</text>`,
    `<text x="${x0 - 2}" y="${y0 + h}" text-anchor="end" fill="#999" font-size="7">${vminFmt}</text>`,
  ];
}
