/**
 * Generate a static placeholder SVG for blocked animation variants.
 *
 * Used when a variant (e.g. WebGPU MLE) is blocked by an upstream issue —
 * produces a clean informational SVG instead of leaving a broken link.
 */

export interface PlaceholderOpts {
  /** Title shown at the top of the SVG. */
  title: string;
  /** One-line message describing why this variant is blocked. */
  message: string;
  /** Additional detail lines (smaller text). */
  details?: string[];
  /** SVG width (default 800). */
  width?: number;
  /** SVG height (default 200). */
  height?: number;
}

export function generatePlaceholderSvg(opts: PlaceholderOpts): string[] {
  const W = opts.width ?? 800;
  const H = opts.height ?? 200;
  const cx = W / 2;
  const lines: string[] = [];

  lines.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="system-ui,-apple-system,sans-serif" font-size="12">`);
  lines.push(`<rect width="${W}" height="${H}" fill="#fafafa" rx="8"/>`);
  lines.push(`<rect x="2" y="2" width="${W - 4}" height="${H - 4}" fill="none" stroke="#e5e7eb" stroke-width="1" rx="7" stroke-dasharray="6,4"/>`);

  // Title
  lines.push(`<text x="${cx}" y="40" text-anchor="middle" fill="#333" font-size="14" font-weight="600">${escSvg(opts.title)}</text>`);

  // Icon (clock / pending)
  lines.push(`<text x="${cx}" y="80" text-anchor="middle" fill="#9ca3af" font-size="28">\u23F3</text>`);

  // Message
  lines.push(`<text x="${cx}" y="110" text-anchor="middle" fill="#6b7280" font-size="12">${escSvg(opts.message)}</text>`);

  // Detail lines
  if (opts.details) {
    let dy = 130;
    for (const d of opts.details) {
      lines.push(`<text x="${cx}" y="${dy}" text-anchor="middle" fill="#9ca3af" font-size="10">${escSvg(d)}</text>`);
      dy += 16;
    }
  }

  lines.push(`</svg>`);
  return lines;
}

function escSvg(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
