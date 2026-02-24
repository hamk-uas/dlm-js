/**
 * Artifact dependency graph.
 *
 * Centrally declares which source files produce which output artifacts,
 * and the command to regenerate each group.  Used by `check-freshness.ts`
 * to detect stale artifacts and by preflight to warn developers.
 *
 * ## How staleness is detected
 *
 * For each group, the checker compares the mtime of the newest source file
 * against the mtime of the oldest output file.  If any source is newer than
 * any output, the group is stale and the regeneration command is shown.
 *
 * ## Adding a new artifact group
 *
 * 1. Add an entry to `artifactGroups` below.
 * 2. Run `pnpm run check:freshness` to verify the declaration is correct.
 */

import { resolve, dirname } from "node:path";
import { statSync, existsSync } from "node:fs";

export const ROOT = resolve(dirname(new URL(import.meta.url).pathname), "../..");

export interface ArtifactGroup {
  /** Short identifier for the group. */
  id: string;
  /** Human-readable label (shown in reports). */
  label: string;
  /** Glob patterns (relative to repo root) for source files. */
  sources: string[];
  /** Glob patterns or literal paths (relative to root) for output files. */
  outputs: string[];
  /** pnpm command to regenerate this group. */
  command: string;
  /**
   * If true, regeneration is slow (>30 s) — only warn, never auto-run.
   * If false, could be auto-run in strict preflight mode.
   */
  heavy: boolean;
}

export const artifactGroups: ArtifactGroup[] = [
  // ── dlmFit SVGs (scan/assoc/sqrt-assoc variants) ────────────────────────
  {
    id: "fit-svgs",
    label: "dlmFit demo SVGs + timing sidecars",
    sources: [
      "src/index.ts",
      "src/dlmgensys.ts",
      "src/types.ts",
      "scripts/gen-niledemo-svg.ts",
      "scripts/gen-kaisaniemi-svg.ts",
      "scripts/gen-trigar-svg.ts",
      "scripts/gen-ozone-svg.ts",
      "scripts/gen-gapped-svg.ts",
      "scripts/lib/svg-helpers.ts",
      "scripts/lib/svg-placeholder.ts",
      "tests/niledemo-in.json",
      "tests/kaisaniemi-in.json",
      "tests/trigar-in.json",
      "tests/ozone-in.json",
      "tests/gapped-in.json",
    ],
    outputs: [
      "assets/niledemo-scan.svg",
      "assets/niledemo-assoc.svg",
      "assets/niledemo-sqrt-assoc.svg",
      "assets/niledemo-sqrt-assoc-f32.svg",
      "assets/kaisaniemi-scan.svg",
      "assets/kaisaniemi-assoc.svg",
      "assets/trigar-scan.svg",
      "assets/trigar-assoc.svg",
      "assets/ozone-demo-scan.svg",
      "assets/ozone-demo-assoc.svg",
      "assets/gapped-demo-scan.svg",
      "assets/gapped-demo-assoc.svg",
      "assets/timings/gen-niledemo-svg.json",
      "assets/timings/gen-niledemo-svg-assoc.json",
      "assets/timings/gen-niledemo-svg-sqrt-assoc.json",
      "assets/timings/gen-niledemo-svg-sqrt-assoc-f32.json",
      "assets/timings/gen-kaisaniemi-svg.json",
      "assets/timings/gen-kaisaniemi-svg-assoc.json",
      "assets/timings/gen-trigar-svg.json",
      "assets/timings/gen-trigar-svg-assoc.json",
      "assets/timings/gen-gapped-svg.json",
      "assets/timings/gen-gapped-svg-assoc.json",
    ],
    command: "pnpm run gen:svg:fit",
    heavy: false,
  },

  // ── MLE animation SVGs ──────────────────────────────────────────────────
  {
    id: "mle-svgs",
    label: "MLE animation SVGs + frame sidecars",
    sources: [
      "src/index.ts",
      "src/mle.ts",
      "src/dlmgensys.ts",
      "src/types.ts",
      "scripts/collect-nile-mle-frames.ts",
      "scripts/collect-energy-mle-frames.ts",
      "scripts/gen-nile-mle-anim-svg.ts",
      "scripts/gen-energy-mle-anim-svg.ts",
      "scripts/lib/svg-helpers.ts",
      "scripts/lib/svg-anim-helpers.ts",
      "tests/niledemo-in.json",
      "tests/energy-in.json",
    ],
    outputs: [
      "assets/nile-mle-anim-scan.svg",
      "assets/nile-mle-anim-assoc.svg",
      "assets/energy-mle-anim-scan.svg",
      "assets/energy-mle-anim-assoc.svg",
      "assets/timings/collect-nile-mle-frames.json",
      "assets/timings/collect-nile-mle-frames-assoc.json",
      "assets/timings/collect-energy-mle-frames.json",
      "assets/timings/collect-energy-mle-frames-assoc.json",
    ],
    command: "pnpm run bench:wasm   # (includes MLE frame collection + SVG gen)",
    heavy: true,
  },

  // ── WebGPU MLE animations (separate because requires Deno) ──────────────
  {
    id: "mle-svgs-webgpu",
    label: "MLE animation SVGs (WebGPU variants)",
    sources: [
      "src/index.ts",
      "src/mle.ts",
      "src/dlmgensys.ts",
      "src/types.ts",
      "scripts/collect-nile-mle-frames-webgpu.ts",
      "scripts/collect-energy-mle-frames-webgpu.ts",
      "scripts/gen-nile-mle-anim-svg.ts",
      "scripts/gen-energy-mle-anim-svg.ts",
    ],
    outputs: [
      "assets/nile-mle-anim-webgpu.svg",
      "assets/energy-mle-anim-webgpu.svg",
      "assets/timings/collect-nile-mle-frames-webgpu.json",
      "assets/timings/collect-energy-mle-frames-webgpu.json",
    ],
    command: "pnpm run gen:svg:nile-mle-anim && pnpm run gen:svg:energy-mle  # (WebGPU parts use Deno; deno must be in PATH)",
    heavy: true,
  },

  // ── Cross-backend benchmark ─────────────────────────────────────────────
  {
    id: "bench-backends",
    label: "Cross-backend dlmFit benchmark",
    sources: [
      "src/index.ts",
      "src/dlmgensys.ts",
      "src/types.ts",
      "scripts/bench-backends.ts",
    ],
    outputs: [
      "assets/timings/bench-backends.json",
    ],
    command: "pnpm run bench:backends",
    heavy: false,
  },

  // ── MLE benchmark comparison table ──────────────────────────────────────
  {
    id: "bench-mle",
    label: "MLE benchmark comparison (Nile + Kaisaniemi)",
    sources: [
      "src/mle.ts",
      "src/index.ts",
      "src/dlmgensys.ts",
      "src/types.ts",
      "scripts/collect-mle-benchmark.ts",
    ],
    outputs: [
      "assets/timings/collect-mle-benchmark.json",
    ],
    command: "pnpm run bench:mle",
    heavy: false,
  },

  // ── Checkpoint benchmark ────────────────────────────────────────────────
  {
    id: "bench-checkpoint",
    label: "Checkpoint strategy benchmark",
    sources: [
      "src/mle.ts",
      "src/index.ts",
      "src/types.ts",
      "scripts/bench-checkpoint.ts",
    ],
    outputs: [
      "assets/timings/bench-checkpoint.json",
    ],
    command: "pnpm run bench:checkpoint",
    heavy: false,
  },

  // ── Full benchmark table (Deno + WebGPU) ────────────────────────────────
  {
    id: "bench-full",
    label: "Full benchmark table (all backends × algorithms)",
    sources: [
      "src/index.ts",
      "src/dlmgensys.ts",
      "src/types.ts",
      "scripts/bench-full.ts",
    ],
    outputs: [
      "assets/timings/bench-full.json",
    ],
    command: "pnpm run bench:full   # (uses Deno with --unstable-webgpu; deno must be in PATH)",
    heavy: true,
  },
];

// ── Algorithm coverage declarations ───────────────────────────────────────

export interface AlgorithmCoverageTarget {
  /** File to check (relative to repo root). */
  file: string;
  /** Human label for error messages. */
  label: string;
  /**
   * Regex to extract algorithm names from the file.
   * All capture groups from all matches are collected and compared
   * against the known algorithm list.
   */
  pattern: RegExp;
}

/**
 * The canonical list of algorithms is extracted from `src/types.ts`
 * at runtime (see `check-freshness.ts`).  This avoids duplicating
 * the type definition here.
 */
export const ALGORITHM_TYPE_FILE = "src/types.ts";
export const ALGORITHM_TYPE_REGEX = /export\s+type\s+DlmAlgorithm\s*=\s*([^;]+);/;

/**
 * Files where every known algorithm should appear.
 * The pattern should match string literals like 'scan' or "assoc".
 */
export const algorithmCoverageTargets: AlgorithmCoverageTarget[] = [
  {
    file: "scripts/bench-full.ts",
    label: "Full benchmark script (combo loop)",
    pattern: /['"]([a-z][\w-]*)['"](?:\s+as\s+const)/g,
  },
];

// ── Helpers ───────────────────────────────────────────────────────────────

/** Resolve a repo-relative path to absolute. */
export function abs(relPath: string): string {
  return resolve(ROOT, relPath);
}

/** Get mtime of a file, or null if it doesn't exist. */
export function mtimeMs(relPath: string): number | null {
  const p = abs(relPath);
  if (!existsSync(p)) return null;
  return statSync(p).mtimeMs;
}
