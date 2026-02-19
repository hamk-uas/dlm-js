/**
 * Timing slot registry.
 *
 * Each entry maps a slot key (used in `<!-- timing:KEY -->` HTML comments
 * inside .md files) to:
 *   - which script produces it (`script`, repo-root-relative path)
 *   - which field in that script's timing sidecar holds the raw value
 *   - how to format the value into a human-readable string
 *
 * Format codes
 * ────────────
 *  "ms2"  →  value.toFixed(2) + " ms"        e.g. "86.48 ms"
 *  "ms0"  →  Math.round(value) + " ms"        e.g. "2174 ms"
 *  "s1"   →  (value/1000).toFixed(1) + " s"   e.g. "5.9 s"
 *  "int"  →  String(Math.round(value))         e.g. "88"
 *  "pctdiff" → sign + Math.round(value) + "%" e.g. "+32%" or "-1%"
 *  "lik1" →  value.toFixed(1)                  e.g. "1105.0"
 *
 * To add a new timing slot:
 *   1. Add the entry here.
 *   2. Call writeTimingsSidecar() in the relevant script.
 *   3. Wrap the value in .md with <!-- timing:KEY -->VALUE<!-- /timing -->.
 *   4. Run `pnpm run update:timings`.
 */

export type TimingFormat = "ms2" | "ms0" | "s1" | "int" | "pctdiff" | "lik1";

export type TimingSlot = {
  /** Repo-root-relative path of the script that produces this timing. */
  script: string;
  /** Basename of the sidecar JSON file (no path, no extension). */
  sidecar: string;
  /** Key inside the sidecar JSON object. */
  field: string;
  /** How to render the raw numeric value as a string. */
  format: TimingFormat;
  /** One-line human description (appears in --list output). */
  description: string;
};

export const timingRegistry: Record<string, TimingSlot> = {
  // ── dlmFit benchmarks (gen-*.ts scripts) ────────────────────────────────

  "nile-demo:first": {
    script:      "scripts/gen-niledemo-svg.ts",
    sidecar:     "gen-niledemo-svg",
    field:       "firstRunMs",
    format:      "ms2",
    description: "dlmFit first-run wall-clock — Nile demo",
  },
  "nile-demo:warm": {
    script:      "scripts/gen-niledemo-svg.ts",
    sidecar:     "gen-niledemo-svg",
    field:       "warmRunMs",
    format:      "ms2",
    description: "dlmFit warm-run wall-clock — Nile demo",
  },

  "kaisaniemi:first": {
    script:      "scripts/gen-kaisaniemi-svg.ts",
    sidecar:     "gen-kaisaniemi-svg",
    field:       "firstRunMs",
    format:      "ms2",
    description: "dlmFit first-run wall-clock — Kaisaniemi demo",
  },
  "kaisaniemi:warm": {
    script:      "scripts/gen-kaisaniemi-svg.ts",
    sidecar:     "gen-kaisaniemi-svg",
    field:       "warmRunMs",
    format:      "ms2",
    description: "dlmFit warm-run wall-clock — Kaisaniemi demo",
  },

  "trigar:first": {
    script:      "scripts/gen-trigar-svg.ts",
    sidecar:     "gen-trigar-svg",
    field:       "firstRunMs",
    format:      "ms2",
    description: "dlmFit first-run wall-clock — energy/trigar demo",
  },
  "trigar:warm": {
    script:      "scripts/gen-trigar-svg.ts",
    sidecar:     "gen-trigar-svg",
    field:       "warmRunMs",
    format:      "ms2",
    description: "dlmFit warm-run wall-clock — energy/trigar demo",
  },

  "missing:first": {
    script:      "scripts/gen-missing-svg.ts",
    sidecar:     "gen-missing-svg",
    field:       "firstRunMs",
    format:      "ms2",
    description: "dlmFit first-run wall-clock — missing-data demo",
  },
  "missing:warm": {
    script:      "scripts/gen-missing-svg.ts",
    sidecar:     "gen-missing-svg",
    field:       "warmRunMs",
    format:      "ms2",
    description: "dlmFit warm-run wall-clock — missing-data demo",
  },

  // ── dlmMLE benchmarks (collect-*.ts scripts) ────────────────────────────

  "nile-mle:elapsed": {
    script:      "scripts/collect-nile-mle-frames.ts",
    sidecar:     "collect-nile-mle-frames",
    field:       "elapsed",
    format:      "ms0",
    description: "dlmMLE total wall-clock — Nile model (order=1, fit s+w)",
  },
  "nile-mle:iterations": {
    script:      "scripts/collect-nile-mle-frames.ts",
    sidecar:     "collect-nile-mle-frames",
    field:       "iterations",
    format:      "int",
    description: "dlmMLE iteration count — Nile model",
  },
  "nile-mle:lik": {
    script:      "scripts/collect-nile-mle-frames.ts",
    sidecar:     "collect-nile-mle-frames",
    field:       "lik",
    format:      "lik1",
    description: "dlmMLE final −2log L — Nile order=1",
  },

  "energy-mle:elapsed": {
    script:      "scripts/collect-energy-mle-frames.ts",
    sidecar:     "collect-energy-mle-frames",
    field:       "elapsed",
    format:      "s1",
    description: "dlmMLE total wall-clock — energy/AR model (seconds, for prose)",
  },
  "energy-mle:elapsed-ms": {
    script:      "scripts/collect-energy-mle-frames.ts",
    sidecar:     "collect-energy-mle-frames",
    field:       "elapsed",
    format:      "ms0",
    description: "dlmMLE total wall-clock — energy/AR model (ms, for benchmark table)",
  },
  "energy-mle:iterations": {
    script:      "scripts/collect-energy-mle-frames.ts",
    sidecar:     "collect-energy-mle-frames",
    field:       "iterations",
    format:      "int",
    description: "dlmMLE iteration count — energy/AR model",
  },
  "energy-mle:lik": {
    script:      "scripts/collect-energy-mle-frames.ts",
    sidecar:     "collect-energy-mle-frames",
    field:       "lik",
    format:      "lik1",
    description: "dlmMLE final −2log L — energy/AR model",
  },

  // ── MLE comparison-table benchmark (collect-mle-benchmark.ts) ──────────

  "mle-bench:nile-order1:elapsed": {
    script:      "scripts/collect-mle-benchmark.ts",
    sidecar:     "collect-mle-benchmark",
    field:       "nile_order1_elapsed",
    format:      "ms0",
    description: "dlmMLE wall-clock — Nile order=1 (s+w), benchmark run",
  },
  "mle-bench:nile-order1:iterations": {
    script:      "scripts/collect-mle-benchmark.ts",
    sidecar:     "collect-mle-benchmark",
    field:       "nile_order1_iterations",
    format:      "int",
    description: "dlmMLE iteration count — Nile order=1 (s+w)",
  },
  "mle-bench:nile-order1:lik": {
    script:      "scripts/collect-mle-benchmark.ts",
    sidecar:     "collect-mle-benchmark",
    field:       "nile_order1_lik",
    format:      "lik1",
    description: "dlmMLE final −2log L — Nile order=1 (s+w)",
  },

  "mle-bench:nile-order0:elapsed": {
    script:      "scripts/collect-mle-benchmark.ts",
    sidecar:     "collect-mle-benchmark",
    field:       "nile_order0_elapsed",
    format:      "ms0",
    description: "dlmMLE wall-clock — Nile order=0, fit s+w",
  },
  "mle-bench:nile-order0:iterations": {
    script:      "scripts/collect-mle-benchmark.ts",
    sidecar:     "collect-mle-benchmark",
    field:       "nile_order0_iterations",
    format:      "int",
    description: "dlmMLE iteration count — Nile order=0",
  },
  "mle-bench:nile-order0:lik": {
    script:      "scripts/collect-mle-benchmark.ts",
    sidecar:     "collect-mle-benchmark",
    field:       "nile_order0_lik",
    format:      "lik1",
    description: "dlmMLE final −2log L — Nile order=0",
  },

  "mle-bench:kaisaniemi:elapsed": {
    script:      "scripts/collect-mle-benchmark.ts",
    sidecar:     "collect-mle-benchmark",
    field:       "kaisaniemi_elapsed",
    format:      "ms0",
    description: "dlmMLE wall-clock — Kaisaniemi trig model, fit s+w",
  },
  "mle-bench:kaisaniemi:elapsed-s": {
    script:      "scripts/collect-mle-benchmark.ts",
    sidecar:     "collect-mle-benchmark",
    field:       "kaisaniemi_elapsed",
    format:      "s1",
    description: "dlmMLE wall-clock — Kaisaniemi (seconds, for prose)",
  },
  "mle-bench:kaisaniemi:iterations": {
    script:      "scripts/collect-mle-benchmark.ts",
    sidecar:     "collect-mle-benchmark",
    field:       "kaisaniemi_iterations",
    format:      "int",
    description: "dlmMLE iteration count — Kaisaniemi trig model",
  },
  "mle-bench:kaisaniemi:lik": {
    script:      "scripts/collect-mle-benchmark.ts",
    sidecar:     "collect-mle-benchmark",
    field:       "kaisaniemi_lik",
    format:      "lik1",
    description: "dlmMLE final −2log L — Kaisaniemi trig model",
  },

  // ── Checkpoint benchmark (bench-checkpoint.ts) ──────────────────────────

  "ckpt:nile:false-ms": {
    script:      "scripts/bench-checkpoint.ts",
    sidecar:     "bench-checkpoint",
    field:       "nile_false_ms",
    format:      "ms0",
    description: "Checkpoint bench: Nile, checkpoint:false, mean ms (60 iters)",
  },
  "ckpt:nile:false-s": {
    script:      "scripts/bench-checkpoint.ts",
    sidecar:     "bench-checkpoint",
    field:       "nile_false_ms",
    format:      "s1",
    description: "Checkpoint bench: Nile, checkpoint:false, mean seconds (60 iters)",
  },
  "ckpt:nile:true-ms": {
    script:      "scripts/bench-checkpoint.ts",
    sidecar:     "bench-checkpoint",
    field:       "nile_true_ms",
    format:      "ms0",
    description: "Checkpoint bench: Nile, checkpoint:true, mean ms (60 iters)",
  },
  "ckpt:nile:speedup": {
    script:      "scripts/bench-checkpoint.ts",
    sidecar:     "bench-checkpoint",
    field:       "nile_speedup_pct",
    format:      "pctdiff",
    description: "Checkpoint bench: Nile speedup % (checkpoint:true vs false)",
  },
  "ckpt:energy:false-ms": {
    script:      "scripts/bench-checkpoint.ts",
    sidecar:     "bench-checkpoint",
    field:       "energy_false_ms",
    format:      "ms0",
    description: "Checkpoint bench: Energy, checkpoint:false, mean ms (60 iters)",
  },
  "ckpt:energy:true-ms": {
    script:      "scripts/bench-checkpoint.ts",
    sidecar:     "bench-checkpoint",
    field:       "energy_true_ms",
    format:      "ms0",
    description: "Checkpoint bench: Energy, checkpoint:true, mean ms (60 iters)",
  },
  "ckpt:energy:speedup": {
    script:      "scripts/bench-checkpoint.ts",
    sidecar:     "bench-checkpoint",
    field:       "energy_speedup_pct",
    format:      "pctdiff",
    description: "Checkpoint bench: Energy speedup % (checkpoint:true vs false)",
  },
};

/** Format a raw numeric timing value according to an entry's format code. */
export function formatTiming(value: number, format: TimingFormat): string {
  switch (format) {
    case "ms2": return `${value.toFixed(2)} ms`;
    case "ms0": return `${Math.round(value)} ms`;
    case "s1":  return `${(value / 1000).toFixed(1)} s`;
    case "int": return String(Math.round(value));
    case "pctdiff": return `${value >= 0 ? "+" : ""}${Math.round(value)}%`;
    case "lik1":    return value.toFixed(1);
  }
}
