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

  // ── dlmMLE assocScan variant benchmarks (collect-*.ts, forceAssocScan) ──

  "nile-mle-assoc:elapsed": {
    script:      "scripts/collect-nile-mle-frames.ts",
    sidecar:     "collect-nile-mle-frames-assoc",
    field:       "elapsed",
    format:      "ms0",
    description: "dlmMLE total wall-clock — Nile model, assocScan variant",
  },
  "nile-mle-assoc:iterations": {
    script:      "scripts/collect-nile-mle-frames.ts",
    sidecar:     "collect-nile-mle-frames-assoc",
    field:       "iterations",
    format:      "int",
    description: "dlmMLE iteration count — Nile model, assocScan variant",
  },
  "nile-mle-assoc:lik": {
    script:      "scripts/collect-nile-mle-frames.ts",
    sidecar:     "collect-nile-mle-frames-assoc",
    field:       "lik",
    format:      "lik1",
    description: "dlmMLE final −2log L — Nile order=1, assocScan variant",
  },

  "energy-mle-assoc:elapsed": {
    script:      "scripts/collect-energy-mle-frames.ts",
    sidecar:     "collect-energy-mle-frames-assoc",
    field:       "elapsed",
    format:      "s1",
    description: "dlmMLE total wall-clock — energy/AR model, assocScan variant (seconds)",
  },
  "energy-mle-assoc:elapsed-ms": {
    script:      "scripts/collect-energy-mle-frames.ts",
    sidecar:     "collect-energy-mle-frames-assoc",
    field:       "elapsed",
    format:      "ms0",
    description: "dlmMLE total wall-clock — energy/AR model, assocScan variant (ms)",
  },
  "energy-mle-assoc:iterations": {
    script:      "scripts/collect-energy-mle-frames.ts",
    sidecar:     "collect-energy-mle-frames-assoc",
    field:       "iterations",
    format:      "int",
    description: "dlmMLE iteration count — energy/AR model, assocScan variant",
  },
  "energy-mle-assoc:lik": {
    script:      "scripts/collect-energy-mle-frames.ts",
    sidecar:     "collect-energy-mle-frames-assoc",
    field:       "lik",
    format:      "lik1",
    description: "dlmMLE final −2log L — energy/AR model, assocScan variant",
  },

  // ── dlmMLE WebGPU variant benchmarks (collect-*-webgpu.ts, Float32) ──────

  "nile-mle-webgpu:elapsed": {
    script:      "scripts/collect-nile-mle-frames-webgpu.ts",
    sidecar:     "collect-nile-mle-frames-webgpu",
    field:       "elapsed",
    format:      "ms0",
    description: "dlmMLE total wall-clock — Nile model, WebGPU Float32 variant",
  },
  "nile-mle-webgpu:iterations": {
    script:      "scripts/collect-nile-mle-frames-webgpu.ts",
    sidecar:     "collect-nile-mle-frames-webgpu",
    field:       "iterations",
    format:      "int",
    description: "dlmMLE iteration count — Nile model, WebGPU Float32 variant",
  },
  "nile-mle-webgpu:lik": {
    script:      "scripts/collect-nile-mle-frames-webgpu.ts",
    sidecar:     "collect-nile-mle-frames-webgpu",
    field:       "lik",
    format:      "lik1",
    description: "dlmMLE final −2log L — Nile order=1, WebGPU Float32 variant",
  },

  "energy-mle-webgpu:elapsed": {
    script:      "scripts/collect-energy-mle-frames-webgpu.ts",
    sidecar:     "collect-energy-mle-frames-webgpu",
    field:       "elapsed",
    format:      "s1",
    description: "dlmMLE total wall-clock — energy/AR model, WebGPU Float32 variant (seconds)",
  },
  "energy-mle-webgpu:elapsed-ms": {
    script:      "scripts/collect-energy-mle-frames-webgpu.ts",
    sidecar:     "collect-energy-mle-frames-webgpu",
    field:       "elapsed",
    format:      "ms0",
    description: "dlmMLE total wall-clock — energy/AR model, WebGPU Float32 variant (ms)",
  },
  "energy-mle-webgpu:iterations": {
    script:      "scripts/collect-energy-mle-frames-webgpu.ts",
    sidecar:     "collect-energy-mle-frames-webgpu",
    field:       "iterations",
    format:      "int",
    description: "dlmMLE iteration count — energy/AR model, WebGPU Float32 variant",
  },
  "energy-mle-webgpu:lik": {
    script:      "scripts/collect-energy-mle-frames-webgpu.ts",
    sidecar:     "collect-energy-mle-frames-webgpu",
    field:       "lik",
    format:      "lik1",
    description: "dlmMLE final −2log L — energy/AR model, WebGPU Float32 variant",
  },

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

  // ── Cross-backend dlmFit benchmark (bench-backends.ts) ───────────────────

  "bb:nile-o0:cpu-f32":   { script: "scripts/bench-backends.ts", sidecar: "bench-backends", field: "nile_o0__cpu_f32__warm",       format: "ms0", description: "dlmFit warm — Nile order=0, cpu/f32" },
  "bb:nile-o0:wasm-f32":  { script: "scripts/bench-backends.ts", sidecar: "bench-backends", field: "nile_o0__wasm_f32__warm",      format: "ms0", description: "dlmFit warm — Nile order=0, wasm/f32" },
  "bb:nile-o0:wasm-f64":  { script: "scripts/bench-backends.ts", sidecar: "bench-backends", field: "nile_o0__wasm_f64__warm",      format: "ms0", description: "dlmFit warm — Nile order=0, wasm/f64" },

  "bb:nile-o1:cpu-f32":   { script: "scripts/bench-backends.ts", sidecar: "bench-backends", field: "nile_o1__cpu_f32__warm",       format: "ms0", description: "dlmFit warm — Nile order=1, cpu/f32" },
  "bb:nile-o1:wasm-f32":  { script: "scripts/bench-backends.ts", sidecar: "bench-backends", field: "nile_o1__wasm_f32__warm",      format: "ms0", description: "dlmFit warm — Nile order=1, wasm/f32" },
  "bb:nile-o1:wasm-f64":  { script: "scripts/bench-backends.ts", sidecar: "bench-backends", field: "nile_o1__wasm_f64__warm",      format: "ms0", description: "dlmFit warm — Nile order=1, wasm/f64" },

  "bb:kaisaniemi:cpu-f32":  { script: "scripts/bench-backends.ts", sidecar: "bench-backends", field: "kaisaniemi__cpu_f32__warm",   format: "ms0", description: "dlmFit warm — Kaisaniemi trig, cpu/f32" },
  "bb:kaisaniemi:wasm-f32": { script: "scripts/bench-backends.ts", sidecar: "bench-backends", field: "kaisaniemi__wasm_f32__warm",  format: "ms0", description: "dlmFit warm — Kaisaniemi trig, wasm/f32" },
  "bb:kaisaniemi:wasm-f64": { script: "scripts/bench-backends.ts", sidecar: "bench-backends", field: "kaisaniemi__wasm_f64__warm",  format: "ms0", description: "dlmFit warm — Kaisaniemi trig, wasm/f64" },

  "bb:trigar:cpu-f32":  { script: "scripts/bench-backends.ts", sidecar: "bench-backends", field: "trigar__cpu_f32__warm",   format: "ms0", description: "dlmFit warm — Energy trig+AR, cpu/f32" },
  "bb:trigar:wasm-f32": { script: "scripts/bench-backends.ts", sidecar: "bench-backends", field: "trigar__wasm_f32__warm",  format: "ms0", description: "dlmFit warm — Energy trig+AR, wasm/f32" },
  "bb:trigar:wasm-f64": { script: "scripts/bench-backends.ts", sidecar: "bench-backends", field: "trigar__wasm_f64__warm",  format: "ms0", description: "dlmFit warm — Energy trig+AR, wasm/f64" },

  // ── WebGPU dlmFit benchmark (bench-gpu.ts, Deno) ─────────────────────────

  "bb:nile-o0:webgpu-f32":    { script: "scripts/bench-gpu.ts", sidecar: "bench-gpu", field: "nile_o0__webgpu_f32__warm",      format: "ms0", description: "dlmFit warm — Nile order=0, webgpu/f32" },
  "bb:nile-o1:webgpu-f32":    { script: "scripts/bench-gpu.ts", sidecar: "bench-gpu", field: "nile_o1__webgpu_f32__warm",      format: "ms0", description: "dlmFit warm — Nile order=1, webgpu/f32" },
  "bb:kaisaniemi:webgpu-f32": { script: "scripts/bench-gpu.ts", sidecar: "bench-gpu", field: "kaisaniemi__webgpu_f32__warm",   format: "ms0", description: "dlmFit warm — Kaisaniemi trig, webgpu/f32" },
  "bb:trigar:webgpu-f32":     { script: "scripts/bench-gpu.ts", sidecar: "bench-gpu", field: "trigar__webgpu_f32__warm",       format: "ms0", description: "dlmFit warm — Energy trig+AR, webgpu/f32" },

  // ── Backend scaling benchmark (bench-scaling.ts, Deno) ──────────────────
  // WASM/f64 at N=100..102400; WebGPU/f32 at N=100..1600 only.
  // Data: Nile order=1 (m=2) tiled to each N. Warmup=2, Runs=4, median.

  "scale:wasm-f64:n100":    { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "wasm_f64_n100",    format: "ms0", description: "WASM/f64 dlmFit warm — Nile/order=1 tiled to N=100" },
  "scale:wasm-f64:n200":    { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "wasm_f64_n200",    format: "ms0", description: "WASM/f64 dlmFit warm — Nile/order=1 tiled to N=200" },
  "scale:wasm-f64:n400":    { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "wasm_f64_n400",    format: "ms0", description: "WASM/f64 dlmFit warm — Nile/order=1 tiled to N=400" },
  "scale:wasm-f64:n800":    { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "wasm_f64_n800",    format: "ms0", description: "WASM/f64 dlmFit warm — Nile/order=1 tiled to N=800" },
  "scale:wasm-f64:n1600":   { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "wasm_f64_n1600",   format: "ms0", description: "WASM/f64 dlmFit warm — Nile/order=1 tiled to N=1600" },
  "scale:wasm-f64:n3200":   { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "wasm_f64_n3200",   format: "ms0", description: "WASM/f64 dlmFit warm — Nile/order=1 tiled to N=3200" },
  "scale:wasm-f64:n6400":   { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "wasm_f64_n6400",   format: "ms0", description: "WASM/f64 dlmFit warm — Nile/order=1 tiled to N=6400" },
  "scale:wasm-f64:n12800":  { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "wasm_f64_n12800",  format: "ms0", description: "WASM/f64 dlmFit warm — Nile/order=1 tiled to N=12800" },
  "scale:wasm-f64:n25600":  { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "wasm_f64_n25600",  format: "ms0", description: "WASM/f64 dlmFit warm — Nile/order=1 tiled to N=25600" },
  "scale:wasm-f64:n51200":  { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "wasm_f64_n51200",  format: "ms0", description: "WASM/f64 dlmFit warm — Nile/order=1 tiled to N=51200" },
  "scale:wasm-f64:n102400": { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "wasm_f64_n102400", format: "ms0", description: "WASM/f64 dlmFit warm — Nile/order=1 tiled to N=102400" },

  "scale:webgpu-f32:n100":  { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "webgpu_f32_n100",  format: "ms0", description: "WebGPU/f32 dlmFit warm — Nile/order=1 tiled to N=100" },
  "scale:webgpu-f32:n200":  { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "webgpu_f32_n200",  format: "ms0", description: "WebGPU/f32 dlmFit warm — Nile/order=1 tiled to N=200" },
  "scale:webgpu-f32:n400":  { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "webgpu_f32_n400",  format: "ms0", description: "WebGPU/f32 dlmFit warm — Nile/order=1 tiled to N=400" },
  "scale:webgpu-f32:n800":  { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "webgpu_f32_n800",  format: "ms0", description: "WebGPU/f32 dlmFit warm — Nile/order=1 tiled to N=800" },
  "scale:webgpu-f32:n1600": { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "webgpu_f32_n1600", format: "ms0", description: "WebGPU/f32 dlmFit warm — Nile/order=1 tiled to N=1600" },
  "scale:webgpu-f32:n3200": { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "webgpu_f32_n3200", format: "ms0", description: "WebGPU/f32 dlmFit warm — Nile/order=1 tiled to N=3200" },
  "scale:webgpu-f32:n6400": { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "webgpu_f32_n6400", format: "ms0", description: "WebGPU/f32 dlmFit warm — Nile/order=1 tiled to N=6400" },
  "scale:webgpu-f32:n12800":  { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "webgpu_f32_n12800",  format: "ms0", description: "WebGPU/f32 dlmFit warm — Nile/order=1 tiled to N=12800" },
  "scale:webgpu-f32:n25600":  { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "webgpu_f32_n25600",  format: "ms0", description: "WebGPU/f32 dlmFit warm — Nile/order=1 tiled to N=25600" },
  "scale:webgpu-f32:n51200":  { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "webgpu_f32_n51200",  format: "ms0", description: "WebGPU/f32 dlmFit warm — Nile/order=1 tiled to N=51200" },
  "scale:webgpu-f32:n102400": { script: "scripts/bench-scaling.ts", sidecar: "bench-scaling", field: "webgpu_f32_n102400", format: "ms0", description: "WebGPU/f32 dlmFit warm — Nile/order=1 tiled to N=102400" },

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
