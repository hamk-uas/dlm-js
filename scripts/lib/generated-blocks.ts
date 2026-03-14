/**
 * Generated block registry.
 *
 * Each entry maps a block ID (used in `<!-- generated:ID -->...<!-- /generated -->`
 * markers in .md files) to a generator function that reads sidecar data and
 * returns the replacement markdown content.
 *
 * Adding a new generated block:
 *   1. Add a generator function below.
 *   2. Register it in `generatedBlocks`.
 *   3. Wrap the placeholder content in .md with `<!-- generated:ID -->...<!-- /generated -->`.
 *   4. Run `pnpm run update:timings`.
 */

import { readFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";

const root = resolve(dirname(new URL(import.meta.url).pathname), "../..");
const sidecarDir = resolve(root, "assets/timings");

// ── Helpers ────────────────────────────────────────────────────────────────

function readJson<T>(name: string): T | null {
  const p = resolve(sidecarDir, name);
  if (!existsSync(p)) return null;
  return JSON.parse(readFileSync(p, "utf8")) as T;
}

function fmtMs(ms: number): string {
  if (isNaN(ms)) return "crash";
  if (!isFinite(ms)) return ">5 s";
  return `${Math.round(ms)} ms`;
}

function fmtErr(v: number): string {
  if (!isFinite(v) || v === 0) return "0";
  // Use exponential for values outside [0.01, 999]
  if (v >= 0.01 && v < 1000) {
    if (v >= 100) return v.toFixed(0);
    if (v >= 10) return v.toFixed(1);
    return v.toFixed(2);
  }
  return v.toExponential(2);
}

/** Format relative error (sidecar stores percentage; divide by 100). */
function fmtRelErr(pct: number): string {
  if (!isFinite(pct)) return "—";
  if (pct === 0) return "0";
  const rel = pct / 100;
  return fmtErr(rel);
}

// ── bench-full-table ──────────────────────────────────────────────────────

interface BenchFullResult {
  model: string;
  n: number;
  m: number;
  backend: string;
  dtype: string;
  algorithm: string;
  stabLabel: string;
  firstMs: number;
  warmMs: number;
  stable: boolean;
  maxAbsErr: number;
  maxPctErr: number;
}

/** Row specification: which combos to show and in what order. */
interface RowSpec {
  backend: string;
  dtype: string;
  algorithm: string;
  stabLabel: string;
}

/** Display stab label derived from the combo. */
function stabDisplay(r: RowSpec): string {
  if (r.stabLabel === "off") return "off";
  if (r.stabLabel === "joseph+triu") return "joseph+triu";
  if (r.algorithm === "assoc" || r.algorithm === "sqrt-assoc" || r.algorithm === "ud") return "built-in";
  if (r.dtype === "f64") return "triu";
  return "joseph";
}

/** Whether this row is the auto-selected default for its backend×dtype. */
function isDefault(r: RowSpec): boolean {
  if (r.stabLabel !== "default") return false;
  if (r.backend === "webgpu") return r.algorithm === "assoc";
  return r.algorithm === "scan";
}

// Model column order (short labels for the header).
const MODEL_ORDER = [
  "Nile, order=0",
  "Nile, order=1",
  "Kaisaniemi, trig",
  "Energy, trig+AR",
  "Gapped, order=1",
] as const;

const MODEL_HEADERS = ["Nile o=0", "Nile o=1", "Kaisaniemi", "Energy", "Gapped"];

// Row order matching the current README layout (excludes cpu/sqrt-assoc
// which has very large errors due to JS interpreter numerical behaviour).
const ROW_SPECS: RowSpec[] = [
  // cpu f64
  { backend: "cpu", dtype: "f64", algorithm: "scan",  stabLabel: "default" },
  { backend: "cpu", dtype: "f64", algorithm: "scan",  stabLabel: "off" },
  { backend: "cpu", dtype: "f64", algorithm: "assoc", stabLabel: "default" },
  { backend: "cpu", dtype: "f64", algorithm: "ud",    stabLabel: "default" },
  // cpu f32
  { backend: "cpu", dtype: "f32", algorithm: "scan",  stabLabel: "default" },
  { backend: "cpu", dtype: "f32", algorithm: "scan",  stabLabel: "joseph+triu" },
  { backend: "cpu", dtype: "f32", algorithm: "assoc", stabLabel: "default" },
  { backend: "cpu", dtype: "f32", algorithm: "ud",    stabLabel: "default" },
  // wasm f64
  { backend: "wasm", dtype: "f64", algorithm: "scan",       stabLabel: "default" },
  { backend: "wasm", dtype: "f64", algorithm: "scan",       stabLabel: "off" },
  { backend: "wasm", dtype: "f64", algorithm: "assoc",      stabLabel: "default" },
  { backend: "wasm", dtype: "f64", algorithm: "sqrt-assoc", stabLabel: "default" },
  { backend: "wasm", dtype: "f64", algorithm: "ud",         stabLabel: "default" },
  // wasm f32
  { backend: "wasm", dtype: "f32", algorithm: "scan",       stabLabel: "default" },
  { backend: "wasm", dtype: "f32", algorithm: "scan",       stabLabel: "joseph+triu" },
  { backend: "wasm", dtype: "f32", algorithm: "assoc",      stabLabel: "default" },
  { backend: "wasm", dtype: "f32", algorithm: "sqrt-assoc", stabLabel: "default" },
  { backend: "wasm", dtype: "f32", algorithm: "ud",         stabLabel: "default" },
  // webgpu f32
  { backend: "webgpu", dtype: "f32", algorithm: "assoc", stabLabel: "default" },
  { backend: "webgpu", dtype: "f32", algorithm: "scan",  stabLabel: "default" },
  { backend: "webgpu", dtype: "f32", algorithm: "scan",  stabLabel: "joseph+triu" },
  { backend: "webgpu", dtype: "f32", algorithm: "ud",    stabLabel: "default" },
];

function generateBenchFullTable(): string {
  const data = readJson<{ results: BenchFullResult[] }>("bench-full.json");
  if (!data) return "(bench-full.json sidecar not found — run `pnpm run bench:full`)";

  const { results } = data;

  // Index: (backend, dtype, algorithm, stabLabel, model) → result
  const idx = new Map<string, BenchFullResult>();
  for (const r of results) {
    idx.set(`${r.backend}|${r.dtype}|${r.algorithm}|${r.stabLabel}|${r.model}`, r);
  }

  // Build table
  const lines: string[] = [];

  // Header: backend | dtype | algorithm | stab | model1 | Δ% | model2 | Δ% | ...
  const headerCells = MODEL_HEADERS.flatMap(h => [`${h}`, "rel err"]);
  lines.push(
    `| backend | dtype | algorithm | stab | ${headerCells.join(" | ")} |`,
  );
  const sepCells = MODEL_HEADERS.flatMap(() => ["-------", "------"]);
  lines.push(
    `|---------|-------|-----------|------|${sepCells.join("|")}|`,
  );

  let prevBackend = "";
  let prevDtype = "";

  for (const spec of ROW_SPECS) {
    const def = isDefault(spec);
    const b = (s: string) => def ? `**${s}**` : s;

    // Show backend/dtype only on first row of each group
    const showBackend = spec.backend !== prevBackend;
    const showDtype = showBackend || spec.dtype !== prevDtype;
    prevBackend = spec.backend;
    prevDtype = spec.dtype;

    const backendCell = showBackend ? b(spec.backend) : "";
    const dtypeCell = showDtype ? b(spec.dtype) : "";
    const algoCell = b(spec.algorithm);
    const stabCell = b(stabDisplay(spec));

    // Per-model: warm time cell + Δ% cell
    const modelCells: string[] = [];

    for (const model of MODEL_ORDER) {
      const key = `${spec.backend}|${spec.dtype}|${spec.algorithm}|${spec.stabLabel}|${model}`;
      const r = idx.get(key);
      if (!r) {
        modelCells.push(b("—"), b("—"));
      } else if (r.warmMs == null) {
        // Timed out (firstMs > TIMEOUT_MS, no warm run completed)
        modelCells.push(b(">10 s"), b("—"));
      } else if (!r.stable) {
        modelCells.push(b(fmtMs(r.warmMs)), b("NaN"));
      } else {
        modelCells.push(b(fmtMs(r.warmMs)), b(fmtRelErr(r.maxPctErr)));
      }
    }

    lines.push(`| ${backendCell} | ${dtypeCell} | ${algoCell} | ${stabCell} | ${modelCells.join(" | ")} |`);
  }

  return lines.join("\n");
}

// ── Registry ──────────────────────────────────────────────────────────────

export type BlockGenerator = () => string;

export const generatedBlocks: Record<string, BlockGenerator> = {
  "bench-full-table": generateBenchFullTable,
};
