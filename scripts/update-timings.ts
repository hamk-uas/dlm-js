/**
 * update-timings.ts
 *
 * Reads timing sidecar files written by gen-*.ts / collect-*.ts scripts and
 * patches every `<!-- timing:KEY -->VALUE<!-- /timing -->` marker found in
 * .md files throughout the repo.
 *
 * Usage
 * ─────
 *   pnpm run update:timings            # patch md files from existing sidecars
 *   pnpm run update:timings -- --list  # show all known timing slots
 *   pnpm run update:timings -- --dry   # show what would change without writing
 *
 * Workflow
 * ────────
 *  1. Run any gen-* / collect-* script — it writes assets/timings/<name>.json.
 *  2. Run this script to push the fresh values into the .md files.
 *
 * Marker syntax (invisible in rendered Markdown)
 * ───────────────────────────────────────────────
 *   <!-- timing:KEY -->current value<!-- /timing -->
 *
 *     Replaced with formatTiming(sidecar[field]) for the registered KEY.
 *
 *   <!-- computed:EXPR -->current value<!-- /computed -->
 *
 *     EXPR is a JS expression using two helpers:
 *       slot("KEY")    → raw number from the named timing-registry slot's sidecar
 *       static("KEY")  → value from assets/timings/static-references.json
 *
 *     Examples:
 *       static("octave-nile-order1-elapsed-ms") < slot("nile-mle:elapsed") ? "faster" : "slower"
 *       Math.abs(slot("mle-bench:nile-order1:lik") - static("octave-nile-order1-lik")).toFixed(1)
 */

import { readFileSync, writeFileSync, readdirSync, statSync } from "node:fs";
import { resolve, dirname, relative } from "node:path";
import { timingRegistry, formatTiming } from "./lib/timing-registry.ts";
import { readTimingsSidecar } from "./lib/timing-sidecar.ts";
import { generatedBlocks } from "./lib/generated-blocks.ts";

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const args = process.argv.slice(2);
const DRY   = args.includes("--dry");
const LIST  = args.includes("--list");
const CHECK = args.includes("--check");

// ── --list ──────────────────────────────────────────────────────────────────

if (LIST) {
  console.log("Known timing slots:\n");
  const maxKey = Math.max(...Object.keys(timingRegistry).map(k => k.length));
  for (const [key, slot] of Object.entries(timingRegistry)) {
    const sidecar = readTimingsSidecar(slot.sidecar);
    const current = sidecar?.[slot.field] != null
      ? formatTiming(sidecar[slot.field], slot.format, slot.warmth)
      : "(no sidecar yet — run the script to generate it)";
    console.log(`  ${key.padEnd(maxKey + 2)} ${current}`);
    console.log(`  ${"".padEnd(maxKey + 2)} source: ${slot.script}`);
    console.log();
  }
  process.exit(0);
}

// ── Build value map from available sidecars ─────────────────────────────────

const valueMap = new Map<string, number>();    // key → raw numeric value
const gapped: string[] = [];

for (const [key, slot] of Object.entries(timingRegistry)) {
  const sidecar = readTimingsSidecar(slot.sidecar);
  if (sidecar == null || sidecar[slot.field] == null) {
    gapped.push(key);
    continue;
  }
  valueMap.set(key, sidecar[slot.field] as number);
}

if (gapped.length > 0) {
  console.warn(
    `[update-timings] No sidecar data for ${gapped.length} slot(s) — will skip:\n` +
    gapped.map(k => `  ${k}  (run: ${timingRegistry[k].script})`).join("\n")
  );
}

// ── Load static references ─────────────────────────────────────────────────
//    File is optional; missing or extra fields are silently ignored.

const staticRefs: Record<string, number | string> = {};
try {
  const raw = JSON.parse(
    readFileSync(resolve(root, "assets/timings/static-references.json"), "utf8")
  ) as Record<string, unknown>;
  for (const [k, v] of Object.entries(raw)) {
    if (!k.startsWith("_") && (typeof v === "number" || typeof v === "string")) {
      staticRefs[k] = v as number | string;
    }
  }
} catch { /* optional */ }

// ── Evaluate a <!-- computed:EXPR --> expression ───────────────────────────
//
// Expression syntax (standard JS with two helper functions):
//   slot("KEY")   → raw numeric value from a named timing-registry slot
//   static("KEY") → value from assets/timings/static-references.json
//
// Examples:
//   slot("nile-mle:elapsed") < static("octave-nile-order1-elapsed-ms") ? "faster" : "slower"
//   Math.abs(slot("mle-bench:nile-order1:lik") - static("octave-nile-order1-lik")).toFixed(1)

function evalComputed(expr: string): string {
  const slotFn = (key: string): number => {
    const v = valueMap.get(key);
    if (v == null) throw new Error(`slot("${key}") — no value in sidecars`);
    return v;
  };
  const staticFn = (key: string): number | string => {
    const v = staticRefs[key];
    if (v == null) throw new Error(`static("${key}") — not in static-references.json`);
    return v;
  };
  // "static" is a reserved word; rename to _s in the generated function body
  const safeExpr = expr.replace(/\bstatic\(/g, "_s(");
  // eslint-disable-next-line no-new-func
  return String(new Function("slot", "_s", `return (${safeExpr})`)(slotFn, staticFn));
}

// ── Find .md files ──────────────────────────────────────────────────────────

function walkMd(dir: string, results: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    if (name.startsWith(".") || name === "node_modules") continue;
    const full = resolve(dir, name);
    const st = statSync(full);
    if (st.isDirectory()) walkMd(full, results);
    else if (name.endsWith(".md")) results.push(full);
  }
  return results;
}

const mdFiles = walkMd(root);

// ── Patch each file ─────────────────────────────────────────────────────────

// Matches: <!-- timing:KEY -->anything<!-- /timing -->
// Capture group 1 = KEY, group 2 = current value
const MARKER_RE = /<!-- timing:([^>]+) -->([^<]*)<!-- \/timing -->/g;

let totalPatched = 0;

for (const filePath of mdFiles) {
  const original = readFileSync(filePath, "utf8");
  let updated = original;
  let filePatched = 0;

  updated = updated.replace(MARKER_RE, (match, key: string, _oldVal: string, offset: number) => {
    const trimmedKey = key.trim();
    if (!(trimmedKey in timingRegistry)) {
      // Key is not registered at all — warn
      console.warn(`[update-timings] Unknown timing key "${trimmedKey}" in ${relative(root, filePath)}`);
      return match;
    }
    const rawVal = valueMap.get(trimmedKey);
    if (rawVal == null) {
      // Registered but no sidecar yet — leave unchanged (already warned above)
      return match;
    }
    const slot = timingRegistry[trimmedKey];
    // Detect table context: suppress warmth annotation inside Markdown table
    // rows (line starts with |) — warmth goes in the column heading instead.
    const lineStart = updated.lastIndexOf('\n', offset) + 1;
    const linePrefix = updated.substring(lineStart, offset);
    const inTable = linePrefix.trimStart().startsWith('|');
    const warmth = inTable ? undefined : slot.warmth;
    const newVal = formatTiming(rawVal, slot.format, warmth);
    filePatched++;
    return `<!-- timing:${key} -->${newVal}<!-- /timing -->`;
  });

  // Matches: <!-- computed:EXPR -->anything<!-- /computed -->
  // EXPR is a JS expression using slot("key") and static("key") helpers.
  const COMPUTED_RE = /<!-- computed:([^>]+) -->([^<]*)<!-- \/computed -->/g;

  updated = updated.replace(COMPUTED_RE, (match, expr: string, oldVal: string) => {
    try {
      const newVal = evalComputed(expr.trim());
      if (newVal === oldVal) return match; // already current
      filePatched++;
      return `<!-- computed:${expr} -->${newVal}<!-- /computed -->`;
    } catch (e) {
      console.warn(`[update-timings] computed error in ${relative(root, filePath)}: ${e}`);
      return match;
    }
  });

  // Matches: <!-- generated:ID -->...<!-- /generated -->
  // Replaces the entire content between markers with the generator output.
  // Uses dotAll (s flag) so `.` matches newlines in multi-line blocks.
  const GENERATED_RE = /<!-- generated:(\S+) -->\n?([\s\S]*?)<!-- \/generated -->/g;

  updated = updated.replace(GENERATED_RE, (match, id: string, oldContent: string) => {
    const generator = generatedBlocks[id.trim()];
    if (!generator) {
      console.warn(`[update-timings] Unknown generated block "${id}" in ${relative(root, filePath)}`);
      return match;
    }
    const newContent = generator();
    if (newContent + "\n" === oldContent || newContent === oldContent.trim()) return match;
    filePatched++;
    return `<!-- generated:${id} -->\n${newContent}\n<!-- /generated -->`;
  });

  if (updated !== original) {
    totalPatched += filePatched;
    const rel = relative(root, filePath);
    if (DRY || CHECK) {
      console.log(`[${CHECK ? "check" : "dry"}] would update ${filePatched} marker(s) in ${rel}`);
    } else {
      writeFileSync(filePath, updated, "utf8");
      console.log(`updated ${filePatched} marker(s) in ${rel}`);
    }
  }
}

if (totalPatched === 0 && !DRY && !CHECK) {
  console.log("[update-timings] Nothing to update (all markers already current or no sidecars).");
}

if (CHECK && totalPatched > 0) {
  console.error(
    `[update-timings] --check failed: ${totalPatched} marker(s) are out of sync with sidecars.\n` +
    `  Run \`pnpm run update:timings\` to patch, or regenerate sidecars first.`
  );
  process.exit(1);
}
if (CHECK && totalPatched === 0) {
  console.log("[update-timings] Nothing to update (all markers already current or no sidecars).");
}
