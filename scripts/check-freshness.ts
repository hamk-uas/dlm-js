/**
 * check-freshness.ts
 *
 * Detects stale artifacts (SVGs, timing sidecars, benchmark data) and
 * validates algorithm coverage across benchmark/test files.
 *
 * Usage
 * ─────
 *   pnpm run check:freshness              # report stale artifacts + coverage
 *   pnpm run check:freshness -- --status  # machine-readable exit code (0 = fresh, 1 = stale)
 *   pnpm run check:freshness -- --json    # JSON output for tooling
 *
 * How staleness works
 * ───────────────────
 * For each artifact group declared in `scripts/lib/artifact-graph.ts`,
 * the newest source file mtime is compared against the oldest output file
 * mtime.  If any source is newer than any output, the group is stale.
 *
 * Missing output files are always considered stale.
 */

import { readFileSync, existsSync, statSync } from "node:fs";
import { resolve, dirname } from "node:path";
import {
  artifactGroups,
  algorithmCoverageTargets,
  ALGORITHM_TYPE_FILE,
  ALGORITHM_TYPE_REGEX,
  abs,
  mtimeMs,
  ROOT,
  type ArtifactGroup,
} from "./lib/artifact-graph.ts";

// ── Args ──────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const STATUS = args.includes("--status");
const JSON_OUT = args.includes("--json");

// ── Extract known algorithms from DlmAlgorithm type ──────────────────────

function getKnownAlgorithms(): string[] {
  const typesPath = abs(ALGORITHM_TYPE_FILE);
  const content = readFileSync(typesPath, "utf8");
  const match = ALGORITHM_TYPE_REGEX.exec(content);
  if (!match) {
    console.error(`[check-freshness] Could not find DlmAlgorithm type in ${ALGORITHM_TYPE_FILE}`);
    process.exit(2);
  }
  // Parse: 'scan' | 'assoc' | 'sqrt-assoc'
  const raw = match[1];
  const algorithms = [...raw.matchAll(/'([^']+)'/g)].map(m => m[1]);
  if (algorithms.length === 0) {
    console.error(`[check-freshness] No algorithm literals found in DlmAlgorithm type`);
    process.exit(2);
  }
  return algorithms;
}

// ── Staleness check ──────────────────────────────────────────────────────

interface StaleGroup {
  group: ArtifactGroup;
  reason: string;             // "missing" | "outdated"
  missingOutputs: string[];   // outputs that don't exist
  newestSource: string;       // the source file with the newest mtime
  newestSourceMs: number;
  oldestOutput: string;       // the output file with the oldest mtime (if any exist)
  oldestOutputMs: number;
}

function checkStaleness(): StaleGroup[] {
  const stale: StaleGroup[] = [];

  for (const group of artifactGroups) {
    // Find newest source
    let newestSourceMs = 0;
    let newestSource = "";
    for (const src of group.sources) {
      const ms = mtimeMs(src);
      if (ms == null) continue; // source doesn't exist — skip (may be optional)
      if (ms > newestSourceMs) {
        newestSourceMs = ms;
        newestSource = src;
      }
    }
    if (newestSourceMs === 0) continue; // no sources found — skip group

    // Find oldest output + missing outputs
    let oldestOutputMs = Infinity;
    let oldestOutput = "";
    const missingOutputs: string[] = [];

    for (const out of group.outputs) {
      const ms = mtimeMs(out);
      if (ms == null) {
        missingOutputs.push(out);
        continue;
      }
      if (ms < oldestOutputMs) {
        oldestOutputMs = ms;
        oldestOutput = out;
      }
    }

    if (missingOutputs.length > 0) {
      stale.push({
        group,
        reason: "missing",
        missingOutputs,
        newestSource,
        newestSourceMs,
        oldestOutput: oldestOutput || "(none)",
        oldestOutputMs: oldestOutputMs === Infinity ? 0 : oldestOutputMs,
      });
    } else if (newestSourceMs > oldestOutputMs) {
      stale.push({
        group,
        reason: "outdated",
        missingOutputs: [],
        newestSource,
        newestSourceMs,
        oldestOutput,
        oldestOutputMs,
      });
    }
  }

  return stale;
}

// ── Algorithm coverage check ─────────────────────────────────────────────

interface CoverageGap {
  file: string;
  label: string;
  missing: string[];
}

function checkAlgorithmCoverage(): { algorithms: string[]; gaps: CoverageGap[] } {
  const algorithms = getKnownAlgorithms();
  const gaps: CoverageGap[] = [];

  for (const target of algorithmCoverageTargets) {
    const filePath = abs(target.file);
    if (!existsSync(filePath)) {
      gaps.push({ file: target.file, label: target.label, missing: [...algorithms] });
      continue;
    }

    const content = readFileSync(filePath, "utf8");
    const found = new Set<string>();

    // Collect all matches from the pattern
    let m: RegExpExecArray | null;
    const re = new RegExp(target.pattern.source, target.pattern.flags);
    while ((m = re.exec(content)) !== null) {
      // Check all capture groups
      for (let i = 1; i < m.length; i++) {
        if (m[i] && algorithms.includes(m[i])) {
          found.add(m[i]);
        }
      }
    }

    // Also do a simpler check: look for each algorithm as a string literal
    for (const alg of algorithms) {
      if (content.includes(`'${alg}'`) || content.includes(`"${alg}"`)) {
        found.add(alg);
      }
    }

    const missing = algorithms.filter(a => !found.has(a));
    if (missing.length > 0) {
      gaps.push({ file: target.file, label: target.label, missing });
    }
  }

  return { algorithms, gaps };
}

// ── Main ──────────────────────────────────────────────────────────────────

const staleGroups = checkStaleness();
const { algorithms, gaps } = checkAlgorithmCoverage();

let hasIssues = staleGroups.length > 0 || gaps.length > 0;

if (JSON_OUT) {
  const result = {
    stale: staleGroups.map(s => ({
      id: s.group.id,
      label: s.group.label,
      reason: s.reason,
      command: s.group.command,
      heavy: s.group.heavy,
      missingOutputs: s.missingOutputs,
    })),
    coverage: {
      algorithms,
      gaps: gaps.map(g => ({
        file: g.file,
        label: g.label,
        missing: g.missing,
      })),
    },
  };
  console.log(JSON.stringify(result, null, 2));
} else {
  // Human-readable output
  if (staleGroups.length > 0) {
    console.log("[check-freshness] Stale artifacts detected:\n");
    for (const s of staleGroups) {
      const icon = s.group.heavy ? "🐢" : "⚡";
      console.log(`  ${icon} ${s.group.label}`);
      if (s.reason === "missing") {
        console.log(`     reason: ${s.missingOutputs.length} output(s) missing`);
        for (const m of s.missingOutputs.slice(0, 3)) {
          console.log(`       - ${m}`);
        }
        if (s.missingOutputs.length > 3) {
          console.log(`       ... and ${s.missingOutputs.length - 3} more`);
        }
      } else {
        const ageSec = ((s.newestSourceMs - s.oldestOutputMs) / 1000) | 0;
        const ageStr = ageSec > 86400
          ? `${(ageSec / 86400) | 0}d`
          : ageSec > 3600
            ? `${(ageSec / 3600) | 0}h`
            : `${ageSec}s`;
        console.log(`     reason: source ${s.newestSource} is ${ageStr} newer than ${s.oldestOutput}`);
      }
      console.log(`     fix:    ${s.group.command}`);
      console.log();
    }
  }

  if (gaps.length > 0) {
    console.log("[check-freshness] Algorithm coverage gaps:\n");
    console.log(`  Known algorithms (from DlmAlgorithm): ${algorithms.join(", ")}\n`);
    for (const g of gaps) {
      console.log(`  ❌ ${g.label} (${g.file})`);
      console.log(`     missing: ${g.missing.join(", ")}`);
      console.log();
    }
  }

  if (!hasIssues) {
    console.log("[check-freshness] All artifacts fresh. Algorithm coverage complete.");
  }
}

if (STATUS) {
  process.exit(hasIssues ? 1 : 0);
}
