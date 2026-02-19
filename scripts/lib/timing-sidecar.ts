/**
 * Timing sidecar helpers.
 *
 * Each gen-* / collect-* script writes a small JSON file under
 * `assets/timings/<script-basename>.json` immediately after it finishes.
 * `update-timings.ts` reads these sidecars and patches `<!-- timing:KEY -->`
 * markers throughout every .md file in the repo.
 *
 * Sidecar format:  { [field: string]: number }   (all values in ms or a plain
 * integer, unit is determined by the registry entry's `format` field).
 */

import { writeFileSync, readFileSync, mkdirSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { cpus, totalmem } from "node:os";

const root = resolve(dirname(new URL(import.meta.url).pathname), "../..");
const sidecarDir = resolve(root, "assets/timings");
const staticRefsPath = resolve(sidecarDir, "static-references.json");

/** Write timing data for a script.  `scriptBasename` is the filename without
 *  the leading path, e.g. `"gen-niledemo-svg"` (no .ts extension). */
export function writeTimingsSidecar(
  scriptBasename: string,
  data: Record<string, number>,
): void {
  if (!existsSync(sidecarDir)) mkdirSync(sidecarDir, { recursive: true });
  const outPath = resolve(sidecarDir, `${scriptBasename}.json`);
  writeFileSync(outPath, JSON.stringify(data, null, 2) + "\n");
}

/** Read an existing sidecar, or return null if it does not exist yet. */
export function readTimingsSidecar(
  scriptBasename: string,
): Record<string, number> | null {
  const p = resolve(sidecarDir, `${scriptBasename}.json`);
  if (!existsSync(p)) return null;
  return JSON.parse(readFileSync(p, "utf8")) as Record<string, number>;
}

/**
 * Update the `machine` key in `assets/timings/static-references.json` with
 * a short CPU + RAM description derived from `node:os`.  Call this from any
 * script that writes benchmark timings so the README machine footnote stays
 * current automatically.
 *
 * Example output: "Intel(R) Core(TM) Ultra 5 125H, 62 GB RAM"
 */
export function stampMachineInfo(): void {
  const cpu = cpus()[0]?.model?.trim() ?? "unknown CPU";
  const ramGb = Math.round(totalmem() / 1024 ** 3);
  const machine = `${cpu}, ${ramGb} GB RAM`;

  let existing: Record<string, unknown> = {};
  try {
    existing = JSON.parse(readFileSync(staticRefsPath, "utf8")) as Record<string, unknown>;
  } catch { /* file may not exist yet */ }

  existing["machine"] = machine;
  if (!existsSync(sidecarDir)) mkdirSync(sidecarDir, { recursive: true });
  writeFileSync(staticRefsPath, JSON.stringify(existing, null, 2) + "\n");
}
