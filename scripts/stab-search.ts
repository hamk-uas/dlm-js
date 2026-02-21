/**
 * Greedy stabilization search for the Float32 sequential backward smoother.
 *
 * Tests 5 incremental stabilization flags (nSym, nDiag, nLeak, cDiag, cEps)
 * on top of the default joseph+symmetrize baseline across 5 models with m > 2.
 * Uses a greedy algorithm: each round adds the flag that most reduces the error
 * metric, until no further improvement is found.
 *
 * Error metric:  primary   = number of diverged models (NaN/Inf in yhat)
 *                secondary = max relative error on yhat + level-state series
 *                            across all non-diverged models
 * Score:  score = #diverged * 1e6 + maxRelErr   (lower is better)
 *
 * Usage:  pnpm run stab:search
 */

import { defaultDevice, init } from "@hamk-uas/jax-js-nonconsuming";
import { dlmFit } from "../src/index.ts";
import type { DlmStabilization } from "../src/types.ts";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");

// ── Model definitions ────────────────────────────────────────────────────────

interface ModelDef {
  name: string;
  m: number;
  inFile: string;
  refFile: string;
  dlmOpts: Record<string, unknown>;
}

const MODELS: ModelDef[] = [
  {
    name: 'order2',
    m: 3,
    inFile: 'order2-in.json',
    refFile: 'order2-out-m.json',
    dlmOpts: { order: 2 },
  },
  {
    name: 'kaisaniemi',
    m: 4,
    inFile: 'kaisaniemi-in.json',
    refFile: 'kaisaniemi-out-m.json',
    dlmOpts: { order: 1, harmonics: 1 },
  },
  {
    name: 'trigar',
    m: 5,
    inFile: 'trigar-in.json',
    refFile: 'trigar-out-m.json',
    dlmOpts: { order: 1, harmonics: 1, seasonLength: 12, arCoefficients: [0.7] },
  },
  {
    name: 'trig',
    m: 6,
    inFile: 'trig-in.json',
    refFile: 'trig-out-m.json',
    dlmOpts: { order: 1, harmonics: 2, seasonLength: 12 },
  },
  {
    name: 'seasonal',
    m: 13,
    inFile: 'seasonal-in.json',
    refFile: 'seasonal-out-m.json',
    dlmOpts: { order: 1, fullSeasonal: true, seasonLength: 12 },
  },
];

// ── Candidate flags ──────────────────────────────────────────────────────────

const CANDIDATES: (keyof DlmStabilization)[] = ['nSym', 'nDiag', 'nLeak', 'cDiag', 'cEps'];

const FLAG_DESC: Record<keyof DlmStabilization, string> = {
  nSym:  'sym N each step',
  nDiag: 'clamp diag(N)≥0',
  nLeak: 'N*=(1-1e-5)/step',
  cDiag: 'clamp diag(C)≥1e-7',
  cEps:  'C+=1e-6·I',
};

// ── Data loading ─────────────────────────────────────────────────────────────

interface LoadedModel {
  def: ModelDef;
  y: number[];
  s: number;
  w: number[];
  refYhat: number[];
  refLevel: number[];  // smoothed state 0 (level)
}

function loadModels(): LoadedModel[] {
  return MODELS.map(def => {
    const inp  = JSON.parse(readFileSync(resolve(root, `tests/${def.inFile}`), 'utf-8'));
    const ref  = JSON.parse(readFileSync(resolve(root, `tests/${def.refFile}`), 'utf-8'));
    return {
      def,
      y: inp.y,
      s: inp.s,
      w: Array.isArray(inp.w) ? inp.w : [inp.w],
      refYhat:  ref.yhat as number[],
      // ref.x is [state_idx][time_idx] (MATLAB layout)
      refLevel: (ref.x as number[][])[0],
    };
  });
}

// ── Evaluation ───────────────────────────────────────────────────────────────

interface ModelResult {
  name: string;
  m: number;
  diverged: boolean;
  maxRelErrYhat: number;
  maxRelErrLevel: number;
  maxRelErr: number;  // max of yhat and level
}

interface EvalResult {
  models: ModelResult[];
  divergedCount: number;
  maxRelErr: number;
  score: number;
}

function relErr(a: ArrayLike<number>, b: number[]): number {
  let max = 0;
  for (let i = 0; i < b.length; i++) {
    const ref = Math.abs(b[i]);
    const denom = ref < 1e-10 ? 1e-10 : ref;
    const e = Math.abs(a[i] - b[i]) / denom;
    if (e > max) max = e;
  }
  return max;
}

async function evaluate(
  models: LoadedModel[],
  flags: DlmStabilization,
): Promise<EvalResult> {
  const results: ModelResult[] = [];
  let divergedCount = 0;
  let maxRelErr = 0;

  for (const m of models) {
    let diverged = false;
    let maxRelErrYhat = 0;
    let maxRelErrLevel = 0;

    try {
      const fit = await dlmFit(m.y, {
        obsStd: m.s,
        processStd: m.w,
        dtype: 'f32',
        stabilization: flags,
        ...m.def.dlmOpts,
      } as Parameters<typeof dlmFit>[1]);

      // Check for NaN/Inf in yhat
      for (let i = 0; i < fit.yhat.length; i++) {
        if (!isFinite(fit.yhat[i])) { diverged = true; break; }
      }

      if (!diverged) {
        maxRelErrYhat  = relErr(fit.yhat, m.refYhat);
        maxRelErrLevel = relErr(fit.smoothed.series(0), m.refLevel);
      }
    } catch {
      diverged = true;
    }

    if (diverged) {
      divergedCount++;
      results.push({ name: m.def.name, m: m.def.m, diverged: true,
        maxRelErrYhat: Infinity, maxRelErrLevel: Infinity, maxRelErr: Infinity });
    } else {
      const modelMax = Math.max(maxRelErrYhat, maxRelErrLevel);
      if (modelMax > maxRelErr) maxRelErr = modelMax;
      results.push({ name: m.def.name, m: m.def.m, diverged: false,
        maxRelErrYhat, maxRelErrLevel, maxRelErr: modelMax });
    }
  }

  const score = divergedCount * 1e6 + maxRelErr;
  return { models: results, divergedCount, maxRelErr, score };
}

// ── Formatting ───────────────────────────────────────────────────────────────

function fmtErr(e: number): string {
  if (!isFinite(e)) return 'DIVERGED';
  if (e === 0) return '0';
  return e.toExponential(2);
}

function fmtScore(s: number): string {
  if (s >= 1e6) return `∞ (${Math.round(s / 1e6)} diverg)`;
  return s.toExponential(2);
}

function flagsLabel(flags: DlmStabilization): string {
  const active = (Object.keys(flags) as (keyof DlmStabilization)[])
    .filter(k => flags[k]);
  return active.length > 0 ? active.join('+') : 'none';
}

function printEvalRow(label: string, result: EvalResult): void {
  const perModel = result.models.map(r =>
    `${r.name}(m=${r.m}):${fmtErr(r.maxRelErr)}`
  ).join('  ');
  console.log(`  ${label.padEnd(36)} score=${fmtScore(result.score).padEnd(14)} | ${perModel}`);
}

// ── Greedy search ─────────────────────────────────────────────────────────────

async function greedySearch(): Promise<void> {
  const models = loadModels();

  console.log('=== Float32/wasm stabilization greedy search ===\n');
  console.log('Baseline: joseph form (L·C·L\' + K·V²·K\' + W) + symmetrize (always on)');
  console.log('Candidates on top of baseline:\n');
  for (const [k, v] of Object.entries(FLAG_DESC)) {
    console.log(`  ${k.padEnd(8)} — ${v}`);
  }
  console.log();
  console.log('Models (m > 2, all run with wasm/f32, compared to Octave f64 reference):');
  console.log('  ' + models.map(m => `${m.def.name}(m=${m.def.m})`).join('  '));
  console.log();
  console.log('Metric: max relative error on yhat and smoothed level state, across all models.');
  console.log('Score = #diverged * 1e6 + maxRelErr  (lower is better)\n');

  const hdr = '  ' + 'Combination'.padEnd(36) + ' score              | per-model max rel err';
  console.log(hdr);
  console.log('  ' + '─'.repeat(hdr.length - 2));

  // Baseline: joseph only (flags = {})
  const baselineResult = await evaluate(models, {});
  printEvalRow('joseph only (baseline)', baselineResult);

  let activeFlags: DlmStabilization = {};
  let remaining: (keyof DlmStabilization)[] = [...CANDIDATES];
  let bestScore = baselineResult.score;

  for (let round = 1; round <= CANDIDATES.length; round++) {
    console.log(`\nRound ${round}: testing additions to joseph+[${flagsLabel(activeFlags)}]:`);

    let roundBestScore = bestScore;
    let roundBestFlag: keyof DlmStabilization | null = null;
    let roundBestResult: EvalResult | null = null;

    for (const flag of remaining) {
      const testFlags = { ...activeFlags, [flag]: true };
      const result = await evaluate(models, testFlags);
      const label = `joseph+${flagsLabel(testFlags)}`;
      printEvalRow(label, result);
      if (result.score < roundBestScore) {
        roundBestScore = result.score;
        roundBestFlag = flag;
        roundBestResult = result;
      }
    }

    if (roundBestFlag === null || roundBestResult === null) {
      console.log('\n  No improvement in this round. Stopping greedy search.');
      break;
    }

    activeFlags = { ...activeFlags, [roundBestFlag]: true };
    remaining = remaining.filter(f => f !== roundBestFlag);
    bestScore = roundBestScore;

    console.log(`\n  ► Round ${round} winner: +${roundBestFlag}  (score ${fmtScore(bestScore)})`);
    console.log(`    ${FLAG_DESC[roundBestFlag]}`);

    if (remaining.length === 0) {
      console.log('\n  All flags tested.');
      break;
    }
  }

  console.log('\n' + '─'.repeat(70));
  console.log(`Final greedy result:  joseph+[${flagsLabel(activeFlags)}]`);
  console.log(`Final score:          ${fmtScore(bestScore)}`);
  if (bestScore < 1e6) {
    const finalResult = await evaluate(models, activeFlags);
    console.log('\nFinal per-model detail:');
    for (const r of finalResult.models) {
      if (r.diverged) {
        console.log(`  ${r.name.padEnd(12)} m=${r.m}  DIVERGED`);
      } else {
        console.log(`  ${r.name.padEnd(12)} m=${r.m}  yhat=${fmtErr(r.maxRelErrYhat).padEnd(10)} level=${fmtErr(r.maxRelErrLevel)}`);
      }
    }
  }
  console.log();
}

// ── Main ─────────────────────────────────────────────────────────────────────

const availableDevices = await init();
if (!availableDevices.includes('wasm')) {
  console.error('WASM device not available');
  process.exit(1);
}
defaultDevice('wasm');

await greedySearch();
