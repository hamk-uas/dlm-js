/**
 * Minimal repro: WebGPU reverse associativeScan corruption with multi-field
 * compose (3 einsums + 2 adds) and m ≥ 5.
 *
 * Regression introduced by 66670f4: "proper per-element reduction & multi-element
 * codegen in fused block_map shader" removed the bodyHasReductions guard from
 * 8501de9 that had fixed this exact bug class.
 *
 * WASM assoc is correct; WebGPU produces wrong results for indices 0..n-2^⌊log₂n⌋-1.
 *
 * Run:
 *   GPU=nvidia bash scripts/gpu-test.sh run issues/repro-webgpu-reverse-assocscan-multifield.ts
 */

import { describe, it, expect } from 'vitest';
import { defaultDevice, init, numpy as np, lax, jit, tree } from "@hamk-uas/jax-js-nonconsuming";

// ── Compose functions ──

type Elem = { M: np.Array; v: np.Array };

const compose = (a: Elem, b: Elem): Elem => ({
  M: np.matmul(a.M, b.M),
  v: np.add(np.matmul(a.M, b.v), a.v),
});

type BackwardElem = { A: np.Array; b: np.Array; S: np.Array };
const composeBackward = (a: BackwardElem, b_elem: BackwardElem): BackwardElem => {
  const A_comp = np.einsum('nij,njk->nik', b_elem.A, a.A);
  const b_comp = np.add(np.einsum('nij,njk->nik', b_elem.A, a.b), b_elem.b);
  const S_comp = np.add(np.einsum('nij,njk,nlk->nil', b_elem.A, a.S, b_elem.A), b_elem.S);
  return { A: A_comp, b: b_comp, S: S_comp };
};

// ── Deterministic PRNG ──
function makePrng(seed: number) {
  return () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
}

// ── Test: 3-field backward compose ──
async function testBackward(n: number, m: number): Promise<boolean> {
  console.log(`\n═══ Backward compose (3-field): n=${n}, m=${m} ═══`);
  const highestStride = Math.pow(2, Math.floor(Math.log2(n)));
  const boundary = n - highestStride;
  console.log(`Highest stride: ${highestStride}, boundary: ${boundary}`);

  const rng = makePrng(42);
  const A_data: number[][][] = [];
  const b_data: number[][][] = [];
  const S_data: number[][][] = [];
  for (let i = 0; i < n; i++) {
    const aRow: number[][] = [], sRow: number[][] = [];
    for (let j = 0; j < m; j++) {
      const ar: number[] = [], sr: number[] = [];
      for (let k = 0; k < m; k++) {
        ar.push(j === k ? 0.8 + 0.1 * rng() : 0.02 * (rng() - 0.5));
        sr.push(j === k ? 0.5 + 0.5 * rng() : 0.01 * rng());
      }
      aRow.push(ar); sRow.push(sr);
    }
    A_data.push(aRow);
    b_data.push(Array.from({ length: m }, () => [rng() * 10]));
    S_data.push(sRow);
  }

  // WASM reference (reverse)
  defaultDevice("wasm");
  const wasmRes = await jit((A: np.Array, b: np.Array, S: np.Array) =>
    lax.associativeScan(composeBackward, { A, b, S }, { reverse: true }) as BackwardElem
  )(np.array(A_data, { dtype: 'float32' }), np.array(b_data, { dtype: 'float32' }), np.array(S_data, { dtype: 'float32' }));
  const wasmD = await tree.consumeData(wasmRes);
  const wasmB = Array.from(wasmD.b as Float32Array);

  // WebGPU (reverse)
  defaultDevice("webgpu");
  const gpuRes = await jit((A: np.Array, b: np.Array, S: np.Array) =>
    lax.associativeScan(composeBackward, { A, b, S }, { reverse: true }) as BackwardElem
  )(np.array(A_data, { dtype: 'float32' }), np.array(b_data, { dtype: 'float32' }), np.array(S_data, { dtype: 'float32' }));
  const gpuD = await tree.consumeData(gpuRes);
  const gpuB = Array.from(gpuD.b as Float32Array);

  let maxDiff = 0, maxIdx = -1, corrupted = 0;
  for (let i = 0; i < n * m; i++) {
    const diff = Math.abs(wasmB[i] - gpuB[i]);
    if (diff > maxDiff) { maxDiff = diff; maxIdx = i; }
    if (diff > 0.1) corrupted++;
  }

  console.log(`Max |Δ|=${maxDiff.toFixed(4)} at t=${Math.floor(maxIdx / m)}, corrupted=${corrupted}/${n * m}`);

  if (maxDiff > 1) {
    console.log("\nPer-timestep max |Δ|:");
    for (let t = 0; t < Math.min(n, boundary + 5); t++) {
      let tMax = 0;
      for (let j = 0; j < m; j++) {
        tMax = Math.max(tMax, Math.abs(wasmB[t * m + j] - gpuB[t * m + j]));
      }
      const mark = t === boundary ? ' ← boundary' : t === boundary - 1 ? ' ← last corrupted' : '';
      console.log(`  t=${String(t).padStart(3)}: ${tMax.toFixed(6)}${mark}`);
    }
    console.log('  ...');
    for (let t = n - 3; t < n; t++) {
      let tMax = 0;
      for (let j = 0; j < m; j++) {
        tMax = Math.max(tMax, Math.abs(wasmB[t * m + j] - gpuB[t * m + j]));
      }
      console.log(`  t=${String(t).padStart(3)}: ${tMax.toFixed(6)}`);
    }
  }

  const failed = maxDiff > 1;
  console.log(failed ? `❌ FAIL` : `✅ PASS`);
  return failed;
}

// ── Test: 2-field compose (control) ──
async function test2Field(n: number, m: number): Promise<boolean> {
  console.log(`\n═══ 2-field compose: n=${n}, m=${m} ═══`);
  const highestStride = Math.pow(2, Math.floor(Math.log2(n)));
  const boundary = n - highestStride;
  console.log(`Highest stride: ${highestStride}, boundary: ${boundary}`);

  const rng = makePrng(42);
  const M_data: number[][][] = [];
  const v_data: number[][][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[][] = [];
    for (let j = 0; j < m; j++) {
      const r: number[] = [];
      for (let k = 0; k < m; k++) {
        r.push(j === k ? 0.9 + 0.1 * rng() : 0.01 * (rng() - 0.5));
      }
      row.push(r);
    }
    M_data.push(row);
    v_data.push(Array.from({ length: m }, () => [rng() * 10]));
  }

  // WASM reference
  defaultDevice("wasm");
  const wasmScanned = await jit((M: np.Array, v: np.Array) =>
    lax.associativeScan(compose, { M, v }, { reverse: true }) as Elem
  )(np.array(M_data, { dtype: 'float32' }), np.array(v_data, { dtype: 'float32' }));
  const wasmData = await tree.consumeData(wasmScanned);
  const wasmV = Array.from(wasmData.v as Float32Array);

  // WebGPU
  defaultDevice("webgpu");
  const gpuScanned = await jit((M: np.Array, v: np.Array) =>
    lax.associativeScan(compose, { M, v }, { reverse: true }) as Elem
  )(np.array(M_data, { dtype: 'float32' }), np.array(v_data, { dtype: 'float32' }));
  const gpuData = await tree.consumeData(gpuScanned);
  const gpuV = Array.from(gpuData.v as Float32Array);

  let maxDiff = 0, corrupted = 0;
  for (let i = 0; i < n * m; i++) {
    const diff = Math.abs(wasmV[i] - gpuV[i]);
    if (diff > maxDiff) maxDiff = diff;
    if (diff > 0.1) corrupted++;
  }

  console.log(`Max |Δ|=${maxDiff.toFixed(4)}, corrupted=${corrupted}/${n * m}`);
  const failed = maxDiff > 1;
  console.log(failed ? `❌ FAIL` : `✅ PASS`);
  return failed;
}

describe('repro-webgpu-reverse-assocscan-multifield', () => {
  it('reverse assocScan multi-field compose correct on WebGPU', async () => {
    await init("wasm");
    await init("webgpu");

    let anyFail = false;

    console.log("── Part 1: 2-field compose (control) ──");
    anyFail = await test2Field(120, 5) || anyFail;
    anyFail = await test2Field(100, 2) || anyFail;

    console.log("\n── Part 2: 3-field backward compose (failing cases) ──");
    anyFail = await testBackward(120, 5) || anyFail;
    anyFail = await testBackward(100, 2) || anyFail;
    anyFail = await testBackward(128, 5) || anyFail;
    anyFail = await testBackward(65, 5) || anyFail;
    anyFail = await testBackward(200, 5) || anyFail;

    expect(anyFail, "reverse assocScan multi-field compose broken on WebGPU").toBe(false);
  });
});
