/**
 * Minimal repro: WebGPU reverse associativeScan corruption at stride boundary.
 *
 * Tests reverse associativeScan on WebGPU with m=5 state dim (matrix compose).
 * WASM assoc is correct; WebGPU produces wrong results for indices 0..n-2^⌊log₂n⌋-1.
 *
 * Bug pattern:
 *   n=120, m=5 → indices 0..55 (=120-64-1) corrupted, 56..119 correct
 *   n=100, m=2 → TBD (Nile model shows no issue — likely m-dependent)
 *
 * Run with Deno:
 *   DISPLAY=:1 deno run --unstable-webgpu --allow-read --allow-write --allow-env --allow-run \
 *     issues/repro-webgpu-reverse-assocscan-stride.ts
 */

import { defaultDevice, init, numpy as np, lax, jit, tree } from "../node_modules/@hamk-uas/jax-js-nonconsuming/dist/index.js";

type Elem = { M: np.Array; v: np.Array };

// Simple affine compose: (M₁,v₁) ∘ (M₂,v₂) = (M₁·M₂, M₁·v₂ + v₁)
const compose = (a: Elem, b: Elem): Elem => ({
  M: np.matmul(a.M, b.M),
  v: np.add(np.matmul(a.M, b.v), a.v),
});

// Backward smoother compose (3 fields: A, b, S) — matches dlmSmo composeBackward
type BackwardElem = { A: np.Array; b: np.Array; S: np.Array };
const composeBackward = (a: BackwardElem, b_elem: BackwardElem): BackwardElem => {
  const A_comp = np.einsum('nij,njk->nik', b_elem.A, a.A);
  const b_comp = np.add(np.einsum('nij,njk->nik', b_elem.A, a.b), b_elem.b);
  const S_comp = np.add(np.einsum('nij,njk,nlk->nil', b_elem.A, a.S, b_elem.A), b_elem.S);
  return { A: A_comp, b: b_comp, S: S_comp };
};

async function testBackward(n: number, m: number): Promise<boolean> {
  console.log(`\n═══ Backward compose (3-field): n=${n}, m=${m} ═══`);
  const highestStride = Math.pow(2, Math.floor(Math.log2(n)));
  const boundary = n - highestStride;
  console.log(`Highest stride: ${highestStride}, boundary: ${boundary}`);

  let seed = 42;
  const rng = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };

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

async function test(n: number, m: number): Promise<boolean> {
  console.log(`\n═══ Testing n=${n}, m=${m} ═══`);
  const highestStride = Math.pow(2, Math.floor(Math.log2(n)));
  const boundary = n - highestStride;
  console.log(`Highest Kogge-Stone stride: ${highestStride}, expected boundary: ${boundary}`);

  // Deterministic seed: elements near identity
  const M_data: number[][][] = [];
  const v_data: number[][][] = [];
  let seed = 42;
  const rng = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
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

  // --- WASM reference (reverse) ---
  defaultDevice("wasm");
  const wasmScanned = await jit((M: np.Array, v: np.Array) => {
    return lax.associativeScan(compose, { M, v }, { reverse: true }) as Elem;
  })(np.array(M_data, { dtype: 'float32' }), np.array(v_data, { dtype: 'float32' }));
  const wasmData = await tree.consumeData(wasmScanned);
  const wasmV = Array.from(wasmData.v as Float32Array);

  // --- WebGPU (reverse) ---
  defaultDevice("webgpu");
  const gpuScanned = await jit((M: np.Array, v: np.Array) => {
    return lax.associativeScan(compose, { M, v }, { reverse: true }) as Elem;
  })(np.array(M_data, { dtype: 'float32' }), np.array(v_data, { dtype: 'float32' }));
  const gpuData = await tree.consumeData(gpuScanned);
  const gpuV = Array.from(gpuData.v as Float32Array);

  // --- Also test FORWARD scan ---
  defaultDevice("wasm");
  const wasmFwd = await jit((M: np.Array, v: np.Array) => {
    return lax.associativeScan(compose, { M, v }) as Elem;
  })(np.array(M_data, { dtype: 'float32' }), np.array(v_data, { dtype: 'float32' }));
  const wasmFwdData = await tree.consumeData(wasmFwd);
  const wasmFwdV = Array.from(wasmFwdData.v as Float32Array);

  defaultDevice("webgpu");
  const gpuFwd = await jit((M: np.Array, v: np.Array) => {
    return lax.associativeScan(compose, { M, v }) as Elem;
  })(np.array(M_data, { dtype: 'float32' }), np.array(v_data, { dtype: 'float32' }));
  const gpuFwdData = await tree.consumeData(gpuFwd);
  const gpuFwdV = Array.from(gpuFwdData.v as Float32Array);

  // Compare reverse
  let maxDiffRev = 0, maxIdxRev = -1, corruptedRev = 0;
  for (let i = 0; i < n * m; i++) {
    const diff = Math.abs(wasmV[i] - gpuV[i]);
    if (diff > maxDiffRev) { maxDiffRev = diff; maxIdxRev = i; }
    if (diff > 0.1) corruptedRev++;
  }

  // Compare forward
  let maxDiffFwd = 0, maxIdxFwd = -1, corruptedFwd = 0;
  for (let i = 0; i < n * m; i++) {
    const diff = Math.abs(wasmFwdV[i] - gpuFwdV[i]);
    if (diff > maxDiffFwd) { maxDiffFwd = diff; maxIdxFwd = i; }
    if (diff > 0.1) corruptedFwd++;
  }

  console.log(`REVERSE: max |Δ|=${maxDiffRev.toFixed(4)} at t=${Math.floor(maxIdxRev / m)}, corrupted=${corruptedRev}/${n * m}`);
  console.log(`FORWARD: max |Δ|=${maxDiffFwd.toFixed(4)} at t=${Math.floor(maxIdxFwd / m)}, corrupted=${corruptedFwd}/${n * m}`);

  // Per-timestep max error for reverse
  if (maxDiffRev > 1) {
    console.log("\nReverse per-timestep max |Δ| (showing corrupted region + boundary):");
    for (let t = 0; t < n; t++) {
      let tMax = 0;
      for (let j = 0; j < m; j++) {
        tMax = Math.max(tMax, Math.abs(wasmV[t * m + j] - gpuV[t * m + j]));
      }
      if (t < boundary + 3 || t >= n - 3 || tMax > 0.01) {
        const mark = t === boundary ? ' ← boundary' : t === boundary - 1 ? ' ← last corrupted' : '';
        console.log(`  t=${String(t).padStart(3)}: ${tMax.toFixed(6)}${mark}`);
      }
    }
  }

  const failed = maxDiffRev > 1;
  console.log(failed ? `❌ FAIL` : `✅ PASS`);
  return failed;
}

async function main() {
  await init("wasm");
  await init("webgpu");

  let anyFail = false;

  console.log("── Part 1: Simple 2-field compose ──");
  // Main failing case: Energy model (n=120, m=5)
  anyFail = await test(120, 5) || anyFail;
  // Control: Nile model (n=100, m=2)
  anyFail = await test(100, 2) || anyFail;
  // Just past power of 2
  anyFail = await test(129, 5) || anyFail;

  console.log("\n── Part 2: Backward smoother 3-field compose ──");
  anyFail = await testBackward(120, 5) || anyFail;
  anyFail = await testBackward(100, 2) || anyFail;
  anyFail = await testBackward(129, 5) || anyFail;
  anyFail = await testBackward(128, 5) || anyFail;
  anyFail = await testBackward(65, 5) || anyFail;
  anyFail = await testBackward(200, 5) || anyFail;

  if (anyFail) {
    console.log("\n❌ At least one test FAILED — reverse assocScan is broken on WebGPU");
    // @ts-ignore
    if (typeof Deno !== 'undefined') Deno.exit(1);
    else process.exit(1);
  } else {
    console.log("\n✅ All tests passed");
  }
}

main().catch(console.error);
