/**
 * Minimal reproduction: lax.associativeScan dispatch overhead on WebGPU.
 *
 * ─── BACKGROUND ──────────────────────────────────────────────────────────────
 *
 * lax.associativeScan (parallel prefix scan) has O(log n) *depth* in theory:
 *   round 1: compose n/2 independent pairs  → depth 1
 *   round 2: compose n/4 independent pairs  → depth 2
 *   ...
 *   round log₂n: 1 pair                     → depth log₂n
 *
 * Exploiting that depth on GPU requires dispatching ALL n/2 independent
 * compose calls within a round as a SINGLE batched kernel (one launch per
 * op per round = P·log(n) total GPU dispatches for P ops in compose).
 *
 * If instead each compose call dispatches from JS individually, the totals are:
 *   n/2 + n/4 + … + 1   composed calls   ≈ n   (sum of geometric series)
 *   × P ops/call                          → P·n  GPU kernel launches
 *
 * That is O(n) from JS→GPU dispatch alone, regardless of GPU parallelism.
 * The result is WORSE than the sequential scan (n dispatches × P ops = P·n),
 * because the scan visit count is the same but associativeScan redundantly
 * stores intermediate results, adding extra memory traffic.
 *
 * ─── THIS REPRO ──────────────────────────────────────────────────────────────
 *
 * Compose function mirrors the DLM Kalman forward-filter compose used in
 * dlm-js (https://github.com/hamk-uas/dlm-js), scaled down to m=2.
 * Each element is (A: [k,2,2], b: [k,2,1], S: [k,2,2]) and compose does:
 *
 *   A_comp = B.A @ A.A               (einsum nij,njk→nik)  — 1 kernel
 *   Ab     = B.A @ A.b               (einsum nij,njk→nik)  — 1 kernel
 *   b_comp = Ab + B.b                (add)                 — 1 kernel
 *   BSAt   = B.A @ A.S @ B.A'        (einsum nij,njk,nlk→nil) — 1 kernel
 *   S_comp = BSAt + B.S              (add)                 — 1 kernel
 *
 * Total: P = 5 ops × (n-1) compose calls = 5(n-1) sequential GPU dispatches
 * Expected (fused):  5 × log₂(n) dispatches
 *
 * Measured: wall-clock scales linearly with n (doubling n doubles time)
 * rather than logarithmically (doubling n adds one extra round).
 *
 * ─── HOW TO RUN ──────────────────────────────────────────────────────────────
 *
 *   deno run --unstable-webgpu --allow-read --allow-env \
 *     issues/repro-associativescan-dispatch.ts
 *
 * ─── EXPECTED OUTPUT (with kernel fusion) ───────────────────────────────────
 *
 *   N=100  →  T ms   (baseline)
 *   N=200  →  T ms   (+1 round, ~same)
 *   N=400  →  T ms   (+1 round, ~same)
 *   N=800  →  T ms   (+1 round, ~same)
 *   N=1600 →  T ms   (+1 round, ~same)
 *
 * ─── ACTUAL OUTPUT (jax-js fix/jit-scan-einsum-maxargs branch, commit 3112787c) ─
 *
 * A: lax.associativeScan  — FIXED — O(log n), ~17-22ms constant   ✓
 * B: jit(assocScan)       — FIXED — same as A                     ✓
 * C: lax.scan (backward)  — NOT fixed — O(n), 47→1054ms           ✗
 *
 *        N   A:assocScan  ratio   B:jit(scan)  ratio   C:lax.scan  ratio
 *      100          17.5      -          15.8      -         47.6      -
 *      200          16.8  0.96x          16.5  1.04x        102.4  2.15x
 *      400          20.3  1.21x          19.0  1.15x        201.1  1.96x
 *      800          20.8  1.02x          20.6  1.09x        380.4  1.89x
 *     1600          22.6  1.08x          23.0  1.11x       1053.5  2.77x
 *
 * lax.scan dispatches each of its n steps from JS individually.
 * This is the remaining O(n) bottleneck in dlm-js dlmFit WebGPU:
 * the backward RTS smoother uses lax.scan (not associativeScan).
 */

import { numpy as np, lax, jit, DType, defaultDevice, init }
  from "../node_modules/@hamk-uas/jax-js-nonconsuming/dist/index.js";

// ── Config ────────────────────────────────────────────────────────────────

const M       = 2;    // state dimension (mirrors Nile order=1, m=2)
const WARMUP  = 2;    // warm-up runs discarded
const RUNS    = 4;    // timed runs (median reported)
const N_VALUES = [100, 200, 400, 800, 1_600];

// ── Init WebGPU ───────────────────────────────────────────────────────────

await init("webgpu");
defaultDevice("webgpu");
const dtype = DType.Float32;

// ── Build constant element arrays (same as a constant DLM) ────────────────

function makeElements(n: number) {
  // A: [n, M, M] — identity-ish transition
  const aData = new Float32Array(n * M * M);
  for (let i = 0; i < n; i++) {
    aData[i * M * M + 0] = 1.0;  // (0,0)
    aData[i * M * M + 1] = 1.0;  // (0,1)
    aData[i * M * M + 3] = 1.0;  // (1,1)
  }
  // b: [n, M, 1] — per-step correction
  const bData = new Float32Array(n * M * 1);
  for (let i = 0; i < n; i++) {
    bData[i * M] = 0.5;
  }
  // S: [n, M, M] — covariance contribution (symmetric, positive)
  const sData = new Float32Array(n * M * M);
  for (let i = 0; i < n; i++) {
    sData[i * M * M + 0] = 0.1;  // (0,0)
    sData[i * M * M + 3] = 0.1;  // (1,1)
  }
  return {
    A: np.array(aData, { dtype, shape: [n, M, M] }),
    b: np.array(bData, { dtype, shape: [n, M, 1] }),
    S: np.array(sData, { dtype, shape: [n, M, M] }),
  };
}

// ── Compose function (5 ops — the DLM Kalman chain rule) ─────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Elem = { A: any; b: any; S: any };

// compose(earlier, later): DLM Kalman chain rule, 5 ops
function compose(earlier: Elem, later: Elem): Elem {
  // A_comp = later.A @ earlier.A
  const A_comp = np.einsum("nij,njk->nik", later.A, earlier.A);
  // b_comp = later.A @ earlier.b + later.b
  using Ab = np.einsum("nij,njk->nik", later.A, earlier.b);
  const b_comp = np.add(Ab, later.b);
  // S_comp = later.A @ earlier.S @ later.A' + later.S
  using ASAt = np.einsum("nij,njk,nlk->nil", later.A, earlier.S, later.A);
  const S_comp = np.add(ASAt, later.S);
  return { A: A_comp, b: b_comp, S: S_comp };
}

// backwardStep(carry, elem): mirrors RTS backward smoother, 5 ops on [m,m]/[m,1]
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function backwardStep(carry: any, elem: any) {
  using At  = np.einsum("ij->ji", elem.A);           // A'
  using Atr = np.einsum("ij,jk->ik", At, carry.r);   // A' r
  const r_new = np.add(Atr, elem.b);                  // A' r + b  [M,1]
  using AtN   = np.einsum("ij,jk->ik", At, carry.N);  // A' N
  using AtNA  = np.einsum("ij,jk->ik", AtN, elem.A);  // A' N A
  const N_new = np.add(AtNA, elem.S);                  // A' N A + S  [M,M]
  return [{ r: r_new, N: N_new }, { r: r_new, N: N_new }];
}

function makeCarry() {
  return {
    r: np.array(new Float32Array(M), { dtype, shape: [M, 1] }),
    N: np.array(new Float32Array(M * M), { dtype, shape: [M, M] }),
  };
}

// ── Timing helpers ─────────────────────────────────────────────────────────

function median(arr: number[]): number {
  const s = [...arr].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function flush(v: any) { await v?.data?.(); v?.[Symbol.dispose]?.(); }

async function timedRun(
  n: number,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  fn: (n: number) => Promise<any>,
): Promise<number> {
  for (let i = 0; i < WARMUP; i++) await flush(await fn(n));
  const times: number[] = [];
  for (let i = 0; i < RUNS; i++) {
    const t0 = performance.now();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const r: any = await fn(n);
    await r?.data?.();   // GPU->CPU flush to end timer fairly
    times.push(performance.now() - t0);
    r?.[Symbol.dispose]?.();
  }
  return median(times);
}

// Case A: bare associativeScan
async function caseA(n: number) {
  const elems = makeElements(n);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const result = lax.associativeScan(compose, elems) as any;
  result.A[Symbol.dispose](); result.S[Symbol.dispose]();
  elems.A[Symbol.dispose](); elems.b[Symbol.dispose](); elems.S[Symbol.dispose]();
  return result.b;
}

// Case B: jit(assocScan)
const jittedScan = jit((elems: Elem) =>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  lax.associativeScan(compose, elems) as any
);

async function caseB(n: number) {
  const elems = makeElements(n);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const result = await jittedScan(elems) as any;
  elems.A[Symbol.dispose](); elems.b[Symbol.dispose](); elems.S[Symbol.dispose]();
  result.A[Symbol.dispose](); result.S[Symbol.dispose]();
  return result.b;
}

// Case C: lax.scan backward (sequential, mirrors RTS smoother)
async function caseC(n: number) {
  const elems = makeElements(n);
  const carry0 = makeCarry();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [finalCarry, outs] = lax.scan(backwardStep, carry0, elems, { reverse: true }) as any;
  carry0.r[Symbol.dispose](); carry0.N[Symbol.dispose]();
  elems.A[Symbol.dispose](); elems.b[Symbol.dispose](); elems.S[Symbol.dispose]();
  outs.r?.[Symbol.dispose]?.(); outs.N?.[Symbol.dispose]?.();
  finalCarry.N[Symbol.dispose]();
  return finalCarry.r;
}

// Pre-warm jit for Case B
{
  const e0 = makeElements(100);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const r0 = await jittedScan(e0) as any;
  r0.A[Symbol.dispose](); r0.b[Symbol.dispose](); r0.S[Symbol.dispose]();
  e0.A[Symbol.dispose](); e0.b[Symbol.dispose](); e0.S[Symbol.dispose]();
}

console.log(`
lax.associativeScan + lax.scan dispatch repro
backend: webgpu/float32  m=${M}  warmup=${WARMUP}  runs=${RUNS} (median)

  A: bare lax.associativeScan(compose, elems)
  B: jit(fn)(elems)  where fn calls lax.associativeScan
  C: lax.scan(backwardStep, carry, elems, {reverse:true})  <- sequential

ratio ~1.0x per doubling of N  =>  O(log n) / O(1): kernel fusion working
ratio ~2.0x per doubling of N  =>  O(n): per-step JS dispatch happening
`);

const cols = [8, 14, 9, 14, 9, 13, 9];
console.log([
  "N".padStart(cols[0]),
  "A:assocScan".padStart(cols[1]),  "ratio".padStart(cols[2]),
  "B:jit(scan)".padStart(cols[3]),  "ratio".padStart(cols[4]),
  "C:lax.scan".padStart(cols[5]),   "ratio".padStart(cols[6]),
].join("  "));
console.log("-".repeat(cols.reduce((a, b) => a + b) + cols.length * 2));

let [pA, pB, pC] = [null as number | null, null as number | null, null as number | null];
for (const n of N_VALUES) {
  const msA = await timedRun(n, caseA);
  const msB = await timedRun(n, caseB);
  const msC = await timedRun(n, caseC);
  const r = (ms: number, p: number | null) => p !== null ? (ms/p).toFixed(2)+"x" : "-";
  console.log([
    String(n).padStart(cols[0]),
    msA.toFixed(1).padStart(cols[1]),  r(msA,pA).padStart(cols[2]),
    msB.toFixed(1).padStart(cols[3]),  r(msB,pB).padStart(cols[4]),
    msC.toFixed(1).padStart(cols[5]),  r(msC,pC).padStart(cols[6]),
  ].join("  "));
  [pA, pB, pC] = [msA, msB, msC];
}

console.log(`
Expected (ideal): A~1x (assocScan fused), B~1x (same inside jit),
                  C~2x (sequential scan is O(n) depth, inherent)
If B~2x (!=A), jit() changes associativeScan dispatch behaviour.
If C >> A in absolute time, lax.scan dispatches each step from JS
  rather than emitting a native GPU loop -- that is the remaining
  bottleneck in dlm-js dlmFit WebGPU after the assocScan fix.
`);
