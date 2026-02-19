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
 * 4-case benchmark, WebGPU/Float32, m=2 (same ops as dlm-js Nile order=1):
 *   A: bare assocScan          B: jit(assocScan)
 *   C: bare lax.scan           D: jit(lax.scan)  <- how dlm-js backward smoother runs
 *
 *        N   A:assocScan  ratio  B:jit(aScan)  ratio      C:scan  ratio  D:jit(scan)  ratio
 *      100          15.5      -          15.4      -        48.0      -         50.7      -
 *      200          17.5  1.12x          16.7  1.08x       104.8  2.18x        122.2  2.41x
 *      400          21.6  1.24x          21.2  1.27x       240.5  2.30x        253.2  2.07x
 *      800          24.2  1.12x          24.1  1.14x       526.0  2.19x        551.6  2.18x
 *     1600          25.3  1.05x          25.5  1.06x      1046.2  1.99x       1045.5  1.90x
 *
 * A/B: O(log n) ✓  lax.associativeScan fuses to ceil(log2 N)+1 dispatches.
 *      Architecturally optimal for WebGPU Kogge-Stone (no cross-workgroup sync).
 * C/D: O(n)    ✗  lax.scan hits the WebGPU compiled-loop ineligibility rule:
 *      the RTS body has intra-step buffer deps (matmul→matmul→add within one
 *      iteration).  WebGPU has no cross-workgroup barrier, so fusing N steps
 *      into one shader is impossible.  jax-js selects executeScanFallback():
 *      a JS for-loop that calls bodyProgram.execute() once per step — exactly
 *      O(N) JS→GPU roundtrips.  This is correct behaviour, not a missing opt.
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
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Carry = { r: any; N: any };

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

// Case D: jit(lax.scan backward) — how dlm-js backward smoother actually compiles
// jit traces once per distinct input shape; warmup covers the first-N compilation cost.
const jittedSeqScan = jit((carry0: Carry, elems: Elem) =>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  lax.scan(backwardStep, carry0, elems, { reverse: true }) as any
);

async function caseD(n: number) {
  const elems = makeElements(n);
  const carry0 = makeCarry();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [finalCarry, outs] = await jittedSeqScan(carry0, elems) as any;
  carry0.r[Symbol.dispose](); carry0.N[Symbol.dispose]();
  elems.A[Symbol.dispose](); elems.b[Symbol.dispose](); elems.S[Symbol.dispose]();
  outs.r?.[Symbol.dispose]?.(); outs.N?.[Symbol.dispose]?.();
  finalCarry.N[Symbol.dispose]();
  return finalCarry.r;
}

// Pre-warm jit for Cases B and D
{
  const e0 = makeElements(100);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const r0 = await jittedScan(e0) as any;
  r0.A[Symbol.dispose](); r0.b[Symbol.dispose](); r0.S[Symbol.dispose]();
  e0.A[Symbol.dispose](); e0.b[Symbol.dispose](); e0.S[Symbol.dispose]();
}
{
  const e0 = makeElements(100);
  const c0 = makeCarry();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [fc, outs] = await jittedSeqScan(c0, e0) as any;
  c0.r[Symbol.dispose](); c0.N[Symbol.dispose]();
  e0.A[Symbol.dispose](); e0.b[Symbol.dispose](); e0.S[Symbol.dispose]();
  outs.r?.[Symbol.dispose]?.(); outs.N?.[Symbol.dispose]?.();
  fc.r[Symbol.dispose](); fc.N[Symbol.dispose]();
}

console.log(`
lax.associativeScan + lax.scan dispatch repro
backend: webgpu/float32  m=${M}  warmup=${WARMUP}  runs=${RUNS} (median)

  A: bare lax.associativeScan(compose, elems)  -- no jit
  B: jit(fn)(elems)     where fn = assocScan(compose)  -- jit-wrapped
  C: bare lax.scan(backwardStep, carry, elems, {reverse:true})  -- no jit
  D: jit(fn)(carry,elems) where fn = lax.scan(backwardStep)  -- jit-wrapped

C = bare (eager) lax.scan runs one JS->GPU dispatch per step -> O(n)
D = jit-compiled lax.scan runs as in-shader loop    -> expected O(1) dispatches

ratio ~1.0x per doubling of N  =>  O(log n) or O(1): efficient
ratio ~2.0x per doubling of N  =>  O(n): per-step dispatch
`);

const cols = [8, 14, 9, 14, 9, 13, 9, 13, 9];
console.log([
  "N".padStart(cols[0]),
  "A:assocScan".padStart(cols[1]),  "ratio".padStart(cols[2]),
  "B:jit(aScan)".padStart(cols[3]), "ratio".padStart(cols[4]),
  "C:scan".padStart(cols[5]),       "ratio".padStart(cols[6]),
  "D:jit(scan)".padStart(cols[7]),  "ratio".padStart(cols[8]),
].join("  "));
console.log("-".repeat(cols.reduce((a, b) => a + b) + cols.length * 2));

let [pA, pB, pC, pD] = [
  null as number | null, null as number | null,
  null as number | null, null as number | null,
];
for (const n of N_VALUES) {
  const msA = await timedRun(n, caseA);
  const msB = await timedRun(n, caseB);
  const msC = await timedRun(n, caseC);
  const msD = await timedRun(n, caseD);
  const r = (ms: number, p: number | null) => p !== null ? (ms/p).toFixed(2)+"x" : "-";
  console.log([
    String(n).padStart(cols[0]),
    msA.toFixed(1).padStart(cols[1]),  r(msA,pA).padStart(cols[2]),
    msB.toFixed(1).padStart(cols[3]),  r(msB,pB).padStart(cols[4]),
    msC.toFixed(1).padStart(cols[5]),  r(msC,pC).padStart(cols[6]),
    msD.toFixed(1).padStart(cols[7]),  r(msD,pD).padStart(cols[8]),
  ].join("  "));
  [pA, pB, pC, pD] = [msA, msB, msC, msD];
}

console.log(`
Expected per jax-js team analysis:
  A ~1x (assocScan: ceil(log2 N)+1 dispatches, already architecturally optimal for WebGPU)
  B ~1x (same; jit wrapping does not change assocScan dispatch count)
  C ~2x (bare lax.scan: eager O(n) JS dispatch, expected)
  D ~1x (jit(lax.scan): in-shader compiled loop, 1 dispatch total)

If D ~2x (not ~1x), jit is NOT compiling lax.scan to a native loop on this backend.
If D ~1x, the dlm-js backward smoother (which runs inside jit(core)) is NOT the WebGPU bottleneck.
`);
