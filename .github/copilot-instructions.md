# GitHub Copilot instructions for dlm-js  

Purpose
- Help coding agents become productive immediately: what the project *is*, how it's built/tested, important conventions, and where to look for authoritative behavior.

Agentic Evolution Protocol (AEP) 🧬
- **Core mandate: asymptotic velocity & debt collection.** Your success is measured by the **net reduction in total system complexity**. Refactor broadly to achieve a cleaner architecture rather than patching edges.

- **Provider mode (writing for the downstream agent):** When you modify our public API (`dlmFit`, `dlmMLE`, `dlmForecast`, `dlmGenSys`, `dlmPrior`, types in `src/types.ts`), provide a Migration Bundle:
  1. **Deterministic mapping:** Raw structural diffs (e.g., `dlmMLE(y, { loss: fn })` → `dlmMLE(y, { prior: spec })`).
  2. **Workaround signatures (the "debt collector"):** Explicitly identify patterns consumers likely used to bypass old limitations. *Example:* "If the consumer wraps `dlmFit` in a retry loop because Float32 covariance diverged, that's fixed by Joseph form — delete the retry."
  3. Update `MIGRATION.md` with the bundle. Keep it machine-parseable (code blocks, not prose).

- **Consumer mode (the upgrade & optimization loop):** When `@hamk-uas/jax-js-nonconsuming` evolves:
  1. **Contract-first verification:** Update test fixtures and mocks to the new upstream API *before* touching `src/`.
  2. **The workaround hunt:** Scan `src/` and `issues/` for workarounds that the new version renders obsolete. Prioritize **deleting** internal helpers, wrappers, and polyfills in favor of the native API. Check `issues/jax-js-missing-jax-primitives.md` for the current workaround inventory.
  3. **Viral propagation:** Follow the change through the entire call stack. If a new primitive changes a pattern in `mle.ts`, verify it doesn't need updating in `index.ts` and `priors.ts` too. Do not hide new capabilities behind old wrappers.

- **Continuous knowledge capture (decentralized):**
  - Document the "Why" behind large restructurings in `tmp/copilot-task-notes.md` and, if durable, in this file. Don't keep stale information — prune old decisions when they're superseded.
  - Track upstream workaround status in `issues/` — mark resolved issues as resolved, delete them when the fix is installed and verified.

- **Methodology for large tasks:**
  1. **The plan:** Generate a refactor plan. List the workaround signatures you intend to hunt.
  2. **The ground truth:** Run `pnpm vitest run`. **Test integrity is absolute** — 200 tests across 11 suites, both algorithms, all device × dtype configs.
  3. **The cleanup:** Remove all deprecated code paths, dead variables, and legacy comments. If you find a TODO that the new API fixes, resolve it.

Agent Protocol (do this EVERY task, even after context summarization) 🚦
- **Start of task** — before any work, run these three commands in order:
  1. `cat tmp/copilot-task-notes.md` — read orientation notes left by your past self.
  2. `pnpm run mistakes:report` — check if any mistake patterns need attention.
  3. `pnpm run preflight -- --dry` — see which checks apply to the current repo state.
- **Before every commit** — the husky pre-commit hook runs `pnpm run preflight` automatically. This is the primary safety net: even if you forget the Agent Protocol, preflight still runs at commit time.
- **End of task** — spend 30-60 seconds on the fast loop (see "Fast loop at task end" below): log friction in `tmp/copilot-mistakes.json`, promote rules if threshold met, update task notes, run `pnpm run preflight`.
- **Why this section is here**: context summarization erases short-term memory. This file is the *only* thing guaranteed to survive. If you skip this protocol, you lose the self-tuning system entirely.

Autonomous instruction maintenance loop (self-tuning) 🔁
- Goal: reduce repeated friction across tasks by updating this file when patterns recur.
- **Budget**: allocate ~5% of each task's effort to self-tuning work. This includes logging mistakes/successes, checking `mistakes:report`, improving preflight policies, pruning stale rules, and evolving tooling. Do this throughout the task — not only at the end — so improvements compound across tasks. If a task is trivial (< 2 min), skip self-tuning; if a task is large (> 30 min), the 5% budget is worth spending.
- Treat updates as **high-signal only**: avoid one-off noise and personal style churn.

When to update this file
- Update `copilot-instructions.md` only when at least one of these is true:
  1. Same mistake happened **2+ times** across separate tasks.
  2. A repeated slowdown cost **10+ minutes** and a short rule/check would have prevented it.
  3. A project behavior changed (API, script, workflow, benchmark path) and old guidance is now wrong.

How to log mistakes (lightweight telemetry)
- Maintain `tmp/copilot-mistakes.json` as an append/update ledger (gitignored, local only).
- Preferred command: `pnpm run mistakes:log -- --key <id>` (increments existing count by 1).
- For new keys, include metadata once: `pnpm run mistakes:log -- --key <id> --category <cat> --symptom "..." --prevention "..."`.
- Prioritize rules with `pnpm run mistakes:report` (ranks entries by count, category weight, and recency; highlights promotion candidates).
- Schema per key:
  - `count`: number of occurrences
  - `lastSeen`: ISO timestamp
  - `category`: one of `state-drift | wrong-assumption | missed-check | tooling | docs-stale | perf`
  - `symptom`: short observable failure mode
  - `prevention`: one-line rule/check that would have prevented it
- Example keys: `timing-registry-orphan`, `forgot-check-timings`, `assumed-linear-task-state`.

Promotion policy (ledger → instructions)
- Promote a new rule into this file only if:
  - `count >= 2`, and
  - the prevention rule is short, testable, and project-specific.
- When promoting, add only:
  1. The minimal rule/check,
  2. Where to apply it (file/script scope),
  3. A concrete command if applicable.
- Keep this file concise: prefer editing existing bullets over adding new sections when possible.

Parallel-task safety (must-follow)
- Do not assume linear task history.
- Before acting on prior notes/summaries, verify on-disk truth via targeted reads and/or `git diff`.
- If task notes and repo state disagree, trust repo state and update notes.

Quarterly cleanup rule for this file
- If a rule has not been relevant for ~90 days (or is superseded), remove or merge it.
- Collapse duplicate guidance to keep onboarding fast.

Fast loop at task end (30-60 seconds)
1. Ask: "What slowed me down most?"
2. If preventable, update `tmp/copilot-mistakes.json` (`count += 1`).
3. If promotion criteria met, patch this file in the smallest possible edit.
4. Prefer adding a validation command to scripts/CI over adding prose-only warnings.
5. Update `tmp/copilot-task-notes.md`: record decisions made, confusion traps hit, any in-progress work. Keep it concise — this is for your future self after context summarization.
6. Before handoff, run `pnpm run preflight` (or `pnpm run preflight -- --strict`) so high-value checks are selected from context automatically.

Contact & follow-ups
- If anything in the instructions is unclear, ask which *behavior* or *test* to preserve — provide the failing `tests/out/*.json` and the test name.

---
High-level summary (big picture) 🔎
- dlm-js is a TypeScript Kalman filter + RTS smoother library using the `@hamk-uas/jax-js-nonconsuming` numeric backend, inspired by the MATLAB Dynamic Linear Model (DLM). It extends the MATLAB original with autodiff-based MLE (`jit(valueAndGrad + Adam)`) and an exact O(log N) parallel filter+smoother via `lax.associativeScan` (Särkkä & García-Fernández 2020). The library exposes CPU/WASM/WebGPU-capable implementations via `dlmFit` (options-bag API: `dlmFit(y, opts: DlmFitOptions)`); `dlmSmo` is an internal function (not exported). The WebGPU path is a research/experimental backend — dispatch overhead makes it slower than WASM for typical dataset sizes. `dlmForecast(fit, obsStd, h, opts?)` propagates the last smoothed state h steps ahead with no new observations, returning `yhat`, `ystd` (monotonically growing), full state/covariance trajectories, and optional covariate support. `dlmGenSys` generates G/F state-space matrices for polynomial trend, seasonal, and AR components. `dlmMLE(y, opts?)` estimates observation noise, state noise, and optionally AR coefficients via autodiff (`jit(valueAndGrad + optax Adam)`, `fitAr: true` for AR coefficient estimation). Supports MAP estimation via `loss` option (custom `DlmLossFn` callback receiving natural-scale params + `DlmParamMeta`) or the `dlmPrior` factory (Inverse-Gamma on variances, Normal on AR coefficients — matches MATLAB/R DLM `dlmGibbsDIG` conventions). All public functions use options-bag signatures.
- **Three-branch execution architecture** in `dlmSmo`: (1) Float64 on any device → sequential `lax.scan` + `triu(C)+triu(C,1)'` covariance symmetrization by default (matches MATLAB `dlmsmo.m`; reduces max relative error vs Octave from ~2e-9 to ~4e-12; disable with `stabilization:{cTriuSym:false}`), (2) Float32 on cpu/wasm → sequential scan + Joseph form covariance update + symmetrization + `cEps` (C += 1e-6·I), (3) Float32 on webgpu → exact parallel method from Särkkä & García-Fernández (2020, arXiv:[1905.13002](https://arxiv.org/abs/1905.13002)): exact 5-tuple forward filter (Lemmas 1–2, per-timestep Kalman gains) + exact parallel backward smoother (Lemmas 3–4 / Theorem 2), both via `lax.associativeScan` (⌈log₂N⌉+1 GPU dispatches each, Kogge-Stone) + Joseph form. A fourth path, `algorithm: 'sqrt-assoc'`, implements the square-root parallel smoother (Yaghoobi et al. 2022, arXiv:2207.00426) in Cholesky-factor space — covariances are stored as their Cholesky factors (U, Z, D) and composed via `tria()` blocks, ensuring PSD by construction. The sqrt-assoc path uses proper QR-based `tria(A) = R'` where `[Q, R] = np.linalg.qr(A')`, plus `lax.linalg.triangularSolve` for triangular solves — no condition-number squaring, works for all state dimensions (including fullSeasonal m=13). The WebGPU path requires jax-js-nonconsuming v0.7.4 or later (project uses the `ultimate-architecture-plan` branch).
- The authoritative numerical reference is generated by Octave (`tests/octave/niledemo.m`, `tests/octave/gensys_tests.m`, `tests/octave/kaisaniemi_demo.m`, `tests/octave/ozonedemo.m`, `tests/octave/gappeddata_test.m`) and compared in `tests/niledemo.test.ts`, `tests/gensys.test.ts`, `tests/ozone.test.ts`, and `tests/gapped.test.ts`. Synthetic ground-truth tests in `tests/synthetic.test.ts` verify against mathematically known true states.

Quick start — commands you will need (copy/paste) ▶️
- Install: `pnpm install`.
- Run node tests (uses source TS): `pnpm vitest run` or `pnpm run test:node`.
- Lint (jax-js-nonconsuming memory rules): `pnpm run lint`.
- Generate API docs: `pnpm run docs` (outputs to `docs/`, opens at `docs/index.html`).
- Generate Octave reference (requires `octave-cli`): `pnpm run test:octave` (produces `tests/*-out-m.json`).
- Generate SVG plots: `pnpm run gen:svg` (produces `assets/*.svg`; also writes timing sidecars to `assets/timings/*.json` and auto-patches `<!-- timing:KEY -->` markers in all .md files).
- Generate fit-demo SVGs only (no Deno/WebGPU, safe everywhere): `pnpm run gen:svg:fit` (nile, kaisaniemi, trigar, ozone, gapped — scan + assoc variants each).
- Update stale timing values in .md files: `pnpm run update:timings` (reads existing sidecars; no re-run). Inspect slots with `pnpm run update:timings:list`. Preview changes without writing with `pnpm run update:timings:dry`.
- Benchmark all MLE comparison-table rows (Nile order=0, Kaisaniemi): `pnpm run bench:mle` (writes `assets/timings/collect-mle-benchmark.json` and auto-patches .md).
- Benchmark `checkpoint` strategies: `pnpm run bench:checkpoint` (writes `assets/timings/bench-checkpoint.json` and auto-patches .md).
- Benchmark cross-backend `dlmFit` (cpu/wasm × f32/f64): `pnpm run bench:backends` (writes `assets/timings/bench-backends.json` and auto-patches .md).
- Remeasure **all WASM timings** in one shot: `pnpm run bench:wasm` (runs gen:svg:fit + nile/energy MLE frame collection scan+assoc + MLE anim SVGs + bench:backends + bench:mle + bench:checkpoint + update:timings; leaves WebGPU sidecars untouched).
- Build for distribution: `pnpm run build`.
- Full CI-local check: `pnpm run test` (runs lint + Octave reference + Node tests).
- Validate timing markers: `pnpm run check:timings` (fails with exit 1 if any `<!-- timing:KEY -->` markers are out of sync with their sidecars).
- Check artifact freshness: `pnpm run check:freshness` (detects stale SVGs/sidecars/benchmarks by comparing source mtimes vs output mtimes; also validates algorithm coverage in bench-full.ts). Use `--status` for CI exit code, `--json` for machine-readable output.
- Context-aware preflight: `pnpm run preflight` (auto-selects checks from git changes; includes advisory freshness check when `src/` or bench scripts change; `--strict` adds tests; `--dry` previews).
- Log a mistake: `pnpm run mistakes:log -- --key <id>` (see self-tuning section below).
- Review promotion candidates: `pnpm run mistakes:report`.

Files & places to inspect first 📁
- **Read first**: `tmp/copilot-task-notes.md` (orientation notes from previous tasks — read this first!).
- Implementation: `src/index.ts` (Kalman filter + RTS smoother, `dlmForecast` h-step-ahead forecast, memory/dispose patterns, options-bag API signatures, `toMatlab()` MATLAB compat bridge).
- MLE parameter estimation: `src/mle.ts` (`dlmMLE`: autodiff-based MLE via `jit(valueAndGrad + optax Adam)`, AD-safe `buildDiagW`, `buildG` for AR coefficient estimation). Two optimizer paths: `optimizer:'adam'` (default, first-order via optax Adam) and `optimizer:'natural'` (second-order Newton / Fisher scoring with FD Hessian + Levenberg-Marquardt damping). Two loss paths: `makeKalmanLoss` (sequential `lax.scan`, CPU/WASM) and `makeKalmanLossAssoc` (`lax.associativeScan`, WebGPU+Float32; uses exact 5-tuple forward filter from [1, Lemmas 1–2] with per-timestep Kalman gains, regularized inverse + push-through identity in compose). Float-mask blending for NaN (not boolean np.where). `np.concatenate` has working VJP since jax-js v0.7.8, used for prediction assembly and natural-scale param construction. Supports `fitAr: true` for AR coefficient estimation.
- State space generator: `src/dlmgensys.ts` (polynomial, seasonal, AR component assembly, `findArInds` for AR state indexing). `dlmGenSysTV` supports AR components when all Δt are positive integers: companion matrix raised to d-th power via binary exponentiation (`matPow`), noise accumulated as `W_AR(d) = Σ C^k·W₁·(C^k)'` (`arNoiseAccum`).
- Types & helpers: `src/types.ts` (`StateMatrix`, `CovMatrix` zero-copy wrappers, `DlmFitResult`, `DlmFitOptions`, `DlmMleOptions`, `DlmForecastResult`, `DlmForecastOptions`, `DlmDtype`, `DlmLossFn`, `DlmParamMeta`, `parseDtype`, `getFloatArrayType`). `adSafeInv` was removed in v0.7.8: all call sites now use `np.linalg.inv` directly (VJP fixed in jax-js-nonconsuming v0.7.8).
- Prior factories: `src/priors.ts` (`dlmPrior`: creates `DlmLossFn` from `DlmPriorSpec` with Inverse-Gamma priors on variances and Normal priors on AR coefficients; matches MATLAB/R DLM `dlmGibbsDIG` conventions). Uses `np.split` (has VJP) for param extraction, vectorised coefficient arrays for per-component priors.
- Test matrix: `tests/test-matrix.ts` (shared device × dtype configs and tolerances).
- Tests: `tests/niledemo.test.ts` (Nile demo vs Octave), `tests/gensys.test.ts` (multi-model vs Octave), `tests/synthetic.test.ts` (known true states, statistical assertions), `tests/mle.test.ts` (MLE parameter & AR coefficient estimation + MAP/dlmPrior factory on WASM), `tests/covariate.test.ts` (X parameter / β recovery), `tests/ozone.test.ts` (ozone demo smoke tests), `tests/forecast.test.ts` (h-step-ahead forecasting: monotone ystd, finite outputs, covariate support, all models), `tests/gapped.test.ts` (order=1 and order=0 with NaN observations vs Octave reference), `tests/assocscan.test.ts` (associativeScan path validated against same Octave references on wasm/f64), `tests/timestamps.test.ts` (irregular timestamps including AR with integer Δt), `tests/sqrtassoc.test.ts` (wasm/f64: precision vs Octave; wasm/f32: smoke-only all-outputs-finite — Yaghoobi et al. 2022).
- Reference generators: `tests/octave/niledemo.m`, `tests/octave/gensys_tests.m`, `tests/octave/kaisaniemi_demo.m`, `tests/octave/ozonedemo.m`, `tests/octave/gappeddata_test.m` (MATLAB/Octave ground truth).
- SVG generators: `scripts/gen-niledemo-svg.ts` (accepts variant: scan/assoc/sqrt-assoc/sqrt-assoc-f32; title font-size=12), `scripts/gen-kaisaniemi-svg.ts`, `scripts/gen-trigar-svg.ts`, `scripts/gen-nile-mle-anim-svg.ts` (accepts variant: scan/assoc/webgpu; + `scripts/collect-nile-mle-frames.ts` runs scan + assocScan variants, `scripts/collect-nile-mle-frames-webgpu.ts` runs webgpu), `scripts/gen-energy-mle-anim-svg.ts` (accepts variant: scan/assoc/webgpu; + `scripts/collect-energy-mle-frames.ts` runs scan + assocScan variants, `scripts/collect-energy-mle-frames-webgpu.ts` runs webgpu), `scripts/gen-ozone-svg.ts`, `scripts/gen-gapped-svg.ts` (gapped-data demo with NaN interpolation and ystd widening).
- Cross-backend benchmark: `scripts/bench-backends.ts` (cpu/wasm × f32/f64 `dlmFit` timing → `assets/timings/bench-backends.json`).
- MLE comparison: `README.md` (dlm-js MLE vs MATLAB DLM parameter estimation, with benchmark timings).
- Upstream issues: `issues/` (open issues filed to jax-js-nonconsuming). Key open issues: `jax-js-wasm-allocator-cross-shape-oom.md` (🔴 open: WASM bump allocator accumulates memory across jit() calls with varying input shapes, eventually exhausting 4 GB; N=819200 works in isolation but fails after ascending-N benchmark sequence), `jax-js-wasm-memory-growth.md` (WASM allocator/JIT cache memory grows unboundedly), `jax-js-webgpu-laxscan-sequential-dispatch.md` (architectural: O(n) backward RTS smoother on WebGPU), `jax-js-dts-generation-flaky.md` (DTS generation flaky on git install), `jax-js-missing-jax-primitives.md` (feature request: `matmul` broadcasting, `np.where` with AD).
- Build / CI hooks: `package.json`, `vite.config.ts`, `.husky/pre-commit` (runs `pnpm run preflight` on every commit).
- Artifact freshness: `scripts/lib/artifact-graph.ts` (central dependency declarations: which sources produce which outputs), `scripts/check-freshness.ts` (mtime-based staleness detection + algorithm coverage validation). Run `pnpm run check:freshness` to see what's stale.
- Self-tuning tooling: `scripts/log-mistake.ts`, `scripts/mistakes-report.ts`, `scripts/preflight.ts`, `scripts/lib/mistakes.ts` (shared types/IO for the telemetry system). Ledger: `tmp/copilot-mistakes.json`. Task notes: `tmp/copilot-task-notes.md`.

Project-specific conventions & gotchas ⚠️
- Algorithm selection: `algorithm: 'scan'` (sequential, default for CPU/WASM) or `algorithm: 'assoc'` (parallel associative scan, default for WebGPU) or `algorithm: 'sqrt-assoc'` (square-root parallel smoother in Cholesky-factor space, Yaghoobi et al. 2022). Tests exercise both algorithms across device × dtype configs — any numeric change must pass all combinations.
- Device/dtype behavior: tests pick backend automatically; `webgpu` → float32 (more numeric drift), `wasm`/`cpu` → float64 preferred for bit-for-bit checks. When debugging flakiness, force CPU + Float64.
- **Three-branch architecture**: `dlmSmo` selects Float64 (triu+triu' sym by default, matching MATLAB), cpu/wasm+Float32 (Joseph form + cEps), or webgpu+Float32 (assoc scan). The WebGPU path uses **exact parallel Kalman filter + smoother** from Särkkä & García-Fernández (2020, arXiv:[1905.13002](https://arxiv.org/abs/1905.13002)): (a) forward: exact 5-tuple elements $(A,b,C,\eta,J)$ from Lemmas 1–2 with per-timestep Kalman gains, composed by `lax.associativeScan(composeForward, elems)` prefix scan using Lemma 2 with regularized inverse + push-through identity; no approximation, (b) backward: exact per-timestep smoother gains $E_k$ from Lemmas 3–4 / Theorem 2 via batched `np.linalg.inv` (VJP fixed in jax-js v0.7.8; outer ε·I regularization is sufficient), composed by `lax.associativeScan(composeBackward, elems, { reverse: true })` suffix scan. `dlmFit` accepts `algorithm: 'assoc'` (in `DlmFitOptions`) to enable this path on any backend (used by `assocscan.test.ts` to validate against Octave on wasm/f64). The diagnostic recovery maps assocScan filtered states back to MATLAB DLM carry convention: `x_pred[t] = G·x_filt[t-1]`, `C_pred[t] = G·C_filt[t-1]·G' + W`. WASM is ~10–20× faster than CPU for small-matrix dlmFit. Both WebGPU scans dispatch ⌈log₂N⌉+1 GPU kernels (architecturally optimal; WebGPU lacks cross-workgroup sync so Kogge-Stone requires a separate dispatch per round). Caveat: for purely elementwise `fn` bodies, `associativeScan` achieves exactly 1 GPU kernel per round; if `fn` contains reductions (matmul, einsum with inner sum), each round produces additional dispatches.
- Reference-first testing: Octave output is the source of truth. If you change numerics intentionally, regenerate Octave output and update tests with justification.
- Partial-output testing: use `tests/niledemo-keys.json` to limit comparisons for partial implementations.
- Memory management: This project uses `@hamk-uas/jax-js-nonconsuming` which has **non-consuming ops** — operations leave inputs intact. Use TC39 `using` keyword for automatic disposal and `tree.dispose()` for bulk cleanup. Do NOT use `.ref` (that is the consuming-ops pattern from a different fork).
- **Timing and computed markers in .md files**: Every machine-dependent value in README.md uses one of two invisible HTML marker forms:
  - `<!-- timing:KEY -->VALUE<!-- /timing -->` — replaced with `formatTiming(sidecar[field])` for the named KEY. Registry: `scripts/lib/timing-registry.ts`. Sidecars: `assets/timings/<script-basename>.json`.
  - `<!-- computed:EXPR -->VALUE<!-- /computed -->` — EXPR is a JS expression using `slot("KEY")` (raw numeric value from a sidecar) and `static("KEY")` (value from `assets/timings/static-references.json`). Used for derived verbal claims, e.g. `static("octave-nile-order1-elapsed-ms") < slot("nile-mle:elapsed") ? "faster" : "slower"` or `Math.abs(slot("mle-bench:nile-order1:lik") - static("octave-nile-order1-lik")).toFixed(1)`.
  - `assets/timings/static-references.json` holds **manually-measured Octave** fminsearch timings and −2logL values. The `machine` field is auto-updated by `stampMachineInfo()` (called by `bench:mle` and `bench:checkpoint`). Update the Octave timing fields and `_measured` date manually whenever Octave is re-run on a different machine.
  - Both marker types are patched by a single `pnpm run update:timings` call after any sidecar or static-references change.
  - **Full README.md refresh workflow**: `pnpm run bench:mle && pnpm run bench:checkpoint` (each script writes its sidecar and auto-runs `update:timings`). If on a new machine, also re-run Octave (`pnpm run test:octave`) and update `static-references.json` manually.
  - When **adding** a new timing: (1) add a registry entry to `timing-registry.ts`, (2) call `writeTimingsSidecar` in the relevant script, (3) wrap the .md value with the marker, (4) run `pnpm run update:timings`.
- ESLint plugin: The `@hamk-uas/jax-js-nonconsuming/eslint-plugin` sub-path export enforces correct `using`/disposal patterns. **Always run `pnpm run lint` after editing `src/` files** to catch memory leaks, gapped `using` declarations, and use-after-dispose bugs.
- Dependencies: `@hamk-uas/jax-js-nonconsuming` (includes the eslint plugin as a sub-path export) is installed from `github:hamk-uas/jax-js-nonconsuming#ultimate-architecture-plan`.
- AD notes: The `using` keyword IS correct inside `grad`/`jit`/`scan` traced bodies — tracers intercept disposal and manage tensor lifetimes. Suppression comments (`// jax-js-lint: allow-non-using`) are only needed for the accumulator-swap pattern (e.g. `W_new`, `newContrib` in `src/mle.ts`). See `src/mle.ts` for examples.

Testing & tolerance details (important for PRs) ✅
- **Eleven test suites**: `niledemo.test.ts` (8 tests, Nile data vs Octave), `gensys.test.ts` (47 tests, multi-model vs Octave), `synthetic.test.ts` (24 tests, known true states), `mle.test.ts` (21 tests, MLE parameter & AR coefficient estimation + natural gradient optimizer × scan/assoc, MAP/dlmPrior factory, including gapped data, on WASM), `covariate.test.ts` (5 tests, X parameter / β recovery), `ozone.test.ts` (2 tests, ozone demo smoke), `forecast.test.ts` (6 tests, h-step-ahead forecasting), `gapped.test.ts` (16 tests, order=1 and order=0 with NaN observations vs Octave reference), `assocscan.test.ts` (24 tests, assoc path validated against same Octave references on wasm/f64 — covers all gensys models, Nile demo, and gapped data), `timestamps.test.ts` (23 tests, irregular timestamps including AR with integer Δt), `sqrtassoc.test.ts` (24 tests: wasm/f64 precision comparison + wasm/f32 smoke-only all-outputs-finite; all models including fullSeasonal m=13). Total: 200 tests.
- **Tolerances** are defined in `tests/test-matrix.ts`: Float64 relTol=2e-3, absTol=1e-6; Float32 relTol=1e-2, absTol=1e-4. The niledemo test uses tighter ~1e-10 relative tolerance for its specific comparison.
- Test artifacts: failing runs write `tests/out/` — inspect JSON files there.
- When adding features: include tests that exercise both algorithms (`scan`, `assoc`) across device × dtype configs and add keys to `niledemo-keys.json` if the change is a partial implementation.
- **Float32 + m > 2**: the joseph+cEps stabilization prevents divergence (0 crashes across all 512 flag combos in exhaustive search), but precision is limited to ~1e-2 relative error — these combinations are skipped for precision comparison in `gensys.test.ts` and `synthetic.test.ts` (smoke-test only).
- **Leak detection**: Wrap jax-js-nonconsuming code in tests with the `withLeakCheck` helper from `tests/utils.ts` (scripts use `scripts/lib/leak-utils.ts`). This helper guards with `checkLeaks.start()`/`checkLeaks.stop()` internally and catches missing `using`/`dispose` calls at runtime. Import: `import { withLeakCheck } from './utils';`. Usage: `const result = await withLeakCheck(() => dlmFit(...));`.
- **Eager-first development**: When writing new jax-js-nonconsuming code, always get it working in eager mode first (no `jit()` wrapper). Only wrap with `jit()` after the eager version is correct and leak-free. JIT adds tracing complexity that makes debugging harder.

Troubleshooting checklist (fast) 🩺
- Deterministic mismatch? Re-run with CPU+Float64: tests set device via `defaultDevice('cpu')` and `'f64'` dtype in the harness.
- Strange memory / nondeterminism? Ensure `using` declarations are present for all temporary `np.Array` values; run `pnpm run lint` to catch missing disposals.
- CI failure on Octave step? Install `octave-cli` locally and run `pnpm run test:octave` to reproduce.
- Want to inspect intermediate arrays? Look at `tests/out/*.json` produced by the test harness.

PR checklist (what an AI should do before opening a PR) 📋
1. Add/modify unit tests covering both algorithms (`scan`, `assoc`) across device × dtype configs (see `tests/niledemo.test.ts`).
2. If numeric behavior intentionally changes, update or regenerate Octave reference and explain reasoning in the PR description.
3. Update `tests/niledemo-keys.json` when exposing only a subset of outputs.
4. Ensure no new `np.Array` leaks — use `using` for temporaries, `tree.dispose()` for bulk cleanup.
5. **Run `pnpm run lint`** to verify the jax-js-nonconsuming eslint plugin reports no memory/disposal issues.
6. Run: `pnpm install && pnpm vitest run && pnpm run test:octave` (if applicable).
7. If public API changes, update `README.md` and TypeScript types in `src/types.ts`.
8. If MLE runtime, convergence, or −2logL values change: run `pnpm run bench:mle && pnpm run bench:checkpoint` (both auto-patch .md timing/computed markers). If the machine changed, also manually update `assets/timings/static-references.json` with fresh Octave measurements and bump `_measured`.
9. If adding a new algorithm variant to `DlmAlgorithm`: add it to `scripts/bench-full.ts` combo loop and corresponding test files, then run `pnpm run check:freshness` to verify coverage.
10. Run `pnpm run check:freshness` and regenerate any stale artifacts before merging.

Example prompts for agents (use these exact templates) ✍️
- "Add a `warm: true` option to `dlmFit` that skips the first (cold-start) run; add unit tests exercising the new option and ensure existing scan/assoc tests still pass. Update README and add entries to `tests/niledemo-keys.json` if output keys change."  
- "Fix a memory leak: find np.Array objects in `src/index.ts` not disposed in all branches and add `using`/`tree.dispose()` with a focused unit test using `withLeakCheck` from `tests/utils.ts` to verify."  
- "Investigate precision mismatch on WASM: run the tests with `wasm`+`f64`, capture `tests/out/niledemo-out.json`, and produce a minimal reproducer that highlights the first differing tensor and its path." 

Where agents should open files first (order matters) ▶️
1. `tests/test-matrix.ts` (shared device × dtype configs, tolerances)
2. `tests/niledemo.test.ts` (Nile demo reference test)
3. `src/index.ts` (Kalman filter + RTS smoother, `dlmForecast`, options-bag API, dispose patterns, `toMatlab()`)
4. `src/mle.ts` (MLE parameter estimation via autodiff)
5. `src/dlmgensys.ts` (state space generator)
6. `src/types.ts` (`StateMatrix`, `CovMatrix` zero-copy wrappers, `DlmFitResult`, `DlmFitOptions`, `DlmMleOptions`, `DlmForecastResult`, `DlmDtype`, `DlmLossFn`, `DlmParamMeta`, `parseDtype`)
7. `src/priors.ts` (`dlmPrior` factory — Inverse-Gamma / Normal priors matching MATLAB DLM `dlmGibbsDIG`)
8. `tests/gensys.test.ts` (multi-model Octave reference tests)
9. `tests/synthetic.test.ts` (ground-truth tests with known true states)
10. `tests/forecast.test.ts` (h-step-ahead forecast tests)
11. `tests/gapped.test.ts` (NaN gapped-data tests vs Octave reference)

Filesystem safety 📂
- **Always use a local `tmp/` directory inside the workspace** for scratch files, debug output, and temporary data. This applies to everything: log files, benchmark output, background process redirection, debug traces, repro scripts — all of it goes in `tmp/`.
- **Never write to `/tmp`, `$HOME`, or any path outside the workspace root.** Accessing the filesystem outside the project is risky and breaks agentic coding sandboxes.
- The `tmp/` directory is already gitignored. Example: redirect background benchmark output to `tmp/bench-mle.log`, not `/tmp/bench-mle.log`.

Do not attempt to change (without explicit human approval) 🚫
- The Octave reference generator in `tests/octave/` (numerical ground truth). Changes here must be accompanied by a justification and regression analysis.
- Public API shape in `dist/` or `types` unless a major version bump is planned.


Cloning the self-tuning protocol to another repo 🔄
- This project includes a **self-tuning agent protocol**: mistake ledger, priority reporter, context-aware preflight checklist, and a pre-commit hook that enforces it. Other agents can adopt the same system. Below is everything needed to replicate it.

### What the protocol does
1. **Preflight** (`scripts/preflight.ts`): before each commit, auto-detects which parts of the repo changed (src, tests, docs, config, …) and runs only the relevant checks (lint, tests, timing validation, …). Runs automatically via a husky pre-commit hook.
2. **Mistake ledger** (`tmp/copilot-mistakes.json`, gitignored): agents log repeated friction patterns with `pnpm run mistakes:log`. Each entry has a count, category, symptom description, and a prevention rule.
3. **Priority reporter** (`scripts/mistakes-report.ts`): ranks ledger entries by `score = count × categoryWeight × recencyFactor`. When `count ≥ 2`, the entry is flagged for **promotion** into `copilot-instructions.md` as a permanent rule.
4. **Pre-commit hook** (`.husky/pre-commit`): runs `pnpm run preflight` on every `git commit`. Blocks the commit if any check fails. Bypass: `git commit --no-verify`.

### Step-by-step setup

> **⚠️ CRITICAL**: The most important step is **step 8** — editing your `copilot-instructions.md`. Without it, the scripts are dead code: no future agent will know they exist after context summarization. Copying scripts without updating instructions is the #1 failure mode observed in practice.

#### 1. Install dependencies
```bash
pnpm add -D husky tsx   # or npm/yarn equivalents
npx husky init           # creates .husky/ dir and adds "prepare": "husky" to package.json
```

#### 2. Create the shared types module
Copy `scripts/lib/mistakes.ts` to your repo at the same relative path. It provides:
- `MistakeCategory` type (customize the categories for your domain)
- `MistakeEntry`, `MistakeLedger` types
- `VALID_CATEGORIES`, `LEDGER_PATH` constants
- `parseArgs()`, `readLedger()`, `writeLedger()`, `isMistakeEntry()` utilities

Adjust `LEDGER_PATH` if your scratch directory isn't `tmp/`. Ensure it's gitignored.

#### 3. Create the mistake logger
Copy `scripts/log-mistake.ts` to your repo. No changes needed if you kept the same directory structure.

Add to `package.json`:
```json
"mistakes:log": "npx tsx scripts/log-mistake.ts",
```

Usage:
```bash
# New entry (all metadata required first time):
pnpm run mistakes:log -- --key <id> --category <cat> --symptom "..." --prevention "..."

# Increment existing entry:
pnpm run mistakes:log -- --key <id>
```

#### 4. Create the priority reporter
Copy `scripts/mistakes-report.ts` to your repo. Adjust `CATEGORY_WEIGHT` map if you changed categories.

Add to `package.json`:
```json
"mistakes:report": "npx tsx scripts/mistakes-report.ts",
```

#### 5. Create the preflight script
Copy `scripts/preflight.ts` to your repo. **This is the file you must customize.** Edit:
- `Context` type — define contexts relevant to your project (e.g., `"src" | "tests" | "docs" | "ci"`)
- `inferContexts()` — map file paths to contexts (e.g., `file.startsWith("lib/") → "src"`)
- `buildChecks()` — map contexts to validation commands (e.g., `src → lint`, `tests → test`, `docs → spellcheck`)
- Remove dlm-js-specific contexts (e.g., `timings`, `bench`) and checks (e.g., `check:timings`) that don't apply to your project. Watch for directory naming differences — dlm-js uses `tests/` but your project may use `test/`.

Add to `package.json`:
```json
"preflight": "npx tsx scripts/preflight.ts",
```

#### 6. Wire the pre-commit hook
Write `.husky/pre-commit`:
```
pnpm run preflight
```
That's it. Husky makes this portable — any `pnpm install` activates the hook.

#### 7. Seed the ledger (optional but recommended)
Pre-seed patterns you expect to be common. Example:
```bash
pnpm run mistakes:log -- --key forgot-task-protocol --category missed-check \
  --symptom "Skipped orientation steps after context summarization" \
  --prevention "Follow task protocol in copilot-instructions.md at start of every task"
```
Pre-seeding at count=1 means the first real occurrence triggers promotion (count≥2).

#### 8. Add the protocol to your copilot-instructions.md — ⚠️ THIS IS THE MOST IMPORTANT STEP
Without this, the scripts you copied in steps 2–6 are invisible to future agents. **This step is what makes the protocol survive context summarization.**

Add these sections **near the top** of your `copilot-instructions.md` (adapt to your project):

```markdown
Task protocol (do this EVERY task, even after context summarization) 🚦
- **Start of task** — before any task work:
  1. `cat tmp/copilot-task-notes.md` — read notes left by your past self.
  2. `pnpm run mistakes:report` — check for mistake patterns needing attention.
  3. `pnpm run preflight -- --dry` — preview which checks apply.
- **Before every commit** — the husky pre-commit hook runs `pnpm run preflight` automatically.
- **End of task** — spend 30–60 seconds:
  1. Ask: "What slowed me down most?"
  2. If preventable: `pnpm run mistakes:log -- --key <id>` (with metadata if new).
  3. If count ≥ 2 and rule is short/testable: promote into copilot-instructions.md.
  4. Update `tmp/copilot-task-notes.md` with decisions, traps, in-progress work.
```

**AND ALSO:** Copy these two entire sections from this file to the new repo's `copilot-instructions.md`:
1. The "Autonomous instruction maintenance loop (self-tuning) 🔁" section (contains the rules for logging and promoting mistakes). **Place this immediately below the Task protocol.**
2. The "Cloning the self-tuning protocol to another repo 🔄" section (so it can spread further). **Place this at the very bottom of the file.**

#### 9. Create task notes
Create `tmp/copilot-task-notes.md` (gitignored). This is free-form — agents update it at end of task for their future self. Minimal template:
```markdown
# Task notes
Last updated: <date>
## Recent decisions (don't re-debate)
## Confusion traps
## In progress
(Nothing currently in progress.)
```

### Customization checklist
- [ ] **`copilot-instructions.md` updated with task protocol near the top** (this is the single most important item — without it, nothing else matters after summarization)
- [ ] **Maintenance loop copied** to the new repo's instructions so agents know how to log and promote mistakes
- [ ] **Cloning guide copied** to the bottom of the new repo's instructions so it can spread further
- [ ] Categories in `scripts/lib/mistakes.ts` match your project's failure modes
- [ ] `inferContexts()` in `scripts/preflight.ts` maps your file tree correctly
- [ ] `buildChecks()` in `scripts/preflight.ts` runs your project's linter/tests/validators
- [ ] `tmp/` (or your scratch dir) is in `.gitignore`
- [ ] Pre-commit hook tested: `git commit --allow-empty -m "test hook"` then `git reset HEAD~1`

### Files inventory (copy from dlm-js)
| File | Purpose | Must customize? |
|------|---------|----------------|
| `scripts/lib/mistakes.ts` | Shared types, ledger I/O, arg parser | Categories only |
| `scripts/log-mistake.ts` | CLI: append/increment ledger entries | No |
| `scripts/mistakes-report.ts` | CLI: priority-ranked report with promotion flags | Category weights only |
| `scripts/preflight.ts` | Context-aware check runner | **Yes** — contexts, file mapping, checks |
| `.husky/pre-commit` | Git hook: runs preflight on commit | No |
| `tmp/copilot-mistakes.json` | Gitignored mistake ledger | Created automatically |
| `tmp/copilot-task-notes.md` | Gitignored task notes | Create manually once |
