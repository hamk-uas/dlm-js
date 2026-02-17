# dlm-js — a minimal jax-js port of dynamic linear model

A minimal [jax-js](https://jax-js.com/) port of [dynamic linear model](https://mjlaine.github.io/dlm/dlmtut.html) (MATLAB). 

<img width="1277" height="453" alt="image" src="https://github.com/user-attachments/assets/264af73a-d797-45a9-93a6-1fc5cc0503a2" />

*Niledemo main output from dlm-js (which uses jax-js) and from the MATLAB dlm implementation (using Octave). The JIT-compiled dlm-js computation lasts about 60 ms (or 24 ms on successive runs with cached compilation) using `lax.scan` from [a non-consuming ownership jax-js fork](https://github.com/hamk-uas/jax-js-nonconsuming).*

## Features
✅ implemented, ❌ not implemented, — will not be implemented

| Feature | dlm&#8209;js | dlm | Description |
| --- | --- | --- | --- |
| Plotting | — | ✅ | dlm-js is a computation-only library. Plotting is not planned to be implemented. |
| State space generation | ✅ | ✅ | `dlmgensys` generates G, F matrices for polynomial trend, full/trigonometric seasonal, and AR(p) components. |
| Arbitrary state dimension | ✅ | ✅ | Kalman filter and RTS smoother support state dimension m ≥ 1, matching MATLAB's `dlmfit`/`dlmsmo`. |
| float32 computation | ✅ | ❌ | dlm-js dtype is configurable whereas dlm works in float64 in Octave. Float32 is numerically stable for state dimension m ≤ 2; higher dimensions may diverge due to catastrophic cancellation in covariance updates. GPU acceleration can be used when float32 is selected, but the serial Kalman algorithm is slow on GPU. Using the wasm backend is recommended instead. |
| float64 computation | ✅ | ✅ | With float64, results match the MATLAB reference within ~2e-3 relative tolerance (or < 1e-6 absolute tolerance for small covariance elements). See [numerical precision notes](#numerical-precision). |
| Device × dtype test matrix | ✅ | — | All tests run on every available (device, dtype) combination: cpu/f64, cpu/f32, wasm/f64, wasm/f32, webgpu/f32. |
| Synthetic ground-truth tests | ✅ | — | Tests against known true states from a seeded generating process. Verifies noise reduction, calibrated uncertainty, positive covariance — independent of any reference implementation. |

## Numerical precision

Since jax-js v0.2.1, Float64 dot product reductions use Kahan compensated summation, reducing per-dot rounding from O(m·ε) to O(ε²). This improved the seasonal model (m=13) from ~3e-5 to ~1.8e-5 worst-case relative error.

However, the dominant error source is **not** summation accuracy — it is catastrophic cancellation in the RTS backward smoother step `C_smooth = C - C·N·C`. When the smoothing correction nearly equals the prior covariance, the subtraction amplifies any rounding in the operands. Kahan summation cannot fix this because it only improves the individual dot products, not the outer subtraction. See detailed comments in `src/index.ts`.

Precision issues have been filed upstream: [issues/](issues/).

## TODO

* Test the built library (in `dist/`)
* Implement remaining dlm features (covariates, spline mode, missing data handling)
* Human review the AI-generated DLM port
* Document the library API

## Project structure

```
├── dist/                # Compiled and bundled output (after build)
├── issues/              # Drafted GitHub issues for upstream jax-js
├── src/                 # Library TypeScript sources
│   ├── index.ts             # Main source: dlmSmo (Kalman+RTS), dlmFit (two-pass fitting)
│   ├── dlmgensys.ts         # State space generator: polynomial, seasonal, AR components
│   └── types.ts             # TypeScript type definitions and helpers
├── tests/               # Test suite
│   ├── octave/              # Octave reference output generators
│   │   ├── dlm/                 # Minimal MATLAB dlm implementation (Marko Laine)
│   │   ├── niledemo.m           # Niledemo — pre-existing MATLAB DLM demo script
│   │   └── gensys_tests.m      # Additional model tests (synthetic data, generated for this project)
│   ├── out/                 # Test outputs (gitignored)
│   ├── test-matrix.ts       # Shared device × dtype test configurations and tolerances
│   ├── niledemo-in.json     # Niledemo input data
│   ├── niledemo-keys.json   # Output keys to compare (for partial implementations)
│   ├── niledemo-out-m.json  # Niledemo reference output from Octave
│   ├── niledemo.test.ts     # Niledemo integration test
│   ├── gensys.test.ts       # dlmgensys unit tests + multi-model integration tests
│   ├── synthetic.test.ts    # Synthetic ground-truth tests (known true states, statistical assertions)
│   ├── {order0,order2,seasonal,trig,level}-{in,out-m}.json  # Test data (see below)
│   └── utils.ts             # Test utility functions
├── LICENSE              # License (does not apply to tests/octave/dlm/)
├── package.json         # Node.js package information
├── README.md            # This readme          
├── tsconfig.json        # Configuration file of the TypeScript project
└── vite.config.ts       # Configuration file of the Vite project
```

### Test data origins

| Test | Source | Description |
| --- | --- | --- |
| `niledemo` | Pre-existing MATLAB DLM demo (`niledemo.m` by Marko Laine) | Annual Nile river flow, order=1 (local linear trend), m=2. Input/reference generated by Octave. |
| `order0` | Generated for this project (`gensys_tests.m`) | Nile data with order=0 (local level), m=1. Tests the simplest model (scalar state). |
| `level` | Generated for this project (`gensys_tests.m`) | First 50 Nile values with order=0, m=1. Compact edge-case test. |
| `order2` | Generated for this project (`gensys_tests.m`) | Synthetic quadratic signal + deterministic "noise" (sin/cos), order=2, m=3. Tests higher polynomial trend. |
| `seasonal` | Generated for this project (`gensys_tests.m`) | Synthetic monthly data (10 years) with trend + 3 harmonics, fullseas=1, ns=12, m=13. Tests full seasonal decomposition. |
| `trig` | Generated for this project (`gensys_tests.m`) | Same synthetic monthly data, trig=2, ns=12, m=6. Tests trigonometric seasonal with fewer states. |

All generated test data uses deterministic signals (no random noise) so reference outputs are exactly reproducible across platforms.

### Synthetic ground-truth tests

In addition to the Octave reference tests above, `synthetic.test.ts` generates state-space data from a **known generating process** with known true hidden states (using a seeded PRNG with Box-Muller transform for reproducible Gaussian noise). The DLM smoother is then tested against mathematical ground truth rather than another implementation's rounding:

- **Finite outputs**: No NaN/Inf in any result field
- **Positive covariance**: Smoothed covariance diagonals `C[k][k][t] > 0` for all states and timesteps
- **Noise reduction**: Smoother RMSE < observation RMSE (the smoother actually reduces noise)
- **Calibrated uncertainty**: True states fall within the 95% posterior credible intervals at roughly the nominal rate

Models tested: local level (m=1) at moderate/high/low SNR, local linear trend (m=2), trigonometric seasonal (m=6), and full seasonal (m=13). All run across the full device × dtype matrix. Float32 is skipped for m > 2 (see float32 row in the features table).

## Development

### Install Node.js

[Install Node.js](https://nodejs.org/en/download/) to be able to run JavaScript locally. The installation includes the npm package manager.

### Install pnpm globally

This project uses [pnpm](https://pnpm.io/) for fast, disk-efficient dependency management. Install it using npm:

```shell
npm install -g pnpm
```

### Install dependencies using pnpm

Install dlm-js dependencies automatically using pnpm:

```shell
pnpm install
```

### Install Octave

Octave version 10.3.0 is known to work. Other versions will likely work too.

Install Octave and add the folder containing `octave-cli` or `octave-cli.exe` to system path.

### Building and bundling

This project is written in TypeScript. You need to build (compile) it before use:

```shell
npm run build
```
This does two things:
  - **Compiles TypeScript (`src/index.ts`) to ESM and CommonJS JavaScript (`dist/dlm-js.es.js`, `dist/dlm-js.cjs.js`) and type definitions (`dist/index.d.ts`).** TypeScript lets you write code with types, but Node.js and browsers only run JavaScript. The build step converts your code to JavaScript.
  - **Bundles the code with Vite for use as a library (outputs ESM and CommonJS formats in `dist/`).** Vite bundles your code so it can be used easily in other projects, in Node.js or browsers, and optimizes it for distribution.

### Generate reference output using Octave

```shell
pnpm run test:octave
```

This generates Octave reference outputs:
- `tests/niledemo-out-m.json` (from `niledemo.m` — pre-existing MATLAB DLM demo)
- `tests/{order0,order2,seasonal,trig,level}-out-m.json` (from `gensys_tests.m` — generated for this project)

It will also generate test input files unless they already exist.

### Run tests

You can run all tests directly (no build step needed) with:

```shell
pnpm vitest run
```

or

```shell
pnpm run test:node
```

This runs `niledemo.test.ts`, `gensys.test.ts`, and `synthetic.test.ts` against all available device × dtype combinations. Vitest compiles TypeScript on the fly.

To run the full CI-local check (lint + Octave reference generation + tests):

```shell
pnpm run test
```

### Authors
* Marko Laine -- Original dlm and mcmcstat sources in `tests/octave/dlm/` and `tests/octave/niledemo.m`
* Olli Niemitalo (Olli.Niemitalo@hamk.fi) -- Framework and initial human-assisted AI port of DLM
* ########### -- Refined DLM port

### Copyright
* 2013-2017 Marko Laine -- Original dlm and mcmcstat sources in `tests/octave/dlm/` and `tests/octave/niledemo.m`
* 2026 HAMK Häme University of Applied Sciences
* 2026 ######
  
### License
MIT license
