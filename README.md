# dlm-js — a minimal jax-js port of dynamic linear model

A minimal [jax-js](https://jax-js.com/) port of [dynamic linear model](https://mjlaine.github.io/dlm/dlmtut.html) (MATLAB). 

<img width="1277" height="453" alt="image" src="https://github.com/user-attachments/assets/264af73a-d797-45a9-93a6-1fc5cc0503a2" />

*Niledemo main output from dlm-js (which uses jax-js) and from the MATLAB dlm implementation (using Octave). The dlm-js computation lasts about 2 seconds.*

## Features
✅ implemented, ❌ not implemented, — will not be implemented

| Feature | dlm&#8209;js | dlm | Description |
| --- | --- | --- | --- |
| Plotting | — | ✅ | dlm-js is a computation-only library. Plotting is not planned to be implemented. |
| float32 computation | ✅ | ❌ | With float32, niledemo results differ by a factor less than 1 %. dlm-js dtype is configurable whereas dlm works in float64 in Octave. GPU acceleration can be used when float32 is selected, but it is very slow due to the serial algorithm. Using the wasm backend is recommended instead. |
| float64 computation | ✅ | ✅ | With float64, niledemo results differ by a factor less than 1e-10. |

## TODO

* Test the built library (in `dist/`)
* Choose the important dlm functions and non-default input variables and output variables for implementation
* Human review the AI-generated DLM port. It does pass niledemo test, but has not been properly reviewed.
* Document the library

## Project structure

```
├── dist/                # Compiled and bundled output (after build)
├── src/                 # Library TypeScript sources
|   └── index.ts             # Main source file
├── tests/               # Test suite
│   ├── octave/              # Octave reference output generators
│   │   ├── dlm/                 # Minimal MATLAB dlm implementation
│   │   └── niledemo.m           # Niledemo Octave script to generate reference output (and input)
│   ├── out/                 # Test outputs
|   │   └── niledemo-out.json    # Niledemot test output from Node.js
|   ├── niledemo-in.json     # Niledemo test input
|   ├── niledemo-keys.json   # Niledemo list of output keys to test, for partial implementations
|   ├── niledemo-out-m.json  # Niledemo reference output from Octave
|   ├── niledemo.test.ts     # Niledemo test
|   └── utils.ts             # Test utility functions
├── .gitignore           # Ignore file for git
├── .npmignore           # Ignore file for npm
├── LICENSE              # License (does not apply to tests/octave/dlm/)
├── package.json         # Node.js package information
├── pnpm-lock.yaml       # Locked dependencies for reproducible installs (update with "pnpm update")
├── README.md            # This readme          
├── tsconfig.json        # Configuration file of the TypeScript project
├── vite.config.ts       # Configuration file of the Vite project
```

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
npm run test:octave
```

This will generate `tests/out/niledemo-out-m.json`. It will also generate `tests/niledemo-in.json`, unless it already exists, and will use that as its input.

### Run niledemo test

You can run the niledemo test directly (no build step needed) with:

```shell
pnpm vitest run
```

or

```shell
pnpm run test:node
```

This runs the test using the source code in `src/` (not the built output). Vitest compiles your TypeScript on the fly.

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
