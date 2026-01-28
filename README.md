# dlm-js — a minimal jax-js port of dynamic linear model

A minimal [jax-js](https://jax-js.com/) port of [dynamic linear model](https://mjlaine.github.io/dlm/dlmtut.html) (MATLAB).

## Features
✅ implemented, ❌ not implemented, — will not be implemented

| Feature | dlm&#8209;js | dlm (MATLAB) | Description |
| --- | --- | --- | --- |
| Feature 1 | ✅ | ✅ | (Describe here the feature implemented or implemetation details) |
| Plotting | — | ✅ | This is a computation-only library. Plotting is not planned to be implemented. |
| float32 computation | ✅ | ❌ | (see below) |
| float64 computation | ❌ | ✅ | The solver from jax-js currently only supports float32 data. At the moment we favor using the existing jax-js solver for ease of initial development. We have tested that "manual" float64 solves can be implemented in jax-js and result in near-exact matches to results from Octave. We may switch to or also provide float64 for testing or accuracy reasons or when jax-js implements float64 solves. dlm works in float64 in Octave. |

## TODO

* Test the built library
* Document the library
* Choose the important dlm functions and output variables for implementation
* Rework initial AI-generated DLM port

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
|   |   ├── niledemo-out.json    # Niledemot test output from Node.js
|   │   └── niledemo-out-m.json  # Niledemo reference output from Octave
|   ├── niledemo-in.json     # Niledemo test input
|   ├── niledemo-keys.json   # Niledemo list of tested output keys, for partial implementations
|   └── niledemo.ts          # Niledemo test with 1 % error tolerance
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

This project uses [pnpm](https://pnpm.io/) for fast, disk-efficient dependency management. Install pnpm and Node.js:

Install Node.js ([instructions](https://nodejs.org/en/download/)).

### Install pnpm globally

```shell
npm install -g pnpm
```

### Install dependencies using pnpm

```shell
pnpm install
```

### Install Octave

Octave version 10.3.0 is known to work. Other versions will likely work too.

Install Octave and add the folder containing `octave_cli` or `octave_cli.exe` to system path.

### Building and bundling

This project is written in TypeScript. You need to build (compile) it before use:

```shell
npm run build
```
- This does two things:
  - Compiles TypeScript (`src/index.ts`) to ESM and CommonJS JavaScript (`dist/dlm-js.es.js`, `dist/dlm-js.cjs.js`) and type definitions (`dist/index.d.ts`)
  - Bundles the code with Vite for use as a library (outputs ESM and CommonJS formats in `dist/`)

**What does this mean?**

- TypeScript lets you write code with types, but Node.js and browsers only run JavaScript. The build step converts your code to JavaScript.
- Vite bundles your code so it can be used easily in other projects, in Node.js or browsers, and optimizes it for distribution.

### Generate reference output using Octave

```shell
npm run test:octave
```

This will generate `tests/out/niledemo-out-m.json`. It will also generate `tests/niledemo-in.json`, unless it already exists, and will use that as its input.

### Run niledemo test

After building, you can run the niledemo test with Node.js:

```shell
pnpm vitest run
```
This will run the niledemo test using the built library and write the output to `tests/out/niledemo-out.json`.

### Authors
* Marko Laine -- Original dlm and mcmcstat sources
* Olli Niemitalo (Olli.Niemitalo@hamk.fi) -- Framework and initial (human-assisted AI port of DLM)
* ########### -- Refined DLM port

### Copyright
* 2013-2017 Marko Laine -- dlm and mcmcstat
* 2026 HAMK Häme University of Applied Sciences
* 2026 ######
  
### License
MIT license
