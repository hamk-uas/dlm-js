# dlm-js — a minimal jax-js port of dynamic linear model

A minimal [jax-js](https://jax-js.com/) port of [dynamic linear model](https://mjlaine.github.io/dlm/dlmtut.html) for MATLAB.

## Features
✅ implemented, ❌ not implemented

| Feature | minidlm | dlm | description |
| --- | --- | --- | --- |
| Feature 1 | ✅ | ✅ | (Describe here the feature implemented or implemetation details) |
| Plotting | ❌ | ✅ | This is a computation-only library. Plotting is not planned to be implemented. |
| float32 computation | ✅ | ❌ | The solver from jax-js currently only supports float32 data. At the moment we favor using them for ease of initial development. dlm works in float64 in Octave. |
| float64 computation | ❌ | ✅ | We have tested that "manual" float64 solves can be implemented in jax-js and result in near-exact matches to results from Octave. We may switch to float64 for testing or accuracy reasons or when jax-js implements float64 solves. |

## Project structure

| Path | Description |
| --- | --- |
| `/` | Project root |
| `/dlm-m/` | Original dlm and mcmcstat MATLAB sources including only essential dependencies for features ported to minidlm |
| `/test/` | Tests for verifying the correctness and accuracy of results |
| `/test/niledemo/` | Niledemo test |
| `/test/niledemo/niledemo.m` | Niledemo test: Octave script to generate and read (for fairness) the input JSON, and to compute Octave output |
| `/test/niledemo/niledemo.js` | Niledemo test: JavaScript script to read the input JSON and to compute JS output |
| `/test/niledemo/niledemo-in.json` | Niledemo test: Input JSON, containing input data to the computation |
| `/test/niledemo/niledemo-out-js.json` | Niledemo test: JS output JSON |
| `/test/niledemo/niledemo-out-m.json` | Niledemo test: Octave output JSON |
| `/dlm-js.js` | dlm-js JavaScript module |

## Usage

## Prerequisites
### npm and Node.js
Install npm and Node.js, [see instructions](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm).

## Basic use

...

## Testing and development

### Prerequisites
#### Octave
Octave version 7.2.0 will work. Later versions may work.

Install Octave and add the folder containing `octave_cli` or `octave_cli.exe` to system path.

### Generate reference outputs using Octave

```shell
cd tests/niledemo
octave-cli niledemo.m
```

### Generate jax-js outputs

```shell
node run-niledemo.js
```

### Authors
* Marko Laine -- Original DLM and MCMCStat sources
* Olli Niemitalo (Olli.Niemitalo@hamk.fi) -- Framework and initial AI-assisted DLM port
* ########### -- Refined DLM port

### Copyright
* 2013-2017 Marko Laine -- DLM and MCMCStat (Folder `dlm-m`)
* 2026 HAMK Häme University of Applied Sciences -- Framework
* 2026 ###### -- Refined DLM port
  
### License
MIT license
