# Minimal jax-js port of dynamic linear model

A minimal [jax-js](https://jax-js.com/) port of [dynamic linear model](https://mjlaine.github.io/dlm/dlmtut.html).

THESE INSTRUCTIONS ARE WORK IN PROGRESS

# Prerequisites
## Octave
Octave version 7.2.0 will work. Later versions may work.

Install Octave and add the folder containing `octave_cli` or `octave_cli.exe` to system path.
## npm and Node.js
Install npm and Node.js, [see instructions](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm).

# Testing
## Generate reference outputs using Octave

```shell
cd tests/niledemo
octave-cli niledemo.m
```

## Generate jax-js outputs

```shell
node run-niledemo.js
```

## Authors

## Copyright
DLM and MCMCStat (from which some files have been included in `dlm-m`) copyright 2017-2021 Marko Laine
