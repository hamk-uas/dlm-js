# Git dependency: prepare script may silently fail to generate .d.ts

## Filed to

[hamk-uas/jax-js-nonconsuming#5](https://github.com/hamk-uas/jax-js-nonconsuming/issues/5)

## Problem

When `@jax-js-nonconsuming/jax` is installed as a git dependency, the `prepare` script runs
`tsdown` to build `dist/`. In some CI environments (observed in GitHub Actions), the DTS generation
step can **silently fail**, producing `dist/index.js` but **no `dist/index.d.ts`**.

This causes downstream consumers to get TS7016 errors from TypeDoc, `tsc`, or IDE tooling.

Separate from #4 (Node 22 type stripping). This issue can occur on any Node version.

## Our workarounds (all working)

- `"skipErrorChecking": true` in `typedoc.json` (TypeDoc still generates docs)
- `"skipLibCheck": true` in `tsconfig.json` (already standard)
- `"moduleResolution": "Bundler"` in `tsconfig.json` (modern ESM resolution)

## Suggested upstream fix

The core issue is that tsdown DTS generation fails silently. The `prepare` script should verify
that `dist/index.d.ts` was actually produced and fail loudly if not.
