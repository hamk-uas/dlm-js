# Git dependency: prepare script may silently fail to generate .d.ts

## Filed to

[hamk-uas/jax-js-nonconsuming#5](https://github.com/hamk-uas/jax-js-nonconsuming/issues/5)

## Status

**Open** upstream. [PR #6](https://github.com/hamk-uas/jax-js-nonconsuming/pull/6) added a hard error guard, but this broke downstream CI installs (the guard triggered on the exact silent failure it was meant to detect). [PR #7](https://github.com/hamk-uas/jax-js-nonconsuming/pull/7) downgraded the guard to a warning so install succeeds. The underlying DTS generation flakiness remains unfixed.

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

The core issue is that tsdown DTS generation fails silently. The `prepare` script now warns
(PR #7) but does not fix the root cause. A proper fix would ensure tsdown reliably generates
`.d.ts` files, or ship pre-built `dist/` in the repo.
