# `ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING` on Node 22+ when installed as git dependency

## Filed to

[hamk-uas/jax-js-nonconsuming#4](https://github.com/hamk-uas/jax-js-nonconsuming/issues/4)

## Problem

When jax-js-nonconsuming is installed from git (the only install method), the package manager runs
the `prepare` script, which invokes `tsdown`. tsdown loads `tsdown.config.ts` — a TypeScript file
that now lives under `node_modules/`. On Node 22+, Node's built-in TypeScript type stripping
explicitly rejects `.ts` files inside `node_modules/`, throwing:

```
ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING
```

This means **every consumer using Node 22+ in CI or locally** will fail at `pnpm install` /
`npm install` time.

The ESLint plugin README already documents this error for the ESLint config loading case, but the
build-time `tsdown.config.ts` case affects a much wider audience — anyone who installs the package
at all.

## Reproduction

```bash
# With Node 22+
node --version  # v22.x.x or v23.x.x
npm install github:hamk-uas/jax-js-nonconsuming
# → ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING in tsdown.config.ts
```

## Current workaround

Downstream consumers must pin Node 20 in CI:

```yaml
# .github/workflows/deploy.yaml
- uses: actions/setup-node@v4
  with:
    node-version: 20  # Cannot use 22+ due to tsdown.config.ts
```

This is not ideal since Node 20 reaches EOL in April 2026 and Node 22 is the current LTS.

## Suggested fixes (any one would work)

1. **Rename `tsdown.config.ts` → `tsdown.config.mjs`** (or `.js`). tsdown supports JS configs.
   This is the minimal change.

2. **Ship pre-built `dist/` artifacts** so the `prepare` script is either removed or becomes a
   no-op when `dist/` already exists. This avoids the build step entirely for consumers.

3. **Publish to npm** as a proper scoped package, which would eliminate the git-dependency `prepare`
   step altogether.

## Impact

- Affects all downstream consumers (not just dlm-js).
- Forces downgrade to Node 20 LTS, which reaches EOL April 2026.
- The same `.ts`-under-`node_modules` pattern will also break `typedoc.ts` and any other root-level
  `.ts` scripts if they are referenced during install.

## References

- [Node.js docs: ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING](https://nodejs.org/api/errors.html#err_unsupported_node_modules_type_stripping)
- dlm-js fix: `52765d2` (pinned Node 20 in CI)
- ESLint plugin README already documents this for eslint.config case
