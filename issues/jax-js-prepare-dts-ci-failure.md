# jax-js-nonconsuming: `prepare` script fails on GitHub Actions (DTS generation produces 0 files)

**Status:** 🔴 Open

## Summary

The `prepare` script in `@hamk-uas/jax-js-nonconsuming` fails on GitHub Actions runners because `tsdown` DTS generation silently produces 0 `.d.ts` files, causing the `test -f dist/index.d.ts` guard to fail with exit code 1. This breaks `pnpm install --frozen-lockfile` for any downstream consumer CI that fetches the package from git.

## Affected version

- Tag: `v0.7.9`
- Commit: `0972c61ae07010a8d30bfc2397fcbe8775ade439`

## Root cause hypothesis

The `prepare` script runs:

```bash
tsdown && test -f dist/index.d.ts && (husky || true)
```

`tsdown` builds 5 workspace packages (jax, eslint-plugin, loaders, onnx, optax). The ESM/CJS bundles are generated successfully, but the DTS generation pass for each package reports `0 files, total: 0.00 kB`. Since no `.d.ts` files are emitted, `test -f dist/index.d.ts` fails.

The DTS pass works locally but not on GitHub Actions `ubuntu-latest` runners. Possible causes:
1. Missing TypeScript version or tsconfig resolution difference in CI environment
2. `tsdown` DTS plugin silently swallowing errors (no error output, just 0 files)
3. Race condition or environment difference in the monorepo workspace build

## Reproduction

The failure is observable in any GitHub Actions workflow that installs `@hamk-uas/jax-js-nonconsuming` from git:

```yaml
# .github/workflows/deploy-pages.yaml (dlm-js)
- run: pnpm install --frozen-lockfile
```

CI log excerpt:

```
. prepare: ℹ [loaders] [CJS] 0 files, total: 0.00 kB
. prepare: ✔ [loaders] Build complete in 647ms
. prepare: ℹ [optax] [CJS] 0 files, total: 0.00 kB
. prepare: ✔ [optax] Build complete in 662ms
. prepare: ℹ [eslint-plugin] [CJS] 0 files, total: 0.00 kB
. prepare: ✔ [eslint-plugin] Build complete in 668ms
. prepare: ℹ [onnx] [CJS] 0 files, total: 0.00 kB
. prepare: ✔ [onnx] Build complete in 670ms
. prepare: ℹ [jax] [CJS] 0 files, total: 0.00 kB
. prepare: ✔ [jax] Build complete in 767ms
. prepare: Failed
 ELIFECYCLE  Command failed with exit code 1.
 ERR_PNPM_PREPARE_PACKAGE  Failed to prepare git-hosted package
```

Note: the `[CJS] 0 files` lines above are the DTS output round (third pass after ESM + CJS), not the actual CJS round. The ESM and CJS bundles build correctly. Only `dist/index.d.ts` is missing.

## Impact on dlm-js

- **GitHub Pages deployment is broken.** The `deploy-pages.yaml` workflow cannot run `pnpm install --frozen-lockfile` to install dependencies and generate TypeDoc API docs.
- Any future CI workflow (tests, linting, etc.) would also fail.
- Local development is unaffected (tsdown DTS generation works locally).

## Suggested fix

Options (in order of preference):

1. **Investigate why DTS generation produces 0 files on CI.** The tsdown build logs show all 5 packages complete successfully, but the DTS pass emits nothing. Adding `--verbose` or `--debug` to tsdown might reveal the root cause.

2. **Make the prepare script more resilient.** Instead of hard-failing on missing DTS, emit a warning and continue. Consumers that need types can run `tsdown --dts` separately:
   ```bash
   tsdown && (test -f dist/index.d.ts || echo "Warning: DTS not generated") && (husky || true)
   ```

3. **Ship pre-built DTS files.** Include `dist/*.d.ts` in the git tree so the `prepare` script only needs to build JS bundles (which work reliably).

## Workaround locations

| File | Description |
|------|-------------|
| `dlm-js/.github/workflows/deploy-pages.yaml` | Currently broken — needs upstream fix to resume CI deploys |
