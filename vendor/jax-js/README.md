# Vendored jax-js

This is a vendored build of [@jax-js/jax](https://github.com/olli4/jax-js).

**Source:** https://github.com/olli4/jax-js/tree/feat/scan

## Updating

To update to a newer build:

```bash
cd ../jax-js
git checkout feat/scan
pnpm install && pnpm build

cd ../dlm-js
cp -r ../jax-js/dist vendor/jax-js/
cp ../jax-js/package.json vendor/jax-js/
pnpm install
```
