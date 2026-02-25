# ESLint plugin: `require-using` rule misses inline nested `np.*()` temporaries

**Status**: 🔴 Open  
**Package**: `@hamk-uas/jax-js-nonconsuming/eslint-plugin`  
**Filed**: 2026-02-25  
**Upstream commit**: `d08dd54`  
**Context**: dlm-js Kalman filter library — inline `np.reshape()` intermediates leaked GPU buffers for months without lint warnings

## Summary

The `require-using` rule only visits `VariableDeclaration` nodes. When an array-producing call (`np.reshape`, `np.eye`, etc.) is passed directly as an argument to another array-producing call, there is no variable declaration — so the rule never fires. The intermediate `np.Array` is never disposed.

## Reproduction

```ts
// ❌ LEAKS — but lint passes clean
G_scan = np.tile(np.reshape(G, [1, m, m]), [n, 1, 1]);
//                ^^^^^^^^^^^^^^^^^^^^^^^^^^
//                Returns a new np.Array (incRef on G's buffer).
//                np.tile creates a new buffer and releases its ref on the
//                tile input — but the reshape Array object itself is never
//                disposed, so it holds an extra incRef on G's slot forever.

// ✅ CORRECT — lint would also pass (using declaration present)
{
  using G_3d = np.reshape(G, [1, m, m]);
  G_scan = np.tile(G_3d, [n, 1, 1]);
}
```

## Root cause

In [the plugin source](packages/eslint-plugin/dist/index.js), `require-using` only has a `VariableDeclaration` visitor:

```js
create(context) {
  return { VariableDeclaration(node) {
    // ... checks each declarator for isArrayProducingCall(decl.init)
    // Only fires when: const x = np.reshape(...)
    // Never fires when: np.tile(np.reshape(...))
  }};
}
```

There is no `CallExpression` visitor that checks whether arguments to array-producing calls are themselves array-producing calls.

## Suggested fix

Add a `CallExpression` visitor (or extend the existing rule) that flags array-producing calls nested as arguments to other calls:

```js
CallExpression(node) {
  if (!isArrayProducingCall(node)) return;
  for (const arg of node.arguments) {
    if (isArrayProducingCall(arg)) {
      context.report({
        node: arg,
        messageId: "nestedArrayLeak",
        data: { callee: getCalleeName(arg) },
      });
    }
  }
}
```

This would catch the exact patterns like `np.tile(np.reshape(...))`, `np.multiply(np.array(...), np.eye(...))`, etc.

### Edge cases to consider

1. **Return statements**: `return np.reshape(x, shape)` — the result escapes, no leak. The rule should only flag nested calls, not top-level calls in return/yield.

2. **Method chains**: `np.reshape(G, [1,m,m]).mul(np.array(2))` — the reshape result is the receiver, not an argument. The `.mul()` return value is the concern, not the reshape.

3. **Traced contexts (jit/grad/scan bodies)**: Inside `jit()` or `grad()`, disposal is no-op (tracers intercept it). The nested pattern is still semantically a leak in eager mode, but harmless under trace. The rule could optionally suppress warnings inside known traced closures — but it's cleaner to always require explicit disposal since `using` is documented as safe in both modes.

## Impact on dlm-js

We had **3 leak sites** in `src/index.ts` that went undetected:
- `np.tile(np.reshape(G, [1, m, m]), ...)` — `dlmFit` line 1612
- `np.tile(np.reshape(W, [1, m, m]), ...)` — `dlmFit` line 1613
- `np.tile(np.reshape(F, [1, 1, m]), ...)` — `dlmSmo` line 161

Each leaked 1 GPU buffer slot per `dlmFit` call (3 slots total per call × 2 passes = 6 slots). Over 109 frame-collection calls, that's 654 leaked slots. The bytes are small (~4 bytes each for view refs), but the slot count growth is unbounded in long-running workloads.

Fixed in dlm-js by extracting each `np.reshape` into a `using` block.

## Affected files in upstream

- `packages/eslint-plugin/src/rules/require-using.ts` — needs `CallExpression` visitor
