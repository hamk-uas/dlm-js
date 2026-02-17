# einsum multi-operand contraction: option for fused reduction

> **Update:** Kahan compensated summation for Float64 dot reductions shipped in jax-js-nonconsuming v0.2.1. As predicted in "Don't bother" (option 3 below), the remaining benefit of einsum fusion is small once individual dots are compensated. The dominant error source is catastrophic cancellation in `C - C·N·C`, not dot product accumulation. This issue is kept for reference but is low priority.

## Summary

`einsum` with 3+ operands (e.g., `'ij,jk,kl->il'`) decomposes the contraction into sequential pairwise `dot()` calls via an optimal contraction path. Each intermediate materialization introduces its own rounding. For precision-sensitive workloads, a fused kernel that contracts all operands in a single pass over the reduction index would reduce intermediate rounding.

## Context

In our DLM (Kalman filter) port, the most common 3-operand einsum patterns are:

```typescript
np.einsum('ij,jk,lk->il', G, C, L)    // G·C·L'  (forward covariance update)
np.einsum('ij,jk,kl->il', C, N, C)    // C·N·C   (backward covariance update)
np.einsum('ji,jk,kl->il', L, N, L)    // L'·N·L  (information matrix update)
```

Currently each decomposes into two `dot` calls:
1. `tmp = dot(A, B)` — materializes an m×m intermediate
2. `result = dot(tmp, C)` (or `dot(tmp, C')`)

For `C·N·C` specifically, the intermediate `tmp = C·N` accumulates O(m·ε) rounding in each element. Then `dot(tmp, C)` introduces another O(m·ε) rounding. The final result has ~O(m²·ε) accumulated error before the catastrophic subtraction in `C - C·N·C` amplifies it further.

## What a fused kernel could do

For the pattern `A·B·C` where all are m×m, a fused triple loop:

```
result[i][l] = Σ_j Σ_k A[i][j] * B[j][k] * C[k][l]
```

could use a single (compensated) accumulator over the j×k product space. This avoids materializing the intermediate and reduces accumulated rounding from O(m²·ε) to O(m²·ε) with a smaller constant (single accumulation over m² terms vs. two sequential accumulations over m terms each feeding into another m-term accumulation).

With Kahan summation on the single accumulator, it drops to O(ε²) regardless of m.

## Practical considerations

- **Fused triple-product kernels are unusual** — most BLAS libraries don't have them. The standard approach is `DGEMM` + `DGEMM` (two calls). But jax-js generates its own kernels, so there's no BLAS constraint.

- **JIT already fuses element-wise ops** but not reduction ops like matmul. Fusing reductions is fundamentally harder because of the shared accumulator.

- **Cost-benefit**: For small m (1–13, typical in time series models), the compute is negligible either way. For large m (neural network layers, m = 512+), pairwise dot is actually better because it can use tiled/blocked algorithms. So this optimization, if implemented, would likely only help small-to-medium m.

- **This is lower priority than compensated summation in `dot`** (see companion issue). If dot's inner reduction becomes compensated, the benefit of fusing across operands becomes much smaller.

## Possible approaches

1. **Einsum path hint**: Allow users to request fused contraction for specific patterns, e.g., `np.einsum('ij,jk,kl->il', A, B, C, { fused: true })`. The einsum implementation could check if a fused kernel exists for the pattern.

2. **Automatic for small dimensions**: If all contracted dimensions are ≤ some threshold (e.g., 64), use a fused kernel. For large dimensions, stick with pairwise for cache efficiency.

3. **Don't bother**: If compensated summation in `dot` lands (companion issue), the remaining benefit of fusion is small. This issue could be closed as won't-fix.

We're filing this mainly for completeness — the compensated summation issue is the higher-impact request.

## Measured impact

With the current naive-summation pairwise decomposition (Float64):

| Pattern | State dim m | Max relative error vs MATLAB |
|---|---|---|
| `G·C·L'` (forward) | 13 | ~3e-5 |
| `C·N·C` (backward, before subtraction) | 6 | ~8e-4 (after C - C·N·C cancellation) |
| `L'·N·L` (backward N update) | 13 | contributes to above via N accumulation |

These are acceptable for most applications. The issue is primarily relevant if jax-js wants to compete with LAPACK-backed libraries on numerical accuracy for Float64 scientific computing.

## Synthetic ground-truth perspective

Testing the DLM against known true hidden states (rather than Octave reference) shows that the ~1e-4 relative errors reported above are **implementation disagreements** between jax-js and LAPACK rounding pathways. They do not affect the DLM's ability to recover hidden states from noisy observations.

The smoothed state RMSE is ~1–3 (dominated by statistical estimation uncertainty), while the entire v0.2.0 vs v0.2.1 rounding difference is ~2e-10 — ten orders of magnitude smaller. Even the full naive-summation rounding (without Kahan) is negligible at the precision that matters for DLM end users.

This further supports option 3 ("Don't bother") for the einsum fusion request. The benefit would only be visible in implementation-vs-implementation comparisons, not in actual model accuracy.
