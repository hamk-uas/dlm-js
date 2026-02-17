# Compensated or pairwise summation in dot product / reduction kernels

## Summary

The inner-product reduction in `dot` (and by extension `matmul`, `einsum`) uses naive summation (`acc += a[i] * b[i]`). This accumulates O(n·ε) rounding error where n is the reduction dimension and ε is machine epsilon. For Float64 Kalman filter workloads with state dimension m = 6–13, this produces relative errors up to ~1e-4 against a MATLAB/Octave reference — significantly worse than the ~1e-14 achievable with compensated summation.

## Context — how we found this

We're porting a MATLAB Dynamic Linear Model (DLM) library to jax-js. The DLM runs a Kalman filter (forward) + RTS smoother (backward), which chains many matrix multiplications per timestep over ~100 timesteps. We compared Float64 outputs against Octave's reference (which uses LAPACK/BLAS with 80-bit intermediate accumulators on x86).

Worst measured errors (Float64, `wasm` and `cpu` backends identical):

| State dim (m) | Max relative error | Where |
|---|---|---|
| 1 | 2.2e-8 | `s2` (scalar) |
| 2 | 4.7e-7 | `lik` (log-likelihood sum) |
| 3 | 1.6e-7 | `resid0` |
| 6 | 8.4e-4 | `Cf[0][4]` (covariance) |
| 13 | 2.9e-5 | `Cf[0][2]` (covariance) |

Float32 becomes numerically unstable for m > 2 — covariance matrices go negative, producing NaN.

The dominant error source is the backward smoother step `C_smooth = C - C·N·C`, where catastrophic cancellation amplifies any rounding in the `C·N·C` triple product. But the rounding itself originates in the dot product reductions inside `einsum`/`matmul`.

## What we're requesting

More accurate summation in the reduction kernel used by `dot`. Two well-known approaches:

### Option A: Kahan compensated summation

```
let sum = 0, c = 0;
for (let i = 0; i < n; i++) {
  const y = a[i] * b[i] - c;
  const t = sum + y;
  c = (t - sum) - y;
  sum = t;
}
```

Error: O(ε²) regardless of n. Cost: ~2× more additions per element (all register ops, no memory bandwidth impact). This is what CPython's `math.fsum` uses.

### Option B: Pairwise (recursive) summation

Split the reduction in half recursively, sum each half, then add. Error: O(log n · ε). This is what NumPy uses for `np.sum` and `np.dot`. It's cache-friendlier than Kahan on large arrays.

## Backend considerations

We understand the backends have very different constraints:

- **CPU (interpreted JS)**: Either approach is straightforward. Kahan adds ~4 register ops per iteration. Pairwise needs a recursive call or explicit stack but is more vectorization-friendly.

- **WASM (JIT-compiled)**: Same as CPU — both approaches map naturally to WASM scalar ops. The reduction loop in the JIT-compiled WASM is currently a simple accumulate; changing it to Kahan or pairwise is localized.

- **WebGPU**: GPU reductions are typically done via workgroup shared memory with tree reduction. Pairwise summation is a natural fit for tree reductions. Kahan is harder to parallelize but can be done per-thread before the tree reduce. However, for ML workloads (neural networks), the current naive summation is fine — Float32 accumulation is the norm and precision isn't the bottleneck. So GPU might want to keep naive summation as the default for throughput.

- **WebGL**: Presumably same as WebGPU.

## Possible approaches

1. **Per-backend choice**: Use compensated/pairwise on CPU/WASM, naive on GPU. The backends already have separate reduction implementations, so this might be straightforward.

2. **Op-level option**: An option on `dot`/`einsum` like `precision='high'` (cf. JAX's `precision` parameter on `jnp.dot`). This would let performance-sensitive code opt out.

3. **Global setting**: A flag like `jax.config.update('jax_default_matmul_precision', 'high')` — JAX has exactly this.

4. **Always-on for Float64**: Since Float64 users are explicitly opting into precision over speed, compensated summation could be the unconditional default for Float64 reductions. Float32 stays naive.

We don't have a strong opinion on which approach — any improvement over naive summation would substantially help our use case.

## Reproduction

```typescript
import { numpy as np, DType, init, defaultDevice } from '@jax-js-nonconsuming/jax';

await init();
defaultDevice('cpu');

// Simulate accumulated rounding: sum 1000 terms of 1e-8
// Exact answer: 1e-5
// Naive f64 summation will show ~1e-19 error (small n),
// but chain 100 matmuls and it compounds.

// More realistic: compare einsum triple product C·N·C
// against a known reference where C·N·C ≈ C (near-cancellation).
using C = np.array([[1.0, 0.5], [0.5, 1.0]], { dtype: DType.Float64 });
using N = np.array([[0.999, 0.499], [0.499, 0.999]], { dtype: DType.Float64 });
using result = np.subtract(C, np.einsum('ij,jk,kl->il', C, N, C));
// result should be small and positive-definite
// With naive summation, off-diagonals can show ~1e-16 asymmetry
console.log(await result.data());
```

## References

- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, §4.3 (compensated summation)
- NumPy uses pairwise summation: [numpy/core/src/npymath](https://github.com/numpy/numpy/blob/main/numpy/_core/src/npymath/npy_math_internal.h.src)
- JAX `precision` parameter: [jax.numpy.dot](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.dot.html)
- Python `math.fsum`: uses Shewchuk's algorithm (exact summation)
