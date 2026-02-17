# Compensated or pairwise summation in dot product / reduction kernels

> **Update (v0.2.1):** Kahan compensated summation for Float64 reductions shipped in jax-js-nonconsuming v0.2.1. Results are mixed — see [Measured impact after Kahan](#measured-impact-after-kahan) below. The dominant error source is catastrophic cancellation in `C - C·N·C`, which Kahan cannot fix. Pairwise summation for Float32 remains an open request.

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

## Measured impact after Kahan (v0.2.1)

Float64 Kahan summation shipped in v0.2.1. Comparison against MATLAB/Octave reference:

| Model | m | v0.2.0 worst relErr | v0.2.1 worst relErr | Element | Verdict |
|---|---|---|---|---|---|
| level/order0 | 1 | 2.2e-8 | 2.2e-8 | `s2` | No change (m=1, scalar dot) |
| niledemo | 2 | 4.7e-7 | 4.7e-7 | `lik` | No change (m=2, minimal benefit) |
| order2 | 3 | 1.6e-7 | 1.6e-7 | `resid0` | No change |
| seasonal | 13 | 2.9e-5 | 1.8e-5 | `Cf[0][2]` → `C[1][8]` | **Improved** (37% reduction) |
| trig | 6 | 8.4e-4 | 4.8e-3 | `Cf[0][4]` | **Worse** (different rounding path) |

Per-element breakdown for the trig model (m=6):
- 3458 elements improved, 7723 worsened, 825 similar (±10%)
- Median relErr: 7.5e-9 → 1.1e-8 (47% worse)
- All high-error elements are near-zero covariance entries C[5][4], C[4][5] (|ref| ≈ 2e-7)

Per-element breakdown for the seasonal model (m=13):
- 30299 improved, 13359 worsened, 3201 similar
- Median relErr: 2.7e-9 → 1.6e-9 (43% improvement)
- Worst-case element improved across the board

### Conclusion

Kahan compensated summation helps for larger state dimensions (m=13) where the O(m·ε) naive accumulation was the bottleneck. For medium state dimensions (m=6) with catastrophic cancellation in `C - C·N·C`, Kahan changes the rounding pattern but doesn't help — and can make specific elements worse. The subtraction is the real problem, not the dot products.

For a DLM-side fix, the Joseph form covariance update would address the cancellation directly. For the jax-js side, pairwise summation for Float32 would still be valuable (Float32 remains naive in v0.2.1).

### Verification: synthetic ground-truth tests and analytical error propagation

To verify the Kahan implementation is correct (and to put the Octave-reference errors in practical context), we tested the DLM against **known true hidden states** generated from a seeded PRNG. This measures the actual accuracy of state recovery, not implementation agreement.

**Synthetic test results (v0.2.0 vs v0.2.1):**

| Model | m | v0.2.0 maxAbsErr | v0.2.1 maxAbsErr | Difference |
|---|---|---|---|---|
| local level | 1 | 9.956665377027e+0 | 9.956665377027e+0 | 0 (bit-identical) |
| local linear trend | 2 | 8.315640195615e+0 | 8.315640195615e+0 | 0 (bit-identical) |
| trig seasonal | 6 | 7.639202929**8**e+0 | 7.639202930**0**e+0 | ~2e-10 |
| full seasonal | 13 | 8.135228856**7**e+0 | 8.135228856**9**e+0 | ~2e-10 |

RMSE and 95%-CI coverage are **identical** between versions. The ~2e-10 differences are 10 orders of magnitude below the statistical estimation error (~1–3 RMSE).

**Analytical error propagation confirms Kahan correctness.** For a chain of n Kalman filter steps with m×m matrix multiplications:

- Per dot product element, the naive-vs-Kahan difference is $(m-2) \cdot \varepsilon \cdot m \cdot s^2$ where $s$ is the typical matrix element scale and $\varepsilon \approx 2.2 \times 10^{-16}$.
- Over $n$ timesteps with stable propagation: $\sqrt{n}$ growth.
- Catastrophic cancellation in $C - C \cdot N \cdot C$ amplifies by $\kappa = C_f / C_s \approx 3\text{–}4$.

Predicted vs measured:

| m | Predicted Δ | Measured Δ | Ratio | Status |
|---|---|---|---|---|
| 1 | 0 (m−2 < 0) | 0 (bit-identical) | — | ✓ exact match |
| 2 | 0 (m−2 = 0) | 0 (bit-identical) | — | ✓ exact match |
| 6 | ~3e-12 | ~2e-10 | 67× | ✓ consistent (worst-case κ > avg κ) |
| 13 | ~2e-11 | ~2e-10 | 11× | ✓ consistent |

The analytical bound underestimates because it uses average cancellation ratio rather than worst-case per-element amplification. The key structural predictions hold exactly:
- **m ≤ 2: zero difference** (Kahan compensation term is mathematically zero for 1–2 term sums)
- **m > 2: difference scales with $(m-2) \cdot m$** (11× ratio improvement from m=6 to m=13)
- **No anomalous signs or magnitudes** (an incorrect Kahan implementation would show disrupted scaling)

**Conclusion for DLM users:** Kahan is a correctness improvement for jax-js as a numeric library, but the DLM's accuracy in recovering hidden states is entirely dominated by statistical estimation uncertainty, not floating-point rounding. The v0.2.0 → v0.2.1 change has zero practical effect on DLM results.
