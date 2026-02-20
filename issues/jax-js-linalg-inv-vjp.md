# `np.linalg.inv` VJP returns wrong gradient

**Discovered**: 2025-06-14 during dlm-js development  
**Affects**: `np.linalg.inv`, `np.linalg.solve` — both have broken backward pass  
**Severity**: High — any gradient flowing through matrix inverse is wrong  
**Workaround**: `adSafeInv` in `src/types.ts` (analytic cofactor expansion for m ≤ 3; Schur-complement block inversion for m ≥ 4)

## Summary

The VJP (backward pass) of `np.linalg.inv(X)` adds a spurious `−inv(X)` term
to the gradient. The correct VJP for `Y = inv(X)` is:

```
dL/dX = −Y^T · dL/dY · Y^T
```

The jax-js implementation appears to compute `dL/dX = −Y^T · dL/dY · Y^T − Y^T`
(or an equivalent error), producing gradients that are off by a factor of `(x + 1)`
for scalar matrices (where `x` is the matrix element).

`np.linalg.solve(X, I)` has the identical bug (likely shares VJP code with `inv`).

## Minimal Reproducer

```typescript
import { DType, defaultDevice, numpy as np, valueAndGrad } from '@hamk-uas/jax-js-nonconsuming';

defaultDevice('wasm');
const dtype = DType.Float64;

// f(θ) = sum(inv([[θ]])) = 1/θ
// f'(θ) = −1/θ²
// At θ=5: f'(5) = −0.04

const f = (t: np.Array) => {
  using X = np.reshape(t, [1, 1]);
  using invX = np.linalg.inv(X);
  return np.sum(invX);
};

const t = np.array(5, { dtype });
const [val, grad] = valueAndGrad(f)(t);
console.log('val:', (val.dataSync() as Float64Array)[0]);   // 0.2  ✓
console.log('grad:', (grad.dataSync() as Float64Array)[0]); // −0.24  ✗ (should be −0.04)
// Error ratio: 0.24 / 0.04 = 6 = θ + 1
```

## Error Pattern

| θ   | X = [[θ]] | Correct −1/θ² | AD gradient  | Ratio (AD/correct) |
|-----|-----------|---------------|-------------|---------------------|
| 2   | [[2]]     | −0.2500       | −0.7500     | 3  (= θ + 1)       |
| 3   | [[3]]     | −0.1111       | −0.4444     | 4  (= θ + 1)       |
| 5   | [[5]]     | −0.0400       | −0.2400     | 6  (= θ + 1)       |
| 10  | [[10]]    | −0.0100       | −0.1100     | 11 (= θ + 1)       |

The pattern `AD = correct × (θ + 1)` holds for all tested values. Equivalently:

```
AD_gradient = −1/θ² − 1/θ = −(1 + θ)/θ²
```

The extra `−1/θ` term corresponds to `−inv(X)` in matrix language.

## Verified Across

- **1×1 matrices**: error ratio = θ + 1 (always)
- **2×2 diagonal**: same ratio, same error on affected diagonal element
- **3×3 diagonal**: same pattern
- **3D batched `[n, m, m]`**: same errors
- **`np.linalg.solve(X, I)`**: identical errors (same VJP bug)
- **`np.divide(1, θ)`**: **CORRECT** — 0.00% error (scalar division VJP is fine)

## Impact on dlm-js

This bug caused the `makeKalmanLossAssoc` gradient (used for MLE via `lax.associativeScan`)
to have the **wrong sign** for the observation noise parameter. The `composeForward` function
uses `inv(I + C·J)`, and the broken VJP contaminated all gradient components flowing through it.

Symptoms:
- SCAN path (sequential, no `inv` in gradient computation): correct gradients
- ASSOC path (parallel, uses `inv` in compose): wrong gradient sign → Adam step
  increases loss instead of decreasing it

## dlm-js Workaround

Implemented `adSafeInv(X, m, dtype)` in `src/types.ts`:

- **m ≤ 3**: analytic cofactor expansion using only ops with correct VJPs
  (multiply, subtract, divide, split, concatenate). Covers Nile (m=2), order-2 (m=3).
- **m ≥ 4**: recursive Schur-complement block inversion. Partition X = [[A, B], [C, D]]
  with k = ⌊m/2⌋, j = m−k, then apply:

  ```
  X⁻¹ = [[ S⁻¹,              −S⁻¹·B·D⁻¹          ],
          [ −D⁻¹·C·S⁻¹,   D⁻¹ + D⁻¹·C·S⁻¹·B·D⁻¹ ]]
  ```

  where S = A − B·D⁻¹·C. Both sub-inverses call `adSafeInv` recursively;
  recursion bottoms out at m ≤ 3. Covers kaisaniemi (m=6), energy (m=6),
  ar2 (m=4), trigar (m=5), ozone (m=7).

  **Float32 sub-block stabilization**: before inverting D, another ε·I is
  added. The outer ε·I that `composeForward` applies to the full m×m matrix
  does not protect extracted Schur sub-blocks when float32 rounding makes
  `(C·J)` diagonal entries approach −1. Without this, energy WebGPU/f32
  training exhibited a single-iteration loss spike to ~139 600 at iter 148
  that immediately recovered but contaminated Adam's moment estimates.
  With sub-block regularization the spike is eliminated.

All ops used (einsum, add, subtract, multiply, split, concatenate) have correct VJPs.
All 112 tests pass, and ASSOC/SCAN gradients match exactly for all model sizes.
