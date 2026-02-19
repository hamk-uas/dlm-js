# Replace DARE with exact parallel forward Kalman filter (Lemmas 1–2)

## The misconception

When implementing the WebGPU parallel Kalman filter + RTS smoother, we cited Särkkä & García-Fernández (2020) [1] and correctly used **Lemmas 5–6** (backward smoother elements) and **Theorem 2** (associative composition) for the backward RTS smoother pass. However, we **overlooked Lemmas 1–2** of the *same paper*, which provide an equally exact parallel forward Kalman filter using 5-tuple elements $(A_k, b_k, C_k, \eta_k, J_k)$ with an associative compose rule.

Instead, we independently devised a DARE (Discrete Algebraic Riccati Equation) steady-state approximation for the forward filter — a method from classical control theory that computes a single $K_{ss}$ and applies it at every timestep. This introduces ~1–2% early-step bias because the first ~5 steps use a converged gain rather than the true time-varying Kalman gain. The exact parallel forward filter from Lemmas 1–2 has **no such approximation** — it produces the same filtered states and covariances as the sequential filter, up to floating-point reordering.

**In summary:** we read the backward-smoother half of the paper but missed the forward-filter half, then reinvented an inferior workaround for the part we missed.

## Current architecture (DARE, to be replaced)

### Forward filter (3-tuple + constant $K_{ss}$)

1. **DARE pre-computation** (`solveDAREForKss` in `src/index.ts`, lines 15–107): Iterates the Riccati equation to convergence, producing steady-state $K_{ss}$.
2. **Per-timestep elements**: $(A_t, b_t, \Sigma_t)$ where $A_t = G - K_{ss} F$ (constant), $b_t = K_{ss} y_t$, $\Sigma_t = W + V^2 K_{ss} K_{ss}^\top$ (constant when $V^2$ is constant).
3. **Compose**: Simple affine — $(A_2 A_1, A_2 b_1 + b_2, A_2 \Sigma_1 A_2^\top + \Sigma_2)$. No matrix inverse needed.

### Forward filter in MLE (`makeKalmanLossAssoc` in `src/mle.ts`)

Same as above but with the 50-step Riccati iteration **unrolled inside the AD tape** so gradients flow through $\theta \to (W, V^2) \to K_{ss} \to \text{loss}$.

### Backward smoother (exact, unchanged)

Already uses Lemmas 5–6 with per-timestep gains $E_k$ via batched `np.linalg.inv`. This pass is **not affected** by the refactor.

## Target architecture (exact 5-tuple from Lemmas 1–2)

### Forward filter elements (Lemma 1)

For $k \geq 2$:

$$
\begin{aligned}
S_k &= H_k Q_{k-1} H_k^\top + R_k \\
K_k &= Q_{k-1} H_k^\top S_k^{-1} \\
A_k &= (I - K_k H_k) F_{k-1} \\
b_k &= u_{k-1} + K_k (y_k - H_k u_{k-1} - d_k) \\
C_k &= (I - K_k H_k) Q_{k-1} \\
\eta_k &= F_{k-1}^\top H_k^\top S_k^{-1} (y_k - H_k u_{k-1} - d_k) \\
J_k &= F_{k-1}^\top H_k^\top S_k^{-1} H_k F_{k-1}
\end{aligned}
$$

**Mapping to dlm-js notation** (scalar observation, $H_k = F$ row vector, $F_{k-1} = G$ transition, $Q_{k-1} = W$ state noise, $R_k = V^2$ obs noise, $u_{k-1} = 0$ no control, $d_k = 0$ no obs offset):

$$
\begin{aligned}
S_k &= F W F^\top + V^2 \quad (\text{scalar}) \\
K_k &= W F^\top / S_k \quad [m \times 1] \\
A_k &= (I - K_k F) G \quad [m \times m] \\
b_k &= K_k y_k \quad [m] \\
C_k &= (I - K_k F) W \quad [m \times m] \\
\eta_k &= G^\top F^\top y_k / S_k \quad [m] \\
J_k &= G^\top F^\top F G / S_k \quad [m \times m]
\end{aligned}
$$

**Note on constant elements**: When $G$, $F$, $W$, $V^2$ are time-invariant (as in dlm-js DLMs without covariates), the gain $K_k$, transition $A_k$, covariance $C_k$, and matrix $J_k$ are **all constant** — only $b_k$ and $\eta_k$ vary with $y_k$. This means only 2 of the 5 tuple components need per-timestep computation; the other 3 are batch-broadcast constants.

**First element** ($k = 1$): uses the prior $m_1^-$, $P_1^-$:

$$A_1 = 0, \quad b_1 = m_1^- + K_1 (y_1 - F m_1^- - d_1), \quad C_1 = P_1^- - K_1 S_1 K_1^\top$$

where $S_1 = F P_1^- F^\top + V^2$, $K_1 = P_1^- F^\top S_1^{-1}$.

### Forward filter compose (Lemma 2)

$$(A_j, b_j, C_j, \eta_j, J_j) \oplus (A_i, b_i, C_i, \eta_i, J_i) =$$

$$
\begin{aligned}
M_{ij} &= (I + C_i J_j)^{-1} \\
A_{ij} &= A_j M_{ij} A_i \\
b_{ij} &= A_j M_{ij} (b_i + C_i \eta_j) + b_j \\
C_{ij} &= A_j M_{ij} C_i A_j^\top + C_j \\
\eta_{ij} &= A_i^\top (I + J_j C_i)^{-1} (\eta_j - J_j b_i) + \eta_i \\
J_{ij} &= A_i^\top (I + J_j C_i)^{-1} J_j A_i + J_i
\end{aligned}
$$

**Key difference from current 3-tuple compose**: requires $(I + C_i J_j)^{-1}$ (and its transpose-partner $(I + J_j C_i)^{-1}$) at each compose step. This will use `np.linalg.inv`, matching the pattern already used by the backward smoother.

**Note**: $(I + J_j C_i)^{-1} = I - J_j (I + C_i J_j)^{-1} C_i$ (push-through identity), so only one $[m \times m]$ inverse is needed — compute $M$ then derive the other.

### After the prefix scan

The composed elements $(A_{\text{comp},k}, b_{\text{comp},k}, C_{\text{comp},k}, \_, \_)$ give:

$$x_{\text{filt},k} = A_{\text{comp},k} \, x_0 + b_{\text{comp},k}, \quad P_{\text{filt},k} = A_{\text{comp},k} \, P_0 \, A_{\text{comp},k}^\top + C_{\text{comp},k}$$

The $\eta$ and $J$ components are auxiliary — needed only during composition, not for the final state/covariance extraction.

## Implementation plan

### Step 1: `dlmSmo` forward filter in `src/index.ts`

1. **Delete** `solveDAREForKss` (lines 15–107) and its call site (lines 217–228).
2. **Replace** the forward scan element construction (lines 468–519) with 5-tuple elements per Lemma 1.
3. **Replace** the forward compose function (lines 525–537) with Lemma 2 compose using `np.linalg.inv`.
4. **Remove** the `sysData` parameter from `dlmSmo` (it was only used to pass DARE inputs).
5. **Remove** the `dare` disposal block (lines 808–813).
6. **Verify**: run `pnpm vitest run` — the WebGPU path should now match the sequential path exactly (within float32 rounding), eliminating the ~1–2% early-step bias.

### Step 2: MLE loss in `src/mle.ts`

1. **Replace** the traced DARE (50 unrolled Riccati iterations, lines 346–385) with 5-tuple element construction per Lemma 1.
2. **Replace** the MLE compose function with Lemma 2 compose.
3. **Remove** the `dareIter` parameter.
4. **Eliminate** the `mean(V²)` policy — the exact forward filter uses per-timestep $V^2(t)$ directly in the element construction.
5. **Verify**: MLE convergence should improve (−2logL should match the sequential path, currently ~1121.5 vs ~1104.9 for Nile).

### Step 3: Missing-data (NaN) support

The element construction must handle NaN observations. For missing timesteps:
- Set $K_k = 0$ (no observation update)
- $A_k = G$ (pure prediction), $b_k = 0$, $C_k = W$
- $\eta_k = 0$, $J_k = 0$

This matches the current float-mask blending pattern.

### Step 4: Tests and validation

1. Run the full test suite: `pnpm vitest run`.
2. The WebGPU tolerance (currently `relTol=1e-2` to accommodate DARE bias) may be tightened.
3. Run `pnpm run bench:backends` to update timing tables.
4. Run `pnpm run bench:mle` to update MLE comparison tables.
5. Update documentation to remove DARE references and mark both passes as exact.

### Step 5: Cleanup

1. Remove all "planned" / "currently DARE" annotations from documentation.
2. Update the math provenance table in `mle-comparison.md`.
3. Remove `issues/dare-to-exact-parallel.md` or mark as completed.

## Performance considerations

- **Compose cost increases**: The 5-tuple compose requires one $[m \times m]$ matrix inverse per compose step (via push-through identity, only one `np.linalg.inv` call needed). For $m \leq 5$ (typical DLMs), this is negligible. The backward smoother already does per-element inversions.
- **Element count increases**: 5 tensors per element vs 3. More GPU memory, but each tensor is $[m \times m]$ or $[m]$ — tiny for practical $m$.
- **Constant elements**: For time-invariant systems, $A_k$, $C_k$, $J_k$ are constant across all timesteps — only $b_k$ and $\eta_k$ vary. This can be exploited to reduce memory and compute.

## References

1. Särkkä, S. & García-Fernández, Á. F. (2020). [Temporal Parallelization of Bayesian Smoothers](https://arxiv.org/abs/1905.13002). *IEEE Transactions on Automatic Control*, 66(1), 299–306. doi:[10.1109/TAC.2020.2976316](https://doi.org/10.1109/TAC.2020.2976316).
2. Yaghoobi, M., Corenflos, A., Hassan, S. S. & Särkkä, S. (2021). [Parallel Iterated Extended and Sigma-Point Kalman Smoothers](https://arxiv.org/abs/2102.00514). *ICASSP 2021*. — Applies the 5-tuple formulation from [1, Lemma 1] to nonlinear models; confirms the compose rule in Eq. 12–15 (linear case equivalent to [1, Lemma 2]).
