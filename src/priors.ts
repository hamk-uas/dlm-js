/**
 * Prior distribution factories for MAP estimation in dlm-js.
 *
 * Provides {@link dlmPrior} — a factory that creates a {@link DlmLossFn}
 * callback matching MATLAB / R DLM `dlmGibbsDIG` Inverse-Gamma prior
 * conventions.
 *
 * @module
 */
import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";
import type { DlmLossFn, DlmParamMeta } from "./types";

/**
 * Inverse-Gamma(shape, rate) prior specification for a variance parameter.
 *
 * Parameterisation: `p(σ² | α, β) ∝ (σ²)^{−(α+1)} · exp(−β/σ²)`
 * where `α` = shape, `β` = rate.
 *
 * Common choices:
 * - Weakly informative: `{ shape: 2, rate: 1 }` (mode = 1/3)
 * - Non-informative (Jeffreys): `{ shape: 0.001, rate: 0.001 }`
 */
export interface InverseGammaPrior {
  /** Shape parameter α > 0. */
  shape: number;
  /** Rate parameter β > 0. */
  rate: number;
}

/**
 * Normal(mean, std) prior specification for a real-valued parameter.
 *
 * Used for AR coefficients: `p(φ | μ, σ) ∝ exp(−(φ−μ)² / (2σ²))`.
 */
export interface NormalPrior {
  /** Prior mean μ. */
  mean: number;
  /** Prior standard deviation σ > 0. */
  std: number;
}

/**
 * Prior specification for {@link dlmPrior}.
 *
 * Matches MATLAB/R DLM `dlmGibbsDIG` conventions: Inverse-Gamma priors
 * on observation and system variance(s), with optional Normal priors on
 * AR coefficients.
 *
 * @example Weakly informative IG on all variances
 * ```ts
 * import { dlmMLE, dlmPrior } from 'dlm-js';
 *
 * const prior = dlmPrior({
 *   obsVar:     { shape: 2, rate: 1 },
 *   processVar: { shape: 2, rate: 1 },
 * });
 * const result = await dlmMLE(y, { order: 1, loss: prior });
 * ```
 */
export interface DlmPriorSpec {
  /**
   * IG(shape, rate) prior on observation variance s².
   * Ignored when `obsStdFixed` is set in dlmMLE options.
   */
  obsVar?: InverseGammaPrior;
  /**
   * IG(shape, rate) prior on process variance(s) wᵢ².
   * Single spec → recycled for all m components (MATLAB DLM convention).
   * Array → per-component (length should equal state dimension m).
   */
  processVar?: InverseGammaPrior | InverseGammaPrior[];
  /**
   * Normal(mean, std) prior on AR coefficient(s) φⱼ.
   * Single spec → recycled for all p coefficients.
   * Array → per-coefficient.  Only used when `fitAr: true`.
   */
  arCoef?: NormalPrior | NormalPrior[];
}

/**
 * Create a {@link DlmLossFn} that adds Bayesian prior penalties to the
 * Kalman deviance (−2·logL).
 *
 * The returned function computes:
 *   `objective(θ) = deviance(θ) + Σ [−2 · log prior(paramᵢ)]`
 *
 * Prior penalties (dropping constants that don't affect optimisation):
 * - **Inverse-Gamma(α, β) on variance σ²** (params provides σ, the std dev):
 *     `penalty = 4(α+1)·log(σ) + 2β/σ²`
 * - **Normal(μ, σ_p) on coefficient φ**:
 *     `penalty = (φ − μ)² / σ_p²`
 *
 * All operations are AD-safe (`np.split`, `np.log`, `np.multiply`, etc.)
 * and compose inside `jit()` with zero overhead.
 *
 * @example
 * ```ts
 * import { dlmMLE, dlmPrior } from 'dlm-js';
 *
 * const prior = dlmPrior({
 *   obsVar:     { shape: 2, rate: 100 },
 *   processVar: { shape: 2, rate: 10 },
 * });
 * const result = await dlmMLE(y, { order: 1, loss: prior });
 * // result.priorPenalty > 0
 * ```
 */
export function dlmPrior(spec: DlmPriorSpec): DlmLossFn {
  // Pre-validate at factory time
  if (spec.obsVar) {
    if (spec.obsVar.shape <= 0 || spec.obsVar.rate <= 0)
      throw new Error('dlmPrior: obsVar shape and rate must be > 0');
  }
  if (spec.processVar) {
    const pv = Array.isArray(spec.processVar) ? spec.processVar : [spec.processVar];
    for (const p of pv) {
      if (p.shape <= 0 || p.rate <= 0)
        throw new Error('dlmPrior: processVar shape and rate must be > 0');
    }
  }
  if (spec.arCoef) {
    const ac = Array.isArray(spec.arCoef) ? spec.arCoef : [spec.arCoef];
    for (const a of ac) {
      if (a.std <= 0)
        throw new Error('dlmPrior: arCoef std must be > 0');
    }
  }

  return (deviance: np.Array, params: np.Array, meta: DlmParamMeta): np.Array => {
    const { nObs, nProcess, nAr } = meta;

    // Split params into [obs?, process, ar?] parts — np.split has working VJP
    const splitPts: number[] = [];
    if (nObs > 0 && (nAr > 0 || spec.obsVar)) splitPts.push(nObs);
    if (nAr > 0) splitPts.push(nObs + nProcess);
    const parts = splitPts.length > 0 ? np.split(params, splitPts, 0) : [params];

    let partIdx = 0;
    const obsPart  = nObs > 0 && splitPts.length > 0 ? parts[partIdx++] : undefined;
    const processPart = parts[partIdx++];
    const arPart   = nAr > 0 && splitPts.includes(nObs + nProcess) ? parts[partIdx++] : undefined;

    // Accumulator — starts from deviance (not owned by us, never dispose it)
    // jax-js-lint: allow-non-using
    let total = deviance;
    const swapTotal = (penalty: np.Array) => {
      // jax-js-lint: allow-non-using
      const newTotal = np.add(total, penalty);
      if (total !== deviance) total.dispose();
      total = newTotal;
    };

    // ── Inverse-Gamma on observation variance ──
    // IG(α, β) on s² where params has s (std dev):
    //   −2 log p(s²) = 4(α+1)·log(s) + 2β/s²
    if (spec.obsVar && obsPart) {
      const { shape: alpha, rate: beta } = spec.obsVar;
      using logS = np.log(obsPart);
      using sumLogS = np.sum(logS);
      using shapeCoeff = np.array(4 * (alpha + 1));
      using shapeTerm = np.multiply(shapeCoeff, sumLogS);
      using s2 = np.square(obsPart);
      using one_s = np.array(1);
      using invS2 = np.divide(one_s, s2);
      using sumInvS2 = np.sum(invS2);
      using rateCoeff = np.array(2 * beta);
      using rateTerm = np.multiply(rateCoeff, sumInvS2);
      using penalty = np.add(shapeTerm, rateTerm);
      swapTotal(penalty);
    }

    // ── Inverse-Gamma on process variance(s) ──
    // Vectorised: build coefficient arrays for per-component (α_i, β_i)
    if (spec.processVar && processPart) {
      const pvArr = Array.isArray(spec.processVar)
        ? spec.processVar : new Array(nProcess).fill(spec.processVar);
      const n = Math.min(pvArr.length, nProcess);
      // Coefficients: 4(α_i+1) for shape, 2β_i for rate
      const shapeCoeffs = Array.from({ length: n }, (_, i) => 4 * (pvArr[i].shape + 1));
      const rateCoeffs  = Array.from({ length: n }, (_, i) => 2 * pvArr[i].rate);
      // Pad to nProcess if spec array is shorter (extra components get 0 penalty)
      while (shapeCoeffs.length < nProcess) { shapeCoeffs.push(0); rateCoeffs.push(0); }

      using logW = np.log(processPart);
      using sc = np.array(shapeCoeffs);
      using _scLogW = np.multiply(sc, logW);
      using shapePenalty = np.sum(_scLogW);

      using w2 = np.square(processPart);
      using one_w = np.array(1);
      using invW2 = np.divide(one_w, w2);
      using rc = np.array(rateCoeffs);
      using _rcInvW2 = np.multiply(rc, invW2);
      using ratePenalty = np.sum(_rcInvW2);

      using penalty = np.add(shapePenalty, ratePenalty);
      swapTotal(penalty);
    }

    // ── Normal on AR coefficients ──
    // −2 log N(φ | μ, σ_p) = (φ − μ)²/σ_p²
    if (spec.arCoef && arPart) {
      const acArr = Array.isArray(spec.arCoef)
        ? spec.arCoef : new Array(nAr).fill(spec.arCoef);
      const n = Math.min(acArr.length, nAr);
      const means  = Array.from({ length: n }, (_, i) => acArr[i].mean);
      const invVar = Array.from({ length: n }, (_, i) => 1 / (acArr[i].std * acArr[i].std));
      while (means.length < nAr) { means.push(0); invVar.push(0); }

      using mu = np.array(means);
      using iv = np.array(invVar);
      using diff = np.subtract(arPart, mu);
      using _diff2 = np.square(diff);
      using _weighted = np.multiply(iv, _diff2);
      using penalty = np.sum(_weighted);
      swapTotal(penalty);
    }

    return total;
  };
}
