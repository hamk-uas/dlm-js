import { describe, it, expect } from 'vitest';
import { dlmFit, dlmGenSys, dlmGenSysTV, dlmMLE, dlmForecast, dlmPrior } from '../src/index';

/**
 * Verify that all public API functions throw on unknown option keys.
 * Prevents silent typos and MATLAB-name mismatches (e.g. `trig` vs `harmonics`).
 */
describe('unknown option validation', () => {
  // ── dlmGenSys ──
  it('dlmGenSys throws on MATLAB name trig with hint', () => {
    expect(() => dlmGenSys({ trig: 1 } as never)).toThrow(/use 'harmonics' instead/i);
  });

  it('dlmGenSys throws on MATLAB name ns with hint', () => {
    expect(() => dlmGenSys({ ns: 12 } as never)).toThrow(/use 'seasonLength' instead/i);
  });

  it('dlmGenSys throws on MATLAB name fullseas with hint', () => {
    expect(() => dlmGenSys({ fullseas: true } as never)).toThrow(/use 'fullSeasonal' instead/i);
  });

  it('dlmGenSys throws on MATLAB name arphi with hint', () => {
    expect(() => dlmGenSys({ arphi: [0.5] } as never)).toThrow(/use 'arCoefficients' instead/i);
  });

  it('dlmGenSys throws on arbitrary typo without hint', () => {
    expect(() => dlmGenSys({ ordeer: 1 } as never)).toThrow(/unknown option 'ordeer'/);
    expect(() => dlmGenSys({ ordeer: 1 } as never)).not.toThrow(/MATLAB/);
  });

  it('dlmGenSys accepts all valid keys', () => {
    expect(() => dlmGenSys({
      order: 1, harmonics: 0, seasonLength: 12,
      fullSeasonal: false, arCoefficients: [], spline: false,
    })).not.toThrow();
  });

  // ── dlmGenSysTV ──
  it('dlmGenSysTV throws on unknown key', () => {
    expect(() => dlmGenSysTV({ trig: 1 } as never, [0, 1, 2], [1])).toThrow(/unknown option 'trig'/i);
  });

  // ── dlmFit ──
  it('dlmFit rejects MATLAB name trig with hint', async () => {
    await expect(dlmFit([1, 2, 3], {
      obsStd: 1, processStd: [1, 1], trig: 1,
    } as never)).rejects.toThrow(/use 'harmonics' instead/i);
  });

  it('dlmFit rejects MATLAB name ns with hint', async () => {
    await expect(dlmFit([1, 2, 3], {
      obsStd: 1, processStd: [1, 1], ns: 12,
    } as never)).rejects.toThrow(/use 'seasonLength' instead/i);
  });

  it('dlmFit rejects arbitrary typo', async () => {
    await expect(dlmFit([1, 2, 3], {
      obsStd: 1, processStd: [1, 1], ordeer: 1,
    } as never)).rejects.toThrow(/unknown option 'ordeer'/i);
  });

  it('dlmFit rejects unknown stabilization key', async () => {
    await expect(dlmFit([1, 2, 3], {
      obsStd: 1, processStd: [1, 1],
      stabilization: { unknownFlag: true } as never,
    })).rejects.toThrow(/unknown option 'unknownFlag'/i);
  });

  it('dlmFit rejects unknown stabilization preset', async () => {
    await expect(dlmFit([1, 2, 3], {
      obsStd: 1, processStd: [1, 1],
      stabilization: 'turbo' as never,
    })).rejects.toThrow(/unknown stabilization preset 'turbo'/i);
  });

  it('dlmFit rejects deprecated cEps flag', async () => {
    await expect(dlmFit([1, 2, 3], {
      obsStd: 1, processStd: [1, 1],
      stabilization: { cEps: true } as never,
    })).rejects.toThrow(/unknown option 'cEps'/i);
  });

  // ── dlmMLE ──
  it('dlmMLE rejects MATLAB name trig with hint', async () => {
    await expect(dlmMLE([1, 2, 3], {
      trig: 1,
    } as never)).rejects.toThrow(/use 'harmonics' instead/i);
  });

  it('dlmMLE rejects MATLAB name fitar with hint', async () => {
    await expect(dlmMLE([1, 2, 3], {
      fitar: true,
    } as never)).rejects.toThrow(/use 'params\.arCoefficients\.fit' instead/i);
  });

  it('dlmMLE rejects unknown params.processStd key', async () => {
    await expect(dlmMLE([1, 2, 3], {
      params: { processStd: { observation: 1 } as never },
    })).rejects.toThrow(/unknown option 'observation'/i);
  });

  it('dlmMLE rejects wrong params.processStd.init length', async () => {
    await expect(dlmMLE([1, 2, 3], {
      order: 1,
      params: { processStd: { init: [0.2] } },
    })).rejects.toThrow(/params\.processStd\.init must have length 2, got 1/i);
  });

  it('dlmMLE rejects wrong params.processStd.groups length', async () => {
    await expect(dlmMLE([1, 2, 3], {
      order: 1,
      harmonics: 1,
      params: { processStd: { groups: [0, 1, 2] } },
    })).rejects.toThrow(/groups must have length 4, got 3/i);
  });

  it('dlmMLE rejects conflicting fixed values within one processStd group', async () => {
    await expect(dlmMLE([1, 2, 3], {
      order: 1,
      harmonics: 1,
      params: {
        processStd: {
          groups: [0, 1, 2, 2],
          fixed: [undefined, undefined, 0.1, 0.2],
        },
      },
    })).rejects.toThrow(/group '2' has conflicting fixed values 0\.1 and 0\.2/i);
  });

  it('dlmMLE rejects wrong params.arCoefficients.init length', async () => {
    await expect(dlmMLE([1, 2, 3], {
      order: 0,
      arCoefficients: [0.5],
      params: { arCoefficients: { fit: true, init: [0.4, 0.3] } },
    })).rejects.toThrow(/params\.arCoefficients\.init must have length 1, got 2/i);
  });

  it('dlmMLE rejects unknown callbacks key', async () => {
    await expect(dlmMLE([1, 2, 3], {
      callbacks: { onStep: () => {} } as never,
    })).rejects.toThrow(/unknown option 'onStep'/i);
  });

  it('dlmMLE rejects unknown adamOpts key', async () => {
    await expect(dlmMLE([1, 2, 3], {
      adamOpts: { beta1: 0.9 } as never,
    })).rejects.toThrow(/unknown option 'beta1'/i);
  });

  it('dlmMLE rejects unknown naturalOpts key', async () => {
    await expect(dlmMLE([1, 2, 3], {
      optimizer: 'natural',
      naturalOpts: { learningRate: 0.01 } as never,
    })).rejects.toThrow(/unknown option 'learningRate'/i);
  });

  // ── dlmForecast ──
  it('dlmForecast rejects unknown key', async () => {
    // Create a minimal mock fit result
    const mockFit = {
      p: 1, n: 3, m: 2,
      G: [[1, 1], [0, 1]],
      F: [1, 0],
      W: [[1, 0], [0, 1]],
      smoothed: { get: () => 0 },
      smoothedCov: { get: () => 0 },
      covariates: [],
    } as never;
    await expect(dlmForecast(mockFit, 5, {
      algorithm: 'scan',
    } as never)).rejects.toThrow(/unknown option 'algorithm'/i);
  });

  // ── dlmPrior ──
  it('dlmPrior throws on unknown key', () => {
    expect(() => dlmPrior({
      obsVariance: { shape: 2, rate: 1 },
    } as never)).toThrow(/unknown option 'obsVariance'/i);
  });

  it('dlmPrior accepts all valid keys', () => {
    expect(() => dlmPrior({
      obsVar: { shape: 2, rate: 1 },
      processVar: { shape: 2, rate: 1 },
      arCoef: { mean: 0, std: 1 },
    })).not.toThrow();
  });
});
