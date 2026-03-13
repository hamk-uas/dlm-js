/**
 * Repro: O8c command tape DUS CopyBufferToBuffer size mismatch
 *
 * Run in Chromium via Playwright (needs WebGPU):
 *   pnpm vitest run issues/repro-o8c-dus-sqrt-assoc.ts -c tests/vitest.nvidia.config.ts
 *
 * Expected: sqrt-assoc yhat values should be close to assoc yhat values.
 * Actual: sqrt-assoc produces garbage for m=1 models on WebGPU due to
 *   CopyBufferToBuffer size mismatch in the O8c command tape DUS encoding.
 */
import { dlmFit, toMatlab } from '../src/index';
import { it, expect } from 'vitest';
import { commands } from 'vitest/browser';

it('sqrt-assoc m=1 on WebGPU should match assoc', async () => {
  const input = JSON.parse(await commands.readFile('tests/order0-in.json'));

  const sqrtResult = await dlmFit(input.y, {
    obsStd: input.s, processStd: [input.w],
    dtype: 'f32', order: 0, algorithm: 'sqrt-assoc',
  });
  const sqrtYhat = toMatlab(sqrtResult).yhat as Float32Array;

  const assocResult = await dlmFit(input.y, {
    obsStd: input.s, processStd: [input.w],
    dtype: 'f32', order: 0, algorithm: 'assoc',
  });
  const assocYhat = toMatlab(assocResult).yhat as Float32Array;

  // Compare: maxAbsErr should be < 1.0 (f32 precision)
  let maxAbsErr = 0;
  for (let i = 0; i < sqrtYhat.length; i++) {
    maxAbsErr = Math.max(maxAbsErr, Math.abs(sqrtYhat[i] - assocYhat[i]));
  }
  console.log('maxAbsErr (sqrt-assoc vs assoc):', maxAbsErr);
  console.log('sqrt-assoc yhat[0..4]:', Array.from(sqrtYhat.slice(0, 5)));
  console.log('assoc yhat[0..4]:', Array.from(assocYhat.slice(0, 5)));

  // This will FAIL: maxAbsErr is ~1000 due to O8c DUS corruption
  expect(maxAbsErr).toBeLessThan(1.0);
});
