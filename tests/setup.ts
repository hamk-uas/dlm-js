/**
 * Vitest global setup — leak detection for every test.
 *
 * Wraps each test in checkLeaks.start() / stop() so that any un-disposed
 * np.Array backend slots are caught automatically.
 */

import { checkLeaks } from '@hamk-uas/jax-js-nonconsuming';
import { afterEach, beforeEach, expect } from 'vitest';

beforeEach(() => {
  checkLeaks.start();
});

afterEach(() => {
  const result = checkLeaks.stop();
  expect(result.userLeaked, result.summary).toBe(0);
});
