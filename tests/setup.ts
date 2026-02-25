/**
 * Vitest global setup — leak detection for every test.
 *
 * Wraps each test in checkLeaks.start() / stop() so that any un-disposed
 * np.Array backend slots are caught automatically.  This replaces the
 * per-call `withLeakCheck` wrapper that was easy to forget and, worse,
 * silently discarded the leak report.
 *
 * ## Upstream leak budget
 *
 * Some tests (notably dlmMLE) have irreducible leaks caused by upstream
 * jax-js-nonconsuming bugs:
 *  - `[Symbol.dispose]()` is a no-op during PE tracing, so inline
 *    constants inside JIT-traced bodies are never freed.
 *  - `ClosedJaxpr.dispose()` is non-recursive — nested scan/assocScan
 *    sub-jaxpr constants are never freed.
 *
 * These leaks are tagged as `userLeaked` because the creation site is in
 * our source files, even though the root cause is upstream.
 *
 * Tests can declare a per-test leak budget via:
 *   globalThis.__jaxUserLeakBudget = N;
 * The budget resets to 0 after every test.  Tests without a budget must
 * have exactly 0 user-leaked slots.
 */

import { checkLeaks } from '@hamk-uas/jax-js-nonconsuming';
import { afterEach, beforeEach, expect } from 'vitest';

declare global {
  // eslint-disable-next-line no-var
  var __jaxUserLeakBudget: number | undefined;
}

beforeEach(() => {
  checkLeaks.start();
});

afterEach(() => {
  const result = checkLeaks.stop();
  const budget = globalThis.__jaxUserLeakBudget ?? 0;
  globalThis.__jaxUserLeakBudget = undefined;
  expect(result.userLeaked, result.summary).toBeLessThanOrEqual(budget);
});
