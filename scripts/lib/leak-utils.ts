/**
 * Shared leak-check helper for dlm-js scripts.
 *
 * Mirrors `withLeakCheck` from `tests/utils.ts` — kept separate because
 * scripts cannot import from the `tests/` directory.
 *
 * Usage:
 *   import { withLeakCheck } from './lib/leak-utils.ts';
 *   const result = await withLeakCheck(() => dlmFit(y, s, w, dtype));
 */

import { checkLeaks } from '@hamk-uas/jax-js-nonconsuming';

/**
 * Run `fn` inside a checkLeaks guard. Throws if any np.Array objects leak.
 * Use this to wrap every `dlmFit`/`dlmSmo`/`dlmForecast`/`dlmMLE` call.
 */
export const withLeakCheck = async <T>(fn: () => Promise<T>): Promise<T> => {
  const guard = checkLeaks.start();
  try {
    return await fn();
  } finally {
    checkLeaks.stop(guard);
  }
};
