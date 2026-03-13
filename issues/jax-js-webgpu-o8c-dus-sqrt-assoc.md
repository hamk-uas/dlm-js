# O8c command tape DUS CopyBufferToBuffer size mismatch for small tensors in assocScan

**Status**: 🔴 Open

## Summary

The v0.9.1 O8c command tape DUS encoding (`634ac28e`) produces a `CopyBufferToBuffer` validation error when `lax.associativeScan` composes tuples containing small tensors (m=1 or m=2 state dimensions). The copy size exceeds the source buffer size, causing data corruption in the affected scan outputs.

## Affected version

`v0.9.1` (`44b31d3`) — specifically the O8c DUS commit `634ac28e`.

## Root cause hypothesis

The DUS pre-encoded `copyBufferToBuffer` computes `size` from the *target* buffer (or some aligned block size) rather than the *source* buffer when the source/update tensor is smaller than the target's aligned extent. For sqrt-assoc scan tuples with m=1 (each field is `[n,1,1]` = 100 elements = 400 bytes), the tape tries to copy 2048 bytes from a 800-byte buffer. The ~2.5× ratio suggests rounding to a 2048-byte alignment boundary.

## Reproduction

Run the sqrt-assoc algorithm on WebGPU with a model that has small state dimension (m=1 or m=2). The standard assoc algorithm on the same data does not trigger the error.

```
pnpm run bench:full  # run in Chromium via Playwright; observe stderr for CopyBufferToBuffer errors
```

Or use the standalone repro script:
```
DISPLAY=:1 pnpm vitest run tests/webgpu-scan.test.ts -c tests/vitest.nvidia.config.ts
```

See `issues/repro-o8c-dus-sqrt-assoc.ts` for a minimal repro (requires WebGPU; runs in Chromium via Playwright).

### WebGPU validation error (stderr)

```
Copy range (offset: 0, size: 2048) does not fit in [Buffer (unlabeled)] size (800).
 - While validating source [Buffer (unlabeled)] copy size.
 - While encoding [CommandEncoder (unlabeled)].CopyBufferToBuffer(...)
 - While finishing [CommandEncoder (unlabeled)].
```

### Observed results (bench-full, webgpu/f32/sqrt-assoc vs Octave reference)

| Model | m | maxAbsErr | maxPctErr | Status |
|-------|---|-----------|-----------|--------|
| Nile order=0 | 1 | 1083 | 100% | ❌ Corrupted |
| Nile order=1 | 2 | 986 | 156,233% | ❌ Corrupted |
| Kaisaniemi trig | 4 | 0.029 | 202% | ✅ Expected f32 |
| Energy trig+AR | 5 | 0.007 | 4% | ✅ Good |
| Gapped order=1 | 2 | 1026 | 142,000% | ❌ Corrupted |

The same models produce correct results with assoc (non-sqrt) on WebGPU/f32 (maxAbsErr < 0.02 for all), and with sqrt-assoc on WASM/f32 (all finite, reasonable errors).

## Impact on dlm-js

- sqrt-assoc on WebGPU is corrupted for m ≤ 2 (state dimensions up to 2). m ≥ 4 works correctly.
- **Workaround locations**: `tests/webgpu-scan.test.ts` (SQRT_ASSOC_SKIP set skips m≤2 models).
- No impact on other WebGPU algorithms (scan, assoc, ud) — only the sqrt-assoc tuple composition hits this DUS edge case.

## Suggested fix

Check that the `copyBufferToBuffer` size in the DUS tape encoding is `min(copySize, sourceBuffer.size)` or uses the source buffer's actual size rather than the aligned target extent. The DUS copy should respect both source and destination buffer sizes.
