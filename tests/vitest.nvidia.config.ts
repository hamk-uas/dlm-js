/**
 * Vitest config for NVIDIA GPU tests and benchmarks.
 *
 * Selects the NVIDIA discrete GPU (Chromium's default Vulkan adapter priority).
 * Enables f16 via Dawn feature flag.
 *
 * Usage:
 *   pnpm vitest run tests/webgpu-scan.test.ts -c tests/vitest.nvidia.config.ts
 *   GPU=nvidia bash scripts/gpu-test.sh run tests/webgpu-scan.test.ts
 */
import { gpuConfig } from "./gpu-config";

export default gpuConfig("nvidia");
