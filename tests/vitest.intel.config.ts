/**
 * Vitest config that forces the Intel integrated GPU.
 *
 * Uses VK_DRIVER_FILES to bypass the NVIDIA discrete GPU and force
 * Chromium's Vulkan backend to use the Intel mesa driver.
 *
 * Usage:
 *   pnpm vitest run tests/webgpu-scan.test.ts -c tests/vitest.intel.config.ts
 *   GPU=intel bash scripts/gpu-test.sh run tests/webgpu-scan.test.ts
 */
import { gpuConfig } from "./gpu-config";

export default gpuConfig("intel");
