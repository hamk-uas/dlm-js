/**
 * Shared GPU vitest config factory.
 *
 * Both NVIDIA and Intel GPU configs use this factory so that launch args,
 * server headers, aliases, and timeouts stay in sync.
 *
 * Usage from a GPU config:
 *   import { gpuConfig } from "./gpu-config";
 *   export default gpuConfig("nvidia");
 */
import fs from "node:fs";
import path from "node:path";

import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";

/**
 * Detect the correct DISPLAY. Falls back to $DISPLAY, then probes
 * /tmp/.X11-unix/ for the highest-numbered socket.
 */
function detectDisplay(): string {
  const envDisplay = process.env.DISPLAY;
  // Quick sanity check: does the X socket exist for the env DISPLAY?
  if (envDisplay) {
    const num = envDisplay.replace(/^:/, "");
    if (fs.existsSync(`/tmp/.X11-unix/X${num}`)) return envDisplay;
  }
  // Probe for the highest-numbered X socket
  try {
    const sockets = fs.readdirSync("/tmp/.X11-unix")
      .filter(f => /^X\d+$/.test(f))
      .map(f => parseInt(f.slice(1), 10))
      .sort((a, b) => b - a);
    if (sockets.length > 0) return `:${sockets[0]}`;
  } catch { /* no X11 sockets */ }
  return envDisplay ?? ":0";
}

/** Chromium args shared by all GPU configs. */
const COMMON_ARGS = [
  "--no-sandbox",
  "--headless=new",
  "--use-angle=vulkan",
  "--enable-features=Vulkan",
  "--disable-vulkan-surface",
  "--enable-unsafe-webgpu",
];

/** Per-GPU overrides: extra Chromium args and env vars. */
const GPU_PROFILES = {
  nvidia: {
    args: ["--enable-dawn-features=vulkan_enable_f16_on_nvidia"],
    env: {},
  },
  intel: {
    args: ["--use-vulkan=native", "--force-gpu-mem-available-mb=4096"],
    env: {
      // Force Vulkan loader to only load the Intel mesa driver.
      VK_DRIVER_FILES: "/usr/share/vulkan/icd.d/intel_icd.json",
    },
  },
} as const;

export type GpuProfile = keyof typeof GPU_PROFILES;

/**
 * Build a complete vitest config for the given GPU.
 *
 * The config disables leak-checking (`setupFiles: []`) and uses a 300 s
 * timeout — GPU benchmarks are long-running and allocate freely.
 */
export function gpuConfig(gpu: GpuProfile) {
  const profile = GPU_PROFILES[gpu];

  return defineConfig({
    resolve: {
      alias: {
        "@hamk-uas/jax-js-nonconsuming/optax": path.resolve(
          __dirname,
          "..",
          "node_modules/@hamk-uas/jax-js-nonconsuming/packages/optax/dist/index.js",
        ),
        "@hamk-uas/jax-js-nonconsuming": path.resolve(
          __dirname,
          "..",
          "node_modules/@hamk-uas/jax-js-nonconsuming/dist/index.js",
        ),
      },
    },
    esbuild: {
      supported: { using: false },
    },
    server: {
      headers: {
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Opener-Policy": "same-origin",
      },
    },
    test: {
      watch: false,
      browser: {
        enabled: true,
        headless: true,
        screenshotFailures: false,
        provider: playwright({
          launchOptions: {
            args: [...COMMON_ARGS, ...profile.args],
            env: {
              DISPLAY: detectDisplay(),
              XAUTHORITY:
                process.env.XAUTHORITY ??
                `/run/user/${process.getuid?.() ?? 1000}/gdm/Xauthority`,
              ...profile.env,
            },
          },
        }),
        instances: [{ browser: "chromium" }],
      },
      testTimeout: 300_000,
      passWithNoTests: true,
      include: [
        "**/*.{test,spec}.?(c|m)[jt]s?(x)",
        "scripts/**/*.ts",
        "issues/**/*.ts",
      ],
      setupFiles: [],
    },
  });
}
