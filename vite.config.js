
import { defineConfig } from 'vite';
import { resolve } from 'path';
import dts from 'vite-plugin-dts';

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'DlmJs',
      fileName: (format) => `dlm-js.${format}.js`,
      formats: ['es', 'cjs'],
    },
    rollupOptions: {
      // Externalize deps you don’t want bundled
      external: ['@jax-js/jax'],
      output: {},
    },
  },
  plugins: [
    dts({ insertTypesEntry: true }),
  ],
});