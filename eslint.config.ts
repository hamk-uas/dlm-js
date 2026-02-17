import jaxJs from "@hamk-uas/jax-js-nonconsuming/eslint-plugin";
import tsParser from "@typescript-eslint/parser";

export default [
  {
    ignores: ["dist/**", "node_modules/**", "tmp/**"],
  },
  {
    files: ["src/**/*.ts"],
    ...jaxJs.configs.recommended,
    languageOptions: {
      ...(jaxJs.configs.recommended.languageOptions ?? {}),
      parser: tsParser,
    },
  },
  {
    files: ["tests/**/*.ts"],
    plugins: { "jax-js": jaxJs },
    languageOptions: {
      parser: tsParser,
    },
    rules: {
      "jax-js/require-using": "off",
      "jax-js/no-use-after-dispose": "off",
      "jax-js/no-unnecessary-ref": "off",
      "jax-js/no-array-chain": "off",
    },
  },
];
