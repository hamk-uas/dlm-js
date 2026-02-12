import jaxJs from "@hamk-uas/eslint-plugin-jax-js";
import tsParser from "@typescript-eslint/parser";

const recommended = jaxJs.configs.recommended;

export default [
  {
    ignores: ["dist/**", "node_modules/**"],
  },
  {
    files: ["**/*.ts"],
    ...recommended,
    languageOptions: {
      ...(recommended.languageOptions ?? {}),
      parser: tsParser,
    },
  },
];
