import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";
import { SMALL_TEXT_SELECTORS } from "./eslint-rules/small-text.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    files: ["src/**/*.{ts,tsx}"],
    rules: {
      // WB.4: no font size below 13 px (eslint-rules/small-text.mjs).
      "no-restricted-syntax": ["error", ...SMALL_TEXT_SELECTORS],
    },
  },
];

export default eslintConfig;
