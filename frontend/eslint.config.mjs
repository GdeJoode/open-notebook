import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

// WB.4 type scale: 13 px is the floor (text-xs). An arbitrary font size below
// it (`text-[10px]`, `text-[0.7rem]`) is refused; colours and other arbitrary
// `text-[...]` values (`text-[#f97316]`, `text-[var(--x)]`) are not sizes and
// pass. rem/em are read at 16 px: below 0.8125.
const BELOW_13PX = String.raw`text-\[(?:(?:\d|1[0-2])(?:\.\d+)?px|0?\.(?:[0-7]\d*|80\d*|81[01]\d*|812[0-4]\d*)r?em)\]`;
const SMALL_TEXT_MESSAGE =
  "Font sizes below 13 px are not used (WB.4 type scale): use text-xs (13 px) or larger.";

const eslintConfig = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    files: ["src/**/*.{ts,tsx}"],
    rules: {
      "no-restricted-syntax": [
        "error",
        { selector: `Literal[value=/${BELOW_13PX}/]`, message: SMALL_TEXT_MESSAGE },
        { selector: `TemplateElement[value.raw=/${BELOW_13PX}/]`, message: SMALL_TEXT_MESSAGE },
      ],
    },
  },
];

export default eslintConfig;
