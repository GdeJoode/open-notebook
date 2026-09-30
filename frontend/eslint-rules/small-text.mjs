// WB.4 type scale: 13 px is the floor (text-xs). An arbitrary Tailwind font
// size below it is refused. Colours and other arbitrary `text-[...]` values
// (`text-[#f97316]`, `text-[var(--x)]`) are not sizes and pass.
//
// Read at 16 px to the rem/em and 4/3 px to the pt, "below 13 px" is:
//   px  < 13      (0-12.x, .x)
//   rem/em < 0.8125
//   pt  < 9.75
// The pattern spells those bounds out digit by digit; small-text.test.ts
// checks it against the numbers over a grid of values.
const PX = String.raw`(?:(?:\d|1[0-2])(?:\.\d+)?|\.\d+)px`;
const REM = String.raw`(?:0|0?\.(?:[0-7]\d*|8(?:0\d*|1(?:[01]\d*|2(?:[0-4]\d*)?)?)?))r?em`;
const PT = String.raw`(?:[0-8](?:\.\d+)?|9(?:\.(?:[0-6]\d*|7(?:[0-4]\d*)?))?|\.\d+)pt`;

export const BELOW_13PX = String.raw`text-\[(?:${PX}|${REM}|${PT})\]`;

export const SMALL_TEXT_MESSAGE =
  "Font sizes below 13 px are not used (WB.4 type scale): use text-xs (13 px) or larger.";

/** `no-restricted-syntax` options: class names in strings and template parts. */
export const SMALL_TEXT_SELECTORS = [
  { selector: `Literal[value=/${BELOW_13PX}/]`, message: SMALL_TEXT_MESSAGE },
  { selector: `TemplateElement[value.raw=/${BELOW_13PX}/]`, message: SMALL_TEXT_MESSAGE },
];
