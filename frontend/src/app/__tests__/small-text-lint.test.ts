/**
 * WB.4: the eslint rule that keeps font sizes at 13 px or more
 * (`eslint-rules/small-text.mjs`). The pattern is checked against the numbers
 * over a grid of values and units, then run through ESLint on class names in
 * JSX, `cn()` strings and template literals.
 */
/* eslint-disable no-restricted-syntax -- the probes below are what the rule must flag */
import { Linter } from 'eslint'
import tsParser from '@typescript-eslint/parser'
import { describe, expect, it } from 'vitest'

import { BELOW_13PX, SMALL_TEXT_SELECTORS } from '../../../eslint-rules/small-text.mjs'

const PX_PER: Record<string, number> = { px: 1, rem: 16, em: 16, pt: 4 / 3 }
const pattern = new RegExp(BELOW_13PX)

function spellings(value: number): string[] {
  const out = new Set<string>()
  for (const digits of [0, 1, 2, 3, 4]) out.add(value.toFixed(digits))
  return [...out].flatMap((s) => (s.startsWith('0.') ? [s, s.slice(1)] : [s]))
}

describe('the small-text pattern', () => {
  it('flags exactly the sizes below 13 px, in px, rem, em and pt', () => {
    const wrong: string[] = []
    const values = [
      ...Array.from({ length: 2001 }, (_, i) => i / 100), // 0.00 .. 20.00
      0.8, 0.81, 0.812, 0.8124, 0.8125, 0.8126, 0.813, 0.82, 9.7, 9.74, 9.749, 9.75, 9.76, 12.99, 13,
    ]
    for (const unit of Object.keys(PX_PER)) {
      for (const value of values) {
        for (const text of spellings(value)) {
          const px = Number(text) * PX_PER[unit]
          const cls = `text-[${text}${unit}]`
          if (pattern.test(cls) !== px < 13 - 1e-9) wrong.push(`${cls} (${px.toFixed(3)} px)`)
        }
      }
    }
    expect([...new Set(wrong)]).toEqual([])
  })

  it('leaves colours and variables alone', () => {
    for (const cls of ['text-[#10a0ff]', 'text-[var(--x)]', 'text-[color:red]', 'text-xs', 'ring-[3px]']) {
      expect(pattern.test(cls), cls).toBe(false)
    }
  })
})

describe('the eslint rule', () => {
  const lint = (code: string) =>
    new Linter({ configType: 'flat' }).verify(code, [{
      files: ['**/*.tsx'],
      languageOptions: { parser: tsParser, parserOptions: { ecmaFeatures: { jsx: true } } },
      rules: { 'no-restricted-syntax': ['error', ...SMALL_TEXT_SELECTORS] },
    }], 'probe.tsx')

  it('reports small sizes in JSX, strings and template literals', () => {
    const code = [
      'const n = 1',
      'export const a = <span className="text-[10px]" />',
      'export const b = <span className="sm:text-[0.8rem] px-1" />',
      "export const c = cn('p-1', 'text-[0.8em]')",
      'export const d = <span className={`p-1 text-[9pt] ${n}`} />',
    ].join('\n')
    expect(lint(code).map((m) => m.line)).toEqual([2, 3, 4, 5])
  })

  it('passes the type scale, 13 px and up, and colours', () => {
    const code = [
      'export const a = <span className="text-xs text-[13px] text-[0.8125rem] text-[10pt]" />',
      'export const b = <span className="text-[#10a0ff] focus-visible:ring-[3px]" />',
    ].join('\n')
    expect(lint(code)).toEqual([])
  })
})
