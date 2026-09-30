/**
 * WB.5: nothing in the app is fixed wider than a 390 px phone leaves (390 px
 * less the dialogs' 2rem margin). A dialog, popover or placeholder that wants
 * a fixed width clamps it to the viewport: `max-w-[min(560px,100vw-2rem)]`.
 * Derived from the source, not from a list: a new fixed width fails here.
 */
import { readdirSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

const SRC = path.resolve(__dirname, '../../..')
const PHONE_ROOM = 390 - 32

function sourceFiles(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) return entry.name === '__tests__' ? [] : sourceFiles(full)
    return /\.tsx?$/.test(entry.name) && !/\.test\.tsx?$/.test(entry.name) ? [full] : []
  })
}

/** Width classes with a bare pixel size (any variant prefix), with their px. */
function fixedWidths(text: string): { token: string; px: number }[] {
  const re = /(?<![\w-])((?:[\w-]+:)*(?:min-w|max-w|w)-\[(\d+)px\])(?![\w-])/g
  return [...text.matchAll(re)].map((m) => ({ token: m[1], px: Number(m[2]) }))
}

describe('fixed widths', () => {
  it('the pattern finds bare pixel widths and passes clamped ones', () => {
    expect(fixedWidths('a sm:max-w-[560px] w-[595px] min-w-[960px] w-[40px]').map((w) => w.token))
      .toEqual(['sm:max-w-[560px]', 'w-[595px]', 'min-w-[960px]', 'w-[40px]'])
    expect(fixedWidths('sm:max-w-[min(560px,100vw-2rem)] w-[min(90vw,720px)] max-w-[80%]')).toEqual([])
  })

  it('no class fixes a width wider than a phone leaves', () => {
    const wide = sourceFiles(SRC).flatMap((file) =>
      fixedWidths(readFileSync(file, 'utf-8'))
        .filter((w) => w.px > PHONE_ROOM)
        .map((w) => `${path.relative(SRC, file)}: ${w.token}`),
    )
    expect(wide).toEqual([])
  })
})
