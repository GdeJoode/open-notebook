/**
 * WB.5: nothing in the app is fixed wider than a 390 px phone leaves (390 px
 * less the dialogs' 2rem margin). A dialog, popover or placeholder that wants
 * a fixed width clamps it to the viewport: `max-w-[min(560px,100vw-2rem)]`.
 * Derived from the source, not from a list: a new fixed width fails here.
 *
 * Review round 1: a named size (`max-w-2xl`) on a dialog's content does the
 * same. Unprefixed it replaces the base `max-w-[calc(100%-2rem)]`; with a
 * breakpoint it must fit that breakpoint's width less the 2rem margin.
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

/** Tailwind v4's `--container-*` sizes (the theme keeps them), in px. */
const NAMED: Record<string, number> = {
  '3xs': 256, '2xs': 288, xs: 320, sm: 384, md: 448, lg: 512, xl: 576, '2xl': 672,
  '3xl': 768, '4xl': 896, '5xl': 1024, '6xl': 1152, '7xl': 1280,
}
const BREAKPOINTS: Record<string, number> = { sm: 640, md: 768, lg: 1024, xl: 1280, '2xl': 1536 }

/** The class strings on dialog, alert dialog and sheet content in `text`. */
function contentClasses(text: string): string[] {
  const re = /<(?:Dialog|AlertDialog|Sheet)Content\b[^<]*?className=(?:"([^"]*)"|\{`([^`]*)`\}|\{cn\(([^)]*)\)\})/g
  return [...text.matchAll(re)].map((m) => m[1] ?? m[2] ?? m[3])
}

/** Named max-w sizes too wide for the viewport they apply at. */
function unclampedNamed(classes: string): string[] {
  return classes.split(/[\s'"]+/).filter((token) => {
    const m = /^(?:([\w-]+):)?max-w-(\w+)$/.exec(token)
    if (!m || !(m[2] in NAMED)) return false
    const room = m[1] ? (BREAKPOINTS[m[1]] ?? Infinity) - 32 : PHONE_ROOM
    return NAMED[m[2]] > room
  })
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

  it('the dialog pattern finds named sizes too wide where they apply', () => {
    const tsx = `<DialogContent className="max-w-2xl sm:max-w-3xl w-full">
      <AlertDialogContent className="sm:max-w-lg">
      <SheetContent side="right" className={cn('w-full sm:max-w-md', x)}>`
    expect(contentClasses(tsx).flatMap(unclampedNamed)).toEqual(['max-w-2xl', 'sm:max-w-3xl'])
    expect(unclampedNamed('max-w-xs sm:max-w-xl md:max-w-2xl sm:max-w-[min(42rem,100vw-2rem)]')).toEqual([])
  })

  it('no dialog, alert dialog or sheet takes a named width wider than its viewport', () => {
    const wide = sourceFiles(SRC).flatMap((file) =>
      contentClasses(readFileSync(file, 'utf-8'))
        .flatMap(unclampedNamed)
        .map((token) => `${path.relative(SRC, file)}: ${token}`),
    )
    expect(wide).toEqual([])
  })
})
