/**
 * WB.4 AC: every dashboard page uses `PageHeader` / `PageBody`. The pages are
 * read from `app/(dashboard)/**\/page.tsx`, so a new page is held to it without
 * being listed. The rendered check (one header, one h1, inside the body) is
 * `e2e/track-wb/layout.spec.ts`.
 */
import { readdirSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

const SRC = path.resolve(__dirname, '../../..')
const DASHBOARD = path.join(SRC, 'app', '(dashboard)')

function findPages(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) return findPages(full)
    return entry.name === 'page.tsx' ? [full] : []
  })
}

/** Components a page hands its header to; each renders `PageHeader` itself. */
const HEADER_WRAPPERS: Record<string, string> = {
  NotebookHeader: 'app/(dashboard)/notebooks/components/NotebookHeader.tsx',
  SourceDetailContent: 'components/source/SourceDetailContent.tsx',
}

const read = (file: string) =>
  readFileSync(file, 'utf8')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:])\/\/.*$/gm, '$1')

const uses = (src: string, component: string) =>
  new RegExp(`<${component}[\\s/>]`).test(src) &&
  new RegExp(`import \\{[^}]*\\b${component}\\b[^}]*\\} from '[^']+'`).test(src)

const pages = findPages(DASHBOARD)
const rel = (file: string) => path.relative(SRC, file).split(path.sep).join('/')

describe('dashboard pages', () => {
  it('are found', () => {
    expect(pages.map(rel)).toContain('app/(dashboard)/page.tsx')
    expect(pages.map(rel)).toContain('app/(dashboard)/sources/[id]/page.tsx')
  })

  describe.each(pages.map((p) => [rel(p), p]))('%s', (_name, file) => {
    const src = read(file)

    it('sits in PageBody', () => {
      expect(uses(src, 'PageBody')).toBe(true)
    })

    it('has a PageHeader, its own or through a header component', () => {
      const through = Object.keys(HEADER_WRAPPERS).filter((w) => uses(src, w))
      expect(uses(src, 'PageHeader') || through.length > 0).toBe(true)
    })

    it('writes no h1 of its own', () => {
      expect(src).not.toMatch(/<h1[\s>]/)
    })
  })

  describe.each(Object.entries(HEADER_WRAPPERS))('header component %s', (_name, file) => {
    const src = read(path.join(SRC, file))

    // No h1 check here: SourceDetailContent renders the source's own markdown.
    it('renders PageHeader', () => {
      expect(uses(src, 'PageHeader')).toBe(true)
    })
  })
})
