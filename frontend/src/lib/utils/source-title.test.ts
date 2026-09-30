import { readdirSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

import { displaySourceTitle } from './source-title'

describe('displaySourceTitle', () => {
  it('drops a leading 12-hex upload prefix and nothing else', () => {
    expect(displaySourceTitle('410a65bba3f8__Convenant Regio Deal Waterwegregio')).toBe(
      'Convenant Regio Deal Waterwegregio',
    )
    expect(displaySourceTitle('Convenant Regio Deal')).toBe('Convenant Regio Deal')
    // Same cases as the server's twin (apps/app-main/tests/test_source_title.py).
    expect(displaySourceTitle('410a65bba3f__Short')).toBe('410a65bba3f__Short')
    expect(displaySourceTitle('410A65BBA3F8__Upper')).toBe('410A65BBA3F8__Upper')
    expect(displaySourceTitle('Notes on 410a65bba3f8__x')).toBe('Notes on 410a65bba3f8__x')
    expect(displaySourceTitle('410a65bba3f8__')).toBe('')
    expect(displaySourceTitle('')).toBe('')
    expect(displaySourceTitle(null)).toBe('')
    expect(displaySourceTitle(undefined)).toBe('')
  })
})

/**
 * AC (WB.3): no displayed source title starts with the upload prefix.
 *
 * Every line of `src/` (tests excepted) that reads a `.title`, `.source_title`
 * or `sourceTitle`, and every attention item label (a source item's label is
 * its title), must either go through `displaySourceTitle(` on that line or be
 * listed below as something that is not a source title shown to the owner. A
 * new line that is neither fails here, so the set is derived from the code, not
 * sampled.
 */
const SRC = path.resolve(__dirname, '../..')

type Exempt = { file: string; line: string; why: string }

const NOT_A_SHOWN_SOURCE_TITLE: Exempt[] = [
  // Not sources.
  { file: 'app/(dashboard)/page.tsx', line: '{group.title}', why: 'attention group heading' },
  { file: 'app/(dashboard)/notebooks/components/NoteEditorDialog.tsx', line: 'source?.title', why: 'a note (its `source` prop is the note)' },
  { file: 'app/(dashboard)/notebooks/components/NoteEditorDialog.tsx', line: 'title: data.title || undefined,', why: 'note form payload' },
  { file: 'app/(dashboard)/notebooks/components/NotesColumn.tsx', line: 'note.title', why: 'a note' },
  { file: 'app/(dashboard)/transformations/components/TransformationCard.tsx', line: 'transformation.title', why: 'a transformation' },
  { file: 'app/(dashboard)/transformations/components/TransformationEditorDialog.tsx', line: 'title', why: 'transformation form' },
  { file: 'components/layout/AppSidebar.tsx', line: 'section.title', why: 'nav section' },
  { file: 'components/notebooks/audit/AuditWidget.tsx', line: 'f.title', why: 'audit finding sentence; the server strips the source title it names (source_title.py)' },
  { file: 'components/podcasts/GeneratePodcastDialog.tsx', line: 'note.title', why: 'a note' },
  { file: 'components/source/SessionManager.tsx', line: 'session.title', why: 'a chat session' },
  { file: 'components/source/SourceDetailContent.tsx', line: 'trans.title', why: 'a transformation' },
  { file: 'components/sources/steps/ProcessingStep.tsx', line: 'transformation.title', why: 'a transformation' },
  { file: 'components/sources/steps/SourceTypeStep.tsx', line: 'errors.title', why: 'form validation error' },
  { file: 'components/ui/checkbox-list.tsx', line: 'item.title', why: 'generic list; its callers pass notebooks and transformations' },
  { file: 'components/ui/wizard-container.tsx', line: 'step.title', why: 'wizard step' },
  // Source titles, but not shown here.
  { file: 'components/source/SourceDetailContent.tsx', line: 'title === source.title', why: 'compares the edit with the stored title' },
  { file: 'components/sources/AddSourceDialog.tsx', line: 'data.title', why: 'the title typed into the create form' },
  { file: 'components/sources/pipeline/CreateSourcePipeline.tsx', line: 'data.title', why: 'the title typed into the create form' },
  { file: 'components/sources/pipeline/CreateSourcePipeline.tsx', line: 'request.title = undefined', why: 'request payload' },
  { file: 'lib/api/sources.ts', line: 'data.title', why: 'request payload' },
  // Passed on raw, shown through displaySourceTitle where rendered.
  { file: 'components/sources/AddExistingSourceDialog.tsx', line: "title: r.title || 'Untitled',", why: 'rendered as {displaySourceTitle(source.title)} in the same file' },
  { file: 'components/sources/pipeline/CreateSourcePipeline.tsx', line: 'title: result.title || file.name', why: 'rendered as displaySourceTitle(entry.title) in the same file' },
  { file: 'components/sources/pipeline/CreateSourcePipeline.tsx', line: 'title: s.title,', why: 'CompletionTab renders displaySourceTitle(source.title)' },
  { file: 'components/sources/pipeline/CreateSourcePipeline.tsx', line: 'sourceTitle=', why: 'CompletionTab renders displaySourceTitle(sourceTitle)' },
  { file: 'components/sources/pipeline/tabs/CompletionTab.tsx', line: 'sourceTitle', why: 'prop declaration; rendered through displaySourceTitle' },
]

const TITLE_READ = [/\.(title|source_title)\b/, /\bsourceTitle\b/]
const ATTENTION_LABEL = /\bitem\.label\b/

function sourceFiles(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const full = path.join(dir, e.name)
    if (e.isDirectory()) return e.name === '__tests__' ? [] : sourceFiles(full)
    return /\.tsx?$/.test(e.name) && !/\.(test|spec)\.tsx?$/.test(e.name) ? [full] : []
  })
}

type Hit = { file: string; lineNo: number; text: string }

function titleReads(): Hit[] {
  const hits: Hit[] = []
  for (const full of sourceFiles(SRC)) {
    const file = path.relative(SRC, full).split(path.sep).join('/')
    const body = readFileSync(full, 'utf-8')
    const readsAttention = body.includes("from '@/lib/api/attention'")
    body.split('\n').forEach((text, i) => {
      if (
        TITLE_READ.some((re) => re.test(text)) ||
        (readsAttention && ATTENTION_LABEL.test(text))
      ) {
        hits.push({ file, lineNo: i + 1, text: text.trim() })
      }
    })
  }
  return hits
}

const exemptFor = (h: Hit) =>
  NOT_A_SHOWN_SOURCE_TITLE.find((e) => e.file === h.file && h.text.includes(e.line))

describe('every rendered source title goes through displaySourceTitle', () => {
  const hits = titleReads()

  it('finds the title reads it guards (the scan is not empty)', () => {
    // A few that must be there: the list, the detail header, a card, a Tray item.
    const files = new Set(hits.map((h) => h.file))
    for (const f of [
      'app/(dashboard)/sources/page.tsx',
      'components/source/SourceDetailContent.tsx',
      'components/sources/SourceCard.tsx',
      'app/(dashboard)/page.tsx',
      'components/notebooks/schema/CoverageStatsTable.tsx',
      'app/(dashboard)/knowledge-graph/components/DocumentGraphView.tsx',
      'lib/utils/triage.ts',
    ]) {
      expect(files, f).toContain(f)
    }
  })

  it('has no title read that neither strips nor is exempt', () => {
    const unclassified = hits
      .filter((h) => !h.text.includes('displaySourceTitle(') && !exemptFor(h))
      .map((h) => `${h.file}:${h.lineNo}: ${h.text}`)
    expect(unclassified).toEqual([])
  })

  it('has no exemption that matches nothing any more', () => {
    const stale = NOT_A_SHOWN_SOURCE_TITLE.filter(
      (e) => !hits.some((h) => h.file === e.file && h.text.includes(e.line)),
    )
    expect(stale).toEqual([])
  })
})
