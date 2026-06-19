/**
 * Track I — Phase I.F — Document structure graph E2E spec.
 *
 * AC mapping → docs/tracks/I-docling-studio/plan.md §I.F:
 *
 *   AC5  UI: clicking a node highlights the bbox in the middle pane.
 *        → the Structure tab renders the Sigma graph from a mocked
 *          /structure-graph payload; the graph canvas mounts and the node-count
 *          stat is visible, all without a console error. Node-level click →
 *          bbox highlight is asserted at the unit/wiring level (StructureViewer
 *          forwards onSelectNode to setActiveChunkId, which the PdfChunkViewer
 *          consumes); a deterministic per-node click on the WebGL canvas is not
 *          reliable headless, so the visual highlight is verified manually and
 *          documented as live-deferred in the phase self-review.
 *   AC7  page-limit guard — exercised at the API-test level (422 above 500);
 *        the frontend always requests the default (100).
 *
 * Fully route-mocked: no live backend. Mirrors the I.D-4 result-tabs fixtures.
 *
 * NOTE: validated via `playwright test --list` in the sandbox (no browser run);
 * a full headed run is deferred to an environment with the dev server up.
 */

import { expect, test, type Page } from '@playwright/test'
import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const SOURCE_ID = 'src_track_i_f_fixture'

const PNG_1x1 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='

function makeChunks(n: number) {
  return Array.from({ length: n }, (_, i) => ({
    id: `chunk:${i}`,
    text: `Chunk number ${i} body text for the structure-graph fixture.`,
    order: i,
    physical_page: 0,
    printed_page: null,
    chapter: null,
    paragraph_number: null,
    element_type: i % 4 === 0 ? 'section_header' : 'paragraph',
    positions: [[0, 0.1, 0.9, 0.1, 0.2]],
    metadata: { section_path: ['Chapter 1', 'Section 1.1'], section_level: 1 },
    is_content: true,
  }))
}

/** A small but well-formed structure-graph payload (sections + leaves + edges). */
function makeStructureGraph() {
  const nodes = [
    {
      id: 'doc_node:s0',
      self_ref: '#/sections/0',
      element_type: 'section_header',
      text: 'Chapter 1',
      page: null,
      level: 0,
      sequence: 0,
      bbox: null,
      chunk_id: null,
    },
    {
      id: 'doc_node:l0',
      self_ref: '#/chunks/0',
      element_type: 'paragraph',
      text: 'First paragraph',
      page: 0,
      level: 1,
      sequence: 1,
      bbox: [0.1, 0.1, 0.9, 0.2],
      chunk_id: 'chunk:0',
    },
    {
      id: 'doc_node:l1',
      self_ref: '#/chunks/1',
      element_type: 'paragraph',
      text: 'Second paragraph',
      page: 0,
      level: 1,
      sequence: 2,
      bbox: [0.1, 0.3, 0.9, 0.4],
      chunk_id: 'chunk:1',
    },
  ]
  const edges = [
    { source: 'doc_node:s0', target: 'doc_node:l0', type: 'parent_of' },
    { source: 'doc_node:s0', target: 'doc_node:l1', type: 'parent_of' },
    { source: 'doc_node:l0', target: 'doc_node:l1', type: 'next_node' },
    { source: 'chunk:0', target: 'doc_node:l0', type: 'derived_from' },
    { source: 'chunk:1', target: 'doc_node:l1', type: 'derived_from' },
  ]
  return { nodes, edges, total_nodes: nodes.length, truncated: false }
}

async function mockInspect(page: Page): Promise<void> {
  const source = {
    id: SOURCE_ID,
    title: 'Track I.F fixture source',
    topics: [],
    asset: { file_path: '/tmp/fixture.pdf' },
    embedded: false,
    embedded_chunks: 0,
    insights_count: 0,
    entity_count: 0,
    relation_count: 0,
    created: '2026-06-01T00:00:00Z',
    updated: '2026-06-01T00:00:00Z',
    file_available: true,
    full_text: '# Chapter 1\n\nFirst paragraph',
    notebooks: [],
    metadata: { parser_engine_used: 'docling' },
  }

  await page.route(`**/api/sources/${encodeURIComponent(SOURCE_ID)}`, (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(source),
    })
  )

  await page.route(
    `**/api/sources/${encodeURIComponent(SOURCE_ID)}/chunks`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          chunks: makeChunks(8),
          total_chunks: 8,
          has_spatial_data: true,
        }),
      })
  )

  await page.route(
    `**/api/sources/${encodeURIComponent(SOURCE_ID)}/structure-graph*`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(makeStructureGraph()),
      })
  )

  await page.route(
    `**/api/sources/${encodeURIComponent(SOURCE_ID)}/page-count`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          page_count: 1,
          pages: [{ page_number: 1, width: 595, height: 842 }],
        }),
      })
  )

  await page.route(/\/api\/sources\/[^/]+\/page-preview.*/, (route) =>
    route.fulfill({
      status: 200,
      contentType: 'image/png',
      body: Buffer.from(PNG_1x1, 'base64'),
    })
  )

  await page.route(`**/api/insights*`, (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '[]' })
  )
  await page.route(`**/api/transformations*`, (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '[]' })
  )
  await page.route(`**/api/zotero/status`, (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ connected: false }),
    })
  )
}

const INSPECT_PATH = `/sources/${encodeURIComponent(SOURCE_ID)}/inspect`

function watchConsole(page: Page): string[] {
  const errors: string[] = []
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(msg.text())
  })
  page.on('pageerror', (err) => errors.push(err.message))
  return errors
}

test.describe('I.F: document structure graph', () => {
  test.setTimeout(60_000)
  test.use({ navigationTimeout: 45_000 })

  test('structure tab renders the graph without console errors (AC5)', async ({
    page,
  }) => {
    const errors = watchConsole(page)

    await mockDashboardChrome(page)
    await mockInspect(page)
    await gotoAndWait(page, INSPECT_PATH)

    await page.getByRole('tab', { name: 'Structure' }).click()
    await expect(page.getByRole('tab', { name: 'Structure' })).toHaveAttribute(
      'aria-selected',
      'true'
    )

    // The Sigma graph mounts a canvas + the node-count stat.
    await expect(page.getByText('3 nodes')).toBeVisible({ timeout: 15_000 })
    await expect(page.locator('canvas').first()).toBeVisible()

    expect(errors).toEqual([])
  })

  test('empty structure graph shows the empty state', async ({ page }) => {
    await mockDashboardChrome(page)
    await mockInspect(page)
    // Override with an empty payload.
    await page.route(
      `**/api/sources/${encodeURIComponent(SOURCE_ID)}/structure-graph*`,
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            nodes: [],
            edges: [],
            total_nodes: 0,
            truncated: false,
          }),
        })
    )
    await gotoAndWait(page, INSPECT_PATH)

    await page.getByRole('tab', { name: 'Structure' }).click()
    await expect(page.getByText('No structure graph')).toBeVisible({
      timeout: 15_000,
    })
  })
})
