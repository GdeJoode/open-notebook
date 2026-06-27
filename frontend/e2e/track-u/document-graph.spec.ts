/**
 * Track U — Phase U.4 — Document-graph view E2E spec.
 *
 * AC mapping → docs/tracks/U-document-graph/plan.md §U.4:
 *
 *   AC1  Documents as nodes linked via shared entities; entity layer            → the
 *        summary reports the documents + concepts; the link list names the shared
 *        concept; toggling "Show shared concepts" flips bipartite↔collapsed.
 *   AC2  Per-edge basis (shared concept) surfaced.                              → the
 *        link-list alternative shows "… via Regio Deal".
 *   AC3  Loading / empty / error states handled.                               → empty
 *        (0 edges) renders the friendly regenerate hint; error renders retry.
 *   AC4  a11y: accessible summary, keyboard-reachable controls, not colour-only → the
 *        aria-live summary + the legend's ◆ glyph + the link list cover this.
 *
 * Fully route-mocked (no live backend), matching the project's E2E convention.
 * The U.2 backend exists in merged code, but this spec mocks the response so it
 * runs against a built+served frontend with no DB.
 */

import { expect, test, type Route } from '@playwright/test'

import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const DOC_A = 'source:hjrb1cjvzv86oyaojblj'
const DOC_B = 'source:dndibxmjveoxk7tfqfsl'
const PAPER = 'source:bc6xax5piw8cntg5nomc'
const REGIO_DEAL = 'entity:regio-deal'
const BREDE_WELVAART = 'entity:brede-welvaart'

const SOURCES = [
  { id: DOC_A, title: 'Convenant Regio Deal Noord-Holland Noord', embedded: true, embedded_chunks: 0, insights_count: 0, entity_count: 151, relation_count: 0, created: '', updated: '', asset: null },
  { id: DOC_B, title: 'Convenant Regio Deal Midden-Limburg', embedded: true, embedded_chunks: 0, insights_count: 0, entity_count: 110, relation_count: 0, created: '', updated: '', asset: null },
  { id: PAPER, title: 'Cohesion Policy & Institutional Quality', embedded: true, embedded_chunks: 0, insights_count: 0, entity_count: 0, relation_count: 0, created: '', updated: '', asset: null },
]

const DOCUMENT_GRAPH = {
  count: 4,
  edges: [
    { id: 'mentions:1', source: DOC_A, target: REGIO_DEAL, weight: 1.336, concept_name: 'Regio Deal', concept_type: 'programme', document_frequency: 4 },
    { id: 'mentions:2', source: DOC_B, target: REGIO_DEAL, weight: 1.336, concept_name: 'Regio Deal', concept_type: 'programme', document_frequency: 4 },
    { id: 'mentions:3', source: DOC_A, target: BREDE_WELVAART, weight: 0.2, concept_name: 'brede welvaart', concept_type: 'topic', document_frequency: 4 },
    { id: 'mentions:4', source: DOC_B, target: BREDE_WELVAART, weight: 0.2, concept_name: 'brede welvaart', concept_type: 'topic', document_frequency: 4 },
  ],
}

function json(route: Route, body: unknown, status = 200): Promise<void> {
  return route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

async function mockKgChrome(page: import('@playwright/test').Page) {
  await mockDashboardChrome(page)
  // The KG page's table tab fires these on mount; stub them so the page boots.
  await page.route('**/api/knowledge-graph/entity-types', (route) => json(route, []))
  await page.route('**/api/knowledge-graph/entities?*', (route) =>
    json(route, { items: [], total: 0, limit: 50, offset: 0 }),
  )
  await page.route('**/api/sources?*', (route) => json(route, SOURCES))
  await page.route('**/api/sources', (route) => json(route, SOURCES))
}

test.describe('Track U.4 — document graph view', () => {
  test('renders the document↔concept graph and toggles the entity layer', async ({
    page,
  }) => {
    await mockKgChrome(page)
    await page.route('**/api/knowledge-graph/document-graph*', (route) =>
      json(route, DOCUMENT_GRAPH),
    )

    await gotoAndWait(page, '/knowledge-graph')
    if (page.url().includes('/login')) {
      test.skip(true, 'auth-gated stack — covered by the auth-disabled run')
    }

    await page.getByRole('tab', { name: /Document Graph/ }).click()

    // AC1/AC4: the accessible summary reports the documents + shared concepts.
    // All three sources are document nodes (the two convenanten + the isolated
    // paper, injected from the source list so it isn't dropped).
    const summary = page.getByTestId('document-graph-summary')
    await expect(summary).toBeVisible()
    await expect(summary).toContainText('3 documents')
    await expect(summary).toContainText('2 shared concepts')
    // AC1: the isolated paper still appears (callout), not dropped.
    await expect(summary).toContainText('1 isolated document')
    await expect(summary).toContainText('Cohesion Policy & Institutional Quality')

    // AC2: the per-link "why" (shared concept) is surfaced in the link list.
    const links = page.getByTestId('document-graph-link-list')
    await expect(links).toContainText('Regio Deal')

    // AC1: hiding the entity layer collapses to a single doc↔doc link.
    const toggle = page.getByTestId('entity-layer-toggle')
    await toggle.click()
    await expect(summary).toContainText('0 shared concepts')
    await expect(summary).toContainText('directly via the concepts they share')
    await expect(links).toContainText('via Regio Deal, brede welvaart')
  })

  test('the named-only preset re-queries with a higher min_weight (AC1)', async ({
    page,
  }) => {
    const urls: string[] = []
    await mockKgChrome(page)
    await page.route('**/api/knowledge-graph/document-graph*', (route) => {
      urls.push(route.request().url())
      return json(route, DOCUMENT_GRAPH)
    })

    await gotoAndWait(page, '/knowledge-graph')
    if (page.url().includes('/login')) {
      test.skip(true, 'auth-gated stack — covered by the auth-disabled run')
    }
    await page.getByRole('tab', { name: /Document Graph/ }).click()
    await expect(page.getByTestId('document-graph-summary')).toBeVisible()

    await page.getByTestId('named-only-preset').click()
    await expect
      .poll(() => urls.some((u) => /min_weight=0\.3/.test(u)))
      .toBe(true)
  })

  test('empty projection shows the regenerate hint, not a crash (AC3)', async ({
    page,
  }) => {
    await mockKgChrome(page)
    await page.route('**/api/knowledge-graph/document-graph*', (route) =>
      json(route, { edges: [], count: 0 }),
    )

    await gotoAndWait(page, '/knowledge-graph')
    if (page.url().includes('/login')) {
      test.skip(true, 'auth-gated stack — covered by the auth-disabled run')
    }
    await page.getByRole('tab', { name: /Document Graph/ }).click()

    const empty = page.getByTestId('document-graph-empty')
    await expect(empty).toBeVisible()
    await expect(empty).toContainText('No shared-concept links yet')
    await expect(empty).toContainText('regenerate')
  })

  test('an error from the projection shows a retry alert (AC3)', async ({
    page,
  }) => {
    await mockKgChrome(page)
    await page.route('**/api/knowledge-graph/document-graph*', (route) =>
      json(route, { detail: 'boom' }, 500),
    )

    await gotoAndWait(page, '/knowledge-graph')
    if (page.url().includes('/login')) {
      test.skip(true, 'auth-gated stack — covered by the auth-disabled run')
    }
    await page.getByRole('tab', { name: /Document Graph/ }).click()

    await expect(page.getByTestId('document-graph-error')).toBeVisible()
    await expect(
      page.getByRole('button', { name: /Retry/ }),
    ).toBeVisible()
  })
})
