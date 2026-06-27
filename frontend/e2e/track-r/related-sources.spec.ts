/**
 * Track R — Phase R.5 — "Related sources" UI surface E2E spec.
 *
 * AC mapping → docs/tracks/R-hybrid-search/plan.md §R.5:
 *
 *   AC1  Source detail shows a "Related sources" section driven by
 *        /related-hybrid; each result has title (linked) + fused score +
 *        per-result provenance ("why": shared entities and/or text-similarity).
 *        → open the Related tab, assert a result row renders a linked title,
 *          a fused score badge, and a why-explanation naming a shared entity
 *          and the text-similarity note.
 *   AC2  Loading / empty / error states handled gracefully (no crash, no raw
 *        error dump). → a second test mocks an empty /related-hybrid response
 *        (the no-embedding/no-related case) and asserts the friendly empty
 *        message renders, not an error.
 *   AC3  a11y: keyboard-navigable, aria-labelled, not colour-only. → the result
 *          title is a real link with an aria-label that includes the "why"; the
 *          section is aria-labelled; the why text is plain text (asserted on by
 *          text, not colour). Page produces no uncaught errors.
 *   AC4  This spec covers render + a result row's why-explanation.
 *
 * Fully route-mocked (no live backend), matching the project's E2E convention
 * (see track-i/inspect-workspace.spec.ts). The source-detail page also fetches
 * insights / transformations / chunks / chat sessions on mount; those are
 * stubbed with empty fixtures so the page boots into the Related tab.
 */

import { expect, test, type Page } from '@playwright/test'

import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const SOURCE_ID = 'src_track_r_fixture'
const RELATED_ID = 'src_related_one'
const RELATED_ID_2 = 'src_related_two'

const SOURCE = {
  id: SOURCE_ID,
  title: 'Track R.5 fixture source',
  topics: [],
  asset: { file_path: '/tmp/fixture.pdf' },
  embedded: true,
  embedded_chunks: 12,
  insights_count: 0,
  entity_count: 5,
  relation_count: 2,
  created: '2026-06-01T00:00:00Z',
  updated: '2026-06-01T00:00:00Z',
  file_available: true,
  full_text: 'Fixture content',
  notebooks: [],
  metadata: { parser_engine_used: 'docling' },
}

// A result that matched on BOTH signals: shared KG entities (the kg signal)
// AND dense text similarity (the dense signal). Drives the "why matched" copy.
const RELATED_BOTH = {
  id: RELATED_ID,
  title: 'Regional cohesion governance report',
  fused_score: 0.0488,
  dense: { present: true, score: 0.81, rank: 1, contribution: 0.0123 },
  kg: { present: true, score: 4.2, rank: 1, contribution: 0.0365 },
  kg_entities: [
    {
      entity_id: 'entity:bzk',
      name: 'BZK',
      type: 'organization',
      document_frequency: 3,
      weight: 2.1,
      via_relation: false,
    },
    {
      entity_id: 'entity:cohesion',
      name: 'Cohesion Policy',
      type: 'programme',
      document_frequency: 2,
      weight: 1.7,
      via_relation: false,
    },
  ],
}

// A dense-only result (no shared entities) — exercises the text-similarity-only
// branch of the why-explanation.
const RELATED_DENSE_ONLY = {
  id: RELATED_ID_2,
  title: 'Adjacent topic memo',
  fused_score: 0.0161,
  dense: { present: true, score: 0.62, rank: 2, contribution: 0.0161 },
  kg: { present: false, score: null, rank: null, contribution: 0 },
  kg_entities: [],
}

/**
 * Stub every endpoint the source-detail page touches on mount, plus the
 * R.5 hybrid endpoint with the supplied body. Specific routes are registered
 * before the catch-all `/sources/{id}` so they win.
 */
async function mockSourceDetail(
  page: Page,
  relatedBody: unknown
): Promise<void> {
  // R.5: the hybrid related-sources endpoint (with query params).
  await page.route(
    `**/api/sources/${encodeURIComponent(SOURCE_ID)}/related-hybrid*`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(relatedBody),
      })
  )

  await page.route(
    `**/api/sources/${encodeURIComponent(SOURCE_ID)}/chunks`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          chunks: [],
          total_chunks: 0,
          has_spatial_data: false,
        }),
      })
  )

  await page.route(
    `**/api/sources/${encodeURIComponent(SOURCE_ID)}/chat/sessions`,
    (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: '[]' })
  )

  // The bare source GET — registered after the more specific routes above.
  await page.route(
    `**/api/sources/${encodeURIComponent(SOURCE_ID)}`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(SOURCE),
      })
  )

  // Misc fixtures the detail page / chat shell may touch.
  await page.route('**/api/insights*', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '[]' })
  )
  await page.route('**/api/transformations*', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '[]' })
  )
  await page.route('**/api/zotero/status', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ connected: false }),
    })
  )
}

const DETAIL_PATH = `/sources/${encodeURIComponent(SOURCE_ID)}`

async function openRelatedTab(page: Page): Promise<void> {
  const relatedTab = page.getByRole('tab', { name: 'Related' })
  await expect(relatedTab).toBeVisible({ timeout: 15_000 })
  await relatedTab.click()
}

test.describe('R.5: Related sources on the source detail page', () => {
  test.setTimeout(60_000)
  test.use({ navigationTimeout: 45_000 })

  test('renders a ranked result with a linked title + why-matched provenance', async ({
    page,
  }) => {
    const pageErrors: string[] = []
    page.on('pageerror', (err) => pageErrors.push(err.message))

    await mockDashboardChrome(page)
    await mockSourceDetail(page, [RELATED_BOTH, RELATED_DENSE_ONLY])
    await gotoAndWait(page, DETAIL_PATH)
    if (page.url().includes('/login')) {
      test.skip(true, 'auth-gated stack — covered by the auth-disabled run')
    }

    await openRelatedTab(page)

    // AC1/AC3 — the section is present and aria-labelled.
    await expect(
      page.getByRole('region', { name: 'Related sources' })
    ).toBeVisible()

    // AC1 — a result row renders with a linked title pointing at the source.
    const row = page.getByTestId(`related-source-row-${RELATED_ID}`)
    await expect(row).toBeVisible()
    const title = row.getByTestId('related-source-title')
    await expect(title).toHaveText('Regional cohesion governance report')
    await expect(title).toHaveAttribute(
      'href',
      `/sources/${encodeURIComponent(RELATED_ID)}`
    )

    // AC1 — fused score is shown.
    await expect(row).toContainText('0.049')

    // AC1/AC4 — the "why matched" explanation names the shared entities AND the
    // text-similarity note (this result matched on both signals).
    const why = row.getByTestId('related-source-why')
    await expect(why).toContainText('Shares entities')
    await expect(why).toContainText('BZK')
    await expect(why).toContainText('Cohesion Policy')
    await expect(why).toContainText('similar text')

    // AC3 — the link's accessible name carries the "why" (screen-reader path,
    // not colour-only). Plain-text assertion, not a colour assertion.
    const ariaLabel = await title.getAttribute('aria-label')
    expect(ariaLabel).toContain('Regional cohesion governance report')
    expect(ariaLabel).toContain('BZK')

    // AC1 — the dense-only result explains itself as text-similarity only.
    const denseRow = page.getByTestId(`related-source-row-${RELATED_ID_2}`)
    await expect(denseRow).toBeVisible()
    await expect(denseRow.getByTestId('related-source-why')).toContainText(
      'Similar text content'
    )

    expect(pageErrors).toEqual([])
  })

  test('renders a friendly empty state for a source with no related sources', async ({
    page,
  }) => {
    const pageErrors: string[] = []
    page.on('pageerror', (err) => pageErrors.push(err.message))

    await mockDashboardChrome(page)
    // Empty array = the backend's no-embedding / no-shared-entities response.
    await mockSourceDetail(page, [])
    await gotoAndWait(page, DETAIL_PATH)
    if (page.url().includes('/login')) {
      test.skip(true, 'auth-gated stack — covered by the auth-disabled run')
    }

    await openRelatedTab(page)

    const region = page.getByRole('region', { name: 'Related sources' })
    await expect(region).toBeVisible()

    // AC2 — empty state is a friendly message, NOT an error alert / raw dump.
    // Scope the alert check to the section so a global toaster region (sonner)
    // doesn't count as a related-sources error.
    await expect(page.getByText('No related sources yet')).toBeVisible()
    await expect(region.getByRole('alert')).toHaveCount(0)
    await expect(
      page.getByTestId(`related-source-row-${RELATED_ID}`)
    ).toHaveCount(0)

    expect(pageErrors).toEqual([])
  })
})
