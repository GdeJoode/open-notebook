/**
 * Track UX — Phase UX.5 — lean 4-step creation flow E2E spec.
 *
 * AC mapping → docs/tracks/UX-pipeline-alignment/plan.md §UX.5:
 *   AC1  Exactly 4 steps (Input → Organize → Processing → Done).      → the stepper
 *   AC2  Live PipelineStatus tracker during processing (Embed before   → the live
 *        Extract), driven by `processing_stage`.                          tracker
 *   AC3  Transitions to Done on `processing_stage === 'complete'`.     → the Done copy
 *
 * Fully route-mocked (no live backend), matching the project's E2E convention.
 * The create call is mocked to return a source id, and GET /sources/{id} is
 * mocked to advance the `processing_stage` spine across successive polls
 * (ingested → embedded → extracted → graphed → complete) so the live tracker
 * walks the spine and the flow lands on Done — no DB / worker required.
 *
 * NOTE: this spec runs against a *served* frontend (PLAYWRIGHT_BASE_URL, default
 * http://localhost:8502). It does not stand up its own web server. If no built
 * frontend is being served the whole file is skipped at runtime rather than
 * reporting a false pass.
 */

import { expect, test, type Route, type Page } from '@playwright/test'

import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const SOURCE_ID = 'source:uxfive'

// The stage spine the mocked source walks across successive GET /sources/{id}
// polls. The last value (`complete`) sticks so the flow settles on Done.
const STAGE_SEQUENCE = [
  'ingested',
  'embedded',
  'extracted',
  'graphed',
  'complete',
] as const

function json(route: Route, body: unknown, status = 200): Promise<void> {
  return route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

function sourceBody(stage: string) {
  return {
    id: SOURCE_ID,
    title: 'A UX.5 source',
    topics: [],
    asset: null,
    embedded: stage !== 'ingested',
    embedded_chunks: stage === 'ingested' ? 0 : 12,
    insights_count: 0,
    entity_count: ['extracted', 'graphed', 'complete'].includes(stage) ? 7 : 0,
    relation_count: ['graphed', 'complete'].includes(stage) ? 3 : 0,
    created: '2026-07-01T00:00:00Z',
    updated: '2026-07-01T00:00:00Z',
    full_text: 'body',
    processing_stage: stage,
  }
}

async function mockCreateFlow(page: Page): Promise<void> {
  await mockDashboardChrome(page)
  await page.route('**/api/settings', (route) =>
    json(route, { default_embedding_option: 'ask' }),
  )
  await page.route('**/api/notebooks*', (route) => json(route, []))

  // Create returns the source id; the poll then advances the stage.
  await page.route('**/api/sources', (route) => {
    if (route.request().method() === 'POST') {
      return json(route, sourceBody('ingested'))
    }
    return json(route, [])
  })

  // Job axis (spinner only) + logs (SSE — fulfilled empty so the console mounts).
  // Registered before the generic GET so the more-specific globs win. The id
  // keeps its literal colon: the browser does not percent-encode it in the path.
  await page.route(`**/api/sources/${SOURCE_ID}/status`, (route) =>
    json(route, { status: 'running' }),
  )
  await page.route(`**/api/sources/${SOURCE_ID}/logs*`, (route) =>
    route.fulfill({ status: 200, contentType: 'text/event-stream', body: '' }),
  )

  // GET /sources/{id} advances the spine one step per poll (clamped to complete).
  let pollCount = 0
  await page.route(`**/api/sources/${SOURCE_ID}`, (route) => {
    const stage = STAGE_SEQUENCE[Math.min(pollCount, STAGE_SEQUENCE.length - 1)]
    pollCount += 1
    return json(route, sourceBody(stage))
  })
}

test.describe('Track UX.5 — lean creation flow', () => {
  test('create → live tracker advances the spine → Done', async ({ page }) => {
    await mockCreateFlow(page)

    await gotoAndWait(page, '/sources/new')
    if (page.url().includes('/login')) {
      test.skip(true, 'auth-gated stack — covered by the auth-disabled run')
    }

    // AC1 — exactly the 4 lean steps, no mandatory Config step.
    for (const label of ['Input', 'Organize', 'Processing', 'Done']) {
      await expect(page.getByRole('button', { name: label, exact: true })).toBeVisible()
    }
    await expect(page.getByRole('button', { name: 'Config', exact: true })).toHaveCount(0)

    // Input — text source.
    await page.getByRole('tab', { name: /Text/i }).click()
    await page.getByPlaceholder('Give your source a descriptive title').fill('A UX.5 source')
    await page.getByPlaceholder('Paste or type your content here...').fill('Body content')

    // Advance to Organize, then submit. (`Next` is exact so it doesn't match the
    // Next.js dev-tools button present under `next dev`.)
    await page.getByRole('button', { name: 'Next', exact: true }).click()
    await page.getByRole('button', { name: 'Submit', exact: true }).click()

    // AC2 — the live tracker renders the spine with Embed before Extract.
    const live = page.locator('[data-variant="live"]')
    await expect(live).toBeVisible()
    const embedNode = live.locator('[data-node-key="embed"]')
    const extractNode = live.locator('[data-node-key="extract"]')
    await expect(embedNode).toBeVisible()
    await expect(extractNode).toBeVisible()
    const embedBox = await embedNode.boundingBox()
    const extractBox = await extractNode.boundingBox()
    expect(embedBox && extractBox && embedBox.x < extractBox.x).toBeTruthy()

    // AC3 — the spine walks to complete and the flow lands on Done.
    await expect(page.getByText('Source Created Successfully')).toBeVisible({
      timeout: 20_000,
    })
  })
})
