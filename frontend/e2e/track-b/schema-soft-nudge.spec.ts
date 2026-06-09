/**
 * B.3c acceptance criteria:
 *
 * 1. When an `extension_suggested` event exists for the notebook, the
 *    SchemaSoftNudge banner appears on the workspace view.
 * 2. Clicking `[Use as-is]` calls the mark_read API → banner hides.
 * 3. Clicking `[Don't show again]` calls dismiss_nudge → subsequent
 *    polls don't show the banner (because the schema flag now reports
 *    `soft_nudge_dismissed=true`).
 * 4. Toggling the review_required Switch on the Schema tab calls the
 *    review_required endpoint and persists across reloads.
 * 5. When the paused-extraction endpoint reports `paused_count > 0`,
 *    the ExtractionPausedBanner appears with a Resume button that
 *    calls /extraction/resume.
 *
 * All API calls are mocked — no live backend.
 */

import { expect, test } from '@playwright/test'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const NOTEBOOK_ID = 'notebook:b3c-fix'

const FIX_NOTEBOOK = {
  id: NOTEBOOK_ID,
  name: 'B.3c fixture notebook',
  description: 'Used by schema-soft-nudge e2e spec',
  archived: false,
  created: '2026-06-01T00:00:00Z',
  updated: '2026-06-01T00:00:00Z',
}

const FIX_SCHEMA_BASE = {
  notebook_id: NOTEBOOK_ID,
  base_ontology: 'scholarly',
  base_ontology_types: [
    {
      name: 'Article',
      description: 'A published scholarly article.',
      parent_type: null,
      properties: [{ name: 'title', data_type: 'string', required: true }],
    },
  ],
  accepted_extensions: [],
  pending_extensions: [
    {
      extension_id: 'ext-pen-1',
      type_name: 'Cohort',
      rationale: 'Three sources mention a cohort.',
      properties: [],
    },
  ],
  coverage_pct: 0.82,
  review_required: false,
  soft_nudge_dismissed: false,
}

const FIX_EVENT_EXTENSION_SUGGESTED = {
  id: 'notebook_event:e1',
  event_type: 'extension_suggested',
  message: 'Coverage 0.82 — 2 candidate extensions found',
  source_id: 'source:s1',
  created_at: '2026-06-01T10:00:00Z',
  read_at: null,
}

// Accepts either encoded (notebook%3Ab3c-fix) or raw colon spellings —
// same rationale as in schema-tab-view.spec.ts.
const NB_ID_ENCODED_RE = '(?:notebook:b3c-fix|notebook%3Ab3c-fix)'

test.describe('B.3c: schema soft-nudge + pause toggle', () => {
  test.setTimeout(60_000)
  test.use({ navigationTimeout: 45_000 })

  test('extension_suggested event shows the soft-nudge banner', async ({
    page,
  }) => {
    await mockDashboardChrome(page)

    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(FIX_NOTEBOOK),
        }),
    )

    // Workspace also fetches sources + notes; return empty arrays so
    // the page renders quickly without hitting the 3-column components'
    // loading states forever.
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}/sources(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        }),
    )
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}/notes(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        }),
    )

    // Schema with dismiss=false so the banner is not suppressed.
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}/schema(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(FIX_SCHEMA_BASE),
        }),
    )

    // Events endpoint returns one extension_suggested event.
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}/events(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([FIX_EVENT_EXTENSION_SUGGESTED]),
        }),
    )

    // No paused extraction.
    await page.route(
      new RegExp(
        `/api/notebooks/${NB_ID_ENCODED_RE}/extraction/paused(?:\\?.*)?$`,
      ),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            notebook_id: NOTEBOOK_ID,
            paused_count: 0,
            paused_source_ids: [],
          }),
        }),
    )

    await page.goto(`/notebooks/${NOTEBOOK_ID}`)

    const banner = page.getByTestId('schema-soft-nudge')
    await expect(banner).toBeVisible({ timeout: 10_000 })
    await expect(banner).toContainText('extension')
    await expect(banner).toHaveAttribute(
      'data-event-type',
      'extension_suggested',
    )
  })

  test('clicking Use as-is calls mark_read and banner disappears', async ({
    page,
  }) => {
    await mockDashboardChrome(page)

    // The events endpoint flips to an empty list after mark_read is
    // called — track via a flag on the test runner side.
    let markReadCalled = false

    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(FIX_NOTEBOOK),
        }),
    )
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}/sources(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        }),
    )
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}/notes(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        }),
    )
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}/schema(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(FIX_SCHEMA_BASE),
        }),
    )
    await page.route(
      new RegExp(
        `/api/notebooks/${NB_ID_ENCODED_RE}/extraction/paused(?:\\?.*)?$`,
      ),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            notebook_id: NOTEBOOK_ID,
            paused_count: 0,
            paused_source_ids: [],
          }),
        }),
    )

    // GET events — returns the event before mark_read, empty after.
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}/events(?:\\?.*)?$`),
      (route) => {
        const body = markReadCalled
          ? '[]'
          : JSON.stringify([FIX_EVENT_EXTENSION_SUGGESTED])
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body,
        })
      },
    )

    // POST mark_read — assertion-side flag + success echo.
    const markReadUrlRe = new RegExp(
      `/api/notebooks/${NB_ID_ENCODED_RE}/events/[^/]+/mark_read(?:\\?.*)?$`,
    )
    await page.route(markReadUrlRe, (route) => {
      markReadCalled = true
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          event_id: 'notebook_event:e1',
          success: true,
        }),
      })
    })

    await page.goto(`/notebooks/${NOTEBOOK_ID}`)

    const banner = page.getByTestId('schema-soft-nudge')
    await expect(banner).toBeVisible({ timeout: 10_000 })

    const useAsIs = page.getByTestId('schema-soft-nudge-use-as-is')
    await expect(useAsIs).toBeVisible()

    const [request] = await Promise.all([
      page.waitForRequest((req) => markReadUrlRe.test(req.url())),
      useAsIs.click(),
    ])
    expect(request.method()).toBe('POST')

    await expect(banner).toBeHidden({ timeout: 5_000 })
  })

  test("clicking Don't show again calls dismiss_nudge and prevents re-show", async ({
    page,
  }) => {
    await mockDashboardChrome(page)

    let dismissCalled = false

    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(FIX_NOTEBOOK),
        }),
    )
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}/sources(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        }),
    )
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}/notes(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        }),
    )

    // Schema flips soft_nudge_dismissed=true after dismiss.
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}/schema(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            ...FIX_SCHEMA_BASE,
            soft_nudge_dismissed: dismissCalled,
          }),
        }),
    )

    // Events endpoint keeps returning the event — the dismiss should
    // suppress the banner even though events still exist.
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}/events(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([FIX_EVENT_EXTENSION_SUGGESTED]),
        }),
    )

    await page.route(
      new RegExp(
        `/api/notebooks/${NB_ID_ENCODED_RE}/extraction/paused(?:\\?.*)?$`,
      ),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            notebook_id: NOTEBOOK_ID,
            paused_count: 0,
            paused_source_ids: [],
          }),
        }),
    )

    const dismissUrlRe = new RegExp(
      `/api/notebooks/${NB_ID_ENCODED_RE}/schema/dismiss_nudge(?:\\?.*)?$`,
    )
    await page.route(dismissUrlRe, (route) => {
      dismissCalled = true
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          notebook_id: NOTEBOOK_ID,
          soft_nudge_dismissed: true,
        }),
      })
    })

    await page.goto(`/notebooks/${NOTEBOOK_ID}`)
    const banner = page.getByTestId('schema-soft-nudge')
    await expect(banner).toBeVisible({ timeout: 10_000 })

    const dismissBtn = page.getByTestId('schema-soft-nudge-dismiss')
    const [request] = await Promise.all([
      page.waitForRequest((req) => dismissUrlRe.test(req.url())),
      dismissBtn.click(),
    ])
    expect(request.method()).toBe('POST')

    // Banner disappears even though events still exist (dismiss flag
    // is the suppression source-of-truth).
    await expect(banner).toBeHidden({ timeout: 5_000 })
  })

  test('toggling review_required calls the API endpoint', async ({
    page,
  }) => {
    await mockDashboardChrome(page)

    let currentReviewRequired = false

    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(FIX_NOTEBOOK),
        }),
    )
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}/schema(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            ...FIX_SCHEMA_BASE,
            review_required: currentReviewRequired,
          }),
        }),
    )
    await page.route(
      new RegExp(
        `/api/notebooks/${NB_ID_ENCODED_RE}/pass1_results(?:\\?.*)?$`,
      ),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        }),
    )

    const reviewReqUrlRe = new RegExp(
      `/api/notebooks/${NB_ID_ENCODED_RE}/schema/review_required(?:\\?.*)?$`,
    )
    await page.route(reviewReqUrlRe, async (route) => {
      const data = route.request().postDataJSON() as { enabled?: boolean }
      currentReviewRequired = !!data.enabled
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          notebook_id: NOTEBOOK_ID,
          review_required: currentReviewRequired,
        }),
      })
    })

    await page.goto(`/notebooks/${NOTEBOOK_ID}/schema`)

    const switchEl = page.getByTestId('review-required-switch')
    await expect(switchEl).toBeVisible({ timeout: 10_000 })
    await expect(switchEl).toHaveAttribute('aria-checked', 'false')

    const [request] = await Promise.all([
      page.waitForRequest((req) => reviewReqUrlRe.test(req.url())),
      switchEl.click(),
    ])
    expect(request.method()).toBe('POST')
    expect(request.postDataJSON()).toEqual({ enabled: true })
  })

  test('paused extraction shows banner; Resume calls /extraction/resume', async ({
    page,
  }) => {
    await mockDashboardChrome(page)

    let resumeCalled = false

    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(FIX_NOTEBOOK),
        }),
    )
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}/sources(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        }),
    )
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}/notes(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        }),
    )

    // Schema with dismiss=true so the soft-nudge banner doesn't compete.
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}/schema(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            ...FIX_SCHEMA_BASE,
            review_required: true,
            soft_nudge_dismissed: true,
          }),
        }),
    )
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}/events(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        }),
    )

    // Paused endpoint reports two paused sources before resume, zero after.
    await page.route(
      new RegExp(
        `/api/notebooks/${NB_ID_ENCODED_RE}/extraction/paused(?:\\?.*)?$`,
      ),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            notebook_id: NOTEBOOK_ID,
            paused_count: resumeCalled ? 0 : 2,
            paused_source_ids: resumeCalled
              ? []
              : ['source:s1', 'source:s2'],
          }),
        }),
    )

    const resumeUrlRe = new RegExp(
      `/api/notebooks/${NB_ID_ENCODED_RE}/extraction/resume(?:\\?.*)?$`,
    )
    await page.route(resumeUrlRe, (route) => {
      resumeCalled = true
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          notebook_id: NOTEBOOK_ID,
          resumed_count: 2,
          sentinel_added: true,
        }),
      })
    })

    await page.goto(`/notebooks/${NOTEBOOK_ID}`)

    const pausedBanner = page.getByTestId('extraction-paused-banner')
    await expect(pausedBanner).toBeVisible({ timeout: 10_000 })
    await expect(pausedBanner).toContainText('2 sources')

    const resumeBtn = page.getByTestId('extraction-paused-resume')
    const [request] = await Promise.all([
      page.waitForRequest((req) => resumeUrlRe.test(req.url())),
      resumeBtn.click(),
    ])
    expect(request.method()).toBe('POST')

    await expect(pausedBanner).toBeHidden({ timeout: 5_000 })
  })
})
