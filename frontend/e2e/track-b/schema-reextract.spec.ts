/**
 * B.3d acceptance criteria for the ReextractPromptBanner:
 *
 *   1. When the events endpoint returns at least one unread
 *      schema_changed event, the banner appears on the notebook
 *      workspace view with the affected-source count.
 *   2. Clicking [Re-extract all] POSTs every candidate source id to
 *      /schema/reextract and dismisses the banner.
 *   3. Clicking [Re-extract selected…] opens a modal multi-select
 *      that defaults all candidates on; submitting POSTs only the
 *      checked subset.
 *   4. Clicking [Later] marks each unread event read and dismisses
 *      the banner without posting any jobs.
 *
 * Fully mocked — no live backend.
 */

import { expect, test } from '@playwright/test'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const NOTEBOOK_ID = 'notebook:b3d-reextract'

const FIX_NOTEBOOK = {
  id: NOTEBOOK_ID,
  name: 'B.3d re-extract notebook',
  description: 'Used by schema-reextract e2e spec',
  archived: false,
  created: '2026-06-05T00:00:00Z',
  updated: '2026-06-05T00:00:00Z',
}

const FIX_EVENTS_UNREAD = [
  {
    id: 'notebook_event:evt-1',
    notebook_id: NOTEBOOK_ID,
    event_type: 'schema_changed',
    payload: { op: 'rename', old_name: 'Author', new_name: 'ResearchFellow' },
    created_at: '2026-06-05T12:00:00Z',
    read_at: null,
  },
]

const FIX_CANDIDATES = {
  notebook_id: NOTEBOOK_ID,
  source_ids: ['source:alpha', 'source:beta', 'source:gamma'],
  count: 3,
}

// Accept both encoded and raw notebook ids — the notebooksApi client
// uses raw, the schema client uses encodeURIComponent.
const NB_ID_RE = '(?:notebook:b3d-reextract|notebook%3Ab3d-reextract)'

test.describe('B.3d: schema-change re-extract prompt', () => {
  test.setTimeout(60_000)
  test.use({ navigationTimeout: 45_000 })

  test.beforeEach(async ({ page }) => {
    await mockDashboardChrome(page)

    // Notebook fetch
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(FIX_NOTEBOOK),
        }),
    )

    // Sources + notes are needed by the page chrome but we don't care
    // about their shapes for this spec.
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/sources(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        }),
    )
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/notes(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        }),
    )
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/chat-sessions(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        }),
    )

    // Candidate list — same for every test
    await page.route(
      new RegExp(
        `/api/notebooks/${NB_ID_RE}/schema/reextract_candidates(?:\\?.*)?$`,
      ),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(FIX_CANDIDATES),
        }),
    )
  })

  test('banner shows count when schema_changed events exist, then [Re-extract all] POSTs every candidate', async ({
    page,
  }) => {
    // Events endpoint serves the unread event on the first poll, then
    // empties out once the event has been marked read.
    const eventState = { unread: [...FIX_EVENTS_UNREAD] }
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/events(?:\\?.*)?$`),
      async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(eventState.unread),
        })
      },
    )

    let reextractPayload: unknown = null
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/schema/reextract$`),
      async (route) => {
        if (route.request().method() === 'POST') {
          reextractPayload = route.request().postDataJSON()
          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
              notebook_id: NOTEBOOK_ID,
              jobs_enqueued: 3,
              source_ids: FIX_CANDIDATES.source_ids,
              enqueued_source_ids: FIX_CANDIDATES.source_ids,
              skipped_source_ids: [],
            }),
          })
        } else {
          await route.fallback()
        }
      },
    )

    // Mark-read flushes the event from the unread list — the next
    // events poll then returns [] and the banner disappears.
    let markReadCalls = 0
    await page.route(
      new RegExp(
        `/api/notebooks/${NB_ID_RE}/events/[^/]+/mark_read$`,
      ),
      async (route) => {
        if (route.request().method() === 'POST') {
          markReadCalls += 1
          eventState.unread = []
          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ ok: true }),
          })
        } else {
          await route.fallback()
        }
      },
    )

    await page.goto(`/notebooks/${NOTEBOOK_ID}`)

    const banner = page.getByTestId('reextract-prompt-banner')
    await expect(banner).toBeVisible({ timeout: 10_000 })
    await expect(banner).toContainText('3 affected source')

    await page.getByTestId('reextract-all').click()

    await expect.poll(() => reextractPayload).toEqual({
      source_ids: FIX_CANDIDATES.source_ids,
    })
    // mark_read is called for the one unread event.
    await expect.poll(() => markReadCalls).toBeGreaterThanOrEqual(1)
    // Banner disappears after the next poll returns no unread events.
    await expect(banner).toBeHidden({ timeout: 35_000 })
  })

  test('[Re-extract selected…] opens dialog and POSTs only the checked subset', async ({
    page,
  }) => {
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/events(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(FIX_EVENTS_UNREAD),
        }),
    )

    let reextractPayload: unknown = null
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/schema/reextract$`),
      async (route) => {
        if (route.request().method() === 'POST') {
          reextractPayload = route.request().postDataJSON()
          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
              notebook_id: NOTEBOOK_ID,
              jobs_enqueued: 1,
              source_ids: ['source:alpha'],
              enqueued_source_ids: ['source:alpha'],
              skipped_source_ids: [],
            }),
          })
        } else {
          await route.fallback()
        }
      },
    )

    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/events/[^/]+/mark_read$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ ok: true }),
        }),
    )

    await page.goto(`/notebooks/${NOTEBOOK_ID}`)
    await expect(page.getByTestId('reextract-prompt-banner')).toBeVisible({
      timeout: 10_000,
    })

    await page.getByTestId('reextract-selected').click()

    const dialog = page.getByTestId('reextract-select-dialog')
    await expect(dialog).toBeVisible()

    // All checkboxes default to checked — uncheck beta and gamma so
    // only source:alpha remains.
    await page.getByTestId('reextract-pick-source:beta').click()
    await page.getByTestId('reextract-pick-source:gamma').click()

    await page.getByTestId('reextract-select-submit').click()

    await expect.poll(() => reextractPayload).toEqual({
      source_ids: ['source:alpha'],
    })
  })

  test('[Later] marks every unread event read and hides the banner without posting jobs', async ({
    page,
  }) => {
    const eventState = { unread: [...FIX_EVENTS_UNREAD] }
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/events(?:\\?.*)?$`),
      async (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(eventState.unread),
        }),
    )

    let reextractCalls = 0
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/schema/reextract$`),
      async (route) => {
        if (route.request().method() === 'POST') {
          reextractCalls += 1
        }
        await route.fallback()
      },
    )

    let markedReadIds: string[] = []
    await page.route(
      new RegExp(
        `/api/notebooks/${NB_ID_RE}/events/([^/]+)/mark_read$`,
      ),
      async (route) => {
        const match = route.request().url().match(/\/events\/([^/]+)\/mark_read/)
        if (match) markedReadIds.push(decodeURIComponent(match[1]))
        eventState.unread = []
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ ok: true }),
        })
      },
    )

    await page.goto(`/notebooks/${NOTEBOOK_ID}`)
    await expect(page.getByTestId('reextract-prompt-banner')).toBeVisible({
      timeout: 10_000,
    })

    await page.getByTestId('reextract-later').click()

    await expect.poll(() => markedReadIds.length).toBeGreaterThanOrEqual(1)
    // No re-extract submissions.
    expect(reextractCalls).toBe(0)
    await expect(page.getByTestId('reextract-prompt-banner')).toBeHidden({
      timeout: 35_000,
    })
  })

  test('banner is absent when there are no unread schema_changed events', async ({
    page,
  }) => {
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/events(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        }),
    )

    await page.goto(`/notebooks/${NOTEBOOK_ID}`)
    // Allow the page to settle.
    await page.waitForLoadState('networkidle')
    await expect(page.getByTestId('reextract-prompt-banner')).toHaveCount(0)
  })
})
