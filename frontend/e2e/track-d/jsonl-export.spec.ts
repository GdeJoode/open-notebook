/**
 * Track-D Phase D.2 acceptance:
 *
 * 1. Clicking "Export JSONL" on the notebook header opens the
 *    JsonlExportButton popover.
 * 2. The preview-counts widget renders "{N} entities, {M} relations"
 *    from the mocked GET /export-preview endpoint.
 * 3. Adjusting the min_confidence slider triggers a debounced refetch
 *    of /export-preview within ~500ms of the last change.
 * 4. Toggling the include_orphans switch flips its state.
 * 5. Clicking "Download" POSTs to /export-jsonl and the browser
 *    receives a download event with the suggested filename parsed from
 *    Content-Disposition.
 * 6. The captured POST payload carries the toggled filter knobs --
 *    proves the switch state round-trips into the request body.
 *
 * Fully mocked -- no live backend.
 *
 * Mirrors the D.1c obsidian-export spec structure; the only meaningful
 * differences are the popover wrapper (vs a dialog) and the lack of a
 * mode toggle.
 */

import { expect, test } from '@playwright/test'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const NOTEBOOK_ID = 'notebook:d2-fix'

const FIX_NOTEBOOK = {
  id: NOTEBOOK_ID,
  name: 'D.2 fixture notebook',
  description: 'Used by jsonl-export e2e spec',
  archived: false,
  created: '2026-06-01T00:00:00Z',
  updated: '2026-06-01T00:00:00Z',
  source_count: 0,
  note_count: 0,
}

const FIX_SETTINGS = {
  vault_path: '',
  vault_entities_folder: 'Entities',
  vault_sync_on_startup: false,
}

test.describe('D.2: JSONL export popover', () => {
  // First-route compile can take a while on cold dev runs -- same
  // mitigation as the D.1c spec.
  test.setTimeout(60_000)
  test.use({ navigationTimeout: 45_000 })

  // Match both raw and url-encoded forms of the notebook id.
  const NB_ID_RE = '(?:notebook:d2-fix|notebook%3Ad2-fix)'

  test.beforeEach(async ({ page }) => {
    await mockDashboardChrome(page)

    // Notebook detail.
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(FIX_NOTEBOOK),
        }),
    )

    // Settings (read by the Obsidian dialog -- harmless for JSONL but
    // the page mounts both surfaces).
    await page.route(/\/api\/settings(?:\?.*)?$/, (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(FIX_SETTINGS),
      }),
    )

    // Sources + notes -- workspace page fetches them; empty keeps the
    // page light.
    await page.route(
      new RegExp(`/api/sources(?:\\?.*notebook_id=${NB_ID_RE}.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        }),
    )
    await page.route(
      new RegExp(`/api/notes(?:\\?.*notebook_id=${NB_ID_RE}.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        }),
    )
  })

  test('preview counts render, slider triggers refetch, download fires with toggled filter', async ({
    page,
  }) => {
    // Pageerror watchdog (Nit 12 pattern from D.1c). Any uncaught page
    // error during the popover lifecycle is a regression.
    const pageErrors: string[] = []
    page.on('pageerror', (err) => pageErrors.push(err.message))

    // Track preview calls so we can verify the debounce + refetch. The
    // query string carries the filter -- we vary the response based on
    // min_confidence so the FE proves the new value actually reached
    // the backend.
    const previewCalls: string[] = []
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/export-preview(?:\\?.*)?$`),
      async (route) => {
        previewCalls.push(route.request().url())
        const url = new URL(route.request().url())
        const minConf = Number(url.searchParams.get('min_confidence') ?? 0.9)
        // When min_confidence drops below the default 0.9 the count
        // jumps up (the looser filter retains more entities).
        const body =
          minConf < 0.5
            ? { entity_count: 120, relation_count: 240 }
            : { entity_count: 18, relation_count: 32 }
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(body),
        })
      },
    )

    // Mock the JSONL endpoint. ``access-control-expose-headers`` lets
    // axios on the page side read ``content-disposition`` (same lesson
    // as the D.1c spec). We capture the POST payload so the assertion
    // below proves the boolean filter switch round-trips into the body.
    let capturedPayload:
      | {
          filter: {
            min_connections: number
            min_confidence: number
            include_orphans: boolean
            include_archived: boolean
          }
        }
      | undefined
    const TINY_ZIP_B64 = 'UEsFBgAAAAAAAAAAAAAAAAAAAAAAAA=='
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/export-jsonl$`),
      async (route) => {
        capturedPayload = route.request().postDataJSON() as typeof capturedPayload
        await route.fulfill({
          status: 200,
          contentType: 'application/zip',
          headers: {
            'content-disposition':
              'attachment; filename="d2-notebook-jsonl.zip"',
            'access-control-expose-headers':
              'Content-Disposition, Content-Type',
          },
          body: Buffer.from(TINY_ZIP_B64, 'base64'),
        })
      },
    )

    await page.goto(`/notebooks/${NOTEBOOK_ID}`)

    // The header button opens the popover.
    const openButton = page.getByTestId('open-jsonl-export-popover')
    await expect(openButton).toBeVisible()
    await openButton.click()

    const popover = page.getByTestId('jsonl-export-popover')
    await expect(popover).toBeVisible()

    // Preview counts render with the mocked numbers (min_confidence at
    // 0.9 default -> 18 entities / 32 relations).
    await expect(
      page.getByTestId('export-preview-loaded'),
    ).toBeVisible({ timeout: 5_000 })
    await expect(
      page.getByTestId('export-preview-entity-count'),
    ).toHaveText('18')
    await expect(
      page.getByTestId('export-preview-relation-count'),
    ).toHaveText('32')

    // Snapshot the call count BEFORE the slider interaction so the
    // delta below isolates the refetch caused by the slider.
    const callsBeforeSlider = previewCalls.length

    // Adjust the min_confidence slider. Radix slider has a focusable
    // thumb -- we focus it then press ArrowLeft to bump the value DOWN
    // from 0.9 toward 0 (which crosses our mock threshold of 0.5 and
    // flips the preview counts). 10 presses at step 0.05 = 0.5 decrement.
    const slider = page.getByTestId('jsonl-min-confidence-slider')
    await slider.focus()
    for (let i = 0; i < 10; i++) {
      await page.keyboard.press('ArrowLeft')
    }

    // Wait for the debounced refetch. The hook debounce is 300ms; the
    // entity count 120 is the proof the refetch happened with
    // min_confidence < 0.5.
    await expect(
      page.getByTestId('export-preview-entity-count'),
    ).toHaveText('120', { timeout: 2_500 })

    // Debounce-regression check: even though we pressed Arrow 10 times,
    // the refetch should have fired AT MOST twice (initial + once after
    // the debounced settle). The "one fetch per keystroke" failure mode
    // would show delta=10.
    const refetchDelta = previewCalls.length - callsBeforeSlider
    expect(refetchDelta).toBeLessThanOrEqual(2)
    expect(refetchDelta).toBeGreaterThanOrEqual(1)

    // Toggle the include_orphans switch BEFORE submit so the captured
    // POST payload proves it round-trips into the request body. The
    // default is ``false``; flipping to ``true`` is the only way to
    // verify the wiring -- a regression that stripped the field from
    // the payload would slip through if we relied on defaults.
    await page.getByTestId('jsonl-include-orphans-switch').click()

    // Click Download -> browser download fires.
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      page.getByTestId('jsonl-export-submit').click(),
    ])
    expect(download.suggestedFilename()).toBe('d2-notebook-jsonl.zip')

    // Captured payload assertions: toggled switch landed in the body
    // alongside the slider-driven min_confidence.
    expect(capturedPayload).toBeDefined()
    expect(capturedPayload!.filter.include_orphans).toBe(true)
    expect(capturedPayload!.filter.include_archived).toBe(false)
    // Min_confidence should be the post-slider value (~0.4 after 10
    // arrow-left presses). The exact value depends on Radix step
    // rounding; we assert it's *below* the 0.5 threshold rather than
    // pinning a brittle exact float.
    expect(capturedPayload!.filter.min_confidence).toBeLessThan(0.5)

    // Popover closes after the download completes.
    await expect(popover).toBeHidden({ timeout: 5_000 })

    // No uncaught page errors during the entire flow.
    expect(pageErrors).toEqual([])
  })
})
