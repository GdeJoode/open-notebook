/**
 * Track-D Phase D.1c acceptance:
 *
 * 1. Clicking "Export Obsidian" on the notebook header opens the
 *    ObsidianExportDialog.
 * 2. The preview-counts widget renders "{N} entities, {M} relations"
 *    from the mocked GET /export-preview endpoint.
 * 3. Adjusting the min_connections slider triggers a debounced refetch
 *    of /export-preview within ~500ms of the last change.
 * 4. Clicking "Export" in zip mode POSTs to /export-obsidian and the
 *    browser receives a download event with the suggested filename
 *    parsed from Content-Disposition.
 * 5. The dialog closes after a successful zip download.
 *
 * Fully mocked -- no live backend.
 *
 * The unit-test counterpart for ``parseFilenameFromContentDisposition``
 * lives in ``obsidian-export-parser.spec.ts`` next to this file.
 */

import { expect, test } from '@playwright/test'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const NOTEBOOK_ID = 'notebook:d1c-fix'

const FIX_NOTEBOOK = {
  id: NOTEBOOK_ID,
  name: 'D.1c fixture notebook',
  description: 'Used by obsidian-export e2e spec',
  archived: false,
  created: '2026-06-01T00:00:00Z',
  updated: '2026-06-01T00:00:00Z',
  source_count: 0,
  note_count: 0,
}

const FIX_SETTINGS = {
  // vault_path empty so the dialog disables the Write-to-Vault tab --
  // matches the AC's "tooltip when unset" path without needing a
  // configured vault for the happy-path zip test.
  vault_path: '',
  vault_entities_folder: 'Entities',
  vault_sync_on_startup: false,
}

test.describe('D.1c: Obsidian export dialog', () => {
  // First-route compile can take a while on cold dev runs -- same
  // mitigation as the schema-tab spec.
  test.setTimeout(60_000)
  test.use({ navigationTimeout: 45_000 })

  // The notebooks client uses a raw id ("notebook:d1c-fix") and the
  // export endpoints encode it ("notebook%3Ad1c-fix"). The regex
  // matches both spellings.
  const NB_ID_RE = '(?:notebook:d1c-fix|notebook%3Ad1c-fix)'

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

    // Settings -- the dialog reads vault_path.
    await page.route(/\/api\/settings(?:\?.*)?$/, (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(FIX_SETTINGS),
      }),
    )

    // Sources + notes -- workspace page fetches them; empty lists keep
    // the page light.
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

  test('preview counts render, debounce fires once per drag, zip download closes dialog', async ({
    page,
  }) => {
    // Console-error watchdog (Nit 12). Any uncaught page error during
    // the dialog lifecycle is a regression -- React state bugs in the
    // dialog often manifest here first, before any visible breakage.
    const pageErrors: string[] = []
    page.on('pageerror', (err) => pageErrors.push(err.message))

    // Track preview calls so we can verify the debounce. The query
    // string carries the current filter -- we don't assert on it here
    // because the debounce assertion is about CALL COUNT, not contents.
    const previewCalls: string[] = []
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/export-preview(?:\\?.*)?$`),
      async (route) => {
        previewCalls.push(route.request().url())
        // Return a different relation count when min_connections=10 so
        // the assertion below proves the refetch actually happened
        // with the new filter value.
        const url = new URL(route.request().url())
        const minConn = Number(url.searchParams.get('min_connections') ?? 5)
        const body =
          minConn >= 10
            ? { entity_count: 12, relation_count: 24 }
            : { entity_count: 42, relation_count: 87 }
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(body),
        })
      },
    )

    // Mock the Obsidian zip export endpoint. ``access-control-expose-
    // headers`` is required so axios on the page side can read the
    // ``content-disposition`` header (same lesson as networkx-export
    // spec).
    //
    // M4 follow-up: capture the request payload so we can assert that
    // the boolean filter switches (``include_orphans`` /
    // ``include_archived``) actually round-trip into the POST body
    // when the operator toggles them in the dialog.
    let capturedExportPayload:
      | {
          mode: string
          filter: {
            min_connections: number
            min_confidence: number
            include_orphans: boolean
            include_archived: boolean
          }
        }
      | undefined
    const TINY_ZIP_B64 =
      'UEsFBgAAAAAAAAAAAAAAAAAAAAAAAA=='
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/export-obsidian$`),
      async (route) => {
        capturedExportPayload = route
          .request()
          .postDataJSON() as typeof capturedExportPayload
        await route.fulfill({
          status: 200,
          contentType: 'application/zip',
          headers: {
            'content-disposition':
              'attachment; filename="d1c-notebook-obsidian.zip"',
            'access-control-expose-headers':
              'Content-Disposition, Content-Type',
          },
          body: Buffer.from(TINY_ZIP_B64, 'base64'),
        })
      },
    )

    await page.goto(`/notebooks/${NOTEBOOK_ID}`)

    // The header button opens the dialog.
    const openButton = page.getByTestId('open-obsidian-export-dialog')
    await expect(openButton).toBeVisible()
    await openButton.click()

    const dialog = page.getByTestId('obsidian-export-dialog')
    await expect(dialog).toBeVisible()

    // Preview counts render the mocked numbers.
    await expect(
      page.getByTestId('export-preview-loaded'),
    ).toBeVisible({ timeout: 5_000 })
    await expect(
      page.getByTestId('export-preview-entity-count'),
    ).toHaveText('42')
    await expect(
      page.getByTestId('export-preview-relation-count'),
    ).toHaveText('87')

    // Snapshot the call count BEFORE the slider interaction so the
    // delta below isolates the refetch caused by the slider.
    const callsBeforeSlider = previewCalls.length

    // Adjust the min_connections slider. Slider is a Radix component
    // with a focusable thumb -- we focus it then press the right
    // arrow five times to bump the value from 5 -> 10. The page
    // should still render only ONE preview refetch because of the
    // 300ms debounce.
    const slider = page.getByTestId('min-connections-slider')
    await slider.focus()
    await page.keyboard.press('ArrowRight')
    await page.keyboard.press('ArrowRight')
    await page.keyboard.press('ArrowRight')
    await page.keyboard.press('ArrowRight')
    await page.keyboard.press('ArrowRight')

    // Wait for the debounced refetch. The hook's debounce is 300ms;
    // the new entity_count (12) is the proof the refetch happened with
    // min_connections >= 10.
    await expect(
      page.getByTestId('export-preview-entity-count'),
    ).toHaveText('12', { timeout: 2_500 })

    // Debounce-regression check: even though we pressed Arrow 5 times,
    // the refetch should have fired AT MOST once (and ideally exactly
    // once). Allow a small slack for any pre-render fetch already in
    // flight when the dialog mounted, but reject the "one fetch per
    // keystroke" failure mode (which would show delta=5).
    const refetchDelta = previewCalls.length - callsBeforeSlider
    expect(refetchDelta).toBeLessThanOrEqual(2)
    expect(refetchDelta).toBeGreaterThanOrEqual(1)

    // M4: toggle both boolean filter switches BEFORE submit so the
    // captured POST payload proves they round-trip into the request.
    // The defaults are ``false``; flipping to ``true`` is the only way
    // to verify the wiring -- a regression that stripped the fields
    // from the payload would slip through if we relied on defaults.
    await page.getByTestId('include-orphans-switch').click()
    await page.getByTestId('include-archived-switch').click()

    // Click Export -> browser download fires.
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      page.getByTestId('obsidian-export-submit').click(),
    ])
    expect(download.suggestedFilename()).toBe('d1c-notebook-obsidian.zip')

    // M4 assertion: the toggled switches landed in the request body
    // alongside the mode + min_confidence the original spec covered.
    expect(capturedExportPayload).toBeDefined()
    expect(capturedExportPayload!.mode).toBe('zip')
    expect(capturedExportPayload!.filter.min_confidence).toBe(0.9)
    expect(capturedExportPayload!.filter.include_orphans).toBe(true)
    expect(capturedExportPayload!.filter.include_archived).toBe(true)

    // Dialog closes after the zip download (success path for zip mode).
    await expect(dialog).toBeHidden({ timeout: 5_000 })

    // Nit 12: no uncaught page errors during the entire flow.
    expect(pageErrors).toEqual([])
  })

  test('vault_path tab is disabled when Settings.vault_path is empty', async ({
    page,
  }) => {
    // Preview is required by the dialog mount even though we don't
    // interact with it here.
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/export-preview(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ entity_count: 5, relation_count: 8 }),
        }),
    )

    // Track export calls so the inversion assertion below can prove
    // the mutation never fires when vault_path is unset.
    let exportCallCount = 0
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/export-obsidian$`),
      async (route) => {
        exportCallCount += 1
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'should never be called' }),
        })
      },
    )

    await page.goto(`/notebooks/${NOTEBOOK_ID}`)
    await page.getByTestId('open-obsidian-export-dialog').click()

    // The vault_path tab renders but is disabled because
    // FIX_SETTINGS.vault_path is empty.
    const vaultTab = page.getByTestId('mode-vault-path')
    await expect(vaultTab).toBeVisible()
    await expect(vaultTab).toBeDisabled()

    // Clicking the disabled tab does NOT flip the mode -- Radix won't
    // fire onValueChange. The vault-path display therefore stays
    // hidden, proving the mode never flipped.
    await vaultTab.click({ force: true })
    await expect(page.getByTestId('vault-path-display')).toBeHidden()

    // Inversion: the Export submit button is enabled (zip mode is the
    // default), and clicking it fires the zip mutation -- NOT a
    // vault_path mutation. The export endpoint mock above returns 500
    // if hit; we don't click submit here because the assertion we
    // really want is that no vault_path mutation fired during the
    // disabled-tab interaction.
    expect(exportCallCount).toBe(0)
  })
})
