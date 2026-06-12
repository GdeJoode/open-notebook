/**
 * B.6 acceptance criteria for the cross-notebook merge UI:
 *
 *   1. The notebook card overflow menu offers "Merge into…".
 *   2. The dialog renders a multi-select of other (non-target) notebooks.
 *   3. Dry-run preview surfaces entity + relation counters.
 *   4. Confirm fires the same call without dry_run and closes the dialog
 *      on success.
 *
 * Fully mocked — no live backend. Mirrors the B.3b spec's helper usage
 * for consistency.
 */

import { expect, test } from '@playwright/test'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const TARGET_NB = {
  id: 'notebook:b6-target',
  name: 'B.6 Target Notebook',
  description: 'Receives the merged graph',
  archived: false,
  created: '2026-06-01T00:00:00Z',
  updated: '2026-06-01T00:00:00Z',
  source_count: 0,
  note_count: 0,
}

const SOURCE_NB_A = {
  id: 'notebook:b6-src-a',
  name: 'B.6 Source A',
  description: 'first source',
  archived: false,
  created: '2026-06-01T00:00:00Z',
  updated: '2026-06-01T00:00:00Z',
  source_count: 0,
  note_count: 0,
}

const SOURCE_NB_B = {
  id: 'notebook:b6-src-b',
  name: 'B.6 Source B',
  description: 'second source',
  archived: false,
  created: '2026-06-01T00:00:00Z',
  updated: '2026-06-01T00:00:00Z',
  source_count: 0,
  note_count: 0,
}

const NOTEBOOKS_ACTIVE = [TARGET_NB, SOURCE_NB_A, SOURCE_NB_B]

test.describe('B.6: cross-notebook merge UI', () => {
  test.setTimeout(60_000)
  test.use({ navigationTimeout: 45_000 })

  test.beforeEach(async ({ page }) => {
    await mockDashboardChrome(page)

    // Notebook listing — the dialog uses this to populate its source picker.
    await page.route(
      /\/api\/notebooks(?:\?[^/]*)?$/,
      (route) => {
        const url = new URL(route.request().url())
        const archived = url.searchParams.get('archived')
        const body = archived === 'true' ? [] : NOTEBOOKS_ACTIVE
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(body),
        })
      },
    )
  })

  test('preview shows counts, confirm fires non-dry-run and closes', async ({
    page,
  }) => {
    const mergeCalls: unknown[] = []

    await page.route(
      '**/api/notebooks/merge',
      async (route) => {
        const payload = route.request().postDataJSON() as {
          dry_run?: boolean
          source_ids: string[]
          target_id: string
        }
        mergeCalls.push(payload)

        // The mocked merge service returns a populated report on both
        // dry-run and the actual commit. UI behaviour differentiates
        // on the request shape, not the response.
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            entities_merged: 2,
            relations_created: 3,
            conflicts: [],
            source_notebook_ids: payload.source_ids,
            target_notebook_id: payload.target_id,
            dry_run: !!payload.dry_run,
          }),
        })
      },
    )

    await page.goto('/notebooks')

    // Wait for the target card to render.
    await expect(page.getByText(TARGET_NB.name, { exact: true })).toBeVisible()

    // Open the per-card overflow menu (opacity-0 by default but click
    // is force-able regardless of opacity).
    await page
      .getByTestId(`notebook-card-menu-${TARGET_NB.id}`)
      .click({ force: true })

    // Click the "Merge into…" menu item.
    await page.getByTestId(`notebook-card-merge-${TARGET_NB.id}`).click()

    // Dialog is up. Pick the two source notebooks (target is excluded).
    // The dialog title carries the target name; scope to it so we don't
    // catch the menu item that was just closed.
    await expect(
      page.getByRole('heading', { name: /Merge into/ }),
    ).toBeVisible()
    // The source-picker rows live inside the dialog; the same names
    // also appear in the underlying notebook listing, so scope to
    // the dialog to disambiguate.
    const dialog = page.getByRole('dialog')
    await dialog.getByText(SOURCE_NB_A.name).click()
    await dialog.getByText(SOURCE_NB_B.name).click()

    // Preview → counts render.
    await page.getByTestId('merge-dialog-preview').click()
    await expect(
      page.getByTestId('merge-dialog-preview-panel'),
    ).toBeVisible({ timeout: 5_000 })
    await expect(page.getByTestId('merge-preview-entities')).toHaveText(
      /2 entities would be merged/,
    )
    await expect(page.getByTestId('merge-preview-relations')).toHaveText(
      /3 relations would be created/,
    )

    // First request should be dry-run.
    await expect.poll(() => mergeCalls.length).toBe(1)
    expect((mergeCalls[0] as { dry_run?: boolean }).dry_run).toBe(true)

    // Confirm → non-dry-run + dialog closes.
    await page.getByTestId('merge-dialog-confirm').click()

    await expect.poll(() => mergeCalls.length).toBe(2)
    const commitCall = mergeCalls[1] as {
      dry_run?: boolean
      source_ids: string[]
      target_id: string
    }
    expect(commitCall.dry_run).toBe(false)
    expect(new Set(commitCall.source_ids)).toEqual(
      new Set([SOURCE_NB_A.id, SOURCE_NB_B.id]),
    )
    expect(commitCall.target_id).toBe(TARGET_NB.id)

    // Dialog dismisses after the commit completes.
    await expect(
      page.getByTestId('merge-dialog-preview-panel'),
    ).toBeHidden({ timeout: 5_000 })
  })
})
