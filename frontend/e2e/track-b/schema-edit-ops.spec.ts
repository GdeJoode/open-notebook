/**
 * B.3b acceptance criteria for the schema edit-ops UI:
 *
 *   1. The schema row has a "⋯" overflow menu offering Rename / Merge / Split / Delete.
 *   2. Each op opens the SchemaEditDialog in the corresponding mode and submits via POST.
 *   3. Pending extensions render LIVE Accept / Reject buttons that POST to the
 *      accept / reject endpoints (no longer the B.3a "coming next release" tooltip).
 *   4. After a successful mutation the cache is replaced with the response body so
 *      the SchemaBrowser reflects the new state without a follow-up GET.
 *
 * Fully mocked — no live backend.
 */

import { expect, test } from '@playwright/test'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const NOTEBOOK_ID = 'notebook:b3b-edit'

const FIX_NOTEBOOK = {
  id: NOTEBOOK_ID,
  name: 'B.3b edit-ops notebook',
  description: 'Used by schema-edit-ops e2e spec',
  archived: false,
  created: '2026-06-01T00:00:00Z',
  updated: '2026-06-01T00:00:00Z',
}

const FIX_SCHEMA = {
  notebook_id: NOTEBOOK_ID,
  base_ontology: 'scholarly',
  base_ontology_types: [
    {
      name: 'Article',
      description: 'A published scholarly article.',
      parent_type: null,
      properties: [{ name: 'title', data_type: 'string', required: true }],
    },
    {
      name: 'Author',
      description: 'A person who authored the work.',
      parent_type: null,
      properties: [{ name: 'name', data_type: 'string', required: true }],
    },
  ],
  accepted_extensions: [],
  pending_extensions: [
    {
      extension_id: 'ext-pen-1',
      type_name: 'Cohort',
      parent_type: 'Group',
      rationale: 'Three sources mention a cohort.',
      properties: [],
    },
  ],
  excluded_types: [],
  coverage_pct: 0.8,
  review_required: false,
  soft_nudge_dismissed: false,
}

test.describe('B.3b: schema edit operations', () => {
  // Dev-server cold-start can exceed 15s, mirror the B.3a budget.
  test.setTimeout(60_000)
  test.use({ navigationTimeout: 45_000 })

  // The frontend has two API clients that build URLs differently —
  // notebooksApi uses the raw id, notebookSchemaApi uses
  // encodeURIComponent. Accept both spellings on each route.
  const NB_ID_RE = '(?:notebook:b3b-edit|notebook%3Ab3b-edit)'

  test.beforeEach(async ({ page }) => {
    await mockDashboardChrome(page)

    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(FIX_NOTEBOOK),
        }),
    )

    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/schema(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(FIX_SCHEMA),
        }),
    )

    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/pass1_results(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        }),
    )
  })

  test('rename op opens dialog, POSTs to /schema/rename, updates cache', async ({
    page,
  }) => {
    let renameRequestPayload: unknown = null

    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/schema/rename$`),
      async (route) => {
        renameRequestPayload = route.request().postDataJSON()
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            ...FIX_SCHEMA,
            accepted_extensions: [
              {
                extension_id: null,
                type_name: 'ResearchFellow',
                parent_type: null,
                description: null,
                rationale: null,
                properties: [],
              },
            ],
          }),
        })
      },
    )

    await page.goto(`/notebooks/${NOTEBOOK_ID}/schema`)

    // Open the overflow menu on the Author row.
    await page.getByTestId('schema-row-actions-Author').click()
    await page.getByTestId('schema-row-rename-Author').click()

    // Dialog opens — the rename input is pre-populated with the row name.
    const renameInput = page.getByTestId('schema-edit-rename-input')
    await expect(renameInput).toBeVisible()
    await renameInput.fill('ResearchFellow')

    await page.getByTestId('schema-edit-submit').click()

    await expect.poll(() => renameRequestPayload).toEqual({
      old_name: 'Author',
      new_name: 'ResearchFellow',
    })

    // Dialog closes on success; the new accepted extension appears.
    await expect(
      page.getByTestId('schema-tree-item-ResearchFellow'),
    ).toBeVisible({ timeout: 5_000 })
  })

  test('merge op submits sorted source list + merged name', async ({
    page,
  }) => {
    let mergeRequestPayload: unknown = null

    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/schema/merge$`),
      async (route) => {
        mergeRequestPayload = route.request().postDataJSON()
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            ...FIX_SCHEMA,
            accepted_extensions: [
              {
                extension_id: null,
                type_name: 'Contributor',
                parent_type: null,
                description: null,
                rationale: null,
                properties: [],
              },
            ],
          }),
        })
      },
    )

    await page.goto(`/notebooks/${NOTEBOOK_ID}/schema`)
    await page.getByTestId('schema-row-actions-Author').click()
    await page.getByTestId('schema-row-merge-Author').click()

    const sources = page.getByTestId('schema-edit-merge-sources')
    await expect(sources).toBeVisible()
    await sources.fill('Author, Editor')
    await page.getByTestId('schema-edit-merge-target').fill('Contributor')

    await page.getByTestId('schema-edit-submit').click()

    await expect.poll(() => mergeRequestPayload).toEqual({
      type_names: ['Author', 'Editor'],
      merged_name: 'Contributor',
    })
  })

  test('split op submits source + into + criterion', async ({ page }) => {
    let splitRequestPayload: unknown = null

    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/schema/split$`),
      async (route) => {
        splitRequestPayload = route.request().postDataJSON()
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            ...FIX_SCHEMA,
            accepted_extensions: [],
          }),
        })
      },
    )

    await page.goto(`/notebooks/${NOTEBOOK_ID}/schema`)
    await page.getByTestId('schema-row-actions-Author').click()
    await page.getByTestId('schema-row-split-Author').click()

    await page
      .getByTestId('schema-edit-split-into')
      .fill('PrimaryAuthor, CorrespondingAuthor')
    await page
      .getByTestId('schema-edit-split-criterion')
      .fill('by listed order in the byline')

    await page.getByTestId('schema-edit-submit').click()

    await expect.poll(() => splitRequestPayload).toEqual({
      type_name: 'Author',
      into: ['PrimaryAuthor', 'CorrespondingAuthor'],
      criterion: 'by listed order in the byline',
    })
  })

  test('delete op DELETEs the type and hides it from the browser', async ({
    page,
  }) => {
    let deleteCalled = false

    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/schema/types/Author$`),
      async (route) => {
        if (route.request().method() === 'DELETE') {
          deleteCalled = true
          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
              ...FIX_SCHEMA,
              excluded_types: ['Author'],
            }),
          })
        } else {
          await route.fallback()
        }
      },
    )

    await page.goto(`/notebooks/${NOTEBOOK_ID}/schema`)
    await page.getByTestId('schema-row-actions-Author').click()
    await page.getByTestId('schema-row-delete-Author').click()

    await page.getByTestId('schema-edit-submit').click()

    await expect.poll(() => deleteCalled).toBe(true)
    // After the cache replacement Author is filtered out via excluded_types.
    await expect(page.getByTestId('schema-tree-item-Author')).toHaveCount(0, {
      timeout: 5_000,
    })
  })

  test('pending extension Accept button POSTs and refreshes the panel', async ({
    page,
  }) => {
    let acceptCalled = false

    await page.route(
      new RegExp(
        `/api/notebooks/${NB_ID_RE}/schema/extensions/Cohort/accept$`,
      ),
      async (route) => {
        acceptCalled = true
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            ...FIX_SCHEMA,
            accepted_extensions: [
              {
                extension_id: 'ext-pen-1',
                type_name: 'Cohort',
                parent_type: 'Group',
                description: null,
                rationale: null,
                properties: [],
              },
            ],
            pending_extensions: [],
          }),
        })
      },
    )

    await page.goto(`/notebooks/${NOTEBOOK_ID}/schema`)

    const acceptBtn = page.getByTestId('accept-ext-pen-1')
    await expect(acceptBtn).toBeEnabled()
    await acceptBtn.click()

    await expect.poll(() => acceptCalled).toBe(true)
    // Pending list empties, accepted shows the new row.
    await expect(
      page.getByTestId('pending-extension-Cohort'),
    ).toHaveCount(0, { timeout: 5_000 })
    await expect(
      page.getByTestId('schema-tree-item-Cohort'),
    ).toBeVisible()
  })
})
