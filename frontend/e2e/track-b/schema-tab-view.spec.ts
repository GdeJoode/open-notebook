/**
 * B.3a acceptance criteria:
 *
 * 1. Navigating to `/notebooks/{id}/schema` renders the SchemaBrowser.
 * 2. The tree shows base-ontology entity types + accepted extensions.
 * 3. CoverageStatsTable shows per-source rows from /pass1_results.
 * 4. PendingExtensionsPanel lists pending extensions with disabled
 *    Accept/Reject buttons.
 * 5. TTL download button triggers a browser download.
 * 6. Keyboard accessibility: tree items selectable via Enter/Space;
 *    download button activatable via keyboard.
 *
 * Fully mocked — no live backend, no DB.
 */

import { expect, test } from '@playwright/test'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const NOTEBOOK_ID = 'notebook:b3a-fix'

const FIX_NOTEBOOK = {
  id: NOTEBOOK_ID,
  name: 'B.3a fixture notebook',
  description: 'Used by schema-tab e2e spec',
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
      properties: [
        { name: 'title', data_type: 'string', required: true },
        { name: 'doi', data_type: 'string', required: false },
      ],
    },
    {
      name: 'Author',
      description: 'A person who authored the work.',
      parent_type: null,
      properties: [{ name: 'name', data_type: 'string', required: true }],
    },
  ],
  accepted_extensions: [
    {
      extension_id: 'ext-acc-1',
      type_name: 'PreprintServer',
      parent_type: 'Organization',
      description: 'A preprint hosting service.',
      properties: [],
    },
  ],
  pending_extensions: [
    {
      extension_id: 'ext-pen-1',
      type_name: 'Cohort',
      parent_type: 'Group',
      rationale: 'Three sources mention a cohort.',
      properties: [],
    },
  ],
  coverage_pct: 0.82,
  review_required: false,
  soft_nudge_dismissed: false,
}

const FIX_PASS1: Array<Record<string, unknown>> = [
  {
    id: 'pass1_results:1',
    source_id: 'source:a',
    source_title: 'Paper A',
    schema_attempted: 'scholarly',
    detected_schema: 'scholarly',
    confidence_in_choice: 0.95,
    coverage_pct: 0.91,
    uncovered_concept_count: 1,
    created: '2026-06-01T10:00:00Z',
  },
  {
    id: 'pass1_results:2',
    source_id: 'source:b',
    source_title: 'Paper B',
    schema_attempted: 'scholarly',
    detected_schema: 'scholarly',
    confidence_in_choice: 0.81,
    coverage_pct: 0.62,
    uncovered_concept_count: 5,
    created: '2026-06-01T11:00:00Z',
  },
]

test.describe('B.3a: schema tab view-only', () => {
  // Dev-server first-route compile can exceed 15s on cold runs.
  test.setTimeout(60_000)
  test.use({ navigationTimeout: 45_000 })

  // The frontend has two API clients that build URLs *differently*:
  //   * `notebooksApi.get` interpolates the raw id, so the request URL
  //     carries a literal colon ("/api/notebooks/notebook:b3a-fix").
  //   * `notebookSchemaApi.get` and friends call `encodeURIComponent`
  //     first, so their URLs contain "%3A" instead.
  // Playwright glob route patterns are literal — `:` does not match
  // `%3A` — so we use a regex matcher on the notebook-id segment to
  // accept both spellings. This keeps the mock robust against either
  // client deciding to flip its encoding strategy in the future.
  const NB_ID_ENCODED_RE = '(?:notebook:b3a-fix|notebook%3Ab3a-fix)'

  test.beforeEach(async ({ page }) => {
    await mockDashboardChrome(page)

    // Notebook detail GET (used by useNotebook hook on the schema page).
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(FIX_NOTEBOOK),
        }),
    )

    // Schema JSON.
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}/schema(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(FIX_SCHEMA),
        }),
    )

    // Pass1 results.
    await page.route(
      new RegExp(
        `/api/notebooks/${NB_ID_ENCODED_RE}/pass1_results(?:\\?.*)?$`,
      ),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(FIX_PASS1),
        }),
    )

    // TTL download — Playwright's downloadEvent fires on the
    // browser's native download trigger from the Blob URL the
    // component creates, so we just need to satisfy the actual
    // GET /schema.ttl with a small Turtle body.
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}/schema\\.ttl(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'text/turtle',
          headers: {
            'content-disposition': `attachment; filename="${NOTEBOOK_ID.replace(
              ':',
              '_',
            )}.ttl"`,
          },
          body: '@prefix on: <https://open-notebook.dev/ontology/> .\non:Article a <http://www.w3.org/2002/07/owl#Class> .\n',
        }),
    )
  })

  test('schema page renders tree, coverage table, and pending extensions', async ({
    page,
  }) => {
    await page.goto(`/notebooks/${NOTEBOOK_ID}/schema`)

    // SchemaBrowser tree renders both base types and the accepted extension.
    const browser = page.getByTestId('schema-browser')
    await expect(browser).toBeVisible()
    await expect(
      page.getByTestId('schema-tree-item-Article'),
    ).toBeVisible()
    await expect(page.getByTestId('schema-tree-item-Author')).toBeVisible()
    await expect(
      page.getByTestId('schema-tree-item-PreprintServer'),
    ).toBeVisible()

    // CoverageStatsTable shows one row per source (deduped by source_id).
    const table = page.getByTestId('coverage-stats-table')
    await expect(table).toBeVisible()
    await expect(page.getByTestId('coverage-row-source:a')).toContainText(
      'Paper A',
    )
    await expect(page.getByTestId('coverage-row-source:a')).toContainText(
      '91%',
    )
    await expect(page.getByTestId('coverage-row-source:b')).toContainText(
      'Paper B',
    )
    await expect(page.getByTestId('coverage-row-source:b')).toContainText(
      '62%',
    )

    // PendingExtensionsPanel lists the pending extension with disabled
    // Accept/Reject buttons.
    const pending = page.getByTestId('pending-extension-Cohort')
    await expect(pending).toBeVisible()
    await expect(pending).toContainText('Three sources mention a cohort.')

    const acceptBtn = page.getByTestId('accept-ext-pen-1')
    const rejectBtn = page.getByTestId('reject-ext-pen-1')
    await expect(acceptBtn).toBeDisabled()
    await expect(rejectBtn).toBeDisabled()
  })

  test('selecting a tree item updates the side panel via keyboard', async ({
    page,
  }) => {
    await page.goto(`/notebooks/${NOTEBOOK_ID}/schema`)

    // Article is selected by default (first item alphabetically) —
    // verify side panel shows its properties.
    await expect(
      page.getByTestId('schema-properties-Article'),
    ).toBeVisible()
    await expect(
      page.getByTestId('schema-properties-Article'),
    ).toContainText('title')

    // Move focus to the Author row and activate via Enter.
    const authorRow = page.getByTestId('schema-tree-item-Author')
    await authorRow.focus()
    await page.keyboard.press('Enter')

    await expect(
      page.getByTestId('schema-properties-Author'),
    ).toBeVisible()
    await expect(
      page.getByTestId('schema-properties-Author'),
    ).toContainText('name')
  })

  test('TTL download button triggers a browser download', async ({
    page,
  }) => {
    await page.goto(`/notebooks/${NOTEBOOK_ID}/schema`)

    const downloadBtn = page.getByTestId('ttl-download-button')
    await expect(downloadBtn).toBeVisible()
    await expect(downloadBtn).toBeEnabled()

    // The download is staged from a Blob URL by the component, so
    // Playwright's `page.waitForEvent('download')` fires once the
    // synthetic <a download> click runs.
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      downloadBtn.click(),
    ])

    // Filename uses the Content-Disposition header from the mock.
    expect(download.suggestedFilename()).toBe(
      `${NOTEBOOK_ID.replace(':', '_')}.ttl`,
    )
  })

  test('Schema link on NotebookHeader is highlighted as active', async ({
    page,
  }) => {
    await page.goto(`/notebooks/${NOTEBOOK_ID}/schema`)

    const schemaLink = page.getByTestId('notebook-tab-schema')
    await expect(schemaLink).toBeVisible()
    await expect(schemaLink).toHaveAttribute('aria-current', 'page')

    // The Workspace link should still be present but NOT marked active.
    const workspaceLink = page.getByTestId('notebook-tab-workspace')
    await expect(workspaceLink).toBeVisible()
    await expect(workspaceLink).not.toHaveAttribute('aria-current', 'page')
  })
})
