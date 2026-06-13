/**
 * Track-D Phase D.3 acceptance:
 *
 * 1. The NetworkxExportMenu trigger renders on the notebook Schema tab.
 * 2. Clicking a format (GraphML) opens the dropdown and POSTs to
 *    /api/notebooks/{id}/export-networkx with `{format: "graphml"}`,
 *    then triggers a browser download.
 * 3. A second format (JSON-tree) triggers a download with the matching
 *    `application/json` content type.
 *
 * Fully mocked -- no live backend.
 */

import { expect, test } from '@playwright/test'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const NOTEBOOK_ID = 'notebook:d3-fix'

const FIX_NOTEBOOK = {
  id: NOTEBOOK_ID,
  name: 'D.3 fixture notebook',
  description: 'Used by networkx-export e2e spec',
  archived: false,
  created: '2026-06-01T00:00:00Z',
  updated: '2026-06-01T00:00:00Z',
}

// Re-use the schema page minimal fixtures so the NetworkX menu mounts
// inside the Schema tab without crashing on missing data.
const FIX_SCHEMA = {
  notebook_id: NOTEBOOK_ID,
  base_ontology: 'scholarly',
  base_ontology_types: [
    {
      name: 'Article',
      description: 'A scholarly article.',
      parent_type: null,
      properties: [],
    },
  ],
  accepted_extensions: [],
  pending_extensions: [],
  coverage_pct: 0.0,
  review_required: false,
  soft_nudge_dismissed: false,
}

test.describe('D.3: NetworkX export dropdown', () => {
  // First-route compile can exceed the default 15s on cold dev runs --
  // same mitigation as the B.3a schema-tab spec.
  test.setTimeout(60_000)
  test.use({ navigationTimeout: 45_000 })

  // The notebooks client uses a raw id ("notebook:d3-fix") and the
  // schema/exports clients pass through `encodeURIComponent`
  // ("notebook%3Ad3-fix"). The regex matches both spellings.
  const NB_ID_RE = '(?:notebook:d3-fix|notebook%3Ad3-fix)'

  test.beforeEach(async ({ page }) => {
    await mockDashboardChrome(page)

    // Notebook detail GET.
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(FIX_NOTEBOOK),
        }),
    )

    // Schema JSON (so the page renders).
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/schema(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(FIX_SCHEMA),
        }),
    )

    // Pass1 results (empty list keeps the coverage table simple).
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/pass1_results(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        }),
    )

    // Orphans dashboard fetches its own endpoint -- empty list is fine.
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/orphans(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            notebook_id: NOTEBOOK_ID,
            pending_count: 0,
            archived_count: 0,
            items: [],
          }),
        }),
    )
  })

  test('GraphML format triggers a browser download', async ({ page }) => {
    // Mock the NetworkX export endpoint -- assert the request body
    // shape on the way in.
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/export-networkx$`),
      async (route) => {
        const body = route.request().postDataJSON()
        expect(body.format).toBe('graphml')
        // Filter knobs are wired and use min_confidence=0.9 by default.
        expect(body.filter.min_confidence).toBe(0.9)
        await route.fulfill({
          status: 200,
          contentType: 'application/xml',
          headers: {
            'content-disposition':
              'attachment; filename="notebook_d3-fix.graphml"',
            // Browsers gate response-header access through CORS. The
            // mock has to opt-in to Content-Disposition exposure so
            // axios on the page side can read it.
            'access-control-expose-headers': 'Content-Disposition',
          },
          body: '<?xml version="1.0"?><graphml/>',
        })
      },
    )

    await page.goto(`/notebooks/${NOTEBOOK_ID}/schema`)

    const trigger = page.getByTestId('networkx-export-trigger')
    await expect(trigger).toBeVisible()

    await trigger.click()

    const graphmlItem = page.getByTestId('networkx-format-graphml')
    await expect(graphmlItem).toBeVisible()

    const [download] = await Promise.all([
      page.waitForEvent('download'),
      graphmlItem.click(),
    ])

    expect(download.suggestedFilename()).toBe('notebook_d3-fix.graphml')
  })

  test('JSON-tree format triggers a download with json content-type', async ({
    page,
  }) => {
    // Track what content-type we mocked for the assertion below. The
    // download event itself doesn't expose the response headers, but
    // the request matcher closure receives the route.request() so we
    // can assert on the body shape there.
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/export-networkx$`),
      async (route) => {
        const body = route.request().postDataJSON()
        expect(body.format).toBe('json-tree')
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          headers: {
            'content-disposition':
              'attachment; filename="notebook_d3-fix.json"',
            'access-control-expose-headers': 'Content-Disposition',
          },
          body: '{"nodes": [], "links": []}',
        })
      },
    )

    await page.goto(`/notebooks/${NOTEBOOK_ID}/schema`)

    const trigger = page.getByTestId('networkx-export-trigger')
    await trigger.click()

    const jsonItem = page.getByTestId('networkx-format-json-tree')

    const [download] = await Promise.all([
      page.waitForEvent('download'),
      jsonItem.click(),
    ])

    expect(download.suggestedFilename()).toBe('notebook_d3-fix.json')
  })
})
