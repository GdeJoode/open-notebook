/**
 * B.5b orphans dashboard acceptance criteria:
 *
 *   1. /notebooks/{id}/schema renders the OrphansDashboard with the
 *      pending + archived tabs and their counts.
 *   2. Pending tab lists per-row entries with a [Reconnect] action.
 *   3. Clicking Reconnect calls POST /orphans/{eid}/reconnect and the
 *      dashboard refetches the orphans response.
 *   4. The empty state renders when both counts are zero.
 *
 * Fully mocked -- no live backend.
 */

import { expect, test } from '@playwright/test'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const NOTEBOOK_ID = 'notebook:b5b-orphans'

const FIX_NOTEBOOK = {
  id: NOTEBOOK_ID,
  name: 'B.5b orphan-lifecycle notebook',
  description: 'Used by orphan-lifecycle e2e spec',
  archived: false,
  created: '2026-06-01T00:00:00Z',
  updated: '2026-06-01T00:00:00Z',
}

const FIX_SCHEMA = {
  notebook_id: NOTEBOOK_ID,
  base_ontology: 'scholarly',
  base_ontology_types: [],
  accepted_extensions: [],
  pending_extensions: [],
  excluded_types: [],
  coverage_pct: 0.0,
  review_required: false,
  soft_nudge_dismissed: false,
}

const ORPHANS_INITIAL = {
  notebook_id: NOTEBOOK_ID,
  pending_count: 2,
  archived_count: 1,
  items: [
    {
      id: 'entity:pending-1',
      canonical_name: 'Alice Researcher',
      entity_type: 'PERSON',
      orphan_status: 'pending_reconnect',
      reconnect_attempts: 1,
      first_orphaned_at: '2026-06-01T10:00:00Z',
      last_reconnect_attempt_at: '2026-06-01T10:00:00Z',
    },
    {
      id: 'entity:pending-2',
      canonical_name: 'OECD',
      entity_type: 'ORG',
      orphan_status: 'pending_reconnect',
      reconnect_attempts: 2,
      first_orphaned_at: '2026-05-28T09:00:00Z',
      last_reconnect_attempt_at: '2026-06-02T11:00:00Z',
    },
    {
      id: 'entity:archived-1',
      canonical_name: 'Outdated Term',
      entity_type: 'MISC',
      orphan_status: 'archived',
      reconnect_attempts: 3,
      first_orphaned_at: '2026-03-01T08:00:00Z',
      last_reconnect_attempt_at: '2026-05-15T08:00:00Z',
    },
  ],
}

const ORPHANS_AFTER_RECONNECT = {
  notebook_id: NOTEBOOK_ID,
  pending_count: 1,
  archived_count: 1,
  items: [
    // entity:pending-1 was successfully reconnected -- only pending-2 remains.
    {
      id: 'entity:pending-2',
      canonical_name: 'OECD',
      entity_type: 'ORG',
      orphan_status: 'pending_reconnect',
      reconnect_attempts: 2,
      first_orphaned_at: '2026-05-28T09:00:00Z',
      last_reconnect_attempt_at: '2026-06-02T11:00:00Z',
    },
    {
      id: 'entity:archived-1',
      canonical_name: 'Outdated Term',
      entity_type: 'MISC',
      orphan_status: 'archived',
      reconnect_attempts: 3,
      first_orphaned_at: '2026-03-01T08:00:00Z',
      last_reconnect_attempt_at: '2026-05-15T08:00:00Z',
    },
  ],
}

test.describe('B.5b: orphan-prune lifecycle dashboard', () => {
  // Dev-server cold start can exceed 15s -- match the B.3a/B.3b budget.
  test.setTimeout(60_000)
  test.use({ navigationTimeout: 45_000 })

  // Accept both spellings of the notebook id segment (raw vs %-encoded
  // colon) so the spec is robust against either API client.
  const NB_ID_RE = '(?:notebook:b5b-orphans|notebook%3Ab5b-orphans)'
  const RECONNECT_ID_RE = '(?:entity:pending-1|entity%3Apending-1)'

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

  test('orphan dashboard renders counts and per-row table', async ({
    page,
  }) => {
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/orphans(?:\\?.*)?$`),
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(ORPHANS_INITIAL),
        }),
    )

    await page.goto(`/notebooks/${NOTEBOOK_ID}/schema`)

    const dashboard = page.getByTestId('orphans-dashboard')
    await expect(dashboard).toBeVisible()

    // Tab badges reflect the counts from the response.
    const pendingTab = page.getByTestId('orphans-tab-pending')
    const archivedTab = page.getByTestId('orphans-tab-archived')
    await expect(pendingTab).toContainText('2')
    await expect(archivedTab).toContainText('1')

    // Pending tab is the default -- both pending rows render.
    await expect(
      page.getByTestId('orphans-row-entity:pending-1'),
    ).toContainText('Alice Researcher')
    await expect(
      page.getByTestId('orphans-row-entity:pending-2'),
    ).toContainText('OECD')

    // Each pending row carries a [Reconnect] action.
    await expect(
      page.getByTestId('reconnect-entity:pending-1'),
    ).toBeEnabled()
  })

  test('clicking Reconnect posts to the endpoint and refreshes the dashboard', async ({
    page,
  }) => {
    // Phase 1: initial fetch returns ORPHANS_INITIAL.
    // Phase 2 (after reconnect): returns ORPHANS_AFTER_RECONNECT.
    let fetchCount = 0
    await page.route(
      new RegExp(`/api/notebooks/${NB_ID_RE}/orphans(?:\\?.*)?$`),
      (route) => {
        fetchCount += 1
        const body =
          fetchCount === 1 ? ORPHANS_INITIAL : ORPHANS_AFTER_RECONNECT
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(body),
        })
      },
    )

    let reconnectCalled = false
    await page.route(
      new RegExp(
        `/api/notebooks/${NB_ID_RE}/orphans/${RECONNECT_ID_RE}/reconnect$`,
      ),
      (route) => {
        reconnectCalled = true
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            entity_id: 'entity:pending-1',
            attempted: 1,
            reconnected: 1,
            new_status: null,
          }),
        })
      },
    )

    await page.goto(`/notebooks/${NOTEBOOK_ID}/schema`)

    const reconnectBtn = page.getByTestId('reconnect-entity:pending-1')
    await expect(reconnectBtn).toBeEnabled()
    await reconnectBtn.click()

    // Wait for the row to disappear from the pending tab after the
    // refresh fires (server returned the reduced list).
    await expect(
      page.getByTestId('orphans-row-entity:pending-1'),
    ).toHaveCount(0)

    // The reconnect endpoint was hit and the table re-fetched.
    expect(reconnectCalled).toBe(true)
    expect(fetchCount).toBeGreaterThanOrEqual(2)

    // Pending count badge is now 1.
    await expect(page.getByTestId('orphans-tab-pending')).toContainText(
      '1',
    )
  })

  test('empty state renders when no orphans exist', async ({ page }) => {
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

    await page.goto(`/notebooks/${NOTEBOOK_ID}/schema`)

    const emptyState = page.getByTestId('orphans-empty-state')
    await expect(emptyState).toBeVisible()
  })
})
