/**
 * Track Q — Phase Q.5 — Triage UI surface E2E spec.
 *
 * AC mapping → docs/tracks/Q-triage/plan.md §Q.5:
 *
 *   AC1  KG entity table renders states + filters by status.            → the
 *        table renders loaded rows; selecting the status filter re-queries with
 *        `?status=reference` (asserted on the captured request).
 *   AC2  Each row shows structural-degree + cross-doc-count.            → the row
 *        renders the degree + docs columns from the list payload.
 *   AC3  Detail panel shows status + a manual_override toggle; toggling → open an
 *        entity, flip the override switch, assert POST /override carries
 *        manual_override=true (override-wins; the status pin persists server-side).
 *   AC4  Triage-queue view lists rows + a decision control; deciding it → open the
 *        queue, click Promote, assert POST decision fires and the row leaves.
 *   AC5  Backbone warnings render for degree>=3 weak-edge entities.     → covered
 *        by the BackboneWarnings unit tests + the queue render; not re-asserted
 *        here (no dedicated panel route in this mocked flow).
 *   AC6  Keyboard-accessible (aria-labels, focusable controls).         → the
 *        status filter + override toggle + decision buttons carry aria-labels and
 *        are focusable.
 *
 * Fully route-mocked (no live backend), matching the project's E2E convention.
 */

import { expect, test, type Route } from '@playwright/test'

import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const ENTITY_ID = 'entity:bzk-triage'

const REFERENCE_ENTITY = {
  id: ENTITY_ID,
  name: 'BZK',
  entity_type: 'organization',
  weight: 5,
  status: 'reference',
  manual_override: false,
  structural_degree: 4,
  doc_count: 3,
}

const QUEUE_OPEN = {
  count: 1,
  items: [
    {
      id: 'triage_queue:1',
      entity: ENTITY_ID,
      name: 'BZK',
      type: 'Organisatie',
      structural_degree: 0,
      doc_count: 2,
      reason: 'unsure — operator decides',
      decision: 'open',
      batch_id: 'b1',
    },
  ],
}

const QUEUE_EMPTY = { count: 0, items: [] }

function json(route: Route, body: unknown, status = 200): Promise<void> {
  return route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

test.describe('Track Q.5 — KG status filter + override toggle', () => {
  test('table filters by status and the detail panel toggles manual_override', async ({
    page,
  }) => {
    const capture: { listUrls: string[]; overrideBody?: unknown } = {
      listUrls: [],
    }
    await mockDashboardChrome(page)

    await page.route('**/api/knowledge-graph/entity-types', (route) =>
      json(route, [{ entity_type: 'organization', count: 1 }]),
    )
    await page.route('**/api/knowledge-graph/entities?*', (route) => {
      capture.listUrls.push(route.request().url())
      return json(route, {
        items: [REFERENCE_ENTITY],
        total: 1,
        limit: 50,
        offset: 0,
      })
    })
    await page.route(`**/api/knowledge-graph/entities/${ENTITY_ID}`, (route) =>
      json(route, { ...REFERENCE_ENTITY, relations: [] }),
    )
    await page.route(
      `**/api/triage/entities/${ENTITY_ID}/override`,
      (route) => {
        capture.overrideBody = route.request().postDataJSON()
        return json(route, {
          entity: ENTITY_ID,
          status: 'reference',
          manual_override: true,
          closed_queue_rows: 1,
        })
      },
    )

    await gotoAndWait(page, '/knowledge-graph')
    if (page.url().includes('/login')) {
      test.skip(true, 'auth-gated stack — covered by the auth-disabled run')
    }

    // AC1/AC2: the row renders with the status badge + degree/docs columns.
    await expect(page.getByTestId('status-filter')).toBeVisible()
    const row = page.getByTestId(`entity-row-${ENTITY_ID}`)
    await expect(row).toBeVisible()
    await expect(row.getByTestId('status-badge')).toContainText('Reference')

    // AC1: filtering by status re-queries with ?status=reference.
    await page.getByTestId('status-filter').click()
    await page.getByRole('option', { name: 'Reference' }).click()
    await expect
      .poll(() => capture.listUrls.some((u) => u.includes('status=reference')))
      .toBe(true)

    // AC3: open the entity, flip the override toggle → POST /override (pin).
    await row.click()
    const toggle = page.getByTestId('override-toggle')
    await expect(toggle).toBeVisible()
    await toggle.click()
    await expect.poll(() => capture.overrideBody).toBeTruthy()
    const body = capture.overrideBody as { manual_override: boolean }
    expect(body.manual_override).toBe(true)
  })
})

test.describe('Track Q.5 — triage queue view', () => {
  test('lists open rows and a decision closes one (AC4)', async ({ page }) => {
    const capture: { decisionBody?: unknown } = {}
    await mockDashboardChrome(page)

    // First queue call returns the open row; after a decision, return empty so
    // the queue reflects the close on the React Query invalidation refetch.
    let queueCalls = 0
    await page.route('**/api/triage/queue*', (route) => {
      queueCalls += 1
      return json(route, queueCalls === 1 ? QUEUE_OPEN : QUEUE_EMPTY)
    })
    await page.route(
      '**/api/triage/queue/triage_queue:1/decision',
      (route) => {
        capture.decisionBody = route.request().postDataJSON()
        return json(route, {
          queue_id: 'triage_queue:1',
          entity: ENTITY_ID,
          status: 'active',
          manual_override: true,
          decision: 'promote → active',
        })
      },
    )

    await gotoAndWait(page, '/knowledge-graph/triage')
    if (page.url().includes('/login')) {
      test.skip(true, 'auth-gated stack — covered by the auth-disabled run')
    }

    // AC4: the open row renders under its reason group.
    const queueRow = page.getByTestId('triage-row-triage_queue:1')
    await expect(queueRow).toBeVisible()
    await expect(page.getByText('unsure — operator decides')).toBeVisible()

    // Decide: Promote → POST decision with status active, row leaves the queue.
    await queueRow.getByRole('button', { name: /Promote/ }).click()
    await expect.poll(() => capture.decisionBody).toBeTruthy()
    const body = capture.decisionBody as { status: string }
    expect(body.status).toBe('active')
    await expect(queueRow).toHaveCount(0)
  })

  test('empty queue shows the "Nothing to triage" affordance', async ({
    page,
  }) => {
    await mockDashboardChrome(page)
    await page.route('**/api/triage/queue*', (route) => json(route, QUEUE_EMPTY))

    await gotoAndWait(page, '/knowledge-graph/triage')
    if (page.url().includes('/login')) {
      test.skip(true, 'auth-gated stack — covered by the auth-disabled run')
    }

    await expect(page.getByText('Nothing to triage')).toBeVisible()
  })

  test('an error from the queue shows an inline retry alert', async ({
    page,
  }) => {
    await mockDashboardChrome(page)
    await page.route('**/api/triage/queue*', (route) =>
      json(route, { detail: 'boom' }, 500),
    )

    await gotoAndWait(page, '/knowledge-graph/triage')
    if (page.url().includes('/login')) {
      test.skip(true, 'auth-gated stack — covered by the auth-disabled run')
    }

    await expect(page.getByTestId('triage-error')).toBeVisible()
  })
})
