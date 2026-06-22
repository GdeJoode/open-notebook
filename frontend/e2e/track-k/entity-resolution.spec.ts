/**
 * Track K — Phase K.6 — Entity-resolution review hub E2E spec.
 *
 * AC mapping → docs/tracks/K-entity-resolution/plan.md §K.6:
 *
 *   AC1  Hub renders in all states (default / loading / error / empty).   → the
 *        default + empty states are asserted here (loaded clusters, then the
 *        "No duplicates pending" affordance against an empty plan).
 *   AC2  Approve → confirmation step showing what merges → POST /apply    → click
 *        Approve, assert the confirm dialog lists the survivor + losers, confirm,
 *        and assert the captured POST /apply body carries that cluster's ids.
 *        The fuzzy-candidate "Merge" goes through the SAME gate — asserted below.
 *   AC3  Reject → force-split overlay (so it is not re-proposed)           → click
 *        Keep separate, assert POST /overlay with kind=split.
 *   AC5  OverlayEditor is keyboard-accessible (Tab/focus, aria-label).     → the
 *        scope radio group is present, labelled, and focusable.
 *   AC6  ExternalIdBadges render external ids as links; none → nothing.    → a
 *        seeded entity detail with external_ids shows the TOOI/DOI badge.
 *   AC7  Open hub → review a cluster → approve → it leaves the queue.      → the
 *        approved cluster disappears after the apply call resolves.
 *
 * Fully route-mocked (no live backend), matching the project's E2E convention.
 */

import { expect, test, type Route } from '@playwright/test'

import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const PLAN_WITH_CLUSTER = {
  scope: 'global',
  total_active_entities: 12,
  cluster_count: 1,
  clusters: [
    {
      new_canonical: 'binnenlandse zaken en koninkrijksrelaties',
      entity_type: 'organization',
      winner_id: 'entity:bzk1',
      loser_ids: ['entity:bzk2', 'entity:bzk3'],
      member_surface_forms: [
        'BZK',
        'Ministerie van BZK',
        'Binnenlandse Zaken',
      ],
      total_source_docs: 4,
      size: 3,
    },
  ],
}

const EMPTY_PLAN = {
  scope: 'global',
  total_active_entities: 12,
  cluster_count: 0,
  clusters: [],
}

const EMPTY_CANDIDATES = {
  scope: 'global',
  total_active_entities: 12,
  auto_merge_count: 0,
  review_count: 0,
  auto_merge: [],
  review: [],
}

const CANDIDATES_WITH_REVIEW = {
  scope: 'global',
  total_active_entities: 12,
  auto_merge_count: 0,
  review_count: 1,
  auto_merge: [],
  review: [
    {
      id_a: 'entity:vws1',
      id_b: 'entity:vws2',
      name_a: 'VWS',
      name_b: 'Volksgezondheid',
      entity_type: 'organization',
      score: 0.82,
      band: 'review',
      method: 'embedding',
      winner_id: 'entity:vws1',
      loser_id: 'entity:vws2',
    },
  ],
}

function json(route: Route, body: unknown, status = 200): Promise<void> {
  return route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

test.describe('Track K.6 — entity-resolution review hub', () => {
  test('loads a cluster, approves with confirmation, and it leaves the queue', async ({
    page,
  }) => {
    const capture: { applyBody?: unknown } = {}
    await mockDashboardChrome(page)

    // First plan call returns the cluster; after apply, return an empty plan so
    // the queue reflects the merge on the React Query invalidation refetch.
    let planCalls = 0
    await page.route('**/api/entity-resolution/plan', (route) => {
      planCalls += 1
      return json(route, planCalls === 1 ? PLAN_WITH_CLUSTER : EMPTY_PLAN)
    })
    await page.route('**/api/entity-resolution/candidates*', (route) =>
      json(route, EMPTY_CANDIDATES),
    )
    await page.route('**/api/entity-resolution/apply', (route) => {
      capture.applyBody = route.request().postDataJSON()
      return json(route, {
        applied: 1,
        skipped: 0,
        results: [
          {
            new_canonical: 'binnenlandse zaken en koninkrijksrelaties',
            entity_type: 'organization',
            winner_id: 'entity:bzk1',
            merged_loser_ids: ['entity:bzk2', 'entity:bzk3'],
            relations_repointed: 2,
            aliases_created: 2,
            skipped: false,
          },
        ],
      })
    })

    await gotoAndWait(page, '/knowledge-graph/resolution')
    if (page.url().includes('/login')) {
      test.skip(true, 'auth-gated stack — covered by the auth-disabled run')
    }

    // AC1 default: the cluster card renders.
    await expect(page.getByTestId('merge-cluster-entity:bzk1')).toBeVisible()
    await expect(page.getByText('Proposed merges')).toBeVisible()

    // AC2: approve opens the confirmation step showing what merges.
    await page.getByRole('button', { name: 'Approve merge' }).click()
    const dialog = page.getByTestId('merge-confirm-dialog')
    await expect(dialog).toBeVisible()
    await expect(dialog.getByText('Survives:')).toBeVisible()

    // Confirm → POST /apply with this cluster's ids.
    await page.getByTestId('merge-confirm-apply').click()
    await expect.poll(() => capture.applyBody).toBeTruthy()
    const body = capture.applyBody as {
      clusters: { winner_id: string; loser_ids: string[] }[]
    }
    expect(body.clusters[0].winner_id).toBe('entity:bzk1')
    expect(body.clusters[0].loser_ids).toEqual(['entity:bzk2', 'entity:bzk3'])

    // AC7: the approved cluster leaves the queue (locally dismissed + refetched).
    await expect(
      page.getByTestId('merge-cluster-entity:bzk1'),
    ).toHaveCount(0)
    await expect(page.getByText('No duplicates pending')).toBeVisible()
  })

  test('rejecting a cluster records a force-split overlay (AC3)', async ({
    page,
  }) => {
    const capture: { overlayBody?: unknown } = {}
    await mockDashboardChrome(page)

    await page.route('**/api/entity-resolution/plan', (route) =>
      json(route, PLAN_WITH_CLUSTER),
    )
    await page.route('**/api/entity-resolution/candidates*', (route) =>
      json(route, EMPTY_CANDIDATES),
    )
    await page.route('**/api/entity-resolution/overlay', (route) => {
      capture.overlayBody = route.request().postDataJSON()
      return json(route, { id: 'alias_overlay:1' })
    })

    await gotoAndWait(page, '/knowledge-graph/resolution')
    if (page.url().includes('/login')) {
      test.skip(true, 'auth-gated stack — covered by the auth-disabled run')
    }

    await expect(page.getByTestId('merge-cluster-entity:bzk1')).toBeVisible()
    await page
      .getByRole('button', { name: 'Reject — keep these entities separate' })
      .click()

    await expect.poll(() => capture.overlayBody).toBeTruthy()
    const body = capture.overlayBody as { kind: string; scope: string }
    expect(body.kind).toBe('split')
    expect(body.scope).toBe('global')
  })

  test('a fuzzy candidate merge also requires the confirmation gate (AC2)', async ({
    page,
  }) => {
    const capture: { applyBody?: unknown } = {}
    await mockDashboardChrome(page)

    await page.route('**/api/entity-resolution/plan', (route) =>
      json(route, EMPTY_PLAN),
    )
    await page.route('**/api/entity-resolution/candidates*', (route) =>
      json(route, CANDIDATES_WITH_REVIEW),
    )
    await page.route('**/api/entity-resolution/apply', (route) => {
      capture.applyBody = route.request().postDataJSON()
      return json(route, {
        applied: 1,
        skipped: 0,
        results: [
          {
            new_canonical: 'VWS',
            entity_type: 'organization',
            winner_id: 'entity:vws1',
            merged_loser_ids: ['entity:vws2'],
            relations_repointed: 1,
            aliases_created: 1,
            skipped: false,
          },
        ],
      })
    })

    await gotoAndWait(page, '/knowledge-graph/resolution')
    if (page.url().includes('/login')) {
      test.skip(true, 'auth-gated stack — covered by the auth-disabled run')
    }

    const card = page.getByTestId('merge-candidate-entity:vws1:entity:vws2')
    await expect(card).toBeVisible()

    // No apply must fire until the confirmation gate is confirmed.
    await card.getByRole('button', { name: 'Merge' }).click()
    const dialog = page.getByTestId('merge-confirm-dialog')
    await expect(dialog).toBeVisible()
    await expect(dialog.getByText('Survives:')).toBeVisible()
    expect(capture.applyBody).toBeUndefined()

    // Confirm → POST /apply carries the candidate's winner + loser ids.
    await page.getByTestId('merge-confirm-apply').click()
    await expect.poll(() => capture.applyBody).toBeTruthy()
    const body = capture.applyBody as {
      clusters: { winner_id: string; loser_ids: string[] }[]
    }
    expect(body.clusters[0].winner_id).toBe('entity:vws1')
    expect(body.clusters[0].loser_ids).toEqual(['entity:vws2'])
  })

  test('empty plan + candidates shows the "No duplicates pending" affordance (AC1)', async ({
    page,
  }) => {
    await mockDashboardChrome(page)
    await page.route('**/api/entity-resolution/plan', (route) =>
      json(route, EMPTY_PLAN),
    )
    await page.route('**/api/entity-resolution/candidates*', (route) =>
      json(route, EMPTY_CANDIDATES),
    )

    await gotoAndWait(page, '/knowledge-graph/resolution')
    if (page.url().includes('/login')) {
      test.skip(true, 'auth-gated stack — covered by the auth-disabled run')
    }

    await expect(page.getByText('No duplicates pending')).toBeVisible()
  })

  test('error from both queues shows an inline retry alert (AC1)', async ({
    page,
  }) => {
    await mockDashboardChrome(page)
    await page.route('**/api/entity-resolution/plan', (route) =>
      json(route, { detail: 'boom' }, 500),
    )
    await page.route('**/api/entity-resolution/candidates*', (route) =>
      json(route, { detail: 'boom' }, 500),
    )

    await gotoAndWait(page, '/knowledge-graph/resolution')
    if (page.url().includes('/login')) {
      test.skip(true, 'auth-gated stack — covered by the auth-disabled run')
    }

    await expect(page.getByTestId('resolution-error')).toBeVisible()
  })

  test('the overlay scope toggle is present, labelled, and focusable (AC5)', async ({
    page,
  }) => {
    await mockDashboardChrome(page)
    await page.route('**/api/entity-resolution/plan', (route) =>
      json(route, EMPTY_PLAN),
    )
    await page.route('**/api/entity-resolution/candidates*', (route) =>
      json(route, EMPTY_CANDIDATES),
    )

    await gotoAndWait(page, '/knowledge-graph/resolution')
    if (page.url().includes('/login')) {
      test.skip(true, 'auth-gated stack — covered by the auth-disabled run')
    }

    const globalScope = page.getByLabel('Scope: Global (everywhere)')
    await expect(globalScope).toBeVisible()
    await globalScope.focus()
    await expect(globalScope).toBeFocused()
  })
})

test.describe('Track K.6 — external-id badges on entity detail (AC6)', () => {
  const ENTITY_ID = 'entity:bzk-detail'

  test('an entity with external_ids shows a linked badge', async ({ page }) => {
    await mockDashboardChrome(page)

    await page.route('**/api/knowledge-graph/entity-types', (route) =>
      json(route, [{ entity_type: 'organization', count: 1 }]),
    )
    await page.route('**/api/knowledge-graph/entities?*', (route) =>
      json(route, {
        items: [
          {
            id: ENTITY_ID,
            name: 'BZK',
            entity_type: 'organization',
            weight: 5,
          },
        ],
        total: 1,
        limit: 50,
        offset: 0,
      }),
    )
    await page.route(`**/api/knowledge-graph/entities/${ENTITY_ID}`, (route) =>
      json(route, {
        id: ENTITY_ID,
        name: 'BZK',
        entity_type: 'organization',
        weight: 5,
        relations: [],
        aliases: ['Ministerie van BZK', 'Binnenlandse Zaken'],
        external_ids: [
          'https://identifier.overheid.nl/tooi/id/ministerie/mnre1034',
        ],
      }),
    )

    await gotoAndWait(page, '/knowledge-graph')
    if (page.url().includes('/login')) {
      test.skip(true, 'auth-gated stack — covered by the auth-disabled run')
    }

    await page.getByText('BZK', { exact: true }).first().click()
    await expect(page.getByTestId('external-id-badges')).toBeVisible()
    await expect(page.getByLabel(/^TOOI:/)).toBeVisible()
  })
})
