/**
 * B.4 frontend acceptance:
 *
 *   1. KG page shows a confidence bar on every entity tile (visual smoke).
 *   2. ConfidenceFilter slider hides entities below the threshold.
 *   3. localStorage persists the threshold across reloads.
 *   4. Relation rows in the entity detail panel also render the bar.
 *
 * Fully mocked — no live backend, no DB.
 */

import { expect, test, type Page } from '@playwright/test'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const FIX_ENTITIES = [
  {
    id: 'entity:low',
    name: 'Low Confidence Entity',
    entity_type: 'Concept',
    weight: 1,
    confidence: 0.3,
  },
  {
    id: 'entity:mid',
    name: 'Mid Confidence Entity',
    entity_type: 'Concept',
    weight: 2,
    confidence: 0.7,
  },
  {
    id: 'entity:high',
    name: 'High Confidence Entity',
    entity_type: 'Concept',
    weight: 5,
    confidence: 0.95,
  },
]

const FIX_ENTITY_DETAIL = {
  id: 'entity:high',
  name: 'High Confidence Entity',
  entity_type: 'Concept',
  weight: 5,
  confidence: 0.95,
  relations: [
    {
      id: 'relation:r1',
      source: 'entity:high',
      target: 'entity:mid',
      relation_type: 'related_to',
      confidence: 0.42,
    },
    {
      id: 'relation:r2',
      source: 'entity:high',
      target: 'entity:low',
      relation_type: 'mentions',
      confidence: 0.88,
    },
  ],
}

async function mockKnowledgeGraphApis(page: Page): Promise<void> {
  await page.route('**/api/knowledge-graph/entities?**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        items: FIX_ENTITIES,
        total: FIX_ENTITIES.length,
        limit: 50,
        offset: 0,
      }),
    }),
  )

  await page.route('**/api/knowledge-graph/entities', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        items: FIX_ENTITIES,
        total: FIX_ENTITIES.length,
        limit: 50,
        offset: 0,
      }),
    }),
  )

  await page.route('**/api/knowledge-graph/entity-types', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([{ entity_type: 'Concept', count: 3 }]),
    }),
  )

  // Axios sends the entity id verbatim (no URL-encoding for path segments),
  // so we match the literal colon form. A second route with the
  // percent-encoded form is added defensively in case any caller wraps
  // the id with encodeURIComponent.
  await page.route('**/api/knowledge-graph/entities/entity:high', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(FIX_ENTITY_DETAIL),
    }),
  )
  await page.route(
    '**/api/knowledge-graph/entities/entity%3Ahigh',
    (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(FIX_ENTITY_DETAIL),
      }),
  )

  // Search endpoint — not used by these specs but a sibling hook
  // (`useSearchEntities`) is registered on mount and may fire if the
  // input value briefly defaults to non-empty. Mock defensively.
  await page.route('**/api/knowledge-graph/search**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([]),
    }),
  )

  // Graph endpoint — the Sigma tab loads lazily but we still want the
  // table render to settle without a 404 spamming the network panel.
  await page.route('**/api/knowledge-graph/graph**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ nodes: [], edges: [] }),
    }),
  )
}

test.describe('B.4: confidence display + filter', () => {
  // Dev-server first-route compile can exceed the default 15s on cold
  // runs. Mirrors the relaxation used by Track-A specs.
  test.setTimeout(60_000)
  test.use({ navigationTimeout: 45_000 })

  test.beforeEach(async ({ page }) => {
    await mockDashboardChrome(page)
    await mockKnowledgeGraphApis(page)
  })

  test('renders a confidence bar on every entity row', async ({ page }) => {
    await page.goto('/knowledge-graph')

    // Wait for the table to populate.
    await expect(page.getByTestId('entity-row-entity:high')).toBeVisible()
    await expect(page.getByTestId('entity-row-entity:mid')).toBeVisible()
    await expect(page.getByTestId('entity-row-entity:low')).toBeVisible()

    // Each row has its own confidence bar.
    for (const id of ['entity:low', 'entity:mid', 'entity:high']) {
      const bar = page
        .getByTestId(`entity-row-${id}`)
        .getByTestId('confidence-bar')
      await expect(bar).toBeVisible()
    }

    // The data-confidence attribute carries the precise value so the
    // tooltip text doesn't need to be hovered to assert correctness.
    await expect(
      page
        .getByTestId('entity-row-entity:low')
        .getByTestId('confidence-bar'),
    ).toHaveAttribute('data-confidence', '0.30')
    await expect(
      page
        .getByTestId('entity-row-entity:high')
        .getByTestId('confidence-bar'),
    ).toHaveAttribute('data-confidence', '0.95')
  })

  test('slider hides entities below the threshold', async ({ page }) => {
    await page.goto('/knowledge-graph')

    // All three start visible.
    await expect(page.getByTestId('entity-row-entity:low')).toBeVisible()
    await expect(page.getByTestId('entity-row-entity:mid')).toBeVisible()
    await expect(page.getByTestId('entity-row-entity:high')).toBeVisible()

    // Drive the slider via keyboard — Radix sliders accept ArrowRight
    // for fine-grained increments (step = 0.05).
    const slider = page.getByTestId('confidence-filter-slider').locator('[role="slider"]')
    await slider.focus()
    // 0 -> 0.05 -> 0.10 -> ... -> 0.50 = 10 steps right.
    for (let i = 0; i < 10; i++) {
      await page.keyboard.press('ArrowRight')
    }

    // Threshold label reflects the new value.
    await expect(page.getByTestId('confidence-filter')).toContainText('0.50')

    // The 0.30-confidence row is hidden; the 0.70 + 0.95 rows remain.
    await expect(page.getByTestId('entity-row-entity:low')).toHaveCount(0)
    await expect(page.getByTestId('entity-row-entity:mid')).toBeVisible()
    await expect(page.getByTestId('entity-row-entity:high')).toBeVisible()
  })

  test('threshold persists across reloads via localStorage', async ({
    page,
  }) => {
    await page.goto('/knowledge-graph')

    // Move the slider to 0.50.
    const slider = page.getByTestId('confidence-filter-slider').locator('[role="slider"]')
    await slider.focus()
    for (let i = 0; i < 10; i++) {
      await page.keyboard.press('ArrowRight')
    }
    await expect(page.getByTestId('confidence-filter')).toContainText('0.50')

    // Verify localStorage was written.
    const stored = await page.evaluate(() =>
      window.localStorage.getItem('kg_confidence_threshold'),
    )
    expect(stored).not.toBeNull()
    expect(parseFloat(stored as string)).toBeCloseTo(0.5, 2)

    // Reload and re-assert: the label is restored and the 0.30 row is
    // still hidden without any user interaction.
    await page.reload()
    await expect(page.getByTestId('confidence-filter')).toContainText('0.50')
    await expect(page.getByTestId('entity-row-entity:low')).toHaveCount(0)
    await expect(page.getByTestId('entity-row-entity:mid')).toBeVisible()
  })

  test('relation rows render their own confidence bars', async ({ page }) => {
    await page.goto('/knowledge-graph')

    // Open detail panel by clicking the high-confidence entity.
    await page.getByTestId('entity-row-entity:high').click()

    // Detail panel shows two relation cards, each with a confidence bar.
    const r1Bar = page
      .getByTestId('relation-row-relation:r1')
      .getByTestId('confidence-bar')
    const r2Bar = page
      .getByTestId('relation-row-relation:r2')
      .getByTestId('confidence-bar')

    await expect(r1Bar).toBeVisible()
    await expect(r1Bar).toHaveAttribute('data-confidence', '0.42')

    await expect(r2Bar).toBeVisible()
    await expect(r2Bar).toHaveAttribute('data-confidence', '0.88')
  })
})
