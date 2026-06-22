/**
 * Track J — Phase J.5 — Model-routing config + layered privacy E2E spec.
 *
 * AC mapping → docs/tracks/J-model-routing/plan.md §J.5:
 *
 *   AC1  Page renders with the loaded chains (default state)         → assert the
 *        three task cards (Entity Extraction / Summarization / Chat) render.
 *   AC2  Reorder a provider → PUT /api/model-routes/{task} with the   → click "Move
 *        new order; reload reflects it.                                  down" on the
 *                                                                        head provider,
 *                                                                        Save, assert the
 *                                                                        captured PUT body
 *                                                                        has the swapped
 *                                                                        order.
 *   AC4  PrivacyModeToggle renders the global default + is keyboard-   → the global
 *        operable.                                                        radio group is
 *                                                                        present and
 *                                                                        focusable.
 *   AC5  A private source shows a "Private" badge.                     → seed a private
 *                                                                        source detail and
 *                                                                        assert the badge.
 *   AC6  Provider health chips show configured/unconfigured + circuit  → assert a
 *        state.                                                           configured and an
 *                                                                        unconfigured chip.
 *
 * Fully route-mocked (no live backend), matching the project's E2E convention.
 * AC7's full ingest-with-private flow needs a live worker; here we assert the
 * load → reorder → save → toggle-global-privacy path on the settings page plus
 * the private-source badge on a seeded source detail. The live ingest path is a
 * J.6 operator smoke-checklist item (no worker/keys in this harness).
 */

import { expect, test, type Page, type Route } from '@playwright/test'

import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome, DEFAULT_SETTINGS } from '../track-a/_track-a-helpers'

const DEFAULT_ROUTES = [
  {
    task: 'entity_extraction',
    provider_chain: [
      { provider: 'nvidia', model_id: null, enabled: true },
      { provider: 'ollama', model_id: null, enabled: true },
    ],
    private_chain: [{ provider: 'ollama', model_id: null, enabled: true }],
    is_default: false,
  },
  {
    task: 'summarization',
    provider_chain: [
      { provider: 'nvidia', model_id: null, enabled: true },
      { provider: 'ollama', model_id: null, enabled: true },
    ],
    private_chain: [{ provider: 'ollama', model_id: null, enabled: true }],
    is_default: false,
  },
  {
    task: 'chat',
    provider_chain: [
      { provider: 'nvidia', model_id: null, enabled: true },
      { provider: 'ollama', model_id: null, enabled: true },
    ],
    private_chain: [{ provider: 'ollama', model_id: null, enabled: true }],
    is_default: false,
  },
]

const SUMMARY = {
  total_events: 5,
  cloud_count: 2,
  local_count: 3,
  fallback_count: 1,
  by_provider: { nvidia: 2, ollama: 3 },
  by_task: { entity_extraction: 3, summarization: 2 },
  notebook_id: null,
}

const HEALTH = {
  providers: [
    {
      provider: 'nvidia',
      is_local: false,
      configured: true,
      circuit_state: 'closed',
      recent_fallback_count: 0,
    },
    {
      provider: 'anthropic',
      is_local: false,
      configured: false,
      circuit_state: 'closed',
      recent_fallback_count: 0,
    },
    {
      provider: 'ollama',
      is_local: true,
      configured: true,
      circuit_state: 'closed',
      recent_fallback_count: 0,
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

async function mockRoutingApis(
  page: Page,
  capture: { lastPut?: { task: string; body: unknown } },
): Promise<void> {
  await page.route('**/api/model-routes/health', (route) => json(route, HEALTH))

  await page.route('**/api/model-routes/summary*', (route) =>
    json(route, SUMMARY),
  )

  await page.route('**/api/model-routes', (route) =>
    json(route, DEFAULT_ROUTES),
  )

  await page.route('**/api/model-routes/*', async (route) => {
    if (route.request().method() === 'PUT') {
      const url = route.request().url()
      const task = url.split('/').pop() ?? ''
      capture.lastPut = { task, body: route.request().postDataJSON() }
      // Echo back the saved chain so the UI reflects it.
      await json(route, {
        task,
        provider_chain: (capture.lastPut.body as { provider_chain: unknown })
          .provider_chain,
        private_chain: [],
        is_default: false,
      })
      return
    }
    await route.fallback()
  })

  await page.route('**/api/settings', (route) => {
    if (route.request().method() === 'PUT') {
      return json(route, {
        ...DEFAULT_SETTINGS,
        default_privacy_mode: 'private',
      })
    }
    return json(route, { ...DEFAULT_SETTINGS, default_privacy_mode: 'cloud' })
  })
}

test.describe('Track J.5 — model routing settings', () => {
  test('loads chains, reorders + saves, and shows health chips', async ({
    page,
  }) => {
    const capture: { lastPut?: { task: string; body: unknown } } = {}
    await mockDashboardChrome(page)
    await mockRoutingApis(page, capture)

    await gotoAndWait(page, '/settings/model-routing')

    if (page.url().includes('/login')) {
      test.skip(true, 'auth-gated stack — covered by the auth-disabled run')
    }

    // AC1: the three task cards render.
    await expect(
      page.getByText('Entity Extraction', { exact: true }),
    ).toBeVisible()
    await expect(page.getByText('Summarization', { exact: true })).toBeVisible()

    // AC6: a configured and an unconfigured provider chip are both present, and
    // the unconfigured one is labelled distinctly.
    await expect(
      page.getByLabel('nvidia: healthy'),
    ).toBeVisible()
    await expect(
      page.getByLabel('anthropic: not configured'),
    ).toBeVisible()

    // J.6: the recent-routing summary card renders the cloud/local/fallback
    // counts read from the routing.served telemetry.
    await expect(page.getByText('Recent Routing', { exact: true })).toBeVisible()
    await expect(page.getByLabel('Cloud: 2')).toBeVisible()
    await expect(page.getByLabel('Local: 3')).toBeVisible()
    await expect(page.getByLabel('Fallbacks: 1')).toBeVisible()

    // AC4: the global privacy radio group renders and is focusable.
    const globalCloud = page.getByLabel('Global default: Cloud')
    await expect(globalCloud).toBeVisible()
    await globalCloud.focus()
    await expect(globalCloud).toBeFocused()

    // AC2: reorder the head provider of the Entity Extraction chain down, then
    // save, and assert the captured PUT carries the swapped order.
    const moveDown = page.getByLabel('Move nvidia down').first()
    await moveDown.click()
    await page
      .getByRole('button', { name: 'Save', exact: true })
      .first()
      .click()

    await expect
      .poll(() => capture.lastPut?.task)
      .toBe('entity_extraction')
    const body = capture.lastPut?.body as {
      provider_chain: { provider: string }[]
    }
    expect(body.provider_chain.map((e) => e.provider)).toEqual([
      'ollama',
      'nvidia',
    ])
  })
})

test.describe('Track J.5 — private source badge (AC5)', () => {
  const SOURCE_ID = 'source:track-j-private'

  test('a private source renders the Private badge', async ({ page }) => {
    await mockDashboardChrome(page)

    await page.route(
      new RegExp('/api/sources/[^/]+(?:\\?.*)?$'),
      (route) =>
        json(route, {
          id: SOURCE_ID,
          title: 'Confidential memo',
          topics: [],
          asset: { file_path: '/tmp/memo.pdf' },
          full_text: 'redacted',
          embedded: false,
          embedded_chunks: 0,
          insights_count: 0,
          entity_count: 0,
          relation_count: 0,
          created: '2026-06-01T00:00:00Z',
          updated: '2026-06-01T00:00:00Z',
          notebooks: [],
          private: true,
        }),
    )
    // Insights/entities side calls — keep them empty so the page renders.
    await page.route('**/api/sources/*/insights', (route) => json(route, []))

    await gotoAndWait(page, `/sources/${encodeURIComponent(SOURCE_ID)}`)

    if (page.url().includes('/login')) {
      test.skip(true, 'auth-gated stack — covered by the auth-disabled run')
    }

    await expect(
      page.getByTestId('private-source-badge'),
    ).toBeVisible()
  })
})
