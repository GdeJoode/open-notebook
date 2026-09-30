/**
 * Track WB — Phase WB.5 — every dialog a menu item opens closes back to a
 * working page.
 *
 * WB.4 review round 1: a Dialog opened from a modal menu's `onSelect` (or
 * `onClick`) left `pointer-events: none` on <body> once it closed, and the
 * page stopped responding. WB.4 moved the notebook header's menus to
 * `useMenuAction`; this spec holds the other menus that open a Dialog or an
 * AlertDialog (found by grepping `DropdownMenuItem`): the notebook's Sources
 * column (Add ▾ and each source card's ⋮), the notebook cards, the source
 * page's actions, the sidebar's Create, and the podcast profiles.
 *
 * Each item is opened by mouse and by keyboard, closed by Escape and by its
 * Cancel/Close button, and then the page must take a real click.
 *
 * Measured before the fix: Add ▾, the notebook card, Reprocess and Create
 * failed (the dialog did not take Escape). The source card's items (the
 * parent's AlertDialog) and the podcast profiles (an AlertDialogTrigger
 * around the item) passed as they were and are held here unchanged.
 */

import { expect, test, type Locator, type Page } from '@playwright/test'

import { gotoAndWait } from '../_helpers'
import { NOTEBOOK, SOURCE, apiSource, json, mockBackend } from './_wb-backend'

const enc = encodeURIComponent

async function backendWithMenus(page: Page) {
  await mockBackend(page)
  const inNotebook = { ...apiSource('source:nb1', 'Convenant Regio Deal Achterhoek', 'complete'),
    asset: { file_path: '/uploads/convenant.pdf' }, stalled: false, needs_you: false }
  // The notebook's own sources (the Sources column): one card with a menu.
  await page.route(/\/api\/sources\/?\?.*notebook_id=notebook(?::|%3A)oe5.*$/, (r) => json(r, [inNotebook]))
  // A source with a file, so Reprocess is enabled.
  const re = SOURCE.replace(':', '(?::|%3A)')
  await page.route(new RegExp(`/api/sources/${re}$`), (r) => json(r, {
    ...apiSource(SOURCE, 'Convenant Regio Deal Achterhoek.pdf', 'complete'),
    asset: { file_path: '/uploads/convenant.pdf' } }))
  await page.route('**/api/episode-profiles', (r) => json(r, [{ id: 'episode_profile:1',
    name: 'Briefing', description: '', speaker_config: 'Duo', outline_provider: 'openai',
    outline_model: 'gpt', transcript_provider: 'openai', transcript_model: 'gpt',
    default_briefing: '', num_segments: 3 }]))
  await page.route('**/api/speaker-profiles', (r) => json(r, [
    { id: 'speaker_profile:1', name: 'Duo', description: '', tts_provider: 'openai',
      tts_model: 'tts', speakers: [] },
    { id: 'speaker_profile:2', name: 'Solo', description: '', tts_provider: 'openai',
      tts_model: 'tts', speakers: [] }]))
}

interface Target {
  item: string
  /** The menu trigger. */
  menu: () => Locator
  /** The layer the item opens. */
  layer: () => Locator
}

/**
 * Opens `t` by mouse or keyboard, closes it by Escape or its button, then
 * checks the page responds: body pointer-events back to auto and a real
 * mouse click on the menu trigger opens its menu again.
 */
async function roundTrip(page: Page, t: Target) {
  for (const open of ['mouse', 'keyboard'] as const) {
    for (const close of ['Escape', 'button'] as const) {
      const what = `${t.item} (${open}, ${close})`
      if (open === 'mouse') {
        await t.menu().click()
        await page.getByRole('menuitem', { name: t.item }).click()
      } else {
        await t.menu().focus()
        await page.keyboard.press('Enter')
        await page.getByRole('menuitem', { name: t.item }).focus()
        await page.keyboard.press('Enter')
      }
      const layer = t.layer()
      await expect(layer, what).toBeVisible()
      if (close === 'Escape') await page.keyboard.press('Escape')
      else await layer.getByRole('button', { name: /^(Cancel|Close)$/ }).first().click()
      await expect(layer, what).toBeHidden()

      await expect.poll(() => page.evaluate(() => getComputedStyle(document.body).pointerEvents),
        { message: what }).toBe('auto')
      // A real click lands: the menu opens again.
      await t.menu().click({ timeout: 5_000 })
      await expect(page.getByRole('menu'), what).toBeVisible()
      await page.keyboard.press('Escape')
      await expect(page.getByRole('menu'), what).toBeHidden()
    }
  }
}

const dialog = (page: Page, name: RegExp) => () => page.getByRole('dialog', { name })
const alert = (page: Page) => () => page.getByRole('alertdialog')

test.describe('WB.5 — menus that open dialogs leave a working page', () => {
  test.setTimeout(240_000)
  test.use({ navigationTimeout: 60_000, viewport: { width: 1440, height: 900 } })

  test('the notebook Sources column: Add ▾ and a source card ⋮', async ({ page }) => {
    await backendWithMenus(page)
    await gotoAndWait(page, `/notebooks/${enc(NOTEBOOK)}`)
    const addMenu = () => page.getByRole('button', { name: 'Add Source' })
    await expect(addMenu()).toBeVisible({ timeout: 60_000 })
    await roundTrip(page, { item: 'Add New Source', menu: addMenu,
      layer: dialog(page, /Add New Source/) })
    await roundTrip(page, { item: 'Add Existing Source', menu: addMenu,
      layer: dialog(page, /Add Existing Source/) })

    const cardMenu = () => page.getByRole('button', { name: 'Source actions' })
    await expect(cardMenu()).toHaveCount(1)
    await roundTrip(page, { item: 'Remove from Notebook', menu: cardMenu, layer: alert(page) })
    await roundTrip(page, { item: 'Delete Source', menu: cardMenu, layer: alert(page) })
  })

  test('a notebook card ⋮: Merge into… and Delete', async ({ page }) => {
    await backendWithMenus(page)
    await gotoAndWait(page, '/notebooks')
    const menu = () => page.getByTestId(`notebook-card-menu-${NOTEBOOK}`)
    await expect(menu()).toBeAttached({ timeout: 60_000 })
    await roundTrip(page, { item: 'Merge into…', menu, layer: dialog(page, /Merge/) })
    await roundTrip(page, { item: 'Delete', menu, layer: alert(page) })
  })

  test('the source page actions: Reprocess Document…', async ({ page }) => {
    await backendWithMenus(page)
    await gotoAndWait(page, `/sources/${enc(SOURCE)}`)
    const menu = () => page.locator('[data-page-header]').getByRole('button', { name: 'Source actions' })
    await expect(menu()).toBeVisible({ timeout: 60_000 })
    await roundTrip(page, { item: 'Reprocess Document...', menu,
      layer: dialog(page, /Reprocess Document/) })
  })

  test('the sidebar Create ▾: Notebook and Podcast', async ({ page }) => {
    await backendWithMenus(page)
    await gotoAndWait(page, '/sources')
    const menu = () => page.locator('.app-sidebar').getByRole('button', { name: 'Create' })
    await expect(menu()).toBeVisible({ timeout: 60_000 })
    await roundTrip(page, { item: 'Notebook', menu, layer: dialog(page, /Create New Notebook/) })
    await roundTrip(page, { item: 'Podcast', menu, layer: dialog(page, /Generate Podcast Episode/) })
  })

  test('the podcast profiles ⋮: Delete', async ({ page }) => {
    await backendWithMenus(page)
    await gotoAndWait(page, '/podcasts?tab=templates')
    await page.getByRole('tab', { name: /Templates/ }).click({ timeout: 60_000 })
    const episode = () => page.getByRole('button', { name: 'Actions for Briefing' })
    await expect(episode()).toBeVisible({ timeout: 30_000 })
    await roundTrip(page, { item: 'Delete', menu: episode, layer: alert(page) })
    const speaker = () => page.getByRole('button', { name: 'Actions for Solo' })
    await roundTrip(page, { item: 'Delete', menu: speaker, layer: alert(page) })
  })
})
