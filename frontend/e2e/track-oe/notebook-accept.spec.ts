/**
 * Track OE — Phase OE.5 — the per-notebook layer retired.
 *
 * AC (docs/tracks/OE-ontology-editor/plan.md §OE.5): a notebook's Pass-1
 * proposal is accepted into the SHARED ontology through the editor's gate
 * (previewed; a blocking candidate refused with its findings, nothing stored),
 * and the Schema tab routes edits to the ontology editor instead of writing
 * per-notebook edits.
 *
 * The in-memory backend (`_ontology-backend.ts`) is the one the editor spec
 * uses, so an accepted type lands in the ontology the editor then shows.
 */

import { expect, test, type Page } from '@playwright/test'

import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'
import { NOTEBOOK, backend } from './_ontology-backend'

async function openSchemaTab(page: Page) {
  const api = backend()
  await mockDashboardChrome(page)
  await page.route('**/api/ontologies**', api.handle)
  await page.route(/\/api\/notebooks\/(?:notebook:oe5|notebook%3Aoe5)(?:[/?].*)?$/, api.handleNotebook)
  await gotoAndWait(page, `/notebooks/${NOTEBOOK}/schema`)
  await expect(page.getByTestId('pending-extensions-panel')).toBeVisible({ timeout: 45_000 })
  return api
}

test.describe('OE.5 — a notebook proposal goes to the shared ontology', () => {
  test.setTimeout(90_000)

  test('Accept → preview → publish: the type is in the shared ontology', async ({ page }) => {
    const api = await openSchemaTab(page)
    await page.getByTestId('accept-pass1::streekfonds').click()
    const dialog = page.getByTestId('accept-extension-dialog')
    await expect(dialog.getByTestId('accept-ontology')).toHaveValue('deals')
    await expect(dialog.getByTestId('accept-parent')).toHaveValue('Deal')
    await expect(dialog.getByTestId('accept-publish')).toBeDisabled()
    await dialog.getByTestId('accept-preview').click()
    await expect(dialog.getByTestId('accept-preview-clean')).toBeVisible()
    await dialog.getByTestId('accept-publish').click()
    await expect(dialog).toBeHidden()

    // Out of the queue; published as a new active version of `deals`.
    await expect(page.getByTestId('pending-extension-Streekfonds')).toHaveCount(0)
    await expect(page.getByTestId('pending-extension-Zwerfbegrip')).toBeVisible()
    expect(api.state.active.content.types.Streekfonds).toMatchObject({ parent_type: 'Deal' })
    expect(api.state.versions.find((v) => v.is_active)?.version).toBe(api.state.active.version)
    // The Schema tab lists it with the base's other types, linked to the editor.
    const edit = page.getByTestId('schema-row-edit-Streekfonds')
    await expect(edit).toHaveAttribute('href', '/ontologies?name=deals&tab=edit&type=Streekfonds')

    // The editor shows the published type, opened on it.
    await edit.click()
    await expect(page.getByTestId('ontology-editor')).toBeVisible({ timeout: 45_000 })
    await expect(page.getByTestId('ontology-type-form').getByRole('heading'))
      .toHaveText('Streekfonds')
  })

  test('a blocking candidate is refused with its findings, and nothing is stored', async ({ page }) => {
    const api = await openSchemaTab(page)
    const before = { version: api.state.active.version, versions: api.state.versions.length }
    await page.getByTestId('accept-pass1::zwerfbegrip').click()
    const dialog = page.getByTestId('accept-extension-dialog')
    // `Nergens` is defined nowhere, so the default is the fallback root.
    await expect(dialog.getByTestId('accept-ontology')).toHaveValue('schema_core')
    await dialog.getByTestId('accept-ontology').selectOption('deals')
    await dialog.getByTestId('accept-preview').click()
    await expect(dialog.getByTestId('accept-preview-blocking')).toContainText('orphan')
    await expect(dialog.getByTestId('accept-publish')).toBeDisabled()

    // The server refuses too, whatever the button says: a direct POST is a 422.
    const status = await page.evaluate(async (url) => (await fetch(url, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ontology: 'deals' }) })).status,
    `/api/notebooks/${encodeURIComponent(NOTEBOOK)}/schema/extensions/Zwerfbegrip/accept`)
    expect(status).toBe(422)
    expect(api.state.active.version).toBe(before.version)
    expect(api.state.versions.length).toBe(before.versions)
    expect(api.state.active.content.types.Zwerfbegrip).toBeUndefined()
    expect(api.state.pending.map((p) => p.type_name)).toContain('Zwerfbegrip')

    // A parent that reaches a base clears the finding and publishes.
    await dialog.getByTestId('accept-parent').fill('Deal')
    await expect(dialog.getByTestId('accept-publish')).toBeDisabled() // the old preview is dropped
    await dialog.getByTestId('accept-preview').click()
    await expect(dialog.getByTestId('accept-preview-clean')).toBeVisible()
    await dialog.getByTestId('accept-publish').click()
    await expect(dialog).toBeHidden()
    expect(api.state.active.content.types.Zwerfbegrip).toMatchObject({ parent_type: 'Deal' })
  })

  test('an open draft of the ontology refuses the accept', async ({ page }) => {
    const api = await openSchemaTab(page)
    api.state.draft = { content: JSON.parse(JSON.stringify(api.state.active.content)),
      base: api.state.active.version }
    await page.getByTestId('accept-pass1::streekfonds').click()
    const dialog = page.getByTestId('accept-extension-dialog')
    await dialog.getByTestId('accept-preview').click()
    await expect(dialog.getByTestId('accept-draft-open')).toBeVisible()
    await expect(dialog.getByTestId('accept-publish')).toBeDisabled()
    expect(api.state.active.content.types.Streekfonds).toBeUndefined()
  })

  test('a proposal already shared is marked covered: no publish, no rejection', async ({ page }) => {
    const api = await openSchemaTab(page)
    const before = { version: api.state.active.version, versions: api.state.versions.length }
    await page.getByTestId('accept-pass1::subsidie').click()
    const dialog = page.getByTestId('accept-extension-dialog')
    await expect(dialog.getByTestId('accept-already-defined')).toContainText('instruments')
    await expect(dialog.getByTestId('accept-publish')).toHaveCount(0)
    await dialog.getByTestId('accept-mark-covered').click()
    await expect(dialog).toBeHidden()
    await expect(page.getByTestId('pending-extension-subsidie')).toHaveCount(0)
    expect(api.state.accepted.find((a) => a.type_name === 'subsidie'))
      .toMatchObject({ covered: true, published_to: 'instruments' })
    expect(api.state.rejected).toEqual([])
    expect(api.state.active.version).toBe(before.version)
    expect(api.state.versions.length).toBe(before.versions)
  })

  test('the Schema tab has no per-notebook edit left', async ({ page }) => {
    await openSchemaTab(page)
    await expect(page.getByTestId('schema-row-edit-Deal')).toBeVisible()
    await expect(page.getByTestId('schema-row-actions-Deal')).toHaveCount(0)
    await expect(page.getByTestId('schema-open-editor'))
      .toHaveAttribute('href', '/ontologies?name=deals&tab=edit')
  })
})
