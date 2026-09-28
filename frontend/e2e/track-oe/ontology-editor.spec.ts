/**
 * Track OE — Phase OE.3 — the shared-ontology editor, end to end.
 *
 * AC (docs/tracks/OE-ontology-editor/plan.md §OE.3): rename → preview →
 * publish → rollback, plus the owner's drag-and-drop re-parenting (and Ctrl+X /
 * Ctrl+V as the keyboard way).
 *
 * Route-mocked, like the project's other E2E specs: a small in-memory backend
 * answers the OE.2 API. It reproduces the contract the review found the
 * first double hid (review OE.3 M7): relationships stored as
 * `source_types`/`target_types`, validation that blocks a parent cycle, 409 on
 * a stale draft and 422 on a blocking one, a rollback that leaves the draft
 * alone (so it goes stale), and a rename that appends the old name to the
 * aliases and moves this ontology's references. Nothing touches a database.
 *
 * OE.4 adds the retype (`/api/ontologies/retype/...`), modelled on
 * `retype_planner.plan_rows` and `EntityRetypeService`: entities keep the type
 * they were written with; the plan types each row's label under the ACTIVE
 * content (name or alias, up the parent chain to a base) and is conservative
 * (every current identity stays held, so a move onto a held `(name_key, type)`
 * is a collision); the preview and publish count the same plan; an apply
 * writes only when the fingerprint it echoes is the current plan's (409
 * otherwise) and returns a rollback id to undo it.
 *
 * The backend lives in `_ontology-backend.ts`, shared with the OE.5 spec
 * (accepting a notebook's proposal publishes into the same ontologies).
 */

import { expect, test, type Page } from '@playwright/test'

import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'
import { backend } from './_ontology-backend'

async function openEditor(page: Page) {
  const api = backend()
  await mockDashboardChrome(page)
  await page.route('**/api/ontologies**', api.handle)
  await gotoAndWait(page, '/ontologies')
  await page.getByText('deals', { exact: true }).first().click()
  await page.getByTestId('ontology-edit-tab').click()
  await expect(page.getByTestId('ontology-editor')).toBeVisible()
  return api
}

const typeName = (page: Page, name: string) =>
  page.getByTestId(`ontology-type-${name}`).getByRole('button', { name, exact: true })

async function confirm(page: Page) {
  await page.getByTestId('confirm-action').click()
}

test.describe('OE.3 — ontology editor', () => {
  test('drag a type onto its new parent', async ({ page }) => {
    const api = await openEditor(page)
    const handle = page.getByRole('button', { name: 'Move Project' })
    const target = page.getByTestId('ontology-type-Deal')
    await handle.hover()
    await page.mouse.down()
    const box = await target.boundingBox()
    if (!box) throw new Error('no target box')
    await page.mouse.move(box.x + 20, box.y + box.height / 2, { steps: 12 })
    await page.mouse.move(box.x + 24, box.y + box.height / 2, { steps: 4 })
    await page.mouse.up()

    await expect.poll(() => api.state.calls.find((c) => c.method === 'PUT')?.body).toEqual({
      name: 'Project', parent_type: 'Deal', graph_relevance: 'active',
    })
    await expect(page.getByTestId('ontology-type-Deal').locator('xpath=..')
      .getByTestId('ontology-type-Project')).toBeVisible()
    await expect(page.getByTestId('ontology-draft-state')).toHaveText('Draft')
  })

  test('the keyboard can move a type too', async ({ page }) => {
    const api = await openEditor(page)
    const handle = page.getByRole('button', { name: 'Move Project' })
    await handle.focus()
    await page.keyboard.press('Space')
    await expect(handle).toHaveAttribute('aria-pressed', 'true') // picked up
    // Deal is listed right above Project under the same parent. The keyboard
    // sensor moves in fixed steps; step up until Deal is the drop target (the
    // target is resolved asynchronously, so wait on it each time).
    const deal = page.getByTestId('ontology-type-Deal')
    for (let step = 0; step < 3; step++) {
      await page.keyboard.press('ArrowUp')
      const over = await deal.evaluate((el) => new Promise<boolean>((resolve) =>
        setTimeout(() => resolve(el.className.includes('bg-primary/10')), 250)))
      if (over) break
    }
    await expect(deal).toHaveClass(/bg-primary\/10/)
    await page.keyboard.press('Space')
    await expect.poll(() => api.state.calls.find((c) => c.method === 'PUT')?.body).toEqual({
      name: 'Project', parent_type: 'Deal', graph_relevance: 'active',
    })
  })

  test('cut and paste a type with Ctrl+X / Ctrl+V', async ({ page }) => {
    const api = await openEditor(page)
    await typeName(page, 'Project').focus()
    await page.keyboard.press('Control+x')
    await expect(page.getByTestId('ontology-cut-state')).toContainText('Project cut')
    await typeName(page, 'Deal').focus()
    await page.keyboard.press('Control+v')
    await expect.poll(() => api.state.calls.find((c) => c.method === 'PUT')?.body).toEqual({
      name: 'Project', parent_type: 'Deal', graph_relevance: 'active',
    })
    await expect(page.getByTestId('ontology-cut-state')).toHaveCount(0)
  })

  test('Ctrl+V on "No parent" makes a type a root; Escape cancels a cut', async ({ page }) => {
    const api = await openEditor(page)
    await typeName(page, 'Project').focus()
    await page.keyboard.press('Control+x')
    await page.keyboard.press('Escape')
    await expect(page.getByTestId('ontology-cut-state')).toHaveCount(0)
    await typeName(page, 'Project').focus()
    await page.keyboard.press('Control+x')
    await page.getByTestId('ontology-root-zone').focus()
    await page.keyboard.press('Control+v')
    await expect.poll(() => api.state.calls.find((c) => c.method === 'PUT')?.body).toEqual({
      name: 'Project', graph_relevance: 'active',
    })
  })

  test('rename → preview → publish → rollback', async ({ page }) => {
    const api = await openEditor(page)
    await typeName(page, 'Project').click()
    await page.getByLabel('New name').fill('Programmaproject')
    await page.getByTestId('ontology-type-rename').click()
    await expect(page.getByTestId('ontology-type-Programmaproject')).toBeVisible()
    // The rename moved the relationship's reference: the row shows the new name.
    await expect(page.getByLabel('BEVAT_PROJECT range')).toHaveValue('Programmaproject')

    // Re-parent it too, so the preview has a type change to show.
    await typeName(page, 'Programmaproject').click()
    await page.getByLabel('Parent type').fill('Deal')
    await page.getByTestId('ontology-type-save').click()
    await page.getByTestId('ontology-preview').click()
    await expect(page.getByTestId('ontology-preview-labels')).toContainText('Programmaproject')

    await page.getByLabel('Publish note').fill('rename Project')
    await page.getByTestId('ontology-publish').click()
    const versions = page.getByTestId('ontology-versions')
    await expect(versions).toContainText('rename Project')
    await expect(page.getByTestId('ontology-draft-state')).toHaveText('No draft')
    expect(api.state.active.content.types.Programmaproject.aliases).toEqual(['Project'])

    await page.getByTestId('ontology-rollback-1.0.0').click()
    await confirm(page)
    await expect.poll(() => api.state.active.version).toBe('1.0.0')
    await expect(page.getByTestId('ontology-type-Project')).toBeVisible()
  })

  test('an inherited type names the ontology that defines it (round-2 major 2)', async ({ page }) => {
    await openEditor(page)
    await expect(page.getByTestId('ontology-type-Person')).toContainText('from schema_core')
    await expect(page.getByTestId('ontology-type-Subsidie')).toContainText('from instruments')
  })

  test('a save does not undo a drag of the selected type (round-2 major 1)', async ({ page }) => {
    const api = await openEditor(page)
    await typeName(page, 'Project').click()
    await page.getByLabel('Description', { exact: true }).fill('beschrijving')
    await typeName(page, 'Project').focus()
    await page.keyboard.press('Control+x')
    await typeName(page, 'Deal').focus()
    await page.keyboard.press('Control+v')
    await expect(page.getByLabel('Parent type')).toHaveValue('Deal')
    await page.getByTestId('ontology-type-save').click()
    await expect.poll(() => api.state.calls.filter((c) => c.method === 'PUT').at(-1)?.body).toEqual({
      name: 'Project', parent_type: 'Deal', graph_relevance: 'active', description: 'beschrijving',
    })
  })

  test('editing a relationship keeps its source_types/target_types (review B1)', async ({ page }) => {
    const api = await openEditor(page)
    await page.getByLabel('BEVAT_PROJECT description').fill('Een deal bevat projecten')
    await page.getByTestId('ontology-relationship-BEVAT_PROJECT').getByRole('button', { name: 'Save' }).click()
    await expect.poll(() => api.state.calls.find((c) => c.method === 'PUT')?.body).toEqual({
      name: 'BEVAT_PROJECT', description: 'Een deal bevat projecten',
      source_types: ['Deal'], target_types: ['Project'],
    })
  })

  test('a parent cycle stays visible and blocks the publish (review B2)', async ({ page }) => {
    await openEditor(page)
    await typeName(page, 'Deal').click()
    await page.getByLabel('Parent type').fill('Project')
    await page.getByTestId('ontology-type-save').click()
    await typeName(page, 'Project').click()
    await page.getByLabel('Parent type').fill('Deal')
    await page.getByTestId('ontology-type-save').click()
    await expect(page.getByRole('alert').first()).toContainText('parent_cycle')
    await expect(page.getByTestId('ontology-type-Deal')).toBeVisible()
    await expect(page.getByTestId('ontology-type-Project')).toBeVisible()
    await expect(page.getByTestId('ontology-publish')).toBeDisabled()
  })

  test('a rollback makes the open draft stale: warned, not publishable (review M5)', async ({ page }) => {
    const api = await openEditor(page)
    // Publish once, so there is a version to roll back to.
    await typeName(page, 'Project').click()
    await page.getByLabel('Description', { exact: true }).fill('eerste')
    await page.getByTestId('ontology-type-save').click()
    await page.getByTestId('ontology-publish').click()
    await expect(page.getByTestId('ontology-draft-state')).toHaveText('No draft')
    // Start a new draft, then roll back underneath it.
    await typeName(page, 'Project').click()
    await page.getByLabel('Description', { exact: true }).fill('tweede')
    await page.getByTestId('ontology-type-save').click()
    await expect(page.getByTestId('ontology-draft-state')).toHaveText('Draft')
    await page.getByTestId('ontology-rollback-1.0.0').click()
    await confirm(page)
    await expect(page.getByTestId('ontology-draft-stale')).toBeVisible()
    await expect(page.getByTestId('ontology-publish')).toBeDisabled()
    expect(api.state.draft).not.toBeNull() // the edits are kept until the user discards
  })
})

test.describe('OE.4 — the retype after a publish', () => {
  async function publishProjectAsRoot(page: Page) {
    await typeName(page, 'Project').focus()
    await page.keyboard.press('Control+x')
    await page.getByTestId('ontology-root-zone').focus()
    await page.keyboard.press('Control+v')
    await expect(page.getByTestId('ontology-draft-state')).toHaveText('Draft')
    await page.getByTestId('ontology-publish').click()
  }

  test('publish → dry run → apply → the dry run is empty, the collision is left', async ({ page }) => {
    const api = await openEditor(page)
    await publishProjectAsRoot(page)

    // The publish reported entities it moves, so the dry run is offered.
    const panel = page.getByTestId('ontology-retype')
    await expect(panel.getByTestId('ontology-retype-changes')).toContainText('Regio Deal Twente')
    await expect(panel.getByTestId('ontology-retype-changes')).toContainText('programme → concept')
    const collisions = panel.getByTestId('ontology-retype-collisions')
    await expect(collisions).toContainText('held by entity:3')
    await expect(collisions.getByRole('link', { name: 'resolution page' }))
      .toHaveAttribute('href', '/knowledge-graph/resolution')

    await panel.getByTestId('ontology-retype-apply').click()
    await confirm(page)
    await expect(panel.getByTestId('ontology-retype-empty')).toBeVisible()
    await expect(panel.getByTestId('ontology-retype-result')).toContainText('Written 1')
    await expect(collisions).toContainText('Groeiagenda') // left for the curator
    expect(Object.fromEntries(api.state.entities.map((e) => [e.id, e.entity_type]))).toEqual({
      'entity:1': 'concept', 'entity:2': 'programme', 'entity:3': 'concept', 'entity:4': 'programme',
    })

    await panel.getByTestId('ontology-retype-undo').click()
    await confirm(page)
    await expect(panel.getByTestId('ontology-retype-changes')).toContainText('Regio Deal Twente')
    expect(api.state.entities.find((e) => e.id === 'entity:1')?.entity_type).toBe('programme')
  })

  test('an apply of a plan that changed since the dry run writes nothing and shows the new plan',
    async ({ page }) => {
      const api = await openEditor(page)
      await publishProjectAsRoot(page)
      const panel = page.getByTestId('ontology-retype')
      await expect(panel.getByTestId('ontology-retype-changes')).toContainText('Regio Deal Twente')
      // Extracted meanwhile, under the old typing.
      api.state.entities.push({ id: 'entity:5', name: 'Fonds West', name_key: 'fonds west',
        label: 'Project', entity_type: 'programme' })

      await panel.getByTestId('ontology-retype-apply').click()
      await confirm(page)
      await expect(panel.getByTestId('ontology-retype-changes')).toContainText('Fonds West')
      expect(api.state.entities.filter((e) => e.entity_type === 'concept').map((e) => e.id))
        .toEqual(['entity:3'])

      await panel.getByTestId('ontology-retype-apply').click()
      await confirm(page)
      await expect(panel.getByTestId('ontology-retype-empty')).toBeVisible()
    })

  test('the retype is there on demand, and says when there is nothing to do', async ({ page }) => {
    await openEditor(page)
    await page.getByTestId('ontology-retype-open').click()
    await expect(page.getByTestId('ontology-retype-empty')).toBeVisible()
    await expect(page.getByTestId('ontology-retype-apply')).toHaveCount(0)
  })
})
