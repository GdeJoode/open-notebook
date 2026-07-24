import { expect, test } from '@playwright/test'
import { gotoAndWait, isOnLoginPage } from '../_helpers'

/**
 * Track G.4 — API Keys settings tab.
 *
 * Flow: open Settings → API Keys tab → generate a key → the plaintext is shown
 * ONCE (copyable) → revoke it → the revoked key no longer authenticates against
 * the agent capability API (`GET /api/v1/agents/openapi.json` → 401).
 *
 * Skips gracefully when the stack requires login (no seeded session), matching
 * the smoke-test convention — the deterministic assertions run on an auth-off
 * stack (the default CI boot).
 */
test.describe('track-g: API Keys settings tab', () => {
  test('generate → reveal once → revoke → key rejected', async ({
    page,
    request,
  }) => {
    await gotoAndWait(page, '/settings')
    if (await isOnLoginPage(page)) {
      test.skip(true, 'stack requires login; no seeded session for this spec')
      return
    }

    // Open the API Keys tab.
    await page.getByRole('tab', { name: 'API Keys' }).click()

    // Generate a key.
    await page.getByTestId('api-keys-generate').click()
    const agentId = `e2e-bot-${Date.now()}`
    await page.getByLabel('Agent ID').fill(agentId)
    await page.getByTestId('api-key-generate-submit').click()

    // The plaintext is revealed exactly once.
    const plaintext = await page
      .getByTestId('api-key-plaintext')
      .textContent()
    expect(plaintext).toMatch(/^ak_/)

    // The key authenticates against the agent API before revocation.
    const okResp = await request.get('/api/v1/agents/openapi.json', {
      headers: { 'X-API-Key': plaintext as string },
    })
    expect(okResp.status()).toBe(200)

    // Close the reveal; the plaintext is gone and never shown again.
    await page.getByRole('button', { name: 'Done' }).click()
    await expect(page.getByTestId('api-key-plaintext')).toHaveCount(0)

    // Revoke the key (confirm in the alert dialog).
    const row = page.locator(`[data-testid^="api-key-row-"]`, {
      hasText: agentId,
    })
    await row.getByRole('button', { name: `Revoke key for ${agentId}` }).click()
    await page.getByRole('button', { name: 'Revoke' }).click()

    // The revoked key no longer authenticates.
    await expect(async () => {
      const resp = await request.get('/api/v1/agents/openapi.json', {
        headers: { 'X-API-Key': plaintext as string },
      })
      expect(resp.status()).toBe(401)
    }).toPass({ timeout: 5000 })
  })
})
