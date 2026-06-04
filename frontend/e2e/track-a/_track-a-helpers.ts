import type { Page } from '@playwright/test'

/**
 * Shared route-mocking helpers for Track A (MinerU) Playwright specs.
 *
 * Every spec relies on these to short-circuit the auth + model-status
 * checks done by the dashboard layout. That way the specs can boot
 * straight into the page under test without depending on backend
 * configuration.
 *
 * IMPORTANT: We patch the API endpoints used by the dashboard shell:
 *   - GET /api/auth/status   → auth_enabled: false (auto-authenticated)
 *   - GET /api/models/status → valid: true (skip /models redirect)
 *   - GET /api/config        → minimal config response
 *
 * Specs additionally route their own feature-specific endpoints.
 */

export async function mockDashboardChrome(page: Page): Promise<void> {
  // Pre-seed the auth store with the "auth not required" shape so the
  // dashboard layout treats the user as authenticated from the very
  // first render. Avoids the race where the layout redirects to
  // /login before the auth/status mock has a chance to respond.
  await page.addInitScript(() => {
    try {
      window.localStorage.setItem(
        'auth-storage',
        JSON.stringify({
          state: {
            isAuthenticated: true,
            token: 'not-required',
            isLoading: false,
            error: null,
            lastAuthCheck: Date.now(),
            isCheckingAuth: false,
            hasHydrated: false,
            authRequired: false,
          },
          version: 0,
        })
      )
    } catch {
      // localStorage may be unavailable in some test contexts; the
      // route mock below is the fallback.
    }
  })

  await page.route('**/api/auth/status', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ auth_enabled: false }),
    })
  )

  await page.route('**/api/models/status', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ valid: true, message: 'ok' }),
    })
  )

  // /api/config is fetched once on page load by the frontend config layer.
  await page.route('**/api/config', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        version: 'e2e',
        latestVersion: null,
        hasUpdate: false,
        dbStatus: { connected: true },
      }),
    })
  )
}

export const DEFAULT_SETTINGS = {
  parser_engine: 'docling',
  docling_min_confidence: 0.95,
  default_content_processing_engine_url: 'auto',
  default_embedding_option: 'ask',
  auto_delete_files: 'yes',
  docling_gpu_enabled: false,
  docling_gpu_device: 'auto',
  docling_pipeline: 'standard',
  docling_vlm_model: 'granite-docling-258m',
  docling_vlm_framework: 'auto',
  docling_ocr_engine: 'easyocr',
  docling_ocr_use_gpu: false,
  docling_table_mode: 'accurate',
  docling_auto_export_images: false,
  docling_image_scale: 2.0,
  docling_chunking_enabled: true,
  docling_chunking_method: 'hybrid',
  docling_chunking_max_tokens: 512,
  input_directory_path: '',
  markdown_directory_path: '',
  output_directory_path: '',
  file_operation: 'copy',
  output_naming_scheme: 'original',
  vault_path: '',
  vault_entities_folder: 'Entities',
  vault_sync_on_startup: false,
  mineru_supported_extensions: ['.pdf', '.docx'],
}
