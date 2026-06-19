import { describe, expect, it } from 'vitest'
import { DEFAULT_PIPELINE_CONFIG } from './sources'

/**
 * Unit tests for the Phase I.D-2 docling-config defaults (AC3).
 *
 * The five conversion toggles must default to today's effective values so
 * existing users see no behaviour change. Picture classification is the
 * subtle one: it must stay *undefined* in the default config so the backend
 * keeps it coupled to the VLM pipeline toggle until the user opts in.
 */
describe('DEFAULT_PIPELINE_CONFIG (I.D-2)', () => {
  it('keeps enrichment on by default', () => {
    expect(DEFAULT_PIPELINE_CONFIG.docling_do_code_enrichment).toBe(true)
    expect(DEFAULT_PIPELINE_CONFIG.docling_do_formula_enrichment).toBe(true)
  })

  it('keeps full-page images off by default', () => {
    expect(DEFAULT_PIPELINE_CONFIG.docling_generate_page_images).toBe(false)
  })

  it('keeps image scale at 2.0', () => {
    expect(DEFAULT_PIPELINE_CONFIG.docling_image_scale).toBe(2.0)
  })

  it('leaves picture classification undefined so it follows the VLM toggle', () => {
    expect(
      DEFAULT_PIPELINE_CONFIG.docling_do_picture_classification
    ).toBeUndefined()
  })
})
