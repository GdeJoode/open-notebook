/**
 * Unit tests for the provider-label / linkify helpers behind `ExternalIdBadges`
 * (Track K.6, AC6). Notably pins that a bare `10.xxxx` DOI is labelled 'DOI'
 * rather than falling through to the generic 'ID'.
 */

import { describe, expect, it } from 'vitest'

import { providerLabel, toHref } from './external-ids'

describe('providerLabel', () => {
  it('labels a doi.org URL as DOI', () => {
    expect(providerLabel('https://doi.org/10.1000/xyz')).toBe('DOI')
  })

  it('labels a doi: scheme as DOI', () => {
    expect(providerLabel('doi:10.1000/xyz')).toBe('DOI')
  })

  it('labels a bare 10.xxxx DOI as DOI (no fall-through to ID)', () => {
    expect(providerLabel('10.1000/xyz')).toBe('DOI')
  })

  it('labels TOOI overheid identifiers', () => {
    expect(
      providerLabel(
        'https://identifier.overheid.nl/tooi/id/ministerie/mnre1034',
      ),
    ).toBe('TOOI')
  })

  it('labels ORCID, Wikidata, and arXiv', () => {
    expect(providerLabel('https://orcid.org/0000-0002-1825-0097')).toBe('ORCID')
    expect(providerLabel('https://www.wikidata.org/wiki/Q42')).toBe('Wikidata')
    expect(providerLabel('https://arxiv.org/abs/1234.5678')).toBe('arXiv')
  })

  it('falls back to ID for an unrecognised uri', () => {
    expect(providerLabel('https://example.com/thing/42')).toBe('ID')
  })
})

describe('toHref', () => {
  it('passes http(s) urls through unchanged', () => {
    expect(toHref('https://doi.org/10.1000/xyz')).toBe(
      'https://doi.org/10.1000/xyz',
    )
  })

  it('links a bare 10.xxxx DOI via doi.org', () => {
    expect(toHref('10.1000/xyz')).toBe('https://doi.org/10.1000/xyz')
  })

  it('links a doi: scheme via doi.org', () => {
    expect(toHref('doi:10.1000/xyz')).toBe('https://doi.org/10.1000/xyz')
  })
})
