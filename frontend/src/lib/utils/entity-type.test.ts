import { describe, expect, it } from 'vitest'

import { entityTypeLabels } from './entity-type'

describe('entityTypeLabels (WB.3)', () => {
  it('puts the canonical type first and the display label second', () => {
    expect(entityTypeLabels('administrative_area', 'Gemeente')).toEqual({
      canonical: 'administrative_area', label: 'Gemeente',
    })
    // The triage table's case: the row stored the display label.
    expect(entityTypeLabels('creative_work', 'Journal/ Publication')).toEqual({
      canonical: 'creative_work', label: 'Journal/ Publication',
    })
  })

  it('drops a label that only restates the canonical type', () => {
    expect(entityTypeLabels('organization', 'Organization').label).toBeNull()
    expect(entityTypeLabels('government_organization', 'GovernmentOrganization').label).toBeNull()
    expect(entityTypeLabels('person', undefined).label).toBeNull()
  })

  it('never promotes the label when the canonical type is unknown', () => {
    expect(entityTypeLabels(undefined, 'Organisatie')).toEqual({ canonical: '', label: 'Organisatie' })
  })
})
