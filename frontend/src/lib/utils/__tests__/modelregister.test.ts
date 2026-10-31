import { describe, expect, it } from 'vitest'
import { Model } from '@/lib/types/models'
import { modelVoorTaak, modellenVanSoort } from '../modelregister'

const model = (naam: string, taken: string[]): Model => ({
  naam,
  versie: 'v1',
  aanroepwijze: 'llama-swap',
  locatie: 'lokaal',
  toegestaan_intern: 'ja',
  taken,
})

const register = [model('werkmodel', ['chat', 'summarization']), model('embed', ['embedding'])]

describe('modelregister', () => {
  it('kiest het eerste model met de taak', () => {
    expect(modelVoorTaak(register, 'chat')?.naam).toBe('werkmodel')
    expect(modelVoorTaak(register, 'embedding')?.naam).toBe('embed')
    expect(modelVoorTaak(register, 'rerank')).toBeUndefined()
    expect(modelVoorTaak(undefined, 'chat')).toBeUndefined()
  })

  it('filtert op soort via de taken', () => {
    expect(modellenVanSoort(register, 'language').map((m) => m.naam)).toEqual(['werkmodel'])
    expect(modellenVanSoort(register, 'embedding').map((m) => m.naam)).toEqual(['embed'])
    expect(modellenVanSoort(register, 'text_to_speech')).toEqual([])
  })
})
