/**
 * Eén model uit het modelregister (`config/modelregister.yaml`, ADR 0007).
 * `GET /api/models` toont het register; aanpassen gaat met een commit.
 */
export interface Model {
  naam: string
  versie: string
  aanroepwijze: 'llama-swap' | 'api' | 'cli'
  locatie: 'lokaal' | 'cloud'
  toegestaan_intern: 'ja' | 'nee'
  taken: string[]
}

export type ModelType = 'language' | 'embedding' | 'text_to_speech' | 'speech_to_text'
