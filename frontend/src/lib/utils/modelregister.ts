/**
 * Hulpfuncties voor het modelregister (ADR 0007). Welk model een taak uitvoert,
 * staat in het register: het eerste model met die taak (zoals
 * `ModelRegister.voor_taak` in de fork).
 */
import { Model, ModelType } from '@/lib/types/models'

const TAKEN_PER_SOORT: Record<ModelType, string[]> = {
  language: ['chat', 'summarization', 'entity_extraction', 'entity_dedup'],
  embedding: ['embedding'],
  text_to_speech: [],
  speech_to_text: [],
}

export function modelVoorTaak(models: Model[] | undefined, taak: string): Model | undefined {
  return models?.find((model) => model.taken.includes(taak))
}

export function modellenVanSoort(models: Model[] | undefined, soort: ModelType): Model[] {
  const taken = TAKEN_PER_SOORT[soort]
  return (models ?? []).filter((model) => model.taken.some((taak) => taken.includes(taak)))
}
