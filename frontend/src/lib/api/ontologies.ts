import apiClient from './client'

export interface OntologySummary {
  name: string
  version: string
  description: string | null
  entity_type_count: number
  relationship_type_count: number
  concept_count: number
}

export interface PropertyDef {
  name: string
  description: string | null
  data_type: string
  required: boolean
  examples: string[] | null
}

export interface EntityTypeDef {
  name: string
  description: string | null
  extraction_hints: string[] | null
  examples: string[] | null
  aliases: string[] | null
  parent_type: string | null
  properties: PropertyDef[]
}

export interface RelationshipTypeDef {
  name: string
  description: string | null
  domain: string[]
  range: string[]
  cardinality: string
  symmetric: boolean
  transitive: boolean
  inverse: string | null
  extraction_hints: string[] | null
}

export interface ConceptDef {
  name: string
  description: string | null
  related_concepts: string[] | null
  examples: string[] | null
}

export interface OntologyDetail extends OntologySummary {
  authors: string[] | null
  extends: string | null
  entity_types: EntityTypeDef[]
  relationship_types: RelationshipTypeDef[]
  concepts: ConceptDef[]
}

export const ontologiesApi = {
  list: async () => {
    const response = await apiClient.get<OntologySummary[]>('/ontologies')
    return response.data
  },

  get: async (name: string) => {
    const response = await apiClient.get<OntologyDetail>(`/ontologies/${name}`)
    return response.data
  },

  getEntityTypes: async (name: string) => {
    const response = await apiClient.get<EntityTypeDef[]>(`/ontologies/${name}/entity-types`)
    return response.data
  },

  getRelationshipTypes: async (name: string) => {
    const response = await apiClient.get<RelationshipTypeDef[]>(`/ontologies/${name}/relationship-types`)
    return response.data
  },
}

// ---------------------------------------------------------------------------
// Editing (Track OE.2 API, used by the OE.3 editor)
// ---------------------------------------------------------------------------

/** An entity type as STORED: what the source set, plus keys the model keeps
 *  without a field (e.g. `graph_relevance`). Sent back whole on every edit so
 *  nothing is dropped. */
export type StoredEntityType = {
  name: string
  description?: string | null
  parent_type?: string | null
  schema_org_type?: string | null
  aliases?: string[] | null
  extraction_hints?: string[] | null
  examples?: string[] | null
  [key: string]: unknown
}

export type StoredRelationshipType = {
  name: string
  description?: string | null
  domain?: string[]
  range?: string[]
  [key: string]: unknown
}

export interface OntologyDraft {
  name: string
  /** False when no draft exists yet: `ontology` is then the active content. */
  exists: boolean
  base_version: string
  ontology: {
    metadata: {
      name: string
      version: string
      extends?: string | null
      includes?: string[] | null
    }
    entity_types?: Record<string, StoredEntityType>
    relationship_types?: Record<string, StoredRelationshipType>
    [key: string]: unknown
  }
}

export interface Finding {
  rule: string
  ontology: string | null
  subject: string | null
  detail: string
  severity: 'block' | 'warn'
  weight: number
}

export interface OntologyPreview {
  findings: Finding[]
  blocking: boolean
  label_changes: { label: string; before: string; after: string; entities: number }[]
  retype: {
    changes: number
    collisions: number
    flag_only: number
    caused_by_this_edit: number
    collisions_caused_by_this_edit: number
  }
  detection: { source: string; title: string | null; notebook: string | null; before: string[]; after: string[] }[]
  detection_error: string | null
  notebook_edits: number | null
}

export interface OntologyVersion {
  name: string
  version: string
  is_active: boolean
  created_at: string
  note: string | null
  created_by: string | null
}

export interface PublishResult {
  name: string
  version: string
  warnings: Finding[]
  retype: { changes: number; collisions: number; flag_only: number; caused_by_this_edit: number }
}

const enc = encodeURIComponent

export const ontologyEditApi = {
  getDraft: async (name: string) =>
    (await apiClient.get<OntologyDraft>(`/ontologies/${enc(name)}/draft`)).data,

  discardDraft: async (name: string) => {
    await apiClient.delete(`/ontologies/${enc(name)}/draft`)
  },

  putEntityType: async (name: string, type: StoredEntityType) =>
    (await apiClient.put<OntologyDraft>(
      `/ontologies/${enc(name)}/draft/entity-types/${enc(type.name)}`, type)).data,

  deleteEntityType: async (name: string, typeName: string) =>
    (await apiClient.delete<OntologyDraft>(
      `/ontologies/${enc(name)}/draft/entity-types/${enc(typeName)}`)).data,

  renameEntityType: async (name: string, typeName: string, newName: string) =>
    (await apiClient.post<OntologyDraft>(
      `/ontologies/${enc(name)}/draft/entity-types/${enc(typeName)}/rename`,
      { new_name: newName })).data,

  putRelationshipType: async (name: string, rel: StoredRelationshipType) =>
    (await apiClient.put<OntologyDraft>(
      `/ontologies/${enc(name)}/draft/relationship-types/${enc(rel.name)}`, rel)).data,

  deleteRelationshipType: async (name: string, relName: string) =>
    (await apiClient.delete<OntologyDraft>(
      `/ontologies/${enc(name)}/draft/relationship-types/${enc(relName)}`)).data,

  validate: async (name: string) =>
    (await apiClient.post<Finding[]>(`/ontologies/${enc(name)}/draft/validate`)).data,

  preview: async (name: string, includeDetection = true) =>
    (await apiClient.post<OntologyPreview>(`/ontologies/${enc(name)}/draft/preview`, null,
      { params: { include_detection: includeDetection } })).data,

  publish: async (name: string, note: string | null) =>
    (await apiClient.post<PublishResult>(`/ontologies/${enc(name)}/draft/publish`, { note })).data,

  versions: async (name: string) =>
    (await apiClient.get<OntologyVersion[]>(`/ontologies/${enc(name)}/versions`)).data,

  rollback: async (name: string, version: string) =>
    (await apiClient.post(`/ontologies/${enc(name)}/rollback/${enc(version)}`)).data,

  exportYaml: async (name: string) =>
    (await apiClient.get<string>(`/ontologies/${enc(name)}/export.yaml`,
      { responseType: 'text' })).data,
}
