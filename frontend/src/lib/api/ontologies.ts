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
