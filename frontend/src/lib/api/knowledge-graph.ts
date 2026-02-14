import apiClient from './client'

export interface Entity {
  id: string
  name: string
  entity_type: string
  weight: number
}

export interface EntityDetail extends Entity {
  embedding?: number[]
  relations: EntityRelation[]
  [key: string]: unknown
}

export interface EntityRelation {
  id: string
  source: string
  target: string
  relation_type: string
}

export interface EntityTypeSummary {
  entity_type: string
  count: number
}

export interface PaginatedEntities {
  items: Entity[]
  total: number
  limit: number
  offset: number
}

export interface GraphNode {
  id: string
  label: string
  entity_type: string
  weight: number
}

export interface GraphEdge {
  source: string
  target: string
  label: string
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export const knowledgeGraphApi = {
  listEntities: async (params?: {
    limit?: number
    offset?: number
    entity_type?: string
  }) => {
    const response = await apiClient.get<PaginatedEntities>('/knowledge-graph/entities', { params })
    return response.data
  },

  getEntity: async (id: string) => {
    const response = await apiClient.get<EntityDetail>(`/knowledge-graph/entities/${id}`)
    return response.data
  },

  getEntityTypes: async () => {
    const response = await apiClient.get<EntityTypeSummary[]>('/knowledge-graph/entity-types')
    return response.data
  },

  getGraphData: async (params?: {
    entity_type?: string
    limit?: number
  }) => {
    const response = await apiClient.get<GraphData>('/knowledge-graph/graph', { params })
    return response.data
  },

  searchEntities: async (q: string, limit?: number) => {
    const response = await apiClient.get<Entity[]>('/knowledge-graph/search', {
      params: { q, limit },
    })
    return response.data
  },
}
