import { QueryClient } from '@tanstack/react-query'

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes
      retry: 2,
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 1,
    },
  },
})

export const QUERY_KEYS = {
  notebooks: ['notebooks'] as const,
  notebook: (id: string) => ['notebooks', id] as const,
  notes: (notebookId?: string) => ['notes', notebookId] as const,
  note: (id: string) => ['notes', id] as const,
  sources: (notebookId?: string) => ['sources', notebookId] as const,
  source: (id: string) => ['sources', id] as const,
  settings: ['settings'] as const,
  sourceChatSessions: (sourceId: string) => ['source-chat', sourceId, 'sessions'] as const,
  sourceChatSession: (sourceId: string, sessionId: string) => ['source-chat', sourceId, 'sessions', sessionId] as const,
  notebookChatSessions: (notebookId: string) => ['notebook-chat', notebookId, 'sessions'] as const,
  notebookChatSession: (sessionId: string) => ['notebook-chat', 'sessions', sessionId] as const,
  podcastEpisodes: ['podcasts', 'episodes'] as const,
  podcastEpisode: (episodeId: string) => ['podcasts', 'episodes', episodeId] as const,
  episodeProfiles: ['podcasts', 'episode-profiles'] as const,
  speakerProfiles: ['podcasts', 'speaker-profiles'] as const,
  ontologies: ['ontologies'] as const,
  ontology: (name: string) => ['ontologies', name] as const,
  entities: (filters?: Record<string, unknown>) => ['knowledge-graph', 'entities', filters] as const,
  entity: (id: string) => ['knowledge-graph', 'entities', id] as const,
  entityTypes: ['knowledge-graph', 'entity-types'] as const,
  graphData: (filters?: Record<string, unknown>) => ['knowledge-graph', 'graph', filters] as const,
  entitySearch: (query: string) => ['knowledge-graph', 'search', query] as const,
  strategies: ['summaries', 'strategies'] as const,
  summaries: (sourceId?: string) => ['summaries', sourceId] as const,
  summary: (id: string) => ['summaries', id] as const,
}
