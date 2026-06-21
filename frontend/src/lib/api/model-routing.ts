/**
 * Typed client for the /api/model-routes endpoints (Track J.5).
 *
 * Mirrors `embedding.ts` / `health.ts`: thin wrappers around `apiClient`
 * returning the response body. Error handling is left to the calling hook
 * (TanStack Query), matching the rest of the API layer.
 */

import apiClient from './client'
import {
  ModelRoute,
  ModelRouteHealth,
  ModelRouteUpdate,
  RoutingTask,
} from '@/lib/types/model-routing'

export const modelRoutingApi = {
  list: async (): Promise<ModelRoute[]> => {
    const response = await apiClient.get<ModelRoute[]>('/model-routes')
    return response.data
  },

  update: async (
    task: RoutingTask,
    update: ModelRouteUpdate,
  ): Promise<ModelRoute> => {
    const response = await apiClient.put<ModelRoute>(
      `/model-routes/${task}`,
      update,
    )
    return response.data
  },

  health: async (): Promise<ModelRouteHealth> => {
    const response = await apiClient.get<ModelRouteHealth>('/model-routes/health')
    return response.data
  },
}
