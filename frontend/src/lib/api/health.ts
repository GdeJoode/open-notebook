/**
 * Typed client wrapper for the /api/health/* endpoints.
 *
 * The backend guarantees these endpoints never return 5xx (see
 * `apps/app-main/src/app_main/api/routers/health.py`), so callers
 * can rely on a `MineruHealthResponse` body in every case — even
 * when the underlying service is down.
 */

import apiClient from './client'
import { MineruHealthResponse } from '@/lib/types/api'

export const healthApi = {
  mineru: async (): Promise<MineruHealthResponse> => {
    const response = await apiClient.get<MineruHealthResponse>('/health/mineru')
    return response.data
  },
}
