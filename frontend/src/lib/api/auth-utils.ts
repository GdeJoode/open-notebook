/**
 * Shared auth token utilities for streaming endpoints that bypass the apiClient interceptor.
 */

export function getAuthToken(): string | null {
  if (typeof window === 'undefined') return null
  const authStorage = localStorage.getItem('auth-storage')
  if (!authStorage) return null
  try {
    const { state } = JSON.parse(authStorage)
    return state?.token ?? null
  } catch {
    return null
  }
}

export function createAuthHeaders(token: string | null): Record<string, string> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (token) headers.Authorization = `Bearer ${token}`
  return headers
}
