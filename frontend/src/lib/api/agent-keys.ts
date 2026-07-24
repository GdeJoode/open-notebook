import apiClient from './client'

/** Agent-key permission scopes, least→most privileged (read < write < admin). */
export type AgentPermission = 'read' | 'write' | 'admin'

/** A key as listed back — secret-free (prefix + metadata, never the plaintext). */
export interface AgentKey {
  id: string
  agent_id: string
  key_prefix: string
  label?: string | null
  permission: AgentPermission
  rate_limit_rpm?: number | null
  revoked: boolean
  created?: string | null
  last_used_at?: string | null
}

/** The mint response — carries the plaintext key EXACTLY ONCE. */
export interface AgentKeyCreated extends AgentKey {
  key: string
}

export interface MintAgentKeyInput {
  agent_id: string
  permission: AgentPermission
  label?: string
  rate_limit_rpm?: number
}

/** One audit-log row for an agent's API calls. */
export interface AgentAuditEntry {
  agent_id?: string
  method?: string
  path?: string
  status?: number
  latency_ms?: number | null
  job_id?: string | null
  ts?: string | null
}

export interface KeyAuditLog {
  key_id: string
  agent_id: string
  entries: AgentAuditEntry[]
}

/**
 * Session-authed agent-key management (Track G.4). These call the operator
 * surface under `/api/agent-keys` (shared-password protected), NOT the
 * `X-API-Key` agent capability routes.
 */
export const agentKeysApi = {
  list: async (): Promise<AgentKey[]> => {
    const res = await apiClient.get<AgentKey[]>('/agent-keys')
    return res.data
  },

  mint: async (input: MintAgentKeyInput): Promise<AgentKeyCreated> => {
    const res = await apiClient.post<AgentKeyCreated>('/agent-keys', input)
    return res.data
  },

  revoke: async (id: string): Promise<void> => {
    await apiClient.delete(`/agent-keys/${id}`)
  },

  auditLog: async (id: string, limit = 100): Promise<KeyAuditLog> => {
    const res = await apiClient.get<KeyAuditLog>(`/agent-keys/${id}/audit-log`, {
      params: { limit },
    })
    return res.data
  },
}
