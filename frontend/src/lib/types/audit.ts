/**
 * Types for the Track F.1 per-notebook quality audit (mirrors the backend
 * `shared.models.audit` shapes).
 */

export type AuditSeverity = 'error' | 'warn' | 'info'

export interface AuditFinding {
  check_id: string
  severity: AuditSeverity
  title: string
  detail?: string | null
  subject?: string | null
  count?: number | null
}

export interface AuditRunResponse {
  notebook_id: string
  run_id: string
  findings: AuditFinding[]
  created?: string | null
}
