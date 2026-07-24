'use client'

import { useMemo } from 'react'
import { AlertTriangle, CheckCircle2, Info, Loader2, RefreshCw } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import {
  useNotebookAudit,
  useRunAudit,
  useRunDeepAudit,
} from '@/lib/hooks/use-audit'
import type { AuditFinding, AuditSeverity } from '@/lib/types/audit'

interface AuditWidgetProps {
  notebookId: string
}

const SEVERITY_ORDER: AuditSeverity[] = ['error', 'warn', 'info']

/** check_ids produced by the F.3 deep (LLM-derived) audit — badged in the UI. */
const DEEP_CHECK_IDS = new Set(['conflicting_facts', 'provenance_gaps'])

const SEVERITY_META: Record<
  AuditSeverity,
  { label: string; variant: 'destructive' | 'secondary' | 'outline' }
> = {
  error: { label: 'Errors', variant: 'destructive' },
  warn: { label: 'Warnings', variant: 'secondary' },
  info: { label: 'Info', variant: 'outline' },
}

/**
 * Per-notebook quality-audit widget (Track F.2).
 *
 * Reads `GET /api/notebooks/{id}/audit` (the latest F.1 run) and renders the
 * six LLM-free checks' findings grouped by severity, with a manual "Re-run"
 * that `POST`s a fresh run. Loading / error / empty states are explicit so the
 * widget always carries the "is this notebook healthy?" signal unambiguously.
 */
export function AuditWidget({ notebookId }: AuditWidgetProps) {
  const { data, isLoading, isError } = useNotebookAudit(notebookId)
  const runAudit = useRunAudit(notebookId)
  const runDeep = useRunDeepAudit(notebookId)

  const grouped = useMemo(() => {
    const g: Record<AuditSeverity, AuditFinding[]> = {
      error: [],
      warn: [],
      info: [],
    }
    for (const f of data?.findings ?? []) {
      if (f.severity in g) g[f.severity].push(f)
    }
    return g
  }, [data])

  const rerunButton = (
    <Button
      type="button"
      size="sm"
      variant="outline"
      onClick={() => runAudit.mutate()}
      disabled={runAudit.isPending}
      data-testid="audit-rerun"
      aria-label="Re-run the quality audit"
    >
      {runAudit.isPending ? (
        <Loader2 className="h-4 w-4 mr-2 animate-spin" aria-hidden="true" />
      ) : (
        <RefreshCw className="h-4 w-4 mr-2" aria-hidden="true" />
      )}
      {runAudit.isPending ? 'Running…' : 'Re-run'}
    </Button>
  )

  const deepButton = (
    <Button
      type="button"
      size="sm"
      variant="ghost"
      onClick={() => runDeep.mutate()}
      disabled={runDeep.isPending}
      data-testid="audit-run-deep"
      aria-label="Run the deep (LLM-derived) audit"
    >
      {runDeep.isPending ? (
        <Loader2 className="h-4 w-4 mr-2 animate-spin" aria-hidden="true" />
      ) : null}
      {runDeep.isPending ? 'Deep audit…' : 'Run deep audit'}
    </Button>
  )

  const header = (
    <div className="flex items-center justify-between">
      <div>
        <h3 className="text-sm font-semibold">Quality audit</h3>
        {data?.created && (
          <p className="text-xs text-muted-foreground" data-testid="audit-last-run">
            Last run {new Date(data.created).toLocaleString()}
          </p>
        )}
      </div>
      <div className="flex items-center gap-1">
        {deepButton}
        {rerunButton}
      </div>
    </div>
  )

  let body: React.ReactNode

  if (isLoading) {
    body = (
      <div role="status" aria-live="polite" className="flex justify-center py-8">
        <LoadingSpinner />
      </div>
    )
  } else if (isError || !data) {
    body = (
      <div
        role="alert"
        className="rounded-md border border-destructive/60 bg-destructive/10 p-3 text-sm text-destructive"
        data-testid="audit-error"
      >
        <AlertTriangle className="h-4 w-4 mr-2 inline" aria-hidden="true" />
        Could not load the audit. Try re-running.
      </div>
    )
  } else if (data.findings.length === 0) {
    body = (
      <div
        className="flex items-center gap-2 rounded-md border border-emerald-500/40 bg-emerald-500/10 p-3 text-sm text-emerald-800 dark:text-emerald-200"
        data-testid="audit-empty"
      >
        <CheckCircle2 className="h-4 w-4 shrink-0" aria-hidden="true" />
        {data.run_id
          ? 'No issues found in the latest audit.'
          : 'No audit has run yet — click Re-run to check this notebook.'}
      </div>
    )
  } else {
    body = (
      <div className="space-y-4" data-testid="audit-findings">
        <div className="flex flex-wrap gap-2" data-testid="audit-counts">
          {SEVERITY_ORDER.map((sev) =>
            grouped[sev].length ? (
              <Badge key={sev} variant={SEVERITY_META[sev].variant}>
                {grouped[sev].length} {SEVERITY_META[sev].label}
              </Badge>
            ) : null,
          )}
        </div>
        {SEVERITY_ORDER.map((sev) =>
          grouped[sev].length ? (
            <section
              key={sev}
              aria-label={`${SEVERITY_META[sev].label} (${grouped[sev].length})`}
              data-testid={`audit-group-${sev}`}
            >
              <h4 className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                {SEVERITY_META[sev].label}
              </h4>
              <ul className="space-y-1">
                {grouped[sev].map((f, i) => (
                  <li
                    key={`${f.check_id}-${f.subject ?? i}`}
                    className="flex items-start gap-2 text-sm"
                    data-testid="audit-finding"
                  >
                    <SeverityIcon severity={sev} />
                    <span>
                      {f.subject ? (
                        <a
                          href={`#${f.subject}`}
                          className="underline decoration-dotted underline-offset-2 hover:decoration-solid focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-sm"
                          data-testid="audit-finding-link"
                        >
                          {f.title}
                        </a>
                      ) : (
                        f.title
                      )}
                      {DEEP_CHECK_IDS.has(f.check_id) && (
                        <Badge
                          variant="outline"
                          className="ml-1.5 align-middle text-[10px] font-normal"
                          data-testid="audit-llm-badge"
                        >
                          LLM
                        </Badge>
                      )}
                    </span>
                  </li>
                ))}
              </ul>
            </section>
          ) : null,
        )}
      </div>
    )
  }

  return (
    <section
      role="region"
      aria-label="Notebook quality audit"
      className="space-y-3 rounded-lg border border-border bg-card p-4"
      data-testid="audit-widget"
    >
      {header}
      {body}
    </section>
  )
}

function SeverityIcon({ severity }: { severity: AuditSeverity }) {
  if (severity === 'error') {
    return (
      <AlertTriangle
        className="h-4 w-4 mt-0.5 shrink-0 text-destructive"
        aria-hidden="true"
      />
    )
  }
  if (severity === 'warn') {
    return (
      <AlertTriangle
        className="h-4 w-4 mt-0.5 shrink-0 text-amber-500"
        aria-hidden="true"
      />
    )
  }
  return (
    <Info className="h-4 w-4 mt-0.5 shrink-0 text-muted-foreground" aria-hidden="true" />
  )
}
