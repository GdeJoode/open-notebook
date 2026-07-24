'use client'

import { Loader2 } from 'lucide-react'

import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { useNotebook, useUpdateNotebook } from '@/lib/hooks/use-notebooks'
import { useNotebookAudit } from '@/lib/hooks/use-audit'

interface LibrarianSettingsProps {
  notebookId: string
}

/**
 * Per-notebook librarian opt-in (Track F.6).
 *
 * A toggle for `notebook.librarian_enabled` (the F.5 periodic-audit opt-in) plus
 * the last-run timestamp read from the latest audit run. Default OFF; disabling
 * it stops the F.5 scheduler from enqueueing this notebook (the enqueue guard
 * lists only opted-in notebooks).
 *
 * No interval control: F.5 does not store a per-notebook interval (the schedule
 * is driven externally), so exposing one here would be a non-functional
 * placeholder. When F.5 grows an interval, add it here.
 */
export function LibrarianSettings({ notebookId }: LibrarianSettingsProps) {
  const { data: notebook, isLoading, isError } = useNotebook(notebookId)
  const update = useUpdateNotebook()
  const { data: audit } = useNotebookAudit(notebookId)

  if (isLoading) {
    return (
      <div role="status" aria-live="polite" className="flex justify-center py-4">
        <LoadingSpinner />
      </div>
    )
  }

  if (isError || !notebook) {
    return (
      <div
        role="alert"
        className="rounded-md border border-destructive/60 bg-destructive/10 p-3 text-sm text-destructive"
        data-testid="librarian-error"
      >
        Could not load librarian settings.
      </div>
    )
  }

  const enabled = notebook.librarian_enabled ?? false
  const lastRun = audit?.created
  const lastRunLabel =
    audit?.run_id && lastRun
      ? `Last audit ${new Date(lastRun).toLocaleString()}`
      : 'Never run'

  return (
    <section
      role="region"
      aria-label="Librarian settings"
      className="space-y-2 rounded-lg border border-border bg-card p-4"
      data-testid="librarian-settings"
    >
      <div className="flex items-center gap-3">
        <Switch
          id="librarian-toggle"
          checked={enabled}
          onCheckedChange={(checked) =>
            update.mutate({ id: notebookId, data: { librarian_enabled: checked } })
          }
          disabled={update.isPending}
          aria-label="Enable the periodic quality audit for this notebook"
          data-testid="librarian-toggle"
        />
        <Label htmlFor="librarian-toggle" className="text-sm cursor-pointer">
          Periodic quality audit (librarian)
        </Label>
        {update.isPending && (
          <Loader2
            className="h-4 w-4 animate-spin text-muted-foreground"
            aria-hidden="true"
          />
        )}
      </div>
      <p className="text-xs text-muted-foreground" data-testid="librarian-last-run">
        {enabled
          ? `On — the audit re-runs on the server's schedule. ${lastRunLabel}.`
          : `Off — enable to re-run the quality audit periodically. ${lastRunLabel}.`}
      </p>
    </section>
  )
}
