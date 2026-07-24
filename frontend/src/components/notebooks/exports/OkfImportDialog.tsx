'use client'

/**
 * OkfImportDialog — OKF import UI (the OKF.5 export-only gap).
 *
 * The inverse of ``OkfExportDialog``: uploads an OKF v0.1 Knowledge Bundle
 * (``.zip``) into this notebook and renders the server's import report —
 * created/matched counts plus the non-silent skip / dangling-link ledger.
 *
 * Import is idempotent server-side: entities upsert on the deterministic
 * ``(name, type)`` dedup and notes/sources are created idempotently, all tagged
 * ``okf-import``. Re-importing the same bundle therefore *matches* rather than
 * duplicates — surfaced in the report as matched (not created) counts.
 *
 * Accessibility: Radix ``Dialog`` traps focus, restores it on close, binds Esc;
 * the error region is ``role="alert"`` and the ledger a keyboard-operable
 * ``<details>`` disclosure.
 */

import { useEffect, useId, useMemo, useRef, useState } from 'react'
import { AlertTriangle, Info, Loader2, Upload } from 'lucide-react'

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { useOkfImport } from '@/lib/hooks/use-okf-import'
import type { OkfImportReport } from '@/lib/types/exports'

interface OkfImportDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  notebookId: string
}

export function OkfImportDialog({
  open,
  onOpenChange,
  notebookId,
}: OkfImportDialogProps) {
  const [file, setFile] = useState<File | null>(null)
  const [applyDedup, setApplyDedup] = useState(true)
  const [successReport, setSuccessReport] = useState<OkfImportReport | null>(
    null,
  )
  const fileInputRef = useRef<HTMLInputElement>(null)

  const { mutate, isPending, error } = useOkfImport(notebookId)

  // Reset to a clean state each time the dialog opens.
  useEffect(() => {
    if (open) {
      setFile(null)
      setApplyDedup(true)
      setSuccessReport(null)
    }
  }, [open])

  const titleId = useId()
  const descriptionId = useId()
  const fileId = useId()
  const dedupId = useId()

  const importDisabled = useMemo(
    () => isPending || file === null,
    [isPending, file],
  )

  const handleImport = () => {
    if (importDisabled || !file) return
    mutate(
      { file, applyDedup },
      { onSuccess: (report) => setSuccessReport(report) },
    )
  }

  const close = () => onOpenChange(false)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="sm:max-w-[560px]"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
        data-testid="okf-import-dialog"
      >
        <DialogHeader>
          <DialogTitle id={titleId}>Import from Open Knowledge Format</DialogTitle>
          <DialogDescription id={descriptionId}>
            Upload an OKF v0.1 bundle (a <code>.zip</code> of Markdown concepts)
            to merge its entities, notes, and sources into this notebook.
          </DialogDescription>
        </DialogHeader>

        {!successReport && (
          <div className="space-y-4">
            <IdempotentBanner />

            <div className="space-y-1.5">
              <Label htmlFor={fileId} className="text-sm">
                OKF bundle (.zip)
              </Label>
              <Input
                ref={fileInputRef}
                id={fileId}
                type="file"
                accept=".zip,application/zip"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                data-testid="okf-import-file"
                aria-label="OKF bundle zip file"
              />
              {file && (
                <p
                  className="text-xs text-muted-foreground"
                  data-testid="okf-import-filename"
                >
                  Selected: <strong>{file.name}</strong> (
                  {Math.max(1, Math.round(file.size / 1024))} KB)
                </p>
              )}
            </div>

            <div className="flex items-center gap-3">
              <Switch
                id={dedupId}
                checked={applyDedup}
                onCheckedChange={setApplyDedup}
                aria-label="Match imported entities against the existing graph"
                data-testid="okf-import-dedup-switch"
              />
              <Label htmlFor={dedupId} className="text-sm cursor-pointer">
                Match against existing entities (recommended)
              </Label>
            </div>

            {error && (
              <div
                role="alert"
                className="flex items-start gap-2 rounded-md border border-destructive/60 bg-destructive/10 p-3 text-sm text-destructive"
                data-testid="okf-import-error"
              >
                <AlertTriangle
                  className="h-4 w-4 mt-0.5 shrink-0"
                  aria-hidden="true"
                />
                <span>
                  {error.message || 'Import failed. Please try again.'}
                </span>
              </div>
            )}
          </div>
        )}

        {successReport && <OkfImportSuccess report={successReport} />}

        <DialogFooter className="gap-2 sm:gap-2">
          <Button
            type="button"
            variant="outline"
            onClick={close}
            disabled={isPending}
            data-testid="okf-import-cancel"
          >
            {successReport ? 'Done' : 'Cancel'}
          </Button>
          {!successReport && (
            <Button
              type="button"
              onClick={handleImport}
              disabled={importDisabled}
              data-testid="okf-import-submit"
              aria-label="Import Open Knowledge Format bundle"
            >
              {isPending ? (
                <Loader2
                  className="h-4 w-4 mr-2 animate-spin"
                  aria-hidden="true"
                />
              ) : (
                <Upload className="h-4 w-4 mr-2" aria-hidden="true" />
              )}
              {isPending ? 'Importing…' : 'Import'}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

/** Idempotency notice: re-importing matches rather than duplicates. */
function IdempotentBanner() {
  return (
    <div
      role="note"
      aria-label="OKF import is idempotent"
      className="flex items-start gap-2 rounded-md border border-sky-500/40 bg-sky-500/10 p-3 text-sm text-sky-800 dark:text-sky-200"
      data-testid="okf-import-idempotent-banner"
    >
      <Info className="h-4 w-4 mt-0.5 shrink-0" aria-hidden="true" />
      <span>
        Import is <strong>idempotent</strong>. Entities are matched on name +
        type and notes/sources are created once, so re-importing the same bundle
        updates rather than duplicates.
      </span>
    </div>
  )
}

interface OkfImportSuccessProps {
  report: OkfImportReport
}

/** Post-import success state: created/matched counts + the skip/dangling ledger. */
function OkfImportSuccess({ report }: OkfImportSuccessProps) {
  const { skipped, dangling_links: dangling } = report
  return (
    <div
      className="space-y-3 rounded-md border border-border bg-card p-3"
      data-testid="okf-import-success"
    >
      <div className="text-sm font-medium">OKF bundle imported</div>
      <div className="text-sm text-muted-foreground">
        Entities <strong>+{report.entities_created}</strong> new /{' '}
        <strong>{report.entities_matched}</strong> matched · Relations{' '}
        <strong>+{report.relations_created}</strong>
      </div>
      <div className="text-sm text-muted-foreground">
        Notes <strong>+{report.notes_created}</strong> /{' '}
        <strong>{report.notes_matched}</strong> · Sources{' '}
        <strong>+{report.sources_created}</strong> /{' '}
        <strong>{report.sources_matched}</strong>
        {report.dedup_merges > 0 ? (
          <>
            {' '}
            · Merged <strong>{report.dedup_merges}</strong>
          </>
        ) : null}
      </div>

      {skipped.length > 0 && (
        <details className="group" data-testid="okf-import-skipped-ledger">
          <summary className="cursor-pointer text-sm font-medium text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-sm">
            Skipped ({skipped.length})
          </summary>
          <ul className="mt-2 space-y-1.5 pl-1">
            {skipped.map((s) => (
              <li
                key={s.path}
                className="text-xs text-muted-foreground"
                data-testid="okf-import-skipped-row"
              >
                <code className="font-mono text-foreground">{s.path}</code>
                {' — '}
                {s.reason}
              </li>
            ))}
          </ul>
        </details>
      )}

      {dangling.length > 0 && (
        <details className="group" data-testid="okf-import-dangling-ledger">
          <summary className="cursor-pointer text-sm font-medium text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-sm">
            Dangling links ({dangling.length})
          </summary>
          <ul className="mt-2 space-y-1.5 pl-1">
            {dangling.map((d, i) => (
              <li
                key={`${d.from}-${d.to}-${i}`}
                className="text-xs text-muted-foreground"
                data-testid="okf-import-dangling-row"
              >
                <code className="font-mono text-foreground">{d.from}</code>
                {' → '}
                <code className="font-mono text-foreground">{d.to}</code>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  )
}
