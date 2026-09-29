'use client'

import { useState } from 'react'

import { AcceptExtensionDialog } from '@/components/notebooks/schema/AcceptExtensionDialog'
import { Button } from '@/components/ui/button'
import { useRejectExtension } from '@/lib/hooks/use-notebook-schema'
import type { ExtensionView } from '@/lib/types/notebook_schema'

interface PendingExtensionsPanelProps {
  extensions: ExtensionView[]
  /**
   * Required for the B.3b accept/reject mutations. When omitted (e.g. a
   * preview render in Storybook), the buttons fall back to disabled —
   * the same behaviour as the B.3a stub.
   */
  notebookId?: string
}

/**
 * The notebook's queue of Pass-1 proposals (B.3b, OE.5).
 *
 * Each row shows the proposed `type_name`, optional `parent_type` and
 * `rationale` from the LLM, plus two action buttons.
 *
 * - Accept opens {@link AcceptExtensionDialog}: the type is published to a
 *   SHARED ontology through the ontology editor's gate (OE.5), and leaves the
 *   queue once published.
 * - Reject drops the proposal; the "no" is remembered.
 *
 * Disabled-while-pending: both buttons on a row disable while its reject is
 * in flight so a user can't double-fire it. Other rows stay enabled.
 */
export function PendingExtensionsPanel({
  extensions,
  notebookId,
}: PendingExtensionsPanelProps) {
  const rejectMutation = useRejectExtension(notebookId ?? '')
  const [accepting, setAccepting] = useState<ExtensionView | null>(null)

  if (extensions.length === 0) {
    return (
      <div
        role="status"
        className="rounded-md border border-dashed p-6 text-sm text-muted-foreground"
        data-testid="pending-extensions-empty"
      >
        No pending extensions. New proposals appear here when pass-1
        extraction finds concepts that don&apos;t fit the current schema.
      </div>
    )
  }

  // Track which row's reject is in flight so we can disable BOTH buttons on
  // that row while leaving every other row interactive.
  const pendingTypeName = rejectMutation.isPending
    ? (rejectMutation.variables as string | undefined)
    : undefined

  return (
    <>
    <ul
      className="flex flex-col gap-2"
      data-testid="pending-extensions-panel"
    >
      {extensions.map((ext, idx) => {
        const key = ext.extension_id ?? `${ext.type_name}-${idx}`
        const buttonsDisabled =
          !notebookId || pendingTypeName === ext.type_name
        return (
          <li
            key={key}
            className="rounded-md border p-3"
            data-testid={`pending-extension-${ext.type_name}`}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <h4 className="text-sm font-medium">{ext.type_name}</h4>
                  {ext.parent_type && (
                    <span className="text-xs text-muted-foreground">
                      extends{' '}
                      <code className="font-mono">{ext.parent_type}</code>
                    </span>
                  )}
                </div>
                {ext.rationale && (
                  <p className="text-xs text-muted-foreground">
                    {ext.rationale}
                  </p>
                )}
                {ext.description && !ext.rationale && (
                  <p className="text-xs text-muted-foreground">
                    {ext.description}
                  </p>
                )}
              </div>

              <div className="flex shrink-0 gap-1.5">
                <Button
                  type="button"
                  size="sm"
                  data-testid={`accept-${key}`}
                  disabled={buttonsDisabled}
                  onClick={() => notebookId && setAccepting(ext)}
                >
                  Accept
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  data-testid={`reject-${key}`}
                  disabled={buttonsDisabled}
                  onClick={() =>
                    notebookId &&
                    rejectMutation.mutate(ext.type_name)
                  }
                >
                  Reject
                </Button>
              </div>
            </div>
          </li>
        )
      })}
    </ul>
    {notebookId && accepting && (
      <AcceptExtensionDialog
        notebookId={notebookId}
        extension={accepting}
        onOpenChange={(open) => {
          if (!open) setAccepting(null)
        }}
      />
    )}
    </>
  )
}
