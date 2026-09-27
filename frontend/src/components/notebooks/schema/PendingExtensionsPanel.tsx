'use client'

import { Button } from '@/components/ui/button'
import {
  useAcceptExtension,
  useRejectExtension,
} from '@/lib/hooks/use-notebook-schema'
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
 * Pending-extensions panel with Accept / Reject mutations (B.3b).
 *
 * Each row shows the proposed `type_name`, optional `parent_type` and
 * `rationale` from the LLM, plus two action buttons.
 *
 * - Accept moves the extension into `accepted_extensions`. The list
 *   updates in place via React Query cache replacement (B.3b mutation
 *   hooks return the full updated schema).
 * - Reject drops the extension.
 *
 * Disabled-while-pending: every button on a row disables when the
 * row's mutation is in flight so a user can't double-fire it. Other
 * rows stay enabled — useful when reviewing a batch.
 *
 * # B.3a → B.3b note
 *
 * The B.3a version of this component intentionally rendered disabled
 * buttons with a "coming in next release" tooltip. That stub is fully
 * superseded here — the buttons live, and the tooltip is gone.
 */
export function PendingExtensionsPanel({
  extensions,
  notebookId,
}: PendingExtensionsPanelProps) {
  const acceptMutation = useAcceptExtension(notebookId ?? '')
  const rejectMutation = useRejectExtension(notebookId ?? '')

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

  // Track which row's mutation is in flight so we can disable BOTH
  // buttons on that row (preventing accept-then-immediately-reject
  // races) while leaving every other row interactive.
  const pendingTypeName =
    acceptMutation.isPending
      ? (acceptMutation.variables as string | undefined)
      : rejectMutation.isPending
        ? (rejectMutation.variables as string | undefined)
        : undefined

  return (
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
                  onClick={() =>
                    notebookId &&
                    acceptMutation.mutate(ext.type_name)
                  }
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
  )
}
