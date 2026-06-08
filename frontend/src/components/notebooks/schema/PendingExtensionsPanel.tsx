'use client'

import { Button } from '@/components/ui/button'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import type { ExtensionView } from '@/lib/types/notebook_schema'

interface PendingExtensionsPanelProps {
  extensions: ExtensionView[]
}

const DISABLED_TOOLTIP = 'Edit ops coming in next release'

/**
 * Read-only list of pending extensions for the Schema tab.
 *
 * Each row shows the proposed `type_name`, optional `parent_type`,
 * and the LLM's `rationale` for why this extension was suggested.
 * The Accept / Reject buttons are present but **disabled** in B.3a
 * — the mutations land in B.3b. Hovering / focusing a disabled
 * button reveals a tooltip explaining the upcoming feature so the
 * UX doesn't feel broken to a user who clicks first.
 *
 * Disabled-button accessibility note: HTML disabled buttons are
 * skipped by Tab and ignore mouse events, so the tooltip can't fire
 * via Radix's default focus/hover triggers. We wrap the button in a
 * span so the TooltipTrigger has a focusable target — common pattern
 * for "show a tooltip on a disabled control".
 */
export function PendingExtensionsPanel({
  extensions,
}: PendingExtensionsPanelProps) {
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

  return (
    <TooltipProvider>
      <ul
        className="flex flex-col gap-2"
        data-testid="pending-extensions-panel"
      >
        {extensions.map((ext, idx) => {
          const key = ext.extension_id ?? `${ext.type_name}-${idx}`
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
                  <DisabledActionButton
                    label="Accept"
                    testId={`accept-${key}`}
                  />
                  <DisabledActionButton
                    label="Reject"
                    testId={`reject-${key}`}
                    variant="outline"
                  />
                </div>
              </div>
            </li>
          )
        })}
      </ul>
    </TooltipProvider>
  )
}

function DisabledActionButton({
  label,
  testId,
  variant = 'default',
}: {
  label: string
  testId: string
  variant?: 'default' | 'outline'
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {/* The wrapping span gives Radix a focusable target for the
            tooltip, since a disabled <button> doesn't receive focus
            or hover events. */}
        <span tabIndex={0} className="inline-block">
          <Button
            type="button"
            size="sm"
            variant={variant}
            disabled
            aria-disabled="true"
            data-testid={testId}
          >
            {label}
          </Button>
        </span>
      </TooltipTrigger>
      <TooltipContent>{DISABLED_TOOLTIP}</TooltipContent>
    </Tooltip>
  )
}
