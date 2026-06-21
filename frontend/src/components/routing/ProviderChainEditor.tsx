'use client'

import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { ProviderChainEntry } from '@/lib/types/model-routing'
import {
  isFirst,
  isLast,
  moveDown,
  moveUp,
  toggleEnabled,
} from '@/lib/utils/provider-chain'
import { ArrowDown, ArrowUp } from 'lucide-react'

interface ProviderChainEditorProps {
  /** Human label for the task this chain serves (e.g. "Chat"). */
  taskLabel: string
  chain: ProviderChainEntry[]
  onChange: (chain: ProviderChainEntry[]) => void
  disabled?: boolean
}

/**
 * Ordered provider-chain editor for one task.
 *
 * Reordering uses accessible up/down arrow buttons (NOT drag — the project has
 * no dnd library). Each row carries the provider name, an enable/disable
 * Switch, and up/down buttons that are disabled at the ends and carry
 * `aria-label`s. A disabled provider is greyed and excluded from the effective
 * chain shown downstream (AC3).
 */
export function ProviderChainEditor({
  taskLabel,
  chain,
  onChange,
  disabled = false,
}: ProviderChainEditorProps) {
  const handleMoveUp = (index: number) => onChange(moveUp(chain, index))
  const handleMoveDown = (index: number) => onChange(moveDown(chain, index))
  const handleToggle = (index: number) => onChange(toggleEnabled(chain, index))

  return (
    <ul
      className="space-y-2"
      aria-label={`Provider chain for ${taskLabel}`}
    >
      {chain.map((entry, index) => (
        <li
          key={`${entry.provider}-${index}`}
          className={`flex items-center gap-3 rounded-lg border px-3 py-2 transition-colors ${
            entry.enabled ? 'bg-card' : 'bg-muted/40 opacity-60'
          }`}
          data-provider={entry.provider}
          data-enabled={entry.enabled}
        >
          <span className="w-5 text-xs text-muted-foreground tabular-nums">
            {index + 1}.
          </span>

          <div className="flex min-w-0 flex-1 flex-col">
            <span
              className={`truncate text-sm font-medium capitalize ${
                entry.enabled ? 'text-foreground' : 'text-muted-foreground'
              }`}
            >
              {entry.provider}
            </span>
            {entry.model_id && (
              <span className="truncate text-xs text-muted-foreground">
                {entry.model_id}
              </span>
            )}
          </div>

          <div className="flex items-center gap-1">
            <Switch
              checked={entry.enabled}
              onCheckedChange={() => handleToggle(index)}
              disabled={disabled}
              aria-label={`${
                entry.enabled ? 'Disable' : 'Enable'
              } ${entry.provider} for ${taskLabel}`}
            />
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={() => handleMoveUp(index)}
              disabled={disabled || isFirst(index)}
              aria-label={`Move ${entry.provider} up`}
            >
              <ArrowUp className="h-4 w-4" aria-hidden="true" />
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={() => handleMoveDown(index)}
              disabled={disabled || isLast(index, chain.length)}
              aria-label={`Move ${entry.provider} down`}
            >
              <ArrowDown className="h-4 w-4" aria-hidden="true" />
            </Button>
          </div>
        </li>
      ))}
    </ul>
  )
}
