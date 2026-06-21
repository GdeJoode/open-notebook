'use client'

import { Badge } from '@/components/ui/badge'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { ProviderHealth } from '@/lib/types/model-routing'
import { AlertTriangle, Check, CircleSlash, X } from 'lucide-react'

interface ProviderHealthChipProps {
  health: ProviderHealth
}

/**
 * A status chip for one provider: configured vs unconfigured (visually
 * distinct — AC6) plus the circuit-breaker state. An unconfigured cloud
 * provider is rendered muted/dashed; a tripped (open) breaker is destructive.
 */
export function ProviderHealthChip({ health }: ProviderHealthChipProps) {
  const { provider, configured, is_local, circuit_state, recent_fallback_count } =
    health

  // Unconfigured cloud provider: visually distinct, dashed/muted.
  if (!configured) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            variant="outline"
            className="border-dashed text-muted-foreground"
            aria-label={`${provider}: not configured`}
          >
            <X className="h-3 w-3" aria-hidden="true" />
            <span className="capitalize">{provider}</span>
            <span className="text-[10px]">not configured</span>
          </Badge>
        </TooltipTrigger>
        <TooltipContent>
          Set this provider&apos;s API key env var to enable it.
        </TooltipContent>
      </Tooltip>
    )
  }

  const open = circuit_state === 'open'
  const halfOpen = circuit_state === 'half_open'

  const variant = open ? 'destructive' : halfOpen ? 'secondary' : 'default'
  const Icon = open ? CircleSlash : halfOpen ? AlertTriangle : Check
  const stateLabel = open
    ? 'circuit open'
    : halfOpen
      ? 'recovering'
      : is_local
        ? 'local'
        : 'healthy'

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge
          variant={variant}
          aria-label={`${provider}: ${stateLabel}`}
          data-circuit-state={circuit_state}
        >
          <Icon className="h-3 w-3" aria-hidden="true" />
          <span className="capitalize">{provider}</span>
          <span className="text-[10px]">{stateLabel}</span>
        </Badge>
      </TooltipTrigger>
      <TooltipContent>
        <div className="space-y-0.5 text-xs">
          <div>Circuit: {circuit_state}</div>
          <div>{is_local ? 'Local provider' : 'Cloud provider'}</div>
          {recent_fallback_count > 0 && (
            <div>Recent fallbacks from: {recent_fallback_count}</div>
          )}
        </div>
      </TooltipContent>
    </Tooltip>
  )
}
