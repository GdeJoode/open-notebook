'use client'

import { Fragment } from 'react'
import type { LucideIcon } from 'lucide-react'
import { Ellipsis } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'

export interface PageOverflowAction {
  label: string
  icon?: LucideIcon
  onSelect: () => void
  destructive?: boolean
  disabled?: boolean
  testId?: string
}

interface PageHeaderProps {
  /** The page's one `<h1>`. */
  title: React.ReactNode
  subtitle?: React.ReactNode
  /** A row above the title: the way back, and on review pages the next step. */
  back?: React.ReactNode
  /** Right-aligned; keep it to the page's main actions. */
  actions?: React.ReactNode
  /** Rarely used actions, behind a "⋯" menu after `actions`. */
  overflow?: PageOverflowAction[]
  overflowLabel?: string
  /** Rows under the title (metadata, filters, section links). */
  children?: React.ReactNode
}

export function PageHeader({
  title,
  subtitle,
  back,
  actions,
  overflow,
  overflowLabel = 'More actions',
  children,
}: PageHeaderProps) {
  const hasOverflow = !!overflow && overflow.length > 0
  return (
    <header data-page-header className="flex shrink-0 flex-col gap-2">
      {back && <div className="flex flex-wrap items-center justify-between gap-2">{back}</div>}
      <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
        <div className="min-w-0 flex-1 space-y-1">
          <h1 className="text-2xl font-bold">{title}</h1>
          {subtitle && <p className="text-muted-foreground">{subtitle}</p>}
        </div>
        {(actions || hasOverflow) && (
          <div className="flex shrink-0 flex-wrap items-center gap-2">
            {actions}
            {hasOverflow && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="outline"
                    size="icon"
                    className="size-8"
                    aria-label={overflowLabel}
                    data-testid="page-overflow-trigger"
                  >
                    <Ellipsis />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-44">
                  {overflow.map((action, i) => (
                    <Fragment key={action.label}>
                      {/* Destructive actions sit apart, last. */}
                      {action.destructive && i > 0 && !overflow[i - 1].destructive && (
                        <DropdownMenuSeparator />
                      )}
                      <DropdownMenuItem
                        variant={action.destructive ? 'destructive' : 'default'}
                        disabled={action.disabled}
                        onSelect={action.onSelect}
                        data-testid={action.testId}
                      >
                        {action.icon && <action.icon />}
                        {action.label}
                      </DropdownMenuItem>
                    </Fragment>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            )}
          </div>
        )}
      </div>
      {children}
    </header>
  )
}
