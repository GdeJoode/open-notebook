import { cn } from '@/lib/utils'

const WIDTHS = {
  wide: 'max-w-7xl',
  form: 'max-w-3xl',
} as const

export type PageWidth = keyof typeof WIDTHS

interface PageBodyProps {
  /** `wide` for tables and boards, `form` for forms and single-column reading. */
  width?: PageWidth
  /**
   * The page owns its scroll regions (a board whose table scrolls, a split
   * view): the body fills the height below the shell instead of scrolling.
   */
  fill?: boolean
  className?: string
  children: React.ReactNode
}

/**
 * The one frame every dashboard page sits in (WB.4): one padding, one width
 * scale, left-aligned against the sidebar. Put the `PageHeader` first inside.
 */
export function PageBody({ width = 'wide', fill = false, className, children }: PageBodyProps) {
  return (
    <div
      data-page-body={width}
      className={cn('flex min-h-0 flex-1 flex-col', !fill && 'overflow-y-auto')}
    >
      <div
        className={cn(
          'flex w-full flex-col gap-6 p-6',
          WIDTHS[width],
          fill && 'min-h-0 flex-1',
          className,
        )}
      >
        {children}
      </div>
    </div>
  )
}
