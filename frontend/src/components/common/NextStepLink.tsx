import Link from 'next/link'
import { ArrowRight } from 'lucide-react'

import { Button } from '@/components/ui/button'

interface NextStepLinkProps {
  href: string
  children: React.ReactNode
  /** The primary action once the page's own queue is empty. */
  primary?: boolean
  'data-testid'?: string
}

/**
 * Where the pipeline goes after this page (WB.2): review pages link onward,
 * not only back.
 */
export function NextStepLink({ href, children, primary = false, ...rest }: NextStepLinkProps) {
  return (
    <Button asChild size="sm" variant={primary ? 'default' : 'outline'}>
      <Link href={href} data-testid={rest['data-testid']}>
        {children}
        <ArrowRight />
      </Link>
    </Button>
  )
}
