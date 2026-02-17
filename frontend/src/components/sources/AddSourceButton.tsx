'use client'

import { useRouter } from 'next/navigation'
import { PlusIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface AddSourceButtonProps {
  defaultNotebookId?: string
  variant?: 'default' | 'outline' | 'ghost'
  size?: 'sm' | 'default' | 'lg'
  className?: string
  iconOnly?: boolean
}

export function AddSourceButton({
  defaultNotebookId,
  variant = 'default',
  size = 'default',
  className,
  iconOnly = false
}: AddSourceButtonProps) {
  const router = useRouter()

  const handleClick = () => {
    const url = defaultNotebookId
      ? `/sources/new?notebook=${defaultNotebookId}`
      : '/sources/new'
    router.push(url)
  }

  return (
    <Button
      onClick={handleClick}
      variant={variant}
      size={size}
      className={className}
    >
      <PlusIcon className={iconOnly ? "h-4 w-4" : "h-4 w-4 mr-2"} />
      {!iconOnly && "Add Source"}
    </Button>
  )
}
