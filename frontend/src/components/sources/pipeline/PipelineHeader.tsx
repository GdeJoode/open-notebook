'use client'

import { useRouter } from 'next/navigation'
import { ArrowLeft } from 'lucide-react'
import { Button } from '@/components/ui/button'

export function PipelineHeader() {
  const router = useRouter()

  return (
    <div className="flex items-center justify-between px-6 py-4 border-b border-border">
      <Button
        variant="ghost"
        size="sm"
        onClick={() => router.push('/sources')}
        className="gap-2 text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to Sources
      </Button>
      <h1 className="text-lg font-semibold">Create New Source</h1>
      <div className="w-[140px]" /> {/* Spacer for centering */}
    </div>
  )
}
