'use client'

import { useCallback } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { AppShell } from '@/components/layout/AppShell'
import { DocumentInspectWorkspace } from '@/components/source/inspect/DocumentInspectWorkspace'

export default function SourceInspectPage() {
  const params = useParams()
  const router = useRouter()
  const sourceId = decodeURIComponent(params.id as string)

  const handleBack = useCallback(() => {
    router.push(`/sources/${encodeURIComponent(sourceId)}`)
  }, [router, sourceId])

  return (
    <AppShell>
      <DocumentInspectWorkspace sourceId={sourceId} onBack={handleBack} />
    </AppShell>
  )
}
