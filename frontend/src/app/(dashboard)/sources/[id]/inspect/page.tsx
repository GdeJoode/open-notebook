'use client'

import { useCallback } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { ArrowLeft } from 'lucide-react'
import { DocumentInspectWorkspace } from '@/components/source/inspect/DocumentInspectWorkspace'
import { PageBody } from '@/components/layout/PageBody'
import { PageHeader } from '@/components/layout/PageHeader'
import { Button } from '@/components/ui/button'

export default function SourceInspectPage() {
  const params = useParams()
  const router = useRouter()
  const sourceId = decodeURIComponent(params.id as string)

  const handleBack = useCallback(() => {
    router.push(`/sources/${encodeURIComponent(sourceId)}`)
  }, [router, sourceId])

  return (
    <PageBody fill>
      <PageHeader
        back={
          <Button variant="ghost" size="sm" className="-ml-3" onClick={handleBack}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>
        }
        title="Inspect document"
      />
      <DocumentInspectWorkspace sourceId={sourceId} />
    </PageBody>
  )
}
