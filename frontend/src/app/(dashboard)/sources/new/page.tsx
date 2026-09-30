'use client'

import Link from 'next/link'
import { ArrowLeft } from 'lucide-react'
import { CreateSourcePipeline } from '@/components/sources/pipeline/CreateSourcePipeline'
import { PageBody } from '@/components/layout/PageBody'
import { PageHeader } from '@/components/layout/PageHeader'

export default function NewSourcePage() {
  return (
    <PageBody width="form" fill>
      <PageHeader
        back={
          <Link
            href="/sources"
            className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Sources
          </Link>
        }
        title="Create New Source"
      />
      <CreateSourcePipeline />
    </PageBody>
  )
}
