'use client'

import { AppShell } from '@/components/layout/AppShell'
import { CreateSourcePipeline } from '@/components/sources/pipeline/CreateSourcePipeline'

export default function NewSourcePage() {
  return (
    <AppShell>
      <CreateSourcePipeline />
    </AppShell>
  )
}
