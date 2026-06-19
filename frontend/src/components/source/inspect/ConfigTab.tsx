'use client'

import { useState } from 'react'
import { toast } from 'sonner'
import { PipelineConfigPanel } from '@/components/sources/pipeline/PipelineConfigPanel'
import {
  DEFAULT_PIPELINE_CONFIG,
  sourcesApi,
  type DoclingPipelineConfig,
} from '@/lib/api/sources'

interface ConfigTabProps {
  sourceId: string
}

/**
 * Config tab of the inspect workspace right pane (I.D-4).
 *
 * Hosts the Docling pipeline-config reprocess panel that previously lived
 * inside PropertiesPanel. Splitting it into its own tab keeps the Properties
 * tab focused on read-only chunk metadata while giving reprocessing room.
 */
export function ConfigTab({ sourceId }: ConfigTabProps) {
  const [reprocessConfig, setReprocessConfig] = useState<DoclingPipelineConfig>({
    ...DEFAULT_PIPELINE_CONFIG,
  })
  const [reprocessing, setReprocessing] = useState(false)

  const handleReprocess = async () => {
    try {
      setReprocessing(true)
      await sourcesApi.reprocess(sourceId, reprocessConfig)
      toast.success('Reprocessing started — document will be re-ingested')
    } catch {
      toast.error('Failed to start reprocessing')
    } finally {
      setReprocessing(false)
    }
  }

  return (
    <div className="h-full overflow-y-auto bg-card p-3">
      <PipelineConfigPanel
        config={reprocessConfig}
        onChange={setReprocessConfig}
        onReprocess={handleReprocess}
        reprocessing={reprocessing}
        collapsible={false}
      />
    </div>
  )
}
