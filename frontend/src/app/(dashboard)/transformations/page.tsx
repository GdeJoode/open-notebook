'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { DefaultPromptEditor } from './components/DefaultPromptEditor'
import { TransformationsList } from './components/TransformationsList'
import { TransformationPlayground } from './components/TransformationPlayground'
import { useTransformations } from '@/lib/hooks/use-transformations'
import { Transformation } from '@/lib/types/transformations'
import { Wand2, Play, RefreshCw } from 'lucide-react'
import { PageBody } from '@/components/layout/PageBody'
import { PageHeader } from '@/components/layout/PageHeader'

export default function TransformationsPage() {
  const [activeTab, setActiveTab] = useState('transformations')
  const [selectedTransformation, setSelectedTransformation] = useState<Transformation | undefined>()
  const { data: transformations, isLoading, refetch } = useTransformations()

  const handlePlayground = (transformation: Transformation) => {
    setSelectedTransformation(transformation)
    setActiveTab('playground')
  }

  return (
    <PageBody>
      <PageHeader
        title="Transformations"
        subtitle="Transformations are prompts that will be used by the LLM to process a source and extract insights, summaries, etc."
        actions={
          <Button variant="outline" size="sm" onClick={() => refetch()} aria-label="Refresh transformations">
            <RefreshCw className="h-4 w-4" />
          </Button>
        }
      />

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Choose a workspace</p>
          <TabsList aria-label="Transformation views" className="w-full max-w-xl">
            <TabsTrigger value="transformations" className="flex items-center gap-2">
              <Wand2 className="h-4 w-4" />
              Transformations
            </TabsTrigger>
            <TabsTrigger value="playground" className="flex items-center gap-2">
              <Play className="h-4 w-4" />
              Playground
            </TabsTrigger>
          </TabsList>
        </div>
        
        <TabsContent value="transformations" className="space-y-6">
          <DefaultPromptEditor />
          <TransformationsList 
            transformations={transformations} 
            isLoading={isLoading}
            onPlayground={handlePlayground}
          />
        </TabsContent>
        
        <TabsContent value="playground">
          <TransformationPlayground 
            transformations={transformations}
            selectedTransformation={selectedTransformation}
          />
        </TabsContent>
      </Tabs>
    </PageBody>
  )
}
