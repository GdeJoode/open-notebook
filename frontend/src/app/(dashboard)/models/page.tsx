'use client'

import { ProviderStatus } from './components/ProviderStatus'
import { ModelTypeSection } from './components/ModelTypeSection'
import { DefaultModelsSection } from './components/DefaultModelsSection'
import { useModels, useModelDefaults, useProviders, useModelStatus } from '@/lib/hooks/use-models'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert'
import { RefreshCw, AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useSearchParams } from 'next/navigation'
import { PageBody } from '@/components/layout/PageBody'
import { PageHeader } from '@/components/layout/PageHeader'

export default function ModelsPage() {
  const searchParams = useSearchParams()
  const setupRequired = searchParams.get('setup_required') === 'true'

  const { data: models, isLoading: modelsLoading, refetch: refetchModels } = useModels()
  const { data: defaults, isLoading: defaultsLoading, refetch: refetchDefaults } = useModelDefaults()
  const { data: providers, isLoading: providersLoading, refetch: refetchProviders } = useProviders()
  const { data: modelStatus } = useModelStatus(setupRequired)

  const handleRefresh = () => {
    refetchModels()
    refetchDefaults()
    refetchProviders()
  }

  const header = (
    <PageHeader
      title="Model Management"
      subtitle="Configure AI models for different purposes across Noesis"
      actions={
        <Button variant="outline" size="sm" onClick={handleRefresh} aria-label="Refresh models">
          <RefreshCw className="h-4 w-4" />
        </Button>
      }
    />
  )

  if (modelsLoading || defaultsLoading || providersLoading) {
    return (
      <PageBody>
        {header}
        <div className="flex items-center justify-center min-h-[60vh]">
          <LoadingSpinner size="lg" />
        </div>
      </PageBody>
    )
  }

  if (!models || !defaults || !providers) {
    return (
      <PageBody>
        {header}
        <div className="text-center py-12">
          <p className="text-muted-foreground">Failed to load models data</p>
        </div>
      </PageBody>
    )
  }

  return (
    <PageBody>
      {header}

      {setupRequired && modelStatus && !modelStatus.valid && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Model configuration required</AlertTitle>
          <AlertDescription>
            <p className="mb-2">
              Some configured models are not available. Please review and fix the issues below before continuing.
            </p>
            <ul className="list-disc pl-4 space-y-1">
              {modelStatus.issues.map((issue, i) => (
                <li key={i}>{issue.message}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      <div className="grid gap-6">
        {/* Provider Status */}
        <ProviderStatus providers={providers} />

        {/* Default Models */}
        <DefaultModelsSection models={models} defaults={defaults} />

        {/* Model Types */}
        <div className="grid gap-6 lg:grid-cols-2">
          <ModelTypeSection 
            type="language" 
            models={models} 
            providers={providers}
            isLoading={modelsLoading}
          />
          <ModelTypeSection 
            type="embedding" 
            models={models} 
            providers={providers}
            isLoading={modelsLoading}
          />
          <ModelTypeSection 
            type="text_to_speech" 
            models={models} 
            providers={providers}
            isLoading={modelsLoading}
          />
          <ModelTypeSection 
            type="speech_to_text" 
            models={models} 
            providers={providers}
            isLoading={modelsLoading}
          />
        </div>
      </div>
    </PageBody>
  )
}
