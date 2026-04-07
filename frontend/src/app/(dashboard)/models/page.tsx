'use client'

import { AppShell } from '@/components/layout/AppShell'
import { ProviderStatus } from './components/ProviderStatus'
import { ModelTypeSection } from './components/ModelTypeSection'
import { DefaultModelsSection } from './components/DefaultModelsSection'
import { useModels, useModelDefaults, useProviders, useModelStatus } from '@/lib/hooks/use-models'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert'
import { RefreshCw, AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useSearchParams } from 'next/navigation'

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

  if (modelsLoading || defaultsLoading || providersLoading) {
    return (
      <AppShell>
        <div className="flex items-center justify-center min-h-[60vh]">
          <LoadingSpinner size="lg" />
        </div>
      </AppShell>
    )
  }

  if (!models || !defaults || !providers) {
    return (
      <AppShell>
        <div className="p-6">
          <div className="text-center py-12">
            <p className="text-muted-foreground">Failed to load models data</p>
          </div>
        </div>
      </AppShell>
    )
  }

  return (
    <AppShell>
      <div className="flex-1 overflow-y-auto">
        <div className="p-6 space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">Model Management</h1>
              <p className="text-muted-foreground mt-1">
                Configure AI models for different purposes across Noesis
              </p>
            </div>
            <Button variant="outline" size="sm" onClick={handleRefresh}>
              <RefreshCw className="h-4 w-4" />
            </Button>
        </div>

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
        </div>
      </div>
    </AppShell>
  )
}
