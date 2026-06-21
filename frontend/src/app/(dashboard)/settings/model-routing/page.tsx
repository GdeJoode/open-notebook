'use client'

import { useEffect, useMemo, useState } from 'react'

import { AppShell } from '@/components/layout/AppShell'
import { PrivacyModeToggle } from '@/components/routing/PrivacyModeToggle'
import { ProviderChainEditor } from '@/components/routing/ProviderChainEditor'
import { ProviderHealthChip } from '@/components/routing/ProviderHealthChip'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import {
  useModelRouteHealth,
  useModelRoutes,
  useUpdateModelRoute,
} from '@/lib/hooks/use-model-routing'
import { useSettings, useUpdateSettings } from '@/lib/hooks/use-settings'
import {
  ModelRoute,
  PrivacyValue,
  ProviderChainEntry,
  RoutingTask,
} from '@/lib/types/model-routing'
import { toWire } from '@/lib/utils/privacy-mode'
import { AlertTriangle } from 'lucide-react'

const TASK_LABELS: Record<RoutingTask, string> = {
  entity_extraction: 'Entity Extraction',
  summarization: 'Summarization',
  chat: 'Chat',
}

export default function ModelRoutingPage() {
  const routesQuery = useModelRoutes()
  const healthQuery = useModelRouteHealth()
  const settingsQuery = useSettings()
  const updateRoute = useUpdateModelRoute()
  const updateSettings = useUpdateSettings()

  if (routesQuery.isLoading) {
    return (
      <AppShell>
        <div className="p-6 space-y-6" aria-busy="true">
          <Skeleton className="h-8 w-64" />
          <Skeleton className="h-40 w-full" />
          <Skeleton className="h-40 w-full" />
        </div>
      </AppShell>
    )
  }

  if (routesQuery.isError || !routesQuery.data) {
    return (
      <AppShell>
        <div className="p-6">
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Failed to load model routing</AlertTitle>
            <AlertDescription>
              Could not fetch the provider chains. Check that the API is
              reachable and try again.
              <div className="mt-3">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => routesQuery.refetch()}
                >
                  Retry
                </Button>
              </div>
            </AlertDescription>
          </Alert>
        </div>
      </AppShell>
    )
  }

  const globalPrivacy: 'cloud' | 'private' =
    settingsQuery.data?.default_privacy_mode ?? 'cloud'

  return (
    <AppShell>
      <div className="flex-1 overflow-y-auto">
        <div className="p-6 space-y-6">
          <div>
            <h1 className="text-2xl font-bold">Model Routing</h1>
            <p className="text-muted-foreground mt-1">
              Order the provider chain each LLM stage tries, set the global
              privacy default, and watch per-provider health.
            </p>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Global Privacy Default</CardTitle>
              <CardDescription>
                Cloud routes through cloud providers with a local fallback.
                Private keeps every LLM call local. Notebooks and documents can
                override this.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <PrivacyModeToggle
                scope="global"
                scopeLabel="Global default"
                value={globalPrivacy}
                disabled={updateSettings.isPending}
                onChange={(value: PrivacyValue) => {
                  const wire = toWire(value)
                  if (wire) {
                    updateSettings.mutate({ default_privacy_mode: wire })
                  }
                }}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Provider Health</CardTitle>
              <CardDescription>
                Configured providers (API key present) and their circuit-breaker
                state. Unconfigured cloud providers are skipped at routing time.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {healthQuery.isLoading ? (
                <Skeleton className="h-8 w-full" />
              ) : healthQuery.data ? (
                <div className="flex flex-wrap gap-2">
                  {healthQuery.data.providers.map((h) => (
                    <ProviderHealthChip key={h.provider} health={h} />
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">
                  Health unavailable.
                </p>
              )}
            </CardContent>
          </Card>

          {routesQuery.data.map((route) => (
            <TaskRouteCard
              key={route.task}
              route={route}
              saving={updateRoute.isPending}
              onSave={(provider_chain, private_chain) =>
                updateRoute.mutate({
                  task: route.task,
                  update: { provider_chain, private_chain },
                })
              }
            />
          ))}
        </div>
      </div>
    </AppShell>
  )
}

interface TaskRouteCardProps {
  route: ModelRoute
  saving: boolean
  onSave: (
    providerChain: ProviderChainEntry[],
    privateChain: ProviderChainEntry[],
  ) => void
}

function TaskRouteCard({ route, saving, onSave }: TaskRouteCardProps) {
  const [chain, setChain] = useState<ProviderChainEntry[]>(route.provider_chain)

  // Re-sync local edit state when the server route changes (e.g. after a save
  // reload or another task's mutation invalidates the list).
  useEffect(() => {
    setChain(route.provider_chain)
  }, [route.provider_chain])

  const dirty = useMemo(
    () => JSON.stringify(chain) !== JSON.stringify(route.provider_chain),
    [chain, route.provider_chain],
  )

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle>{TASK_LABELS[route.task]}</CardTitle>
            <CardDescription>
              {route.is_default
                ? 'Default chain (read-only until saved). Reorder and save to customize.'
                : 'Cloud-mode provider chain. A local fallback is always appended at routing time.'}
            </CardDescription>
          </div>
          {route.is_default && (
            <span className="rounded-md border border-dashed px-2 py-0.5 text-xs text-muted-foreground">
              Default
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <ProviderChainEditor
          taskLabel={TASK_LABELS[route.task]}
          chain={chain}
          onChange={setChain}
          disabled={saving}
        />
        <div className="flex justify-end gap-2">
          {dirty && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setChain(route.provider_chain)}
              disabled={saving}
            >
              Reset
            </Button>
          )}
          <Button
            size="sm"
            onClick={() => onSave(chain, route.private_chain)}
            disabled={saving || (!dirty && !route.is_default)}
          >
            {route.is_default ? 'Save to customize' : 'Save'}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
