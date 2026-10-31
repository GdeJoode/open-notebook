'use client'

import { useModels } from '@/lib/hooks/use-models'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { RefreshCw } from 'lucide-react'
import { PageBody } from '@/components/layout/PageBody'
import { PageHeader } from '@/components/layout/PageHeader'

/**
 * Het modelregister, alleen lezen (ADR 0007). Modellen toevoegen of een
 * standaard kiezen kan hier niet: het register staat in
 * `config/modelregister.yaml` en verandert met een commit en een herstart.
 */
export default function ModelsPage() {
  const { data: models, isLoading, isError, refetch } = useModels()

  const header = (
    <PageHeader
      title="Modellen"
      subtitle="Het modelregister (config/modelregister.yaml); alleen lezen"
      actions={
        <Button variant="outline" size="sm" onClick={() => refetch()} aria-label="Ververs modellen">
          <RefreshCw className="h-4 w-4" />
        </Button>
      }
    />
  )

  if (isLoading) {
    return (
      <PageBody>
        {header}
        <div className="flex items-center justify-center min-h-[60vh]">
          <LoadingSpinner size="lg" />
        </div>
      </PageBody>
    )
  }

  if (isError || !models) {
    return (
      <PageBody>
        {header}
        <div className="text-center py-12">
          <p className="text-muted-foreground">Het modelregister kon niet worden gelezen</p>
        </div>
      </PageBody>
    )
  }

  return (
    <PageBody>
      {header}
      <div className="overflow-x-auto rounded-md border">
        <table className="w-full text-sm">
          <thead className="bg-muted/50 text-left">
            <tr>
              <th className="px-3 py-2 font-medium">Naam</th>
              <th className="px-3 py-2 font-medium">Aanroepwijze</th>
              <th className="px-3 py-2 font-medium">Locatie</th>
              <th className="px-3 py-2 font-medium">Intern toegestaan</th>
              <th className="px-3 py-2 font-medium">Taken</th>
              <th className="px-3 py-2 font-medium">Versie</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr key={model.naam} className="border-t align-top">
                <td className="px-3 py-2 font-medium">{model.naam}</td>
                <td className="px-3 py-2">{model.aanroepwijze}</td>
                <td className="px-3 py-2">{model.locatie}</td>
                <td className="px-3 py-2">{model.toegestaan_intern}</td>
                <td className="px-3 py-2">
                  <div className="flex flex-wrap gap-1">
                    {model.taken.map((taak) => (
                      <Badge key={taak} variant="secondary">{taak}</Badge>
                    ))}
                  </div>
                </td>
                <td className="px-3 py-2 break-all text-xs text-muted-foreground">{model.versie}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </PageBody>
  )
}
