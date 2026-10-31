'use client'

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Label } from '@/components/ui/label'
import { useModels } from '@/lib/hooks/use-models'
import { modellenVanSoort } from '@/lib/utils/modelregister'
import { ModelType } from '@/lib/types/models'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'

interface ModelSelectorProps {
  label?: string
  modelType: ModelType
  value: string
  onChange: (value: string) => void
  placeholder?: string
  disabled?: boolean
}

export function ModelSelector({ 
  label, 
  modelType, 
  value, 
  onChange, 
  placeholder = 'Select a model',
  disabled = false 
}: ModelSelectorProps) {
  const { data: models, isLoading } = useModels()
  
  // Modellen uit het register met een taak van deze soort (ADR 0007)
  const filteredModels = modellenVanSoort(models, modelType)
  return (
    <div className="space-y-2">
      {label && <Label>{label}</Label>}
      <Select value={value} onValueChange={onChange} disabled={disabled || isLoading}>
        <SelectTrigger>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-2">
              <LoadingSpinner size="sm" />
            </div>
          ) : filteredModels.length === 0 ? (
            <div className="text-sm text-muted-foreground py-2 px-2">
              No {modelType.replace('_', ' ')} models available
            </div>
          ) : (
            filteredModels.map((model) => (
              <SelectItem key={model.naam} value={model.naam}>
                <div className="flex items-center justify-between w-full">
                  <span>{model.naam}</span>
                  <span className="text-xs text-muted-foreground ml-2">{model.locatie}</span>
                </div>
              </SelectItem>
            ))
          )}
        </SelectContent>
      </Select>
    </div>
  )
}
