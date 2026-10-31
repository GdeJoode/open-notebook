'use client'

import { useEffect, useMemo, useState } from 'react'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { Settings2, Sparkles } from 'lucide-react'
import { useModels } from '@/lib/hooks/use-models'
import { modellenVanSoort, modelVoorTaak } from '@/lib/utils/modelregister'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'

interface ModelSelectorProps {
  currentModel?: string
  onModelChange: (model?: string) => void
  disabled?: boolean
}

export function ModelSelector({ 
  currentModel, 
  onModelChange,
  disabled = false 
}: ModelSelectorProps) {
  const [open, setOpen] = useState(false)
  const [selectedModel, setSelectedModel] = useState(currentModel || 'default')
  const { data: models, isLoading } = useModels()

  useEffect(() => {
    setSelectedModel(currentModel || 'default')
  }, [currentModel])

  // Taalmodellen uit het register, op naam gesorteerd (ADR 0007)
  const languageModels = useMemo(
    () => [...modellenVanSoort(models, 'language')].sort((a, b) => a.naam.localeCompare(b.naam)),
    [models]
  )

  // De standaard is het registermodel voor chat
  const defaultModel = useMemo(() => modelVoorTaak(models, 'chat'), [models])

  const currentModelName = useMemo(() => {
    if (currentModel) {
      return languageModels.find(model => model.naam === currentModel)?.naam || currentModel
    }
    if (defaultModel) {
      return defaultModel.naam
    }
    return 'Default Model'
  }, [currentModel, languageModels, defaultModel])

  const handleSave = () => {
    onModelChange(selectedModel === 'default' ? undefined : selectedModel)
    setOpen(false)
  }

  const handleReset = () => {
    setSelectedModel('default')
    onModelChange(undefined)
    setOpen(false)
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button 
          variant="outline" 
          size="sm"
          disabled={disabled}
          className="gap-2"
        >
          <Settings2 className="h-4 w-4" />
          <span className="text-xs">
            {currentModelName}
          </span>
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[min(425px,100vw-2rem)]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            Model Configuration
          </DialogTitle>
          <DialogDescription>
            Override the default model for this chat session. Leave empty to use the system default.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid gap-2">
            <Label htmlFor="model">Model</Label>
            <Select value={selectedModel} onValueChange={setSelectedModel}>
              <SelectTrigger id="model">
                <SelectValue placeholder="Select a model (or use default)" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="default">
                  <div className="flex items-center justify-between w-full">
                    <span>
                      {defaultModel ? `Default (${defaultModel.naam})` : 'System Default'}
                    </span>
                    {defaultModel?.locatie && (
                      <span className="text-xs text-muted-foreground ml-2">
                        {defaultModel.locatie}
                      </span>
                    )}
                  </div>
                </SelectItem>
                {isLoading ? (
                  <div className="flex items-center justify-center py-2">
                    <LoadingSpinner size="sm" />
                  </div>
                ) : (
                  languageModels.map((model) => (
                    <SelectItem key={model.naam} value={model.naam}>
                      <div className="flex items-center justify-between w-full">
                        <span>{model.naam}</span>
                        <span className="text-xs text-muted-foreground ml-2">
                          {model.locatie}
                        </span>
                      </div>
                    </SelectItem>
                  ))
                )}
              </SelectContent>
            </Select>
          </div>
          {selectedModel && selectedModel !== 'default' && (
            <div className="rounded-lg bg-muted p-3">
              <p className="text-sm text-muted-foreground">
                This session will use <strong>{languageModels.find(m => m.naam === selectedModel)?.naam}</strong> instead of the default model.
              </p>
            </div>
          )}
        </div>
        <DialogFooter className="flex justify-between">
          <Button variant="outline" onClick={handleReset}>
            Reset to Default
          </Button>
          <Button onClick={handleSave}>
            Save Changes
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
