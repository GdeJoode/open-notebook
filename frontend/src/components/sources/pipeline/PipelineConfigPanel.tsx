'use client'

import { useState } from 'react'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import {
  Settings2,
  ChevronDown,
  ChevronRight,
  RotateCcw,
  Loader2,
} from 'lucide-react'
import type { DoclingPipelineConfig } from '@/lib/api/sources'
import { DEFAULT_PIPELINE_CONFIG } from '@/lib/api/sources'

interface PipelineConfigPanelProps {
  config: DoclingPipelineConfig
  onChange: (config: DoclingPipelineConfig) => void
  onReprocess?: () => void
  reprocessing?: boolean
  collapsible?: boolean
  defaultOpen?: boolean
}

export function PipelineConfigPanel({
  config,
  onChange,
  onReprocess,
  reprocessing = false,
  collapsible = true,
  defaultOpen = false,
}: PipelineConfigPanelProps) {
  const [open, setOpen] = useState(defaultOpen)

  const update = <K extends keyof DoclingPipelineConfig>(
    key: K,
    value: DoclingPipelineConfig[K]
  ) => {
    onChange({ ...config, [key]: value })
  }

  const isVlm = config.docling_pipeline === 'vlm'

  const content = (
    <div className="space-y-4">
      {/* OCR Section */}
      <div className="space-y-2">
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">OCR</h4>
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1">
            <Label className="text-xs">Engine</Label>
            <Select
              value={config.docling_ocr_engine ?? 'easyocr'}
              onValueChange={(v) => update('docling_ocr_engine', v)}
            >
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="easyocr" className="text-xs">EasyOCR</SelectItem>
                <SelectItem value="rapidocr" className="text-xs">RapidOCR</SelectItem>
                <SelectItem value="tesseract" className="text-xs">Tesseract</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Languages</Label>
            <Input
              className="h-8 text-xs"
              value={(config.docling_ocr_languages ?? ['en']).join(', ')}
              onChange={(e) => update(
                'docling_ocr_languages',
                e.target.value.split(',').map(s => s.trim()).filter(Boolean)
              )}
              placeholder="en, nl"
            />
          </div>
        </div>
      </div>

      {/* Table Extraction */}
      <div className="space-y-2">
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Table Extraction</h4>
        <div className="space-y-1">
          <Label className="text-xs">Mode</Label>
          <Select
            value={config.docling_table_mode ?? 'accurate'}
            onValueChange={(v) => update('docling_table_mode', v)}
          >
            <SelectTrigger className="h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="accurate" className="text-xs">
                Accurate (TableFormer)
              </SelectItem>
              <SelectItem value="fast" className="text-xs">
                Fast
              </SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* VLM Pipeline */}
      <div className="space-y-2">
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Vision Language Model</h4>
        <div className="flex items-center gap-2">
          <Switch
            checked={isVlm}
            onCheckedChange={(v) => update('docling_pipeline', v ? 'vlm' : 'standard')}
            className="scale-75"
          />
          <span className="text-xs">Enable VLM pipeline</span>
          {isVlm && <Badge variant="secondary" className="text-[10px]">Picture classification + description</Badge>}
        </div>
        {isVlm && (
          <div className="space-y-1">
            <Label className="text-xs">Model</Label>
            <Select
              value={config.docling_vlm_model ?? 'granite-docling-258m'}
              onValueChange={(v) => update('docling_vlm_model', v)}
            >
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="granite-docling-258m" className="text-xs">
                  Granite Docling 258M
                </SelectItem>
                <SelectItem value="smoldocling-256m" className="text-xs">
                  SmolDocling 256M
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
        )}
      </div>

      {/* Image Extraction */}
      <div className="space-y-2">
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Images</h4>
        <div className="flex items-center gap-2">
          <Switch
            checked={config.docling_auto_export_images ?? true}
            onCheckedChange={(v) => update('docling_auto_export_images', v)}
            className="scale-75"
          />
          <span className="text-xs">Export extracted images</span>
        </div>
        <div className="space-y-1">
          <Label className="text-xs">Scale</Label>
          <Select
            value={String(config.docling_image_scale ?? 2.0)}
            onValueChange={(v) => update('docling_image_scale', parseFloat(v))}
          >
            <SelectTrigger className="h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="0.5" className="text-xs">0.5x (small)</SelectItem>
              <SelectItem value="1.0" className="text-xs">1.0x (standard)</SelectItem>
              <SelectItem value="1.5" className="text-xs">1.5x (high)</SelectItem>
              <SelectItem value="2.0" className="text-xs">2.0x (maximum)</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Chunking */}
      <div className="space-y-2">
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Chunking</h4>
        <div className="flex items-center gap-2">
          <Switch
            checked={config.docling_chunking_enabled ?? true}
            onCheckedChange={(v) => update('docling_chunking_enabled', v)}
            className="scale-75"
          />
          <span className="text-xs">Enable chunking</span>
        </div>
        {(config.docling_chunking_enabled ?? true) && (
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label className="text-xs">Method</Label>
              <Select
                value={config.docling_chunking_method ?? 'hybrid'}
                onValueChange={(v) => update('docling_chunking_method', v)}
              >
                <SelectTrigger className="h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="hybrid" className="text-xs">Hybrid</SelectItem>
                  <SelectItem value="hierarchical" className="text-xs">Hierarchical</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1">
              <Label className="text-xs">Max tokens</Label>
              <Input
                type="number"
                className="h-8 text-xs"
                value={config.docling_chunking_max_tokens ?? 512}
                onChange={(e) => update('docling_chunking_max_tokens', parseInt(e.target.value) || 512)}
                min={64}
                max={4096}
              />
            </div>
          </div>
        )}
      </div>

      {/* Reprocess button */}
      {onReprocess && (
        <div className="pt-2 border-t">
          <Button
            onClick={onReprocess}
            disabled={reprocessing}
            size="sm"
            className="w-full gap-2"
          >
            {reprocessing ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <RotateCcw className="h-3.5 w-3.5" />
            )}
            {reprocessing ? 'Reprocessing...' : 'Reprocess Document'}
          </Button>
        </div>
      )}
    </div>
  )

  if (!collapsible) return content

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="flex items-center gap-2 w-full px-3 py-2 border rounded-md bg-muted/50 hover:bg-muted/80 transition-colors text-left">
        <Settings2 className="h-4 w-4 text-muted-foreground" />
        <span className="text-xs font-medium flex-1">Docling Pipeline Configuration</span>
        <Badge variant="secondary" className="text-[10px]">
          {isVlm ? 'VLM' : 'Standard'} · {config.docling_table_mode ?? 'accurate'}
        </Badge>
        {open ? (
          <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
        )}
      </CollapsibleTrigger>
      <CollapsibleContent className="pt-3 px-1">
        {content}
      </CollapsibleContent>
    </Collapsible>
  )
}
