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
import { Slider } from '@/components/ui/slider'
import {
  Settings2,
  ChevronDown,
  ChevronRight,
  RotateCcw,
  Loader2,
  Shield,
  ShieldCheck,
  ShieldAlert,
  WandSparkles,
} from 'lucide-react'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
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

  const privacyLevel = config.privacy ?? 'internal'

  // Empty string in the Select maps back to `undefined` on the config
  // so the backend uses the global parser_engine setting. This keeps
  // "Use default" as a first-class option rather than overloading an
  // engine value.
  const parserEngineValue = config.parser_engine ?? ''
  const minConfidence = config.docling_min_confidence ?? 0.95

  const content = (
    <div className="space-y-4">
      {/* Parser engine override (A.2) */}
      <div className="space-y-2">
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide flex items-center gap-1.5">
          <WandSparkles className="h-3.5 w-3.5" />
          Parser engine for this run
        </h4>
        <Select
          value={parserEngineValue || 'default'}
          onValueChange={(v) =>
            update('parser_engine', v === 'default' ? undefined : (v as 'simple' | 'docling' | 'mineru' | 'auto'))
          }
        >
          <SelectTrigger className="h-8 text-xs" aria-label="Parser engine for this run">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="default" className="text-xs">
              Use default (current global setting)
            </SelectItem>
            <SelectItem value="simple" className="text-xs">Simple</SelectItem>
            <SelectItem value="docling" className="text-xs">Docling</SelectItem>
            <SelectItem value="mineru" className="text-xs">MinerU</SelectItem>
            <SelectItem value="auto" className="text-xs">Auto (docling with MinerU fallback)</SelectItem>
          </SelectContent>
        </Select>
        {config.parser_engine === 'auto' && (
          <div className="space-y-1 pt-1">
            <Label className="text-xs">
              Confidence threshold: {minConfidence.toFixed(2)}
            </Label>
            <Slider
              min={0}
              max={1}
              step={0.01}
              value={[minConfidence]}
              onValueChange={(value) => update('docling_min_confidence', value[0])}
              className="w-full"
              aria-label="Docling confidence threshold"
            />
          </div>
        )}
      </div>

      {/* Privacy & Model Routing */}
      <div className="space-y-2">
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide flex items-center gap-1.5">
          <Shield className="h-3.5 w-3.5" />
          Privacy & Model
        </h4>
        <RadioGroup
          value={privacyLevel}
          onValueChange={(v) => update('privacy', v as 'public' | 'internal' | 'confidential')}
          className="space-y-1.5"
        >
          <div className="flex items-center gap-2">
            <RadioGroupItem value="public" id="privacy-public" className="h-3.5 w-3.5" />
            <Label htmlFor="privacy-public" className="text-xs flex items-center gap-1.5 cursor-pointer">
              <ShieldCheck className="h-3.5 w-3.5 text-green-500" />
              Public
              <span className="text-muted-foreground">— Cloud API (sneller, beter)</span>
            </Label>
          </div>
          <div className="flex items-center gap-2">
            <RadioGroupItem value="internal" id="privacy-internal" className="h-3.5 w-3.5" />
            <Label htmlFor="privacy-internal" className="text-xs flex items-center gap-1.5 cursor-pointer">
              <Shield className="h-3.5 w-3.5 text-blue-500" />
              Internal
              <span className="text-muted-foreground">— Lokaal (data blijft hier)</span>
            </Label>
          </div>
          <div className="flex items-center gap-2">
            <RadioGroupItem value="confidential" id="privacy-confidential" className="h-3.5 w-3.5" />
            <Label htmlFor="privacy-confidential" className="text-xs flex items-center gap-1.5 cursor-pointer">
              <ShieldAlert className="h-3.5 w-3.5 text-orange-500" />
              Confidentieel
              <span className="text-muted-foreground">— Lokaal, minimale logging</span>
            </Label>
          </div>
        </RadioGroup>
        <div className="text-[10px] text-muted-foreground mt-1 pl-5">
          {privacyLevel === 'public' && 'Extractie en samenvatting via NVIDIA Mistral Medium. Parsing altijd lokaal.'}
          {privacyLevel === 'internal' && 'Alles lokaal via Ollama. Data verlaat de machine niet.'}
          {privacyLevel === 'confidential' && 'Lokaal met minimale logging. Voor vertrouwelijke documenten.'}
        </div>
      </div>

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
            aria-label="Export extracted images"
          />
          <span className="text-xs">Export extracted images</span>
        </div>
        <div className="flex items-center gap-2">
          <Switch
            checked={config.docling_generate_page_images ?? false}
            onCheckedChange={(v) => update('docling_generate_page_images', v)}
            className="scale-75"
            aria-label="Generate full-page images"
          />
          <span className="text-xs">Generate full-page images</span>
        </div>
        <div className="flex items-center gap-2">
          <Switch
            checked={config.docling_do_picture_classification ?? isVlm}
            onCheckedChange={(v) => update('docling_do_picture_classification', v)}
            className="scale-75"
            aria-label="Classify pictures by type"
          />
          <span className="text-xs">Classify pictures by type</span>
        </div>
        <div className="space-y-1">
          <Label className="text-xs">
            Scale: {(config.docling_image_scale ?? 2.0).toFixed(1)}x
          </Label>
          <Slider
            min={1}
            max={4}
            step={0.5}
            value={[config.docling_image_scale ?? 2.0]}
            onValueChange={(value) => update('docling_image_scale', value[0])}
            className="w-full"
            aria-label="Image extraction scale"
          />
        </div>
      </div>

      {/* Enrichment */}
      <div className="space-y-2">
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Enrichment</h4>
        <div className="flex items-center gap-2">
          <Switch
            checked={config.docling_do_code_enrichment ?? true}
            onCheckedChange={(v) => update('docling_do_code_enrichment', v)}
            className="scale-75"
            aria-label="Enrich code blocks"
          />
          <span className="text-xs">Enrich code blocks</span>
        </div>
        <div className="flex items-center gap-2">
          <Switch
            checked={config.docling_do_formula_enrichment ?? true}
            onCheckedChange={(v) => update('docling_do_formula_enrichment', v)}
            className="scale-75"
            aria-label="Extract math formulas"
          />
          <span className="text-xs">Extract math formulas (LaTeX)</span>
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
