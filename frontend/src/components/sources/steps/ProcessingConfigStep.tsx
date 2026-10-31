"use client"

import { useState } from "react"
import { FormSection } from "@/components/ui/form-section"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { SettingsResponse } from "@/lib/types/api"

interface ProcessingConfigStepProps {
  settings?: SettingsResponse
  overrides: Record<string, unknown>
  onOverridesChange: (overrides: Record<string, unknown>) => void
}

export function ProcessingConfigStep({
  settings,
  overrides,
  onOverridesChange,
}: ProcessingConfigStepProps) {
  // Get effective value: override if set, else settings default
  const getValue = <T,>(key: keyof SettingsResponse): T | undefined => {
    if (key in overrides) return overrides[key] as T
    return settings?.[key] as T | undefined
  }

  const setValue = (key: string, value: unknown) => {
    onOverridesChange({ ...overrides, [key]: value })
  }

  // Local string state for OCR languages to allow typing commas
  const [ocrLangsText, setOcrLangsText] = useState(
    () => (getValue<string[]>("docling_ocr_languages") || ["en"]).join(", ")
  )

  const pipeline = getValue<string>("docling_pipeline") || "auto"

  return (
    <div className="space-y-8">
      {/* Document Processing */}
      <FormSection
        title="Document Processing"
        description="Configure the document processing engine and pipeline."
      >
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label className="text-sm">Parser Engine</Label>
            <Select
              value={getValue<string>("parser_engine") || "docling"}
              onValueChange={(v) => setValue("parser_engine", v)}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="docling">Docling</SelectItem>
                <SelectItem value="simple">Simple</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label className="text-sm">Pipeline</Label>
            <Select
              value={pipeline}
              onValueChange={(v) => setValue("docling_pipeline", v)}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">Auto</SelectItem>
                <SelectItem value="standard">Standard</SelectItem>
                <SelectItem value="vlm">VLM</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {pipeline === "vlm" && (
            <>
              <div className="space-y-2">
                <Label className="text-sm">VLM Model</Label>
                <Select
                  value={getValue<string>("docling_vlm_model") || "granite-docling-258m"}
                  onValueChange={(v) => setValue("docling_vlm_model", v)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="granite-docling-258m">Granite Docling 258M</SelectItem>
                    <SelectItem value="smoldocling-256m">SmolDocling 256M</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label className="text-sm">VLM Framework</Label>
                <Select
                  value={getValue<string>("docling_vlm_framework") || "auto"}
                  onValueChange={(v) => setValue("docling_vlm_framework", v)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">Auto</SelectItem>
                    <SelectItem value="transformers">Transformers</SelectItem>
                    <SelectItem value="mlx">MLX</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </>
          )}
        </div>
      </FormSection>

      {/* OCR Settings */}
      <FormSection
        title="OCR Settings"
        description="Configure optical character recognition for scanned documents."
      >
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label className="text-sm">OCR Engine</Label>
            <Select
              value={getValue<string>("docling_ocr_engine") || "auto"}
              onValueChange={(v) => setValue("docling_ocr_engine", v)}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">Auto</SelectItem>
                <SelectItem value="easyocr">EasyOCR</SelectItem>
                <SelectItem value="rapidocr">RapidOCR</SelectItem>
                <SelectItem value="tesseract">Tesseract</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label className="text-sm">Table Mode</Label>
            <Select
              value={getValue<string>("docling_table_mode") || "accurate"}
              onValueChange={(v) => setValue("docling_table_mode", v)}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="accurate">Accurate</SelectItem>
                <SelectItem value="fast">Fast</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2 col-span-2">
            <Label className="text-sm">OCR Languages (comma-separated)</Label>
            <Input
              value={ocrLangsText}
              onChange={(e) => setOcrLangsText(e.target.value)}
              onBlur={() => {
                const langs = ocrLangsText.split(",").map((s) => s.trim()).filter(Boolean)
                setValue("docling_ocr_languages", langs.length > 0 ? langs : ["en"])
              }}
              placeholder="en, de, fr"
            />
          </div>

          <div className="flex items-center justify-between col-span-2">
            <div>
              <Label className="text-sm">OCR GPU Acceleration</Label>
              <p className="text-xs text-muted-foreground">Use GPU for OCR processing</p>
            </div>
            <Switch
              checked={getValue<boolean>("docling_ocr_use_gpu") ?? false}
              onCheckedChange={(v) => setValue("docling_ocr_use_gpu", v)}
            />
          </div>
        </div>
      </FormSection>

      {/* GPU Settings */}
      <FormSection
        title="GPU"
        description="GPU acceleration for the document processing pipeline."
      >
        <div className="grid grid-cols-2 gap-4">
          <div className="flex items-center justify-between col-span-2">
            <div>
              <Label className="text-sm">Enable GPU</Label>
              <p className="text-xs text-muted-foreground">Use GPU for document processing</p>
            </div>
            <Switch
              checked={getValue<boolean>("docling_gpu_enabled") ?? false}
              onCheckedChange={(v) => setValue("docling_gpu_enabled", v)}
            />
          </div>

          <div className="space-y-2">
            <Label className="text-sm">GPU Device</Label>
            <Select
              value={getValue<string>("docling_gpu_device") || "auto"}
              onValueChange={(v) => setValue("docling_gpu_device", v)}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">Auto</SelectItem>
                <SelectItem value="cuda">CUDA</SelectItem>
                <SelectItem value="cpu">CPU</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </FormSection>

      {/* File Management */}
      <FormSection
        title="Files & Directories"
        description="Configure file paths and output options."
      >
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2 col-span-2">
            <Label className="text-sm">Input Directory</Label>
            <Input
              value={getValue<string>("input_directory_path") || ""}
              onChange={(e) => setValue("input_directory_path", e.target.value)}
              placeholder="/path/to/input"
            />
          </div>

          <div className="space-y-2">
            <Label className="text-sm">Markdown Directory</Label>
            <Input
              value={getValue<string>("markdown_directory_path") || ""}
              onChange={(e) => setValue("markdown_directory_path", e.target.value)}
              placeholder="/path/to/markdown"
            />
          </div>

          <div className="space-y-2">
            <Label className="text-sm">Output Directory</Label>
            <Input
              value={getValue<string>("output_directory_path") || ""}
              onChange={(e) => setValue("output_directory_path", e.target.value)}
              placeholder="/path/to/output"
            />
          </div>

          <div className="space-y-2">
            <Label className="text-sm">File Operation</Label>
            <Select
              value={getValue<string>("file_operation") || "copy"}
              onValueChange={(v) => setValue("file_operation", v)}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="copy">Copy</SelectItem>
                <SelectItem value="move">Move</SelectItem>
                <SelectItem value="none">None</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label className="text-sm">Output Naming</Label>
            <Select
              value={getValue<string>("output_naming_scheme") || "original"}
              onValueChange={(v) => setValue("output_naming_scheme", v)}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="original">Original</SelectItem>
                <SelectItem value="timestamp_prefix">Timestamp Prefix</SelectItem>
                <SelectItem value="date_prefix">Date Prefix</SelectItem>
                <SelectItem value="datetime_suffix">DateTime Suffix</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </FormSection>

      {/* Images */}
      <FormSection
        title="Images"
        description="Image export settings for document processing."
      >
        <div className="grid grid-cols-2 gap-4">
          <div className="flex items-center justify-between col-span-2">
            <div>
              <Label className="text-sm">Auto Export Images</Label>
              <p className="text-xs text-muted-foreground">Extract images from documents automatically</p>
            </div>
            <Switch
              checked={getValue<boolean>("docling_auto_export_images") ?? false}
              onCheckedChange={(v) => setValue("docling_auto_export_images", v)}
            />
          </div>

          <div className="space-y-2">
            <Label className="text-sm">Image Scale (1.0 - 4.0)</Label>
            <Input
              type="number"
              min={1}
              max={4}
              step={0.5}
              value={getValue<number>("docling_image_scale") ?? 2.0}
              onChange={(e) => setValue("docling_image_scale", parseFloat(e.target.value) || 2.0)}
            />
          </div>
        </div>
      </FormSection>

    </div>
  )
}
