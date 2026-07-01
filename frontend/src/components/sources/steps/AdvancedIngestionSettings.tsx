"use client"

import { ChevronRight, Settings2 } from "lucide-react"

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

/**
 * Power-user parser overrides on the create-flow Input screen (Track UX.5).
 *
 * This is the collapsed "Advanced ingestion settings" disclosure. Its whole
 * reason to exist is to keep parser/OCR config OUT of the mandatory flow: every
 * field defaults to `Auto` (meaning "let the backend auto-route"). The full
 * parser config surface lives on the post-ingest Reprocess dialog (its primary,
 * reversible home).
 *
 * The component is intentionally PRESENTATIONAL and fully CONTROLLED: the parent
 * (`CreateSourcePipeline`) owns the field state and the open/collapsed state.
 * That is what lets a user's selection survive back-navigation (Input →
 * Organize → Back), which conditionally unmounts this child — an earlier version
 * held the state internally and reset every field to `Auto` on mount, silently
 * wiping the user's parser/OCR/table choice. Lifting the state fixes that: a
 * remount now rehydrates from the parent instead of clearing.
 *
 * The "all-Auto ⇒ no overrides" and "emit only changed fields" contract now
 * lives in the parent, which derives `processing_overrides` from these choices.
 */

export type ParserEngineChoice = "auto" | "docling" | "simple"
export type OcrEngineChoice = "auto" | "easyocr" | "rapidocr" | "tesseract"
export type TableModeChoice = "auto" | "accurate" | "fast"

export interface AdvancedIngestionSettingsProps {
  parserEngine: ParserEngineChoice
  ocrEngine: OcrEngineChoice
  tableMode: TableModeChoice
  onParserEngineChange: (value: ParserEngineChoice) => void
  onOcrEngineChange: (value: OcrEngineChoice) => void
  onTableModeChange: (value: TableModeChoice) => void
  /** Controlled disclosure state (lifted so it survives back-navigation). */
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function AdvancedIngestionSettings({
  parserEngine,
  ocrEngine,
  tableMode,
  onParserEngineChange,
  onOcrEngineChange,
  onTableModeChange,
  open,
  onOpenChange,
}: AdvancedIngestionSettingsProps) {
  return (
    <Collapsible
      open={open}
      onOpenChange={onOpenChange}
      className="rounded-md border bg-muted/30"
    >
      <CollapsibleTrigger
        className="group flex w-full items-center gap-2 px-3 py-2 text-left text-sm font-medium outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-md"
        aria-label="Advanced ingestion settings"
      >
        <ChevronRight
          className="size-4 shrink-0 text-muted-foreground transition-transform group-data-[state=open]:rotate-90"
          aria-hidden
        />
        <Settings2 className="size-4 shrink-0 text-muted-foreground" aria-hidden />
        <span className="flex-1">Advanced ingestion settings</span>
        <span className="text-xs font-normal text-muted-foreground">
          Optional — defaults to Auto
        </span>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="space-y-4 border-t px-3 py-3">
          <p className="text-xs text-muted-foreground">
            Override the parser for this document only. Leave everything on Auto
            to let the backend choose. The full parser/OCR config lives on the
            source&apos;s Reprocess dialog after ingestion.
          </p>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <div className="space-y-1.5">
              <Label className="text-xs" htmlFor="adv-parser-engine">
                Parser engine
              </Label>
              <Select
                value={parserEngine}
                onValueChange={(v) => onParserEngineChange(v as ParserEngineChoice)}
              >
                <SelectTrigger id="adv-parser-engine" aria-label="Parser engine">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">Auto</SelectItem>
                  <SelectItem value="docling">Docling</SelectItem>
                  <SelectItem value="simple">Simple</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <Label className="text-xs" htmlFor="adv-ocr-engine">
                OCR engine
              </Label>
              <Select
                value={ocrEngine}
                onValueChange={(v) => onOcrEngineChange(v as OcrEngineChoice)}
              >
                <SelectTrigger id="adv-ocr-engine" aria-label="OCR engine">
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

            <div className="space-y-1.5">
              <Label className="text-xs" htmlFor="adv-table-mode">
                Table mode
              </Label>
              <Select
                value={tableMode}
                onValueChange={(v) => onTableModeChange(v as TableModeChoice)}
              >
                <SelectTrigger id="adv-table-mode" aria-label="Table mode">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">Auto</SelectItem>
                  <SelectItem value="accurate">Accurate</SelectItem>
                  <SelectItem value="fast">Fast</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

export default AdvancedIngestionSettings
