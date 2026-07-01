"use client"

import { useCallback, useEffect, useState } from "react"
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
import type { SettingsResponse } from "@/lib/types/api"

/**
 * Power-user parser overrides on the create-flow Input screen (Track UX.5).
 *
 * This is the collapsed "Advanced ingestion settings" disclosure. Its whole
 * reason to exist is to keep parser/OCR config OUT of the mandatory flow: every
 * field defaults to `Auto` (meaning "let the backend auto-route"), and the
 * component only contributes a `processing_overrides` key for a field the user
 * actually changed away from `Auto`. When nothing is changed it contributes an
 * empty object, so the create call sends no overrides and the backend picks the
 * engine itself. The full parser config surface lives on the post-ingest
 * Reprocess dialog (its primary, reversible home).
 */

export interface AdvancedIngestionSettingsProps {
  /**
   * Called whenever the effective overrides change. Receives only the fields
   * the user moved off `Auto`; an empty object means "no overrides".
   */
  onOverridesChange: (overrides: Partial<SettingsResponse>) => void
  defaultOpen?: boolean
}

type ParserEngineChoice = "auto" | "docling" | "simple"
type OcrEngineChoice = "auto" | "easyocr" | "rapidocr" | "tesseract"
type TableModeChoice = "auto" | "accurate" | "fast"

const AUTO = "auto"

export function AdvancedIngestionSettings({
  onOverridesChange,
  defaultOpen = false,
}: AdvancedIngestionSettingsProps) {
  const [open, setOpen] = useState(defaultOpen)
  const [parserEngine, setParserEngine] = useState<ParserEngineChoice>(AUTO)
  const [ocrEngine, setOcrEngine] = useState<OcrEngineChoice>(AUTO)
  const [tableMode, setTableMode] = useState<TableModeChoice>(AUTO)

  // Recompute overrides from the current field state. A field left on `Auto`
  // contributes nothing, so an all-`Auto` panel yields `{}` (no overrides) and
  // the backend auto-routes. Changed fields persist across collapse so the user
  // never silently loses a selection.
  const emit = useCallback(
    (
      parser: ParserEngineChoice,
      ocr: OcrEngineChoice,
      table: TableModeChoice
    ) => {
      const overrides: Partial<SettingsResponse> = {}
      if (parser !== AUTO) overrides.parser_engine = parser
      if (ocr !== AUTO) overrides.docling_ocr_engine = ocr
      if (table !== AUTO) overrides.docling_table_mode = table
      onOverridesChange(overrides)
    },
    [onOverridesChange]
  )

  // Keep the parent in sync on mount so it starts from a known (empty) state.
  useEffect(() => {
    onOverridesChange({})
    // Only on mount — subsequent changes flow through the `emit` handlers.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <Collapsible
      open={open}
      onOpenChange={setOpen}
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
                onValueChange={(v) => {
                  const next = v as ParserEngineChoice
                  setParserEngine(next)
                  emit(next, ocrEngine, tableMode)
                }}
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
                onValueChange={(v) => {
                  const next = v as OcrEngineChoice
                  setOcrEngine(next)
                  emit(parserEngine, next, tableMode)
                }}
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
                onValueChange={(v) => {
                  const next = v as TableModeChoice
                  setTableMode(next)
                  emit(parserEngine, ocrEngine, next)
                }}
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
