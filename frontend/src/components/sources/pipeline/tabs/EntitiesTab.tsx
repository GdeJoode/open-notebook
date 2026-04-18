'use client'

import { useState, useEffect, useCallback, useMemo } from 'react'
import {
  Network, Loader2, CheckCircle2, XCircle, Clock,
  Play, Filter, ChevronDown, ShieldCheck, Link2,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Checkbox } from '@/components/ui/checkbox'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Slider } from '@/components/ui/slider'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { cn } from '@/lib/utils'
import { sourcesApi } from '@/lib/api/sources'
import { ontologiesApi, type OntologySummary } from '@/lib/api/ontologies'
import {
  validateEntities, findDuplicates, mergeEntities as mergeEntitiesApi,
  type ValidationReport, type ValidationResult,
  type DeduplicateResponse, type DuplicateCandidate,
} from '@/lib/api/services'
import type { ExtractionResultResponse, RunEntitiesOptions } from '@/lib/types/api'
import type { StepStatus } from '../PipelineStepper'
import type { FileEntry } from './ExtractionTab'
import { EntityGraphView } from './EntityGraphView'

interface EntitiesTabProps {
  status: StepStatus
  extractionComplete?: boolean
  classificationReady?: boolean
  sourceId?: string
  files?: FileEntry[]
  onStart?: () => void // kept for backward compat
}

export function EntitiesTab({
  status,
  extractionComplete,
  classificationReady,
  sourceId,
  files = [],
  onStart,
}: EntitiesTabProps) {
  // Left pane: selected file
  const [selectedFileIndex, setSelectedFileIndex] = useState(0)

  // Middle pane: extraction options
  const [extractorType, setExtractorType] = useState<'llm' | 'langextract'>('llm')
  const [ontologyName, setOntologyName] = useState('general')
  const [ontologies, setOntologies] = useState<OntologySummary[]>([])
  const [langextractOptions, setLangextractOptions] = useState({
    model_id: 'qwen2.5:latest',
    model_url: 'http://localhost:11434',
    temperature: '',
    use_schema_constraints: true,
    fence_output: true,
  })
  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState<string | null>(null)

  // Right pane: results
  const [extractionResult, setExtractionResult] = useState<ExtractionResultResponse | null>(null)
  const [activeResultTab, setActiveResultTab] = useState<'list' | 'visualization' | 'duplicates'>('list')

  // Validation
  const [validationReport, setValidationReport] = useState<ValidationReport | null>(null)
  const [validating, setValidating] = useState(false)
  const [statusFilter, setStatusFilter] = useState<string>('all')

  // Dedup review
  const [dedupResponse, setDedupResponse] = useState<DeduplicateResponse | null>(null)
  const [dedupLoading, setDedupLoading] = useState(false)
  const [dismissedPairs, setDismissedPairs] = useState<Set<string>>(new Set())

  // Filtering options
  const [filteringOptions, setFilteringOptions] = useState({
    dedup_enabled: true,
    fuzzy_dedup_enabled: true,
    embedding_dedup_enabled: true,
    edge_prediction_enabled: true,
    dedup_similarity_threshold: 0.85,
  })
  const [filteringRunning, setFilteringRunning] = useState(false)
  const [filteringStats, setFilteringStats] = useState<{
    entities_before: number
    entities_after: number
    entities_removed: number
    merge_groups: number
    predicted_edges: number
  } | null>(null)

  const selectedFile = files[selectedFileIndex] || null
  const selectedSourceId = selectedFile?.sourceId || sourceId

  // Fetch ontologies once
  useEffect(() => {
    let cancelled = false
    ontologiesApi.list().then(data => {
      if (!cancelled) setOntologies(data)
    }).catch(() => {})
    return () => { cancelled = true }
  }, [])

  // Fetch existing extraction results when selected source changes
  const refreshResult = useCallback(async () => {
    if (!selectedSourceId) {
      setExtractionResult(null)
      return
    }
    try {
      const data = await sourcesApi.getExtractionResult(selectedSourceId)
      if (data.entity_count > 0 || data.relation_count > 0) {
        setExtractionResult(data)
      } else {
        setExtractionResult(null)
      }
    } catch {
      setExtractionResult(null)
    }
  }, [selectedSourceId])

  useEffect(() => {
    refreshResult()
  }, [refreshResult])

  // Run extraction
  const handleRun = async () => {
    if (!selectedSourceId || running) return
    setRunning(true)
    setRunError(null)

    const options: RunEntitiesOptions = {
      ontology_name: ontologyName,
      extractor_type: extractorType,
    }

    if (extractorType === 'langextract') {
      if (langextractOptions.model_id) options.langextract_model_id = langextractOptions.model_id
      if (langextractOptions.model_url) options.langextract_model_url = langextractOptions.model_url
      if (langextractOptions.temperature) {
        const temp = parseFloat(langextractOptions.temperature)
        if (!isNaN(temp)) options.langextract_temperature = temp
      }
      options.langextract_use_schema_constraints = langextractOptions.use_schema_constraints
      options.langextract_fence_output = langextractOptions.fence_output
    }

    try {
      await sourcesApi.runEntities(selectedSourceId, options)
      // Also call parent's onStart for backward compat status tracking
      onStart?.()
      // Poll for result
      pollForResult(selectedSourceId)
    } catch (err) {
      setRunError(err instanceof Error ? err.message : 'Extraction failed')
      setRunning(false)
    }
  }

  // Poll extraction result until entities appear
  const pollForResult = useCallback((sid: string) => {
    let attempts = 0
    const maxAttempts = 120 // ~4 minutes at 2s interval
    const interval = setInterval(async () => {
      attempts++
      try {
        const data = await sourcesApi.getExtractionResult(sid)
        if (data.entity_count > 0 || data.relation_count > 0) {
          setExtractionResult(data)
          setRunning(false)
          clearInterval(interval)
        }
      } catch {
        // keep polling
      }
      if (attempts >= maxAttempts) {
        setRunning(false)
        clearInterval(interval)
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  // Run filtering on existing extraction result
  const handleRunFiltering = async () => {
    if (!selectedSourceId || filteringRunning) return
    setFilteringRunning(true)
    try {
      const stats = await sourcesApi.runFiltering(selectedSourceId, filteringOptions)
      setFilteringStats(stats)
      // Refresh extraction result to show filtered data
      await refreshResult()
    } catch (err) {
      setRunError(err instanceof Error ? err.message : 'Filtering failed')
    } finally {
      setFilteringRunning(false)
    }
  }

  // Run validation on extracted entities
  const handleValidate = async () => {
    if (validating) return
    setValidating(true)
    try {
      const report = await validateEntities({ apply_changes: true })
      setValidationReport(report)
      // Refresh extraction result to show updated entities
      await refreshResult()
    } catch (err) {
      setRunError(err instanceof Error ? err.message : 'Validation failed')
    } finally {
      setValidating(false)
    }
  }

  // Find duplicate candidates
  const handleFindDuplicates = async () => {
    if (dedupLoading) return
    setDedupLoading(true)
    try {
      const resp = await findDuplicates({ review_threshold: 0.6 })
      setDedupResponse(resp)
      setDismissedPairs(new Set())
    } catch (err) {
      setRunError(err instanceof Error ? err.message : 'Dedup scan failed')
    } finally {
      setDedupLoading(false)
    }
  }

  // Merge a duplicate pair
  const handleMerge = async (canonical_id: string, duplicate_id: string, pairKey: string) => {
    try {
      await mergeEntitiesApi(canonical_id, duplicate_id)
      setDismissedPairs(prev => new Set(prev).add(pairKey))
      await refreshResult()
    } catch (err) {
      setRunError(err instanceof Error ? err.message : 'Merge failed')
    }
  }

  // Dismiss (ignore) a duplicate pair
  const handleDismiss = (pairKey: string) => {
    setDismissedPairs(prev => new Set(prev).add(pairKey))
  }

  // If extraction not complete, show locked state
  if (!extractionComplete) {
    return (
      <Card className="flex-1 flex flex-col min-h-0">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 shrink-0">
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <Network className="h-4 w-4" />
            Entity Extraction
          </CardTitle>
          <Badge variant="outline">Locked</Badge>
        </CardHeader>
        <CardContent className="flex items-center justify-center flex-1">
          <p className="text-sm text-muted-foreground">
            Entities available after text extraction completes.
          </p>
        </CardContent>
      </Card>
    )
  }

  // If classification not ready (no ontology selected), show gate
  if (!classificationReady) {
    return (
      <Card className="flex-1 flex flex-col min-h-0">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 shrink-0">
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <Network className="h-4 w-4" />
            Entity Extraction
          </CardTitle>
          <Badge variant="outline" className="border-amber-500 text-amber-600">Needs Classification</Badge>
        </CardHeader>
        <CardContent className="flex items-center justify-center flex-1">
          <p className="text-sm text-muted-foreground text-center max-w-sm">
            At least one ontology must be selected and saved on the Analysis &amp; Classification step before entity extraction can proceed.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="flex-1 flex flex-col min-h-0">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 shrink-0">
        <CardTitle className="text-base font-medium flex items-center gap-2">
          <Network className="h-4 w-4" />
          Entity Extraction
        </CardTitle>
        <StatusBadge status={status} />
      </CardHeader>
      <CardContent className="flex-1 flex flex-col min-h-0">
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr_2fr] gap-3 flex-1 min-h-0">
          {/* Left pane: source list */}
          <SourceListPane
            files={files}
            selectedIndex={selectedFileIndex}
            onSelect={setSelectedFileIndex}
          />

          {/* Middle pane: extraction + filtering options */}
          <ExtractionOptionsPane
            extractorType={extractorType}
            onExtractorTypeChange={setExtractorType}
            ontologyName={ontologyName}
            onOntologyNameChange={setOntologyName}
            ontologies={ontologies}
            langextractOptions={langextractOptions}
            onLangextractOptionsChange={setLangextractOptions}
            running={running}
            runError={runError}
            canRun={!!selectedSourceId && !running}
            onRun={handleRun}
            hasExtractionResult={!!extractionResult}
            filteringOptions={filteringOptions}
            onFilteringOptionsChange={setFilteringOptions}
            filteringRunning={filteringRunning}
            filteringStats={filteringStats}
            onRunFiltering={handleRunFiltering}
          />

          {/* Right pane: tabbed results */}
          <ResultsPane
            extractionResult={extractionResult}
            running={running}
            activeTab={activeResultTab}
            onTabChange={setActiveResultTab}
            validationReport={validationReport}
            validating={validating}
            onValidate={handleValidate}
            statusFilter={statusFilter}
            onStatusFilterChange={setStatusFilter}
            dedupResponse={dedupResponse}
            dedupLoading={dedupLoading}
            dismissedPairs={dismissedPairs}
            onFindDuplicates={handleFindDuplicates}
            onMerge={handleMerge}
            onDismiss={handleDismiss}
          />
        </div>
      </CardContent>
    </Card>
  )
}

/* ------------------------------------------------------------------ */
/* Left pane — source list                                             */
/* ------------------------------------------------------------------ */

function SourceListPane({
  files,
  selectedIndex,
  onSelect,
}: {
  files: FileEntry[]
  selectedIndex: number
  onSelect: (index: number) => void
}) {
  return (
    <div className="border rounded-md overflow-hidden flex flex-col min-h-0">
      <div className="px-3 py-2 border-b bg-muted/50 shrink-0">
        <p className="text-xs font-medium text-muted-foreground">Sources</p>
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="p-1.5 space-y-0.5">
          {files.map((file, i) => (
            <button
              key={i}
              type="button"
              onClick={() => onSelect(i)}
              className={cn(
                "flex items-center gap-2 px-2 py-1.5 rounded text-xs w-full text-left transition-colors",
                i === selectedIndex
                  ? "bg-primary/10 text-primary font-medium"
                  : "hover:bg-muted/50"
              )}
            >
              <FileStatusIcon status={file.status} />
              <span className="truncate flex-1" title={file.name}>
                {file.name}
              </span>
            </button>
          ))}
          {files.length === 0 && (
            <p className="text-xs text-muted-foreground px-2 py-3">No sources available</p>
          )}
        </div>
      </div>
    </div>
  )
}

function FileStatusIcon({ status }: { status: FileEntry['status'] }) {
  switch (status) {
    case 'completed':
      return <CheckCircle2 className="h-3.5 w-3.5 text-green-500 shrink-0" />
    case 'processing':
    case 'creating':
      return <Loader2 className="h-3.5 w-3.5 animate-spin text-primary shrink-0" />
    case 'failed':
      return <XCircle className="h-3.5 w-3.5 text-destructive shrink-0" />
    default:
      return <Clock className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
  }
}

/* ------------------------------------------------------------------ */
/* Middle pane — extraction options                                     */
/* ------------------------------------------------------------------ */

interface LangextractOptions {
  model_id: string
  model_url: string
  temperature: string
  use_schema_constraints: boolean
  fence_output: boolean
}

interface FilteringOpts {
  dedup_enabled: boolean
  fuzzy_dedup_enabled: boolean
  embedding_dedup_enabled: boolean
  edge_prediction_enabled: boolean
  dedup_similarity_threshold: number
}

interface FilteringStats {
  entities_before: number
  entities_after: number
  entities_removed: number
  merge_groups: number
  predicted_edges: number
}

function ExtractionOptionsPane({
  extractorType,
  onExtractorTypeChange,
  ontologyName,
  onOntologyNameChange,
  ontologies,
  langextractOptions,
  onLangextractOptionsChange,
  running,
  runError,
  canRun,
  onRun,
  hasExtractionResult,
  filteringOptions,
  onFilteringOptionsChange,
  filteringRunning,
  filteringStats,
  onRunFiltering,
}: {
  extractorType: 'llm' | 'langextract'
  onExtractorTypeChange: (v: 'llm' | 'langextract') => void
  ontologyName: string
  onOntologyNameChange: (v: string) => void
  ontologies: OntologySummary[]
  langextractOptions: LangextractOptions
  onLangextractOptionsChange: (v: LangextractOptions) => void
  running: boolean
  runError: string | null
  canRun: boolean
  onRun: () => void
  hasExtractionResult: boolean
  filteringOptions: FilteringOpts
  onFilteringOptionsChange: (v: FilteringOpts) => void
  filteringRunning: boolean
  filteringStats: FilteringStats | null
  onRunFiltering: () => void
}) {
  return (
    <div className="border rounded-md overflow-hidden flex flex-col min-h-0">
      <div className="px-3 py-2 border-b bg-muted/50 shrink-0">
        <p className="text-xs font-medium text-muted-foreground">Extraction Options</p>
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-4">
        {/* Extractor Type */}
        <div className="space-y-2">
          <Label className="text-xs font-medium">Extractor</Label>
          <RadioGroup
            value={extractorType}
            onValueChange={(v) => onExtractorTypeChange(v as 'llm' | 'langextract')}
            className="flex gap-3"
            disabled={running}
          >
            <div className="flex items-center space-x-1.5">
              <RadioGroupItem value="llm" id="ext-llm" />
              <Label htmlFor="ext-llm" className="text-xs cursor-pointer">LLM</Label>
            </div>
            <div className="flex items-center space-x-1.5">
              <RadioGroupItem value="langextract" id="ext-langextract" />
              <Label htmlFor="ext-langextract" className="text-xs cursor-pointer">LangExtract</Label>
            </div>
          </RadioGroup>
        </div>

        {/* Ontology */}
        <div className="space-y-1.5">
          <Label className="text-xs font-medium">Ontology</Label>
          <Select value={ontologyName} onValueChange={onOntologyNameChange} disabled={running}>
            <SelectTrigger className="h-8 text-xs">
              <SelectValue placeholder="Select ontology" />
            </SelectTrigger>
            <SelectContent>
              {ontologies.map(o => (
                <SelectItem key={o.name} value={o.name} className="text-xs">
                  {o.name}
                  <span className="text-muted-foreground ml-1">
                    ({o.entity_type_count} types)
                  </span>
                </SelectItem>
              ))}
              {ontologies.length === 0 && (
                <SelectItem value="general" className="text-xs">general</SelectItem>
              )}
            </SelectContent>
          </Select>
        </div>

        {/* LangExtract options (shown when selected) */}
        {extractorType === 'langextract' && (
          <div className="space-y-3 pt-2 border-t">
            <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
              LangExtract Settings
            </p>

            <div className="space-y-1.5">
              <Label htmlFor="le-model-id" className="text-xs">Model ID</Label>
              <Input
                id="le-model-id"
                value={langextractOptions.model_id}
                onChange={(e) => onLangextractOptionsChange({
                  ...langextractOptions, model_id: e.target.value,
                })}
                disabled={running}
                className="h-8 text-xs"
                placeholder="qwen2.5:latest"
              />
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="le-model-url" className="text-xs">Model URL</Label>
              <Input
                id="le-model-url"
                value={langextractOptions.model_url}
                onChange={(e) => onLangextractOptionsChange({
                  ...langextractOptions, model_url: e.target.value,
                })}
                disabled={running}
                className="h-8 text-xs"
                placeholder="http://localhost:11434"
              />
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="le-temp" className="text-xs">Temperature</Label>
              <Input
                id="le-temp"
                type="number"
                step="0.1"
                min="0"
                max="2"
                value={langextractOptions.temperature}
                onChange={(e) => onLangextractOptionsChange({
                  ...langextractOptions, temperature: e.target.value,
                })}
                disabled={running}
                className="h-8 text-xs"
                placeholder="(default)"
              />
            </div>

            <div className="flex items-center space-x-2">
              <Checkbox
                id="le-schema"
                checked={langextractOptions.use_schema_constraints}
                onCheckedChange={(checked) => onLangextractOptionsChange({
                  ...langextractOptions, use_schema_constraints: !!checked,
                })}
                disabled={running}
              />
              <Label htmlFor="le-schema" className="text-xs cursor-pointer">
                Schema constraints
              </Label>
            </div>

            <div className="flex items-center space-x-2">
              <Checkbox
                id="le-fence"
                checked={langextractOptions.fence_output}
                onCheckedChange={(checked) => onLangextractOptionsChange({
                  ...langextractOptions, fence_output: !!checked,
                })}
                disabled={running}
              />
              <Label htmlFor="le-fence" className="text-xs cursor-pointer">
                Fence output
              </Label>
            </div>
          </div>
        )}
      </div>

      {/* Run button */}
      <div className="px-3 py-2 border-t shrink-0 space-y-2">
        {runError && (
          <p className="text-xs text-destructive">{runError}</p>
        )}
        <Button
          type="button"
          size="sm"
          className="w-full gap-2"
          disabled={!canRun}
          onClick={onRun}
        >
          {running ? (
            <>
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Extracting...
            </>
          ) : (
            <>
              <Play className="h-3.5 w-3.5" />
              Run Extraction
            </>
          )}
        </Button>

        {/* Filtering options (shown after extraction has results) */}
        {hasExtractionResult && (
          <Collapsible>
            <CollapsibleTrigger className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground w-full pt-2">
              <Filter className="h-3 w-3" />
              Filtering Options
              <ChevronDown className="h-3 w-3 ml-auto" />
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-2 space-y-2.5">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="f-dedup"
                  checked={filteringOptions.dedup_enabled}
                  onCheckedChange={(c) => onFilteringOptionsChange({
                    ...filteringOptions, dedup_enabled: !!c,
                  })}
                  disabled={filteringRunning}
                />
                <Label htmlFor="f-dedup" className="text-xs cursor-pointer">String dedup</Label>
              </div>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="f-fuzzy"
                  checked={filteringOptions.fuzzy_dedup_enabled}
                  onCheckedChange={(c) => onFilteringOptionsChange({
                    ...filteringOptions, fuzzy_dedup_enabled: !!c,
                  })}
                  disabled={filteringRunning}
                />
                <Label htmlFor="f-fuzzy" className="text-xs cursor-pointer">Fuzzy dedup</Label>
              </div>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="f-embedding"
                  checked={filteringOptions.embedding_dedup_enabled}
                  onCheckedChange={(c) => onFilteringOptionsChange({
                    ...filteringOptions, embedding_dedup_enabled: !!c,
                  })}
                  disabled={filteringRunning}
                />
                <Label htmlFor="f-embedding" className="text-xs cursor-pointer">Embedding dedup</Label>
              </div>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="f-edges"
                  checked={filteringOptions.edge_prediction_enabled}
                  onCheckedChange={(c) => onFilteringOptionsChange({
                    ...filteringOptions, edge_prediction_enabled: !!c,
                  })}
                  disabled={filteringRunning}
                />
                <Label htmlFor="f-edges" className="text-xs cursor-pointer">Edge prediction</Label>
              </div>

              <div className="space-y-1">
                <Label className="text-[10px] text-muted-foreground">
                  Similarity threshold: {filteringOptions.dedup_similarity_threshold.toFixed(2)}
                </Label>
                <Slider
                  value={[filteringOptions.dedup_similarity_threshold]}
                  onValueChange={([v]) => onFilteringOptionsChange({
                    ...filteringOptions, dedup_similarity_threshold: v,
                  })}
                  min={0.5}
                  max={1.0}
                  step={0.05}
                  disabled={filteringRunning}
                  className="w-full"
                />
              </div>

              <Button
                type="button"
                size="sm"
                variant="secondary"
                className="w-full gap-2"
                disabled={filteringRunning}
                onClick={onRunFiltering}
              >
                {filteringRunning ? (
                  <>
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    Filtering...
                  </>
                ) : (
                  <>
                    <Filter className="h-3.5 w-3.5" />
                    Run Filtering
                  </>
                )}
              </Button>

              {filteringStats && (
                <div className="text-[10px] text-muted-foreground space-y-0.5 pt-1 border-t">
                  <p>{filteringStats.entities_before} → {filteringStats.entities_after} entities</p>
                  <p>{filteringStats.entities_removed} removed, {filteringStats.merge_groups} merge groups</p>
                  <p>{filteringStats.predicted_edges} predicted edges</p>
                </div>
              )}
            </CollapsibleContent>
          </Collapsible>
        )}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Right pane — tabbed results                                         */
/* ------------------------------------------------------------------ */

function ResultsPane({
  extractionResult,
  running,
  activeTab,
  onTabChange,
  validationReport,
  validating,
  onValidate,
  statusFilter,
  onStatusFilterChange,
  dedupResponse,
  dedupLoading,
  dismissedPairs,
  onFindDuplicates,
  onMerge,
  onDismiss,
}: {
  extractionResult: ExtractionResultResponse | null
  running: boolean
  activeTab: 'list' | 'visualization' | 'duplicates'
  onTabChange: (tab: 'list' | 'visualization' | 'duplicates') => void
  validationReport: ValidationReport | null
  validating: boolean
  onValidate: () => void
  statusFilter: string
  onStatusFilterChange: (v: string) => void
  dedupResponse: DeduplicateResponse | null
  dedupLoading: boolean
  dismissedPairs: Set<string>
  onFindDuplicates: () => void
  onMerge: (canonicalId: string, duplicateId: string, pairKey: string) => void
  onDismiss: (pairKey: string) => void
}) {
  if (!extractionResult && !running) {
    return (
      <div className="border rounded-md overflow-hidden flex flex-col min-h-0">
        <div className="px-3 py-2 border-b bg-muted/50 shrink-0">
          <p className="text-xs font-medium text-muted-foreground">Results</p>
        </div>
        <div className="flex-1 flex items-center justify-center p-6">
          <p className="text-sm text-muted-foreground text-center">
            Configure extraction options and click Run to extract entities.
          </p>
        </div>
      </div>
    )
  }

  if (running && !extractionResult) {
    return (
      <div className="border rounded-md overflow-hidden flex flex-col min-h-0">
        <div className="px-3 py-2 border-b bg-muted/50 shrink-0">
          <p className="text-xs font-medium text-muted-foreground">Results</p>
        </div>
        <div className="flex-1 flex items-center justify-center p-6">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Extracting entities...
          </div>
        </div>
      </div>
    )
  }

  const entities = extractionResult?.entities ?? []
  const relations = extractionResult?.relations ?? []

  return (
    <Tabs
      value={activeTab}
      onValueChange={(v) => onTabChange(v as 'list' | 'visualization')}
      className="border rounded-md overflow-hidden flex flex-col min-h-0 gap-0"
    >
      <div className="px-3 py-2 border-b bg-muted/50 shrink-0 space-y-2">
        <div className="flex items-center justify-between">
          <TabsList className="h-7">
            <TabsTrigger value="list" className="text-xs h-6 px-2.5">
              Entities & Relations
            </TabsTrigger>
            <TabsTrigger value="visualization" className="text-xs h-6 px-2.5">
              Visualization
            </TabsTrigger>
            <TabsTrigger value="duplicates" className="text-xs h-6 px-2.5">
              Duplicates
              {dedupResponse && dedupResponse.candidates.length - dismissedPairs.size > 0 && (
                <Badge variant="secondary" className="text-[9px] ml-1 px-1">
                  {dedupResponse.candidates.length - dismissedPairs.size}
                </Badge>
              )}
            </TabsTrigger>
          </TabsList>
          <div className="flex items-center gap-1.5">
            <Badge variant="secondary" className="text-[10px]">
              {extractionResult?.entity_count ?? 0} entities
            </Badge>
            <Badge variant="secondary" className="text-[10px]">
              {extractionResult?.relation_count ?? 0} relations
            </Badge>
          </div>
        </div>
        {activeTab === 'list' && (
          <div className="flex items-center gap-2">
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="h-6 text-[10px] gap-1"
              disabled={validating}
              onClick={onValidate}
            >
              {validating ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <ShieldCheck className="h-3 w-3" />
              )}
              Validate
            </Button>
            <Select value={statusFilter} onValueChange={onStatusFilterChange}>
              <SelectTrigger className="h-6 text-[10px] w-[130px]">
                <SelectValue placeholder="Filter status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all" className="text-xs">All statuses</SelectItem>
                <SelectItem value="validated" className="text-xs">Validated</SelectItem>
                <SelectItem value="reclassified" className="text-xs">Reclassified</SelectItem>
                <SelectItem value="enriched" className="text-xs">Enriched</SelectItem>
                <SelectItem value="flagged_generic" className="text-xs">Generic</SelectItem>
                <SelectItem value="unvalidated" className="text-xs">Unvalidated</SelectItem>
              </SelectContent>
            </Select>
            {validationReport && (
              <div className="flex items-center gap-1 ml-auto">
                {Object.entries(validationReport.results_by_status).map(([status, count]) => (
                  <Badge key={status} variant="outline" className="text-[9px]">
                    {status}: {count}
                  </Badge>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      <TabsContent value="list" className="flex-1 min-h-0 overflow-y-auto mt-0">
        <EntityRelationList
          entities={entities}
          relations={relations}
          validationReport={validationReport}
          statusFilter={statusFilter}
        />
      </TabsContent>

      <TabsContent value="visualization" className="flex-1 min-h-0 mt-0">
        <EntityGraphView entities={entities} relations={relations} />
      </TabsContent>

      <TabsContent value="duplicates" className="flex-1 min-h-0 overflow-y-auto mt-0">
        <DuplicateReviewPane
          dedupResponse={dedupResponse}
          dedupLoading={dedupLoading}
          dismissedPairs={dismissedPairs}
          onFindDuplicates={onFindDuplicates}
          onMerge={onMerge}
          onDismiss={onDismiss}
        />
      </TabsContent>
    </Tabs>
  )
}

/* ------------------------------------------------------------------ */
/* Entity & Relation list tables                                       */
/* ------------------------------------------------------------------ */

function ValidationStatusBadge({ status, uri }: { status?: string; uri?: string }) {
  switch (status) {
    case 'validated':
      return (
        <span className="inline-flex items-center gap-0.5">
          <Badge variant="outline" className="text-[9px] border-green-500 text-green-600 gap-0.5">
            <CheckCircle2 className="h-2.5 w-2.5" />
            validated
          </Badge>
          {uri && (
            <a href={uri} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:text-blue-600">
              <Link2 className="h-3 w-3" />
            </a>
          )}
        </span>
      )
    case 'reclassified':
      return (
        <Badge variant="outline" className="text-[9px] border-amber-500 text-amber-600">
          reclassified
        </Badge>
      )
    case 'enriched':
      return (
        <span className="inline-flex items-center gap-0.5">
          <Badge variant="outline" className="text-[9px] border-blue-500 text-blue-600 gap-0.5">
            <Link2 className="h-2.5 w-2.5" />
            enriched
          </Badge>
          {uri && (
            <a href={uri} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:text-blue-600">
              <Link2 className="h-3 w-3" />
            </a>
          )}
        </span>
      )
    case 'flagged_generic':
      return (
        <Badge variant="outline" className="text-[9px] border-gray-400 text-gray-500">
          generic
        </Badge>
      )
    default:
      return (
        <Badge variant="outline" className="text-[9px]">
          ?
        </Badge>
      )
  }
}

function EntityRelationList({
  entities,
  relations,
  validationReport,
  statusFilter,
}: {
  entities: ExtractionResultResponse['entities']
  relations: ExtractionResultResponse['relations']
  validationReport: ValidationReport | null
  statusFilter: string
}) {
  // Build a lookup from entity name to validation result
  const validationLookup = useMemo(() => {
    if (!validationReport) return new Map<string, ValidationResult>()
    const lookup = new Map<string, ValidationResult>()
    for (const r of [
      ...validationReport.validated,
      ...validationReport.reclassified,
      ...validationReport.enriched,
      ...validationReport.flagged_generic,
      ...validationReport.unvalidated,
    ]) {
      lookup.set(r.entity_name, r)
    }
    return lookup
  }, [validationReport])

  // Filter entities by validation status
  const filteredEntities = useMemo(() => {
    if (statusFilter === 'all' || !validationReport) return entities
    return entities.filter(e => {
      const vr = validationLookup.get(e.text)
      if (!vr) return statusFilter === 'unvalidated'
      return vr.status === statusFilter
    })
  }, [entities, statusFilter, validationReport, validationLookup])

  const hasGraphScores = useMemo(
    () => entities.some(e => e.pagerank != null && e.pagerank > 0),
    [entities]
  )

  return (
    <div className="p-3 space-y-4">
      {/* Entities table */}
      <div>
        <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">
          Entities ({filteredEntities.length}{statusFilter !== 'all' ? ` of ${entities.length}` : ''})
        </p>
        {filteredEntities.length > 0 ? (
          <div className="border rounded overflow-hidden">
            <table className="w-full text-xs">
              <thead className="bg-muted/50">
                <tr>
                  <th className="text-left px-2 py-1.5 font-medium">Text</th>
                  <th className="text-left px-2 py-1.5 font-medium">Label</th>
                  {validationReport && (
                    <th className="text-center px-2 py-1.5 font-medium">Status</th>
                  )}
                  {hasGraphScores && (
                    <>
                      <th className="text-right px-2 py-1.5 font-medium">PR</th>
                      <th className="text-center px-2 py-1.5 font-medium">Com.</th>
                    </>
                  )}
                  <th className="text-right px-2 py-1.5 font-medium">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {filteredEntities.slice(0, 100).map((entity, i) => {
                  const vr = validationLookup.get(entity.text)
                  return (
                    <tr key={i} className="border-t">
                      <td className="px-2 py-1.5 truncate max-w-[180px]" title={entity.text}>
                        {entity.text}
                      </td>
                      <td className="px-2 py-1.5">
                        <Badge variant="outline" className="text-[10px]">{entity.label}</Badge>
                      </td>
                      {validationReport && (
                        <td className="px-2 py-1.5 text-center">
                          <ValidationStatusBadge
                            status={vr?.status}
                            uri={vr?.external_uri}
                          />
                        </td>
                      )}
                      {hasGraphScores && (
                        <>
                          <td className="px-2 py-1.5 text-right text-muted-foreground tabular-nums">
                            {entity.pagerank != null ? entity.pagerank.toFixed(4) : '-'}
                          </td>
                          <td className="px-2 py-1.5 text-center text-muted-foreground">
                            {entity.community_id != null ? entity.community_id : '-'}
                          </td>
                        </>
                      )}
                      <td className="px-2 py-1.5 text-right text-muted-foreground">
                        {(entity.confidence * 100).toFixed(0)}%
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            {filteredEntities.length > 100 && (
              <p className="text-[10px] text-muted-foreground text-center py-1.5 border-t">
                Showing 100 of {filteredEntities.length} entities
              </p>
            )}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground italic">No entities found.</p>
        )}
      </div>

      {/* Relations table */}
      <div>
        <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">
          Relations ({relations.length})
        </p>
        {relations.length > 0 ? (
          <div className="border rounded overflow-hidden">
            <table className="w-full text-xs">
              <thead className="bg-muted/50">
                <tr>
                  <th className="text-left px-2 py-1.5 font-medium">Source</th>
                  <th className="text-center px-2 py-1.5 font-medium">Type</th>
                  <th className="text-left px-2 py-1.5 font-medium">Target</th>
                  <th className="text-right px-2 py-1.5 font-medium">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {relations.slice(0, 100).map((rel, i) => (
                  <tr key={i} className="border-t">
                    <td className="px-2 py-1.5 truncate max-w-[140px]" title={rel.source_entity}>
                      {rel.source_entity}
                    </td>
                    <td className="px-2 py-1.5 text-center">
                      <Badge variant="outline" className="text-[10px]">{rel.relation_type}</Badge>
                    </td>
                    <td className="px-2 py-1.5 truncate max-w-[140px]" title={rel.target_entity}>
                      {rel.target_entity}
                    </td>
                    <td className="px-2 py-1.5 text-right text-muted-foreground">
                      {(rel.confidence * 100).toFixed(0)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {relations.length > 100 && (
              <p className="text-[10px] text-muted-foreground text-center py-1.5 border-t">
                Showing 100 of {relations.length} relations
              </p>
            )}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground italic">No relations found.</p>
        )}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Duplicate review pane                                               */
/* ------------------------------------------------------------------ */

function DuplicateReviewPane({
  dedupResponse,
  dedupLoading,
  dismissedPairs,
  onFindDuplicates,
  onMerge,
  onDismiss,
}: {
  dedupResponse: DeduplicateResponse | null
  dedupLoading: boolean
  dismissedPairs: Set<string>
  onFindDuplicates: () => void
  onMerge: (canonicalId: string, duplicateId: string, pairKey: string) => void
  onDismiss: (pairKey: string) => void
}) {
  if (!dedupResponse && !dedupLoading) {
    return (
      <div className="flex flex-col items-center justify-center p-6 gap-3">
        <p className="text-sm text-muted-foreground text-center">
          Scan for duplicate entities across the knowledge base.
        </p>
        <Button size="sm" className="gap-1.5" onClick={onFindDuplicates}>
          <Filter className="h-3.5 w-3.5" />
          Find Duplicates
        </Button>
      </div>
    )
  }

  if (dedupLoading) {
    return (
      <div className="flex items-center justify-center p-6">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" />
          Scanning for duplicates...
        </div>
      </div>
    )
  }

  const candidates = dedupResponse!.candidates.filter(
    c => !dismissedPairs.has(`${c.entity_a_id}:${c.entity_b_id}`)
  )

  return (
    <div className="p-3 space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
          {candidates.length} candidates ({dedupResponse!.auto_merge_count} auto-merged)
        </p>
        <Button size="sm" variant="ghost" className="h-6 text-[10px] gap-1" onClick={onFindDuplicates}>
          <Filter className="h-3 w-3" /> Rescan
        </Button>
      </div>

      {candidates.length === 0 ? (
        <p className="text-xs text-muted-foreground italic text-center py-4">
          No duplicate candidates to review.
        </p>
      ) : (
        <div className="space-y-2">
          {candidates.slice(0, 50).map(c => {
            const pairKey = `${c.entity_a_id}:${c.entity_b_id}`
            return (
              <div key={pairKey} className="border rounded-md p-2.5 space-y-2">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0 space-y-1">
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs font-medium truncate">{c.entity_a_name}</span>
                      <Badge variant="outline" className="text-[9px] shrink-0">{c.entity_a_type}</Badge>
                    </div>
                    <div className="text-[10px] text-muted-foreground flex items-center gap-1">
                      <span>↔</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs font-medium truncate">{c.entity_b_name}</span>
                      <Badge variant="outline" className="text-[9px] shrink-0">{c.entity_b_type}</Badge>
                    </div>
                  </div>
                  <div className="text-right shrink-0 space-y-0.5">
                    <div className="text-[10px] text-muted-foreground">
                      Score: {(c.combined_score * 100).toFixed(0)}%
                    </div>
                    {c.auto_merge && (
                      <Badge variant="secondary" className="text-[9px]">auto</Badge>
                    )}
                  </div>
                </div>
                <div className="flex gap-1.5">
                  <Button
                    size="sm"
                    variant="default"
                    className="h-6 text-[10px] flex-1 gap-1"
                    onClick={() => onMerge(c.entity_a_id, c.entity_b_id, pairKey)}
                  >
                    Merge
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="h-6 text-[10px] flex-1"
                    onClick={() => onDismiss(pairKey)}
                  >
                    Ignore
                  </Button>
                </div>
              </div>
            )
          })}
          {candidates.length > 50 && (
            <p className="text-[10px] text-muted-foreground text-center">
              Showing 50 of {candidates.length} candidates
            </p>
          )}
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Status badge                                                        */
/* ------------------------------------------------------------------ */

function StatusBadge({ status }: { status: StepStatus }) {
  switch (status) {
    case 'running':
      return <Badge variant="default" className="bg-primary">Running</Badge>
    case 'completed':
      return <Badge variant="default" className="bg-green-600">Completed</Badge>
    case 'failed':
      return <Badge variant="destructive">Failed</Badge>
    case 'skipped':
      return <Badge variant="secondary">Skipped</Badge>
    default:
      return <Badge variant="outline">Pending</Badge>
  }
}
