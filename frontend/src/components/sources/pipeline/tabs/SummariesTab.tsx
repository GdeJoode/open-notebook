'use client'

import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import {
  AlignLeft, Loader2, CheckCircle2, XCircle, Clock,
  Play, Save, X, Plus, Sparkles, FileText,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import ReactMarkdown from 'react-markdown'
import { Textarea } from '@/components/ui/textarea'
import { toast } from 'sonner'
import { cn } from '@/lib/utils'
import { preprocessingApi, type PreprocessingResult, type DocumentClassification } from '@/lib/api/preprocessing'
import { getPresets, type PresetInfo } from '@/lib/api/services'
import type { StepStatus } from '../PipelineStepper'
import type { FileEntry } from './ExtractionTab'

const DOCUMENT_TYPES = [
  'academic_paper', 'policy_document', 'legal_document', 'technical_report',
  'transcript', 'news_article', 'book_chapter', 'correspondence',
  'manual', 'financial_report', 'presentation', 'other',
]

const DOMAINS = [
  'science', 'technology', 'law', 'politics', 'healthcare',
  'finance', 'education', 'business', 'engineering', 'other',
]

const FORMALITY_LEVELS = ['formal', 'semi_formal', 'informal']

interface SummariesTabProps {
  status: StepStatus
  extractionComplete?: boolean
  sourceId?: string
  files?: FileEntry[]
  onClassificationReady?: (ready: boolean) => void
}

export function SummariesTab({
  status,
  extractionComplete,
  sourceId,
  files = [],
  onClassificationReady,
}: SummariesTabProps) {
  const [selectedFileIndex, setSelectedFileIndex] = useState(0)

  // Per-source preprocessing results
  const [results, setResults] = useState<Map<string, PreprocessingResult>>(new Map())
  const [loadingIds, setLoadingIds] = useState<Set<string>>(new Set())
  const [errorIds, setErrorIds] = useState<Map<string, string>>(new Map())

  // Processing queue
  const [processingQueue, setProcessingQueue] = useState<string[]>([])
  const [currentlyProcessing, setCurrentlyProcessing] = useState<string | null>(null)
  const initiatedRef = useRef<Set<string>>(new Set())

  // Classification editing
  const [editedClassification, setEditedClassification] = useState<DocumentClassification | null>(null)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [newTopic, setNewTopic] = useState('')
  const [newOntology, setNewOntology] = useState('')

  // Summarization preset
  const [presets, setPresets] = useState<PresetInfo[]>([])
  const [selectedPreset, setSelectedPreset] = useState('auto')
  const [customSystemPrompt, setCustomSystemPrompt] = useState('')
  const [customOutputInstructions, setCustomOutputInstructions] = useState('')

  const selectedFile = files[selectedFileIndex] || null
  const selectedSourceId = selectedFile?.sourceId || sourceId

  // All source IDs to process
  const allSourceIds = useMemo(() => {
    if (files.length > 0) return files.map(f => f.sourceId).filter(Boolean) as string[]
    if (sourceId) return [sourceId]
    return []
  }, [files, sourceId])

  // --- Fetch available presets ---
  useEffect(() => {
    let cancelled = false
    getPresets().then(data => {
      if (!cancelled) setPresets(data.presets ?? [])
    }).catch(() => {})
    return () => { cancelled = true }
  }, [])

  // --- Auto-run: fetch existing results, queue missing ones ---
  useEffect(() => {
    if (!extractionComplete) return

    for (const sid of allSourceIds) {
      if (!sid || initiatedRef.current.has(sid)) continue
      initiatedRef.current.add(sid)

      setLoadingIds(prev => new Set(prev).add(sid))
      preprocessingApi.get(sid).then(existing => {
        if (existing) {
          setResults(prev => new Map(prev).set(sid, existing))
        } else {
          setProcessingQueue(prev => [...prev, sid])
        }
        setLoadingIds(prev => { const n = new Set(prev); n.delete(sid); return n })
      }).catch(() => {
        setProcessingQueue(prev => [...prev, sid])
        setLoadingIds(prev => { const n = new Set(prev); n.delete(sid); return n })
      })
    }
  }, [allSourceIds, extractionComplete])

  // --- Process queue one at a time ---
  useEffect(() => {
    if (currentlyProcessing || processingQueue.length === 0) return

    const nextId = processingQueue[0]
    setCurrentlyProcessing(nextId)
    setProcessingQueue(prev => prev.slice(1))

    preprocessingApi.run(nextId).then(result => {
      setResults(prev => new Map(prev).set(nextId, result))
      setCurrentlyProcessing(null)
    }).catch(err => {
      setErrorIds(prev => new Map(prev).set(nextId, err instanceof Error ? err.message : 'Analysis failed'))
      setCurrentlyProcessing(null)
    })
  }, [currentlyProcessing, processingQueue])

  // --- Sync edited classification when selected source or results change ---
  useEffect(() => {
    if (!selectedSourceId) { setEditedClassification(null); setDirty(false); return }
    const result = results.get(selectedSourceId)
    if (result) {
      setEditedClassification({ ...result.classification })
      setDirty(false)
    } else {
      setEditedClassification(null)
    }
  }, [selectedSourceId, results])

  // --- Report classification readiness to parent ---
  useEffect(() => {
    if (!onClassificationReady) return
    // Ready when ALL sources have results with at least one ontology
    const allReady = allSourceIds.length > 0 && allSourceIds.every(sid => {
      const r = results.get(sid)
      return r && r.classification.suggested_ontologies.length > 0
    })
    onClassificationReady(allReady)
  }, [results, allSourceIds, onClassificationReady])

  // --- Classification field updaters ---
  const updateField = <K extends keyof DocumentClassification>(
    field: K, value: DocumentClassification[K]
  ) => {
    if (!editedClassification) return
    setEditedClassification({ ...editedClassification, [field]: value })
    setDirty(true)
  }

  const removeTopic = (topic: string) => {
    if (!editedClassification) return
    updateField('key_topics', editedClassification.key_topics.filter(t => t !== topic))
  }
  const addTopic = () => {
    if (!editedClassification || !newTopic.trim()) return
    if (!editedClassification.key_topics.includes(newTopic.trim())) {
      updateField('key_topics', [...editedClassification.key_topics, newTopic.trim()])
    }
    setNewTopic('')
  }

  const removeOntology = (ont: string) => {
    if (!editedClassification) return
    updateField('suggested_ontologies', editedClassification.suggested_ontologies.filter(o => o !== ont))
  }
  const addOntology = () => {
    if (!editedClassification || !newOntology.trim()) return
    if (!editedClassification.suggested_ontologies.includes(newOntology.trim())) {
      updateField('suggested_ontologies', [...editedClassification.suggested_ontologies, newOntology.trim()])
    }
    setNewOntology('')
  }

  // --- Save classification ---
  const handleSave = useCallback(async () => {
    if (!selectedSourceId || !editedClassification) return
    setSaving(true)
    try {
      const updated = await preprocessingApi.update(selectedSourceId, {
        classification: editedClassification,
      })
      setResults(prev => new Map(prev).set(selectedSourceId, updated))
      setDirty(false)
      toast.success('Classification saved')
    } catch {
      toast.error('Failed to save classification')
    } finally {
      setSaving(false)
    }
  }, [selectedSourceId, editedClassification])

  // --- Re-run analysis for selected source ---
  const handleRerun = useCallback(async () => {
    if (!selectedSourceId) return
    setLoadingIds(prev => new Set(prev).add(selectedSourceId))
    try {
      const result = await preprocessingApi.run(selectedSourceId)
      setResults(prev => new Map(prev).set(selectedSourceId, result))
      toast.success('Analysis complete')
    } catch {
      toast.error('Analysis failed')
    } finally {
      setLoadingIds(prev => { const n = new Set(prev); n.delete(selectedSourceId); return n })
    }
  }, [selectedSourceId])

  // --- Derived state ---
  const selectedResult = selectedSourceId ? results.get(selectedSourceId) : null
  const isSelectedLoading = selectedSourceId ? loadingIds.has(selectedSourceId) : false
  const isSelectedProcessing = currentlyProcessing === selectedSourceId
  const selectedError = selectedSourceId ? errorIds.get(selectedSourceId) : undefined
  const isSelectedBusy = isSelectedLoading || isSelectedProcessing

  // Analysis status per file
  const getFileAnalysisStatus = (sid: string | undefined): 'done' | 'running' | 'queued' | 'error' | 'pending' => {
    if (!sid) return 'pending'
    if (results.has(sid)) return 'done'
    if (currentlyProcessing === sid || loadingIds.has(sid)) return 'running'
    if (processingQueue.includes(sid)) return 'queued'
    if (errorIds.has(sid)) return 'error'
    return 'pending'
  }

  // Locked state
  if (!extractionComplete) {
    return (
      <Card className="flex-1 flex flex-col min-h-0">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 shrink-0">
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <AlignLeft className="h-4 w-4" />
            Classification
          </CardTitle>
          <Badge variant="outline">Locked</Badge>
        </CardHeader>
        <CardContent className="flex items-center justify-center flex-1">
          <p className="text-sm text-muted-foreground">
            Available after extraction completes.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="flex-1 flex flex-col min-h-0">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 shrink-0">
        <CardTitle className="text-base font-medium flex items-center gap-2">
          <AlignLeft className="h-4 w-4" />
          Classification
        </CardTitle>
        <div className="flex items-center gap-2">
          {currentlyProcessing && (
            <Badge variant="secondary" className="text-xs gap-1">
              <Loader2 className="h-3 w-3 animate-spin" />
              Analyzing...
            </Badge>
          )}
          {processingQueue.length > 0 && (
            <Badge variant="outline" className="text-xs">
              {processingQueue.length} queued
            </Badge>
          )}
          <StatusBadge status={status} />
        </div>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col min-h-0">
        <div className="grid grid-cols-1 lg:grid-cols-[180px_1fr_300px] gap-3 flex-1 min-h-0">
          {/* Left pane: file list */}
          <FileListPane
            files={files}
            selectedIndex={selectedFileIndex}
            onSelect={setSelectedFileIndex}
            getAnalysisStatus={getFileAnalysisStatus}
          />

          {/* Middle pane: summary */}
          <SummaryPane
            result={selectedResult}
            loading={isSelectedBusy}
            error={selectedError}
            onRerun={handleRerun}
            presets={presets}
            selectedPreset={selectedPreset}
            onPresetChange={setSelectedPreset}
            customSystemPrompt={customSystemPrompt}
            onCustomSystemPromptChange={setCustomSystemPrompt}
            customOutputInstructions={customOutputInstructions}
            onCustomOutputInstructionsChange={setCustomOutputInstructions}
          />

          {/* Right pane: classification editor */}
          <ClassificationPane
            classification={editedClassification}
            loading={isSelectedBusy}
            dirty={dirty}
            saving={saving}
            newTopic={newTopic}
            newOntology={newOntology}
            onNewTopicChange={setNewTopic}
            onNewOntologyChange={setNewOntology}
            onUpdateField={updateField}
            onRemoveTopic={removeTopic}
            onAddTopic={addTopic}
            onRemoveOntology={removeOntology}
            onAddOntology={addOntology}
            onSave={handleSave}
          />
        </div>
      </CardContent>
    </Card>
  )
}

/* ------------------------------------------------------------------ */
/* Left pane — file list with analysis status                          */
/* ------------------------------------------------------------------ */

function FileListPane({
  files, selectedIndex, onSelect, getAnalysisStatus,
}: {
  files: FileEntry[]
  selectedIndex: number
  onSelect: (index: number) => void
  getAnalysisStatus: (sid: string | undefined) => string
}) {
  return (
    <div className="border rounded-md overflow-hidden flex flex-col min-h-0">
      <div className="px-3 py-2 border-b bg-muted/50 shrink-0">
        <p className="text-xs font-medium text-muted-foreground">Documents</p>
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="p-1.5 space-y-0.5">
          {files.map((file, i) => {
            const analysisStatus = getAnalysisStatus(file.sourceId)
            return (
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
                <AnalysisStatusIcon status={analysisStatus} />
                <span className="truncate flex-1" title={file.name}>
                  {file.name}
                </span>
              </button>
            )
          })}
          {files.length === 0 && (
            <p className="text-xs text-muted-foreground px-2 py-3">No documents</p>
          )}
        </div>
      </div>
    </div>
  )
}

function AnalysisStatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'done':
      return <CheckCircle2 className="h-3.5 w-3.5 text-green-500 shrink-0" />
    case 'running':
      return <Loader2 className="h-3.5 w-3.5 animate-spin text-primary shrink-0" />
    case 'queued':
      return <Clock className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
    case 'error':
      return <XCircle className="h-3.5 w-3.5 text-destructive shrink-0" />
    default:
      return <Clock className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
  }
}

/* ------------------------------------------------------------------ */
/* Middle pane — summary display                                       */
/* ------------------------------------------------------------------ */

function SummaryPane({
  result, loading, error, onRerun,
  presets, selectedPreset, onPresetChange,
  customSystemPrompt, onCustomSystemPromptChange,
  customOutputInstructions, onCustomOutputInstructionsChange,
}: {
  result: PreprocessingResult | null | undefined
  loading: boolean
  error?: string
  onRerun: () => void
  presets: PresetInfo[]
  selectedPreset: string
  onPresetChange: (v: string) => void
  customSystemPrompt: string
  onCustomSystemPromptChange: (v: string) => void
  customOutputInstructions: string
  onCustomOutputInstructionsChange: (v: string) => void
}) {
  if (loading) {
    return (
      <div className="border rounded-md overflow-hidden flex flex-col min-h-0">
        <div className="px-3 py-2 border-b bg-muted/50 shrink-0">
          <p className="text-xs font-medium text-muted-foreground">Summary</p>
        </div>
        <div className="flex-1 flex items-center justify-center p-6">
          <div className="flex items-center gap-3">
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
            <span className="text-sm text-muted-foreground">Analyzing document...</span>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="border rounded-md overflow-hidden flex flex-col min-h-0">
        <div className="px-3 py-2 border-b bg-muted/50 shrink-0">
          <p className="text-xs font-medium text-muted-foreground">Summary</p>
        </div>
        <div className="flex-1 flex flex-col items-center justify-center p-6 gap-3">
          <div className="flex items-center gap-2 text-destructive">
            <XCircle className="h-4 w-4" />
            <span className="text-sm">{error}</span>
          </div>
          <Button variant="outline" size="sm" className="gap-1.5" onClick={onRerun}>
            <Play className="h-3.5 w-3.5" /> Retry
          </Button>
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="border rounded-md overflow-hidden flex flex-col min-h-0">
        <div className="px-3 py-2 border-b bg-muted/50 shrink-0">
          <p className="text-xs font-medium text-muted-foreground">Summary</p>
        </div>
        <div className="flex-1 flex items-center justify-center p-6">
          <p className="text-sm text-muted-foreground text-center">
            Select a document to view its summary.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="border rounded-md overflow-hidden flex flex-col min-h-0">
      <div className="px-3 py-2 border-b bg-muted/50 shrink-0 space-y-2">
        <div className="flex items-center justify-between">
          <p className="text-xs font-medium text-muted-foreground">Summary</p>
          <Button variant="ghost" size="sm" className="h-6 gap-1 text-xs" onClick={onRerun}>
            <Play className="h-3 w-3" /> Re-run
          </Button>
        </div>
        {/* Preset selector */}
        <div className="flex items-center gap-2">
          <FileText className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
          <Select value={selectedPreset} onValueChange={onPresetChange}>
            <SelectTrigger className="h-7 text-[10px] flex-1">
              <SelectValue placeholder="Select preset" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto" className="text-xs">
                <span className="flex items-center gap-1">
                  <Sparkles className="h-3 w-3" /> Auto-detect
                </span>
              </SelectItem>
              {presets.map(p => (
                <SelectItem key={p.name} value={p.name} className="text-xs">
                  {formatLabel(p.name)}
                </SelectItem>
              ))}
              <SelectItem value="custom" className="text-xs">Custom</SelectItem>
            </SelectContent>
          </Select>
        </div>
        {selectedPreset === 'custom' && (
          <div className="space-y-2 pt-1">
            <div className="space-y-1">
              <Label className="text-[10px] text-muted-foreground">System Prompt</Label>
              <Textarea
                value={customSystemPrompt}
                onChange={e => onCustomSystemPromptChange(e.target.value)}
                className="text-xs min-h-[60px] resize-y"
                placeholder="Custom system prompt..."
              />
            </div>
            <div className="space-y-1">
              <Label className="text-[10px] text-muted-foreground">Output Instructions</Label>
              <Textarea
                value={customOutputInstructions}
                onChange={e => onCustomOutputInstructionsChange(e.target.value)}
                className="text-xs min-h-[60px] resize-y"
                placeholder="Custom output format instructions..."
              />
            </div>
          </div>
        )}
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto p-4">
        {result.naive_summary ? (
          <div className="prose prose-sm dark:prose-invert max-w-none prose-p:text-muted-foreground prose-headings:text-foreground prose-p:leading-relaxed prose-p:text-sm prose-li:text-sm prose-li:text-muted-foreground">
            <ReactMarkdown>{result.naive_summary}</ReactMarkdown>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground italic">No summary available.</p>
        )}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Right pane — editable classification                                */
/* ------------------------------------------------------------------ */

function ClassificationPane({
  classification, loading, dirty, saving,
  newTopic, newOntology,
  onNewTopicChange, onNewOntologyChange,
  onUpdateField, onRemoveTopic, onAddTopic,
  onRemoveOntology, onAddOntology, onSave,
}: {
  classification: DocumentClassification | null
  loading: boolean
  dirty: boolean
  saving: boolean
  newTopic: string
  newOntology: string
  onNewTopicChange: (v: string) => void
  onNewOntologyChange: (v: string) => void
  onUpdateField: <K extends keyof DocumentClassification>(field: K, value: DocumentClassification[K]) => void
  onRemoveTopic: (topic: string) => void
  onAddTopic: () => void
  onRemoveOntology: (ont: string) => void
  onAddOntology: () => void
  onSave: () => void
}) {
  if (loading) {
    return (
      <div className="border rounded-md overflow-hidden flex flex-col min-h-0">
        <div className="px-3 py-2 border-b bg-muted/50 shrink-0">
          <p className="text-xs font-medium text-muted-foreground">Classification</p>
        </div>
        <div className="flex-1 flex items-center justify-center p-4">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        </div>
      </div>
    )
  }

  if (!classification) {
    return (
      <div className="border rounded-md overflow-hidden flex flex-col min-h-0">
        <div className="px-3 py-2 border-b bg-muted/50 shrink-0">
          <p className="text-xs font-medium text-muted-foreground">Classification</p>
        </div>
        <div className="flex-1 flex items-center justify-center p-4">
          <p className="text-xs text-muted-foreground text-center">
            Classification will appear after analysis completes.
          </p>
        </div>
      </div>
    )
  }

  const hasOntologies = classification.suggested_ontologies.length > 0

  return (
    <div className="border rounded-md overflow-hidden flex flex-col min-h-0">
      <div className="px-3 py-2 border-b bg-muted/50 shrink-0 flex items-center justify-between">
        <p className="text-xs font-medium text-muted-foreground">Classification</p>
        {dirty && <Badge variant="secondary" className="text-[10px]">Unsaved</Badge>}
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-3">
        {/* Document Type */}
        <div className="space-y-1">
          <Label className="text-xs">Document Type</Label>
          <Select value={classification.document_type} onValueChange={v => onUpdateField('document_type', v)}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              {DOCUMENT_TYPES.map(dt => (
                <SelectItem key={dt} value={dt} className="text-xs">{formatLabel(dt)}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Domain */}
        <div className="space-y-1">
          <Label className="text-xs">Domain</Label>
          <Select value={classification.domain} onValueChange={v => onUpdateField('domain', v)}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              {DOMAINS.map(d => (
                <SelectItem key={d} value={d} className="text-xs">{formatLabel(d)}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Language */}
        <div className="space-y-1">
          <Label className="text-xs">Language</Label>
          <Input
            value={classification.language}
            onChange={e => onUpdateField('language', e.target.value)}
            className="h-8 text-xs"
            placeholder="en"
          />
        </div>

        {/* Formality */}
        <div className="space-y-1">
          <Label className="text-xs">Formality</Label>
          <Select value={classification.formality_level} onValueChange={v => onUpdateField('formality_level', v)}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              {FORMALITY_LEVELS.map(f => (
                <SelectItem key={f} value={f} className="text-xs">{formatLabel(f)}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Key Topics */}
        <div className="space-y-1">
          <Label className="text-xs">Key Topics</Label>
          <div className="flex flex-wrap gap-1">
            {classification.key_topics.map(t => (
              <Badge key={t} variant="secondary" className="text-[10px] gap-1 pr-1">
                {t}
                <button type="button" onClick={() => onRemoveTopic(t)} className="hover:text-destructive">
                  <X className="h-2.5 w-2.5" />
                </button>
              </Badge>
            ))}
          </div>
          <div className="flex gap-1">
            <Input
              value={newTopic}
              onChange={e => onNewTopicChange(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); onAddTopic() } }}
              className="h-7 text-xs flex-1"
              placeholder="Add topic..."
            />
            <Button type="button" variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={onAddTopic}>
              <Plus className="h-3 w-3" />
            </Button>
          </div>
        </div>

        {/* Ontologies — emphasized as required for entity extraction */}
        <div className={cn("space-y-1 p-2 rounded-md border", hasOntologies ? "border-green-500/30 bg-green-50/50 dark:bg-green-950/20" : "border-amber-500/50 bg-amber-50/50 dark:bg-amber-950/20")}>
          <Label className="text-xs flex items-center gap-1.5">
            Ontologies
            {!hasOntologies && (
              <Badge variant="outline" className="text-[9px] border-amber-500 text-amber-600">Required for entities</Badge>
            )}
          </Label>
          <div className="flex flex-wrap gap-1">
            {classification.suggested_ontologies.map(o => (
              <Badge key={o} variant="secondary" className="text-[10px] gap-1 pr-1">
                {o}
                <button type="button" onClick={() => onRemoveOntology(o)} className="hover:text-destructive">
                  <X className="h-2.5 w-2.5" />
                </button>
              </Badge>
            ))}
          </div>
          <div className="flex gap-1">
            <Input
              value={newOntology}
              onChange={e => onNewOntologyChange(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); onAddOntology() } }}
              className="h-7 text-xs flex-1"
              placeholder="Add ontology..."
            />
            <Button type="button" variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={onAddOntology}>
              <Plus className="h-3 w-3" />
            </Button>
          </div>
        </div>

        {/* Structure toggles */}
        <div className="space-y-2">
          <Label className="text-xs">Structure</Label>
          <div className="flex items-center gap-2">
            <Switch
              checked={classification.has_toc}
              onCheckedChange={v => onUpdateField('has_toc', v)}
              className="scale-75"
            />
            <span className="text-xs">Has TOC</span>
          </div>
          <div className="flex items-center gap-2">
            <Switch
              checked={classification.has_hierarchical_structure}
              onCheckedChange={v => onUpdateField('has_hierarchical_structure', v)}
              className="scale-75"
            />
            <span className="text-xs">Hierarchical</span>
          </div>
        </div>
      </div>

      {/* Save button */}
      <div className="px-3 py-2 border-t shrink-0">
        <Button
          type="button"
          size="sm"
          className="w-full gap-1.5"
          onClick={onSave}
          disabled={!dirty || saving}
        >
          {saving ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Save className="h-3.5 w-3.5" />
          )}
          Save Classification
        </Button>
      </div>
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

function formatLabel(s: string): string {
  return s.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}
