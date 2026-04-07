'use client'

import { useState, useEffect, useCallback } from 'react'
import {
  ScanSearch,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  Play,
  Save,
  X,
  Plus,
  FileText,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Checkbox } from '@/components/ui/checkbox'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import ReactMarkdown from 'react-markdown'
import { toast } from 'sonner'
import { cn } from '@/lib/utils'
import { preprocessingApi, type PreprocessingResult, type DocumentClassification } from '@/lib/api/preprocessing'
import { sourcesApi, DEFAULT_PIPELINE_CONFIG, type DoclingPipelineConfig } from '@/lib/api/sources'
import { PipelineConfigPanel } from '../PipelineConfigPanel'
import type { StepStatus } from '../PipelineStepper'
import type { FileEntry } from './ExtractionTab'

const DOCUMENT_TYPES = [
  'academic_paper',
  'policy_document',
  'legal_document',
  'technical_report',
  'transcript',
  'news_article',
  'book_chapter',
  'correspondence',
  'manual',
  'financial_report',
  'presentation',
  'other',
]

const DOMAINS = [
  'science',
  'technology',
  'law',
  'politics',
  'healthcare',
  'finance',
  'education',
  'business',
  'engineering',
  'other',
]

const FORMALITY_LEVELS = ['formal', 'semi_formal', 'informal']

interface ChunkEntry {
  id: string
  text: string
  order: number
  physical_page: number | null
  element_type: string | null
  is_content: boolean
}

interface PreprocessingTabProps {
  status: StepStatus
  extractionComplete?: boolean
  sourceId?: string
  files?: FileEntry[]
}

export function PreprocessingTab({
  status,
  extractionComplete,
  sourceId,
  files = [],
}: PreprocessingTabProps) {
  // Left pane: selected file
  const [selectedFileIndex, setSelectedFileIndex] = useState(0)

  // Middle pane: result & editable classification
  const [result, setResult] = useState<PreprocessingResult | null>(null)
  const [editedClassification, setEditedClassification] = useState<DocumentClassification | null>(null)
  const [editedFilteredIds, setEditedFilteredIds] = useState<string[]>([])
  const [editedRemovedIds, setEditedRemovedIds] = useState<string[]>([])

  // Right pane: chunks
  const [chunks, setChunks] = useState<ChunkEntry[]>([])
  const [loadingChunks, setLoadingChunks] = useState(false)

  // Pipeline config
  const [pipelineConfig, setPipelineConfig] = useState<DoclingPipelineConfig>({ ...DEFAULT_PIPELINE_CONFIG })
  const [reprocessing, setReprocessing] = useState(false)

  // UI state
  const [running, setRunning] = useState(false)
  const [saving, setSaving] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeResultTab, setActiveResultTab] = useState<string>('summary')
  const [newTopic, setNewTopic] = useState('')
  const [newOntology, setNewOntology] = useState('')

  const selectedFile = files[selectedFileIndex] || null
  const selectedSourceId = selectedFile?.sourceId || sourceId

  // Load result + chunks when selected source changes
  useEffect(() => {
    if (!selectedSourceId) {
      setResult(null)
      setEditedClassification(null)
      setChunks([])
      return
    }

    let cancelled = false

    // Load preprocessing result
    preprocessingApi.get(selectedSourceId).then(r => {
      if (cancelled) return
      if (r) {
        setResult(r)
        setEditedClassification({ ...r.classification })
        setEditedFilteredIds([...r.filtered_chunk_ids])
        setEditedRemovedIds([...r.removed_chunk_ids])
        setDirty(false)
      } else {
        setResult(null)
        setEditedClassification(null)
      }
    })

    // Load chunks
    setLoadingChunks(true)
    sourcesApi.getChunks(selectedSourceId).then(data => {
      if (cancelled) return
      setChunks(
        (data.chunks as ChunkEntry[]).sort((a, b) => a.order - b.order)
      )
      setLoadingChunks(false)
    }).catch(() => {
      if (!cancelled) {
        setChunks([])
        setLoadingChunks(false)
      }
    })

    return () => { cancelled = true }
  }, [selectedSourceId])

  // Run analysis
  const handleRun = useCallback(async () => {
    if (!selectedSourceId) return
    setRunning(true)
    setError(null)
    try {
      const r = await preprocessingApi.run(selectedSourceId)
      setResult(r)
      setEditedClassification({ ...r.classification })
      setEditedFilteredIds([...r.filtered_chunk_ids])
      setEditedRemovedIds([...r.removed_chunk_ids])
      setDirty(false)

      // Refresh chunks in case filtering changed
      const data = await sourcesApi.getChunks(selectedSourceId)
      setChunks(
        (data.chunks as ChunkEntry[]).sort((a, b) => a.order - b.order)
      )
    } catch (e: unknown) {
      const axiosErr = e as { response?: { data?: { detail?: string } }; message?: string }
      const detail = axiosErr?.response?.data?.detail
      setError(detail || axiosErr?.message || 'Preprocessing failed')
    } finally {
      setRunning(false)
    }
  }, [selectedSourceId])

  // Save edits
  const handleSave = useCallback(async () => {
    if (!selectedSourceId || !editedClassification) return
    setSaving(true)
    setError(null)
    try {
      const updated = await preprocessingApi.update(selectedSourceId, {
        classification: editedClassification,
        filtered_chunk_ids: editedFilteredIds,
        removed_chunk_ids: editedRemovedIds,
      })
      setResult(updated)
      setEditedClassification({ ...updated.classification })
      setEditedFilteredIds([...updated.filtered_chunk_ids])
      setEditedRemovedIds([...updated.removed_chunk_ids])
      setDirty(false)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Save failed')
    } finally {
      setSaving(false)
    }
  }, [selectedSourceId, editedClassification, editedFilteredIds, editedRemovedIds])

  // Reprocess with pipeline config
  const handleReprocess = useCallback(async () => {
    if (!selectedSourceId) return
    setReprocessing(true)
    setError(null)
    try {
      await sourcesApi.reprocess(selectedSourceId, pipelineConfig)
      toast.success('Reprocessing started — document will be re-ingested with new settings')
    } catch (e: unknown) {
      const axiosErr = e as { response?: { data?: { detail?: string } }; message?: string }
      const detail = axiosErr?.response?.data?.detail
      setError(detail || axiosErr?.message || 'Reprocessing failed')
    } finally {
      setReprocessing(false)
    }
  }, [selectedSourceId, pipelineConfig])

  // Classification field updaters
  const updateField = <K extends keyof DocumentClassification>(
    field: K,
    value: DocumentClassification[K]
  ) => {
    if (!editedClassification) return
    setEditedClassification({ ...editedClassification, [field]: value })
    setDirty(true)
  }

  // Topic management
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

  // Ontology management
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

  // Chunk toggle
  const toggleChunk = (chunkId: string) => {
    const isCurrentlyKept = editedFilteredIds.includes(chunkId)
    if (isCurrentlyKept) {
      setEditedFilteredIds(prev => prev.filter(id => id !== chunkId))
      setEditedRemovedIds(prev => [...prev, chunkId])
    } else {
      setEditedRemovedIds(prev => prev.filter(id => id !== chunkId))
      setEditedFilteredIds(prev => [...prev, chunkId])
    }
    setDirty(true)
  }

  // Determine initial checkbox state for chunks (before any result exists)
  const isChunkKept = (chunk: ChunkEntry): boolean => {
    if (result) {
      return editedFilteredIds.includes(String(chunk.id))
    }
    // Fallback: use is_content from chunk data
    return chunk.is_content
  }

  // Locked state
  if (!extractionComplete) {
    return (
      <Card className="flex-1 flex flex-col min-h-0">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 shrink-0">
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <ScanSearch className="h-4 w-4" />
            Document Preprocessing
          </CardTitle>
          <Badge variant="outline">Locked</Badge>
        </CardHeader>
        <CardContent className="flex items-center justify-center flex-1">
          <p className="text-sm text-muted-foreground">
            Preprocessing available after extraction completes.
          </p>
        </CardContent>
      </Card>
    )
  }

  const keptCount = result ? editedFilteredIds.length : chunks.filter(c => c.is_content).length
  const removedCount = result ? editedRemovedIds.length : chunks.filter(c => !c.is_content).length
  const totalCount = result ? (editedFilteredIds.length + editedRemovedIds.length) : chunks.length

  return (
    <Card className="flex-1 flex flex-col min-h-0">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 shrink-0">
        <CardTitle className="text-base font-medium flex items-center gap-2">
          <ScanSearch className="h-4 w-4" />
          Document Preprocessing
        </CardTitle>
        <div className="flex items-center gap-2">
          {dirty && <Badge variant="secondary" className="text-xs">Unsaved changes</Badge>}
          <StatusBadge status={running ? 'running' : result ? 'completed' : status} />
        </div>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col min-h-0">
        {error && (
          <div className="flex items-center gap-2 pb-2 text-destructive shrink-0">
            <XCircle className="h-4 w-4" />
            <span className="text-sm">{error}</span>
          </div>
        )}

        {/* Pipeline Configuration (collapsible) */}
        <div className="mb-3 shrink-0">
          <PipelineConfigPanel
            config={pipelineConfig}
            onChange={setPipelineConfig}
            onReprocess={handleReprocess}
            reprocessing={reprocessing}
            collapsible={true}
            defaultOpen={false}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr_2fr] gap-3 flex-1 min-h-0">
          {/* Left pane: file list */}
          <FileListPane
            files={files}
            selectedIndex={selectedFileIndex}
            onSelect={setSelectedFileIndex}
          />

          {/* Middle pane: classification (editable) */}
          <ClassificationPane
            result={result}
            classification={editedClassification}
            running={running}
            saving={saving}
            dirty={dirty}
            newTopic={newTopic}
            newOntology={newOntology}
            onNewTopicChange={setNewTopic}
            onNewOntologyChange={setNewOntology}
            onUpdateField={updateField}
            onRemoveTopic={removeTopic}
            onAddTopic={addTopic}
            onRemoveOntology={removeOntology}
            onAddOntology={addOntology}
            onRun={handleRun}
            onSave={handleSave}
          />

          {/* Right pane: summary + chunks */}
          <ResultsPane
            result={result}
            activeTab={activeResultTab}
            onTabChange={setActiveResultTab}
            chunks={chunks}
            loadingChunks={loadingChunks}
            keptCount={keptCount}
            removedCount={removedCount}
            totalCount={totalCount}
            isChunkKept={isChunkKept}
            onToggleChunk={toggleChunk}
            running={running}
            hasResult={!!result}
          />
        </div>
      </CardContent>
    </Card>
  )
}

/* ------------------------------------------------------------------ */
/* Left pane — file list                                               */
/* ------------------------------------------------------------------ */

function FileListPane({
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
        <p className="text-xs font-medium text-muted-foreground">Files</p>
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
            <p className="text-xs text-muted-foreground px-2 py-3">No files</p>
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
/* Middle pane — editable classification                               */
/* ------------------------------------------------------------------ */

function ClassificationPane({
  result,
  classification,
  running,
  saving,
  dirty,
  newTopic,
  newOntology,
  onNewTopicChange,
  onNewOntologyChange,
  onUpdateField,
  onRemoveTopic,
  onAddTopic,
  onRemoveOntology,
  onAddOntology,
  onRun,
  onSave,
}: {
  result: PreprocessingResult | null
  classification: DocumentClassification | null
  running: boolean
  saving: boolean
  dirty: boolean
  newTopic: string
  newOntology: string
  onNewTopicChange: (v: string) => void
  onNewOntologyChange: (v: string) => void
  onUpdateField: <K extends keyof DocumentClassification>(field: K, value: DocumentClassification[K]) => void
  onRemoveTopic: (topic: string) => void
  onAddTopic: () => void
  onRemoveOntology: (ont: string) => void
  onAddOntology: () => void
  onRun: () => void
  onSave: () => void
}) {
  return (
    <div className="border rounded-md overflow-hidden flex flex-col min-h-0">
      <div className="px-3 py-2 border-b bg-muted/50 shrink-0">
        <p className="text-xs font-medium text-muted-foreground">Classification</p>
      </div>

      {!result && !running ? (
        <div className="flex-1 flex flex-col items-center justify-center p-4 gap-3">
          <p className="text-sm text-muted-foreground text-center">
            Analyze the document to generate a summary, classify its type and domain, and filter noise chunks.
          </p>
          <Button onClick={onRun} size="sm" className="gap-2">
            <Play className="h-4 w-4" />
            Run Analysis
          </Button>
        </div>
      ) : running ? (
        <div className="flex-1 flex items-center justify-center p-4">
          <div className="flex items-center gap-3">
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
            <span className="text-sm text-muted-foreground">Analyzing document...</span>
          </div>
        </div>
      ) : classification ? (
        <>
          <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-3">
            {/* Document Type */}
            <div className="space-y-1">
              <Label className="text-xs">Document Type</Label>
              <Select
                value={classification.document_type}
                onValueChange={v => onUpdateField('document_type', v)}
              >
                <SelectTrigger className="h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {DOCUMENT_TYPES.map(dt => (
                    <SelectItem key={dt} value={dt} className="text-xs">
                      {formatLabel(dt)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Domain */}
            <div className="space-y-1">
              <Label className="text-xs">Domain</Label>
              <Select
                value={classification.domain}
                onValueChange={v => onUpdateField('domain', v)}
              >
                <SelectTrigger className="h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {DOMAINS.map(d => (
                    <SelectItem key={d} value={d} className="text-xs">
                      {formatLabel(d)}
                    </SelectItem>
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
              <Select
                value={classification.formality_level}
                onValueChange={v => onUpdateField('formality_level', v)}
              >
                <SelectTrigger className="h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {FORMALITY_LEVELS.map(f => (
                    <SelectItem key={f} value={f} className="text-xs">
                      {formatLabel(f)}
                    </SelectItem>
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
                    <button
                      type="button"
                      onClick={() => onRemoveTopic(t)}
                      className="hover:text-destructive"
                    >
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

            {/* Suggested Ontologies */}
            <div className="space-y-1">
              <Label className="text-xs">Suggested Ontologies</Label>
              <div className="flex flex-wrap gap-1">
                {classification.suggested_ontologies.map(o => (
                  <Badge key={o} variant="secondary" className="text-[10px] gap-1 pr-1">
                    {o}
                    <button
                      type="button"
                      onClick={() => onRemoveOntology(o)}
                      className="hover:text-destructive"
                    >
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
          </div>

          {/* Action buttons */}
          <div className="px-3 py-2 border-t shrink-0 flex gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="flex-1 gap-1.5"
              onClick={onRun}
              disabled={running}
            >
              {running ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Play className="h-3.5 w-3.5" />
              )}
              Re-run
            </Button>
            <Button
              type="button"
              size="sm"
              className="flex-1 gap-1.5"
              onClick={onSave}
              disabled={!dirty || saving}
            >
              {saving ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Save className="h-3.5 w-3.5" />
              )}
              Save
            </Button>
          </div>
        </>
      ) : null}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Right pane — summary + chunks tabs                                  */
/* ------------------------------------------------------------------ */

function ResultsPane({
  result,
  activeTab,
  onTabChange,
  chunks,
  loadingChunks,
  keptCount,
  removedCount,
  totalCount,
  isChunkKept,
  onToggleChunk,
  running,
  hasResult,
}: {
  result: PreprocessingResult | null
  activeTab: string
  onTabChange: (tab: string) => void
  chunks: ChunkEntry[]
  loadingChunks: boolean
  keptCount: number
  removedCount: number
  totalCount: number
  isChunkKept: (chunk: ChunkEntry) => boolean
  onToggleChunk: (chunkId: string) => void
  running: boolean
  hasResult: boolean
}) {
  if (!hasResult && !running) {
    return (
      <div className="border rounded-md overflow-hidden flex flex-col min-h-0">
        <div className="px-3 py-2 border-b bg-muted/50 shrink-0">
          <p className="text-xs font-medium text-muted-foreground">Results</p>
        </div>
        <div className="flex-1 flex items-center justify-center p-6">
          <p className="text-sm text-muted-foreground text-center">
            Run analysis to see summary and chunk filtering.
          </p>
        </div>
      </div>
    )
  }

  if (running) {
    return (
      <div className="border rounded-md overflow-hidden flex flex-col min-h-0">
        <div className="px-3 py-2 border-b bg-muted/50 shrink-0">
          <p className="text-xs font-medium text-muted-foreground">Results</p>
        </div>
        <div className="flex-1 flex items-center justify-center p-6">
          <div className="flex items-center gap-3">
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
            <span className="text-sm text-muted-foreground">Generating summary and classification...</span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <Tabs
      value={activeTab}
      onValueChange={onTabChange}
      className="border rounded-md overflow-hidden flex flex-col min-h-0 gap-0"
    >
      <div className="px-3 py-2 border-b bg-muted/50 shrink-0">
        <TabsList className="h-7">
          <TabsTrigger value="summary" className="text-xs h-6 px-2.5 gap-1">
            <FileText className="h-3 w-3" />
            Summary
          </TabsTrigger>
          <TabsTrigger value="chunks" className="text-xs h-6 px-2.5 gap-1">
            Chunks
            <Badge variant="secondary" className="text-[10px] h-4 px-1 ml-0.5">
              {keptCount}/{totalCount}
            </Badge>
          </TabsTrigger>
        </TabsList>
      </div>

      <TabsContent value="summary" className="flex-1 min-h-0 overflow-y-auto mt-0">
        <div className="p-4">
          {result?.naive_summary ? (
            <div className="prose prose-sm dark:prose-invert max-w-none prose-p:text-muted-foreground prose-headings:text-foreground prose-p:leading-relaxed prose-p:text-sm prose-li:text-sm prose-li:text-muted-foreground">
              <ReactMarkdown>{result.naive_summary}</ReactMarkdown>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground italic">No summary available.</p>
          )}
        </div>
      </TabsContent>

      <TabsContent value="chunks" className="flex-1 min-h-0 overflow-y-auto mt-0">
        <div className="p-3 space-y-1">
          {/* Header stats */}
          <div className="flex items-center gap-2 pb-2 border-b mb-2">
            <Badge variant="secondary" className="text-[10px]">
              {keptCount} kept
            </Badge>
            <Badge variant="outline" className="text-[10px]">
              {removedCount} removed
            </Badge>
            <span className="text-[10px] text-muted-foreground">
              of {totalCount} total
            </span>
          </div>

          {loadingChunks ? (
            <div className="flex items-center gap-2 py-4 text-sm text-muted-foreground">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Loading chunks...
            </div>
          ) : chunks.length === 0 ? (
            <p className="text-xs text-muted-foreground italic py-4">No chunks found.</p>
          ) : (
            chunks.map(chunk => {
              const kept = isChunkKept(chunk)
              return (
                <div
                  key={chunk.id}
                  className={cn(
                    "flex items-start gap-2 px-2 py-1.5 rounded text-xs transition-colors",
                    !kept && "opacity-50"
                  )}
                >
                  <Checkbox
                    checked={kept}
                    onCheckedChange={() => onToggleChunk(String(chunk.id))}
                    className="mt-0.5 shrink-0"
                    disabled={!hasResult}
                  />
                  <span className="text-[10px] text-muted-foreground w-6 shrink-0 text-right mt-0.5">
                    {chunk.order}
                  </span>
                  {chunk.element_type && (
                    <Badge
                      variant="outline"
                      className="text-[9px] h-4 px-1 shrink-0 mt-0.5"
                    >
                      {chunk.element_type}
                    </Badge>
                  )}
                  <span className="flex-1 text-muted-foreground leading-snug line-clamp-2">
                    {chunk.text?.slice(0, 120) || '(empty)'}
                  </span>
                  {chunk.physical_page != null && (
                    <span className="text-[10px] text-muted-foreground/60 shrink-0 mt-0.5">
                      p.{chunk.physical_page}
                    </span>
                  )}
                </div>
              )
            })
          )}
        </div>
      </TabsContent>
    </Tabs>
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
    default:
      return <Badge variant="outline">Pending</Badge>
  }
}

function formatLabel(s: string): string {
  return s
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}
