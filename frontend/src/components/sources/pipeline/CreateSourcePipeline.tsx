'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { useNotebooks } from '@/lib/hooks/use-notebooks'
import {
  useCreateSource,
  useSourcePipeline,
  useSourceStatus,
  useRetrySource,
} from '@/lib/hooks/use-sources'
import { sourcesApi } from '@/lib/api/sources'
import { useSettings } from '@/lib/hooks/use-settings'
import { useToast } from '@/lib/hooks/use-toast'
import { CreateSourceRequest, SettingsResponse } from '@/lib/types/api'
import type { ProcessingStage } from '@/lib/pipeline/processing-stage'
import { toPipelineCounts } from '@/lib/pipeline/source-counts'
import type { PipelineNodeAction } from '@/lib/pipeline/pipeline-stages'
import { SourceTypeStep } from '../steps/SourceTypeStep'
import { NotebooksStep } from '../steps/NotebooksStep'
import { AdvancedIngestionSettings } from '../steps/AdvancedIngestionSettings'
import { PipelineHeader } from './PipelineHeader'
import { PipelineStepper, type PipelineStep, type StepStatus } from './PipelineStepper'
import { PipelineFooter, type PipelinePhase } from './PipelineFooter'
import { PipelineStatus } from './PipelineStatus'
import { ProcessingLogConsole } from './ProcessingLogConsole'
import { CompletionTab } from './tabs/CompletionTab'
import { BatchModeDialog, type BatchMode } from './BatchModeDialog'

// Same schema as AddSourceDialog
const createSourceSchema = z.object({
  type: z.enum(['link', 'upload', 'text', 'queue']),
  title: z.string().optional(),
  url: z.string().optional(),
  content: z.string().optional(),
  file: z.any().optional(),
  notebooks: z.array(z.string()).optional(),
  embed: z.boolean(),
  async_processing: z.boolean(),
}).refine((data) => {
  if (data.type === 'link') return !!data.url && data.url.trim() !== ''
  if (data.type === 'text') return !!data.content && data.content.trim() !== ''
  if (data.type === 'upload') {
    if (data.file instanceof FileList) return data.file.length > 0
    return !!data.file
  }
  if (data.type === 'queue') return false
  return true
}, {
  message: 'Please provide the required content for the selected source type',
  path: ['type'],
}).refine((data) => {
  if (data.type === 'text') return !!data.title && data.title.trim() !== ''
  return true
}, {
  message: 'Title is required for text sources',
  path: ['title'],
})

type CreateSourceFormData = z.infer<typeof createSourceSchema>

// Lean 4-step creation flow (Track UX.5). The mandatory Config step and the old
// Extract/Postprocess/Classification/Entities/Embed manual tabs are gone: the
// pipeline runs automatically and its progress is surfaced by the live
// `PipelineStatus` tracker on the Processing step.
const STEP_INPUT = 1
const STEP_ORGANIZE = 2
const STEP_PROCESSING = 3
const STEP_DONE = 4
const LAST_CONFIG_STEP = STEP_ORGANIZE

const STEP_LABELS = [
  { id: STEP_INPUT, label: 'Input', phase: 'config' as const },
  { id: STEP_ORGANIZE, label: 'Organize', phase: 'config' as const },
  { id: STEP_PROCESSING, label: 'Processing', phase: 'pipeline' as const, description: 'Runs automatically — ingest, embed, extract, graph' },
  { id: STEP_DONE, label: 'Done', phase: 'pipeline' as const },
]

interface MultiSourceEntry {
  id: string
  title: string
  status: 'pending' | 'creating' | 'processing' | 'completed' | 'failed'
  stage?: ProcessingStage
  fileName?: string
}

function isTerminalEntry(status: MultiSourceEntry['status']): boolean {
  return status === 'completed' || status === 'failed'
}

export function CreateSourcePipeline() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const { toast } = useToast()

  // URL params
  const defaultNotebookId = searchParams.get('notebook') || undefined

  // Phase & navigation state
  const [phase, setPhase] = useState<PipelinePhase>('config')
  const [activeTab, setActiveTab] = useState(STEP_INPUT)
  const [sourceId, setSourceId] = useState<string | undefined>()

  // Multi-file state
  const [sourceIds, setSourceIds] = useState<string[]>([])
  const [multiSources, setMultiSources] = useState<MultiSourceEntry[]>([])
  const [showBatchDialog, setShowBatchDialog] = useState(false)
  const [batchMode, setBatchMode] = useState<BatchMode | null>(null)
  const pendingFormDataRef = useRef<CreateSourceFormData | null>(null)

  // Selection state
  const [selectedNotebooks, setSelectedNotebooks] = useState<string[]>(
    defaultNotebookId ? [defaultNotebookId] : []
  )
  const [processingOverrides, setProcessingOverrides] = useState<Partial<SettingsResponse>>({})

  // Track whether embed was enabled (for the completion summary)
  const [embedEnabled, setEmbedEnabled] = useState(false)

  // API hooks
  const createSource = useCreateSource()
  const retrySource = useRetrySource()
  const { data: notebooks = [], isLoading: notebooksLoading } = useNotebooks()
  const { data: settings } = useSettings()

  const isMulti = sourceIds.length > 1

  // Single-source spine: poll GET /sources/{id} for `processing_stage` (the
  // stage source of truth) and the job axis for the current node's spinner.
  const singleEnabled = !!sourceId && !isMulti && phase !== 'config'
  const { data: pipelineData } = useSourcePipeline(sourceId || '', singleEnabled)
  const statusEnabled = phase === 'processing' && !isMulti
  const { data: statusData } = useSourceStatus(sourceId || '', statusEnabled)

  const singleStage = pipelineData?.processing_stage
  const singleComplete = singleStage === 'complete'

  // Multi-source aggregate: each entry's status is driven by its OWN
  // `processing_stage` (polled below), never a shared inference.
  const multiAllDone =
    multiSources.length > 0 && multiSources.every((s) => isTerminalEntry(s.status))

  const processingComplete = isMulti ? multiAllDone : singleComplete

  // Multi-source stage polling: read each in-flight entry's `processing_stage`
  // from GET /sources/{id} and map it to the entry's status.
  useEffect(() => {
    if (phase !== 'processing' || !isMulti) return

    const pollInterval = setInterval(async () => {
      const stillProcessing = multiSources.filter(
        (s) => s.id && (s.status === 'processing' || s.status === 'creating')
      )
      if (stillProcessing.length === 0) {
        clearInterval(pollInterval)
        return
      }

      for (const entry of stillProcessing) {
        try {
          const source = await sourcesApi.get(entry.id)
          const stage = source.processing_stage
          const nextStatus: MultiSourceEntry['status'] =
            stage === 'complete' ? 'completed' : stage === 'failed' ? 'failed' : 'processing'
          setMultiSources((prev) =>
            prev.map((s) => (s.id === entry.id ? { ...s, stage, status: nextStatus } : s))
          )
        } catch {
          // Ignore transient polling errors — the next tick retries.
        }
      }
    }, 3000)

    return () => clearInterval(pollInterval)
  }, [phase, isMulti, multiSources])

  // Advance the flow to Done once the pipeline settles. `complete` ⇒ Done;
  // `awaiting_schema_review` parks on the gated node (no false Done); `failed`
  // stays on the Processing step with the failed node + Retry.
  useEffect(() => {
    if (phase !== 'processing') return
    if (processingComplete) {
      setPhase('complete')
      setActiveTab(STEP_DONE)
    }
  }, [phase, processingComplete])

  // Form
  const {
    register,
    handleSubmit,
    control,
    watch,
    formState: { errors },
    reset,
  } = useForm<CreateSourceFormData>({
    resolver: zodResolver(createSourceSchema),
    defaultValues: {
      notebooks: defaultNotebookId ? [defaultNotebookId] : [],
      embed: settings?.default_embedding_option === 'always' || settings?.default_embedding_option === 'ask',
      async_processing: true,
    },
  })

  // Initialize defaults when settings load
  useEffect(() => {
    if (settings) {
      const embedValue = settings.default_embedding_option === 'always' ||
                         settings.default_embedding_option === 'ask'
      reset({
        notebooks: defaultNotebookId ? [defaultNotebookId] : [],
        embed: embedValue,
        async_processing: true,
      })
    }
  }, [settings, defaultNotebookId, reset])

  // Watch form values for step validation
  const selectedType = watch('type')
  const watchedUrl = watch('url')
  const watchedContent = watch('content')
  const watchedFile = watch('file')
  const watchedTitle = watch('title')

  // Step validation (config steps only — the pipeline runs itself)
  const isStepValid = useCallback((step: number): boolean => {
    switch (step) {
      case STEP_INPUT:
        if (!selectedType) return false
        if (selectedType === 'queue') return false
        if (selectedType === 'link') return !!watchedUrl && watchedUrl.trim() !== ''
        if (selectedType === 'text') {
          return !!watchedContent && watchedContent.trim() !== '' &&
                 !!watchedTitle && watchedTitle.trim() !== ''
        }
        if (selectedType === 'upload') {
          if (watchedFile instanceof FileList) return watchedFile.length > 0
          return !!watchedFile
        }
        return true
      case STEP_ORGANIZE:
        return true
      default:
        return false
    }
  }, [selectedType, watchedUrl, watchedContent, watchedFile, watchedTitle])

  // Processing step status for the stepper.
  const processingStepStatus: StepStatus = (() => {
    if (phase === 'config') return 'locked'
    if (phase === 'complete' || processingComplete) return 'completed'
    if (!isMulti && singleStage === 'failed') return 'failed'
    return 'running'
  })()

  const doneStepStatus: StepStatus =
    phase === 'complete' || processingComplete ? 'completed' : 'locked'

  // Build step objects for stepper
  const steps: PipelineStep[] = STEP_LABELS.map((s) => {
    let status: StepStatus = 'pending'
    if (s.phase === 'config') {
      if (s.id < activeTab) status = 'completed'
      else if (s.id === activeTab) status = 'active'
      else status = 'pending'
      // Once processing has begun, all config steps are settled.
      if (phase !== 'config') status = 'completed'
    } else if (s.id === STEP_PROCESSING) {
      status = processingStepStatus
    } else {
      status = doneStepStatus
    }
    return { ...s, status }
  })

  // Step click handler
  const handleStepClick = (stepId: number) => {
    const step = steps.find((s) => s.id === stepId)
    if (!step) return
    // Config tabs are navigable while still configuring.
    if (step.phase === 'config' && phase === 'config') {
      setActiveTab(stepId)
      return
    }
    // Pipeline tabs are navigable once unlocked.
    if (step.status !== 'locked') {
      setActiveTab(stepId)
    }
  }

  // Navigation
  const handleNext = () => {
    if (activeTab < LAST_CONFIG_STEP && isStepValid(activeTab)) {
      setActiveTab(activeTab + 1)
    }
  }

  const handleBack = () => {
    if (activeTab > STEP_INPUT) {
      setActiveTab(activeTab - 1)
    }
  }

  // Recovery actions on the live tracker's gated/failed nodes.
  const handleNodeAction = useCallback(
    (action: PipelineNodeAction) => {
      if (!sourceId) return
      if (action === 'retry') {
        retrySource.mutate(sourceId)
      } else if (action === 'review-schema') {
        router.push(`/sources/${sourceId}`)
      }
    },
    [sourceId, retrySource, router]
  )

  // Notebook toggle
  const handleNotebookToggle = (notebookId: string) => {
    setSelectedNotebooks((prev) =>
      prev.includes(notebookId) ? prev.filter((id) => id !== notebookId) : [...prev, notebookId]
    )
  }

  // Build a create request for a single file
  const buildCreateRequest = (data: CreateSourceFormData, file?: File): CreateSourceRequest => {
    const createRequest: CreateSourceRequest = {
      type: data.type === 'queue' ? 'upload' : data.type,
      notebooks: selectedNotebooks,
      url: data.type === 'link' ? data.url : undefined,
      content: data.type === 'text' ? data.content : undefined,
      title: data.title,
      embed: data.embed,
      delete_source: false,
      async_processing: true,
      processing_overrides: Object.keys(processingOverrides).length > 0
        ? processingOverrides
        : undefined,
    }

    if (file) {
      (createRequest as CreateSourceRequest & { file?: File }).file = file
    }

    return createRequest
  }

  // Submit multiple files sequentially (batch upload — preserved from the
  // 9-step flow; only the status source changed to `processing_stage`).
  const submitMultipleFiles = async (data: CreateSourceFormData) => {
    const fileList = data.file as FileList
    const entries: MultiSourceEntry[] = []

    for (let i = 0; i < fileList.length; i++) {
      entries.push({
        id: '',
        title: fileList[i].name,
        status: 'pending',
        fileName: fileList[i].name,
      })
    }
    setMultiSources(entries)

    setEmbedEnabled(!!data.embed || settings?.default_embedding_option === 'always')

    setPhase('processing')
    setActiveTab(STEP_PROCESSING)

    const ids: string[] = []

    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i]
      setMultiSources((prev) => prev.map((e, idx) =>
        idx === i ? { ...e, status: 'creating' as const } : e
      ))

      try {
        const request = buildCreateRequest(data, file)
        // Clear title for individual files — let backend generate from filename
        if (!data.title) {
          request.title = undefined
        }
        const result = await createSource.mutateAsync(request)
        ids.push(result.id)

        setMultiSources((prev) => prev.map((e, idx) =>
          idx === i
            ? { ...e, id: result.id, status: 'processing' as const, title: result.title || file.name }
            : e
        ))
      } catch (error) {
        console.error(`Error creating source for ${file.name}:`, error)
        setMultiSources((prev) => prev.map((e, idx) =>
          idx === i ? { ...e, status: 'failed' as const } : e
        ))
      }
    }

    setSourceIds(ids)
    // Completion is detected by the multi-source stage poller, not here.
  }

  // Form submission
  const onSubmit = async (data: CreateSourceFormData) => {
    const isMultiFile = data.type === 'upload' && data.file instanceof FileList && data.file.length > 1

    if (isMultiFile && batchMode === null) {
      pendingFormDataRef.current = data
      setShowBatchDialog(true)
      return
    }

    if (isMultiFile && batchMode === 'individual') {
      try {
        await submitMultipleFiles(data)
      } catch (error) {
        console.error('Error in multi-file submission:', error)
        toast({
          title: 'Error',
          description: 'Failed to create some sources. Please try again.',
          variant: 'destructive',
        })
      }
      return
    }

    // Single file / link / text submission
    try {
      setEmbedEnabled(!!data.embed || settings?.default_embedding_option === 'always')

      const file = data.type === 'upload' && data.file
        ? (data.file instanceof FileList ? data.file[0] : data.file)
        : undefined

      const createRequest = buildCreateRequest(data, file instanceof File ? file : undefined)

      const result = await createSource.mutateAsync(createRequest)
      setSourceId(result.id)
      setSourceIds([result.id])
      setPhase('processing')
      setActiveTab(STEP_PROCESSING)
    } catch (error) {
      console.error('Error creating source:', error)
      toast({
        title: 'Error',
        description: 'Failed to create source. Please try again.',
        variant: 'destructive',
      })
    }
  }

  // Handle batch mode selection
  const handleBatchModeSelect = (mode: BatchMode) => {
    setBatchMode(mode)
    setShowBatchDialog(false)

    if (pendingFormDataRef.current) {
      if (mode === 'individual') {
        submitMultipleFiles(pendingFormDataRef.current)
      }
      pendingFormDataRef.current = null
    }
  }

  // Reset
  const handleReset = () => {
    reset()
    setPhase('config')
    setActiveTab(STEP_INPUT)
    setSourceId(undefined)
    setSourceIds([])
    setMultiSources([])
    setBatchMode(null)
    setShowBatchDialog(false)
    pendingFormDataRef.current = null
    setSelectedNotebooks(defaultNotebookId ? [defaultNotebookId] : [])
    setProcessingOverrides({})
    setEmbedEnabled(false)
  }

  // Determine source type label for completion
  const getSourceTypeLabel = () => {
    if (selectedType === 'link') return 'Link'
    if (selectedType === 'upload') return 'Upload'
    if (selectedType === 'text') return 'Text'
    return 'Unknown'
  }

  // Compute file count for batch dialog
  const batchFileCount = watchedFile instanceof FileList ? watchedFile.length : 0

  const isProcessingTab = activeTab === STEP_PROCESSING

  return (
    <div className="flex flex-col h-full">
      <PipelineHeader />
      <PipelineStepper
        steps={steps}
        activeTab={activeTab}
        onStepClick={handleStepClick}
      />

      <div className="flex-1 overflow-hidden flex flex-col">
        <div className={isProcessingTab
          ? "flex-1 flex flex-col min-h-0 px-6 py-6"
          : "flex-1 overflow-y-auto max-w-3xl mx-auto w-full px-6 py-6"
        }>
          <form
            onSubmit={handleSubmit(onSubmit)}
            className={isProcessingTab ? "flex-1 flex flex-col min-h-0" : undefined}
          >
            {/* Step 1 — Input (source type + optional advanced ingestion settings) */}
            {activeTab === STEP_INPUT && (
              <div className="space-y-6">
                <SourceTypeStep
                  // @ts-expect-error - Type inference issue with zod schema
                  control={control}
                  register={register}
                  // @ts-expect-error - Type inference issue with zod schema
                  errors={errors}
                />
                <AdvancedIngestionSettings onOverridesChange={setProcessingOverrides} />
              </div>
            )}

            {/* Step 2 — Organize */}
            {activeTab === STEP_ORGANIZE && (
              <NotebooksStep
                notebooks={notebooks}
                selectedNotebooks={selectedNotebooks}
                onToggleNotebook={handleNotebookToggle}
                loading={notebooksLoading}
              />
            )}

            {/* Step 3 — Processing (live pipeline tracker + streaming log) */}
            {activeTab === STEP_PROCESSING && (
              isMulti ? (
                <div className="space-y-3">
                  <p className="text-sm text-muted-foreground">
                    Processing {multiSources.length} sources — each advances through the
                    pipeline independently.
                  </p>
                  {multiSources.map((entry, i) => (
                    <div key={entry.id || i} className="rounded-md border p-3">
                      <div className="mb-2 flex items-center gap-2">
                        <span className="truncate text-sm font-medium" title={entry.title}>
                          {entry.title}
                        </span>
                      </div>
                      <PipelineStatus
                        variant="card"
                        processingStage={entry.stage}
                      />
                    </div>
                  ))}
                </div>
              ) : (
                <PipelineStatus
                  variant="live"
                  processingStage={singleStage}
                  jobStatus={statusData?.status}
                  counts={pipelineData ? toPipelineCounts(pipelineData) : undefined}
                  onNodeAction={handleNodeAction}
                >
                  <ProcessingLogConsole
                    sourceId={sourceId}
                    active={phase === 'processing' && !singleComplete && singleStage !== 'failed'}
                  />
                </PipelineStatus>
              )
            )}

            {/* Step 4 — Done */}
            {activeTab === STEP_DONE && (
              isMulti ? (
                <CompletionTab
                  sourceTitle={`${multiSources.length} sources`}
                  sourceType={getSourceTypeLabel()}
                  embedEnabled={embedEnabled}
                  summariesEnabled={false}
                  sources={multiSources.map((s) => ({
                    id: s.id,
                    title: s.title,
                    status: s.status === 'completed' ? 'completed'
                      : s.status === 'failed' ? 'failed'
                      : 'processing',
                  }))}
                />
              ) : (
                <CompletionTab
                  sourceTitle={pipelineData?.title || watchedTitle}
                  sourceType={getSourceTypeLabel()}
                  chunkCount={pipelineData?.embedded_chunks}
                  entityCount={pipelineData?.entity_count}
                  relationCount={pipelineData?.relation_count}
                  embeddedChunks={pipelineData?.embedded_chunks}
                  insightsCount={pipelineData?.insights_count}
                  embedEnabled={embedEnabled}
                  summariesEnabled={(pipelineData?.insights_count || 0) > 0}
                />
              )
            )}
          </form>
        </div>
      </div>

      <PipelineFooter
        phase={phase}
        activeTab={activeTab}
        lastConfigStep={LAST_CONFIG_STEP}
        isStepValid={isStepValid(activeTab)}
        isSubmitting={createSource.isPending}
        sourceId={sourceIds.length === 1 ? sourceIds[0] : sourceId}
        onBack={handleBack}
        onNext={handleNext}
        onSubmit={handleSubmit(onSubmit)}
        onReset={handleReset}
      />

      <BatchModeDialog
        open={showBatchDialog}
        fileCount={batchFileCount}
        onSelect={handleBatchModeSelect}
        onCancel={() => {
          setShowBatchDialog(false)
          pendingFormDataRef.current = null
        }}
      />
    </div>
  )
}
