'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { useSearchParams } from 'next/navigation'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { useNotebooks } from '@/lib/hooks/use-notebooks'
import { useCreateSource, useSource, useSourceStatus } from '@/lib/hooks/use-sources'
import { sourcesApi } from '@/lib/api/sources'
import { useSettings } from '@/lib/hooks/use-settings'
import { useToast } from '@/lib/hooks/use-toast'
import { CreateSourceRequest, SettingsResponse } from '@/lib/types/api'
import { SourceTypeStep } from '../steps/SourceTypeStep'
import { NotebooksStep } from '../steps/NotebooksStep'
import { ProcessingConfigStep } from '../steps/ProcessingConfigStep'
import { PipelineHeader } from './PipelineHeader'
import { PipelineStepper, type PipelineStep, type StepStatus } from './PipelineStepper'
import { PipelineFooter, type PipelinePhase } from './PipelineFooter'
import { ExtractionTab } from './tabs/ExtractionTab'
import { EntitiesTab } from './tabs/EntitiesTab'
import { EmbeddingTab } from './tabs/EmbeddingTab'
import { PreprocessingTab } from './tabs/PreprocessingTab'
import { SummariesTab } from './tabs/SummariesTab'
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

const STEP_LABELS = [
  { id: 1, label: 'Input', phase: 'config' as const },
  { id: 2, label: 'Organize', phase: 'config' as const },
  { id: 3, label: 'Config', phase: 'config' as const },
  { id: 4, label: 'Extract', phase: 'pipeline' as const, description: 'Required — parses document into text chunks' },
  { id: 5, label: 'Postprocess', phase: 'pipeline' as const, description: 'Optional — review chunks, classify content and filter noise' },
  { id: 6, label: 'Classification', phase: 'pipeline' as const, description: 'Required — classify document type and select ontologies' },
  { id: 7, label: 'Entities', phase: 'pipeline' as const, description: 'Optional — extract people, organizations, concepts for the knowledge graph' },
  { id: 8, label: 'Embed', phase: 'pipeline' as const, description: 'Optional — create vector embeddings for semantic search' },
  { id: 9, label: 'Done', phase: 'pipeline' as const },
]

interface MultiSourceEntry {
  id: string
  title: string
  status: 'pending' | 'creating' | 'processing' | 'completed' | 'failed'
  fileName?: string
}

export function CreateSourcePipeline() {
  const searchParams = useSearchParams()
  const { toast } = useToast()

  // URL params
  const defaultNotebookId = searchParams.get('notebook') || undefined

  // Phase & navigation state
  const [phase, setPhase] = useState<PipelinePhase>('config')
  const [activeTab, setActiveTab] = useState(1)
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
  const [processingOverrides, setProcessingOverrides] = useState<Record<string, unknown>>({})

  // Track classification readiness (ontology selected for all sources)
  const [classificationReady, setClassificationReady] = useState(false)

  // Track whether we've configured embed/transformations
  const [embedEnabled, setEmbedEnabled] = useState(false)
  const [summariesEnabled, setSummariesEnabled] = useState(false)

  // API hooks
  const createSource = useCreateSource()
  const { data: notebooks = [], isLoading: notebooksLoading } = useNotebooks()
  const { data: settings } = useSettings()

  // Poll source data when in processing phase (single source mode)
  const statusEnabled = phase === 'processing' && sourceIds.length <= 1
  const { data: sourceData } = useSource(sourceId || '')
  const { data: statusData } = useSourceStatus(sourceId || '', statusEnabled)

  // Debug: log polling state
  useEffect(() => {
    console.log('[Pipeline] polling debug:', { sourceId, phase, sourceIdsLength: sourceIds.length, statusEnabled, statusData: statusData?.status })
  }, [sourceId, phase, sourceIds.length, statusEnabled, statusData])

  // Track previous auto-advance tab to avoid infinite loops
  const lastAutoAdvancedRef = useRef<number>(0)

  // Multi-source status polling
  useEffect(() => {
    if (phase !== 'processing' || sourceIds.length <= 1) return

    const pollInterval = setInterval(async () => {
      const stillProcessing = multiSources.filter(
        s => s.id && (s.status === 'processing' || s.status === 'creating')
      )
      if (stillProcessing.length === 0) {
        clearInterval(pollInterval)
        return
      }

      for (const entry of stillProcessing) {
        try {
          const statusResult = await sourcesApi.status(entry.id)
          const jobStatus = statusResult.status

          if (jobStatus === 'completed') {
            setMultiSources(prev => prev.map(s =>
              s.id === entry.id ? { ...s, status: 'completed' as const } : s
            ))
          } else if (jobStatus === 'failed') {
            setMultiSources(prev => prev.map(s =>
              s.id === entry.id ? { ...s, status: 'failed' as const } : s
            ))
          }
          // 'running' / 'queued' → keep as 'processing'
        } catch {
          // Ignore polling errors
        }
      }
    }, 3000)

    return () => clearInterval(pollInterval)
  }, [phase, sourceIds.length, multiSources])

  // Form
  const {
    register,
    handleSubmit,
    control,
    watch,
    setValue: setFormValue,
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


  // Step validation
  const isStepValid = useCallback((step: number): boolean => {
    switch (step) {
      case 1:
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
      case 2:
      case 3:
        return true
      default:
        return false
    }
  }, [selectedType, watchedUrl, watchedContent, watchedFile, watchedTitle])

  // Track manual step statuses (user-triggered)
  const [manualStatuses, setManualStatuses] = useState<Record<number, StepStatus>>({
    5: 'pending',  // Preprocess
    6: 'pending',  // Summaries
    7: 'pending',  // Entities
    8: 'pending',  // Embed
  })

  // Derive pipeline stage statuses from polled data
  // Order: 4=Extract, 5=Preprocess, 6=Summaries, 7=Entities, 8=Embed, 9=Done
  const derivePipelineStatuses = useCallback((): Record<number, StepStatus> => {
    if (phase === 'config') {
      return { 4: 'locked', 5: 'locked', 6: 'locked', 7: 'locked', 8: 'locked', 9: 'locked' }
    }

    // Multi-source mode: status based on multiSources state
    if (sourceIds.length > 1) {
      const allDone = multiSources.every(s => s.status === 'completed' || s.status === 'failed')
      const anyProcessing = multiSources.some(s => s.status === 'processing' || s.status === 'creating')

      let extractionStatus: StepStatus = 'running'
      if (allDone) extractionStatus = 'completed'
      else if (anyProcessing) extractionStatus = 'running'

      const unlocked = extractionStatus === 'completed' ? 'pending' : 'locked'
      return {
        4: extractionStatus,
        5: unlocked,
        6: unlocked,
        7: unlocked,
        8: unlocked,
        9: 'locked',
      }
    }

    // Single source mode
    const jobStatus = statusData?.status
    const hasFullText = !!sourceData?.full_text
    const hasEmbeddings = (sourceData?.embedded_chunks || 0) > 0
    const hasInsights = (sourceData?.insights_count || 0) > 0
    const jobFailed = jobStatus === 'failed'
    const jobCompleted = jobStatus === 'completed'

    // Step 4: Extraction (automatic)
    let extractionStatus: StepStatus = phase === 'processing' ? 'running' : 'pending'
    if (hasFullText || jobCompleted) extractionStatus = 'completed'
    else if (jobFailed) extractionStatus = 'failed'
    else if (jobStatus === 'running' || jobStatus === 'queued') extractionStatus = 'running'

    const extractionDone = extractionStatus === 'completed'

    // Step 5: Preprocess (manual trigger)
    let preprocessStatus: StepStatus = extractionDone ? 'pending' : 'locked'
    if (manualStatuses[5] === 'running') preprocessStatus = 'running'
    else if (manualStatuses[5] === 'completed') preprocessStatus = 'completed'
    else if (manualStatuses[5] === 'failed') preprocessStatus = 'failed'

    // Step 6: Summaries (manual trigger)
    let summariesStatus: StepStatus = extractionDone ? 'pending' : 'locked'
    if (manualStatuses[6] === 'running') summariesStatus = 'running'
    else if (manualStatuses[6] === 'completed' || hasInsights) summariesStatus = 'completed'
    else if (manualStatuses[6] === 'failed') summariesStatus = 'failed'

    // Step 7: Entities (manual trigger)
    let entitiesStatus: StepStatus = extractionDone ? 'pending' : 'locked'
    if (manualStatuses[7] === 'running') entitiesStatus = 'running'
    else if (manualStatuses[7] === 'completed') entitiesStatus = 'completed'
    else if (manualStatuses[7] === 'failed') entitiesStatus = 'failed'

    // Step 8: Embed (manual trigger)
    let embeddingStatus: StepStatus = extractionDone ? 'pending' : 'locked'
    if (manualStatuses[8] === 'running') embeddingStatus = 'running'
    else if (manualStatuses[8] === 'completed' || hasEmbeddings) embeddingStatus = 'completed'
    else if (manualStatuses[8] === 'failed') embeddingStatus = 'failed'

    // Step 9: Done — complete when extraction is done (manual steps are optional)
    let doneStatus: StepStatus = 'locked'
    if (extractionDone) doneStatus = 'completed'
    else if (jobFailed) doneStatus = 'failed'

    return {
      4: extractionStatus,
      5: preprocessStatus,
      6: summariesStatus,
      7: entitiesStatus,
      8: embeddingStatus,
      9: doneStatus,
    }
  }, [phase, statusData, sourceData, manualStatuses, sourceIds.length, multiSources])

  const pipelineStatuses = derivePipelineStatuses()

  // Auto-advance: stay on step 4 during extraction.
  // Only advance past step 4 when user clicks "Continue to Preprocessing" (single or multi).
  // For steps 5+ (manual triggers), auto-advance to the running step.
  useEffect(() => {
    if (phase !== 'processing') return

    // If we're on step 4 (extraction), don't auto-advance — user will click Continue
    if (activeTab === 4) return

    // For steps 5+, find the running step and advance to it
    const statusOrder = [5, 6, 7, 8, 9] as const
    let targetTab = activeTab

    for (const stepId of statusOrder) {
      const s = pipelineStatuses[stepId]
      if (s === 'running') {
        targetTab = stepId
        break
      }
    }

    // Auto-advance to completion tab — only when the user is already on Done (tab 9).
    // Never auto-skip past manual tabs 5-8; the user navigates away from those explicitly.
    if (pipelineStatuses[9] === 'completed' && phase === 'processing' && activeTab === 9) {
      setPhase('complete')
      return
    }

    if (pipelineStatuses[9] === 'failed') {
      setPhase('error')
    }

    // Only auto-advance forward, not backward
    if (targetTab > activeTab && targetTab > lastAutoAdvancedRef.current) {
      lastAutoAdvancedRef.current = targetTab
      setActiveTab(targetTab)
    }
  }, [phase, pipelineStatuses, activeTab])

  // Build step objects for stepper
  const steps: PipelineStep[] = STEP_LABELS.map((s) => {
    let status: StepStatus = 'pending'
    if (s.phase === 'config') {
      if (s.id < activeTab) status = 'completed'
      else if (s.id === activeTab) status = 'active'
      else status = 'pending'

      // Once in processing, all config steps are completed
      if (phase !== 'config') status = 'completed'
    } else {
      status = pipelineStatuses[s.id] || 'locked'
      if (s.id === activeTab && status === 'locked') status = 'pending'
    }
    return { ...s, status }
  })

  // Step click handler
  const handleStepClick = (stepId: number) => {
    const step = steps.find(s => s.id === stepId)
    if (!step) return

    // Config tabs are always navigable
    if (step.phase === 'config' && phase === 'config') {
      setActiveTab(stepId)
      return
    }

    // Pipeline tabs only when not locked
    if (step.status !== 'locked') {
      setActiveTab(stepId)
    }
  }

  // Navigation
  const handleNext = () => {
    if (activeTab < 3 && isStepValid(activeTab)) {
      setActiveTab(activeTab + 1)
    }
  }

  const handleBack = () => {
    if (activeTab > 1) {
      setActiveTab(activeTab - 1)
    }
  }

  // Notebook toggle
  const handleNotebookToggle = (notebookId: string) => {
    setSelectedNotebooks(prev =>
      prev.includes(notebookId) ? prev.filter(id => id !== notebookId) : [...prev, notebookId]
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
        ? processingOverrides as Partial<SettingsResponse>
        : undefined,
    }

    if (file) {
      (createRequest as CreateSourceRequest & { file?: File }).file = file
    }

    return createRequest
  }

  // Submit multiple files sequentially
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

    // Record what was enabled for status derivation
    setEmbedEnabled(!!data.embed || settings?.default_embedding_option === 'always')
    setSummariesEnabled(false)

    setPhase('processing')
    setActiveTab(4)
    lastAutoAdvancedRef.current = 4

    const ids: string[] = []

    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i]
      // Update status to creating
      setMultiSources(prev => prev.map((e, idx) =>
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

        setMultiSources(prev => prev.map((e, idx) =>
          idx === i ? { ...e, id: result.id, status: 'processing' as const, title: result.title || file.name } : e
        ))
      } catch (error) {
        console.error(`Error creating source for ${file.name}:`, error)
        setMultiSources(prev => prev.map((e, idx) =>
          idx === i ? { ...e, status: 'failed' as const } : e
        ))
      }
    }

    setSourceIds(ids)
    // Don't auto-complete — polling effect will detect actual completion
  }

  // Form submission
  const onSubmit = async (data: CreateSourceFormData) => {
    // Detect multi-file upload
    const isMultiFile = data.type === 'upload' && data.file instanceof FileList && data.file.length > 1

    if (isMultiFile && batchMode === null) {
      // Store form data and show batch dialog
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

    // Single file / link / text submission (original flow)
    try {
      // Record what was enabled for status derivation
      setEmbedEnabled(!!data.embed || settings?.default_embedding_option === 'always')
      setSummariesEnabled(false)

      const file = data.type === 'upload' && data.file
        ? (data.file instanceof FileList ? data.file[0] : data.file)
        : undefined

      const createRequest = buildCreateRequest(data, file instanceof File ? file : undefined)

      const result = await createSource.mutateAsync(createRequest)
      setSourceId(result.id)
      setSourceIds([result.id])
      setPhase('processing')
      setActiveTab(4)
      lastAutoAdvancedRef.current = 4
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
    setActiveTab(1)
    setSourceId(undefined)
    setSourceIds([])
    setMultiSources([])
    setBatchMode(null)
    setShowBatchDialog(false)
    pendingFormDataRef.current = null
    setSelectedNotebooks(defaultNotebookId ? [defaultNotebookId] : [])
    setProcessingOverrides({})
    setEmbedEnabled(false)
    setSummariesEnabled(false)
    setManualStatuses({ 5: 'pending', 6: 'pending', 7: 'pending', 8: 'pending' })
    setClassificationReady(false)
    lastAutoAdvancedRef.current = 0
  }

  // Manual step trigger handlers
  const handleStartEntities = async () => {
    if (!sourceId) return
    setManualStatuses(prev => ({ ...prev, 7: 'running' }))
    try {
      await sourcesApi.runEntities(sourceId)
    } catch {
      setManualStatuses(prev => ({ ...prev, 7: 'failed' }))
    }
  }

  const handleStartEmbed = async () => {
    if (!sourceId) return
    setManualStatuses(prev => ({ ...prev, 8: 'running' }))
    try {
      await sourcesApi.runEmbed(sourceId)
    } catch {
      setManualStatuses(prev => ({ ...prev, 8: 'failed' }))
    }
  }

  // Auto-detect manual step completion from polled data
  useEffect(() => {
    if (!sourceData) return
    const hasInsights = (sourceData.insights_count || 0) > 0
    const hasEntities = (sourceData.entity_count || 0) > 0
    const hasEmbeddings = (sourceData.embedded_chunks || 0) > 0

    setManualStatuses(prev => {
      const next = { ...prev }
      if (hasInsights && prev[6] === 'running') next[6] = 'completed'
      if (hasEntities && prev[7] === 'running') next[7] = 'completed'
      if (hasEmbeddings && prev[8] === 'running') next[8] = 'completed'
      return next
    })
  }, [sourceData])

  // Handle "Continue to Preprocessing" from ExtractionTab
  const handleExtractionContinue = useCallback(() => {
    setActiveTab(5)
    lastAutoAdvancedRef.current = 5
  }, [])

  const handlePostprocessContinue = useCallback(() => {
    setActiveTab(6)
    lastAutoAdvancedRef.current = 6
  }, [])

  // Determine source type label for completion
  const getSourceTypeLabel = () => {
    if (selectedType === 'link') return 'Link'
    if (selectedType === 'upload') return 'Upload'
    if (selectedType === 'text') return 'Text'
    return 'Unknown'
  }

  // Compute file count for batch dialog
  const batchFileCount = watchedFile instanceof FileList ? watchedFile.length : 0

  return (
    <div className="flex flex-col h-full">
      <PipelineHeader />
      <PipelineStepper
        steps={steps}
        activeTab={activeTab}
        onStepClick={handleStepClick}
      />

      <div className="flex-1 overflow-hidden flex flex-col">
        <div className={activeTab === 4 || activeTab === 5 || activeTab === 6 || activeTab === 7
          ? "flex-1 flex flex-col min-h-0 px-6 py-6"
          : "flex-1 overflow-y-auto max-w-3xl mx-auto w-full px-6 py-6"
        }>
          <form onSubmit={handleSubmit(onSubmit)} className={activeTab === 4 || activeTab === 5 || activeTab === 6 || activeTab === 7 ? "flex-1 flex flex-col min-h-0" : undefined}>
            {/* Config tabs */}
            {activeTab === 1 && (
              <SourceTypeStep
                // @ts-expect-error - Type inference issue with zod schema
                control={control}
                register={register}
                // @ts-expect-error - Type inference issue with zod schema
                errors={errors}
              />
            )}

            {activeTab === 2 && (
              <NotebooksStep
                notebooks={notebooks}
                selectedNotebooks={selectedNotebooks}
                onToggleNotebook={handleNotebookToggle}
                loading={notebooksLoading}
              />
            )}

            {activeTab === 3 && (
              <ProcessingConfigStep
                settings={settings}
                overrides={processingOverrides}
                onOverridesChange={setProcessingOverrides}
              />
            )}

            {/* Pipeline tabs: Extract → Preprocess → Summaries → Entities → Embed → Done */}
            {activeTab === 4 && (
              <ExtractionTab
                overallStatus={pipelineStatuses[4]}
                onContinue={handleExtractionContinue}
                files={multiSources.length > 1
                  ? multiSources.map(s => ({
                      name: s.fileName || s.title,
                      sourceId: s.id || undefined,
                      status: s.status,
                    }))
                  : [{
                      name: selectedType === 'upload'
                        ? (watchedFile instanceof FileList ? watchedFile[0]?.name : (watchedFile as File)?.name) || 'File'
                        : selectedType === 'link'
                          ? watchedUrl || 'URL'
                          : watchedTitle || 'Text',
                      sourceId: sourceId,
                      status: pipelineStatuses[4] === 'completed' ? 'completed'
                        : pipelineStatuses[4] === 'failed' ? 'failed'
                        : pipelineStatuses[4] === 'running' ? 'processing'
                        : 'pending',
                    }]
                }
              />
            )}

            {activeTab === 5 && (
              <PreprocessingTab
                status={pipelineStatuses[5]}
                extractionComplete={pipelineStatuses[4] === 'completed'}
                sourceId={sourceId}
                files={multiSources.length > 1
                  ? multiSources.map(s => ({
                      name: s.fileName || s.title,
                      sourceId: s.id || undefined,
                      status: s.status,
                    }))
                  : [{
                      name: selectedType === 'upload'
                        ? (watchedFile instanceof FileList ? watchedFile[0]?.name : (watchedFile as File)?.name) || 'File'
                        : selectedType === 'link'
                          ? watchedUrl || 'URL'
                          : watchedTitle || 'Text',
                      sourceId: sourceId,
                      status: pipelineStatuses[4] === 'completed' ? 'completed'
                        : pipelineStatuses[4] === 'failed' ? 'failed'
                        : pipelineStatuses[4] === 'running' ? 'processing'
                        : 'pending',
                    }]
                }
                onContinue={handlePostprocessContinue}
              />
            )}

            {activeTab === 6 && (
              <SummariesTab
                status={pipelineStatuses[6]}
                extractionComplete={pipelineStatuses[4] === 'completed'}
                sourceId={sourceId}
                files={multiSources.length > 1
                  ? multiSources.map(s => ({
                      name: s.fileName || s.title,
                      sourceId: s.id || undefined,
                      status: s.status,
                    }))
                  : [{
                      name: selectedType === 'upload'
                        ? (watchedFile instanceof FileList ? watchedFile[0]?.name : (watchedFile as File)?.name) || 'File'
                        : selectedType === 'link'
                          ? watchedUrl || 'URL'
                          : watchedTitle || 'Text',
                      sourceId: sourceId,
                      status: pipelineStatuses[4] === 'completed' ? 'completed'
                        : pipelineStatuses[4] === 'failed' ? 'failed'
                        : pipelineStatuses[4] === 'running' ? 'processing'
                        : 'pending',
                    }]
                }
                onClassificationReady={setClassificationReady}
              />
            )}

            {activeTab === 7 && (
              <EntitiesTab
                status={pipelineStatuses[7]}
                extractionComplete={pipelineStatuses[4] === 'completed'}
                sourceId={sourceId}
                files={multiSources.length > 1
                  ? multiSources.map(s => ({
                      name: s.fileName || s.title,
                      sourceId: s.id || undefined,
                      status: s.status,
                    }))
                  : [{
                      name: selectedType === 'upload'
                        ? (watchedFile instanceof FileList ? watchedFile[0]?.name : (watchedFile as File)?.name) || 'File'
                        : selectedType === 'link'
                          ? watchedUrl || 'URL'
                          : watchedTitle || 'Text',
                      sourceId: sourceId,
                      status: pipelineStatuses[4] === 'completed' ? 'completed'
                        : pipelineStatuses[4] === 'failed' ? 'failed'
                        : pipelineStatuses[4] === 'running' ? 'processing'
                        : 'pending',
                    }]
                }
                onStart={handleStartEntities}
                classificationReady={classificationReady}
              />
            )}

            {activeTab === 8 && (
              <EmbeddingTab
                status={pipelineStatuses[8]}
                embeddedChunks={sourceData?.embedded_chunks}
                errorMessage={statusData?.message}
                extractionComplete={pipelineStatuses[4] === 'completed'}
                onStart={handleStartEmbed}
              />
            )}

            {activeTab === 9 && (
              multiSources.length > 1 ? (
                <CompletionTab
                  sourceTitle={`${multiSources.length} sources`}
                  sourceType={getSourceTypeLabel()}
                  embedEnabled={embedEnabled}
                  summariesEnabled={summariesEnabled}
                  sources={multiSources.map(s => ({
                    id: s.id,
                    title: s.title,
                    status: s.status === 'completed' ? 'completed'
                      : s.status === 'failed' ? 'failed'
                      : 'processing',
                  }))}
                />
              ) : (
                <CompletionTab
                  sourceTitle={sourceData?.title || watchedTitle}
                  sourceType={getSourceTypeLabel()}
                  chunkCount={sourceData?.embedded_chunks}
                  entityCount={undefined}
                  relationCount={undefined}
                  embeddedChunks={sourceData?.embedded_chunks}
                  insightsCount={sourceData?.insights_count}
                  embedEnabled={embedEnabled}
                  summariesEnabled={summariesEnabled}
                />
              )
            )}
          </form>
        </div>
      </div>

      <PipelineFooter
        phase={phase}
        activeTab={activeTab}
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
