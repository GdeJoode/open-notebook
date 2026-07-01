'use client'

import { useState, useEffect, useCallback, useMemo } from 'react'
import { isAxiosError } from 'axios'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { sourcesApi } from '@/lib/api/sources'
import { insightsApi, SourceInsightResponse } from '@/lib/api/insights'
import { transformationsApi } from '@/lib/api/transformations'
import { embeddingApi } from '@/lib/api/embedding'
import { SourceDetailResponse } from '@/lib/types/api'
import { Transformation } from '@/lib/types/transformations'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { InlineEdit } from '@/components/common/InlineEdit'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Link as LinkIcon,
  Upload,
  AlignLeft,
  ExternalLink,
  Download,
  Copy,
  CheckCircle,
  Youtube,
  MoreVertical,
  Trash2,
  Sparkles,
  Plus,
  Lightbulb,
  Database,
  AlertCircle,
  MessageSquare,
  Network,
  GitGraph,
  Play,
  ScanSearch,
  BrainCircuit,
  Settings2,
  BookMarked,
  Lock,
} from 'lucide-react'
import { useRouter } from 'next/navigation'
import { formatDistanceToNow } from 'date-fns'
import { toast } from 'sonner'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { SourceInsightDialog } from '@/components/source/SourceInsightDialog'
import { NotebookAssociations } from '@/components/source/NotebookAssociations'
import { PdfChunkViewer } from '@/components/source/PdfChunkViewer'
import { ParserEngineBadge } from '@/components/source/ParserEngineBadge'
import { SourceEntitiesTab } from '@/components/source/SourceEntitiesTab'
import { SourceContradictions } from '@/components/source/SourceContradictions'
import { RelatedSources } from '@/components/source/RelatedSources'
import { PipelineConfigPanel } from '@/components/sources/pipeline/PipelineConfigPanel'
import { PipelineStatus } from '@/components/sources/pipeline'
import { DEFAULT_PIPELINE_CONFIG, type DoclingPipelineConfig } from '@/lib/api/sources'
import { useSourceChunks, useSourcePipeline } from '@/lib/hooks/use-sources'
import { isPollableStage } from '@/lib/pipeline/processing-stage'
import { toPipelineCounts } from '@/lib/pipeline/source-counts'
import type { PipelineDeepLinkTab, PipelineNodeAction } from '@/lib/pipeline/pipeline-stages'
import { useZoteroStatus, useZoteroPush } from '@/lib/hooks/use-zotero'

interface SourceDetailContentProps {
  sourceId: string
  showChatButton?: boolean
  onChatClick?: () => void
  onClose?: () => void
}

export function SourceDetailContent({
  sourceId,
  showChatButton = false,
  onChatClick,
  onClose
}: SourceDetailContentProps) {
  const router = useRouter()
  const [source, setSource] = useState<SourceDetailResponse | null>(null)
  const [insights, setInsights] = useState<SourceInsightResponse[]>([])
  const [transformations, setTransformations] = useState<Transformation[]>([])
  const [selectedTransformation, setSelectedTransformation] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const [loadingInsights, setLoadingInsights] = useState(false)
  const [creatingInsight, setCreatingInsight] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const [isEmbedding, setIsEmbedding] = useState(false)
  const [isRunningPipeline, setIsRunningPipeline] = useState<string | null>(null)
  const [isDownloadingFile, setIsDownloadingFile] = useState(false)
  const [fileAvailable, setFileAvailable] = useState<boolean | null>(null)
  const [selectedInsight, setSelectedInsight] = useState<SourceInsightResponse | null>(null)
  const [showReprocessDialog, setShowReprocessDialog] = useState(false)
  const [reprocessConfig, setReprocessConfig] = useState<DoclingPipelineConfig>({ ...DEFAULT_PIPELINE_CONFIG })
  const [isReprocessing, setIsReprocessing] = useState(false)
  // Controlled tab state so the pipeline spine can deep-link into a tab.
  const [activeTab, setActiveTab] = useState('content')
  // Fetch chunks data
  const { data: chunksData, isLoading: chunksLoading } = useSourceChunks(sourceId)

  // Track UX.4: poll the pipeline stage for in-flight sources so the detail
  // spine advances live. Bounded — `useSourcePipeline` stops refetching at the
  // terminal (complete/failed) or gated (awaiting_schema_review) stages, and we
  // only enable it while the loaded source is still pollable.
  const pipelineEnabled = source != null && isPollableStage(source.processing_stage)
  const { data: pipelineData } = useSourcePipeline(sourceId, pipelineEnabled)

  // Zotero integration
  const { data: zoteroStatus } = useZoteroStatus()
  const zoteroPush = useZoteroPush()

  const fetchSource = useCallback(async () => {
    try {
      setLoading(true)
      const data = await sourcesApi.get(sourceId)
      setSource(data)
      if (typeof data.file_available === 'boolean') {
        setFileAvailable(data.file_available)
      } else if (!data.asset?.file_path) {
        setFileAvailable(null)
      } else {
        setFileAvailable(null)
      }
    } catch (err) {
      console.error('Failed to fetch source:', err)
      setError('Failed to load source details')
    } finally {
      setLoading(false)
    }
  }, [sourceId])

  const fetchInsights = useCallback(async () => {
    try {
      setLoadingInsights(true)
      const data = await insightsApi.listForSource(sourceId)
      setInsights(data)
    } catch (err) {
      console.error('Failed to fetch insights:', err)
    } finally {
      setLoadingInsights(false)
    }
  }, [sourceId])

  const fetchTransformations = useCallback(async () => {
    try {
      const data = await transformationsApi.list()
      setTransformations(data)
    } catch (err) {
      console.error('Failed to fetch transformations:', err)
    }
  }, [])

  useEffect(() => {
    if (sourceId) {
      void fetchSource()
      void fetchInsights()
      void fetchTransformations()
    }
  }, [fetchInsights, fetchSource, fetchTransformations, sourceId])

  const createInsight = async () => {
    if (!selectedTransformation) {
      toast.error('Please select a transformation')
      return
    }

    try {
      setCreatingInsight(true)
      await insightsApi.create(sourceId, {
        transformation_id: selectedTransformation
      })
      toast.success('Insight created successfully')
      await fetchInsights()
      setSelectedTransformation('')
    } catch (err) {
      console.error('Failed to create insight:', err)
      toast.error('Failed to create insight')
    } finally {
      setCreatingInsight(false)
    }
  }

  const handleUpdateTitle = async (title: string) => {
    if (!source || title === source.title) return

    try {
      await sourcesApi.update(sourceId, { title })
      toast.success('Source title updated')
      setSource({ ...source, title })
    } catch (err) {
      console.error('Failed to update source title:', err)
      toast.error('Failed to update source title')
      await fetchSource()
    }
  }

  const handleEmbedContent = async () => {
    if (!source) return

    try {
      setIsEmbedding(true)
      const response = await embeddingApi.embedContent(sourceId, 'source')
      toast.success(response.message)
      await fetchSource()
    } catch (err) {
      console.error('Failed to embed content:', err)
      toast.error('Failed to embed content')
    } finally {
      setIsEmbedding(false)
    }
  }

  const handleRunSummaries = async () => {
    if (!source) return
    try {
      setIsRunningPipeline('summaries')
      await sourcesApi.runSummaries(sourceId)
      toast.success('Summaries generation started')
      await fetchSource()
      fetchInsights()
    } catch (err) {
      console.error('Failed to run summaries:', err)
      toast.error('Failed to start summaries')
    } finally {
      setIsRunningPipeline(null)
    }
  }

  const handleRunEntities = async () => {
    if (!source) return
    try {
      setIsRunningPipeline('entities')
      await sourcesApi.runEntities(sourceId)
      toast.success('Entity extraction started')
      await fetchSource()
    } catch (err) {
      console.error('Failed to run entity extraction:', err)
      toast.error('Failed to start entity extraction')
    } finally {
      setIsRunningPipeline(null)
    }
  }

  const handleReprocess = async () => {
    if (!source) return
    try {
      setIsReprocessing(true)
      await sourcesApi.reprocess(sourceId, reprocessConfig)
      toast.success('Reprocessing started — document will be re-ingested')
      setShowReprocessDialog(false)
      await fetchSource()
    } catch (err) {
      console.error('Failed to reprocess:', err)
      toast.error('Failed to start reprocessing')
    } finally {
      setIsReprocessing(false)
    }
  }

  const handleRetry = async () => {
    if (!source) return
    try {
      await sourcesApi.retry(sourceId)
      toast.success('Source requeued for processing')
      await fetchSource()
    } catch (err) {
      console.error('Failed to retry source:', err)
      toast.error('Failed to retry processing')
    }
  }

  const handlePipelineNodeClick = (tab: PipelineDeepLinkTab | undefined) => {
    // Ingest/Embed → Chunks, Extract/Graph → Entities, Insights → Insights.
    // Complete has no deep-link (undefined) and is a no-op.
    if (tab) setActiveTab(tab)
  }

  const handlePipelineNodeAction = (action: PipelineNodeAction) => {
    // Gated ⇒ send the user to the Entities tab (schema/entity review lives
    // there); Failed ⇒ requeue via the existing retry path.
    if (action === 'review-schema') {
      setActiveTab('entities')
    } else if (action === 'retry') {
      void handleRetry()
    }
  }

  const handleRunPreprocessing = async () => {
    if (!source) return
    try {
      setIsRunningPipeline('preprocessing')
      await sourcesApi.runPreprocessing(sourceId, reprocessConfig.privacy)
      toast.success('Preprocessing completed')
      await fetchSource()
    } catch (err) {
      console.error('Failed to run preprocessing:', err)
      toast.error('Failed to run preprocessing')
    } finally {
      setIsRunningPipeline(null)
    }
  }

  const extractFilename = (pathOrUrl: string | undefined, fallback: string) => {
    if (!pathOrUrl) {
      return fallback
    }
    const segments = pathOrUrl.split(/[/\\]/)
    return segments.pop() || fallback
  }

  const parseContentDisposition = (header?: string | null) => {
    if (!header) {
      return null
    }
    const match = header.match(/filename\*?=([^;]+)/i)
    if (!match) {
      return null
    }
    const value = match[1].trim()
    if (value.toLowerCase().startsWith("utf-8''")) {
      return decodeURIComponent(value.slice(7))
    }
    return value.replace(/^["']|["']$/g, '')
  }

  const handleDownloadFile = async () => {
    if (!source?.asset?.file_path || isDownloadingFile || fileAvailable === false) {
      return
    }

    try {
      setIsDownloadingFile(true)
      const response = await sourcesApi.downloadFile(source.id)
      const filenameFromHeader = parseContentDisposition(
        response.headers?.['content-disposition'] as string | undefined
      )
      const fallbackName = extractFilename(source.asset.file_path, `source-${source.id}`)
      const filename = filenameFromHeader || fallbackName

      const blobUrl = window.URL.createObjectURL(response.data)
      const link = document.createElement('a')
      link.href = blobUrl
      link.download = filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(blobUrl)
      setFileAvailable(true)
      toast.success('Download started')
    } catch (err) {
      console.error('Failed to download file:', err)
      if (isAxiosError(err) && err.response?.status === 404) {
        setFileAvailable(false)
        toast.error('Original file is no longer available on the server')
      } else {
        toast.error('Failed to download file')
      }
    } finally {
      setIsDownloadingFile(false)
    }
  }

  const getSourceIcon = () => {
    if (!source) return null
    if (source.asset?.url) return <LinkIcon className="h-5 w-5" />
    if (source.asset?.file_path) return <Upload className="h-5 w-5" />
    return <AlignLeft className="h-5 w-5" />
  }

  const getSourceType = () => {
    if (!source) return 'unknown'
    if (source.asset?.url) return 'link'
    if (source.asset?.file_path) return 'file'
    return 'text'
  }

  const handleCopyUrl = useCallback(() => {
    if (source?.asset?.url) {
      navigator.clipboard.writeText(source.asset.url)
      setCopied(true)
      toast.success('URL copied to clipboard')
      setTimeout(() => setCopied(false), 2000)
    }
  }, [source])

  const handleOpenExternal = useCallback(() => {
    if (source?.asset?.url) {
      window.open(source.asset.url, '_blank')
    }
  }, [source])

  const getYouTubeVideoId = (url: string): string | null => {
    const patterns = [
      /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)/,
      /youtube\.com\/watch\?.*v=([^&\n?#]+)/
    ]

    for (const pattern of patterns) {
      const match = url.match(pattern)
      if (match) return match[1]
    }
    return null
  }

  const isYouTubeUrl = useMemo(() => {
    if (!source?.asset?.url) return false
    return !!(getYouTubeVideoId(source.asset.url))
  }, [source?.asset?.url])

  const youTubeVideoId = useMemo(() => {
    if (!source?.asset?.url) return null
    return getYouTubeVideoId(source.asset.url)
  }, [source?.asset?.url])

  const handleDelete = async () => {
    if (!source) return

    if (confirm('Are you sure you want to delete this source?')) {
      try {
        await sourcesApi.delete(source.id)
        toast.success('Source deleted successfully')
        onClose?.()
      } catch (error) {
        console.error('Failed to delete source:', error)
        toast.error('Failed to delete source')
      }
    }
  }

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center p-8">
        <LoadingSpinner />
      </div>
    )
  }

  if (error || !source) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 p-8">
        <p className="text-red-500">{error || 'Source not found'}</p>
      </div>
    )
  }

  // Prefer the live-polled payload for the spine so an in-flight source advances
  // without a manual reload; fall back to the initially-loaded source. The Graph
  // node resolves from `processing_stage` (done at graphed/complete) and its
  // "linked" badge is keyed off that stage via the shared counts adapter.
  const effectiveSource = pipelineData ?? source
  const effectiveStage = effectiveSource.processing_stage
  const jobStatus = (effectiveSource as { status?: string }).status

  // Track UX.6: the manual "Run X" runners are now RECOVERY-only. Each stage
  // runs automatically in the backend auto-chain, so a runner is enabled only
  // when the source is on a recovery stage (`failed` / `awaiting_schema_review`)
  // OR its own output is genuinely missing; otherwise it renders disabled with a
  // "Runs automatically" hint. Reprocess (below) stays always available — it is
  // the parser-config home from UX.5, not a recovery action.
  const isRecoveryStage =
    effectiveStage === 'failed' || effectiveStage === 'awaiting_schema_review'
  const hasText = !!source.full_text
  const recovery = {
    preprocessing: hasText && isRecoveryStage,
    summaries: hasText && (isRecoveryStage || insights.length === 0),
    entities: hasText && (isRecoveryStage || (source.entity_count ?? 0) === 0),
    embed: isRecoveryStage || !source.embedded,
  }

  const renderRecoveryRunner = (opts: {
    runningKey: string
    enabled: boolean
    label: string
    runningLabel: string
    icon: React.ReactNode
    onClick: () => void
    /** Embed tracks its own `isEmbedding` flag rather than `isRunningPipeline`. */
    running?: boolean
  }) => {
    const running = opts.running ?? isRunningPipeline === opts.runningKey
    const disabled = running || !opts.enabled
    // "Runs automatically" applies when we disabled a runner because the stage
    // is on the happy path and the output already exists (not while it is
    // actively running).
    const autoReason = disabled && !running
    return (
      <DropdownMenuItem
        onClick={opts.onClick}
        disabled={disabled}
        title={autoReason ? 'Runs automatically' : undefined}
      >
        {opts.icon}
        <span className="flex-1">{running ? opts.runningLabel : opts.label}</span>
        {autoReason && (
          <span className="ml-2 text-xs text-muted-foreground">
            Runs automatically
          </span>
        )}
      </DropdownMenuItem>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="pb-4 px-2">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <InlineEdit
              value={source.title || ''}
              onSave={handleUpdateTitle}
              className="text-2xl font-bold"
              inputClassName="text-2xl font-bold"
              placeholder="Source title"
              emptyText="Untitled Source"
            />
            <p className="mt-1 text-sm text-muted-foreground">
              Source ID: {source.id}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {getSourceIcon()}
            <Badge variant="secondary" className="text-sm">
              {getSourceType()}
            </Badge>

            {/* Track J.3/J.5: read-only badge for documents processed privately. */}
            {source.private && (
              <Badge
                variant="outline"
                className="text-sm"
                data-testid="private-source-badge"
                aria-label="This source is processed privately"
              >
                <Lock className="h-3.5 w-3.5" aria-hidden="true" />
                Private
              </Badge>
            )}

            {/* Promoted actions */}
            {(source.entity_count ?? 0) > 0 && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => router.push('/knowledge-graph')}
              >
                <GitGraph className="h-4 w-4 mr-2" />
                Knowledge Graph
              </Button>
            )}

            {/* Chat with source button - only in modal */}
            {showChatButton && onChatClick && (
              <Button variant="outline" size="sm" onClick={onChatClick}>
                <MessageSquare className="h-4 w-4 mr-2" />
                Chat with source
              </Button>
            )}

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon">
                  <MoreVertical className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                {source.asset?.file_path && (
                  <>
                    <DropdownMenuItem
                      onClick={handleDownloadFile}
                      disabled={isDownloadingFile || fileAvailable === false}
                    >
                      <Download className="mr-2 h-4 w-4" />
                      {fileAvailable === false
                        ? 'File unavailable'
                        : isDownloadingFile
                          ? 'Preparing download…'
                          : 'Download File'}
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                  </>
                )}
                <DropdownMenuItem
                  onClick={() => setShowReprocessDialog(true)}
                  disabled={!source.asset?.file_path}
                >
                  <Settings2 className="mr-2 h-4 w-4" />
                  Reprocess Document...
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuLabel className="text-xs text-muted-foreground">
                  Re-run / recovery
                </DropdownMenuLabel>
                {renderRecoveryRunner({
                  runningKey: 'preprocessing',
                  enabled: recovery.preprocessing,
                  label: 'Run Preprocessing',
                  runningLabel: 'Preprocessing...',
                  icon: <ScanSearch className="mr-2 h-4 w-4" />,
                  onClick: handleRunPreprocessing,
                })}
                {renderRecoveryRunner({
                  runningKey: 'summaries',
                  enabled: recovery.summaries,
                  label: 'Generate Summaries',
                  runningLabel: 'Generating...',
                  icon: <Sparkles className="mr-2 h-4 w-4" />,
                  onClick: handleRunSummaries,
                })}
                {renderRecoveryRunner({
                  runningKey: 'entities',
                  enabled: recovery.entities,
                  label: 'Extract Entities',
                  runningLabel: 'Extracting...',
                  icon: <BrainCircuit className="mr-2 h-4 w-4" />,
                  onClick: handleRunEntities,
                })}
                {renderRecoveryRunner({
                  runningKey: 'embed',
                  enabled: recovery.embed,
                  label: 'Embed Content',
                  runningLabel: 'Embedding...',
                  icon: <Database className="mr-2 h-4 w-4" />,
                  onClick: handleEmbedContent,
                  running: isEmbedding,
                })}
                {zoteroStatus?.connected && (
                  <DropdownMenuItem
                    onClick={() => zoteroPush.mutate({ sourceId })}
                    disabled={zoteroPush.isPending || !!(source as unknown as Record<string, unknown>).zotero_key}
                  >
                    <BookMarked className="mr-2 h-4 w-4" />
                    {zoteroPush.isPending
                      ? 'Pushing...'
                      : (source as unknown as Record<string, unknown>).zotero_key
                        ? 'In Zotero'
                        : 'Push to Zotero'}
                  </DropdownMenuItem>
                )}
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  className="text-destructive"
                  onClick={handleDelete}
                >
                  <Trash2 className="mr-2 h-4 w-4" />
                  Delete Source
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </div>

      {/* Pipeline spine (Track UX.4) — the collapsed `processing_stage` spine
          replaces the four independent output badges. Fed from the freshest
          available payload (live poll for in-flight sources, else the loaded
          source); clicking a node deep-links into its tab. */}
      <div className="px-2 pb-3 flex flex-col gap-2">
        <PipelineStatus
          variant="detail"
          processingStage={effectiveStage}
          jobStatus={jobStatus}
          counts={toPipelineCounts(effectiveSource)}
          onNodeClick={handlePipelineNodeClick}
          onNodeAction={handlePipelineNodeAction}
        />
        <div className="flex flex-wrap gap-2 text-xs">
          <ParserEngineBadge
            metadata={source.metadata}
            sourceStatus={source.status}
          />
          {!!(source as unknown as Record<string, unknown>).zotero_key && (
            <Badge variant="default" className="gap-1 bg-red-700">
              <BookMarked className="h-3 w-3" /> Zotero
            </Badge>
          )}
        </div>
        {/* Track UX.6 (deferred): contradiction verdicts. Self-gating stub —
            renders nothing until a verdicts read API is wired (see the
            component doc). Safe no-op today. */}
        <SourceContradictions sourceId={sourceId} />
      </div>

      {/* Tabs Content */}
      <div className="flex-1 px-2 min-h-0">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
          <TabsList className="grid w-full grid-cols-6 flex-shrink-0">
            <TabsTrigger value="content">Content</TabsTrigger>
            <TabsTrigger value="insights">
              Insights {insights.length > 0 && `(${insights.length})`}
            </TabsTrigger>
            <TabsTrigger value="chunks">
              Chunks {chunksData?.total_chunks ? `(${chunksData.total_chunks})` : ''}
            </TabsTrigger>
            <TabsTrigger value="entities">
              Entities {(source.entity_count ?? 0) > 0 && `(${source.entity_count})`}
            </TabsTrigger>
            <TabsTrigger value="related">Related</TabsTrigger>
            <TabsTrigger value="details">Details</TabsTrigger>
          </TabsList>

          <TabsContent value="content" className="flex-1 overflow-y-auto mt-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  {isYouTubeUrl && <Youtube className="h-5 w-5" />}
                  Content
                </CardTitle>
                {source.asset?.url && !isYouTubeUrl && (
                  <CardDescription className="flex items-center gap-2">
                    <LinkIcon className="h-4 w-4" />
                    <a
                      href={source.asset.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:underline text-blue-600"
                    >
                      {source.asset.url}
                    </a>
                  </CardDescription>
                )}
              </CardHeader>
              <CardContent>
                {isYouTubeUrl && youTubeVideoId && (
                  <div className="mb-6">
                    <div className="aspect-video rounded-lg overflow-hidden bg-black">
                      <iframe
                        src={`https://www.youtube.com/embed/${youTubeVideoId}`}
                        title="YouTube video"
                        className="w-full h-full"
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                        allowFullScreen
                      />
                    </div>
                    {source.asset?.url && (
                      <div className="mt-2">
                        <a
                          href={source.asset.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-sm text-muted-foreground hover:underline inline-flex items-center gap-1"
                        >
                          <ExternalLink className="h-3 w-3" />
                          Open on YouTube
                        </a>
                      </div>
                    )}
                  </div>
                )}
                <div className="prose prose-sm prose-neutral dark:prose-invert max-w-none prose-headings:font-semibold prose-a:text-blue-600 prose-code:bg-muted prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-p:mb-4 prose-p:leading-7 prose-li:mb-2">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      p: ({ children }) => <p className="mb-4">{children}</p>,
                      h1: ({ children }) => <h1 className="text-2xl font-bold mt-6 mb-4">{children}</h1>,
                      h2: ({ children }) => <h2 className="text-xl font-bold mt-5 mb-3">{children}</h2>,
                      h3: ({ children }) => <h3 className="text-lg font-semibold mt-4 mb-2">{children}</h3>,
                      ul: ({ children }) => <ul className="mb-4 list-disc pl-6">{children}</ul>,
                      ol: ({ children }) => <ol className="mb-4 list-decimal pl-6">{children}</ol>,
                      li: ({ children }) => <li className="mb-1">{children}</li>,
                      table: ({ children }) => (
                        <div className="overflow-x-auto my-4">
                          <table className="min-w-full border-collapse border border-border">{children}</table>
                        </div>
                      ),
                      thead: ({ children }) => <thead className="bg-muted">{children}</thead>,
                      th: ({ children }) => <th className="border border-border px-3 py-2 text-left font-semibold">{children}</th>,
                      td: ({ children }) => <td className="border border-border px-3 py-2">{children}</td>,
                      tr: ({ children }) => <tr className="even:bg-muted/50">{children}</tr>,
                    }}
                  >
                    {source.full_text || 'No content available'}
                  </ReactMarkdown>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="insights" className="flex-1 overflow-y-auto mt-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    <Lightbulb className="h-5 w-5" />
                    Insights
                  </span>
                  <Badge variant="secondary">{insights.length}</Badge>
                </CardTitle>
                <CardDescription>
                  AI-generated insights about this source
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Create New Insight */}
                <div className="rounded-lg border bg-muted/30 p-4">
                  <h3 className="mb-3 text-sm font-semibold flex items-center gap-2">
                    <Sparkles className="h-4 w-4" />
                    Generate New Insight
                  </h3>
                  <div className="flex gap-2">
                    <Select
                      value={selectedTransformation}
                      onValueChange={setSelectedTransformation}
                      disabled={creatingInsight}
                    >
                      <SelectTrigger className="flex-1">
                        <SelectValue placeholder="Select a transformation..." />
                      </SelectTrigger>
                      <SelectContent>
                        {transformations.map((trans) => (
                          <SelectItem key={trans.id} value={trans.id}>
                            {trans.title || trans.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Button
                      size="sm"
                      onClick={createInsight}
                      disabled={!selectedTransformation || creatingInsight}
                    >
                      {creatingInsight ? (
                        <>
                          <LoadingSpinner className="mr-2 h-3 w-3" />
                          Creating...
                        </>
                      ) : (
                        <>
                          <Plus className="mr-2 h-4 w-4" />
                          Create
                        </>
                      )}
                    </Button>
                  </div>
                </div>

                {/* Insights List */}
                {loadingInsights ? (
                  <div className="flex items-center justify-center py-8">
                    <LoadingSpinner />
                  </div>
                ) : insights.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    <Lightbulb className="h-12 w-12 mx-auto mb-3 opacity-50" />
                    <p className="text-sm">No insights yet</p>
                    <p className="text-xs mt-1">Create your first insight using a transformation above</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {insights.map((insight) => (
                      <div key={insight.id} className="rounded-lg border bg-background p-4">
                        <div className="flex items-start justify-between">
                          <div className="flex items-center gap-2">
                            <Badge variant="outline" className="text-xs uppercase">
                              {insight.insight_type}
                            </Badge>
                          </div>
                          {insight.created && (
                            <span className="text-xs text-muted-foreground">
                              {(() => {
                                try {
                                  const date = new Date(insight.created)
                                  if (isNaN(date.getTime())) {
                                    return 'Unknown date'
                                  }
                                  return formatDistanceToNow(date, { addSuffix: true })
                                } catch {
                                  return 'Unknown date'
                                }
                              })()}
                            </span>
                          )}
                        </div>
                        <p className="mt-2 text-sm text-muted-foreground">
                          {insight.content.slice(0, 180)}{insight.content.length > 180 ? '…' : ''}
                        </p>
                        <div className="mt-3 flex justify-end">
                          <Button size="sm" variant="outline" onClick={() => setSelectedInsight(insight)}>
                            View Insight
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="chunks" className="flex-1 min-h-0 overflow-hidden mt-6 flex flex-col">
            {chunksData && chunksData.chunks.length > 0 && (
              <div className="flex-shrink-0 flex justify-end pb-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => router.push(`/sources/${encodeURIComponent(sourceId)}/inspect`)}
                >
                  <ScanSearch className="mr-2 h-4 w-4" />
                  Open Inspect
                </Button>
              </div>
            )}
            <div className="flex-1 min-h-0">
              {chunksLoading ? (
                <div className="flex items-center justify-center py-12">
                  <LoadingSpinner />
                </div>
              ) : chunksData && chunksData.chunks.length > 0 ? (
                <div className="h-full">
                  <PdfChunkViewer
                    sourceId={sourceId}
                    chunks={chunksData.chunks}
                  />
                </div>
              ) : (
                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>No Chunks Available</AlertTitle>
                  <AlertDescription>
                    This source doesn&apos;t have chunk data. Chunks are automatically extracted when documents are processed with Docling.
                    {source?.asset?.file_path?.endsWith('.pdf') && (
                      <span> Try reprocessing this PDF with Docling to enable chunk visualization.</span>
                    )}
                  </AlertDescription>
                </Alert>
              )}
            </div>
          </TabsContent>

          <TabsContent value="entities" className="flex-1 overflow-y-auto mt-6">
            <SourceEntitiesTab sourceId={sourceId} />
          </TabsContent>

          <TabsContent value="related" className="flex-1 overflow-y-auto mt-6">
            <RelatedSources sourceId={sourceId} />
          </TabsContent>

          <TabsContent value="details" className="flex-1 overflow-y-auto mt-6">
            <Card>
              <CardHeader>
                <CardTitle>Details</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Embedding Alert */}
                {!source.embedded && (
                  <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>
                      Content Not Embedded
                    </AlertTitle>
                    <AlertDescription>
                      This content hasn&apos;t been embedded for vector search. Embedding enables advanced search capabilities and better content discovery.
                      <div className="mt-3">
                        <Button
                          onClick={handleEmbedContent}
                          disabled={isEmbedding}
                          size="sm"
                        >
                          <Database className="mr-2 h-4 w-4" />
                          {isEmbedding ? 'Embedding...' : 'Embed Content'}
                        </Button>
                      </div>
                    </AlertDescription>
                  </Alert>
                )}

                {/* Source Information */}
                <div className="space-y-4">
                  {source.asset?.url && (
                    <div>
                      <h3 className="mb-2 text-sm font-semibold">URL</h3>
                      <div className="flex items-center gap-2">
                        <code className="flex-1 rounded bg-muted px-2 py-1 text-sm">
                          {source.asset.url}
                        </code>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={handleCopyUrl}
                        >
                          {copied ? (
                            <CheckCircle className="h-4 w-4" />
                          ) : (
                            <Copy className="h-4 w-4" />
                          )}
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={handleOpenExternal}
                        >
                          <ExternalLink className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  )}

                  {source.asset?.file_path && (
                    <div className="space-y-2">
                      <h3 className="text-sm font-semibold">Uploaded File</h3>
                      <div className="flex flex-wrap items-center gap-2">
                        <code className="rounded bg-muted px-2 py-1 text-sm">
                          {source.asset.file_path}
                        </code>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={handleDownloadFile}
                          disabled={isDownloadingFile || fileAvailable === false}
                        >
                          <Download className="mr-2 h-4 w-4" />
                          {fileAvailable === false
                            ? 'Unavailable'
                            : isDownloadingFile
                              ? 'Preparing…'
                              : 'Download'}
                        </Button>
                      </div>
                      {fileAvailable === false ? (
                        <p className="text-xs text-muted-foreground">
                          Original file is no longer available on the server (likely removed after
                          processing). Upload it again if you need a fresh copy.
                        </p>
                      ) : null}
                    </div>
                  )}

                  {source.topics && source.topics.length > 0 && (
                    <div>
                      <h3 className="mb-2 text-sm font-semibold">Topics</h3>
                      <div className="flex flex-wrap gap-2">
                        {source.topics.map((topic, idx) => (
                          <Badge key={idx} variant="outline">
                            {topic}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Metadata */}
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold">Metadata</h3>
                    <div className="flex items-center gap-2">
                      <Database className="h-3.5 w-3.5 text-muted-foreground" />
                      <Badge variant={source.embedded ? "default" : "secondary"} className="text-xs">
                        {source.embedded ? "Embedded" : "Not Embedded"}
                      </Badge>
                    </div>
                  </div>
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div>
                      <p className="text-xs font-medium text-muted-foreground">Created</p>
                      <p className="text-sm">
                        {formatDistanceToNow(new Date(source.created), { addSuffix: true })}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(source.created).toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs font-medium text-muted-foreground">Updated</p>
                      <p className="text-sm">
                        {formatDistanceToNow(new Date(source.updated), { addSuffix: true })}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(source.updated).toLocaleString()}
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Notebook Associations */}
            <NotebookAssociations
              sourceId={sourceId}
              currentNotebookIds={source.notebooks || []}
              onSave={fetchSource}
            />
          </TabsContent>
        </Tabs>
      </div>

      <SourceInsightDialog
        open={Boolean(selectedInsight)}
        onOpenChange={(open) => {
          if (!open) {
            setSelectedInsight(null)
          }
        }}
        insight={selectedInsight ?? undefined}
      />

      {/* Reprocess Dialog */}
      <Dialog open={showReprocessDialog} onOpenChange={setShowReprocessDialog}>
        <DialogContent className="max-w-md max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Reprocess Document</DialogTitle>
            <DialogDescription>
              The primary place to change parsing / OCR settings. Re-run Docling
              ingestion with different pipeline settings — this replaces the
              existing chunks and extracted content.
            </DialogDescription>
          </DialogHeader>
          <PipelineConfigPanel
            config={reprocessConfig}
            onChange={setReprocessConfig}
            onReprocess={handleReprocess}
            reprocessing={isReprocessing}
            collapsible={false}
          />
        </DialogContent>
      </Dialog>
    </div>
  )
}
