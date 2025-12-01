'use client'

import { useForm, Controller } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { useSettings, useUpdateSettings } from '@/lib/hooks/use-settings'
import { useEffect, useState } from 'react'
import { ChevronDownIcon, SettingsIcon } from 'lucide-react'

const settingsSchema = z.object({
  // Basic Settings
  default_content_processing_engine_doc: z.enum(['auto', 'docling', 'simple']).optional(),
  default_content_processing_engine_url: z.enum(['auto', 'firecrawl', 'jina', 'simple']).optional(),
  default_embedding_option: z.enum(['ask', 'always', 'never']).optional(),
  auto_delete_files: z.enum(['yes', 'no']).optional(),

  // GPU Acceleration Settings (content-core)
  docling_gpu_enabled: z.boolean().optional(),
  docling_gpu_device: z.enum(['auto', 'cuda', 'cpu']).optional(),

  // Pipeline Settings (content-core)
  docling_pipeline: z.enum(['auto', 'standard', 'vlm']).optional(),

  // VLM Settings (content-core)
  docling_vlm_model: z.enum(['granite-docling-258m', 'smoldocling-256m']).optional(),
  docling_vlm_framework: z.enum(['auto', 'transformers', 'mlx']).optional(),

  // OCR Settings (content-core)
  docling_ocr_engine: z.enum(['auto', 'easyocr', 'rapidocr', 'tesseract']).optional(),
  docling_ocr_use_gpu: z.boolean().optional(),

  // Table Processing Settings (content-core)
  docling_table_mode: z.enum(['accurate', 'fast']).optional(),

  // Image Export Settings (content-core) - WIP
  docling_auto_export_images: z.boolean().optional(),
  docling_image_scale: z.number().min(1).max(4).optional(),

  // Chunking Settings (content-core)
  docling_chunking_enabled: z.boolean().optional(),
  docling_chunking_method: z.enum(['hybrid', 'hierarchical']).optional(),
  docling_chunking_max_tokens: z.number().min(128).max(4096).optional(),

  // File Management Settings
  input_directory_path: z.string().optional(),
  markdown_directory_path: z.string().optional(),
  output_directory_path: z.string().optional(),
  file_operation: z.enum(['copy', 'move', 'none']).optional(),
  output_naming_scheme: z.enum(['timestamp_prefix', 'date_prefix', 'datetime_suffix', 'original']).optional(),
})

type SettingsFormData = z.infer<typeof settingsSchema>

export function SettingsForm() {
  const { data: settings, isLoading, error } = useSettings()
  const updateSettings = useUpdateSettings()
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({})
  const [hasResetForm, setHasResetForm] = useState(false)
  
  
  const {
    control,
    handleSubmit,
    reset,
    watch,
    formState: { isDirty }
  } = useForm<SettingsFormData>({
    resolver: zodResolver(settingsSchema),
    defaultValues: {
      default_content_processing_engine_doc: undefined,
      default_content_processing_engine_url: undefined,
      default_embedding_option: undefined,
      auto_delete_files: undefined,
    }
  })

  // Watch values for displaying in sliders
  const imageScale = watch('docling_image_scale') ?? 2.0
  const chunkingMaxTokens = watch('docling_chunking_max_tokens') ?? 512


  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }))
  }

  useEffect(() => {
    if (settings && settings.default_content_processing_engine_doc && !hasResetForm) {
      const formData = {
        // Basic Settings
        default_content_processing_engine_doc: settings.default_content_processing_engine_doc as 'auto' | 'docling' | 'simple',
        default_content_processing_engine_url: settings.default_content_processing_engine_url as 'auto' | 'firecrawl' | 'jina' | 'simple',
        default_embedding_option: settings.default_embedding_option as 'ask' | 'always' | 'never',
        auto_delete_files: settings.auto_delete_files as 'yes' | 'no',

        // GPU Acceleration Settings
        docling_gpu_enabled: settings.docling_gpu_enabled,
        docling_gpu_device: settings.docling_gpu_device as 'auto' | 'cuda' | 'cpu',

        // Pipeline Settings
        docling_pipeline: settings.docling_pipeline as 'auto' | 'standard' | 'vlm',

        // VLM Settings
        docling_vlm_model: settings.docling_vlm_model as 'granite-docling-258m' | 'smoldocling-256m',
        docling_vlm_framework: settings.docling_vlm_framework as 'auto' | 'transformers' | 'mlx',

        // OCR Settings
        docling_ocr_engine: settings.docling_ocr_engine as 'auto' | 'easyocr' | 'rapidocr' | 'tesseract',
        docling_ocr_use_gpu: settings.docling_ocr_use_gpu,

        // Table Processing Settings
        docling_table_mode: settings.docling_table_mode as 'accurate' | 'fast',

        // Image Export Settings
        docling_auto_export_images: settings.docling_auto_export_images,
        docling_image_scale: settings.docling_image_scale,

        // Chunking Settings
        docling_chunking_enabled: settings.docling_chunking_enabled,
        docling_chunking_method: settings.docling_chunking_method as 'hybrid' | 'hierarchical',
        docling_chunking_max_tokens: settings.docling_chunking_max_tokens,

        // File Management Settings
        input_directory_path: settings.input_directory_path,
        markdown_directory_path: settings.markdown_directory_path,
        output_directory_path: settings.output_directory_path,
        file_operation: settings.file_operation as 'copy' | 'move' | 'none',
        output_naming_scheme: settings.output_naming_scheme as 'timestamp_prefix' | 'date_prefix' | 'datetime_suffix' | 'original',
      }
      reset(formData)
      setHasResetForm(true)
    }
  }, [hasResetForm, reset, settings])

  const onSubmit = async (data: SettingsFormData) => {
    await updateSettings.mutateAsync(data)
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Failed to load settings</AlertTitle>
        <AlertDescription>
          {error instanceof Error ? error.message : 'An unexpected error occurred.'}
        </AlertDescription>
      </Alert>
    )
  }

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Content Processing</CardTitle>
          <CardDescription>
            Configure how documents and URLs are processed
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-3">
            <Label htmlFor="doc_engine">Document Processing Engine</Label>
            <Controller
              name="default_content_processing_engine_doc"
              control={control}
              render={({ field }) => (
                <Select
                  key={field.value}
                  value={field.value || ''}
                  onValueChange={field.onChange}
                  disabled={field.disabled || isLoading}
                >
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Select document processing engine" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="auto">Auto (Recommended)</SelectItem>
                      <SelectItem value="docling">Docling</SelectItem>
                      <SelectItem value="simple">Simple</SelectItem>
                    </SelectContent>
                  </Select>
              )}
            />
            <Collapsible open={expandedSections.doc} onOpenChange={() => toggleSection('doc')}>
              <CollapsibleTrigger className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors">
                <ChevronDownIcon className={`h-4 w-4 transition-transform ${expandedSections.doc ? 'rotate-180' : ''}`} />
                Help me choose
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-2 text-sm text-muted-foreground space-y-2">
                <p>• <strong>Docling</strong> provides accurate document processing with support for tables and images. Configure GPU acceleration and processing pipelines in Advanced Settings below.</p>
                <p>• <strong>Simple</strong> will extract content without formatting. OK for simple documents, but loses quality in complex ones.</p>
                <p>• <strong>Auto (recommended)</strong> will try docling and fallback to simple if needed.</p>
              </CollapsibleContent>
            </Collapsible>
          </div>
          
          <div className="space-y-3">
            <Label htmlFor="url_engine">URL Processing Engine</Label>
            <Controller
              name="default_content_processing_engine_url"
              control={control}
              render={({ field }) => (
                <Select
                  key={field.value}
                  value={field.value || ''}
                  onValueChange={field.onChange}
                  disabled={field.disabled || isLoading}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select URL processing engine" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">Auto (Recommended)</SelectItem>
                    <SelectItem value="firecrawl">Firecrawl</SelectItem>
                    <SelectItem value="jina">Jina</SelectItem>
                    <SelectItem value="simple">Simple</SelectItem>
                  </SelectContent>
                </Select>
              )}
            />
            <Collapsible open={expandedSections.url} onOpenChange={() => toggleSection('url')}>
              <CollapsibleTrigger className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors">
                <ChevronDownIcon className={`h-4 w-4 transition-transform ${expandedSections.url ? 'rotate-180' : ''}`} />
                Help me choose
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-2 text-sm text-muted-foreground space-y-2">
                <p>• <strong>Firecrawl</strong> is a paid service (with a free tier), and very powerful.</p>
                <p>• <strong>Jina</strong> is a good option as well and also has a free tier.</p>
                <p>• <strong>Simple</strong> will use basic HTTP extraction and will miss content on javascript-based websites.</p>
                <p>• <strong>Auto (recommended)</strong> will try to use firecrawl (if API Key is present). Then, it will use Jina until reaches the limit (or will keep using Jina if you setup the API Key). It will fallback to simple, when none of the previous options is possible.</p>
              </CollapsibleContent>
            </Collapsible>
          </div>
        </CardContent>
      </Card>

      {/* Advanced Document Processing Settings (Collapsible) */}
      <Card>
        <Collapsible open={expandedSections.advanced} onOpenChange={() => toggleSection('advanced')}>
          <CollapsibleTrigger asChild>
            <CardHeader className="cursor-pointer">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <SettingsIcon className="h-5 w-5" />
                    Advanced Document Processing
                  </CardTitle>
                  <CardDescription>
                    Configure GPU acceleration, processing pipelines, and document extraction options
                  </CardDescription>
                </div>
                <ChevronDownIcon className={`h-5 w-5 transition-transform ${expandedSections.advanced ? 'rotate-180' : ''}`} />
              </div>
            </CardHeader>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <CardContent className="space-y-6 pt-0">
              {/* GPU Acceleration */}
              <div className="space-y-4 p-4 border rounded-lg">
                <h4 className="font-medium">GPU Acceleration</h4>
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Enable GPU Acceleration</Label>
                    <p className="text-sm text-muted-foreground">Use NVIDIA GPU for faster processing (8-14x speedup)</p>
                  </div>
                  <Controller
                    name="docling_gpu_enabled"
                    control={control}
                    render={({ field }) => (
                      <Switch
                        checked={field.value ?? true}
                        onCheckedChange={field.onChange}
                        disabled={isLoading}
                      />
                    )}
                  />
                </div>
                <div className="space-y-2">
                  <Label>GPU Device</Label>
                  <Controller
                    name="docling_gpu_device"
                    control={control}
                    render={({ field }) => (
                      <Select
                        value={field.value || 'auto'}
                        onValueChange={field.onChange}
                        disabled={isLoading}
                      >
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder="Select GPU device" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="auto">Auto (Recommended)</SelectItem>
                          <SelectItem value="cuda">CUDA (NVIDIA GPU)</SelectItem>
                          <SelectItem value="cpu">CPU Only</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  />
                </div>
              </div>

              {/* Processing Pipeline */}
              <div className="space-y-4 p-4 border rounded-lg">
                <h4 className="font-medium">Processing Pipeline</h4>
                <div className="space-y-2">
                  <Label>Pipeline Type</Label>
                  <Controller
                    name="docling_pipeline"
                    control={control}
                    render={({ field }) => (
                      <Select
                        value={field.value || 'standard'}
                        onValueChange={field.onChange}
                        disabled={isLoading}
                      >
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder="Select processing pipeline" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="standard">Standard (GPU-accelerated OCR)</SelectItem>
                          <SelectItem value="vlm">VLM (Vision-Language Model)</SelectItem>
                          <SelectItem value="auto">Auto</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  />
                  <p className="text-sm text-muted-foreground">
                    Standard pipeline uses GPU-accelerated OCR for fast text extraction. VLM pipeline uses vision-language models for enhanced document understanding.
                  </p>
                </div>
              </div>

              {/* VLM Settings (shown when pipeline=vlm) */}
              <div className="space-y-4 p-4 border rounded-lg">
                <h4 className="font-medium">Vision-Language Model Settings</h4>
                <p className="text-sm text-muted-foreground">Configure VLM options (used when pipeline is set to VLM)</p>
                <div className="space-y-2">
                  <Label>VLM Model</Label>
                  <Controller
                    name="docling_vlm_model"
                    control={control}
                    render={({ field }) => (
                      <Select
                        value={field.value || 'granite-docling-258m'}
                        onValueChange={field.onChange}
                        disabled={isLoading}
                      >
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder="Select VLM model" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="granite-docling-258m">Granite Docling 258M (Recommended)</SelectItem>
                          <SelectItem value="smoldocling-256m">SmolDocling 256M</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  />
                </div>
                <div className="space-y-2">
                  <Label>VLM Framework</Label>
                  <Controller
                    name="docling_vlm_framework"
                    control={control}
                    render={({ field }) => (
                      <Select
                        value={field.value || 'auto'}
                        onValueChange={field.onChange}
                        disabled={isLoading}
                      >
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder="Select VLM framework" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="auto">Auto (Recommended)</SelectItem>
                          <SelectItem value="transformers">Transformers (PyTorch)</SelectItem>
                          <SelectItem value="mlx">MLX (Apple Silicon)</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  />
                </div>
              </div>

              {/* OCR Settings */}
              <div className="space-y-4 p-4 border rounded-lg">
                <h4 className="font-medium">OCR Settings</h4>
                <p className="text-sm text-muted-foreground">Configure OCR options (used when pipeline is set to Standard)</p>
                <div className="space-y-2">
                  <Label>OCR Engine</Label>
                  <Controller
                    name="docling_ocr_engine"
                    control={control}
                    render={({ field }) => (
                      <Select
                        value={field.value || 'easyocr'}
                        onValueChange={field.onChange}
                        disabled={isLoading}
                      >
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder="Select OCR engine" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="easyocr">EasyOCR (Recommended)</SelectItem>
                          <SelectItem value="rapidocr">RapidOCR</SelectItem>
                          <SelectItem value="tesseract">Tesseract</SelectItem>
                          <SelectItem value="auto">Auto</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>GPU OCR Acceleration</Label>
                    <p className="text-sm text-muted-foreground">Use GPU for OCR text recognition</p>
                  </div>
                  <Controller
                    name="docling_ocr_use_gpu"
                    control={control}
                    render={({ field }) => (
                      <Switch
                        checked={field.value ?? true}
                        onCheckedChange={field.onChange}
                        disabled={isLoading}
                      />
                    )}
                  />
                </div>
              </div>

              {/* Table Processing */}
              <div className="space-y-4 p-4 border rounded-lg">
                <h4 className="font-medium">Table Processing</h4>
                <div className="space-y-2">
                  <Label>Table Structure Mode</Label>
                  <Controller
                    name="docling_table_mode"
                    control={control}
                    render={({ field }) => (
                      <Select
                        value={field.value || 'accurate'}
                        onValueChange={field.onChange}
                        disabled={isLoading}
                      >
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder="Select table mode" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="accurate">Accurate (Better quality)</SelectItem>
                          <SelectItem value="fast">Fast (Quicker processing)</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  />
                </div>
              </div>

              {/* Image Export Settings (WIP) */}
              <div className="space-y-4 p-4 border rounded-lg opacity-60">
                <h4 className="font-medium">Image Export (Work in Progress)</h4>
                <p className="text-sm text-muted-foreground">These settings are configurable but not yet functional</p>
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Auto Export Images</Label>
                    <p className="text-sm text-muted-foreground">Automatically extract images from documents</p>
                  </div>
                  <Controller
                    name="docling_auto_export_images"
                    control={control}
                    render={({ field }) => (
                      <Switch
                        checked={field.value ?? false}
                        onCheckedChange={field.onChange}
                        disabled={isLoading}
                      />
                    )}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Image Scale: {imageScale.toFixed(1)}x</Label>
                  <Controller
                    name="docling_image_scale"
                    control={control}
                    render={({ field }) => (
                      <Slider
                        min={1}
                        max={4}
                        step={0.5}
                        value={[field.value ?? 2.0]}
                        onValueChange={(value) => field.onChange(value[0])}
                        disabled={isLoading}
                        className="w-full"
                      />
                    )}
                  />
                  <p className="text-sm text-muted-foreground">Higher scale = better quality, larger files</p>
                </div>
              </div>

              {/* Chunking Settings */}
              <div className="space-y-4 p-4 border rounded-lg">
                <h4 className="font-medium">Document Chunking</h4>
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Enable Chunking</Label>
                    <p className="text-sm text-muted-foreground">Split documents into semantic chunks</p>
                  </div>
                  <Controller
                    name="docling_chunking_enabled"
                    control={control}
                    render={({ field }) => (
                      <Switch
                        checked={field.value ?? false}
                        onCheckedChange={field.onChange}
                        disabled={isLoading}
                      />
                    )}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Chunking Method</Label>
                  <Controller
                    name="docling_chunking_method"
                    control={control}
                    render={({ field }) => (
                      <Select
                        value={field.value || 'hybrid'}
                        onValueChange={field.onChange}
                        disabled={isLoading}
                      >
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder="Select chunking method" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="hybrid">Hybrid (Recommended)</SelectItem>
                          <SelectItem value="hierarchical">Hierarchical</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Max Tokens per Chunk: {chunkingMaxTokens}</Label>
                  <Controller
                    name="docling_chunking_max_tokens"
                    control={control}
                    render={({ field }) => (
                      <Slider
                        min={128}
                        max={4096}
                        step={128}
                        value={[field.value ?? 512]}
                        onValueChange={(value) => field.onChange(value[0])}
                        disabled={isLoading}
                        className="w-full"
                      />
                    )}
                  />
                </div>
              </div>
            </CardContent>
          </CollapsibleContent>
        </Collapsible>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Embedding and Search</CardTitle>
          <CardDescription>
            Configure search and embedding options
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-3">
            <Label htmlFor="embedding">Default Embedding Option</Label>
            <Controller
              name="default_embedding_option"
              control={control}
              render={({ field }) => (
                <Select
                  key={field.value}
                  value={field.value || ''}
                  onValueChange={field.onChange}
                  disabled={field.disabled || isLoading}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select embedding option" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ask">Ask</SelectItem>
                    <SelectItem value="always">Always</SelectItem>
                    <SelectItem value="never">Never</SelectItem>
                  </SelectContent>
                </Select>
              )}
            />
            <Collapsible open={expandedSections.embedding} onOpenChange={() => toggleSection('embedding')}>
              <CollapsibleTrigger className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors">
                <ChevronDownIcon className={`h-4 w-4 transition-transform ${expandedSections.embedding ? 'rotate-180' : ''}`} />
                Help me choose
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-2 text-sm text-muted-foreground space-y-2">
                <p>Embedding the content will make it easier to find by you and by your AI agents. If you are running a local embedding model (Ollama, for example), you shouldn&apos;t worry about cost and just embed everything. For online providers, you might want to be careful only if you process a lot of content (like 100s of documents at a day).</p>
                <p>• Choose <strong>always</strong> if you are running a local embedding model or if your content volume is not that big</p>
                <p>• Choose <strong>ask</strong> if you want to decide every time</p>
                <p>• Choose <strong>never</strong> if you don&apos;t care about vector search or do not have an embedding provider.</p>
                <p>As a reference, OpenAI&apos;s text-embedding-3-small costs about 0.02 for 1 million tokens -- which is about 30 times the Wikipedia page for Earth. With Gemini API, Text Embedding 004 is free with a rate limit of 1500 requests per minute.</p>
              </CollapsibleContent>
            </Collapsible>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>File Management</CardTitle>
          <CardDescription>
            Configure file handling and storage options
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-3">
            <Label htmlFor="auto_delete">Auto Delete Files</Label>
            <Controller
              name="auto_delete_files"
              control={control}
              render={({ field }) => (
                <Select
                  key={field.value}
                  value={field.value || ''}
                  onValueChange={field.onChange}
                  disabled={field.disabled || isLoading}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select auto delete option" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="yes">Yes</SelectItem>
                    <SelectItem value="no">No</SelectItem>
                  </SelectContent>
                </Select>
              )}
            />
            <Collapsible open={expandedSections.files} onOpenChange={() => toggleSection('files')}>
              <CollapsibleTrigger className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors">
                <ChevronDownIcon className={`h-4 w-4 transition-transform ${expandedSections.files ? 'rotate-180' : ''}`} />
                Help me choose
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-2 text-sm text-muted-foreground space-y-2">
                <p>Once your files are uploaded and processed, they are not required anymore. Most users should allow Open Notebook to delete uploaded files from the upload folder automatically. Choose <strong>no</strong>, ONLY if you are using Notebook as the primary storage location for those files (which you shouldn&apos;t be at all). This option will soon be deprecated in favor of always downloading the files.</p>
                <p>• Choose <strong>yes</strong> (recommended) to automatically delete uploaded files after processing</p>
                <p>• Choose <strong>no</strong> only if you need to keep the original files in the upload folder</p>
              </CollapsibleContent>
            </Collapsible>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>File Organization</CardTitle>
          <CardDescription>
            Configure directory paths and file organization workflow
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-3">
            <Label htmlFor="input_dir">INPUT Directory Path</Label>
            <Controller
              name="input_directory_path"
              control={control}
              render={({ field }) => (
                <input
                  {...field}
                  type="text"
                  placeholder="./data/input"
                  className="w-full px-3 py-2 border rounded-md"
                  disabled={isLoading}
                  value={field.value || ''}
                />
              )}
            />
            <p className="text-sm text-muted-foreground">Directory for organized input files (copy/move from uploads)</p>
          </div>

          <div className="space-y-3">
            <Label htmlFor="markdown_dir">MARKDOWN Directory Path</Label>
            <Controller
              name="markdown_directory_path"
              control={control}
              render={({ field }) => (
                <input
                  {...field}
                  type="text"
                  placeholder="./data/markdown"
                  className="w-full px-3 py-2 border rounded-md"
                  disabled={isLoading}
                  value={field.value || ''}
                />
              )}
            />
            <p className="text-sm text-muted-foreground">Directory for markdown output with subdirectories for images and tables</p>
          </div>

          <div className="space-y-3">
            <Label htmlFor="output_dir">OUTPUT Directory Path</Label>
            <Controller
              name="output_directory_path"
              control={control}
              render={({ field }) => (
                <input
                  {...field}
                  type="text"
                  placeholder="./data/output"
                  className="w-full px-3 py-2 border rounded-md"
                  disabled={isLoading}
                  value={field.value || ''}
                />
              )}
            />
            <p className="text-sm text-muted-foreground">Directory for final processed files with naming scheme applied</p>
          </div>

          <div className="space-y-3">
            <Label htmlFor="file_operation">File Operation</Label>
            <Controller
              name="file_operation"
              control={control}
              render={({ field }) => (
                <Select
                  key={field.value}
                  value={field.value || ''}
                  onValueChange={field.onChange}
                  disabled={field.disabled || isLoading}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select file operation" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="copy">Copy to INPUT (keep original)</SelectItem>
                    <SelectItem value="move">Move to INPUT (remove from uploads)</SelectItem>
                    <SelectItem value="none">None (keep in uploads)</SelectItem>
                  </SelectContent>
                </Select>
              )}
            />
            <p className="text-sm text-muted-foreground">How to handle uploaded files: copy, move, or leave in place</p>
          </div>

          <div className="space-y-3">
            <Label htmlFor="naming_scheme">Output Naming Scheme</Label>
            <Controller
              name="output_naming_scheme"
              control={control}
              render={({ field }) => (
                <Select
                  key={field.value}
                  value={field.value || ''}
                  onValueChange={field.onChange}
                  disabled={field.disabled || isLoading}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select naming scheme" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="date_prefix">Date Prefix (2025-11-05_filename.pdf)</SelectItem>
                    <SelectItem value="timestamp_prefix">Timestamp Prefix (20251105_143022_filename.pdf)</SelectItem>
                    <SelectItem value="datetime_suffix">DateTime Suffix (filename_20251105_143022.pdf)</SelectItem>
                    <SelectItem value="original">Original (filename.pdf)</SelectItem>
                  </SelectContent>
                </Select>
              )}
            />
            <p className="text-sm text-muted-foreground">Naming convention for files in OUTPUT directory</p>
          </div>
        </CardContent>
      </Card>

      <div className="flex justify-end">
        <Button 
          type="submit" 
          disabled={!isDirty || updateSettings.isPending}
        >
          {updateSettings.isPending ? 'Saving...' : 'Save Settings'}
        </Button>
      </div>
    </form>
  )
}
