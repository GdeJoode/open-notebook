"use client"

import { useState } from "react"
import { Control, FieldErrors, UseFormRegister, useWatch } from "react-hook-form"
import { FileIcon, LinkIcon, FileTextIcon, InboxIcon, FolderOpenIcon, ChevronDownIcon, ChevronRightIcon } from "lucide-react"
import { FormSection } from "@/components/ui/form-section"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Controller } from "react-hook-form"

interface CreateSourceFormData {
  type: 'link' | 'upload' | 'text' | 'queue'
  title?: string
  url?: string
  content?: string
  file?: FileList | File
  notebooks?: string[]
  transformations?: string[]
  embed: boolean
  async_processing: boolean
}

const SOURCE_TYPES = [
  {
    value: 'link' as const,
    label: 'Link',
    icon: LinkIcon,
    description: 'Add a web page or URL',
    disabled: false,
  },
  {
    value: 'upload' as const,
    label: 'Upload',
    icon: FileIcon,
    description: 'Upload a document or file',
    disabled: false,
  },
  {
    value: 'text' as const,
    label: 'Text',
    icon: FileTextIcon,
    description: 'Add text content directly',
    disabled: false,
  },
  {
    value: 'queue' as const,
    label: 'Queue',
    icon: InboxIcon,
    description: 'Files from the enrichment pipeline will appear here.',
    disabled: true,
  },
]

interface SourceTypeStepProps {
  control: Control<CreateSourceFormData>
  register: UseFormRegister<CreateSourceFormData>
  errors: FieldErrors<CreateSourceFormData>
}

export function SourceTypeStep({ control, register, errors }: SourceTypeStepProps) {
  const selectedType = useWatch({ control, name: 'type' })
  const [fileListExpanded, setFileListExpanded] = useState(false)

  return (
    <div className="space-y-6">
      <FormSection
        title="Source Type"
        description="Choose how you want to add your content"
      >
        <Controller
          control={control}
          name="type"
          render={({ field }) => (
            <Tabs
              value={field.value || ''}
              onValueChange={(value) => {
                if (value !== 'queue') {
                  field.onChange(value as 'link' | 'upload' | 'text')
                }
              }}
              className="w-full"
            >
              <TabsList className="grid w-full grid-cols-4">
                {SOURCE_TYPES.map((type) => {
                  const Icon = type.icon
                  return (
                    <TabsTrigger
                      key={type.value}
                      value={type.value}
                      disabled={type.disabled}
                      className="gap-2 relative"
                    >
                      <Icon className="h-4 w-4" />
                      {type.label}
                      {type.disabled && (
                        <Badge variant="secondary" className="ml-1 text-[10px] px-1 py-0 leading-tight">
                          Soon
                        </Badge>
                      )}
                    </TabsTrigger>
                  )
                })}
              </TabsList>

              {SOURCE_TYPES.map((type) => (
                <TabsContent key={type.value} value={type.value} className="mt-4">
                  <p className="text-sm text-muted-foreground mb-4">{type.description}</p>

                  {type.value === 'link' && (
                    <div>
                      <Label htmlFor="url" className="mb-2 block">URL *</Label>
                      <Input
                        id="url"
                        {...register('url')}
                        placeholder="https://example.com/article"
                        type="url"
                      />
                      {errors.url && (
                        <p className="text-sm text-destructive mt-1">{errors.url.message}</p>
                      )}
                    </div>
                  )}

                  {type.value === 'upload' && (
                    <UploadSection
                      register={register}
                      errors={errors}
                      control={control}
                      fileListExpanded={fileListExpanded}
                      onToggleFileList={() => setFileListExpanded(prev => !prev)}
                    />
                  )}

                  {type.value === 'text' && (
                    <div>
                      <Label htmlFor="content" className="mb-2 block">Text Content *</Label>
                      <Textarea
                        id="content"
                        {...register('content')}
                        placeholder="Paste or type your content here..."
                        rows={6}
                      />
                      {errors.content && (
                        <p className="text-sm text-destructive mt-1">{errors.content.message}</p>
                      )}
                    </div>
                  )}
                </TabsContent>
              ))}
            </Tabs>
          )}
        />
        {errors.type && (
          <p className="text-sm text-destructive mt-1">{errors.type.message}</p>
        )}
      </FormSection>

      <FormSection
        title={selectedType === 'text' ? "Title *" : "Title (optional)"}
        description={selectedType === 'text'
          ? "A title is required for text content"
          : "If left empty, a title will be generated from the content"
        }
      >
        <Input
          id="title"
          {...register('title')}
          placeholder="Give your source a descriptive title"
        />
        {errors.title && (
          <p className="text-sm text-destructive mt-1">{errors.title.message}</p>
        )}
      </FormSection>
    </div>
  )
}

function UploadSection({
  register,
  errors,
  control,
  fileListExpanded,
  onToggleFileList,
}: {
  register: UseFormRegister<CreateSourceFormData>
  errors: FieldErrors<CreateSourceFormData>
  control: Control<CreateSourceFormData>
  fileListExpanded: boolean
  onToggleFileList: () => void
}) {
  const watchedFile = useWatch({ control, name: 'file' })
  const fileCount = watchedFile instanceof FileList ? watchedFile.length : watchedFile ? 1 : 0
  const fileNames: string[] = []
  if (watchedFile instanceof FileList) {
    for (let i = 0; i < watchedFile.length; i++) {
      fileNames.push(watchedFile[i].name)
    }
  } else if (watchedFile instanceof File) {
    fileNames.push(watchedFile.name)
  }

  return (
    <div className="space-y-3">
      <Label className="mb-2 block">Files *</Label>
      <div className="flex gap-2">
        <div className="flex-1">
          <Input
            id="file"
            type="file"
            multiple
            {...register('file')}
            accept=".pdf,.doc,.docx,.txt,.md,.epub"
          />
        </div>
        <FolderUploadButton
          onFiles={(files) => {
            // Create a DataTransfer to build a FileList from the folder input
            const dt = new DataTransfer()
            for (const f of files) {
              dt.items.add(f)
            }
            // Programmatically set the file input value
            const fileInput = document.getElementById('file') as HTMLInputElement
            if (fileInput) {
              fileInput.files = dt.files
              // Trigger react-hook-form's change handler
              const event = new Event('change', { bubbles: true })
              fileInput.dispatchEvent(event)
            }
          }}
        />
      </div>

      {fileCount > 0 && (
        <div className="rounded-md border p-3 bg-muted/50">
          <button
            type="button"
            className="flex items-center gap-2 text-sm font-medium w-full text-left"
            onClick={onToggleFileList}
          >
            {fileListExpanded ? (
              <ChevronDownIcon className="h-4 w-4" />
            ) : (
              <ChevronRightIcon className="h-4 w-4" />
            )}
            {fileCount} file{fileCount !== 1 ? 's' : ''} selected
          </button>
          {fileListExpanded && (
            <ul className="mt-2 space-y-1 pl-6">
              {fileNames.map((name, i) => (
                <li key={i} className="text-sm text-muted-foreground flex items-center gap-1.5">
                  <FileIcon className="h-3 w-3 shrink-0" />
                  {name}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <p className="text-xs text-muted-foreground">
        Supported formats: PDF, DOC, DOCX, TXT, MD, EPUB
      </p>
      {errors.file && (
        <p className="text-sm text-destructive mt-1">{errors.file.message}</p>
      )}
    </div>
  )
}

function FolderUploadButton({ onFiles }: { onFiles: (files: File[]) => void }) {
  const handleClick = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.setAttribute('webkitdirectory', '')
    input.setAttribute('directory', '')
    input.multiple = true
    input.accept = '.pdf,.doc,.docx,.txt,.md,.epub'
    input.onchange = () => {
      if (input.files && input.files.length > 0) {
        const files: File[] = []
        for (let i = 0; i < input.files.length; i++) {
          files.push(input.files[i])
        }
        onFiles(files)
      }
    }
    input.click()
  }

  return (
    <Button type="button" variant="outline" size="default" onClick={handleClick} className="gap-2 shrink-0">
      <FolderOpenIcon className="h-4 w-4" />
      Folder
    </Button>
  )
}
