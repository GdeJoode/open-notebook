'use client'

import { useState } from 'react'
import Link from 'next/link'
import { NotebookResponse } from '@/lib/types/api'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Archive, ArchiveRestore, Download, Trash2 } from 'lucide-react'
import { useUpdateNotebook, useDeleteNotebook } from '@/lib/hooks/use-notebooks'
import { useSettings } from '@/lib/hooks/use-settings'
import { PrivacyModeToggle } from '@/components/routing/PrivacyModeToggle'
import { fromWire, toWire } from '@/lib/utils/privacy-mode'
import { PrivacyValue } from '@/lib/types/model-routing'
import { ConfirmDialog } from '@/components/common/ConfirmDialog'
import { formatDistanceToNow } from 'date-fns'
import { InlineEdit } from '@/components/common/InlineEdit'
import { JsonlExportButton } from '@/components/notebooks/exports/JsonlExportButton'
import { ObsidianExportDialog } from '@/components/notebooks/exports/ObsidianExportDialog'
import { OkfExportDialog } from '@/components/notebooks/exports/OkfExportDialog'
import { cn } from '@/lib/utils'

interface NotebookHeaderProps {
  notebook: NotebookResponse
  /**
   * Highlights the active tab in the secondary navigation. Defaults
   * to `"workspace"` — the 3-column page at `/notebooks/[id]`.
   * Pass `"schema"` when rendering on the Schema sub-route.
   */
  activeTab?: 'workspace' | 'schema'
}

export function NotebookHeader({
  notebook,
  activeTab = 'workspace',
}: NotebookHeaderProps) {
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  // D.1c -- Obsidian export dialog open-state. The dialog itself owns
  // its filter + mode internals; we just gate visibility.
  const [obsidianDialogOpen, setObsidianDialogOpen] = useState(false)
  // OKF.5 -- Open Knowledge Format export dialog open-state.
  const [okfDialogOpen, setOkfDialogOpen] = useState(false)

  const updateNotebook = useUpdateNotebook()
  const deleteNotebook = useDeleteNotebook()
  const { data: settings } = useSettings()

  const handlePrivacyChange = (value: PrivacyValue) => {
    updateNotebook.mutate({
      id: notebook.id,
      data: { privacy_mode: toWire(value) },
    })
  }

  const handleUpdateName = async (name: string) => {
    if (!name || name === notebook.name) return
    
    await updateNotebook.mutateAsync({
      id: notebook.id,
      data: { name }
    })
  }

  const handleUpdateDescription = async (description: string) => {
    if (description === notebook.description) return
    
    await updateNotebook.mutateAsync({
      id: notebook.id,
      data: { description: description || undefined }
    })
  }

  const handleArchiveToggle = () => {
    updateNotebook.mutate({
      id: notebook.id,
      data: { archived: !notebook.archived }
    })
  }

  const handleDelete = () => {
    deleteNotebook.mutate(notebook.id)
    setShowDeleteDialog(false)
  }

  return (
    <>
      <div className="border-b pb-6">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3 flex-1">
              <InlineEdit
                value={notebook.name}
                onSave={handleUpdateName}
                className="text-2xl font-bold"
                inputClassName="text-2xl font-bold"
                placeholder="Notebook name"
              />
              {notebook.archived && (
                <Badge variant="secondary">Archived</Badge>
              )}
            </div>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setObsidianDialogOpen(true)}
                data-testid="open-obsidian-export-dialog"
                aria-label="Export this notebook to Obsidian"
              >
                <Download className="h-4 w-4 mr-2" />
                Export Obsidian
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setOkfDialogOpen(true)}
                data-testid="open-okf-export-dialog"
                aria-label="Export this notebook to Open Knowledge Format"
              >
                <Download className="h-4 w-4 mr-2" />
                Export OKF
              </Button>
              <JsonlExportButton notebookId={notebook.id} />
              <Button
                variant="outline"
                size="sm"
                onClick={handleArchiveToggle}
              >
                {notebook.archived ? (
                  <>
                    <ArchiveRestore className="h-4 w-4 mr-2" />
                    Unarchive
                  </>
                ) : (
                  <>
                    <Archive className="h-4 w-4 mr-2" />
                    Archive
                  </>
                )}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowDeleteDialog(true)}
                className="text-red-600 hover:text-red-700"
              >
                <Trash2 className="h-4 w-4 mr-2" />
                Delete
              </Button>
            </div>
          </div>
          
          <InlineEdit
            value={notebook.description || ''}
            onSave={handleUpdateDescription}
            className="text-muted-foreground"
            inputClassName="text-muted-foreground"
            placeholder="Add a description..."
            multiline
            emptyText="Add a description..."
          />
          
          <div className="text-sm text-muted-foreground">
            Created {formatDistanceToNow(new Date(notebook.created), { addSuffix: true })} •
            Updated {formatDistanceToNow(new Date(notebook.updated), { addSuffix: true })}
          </div>

          {/* Track J.3/J.5: per-notebook privacy override (inherits the global default). */}
          <div className="pt-2 max-w-md">
            <PrivacyModeToggle
              scope="notebook"
              scopeLabel="Privacy mode"
              value={fromWire(notebook.privacy_mode)}
              onChange={handlePrivacyChange}
              globalDefault={settings?.default_privacy_mode ?? 'cloud'}
              disabled={updateNotebook.isPending}
            />
          </div>

          {/*
            Secondary nav between the 3-column workspace view and the
            B.3a Schema sub-route. Implemented as <Link>s rather than
            tabs so the underlying routes own their data fetching and
            page state — no shared state to coordinate between views.
          */}
          <nav
            aria-label="Notebook sections"
            className="flex gap-1 pt-2"
            data-testid="notebook-section-nav"
          >
            <NotebookTabLink
              href={`/notebooks/${notebook.id}`}
              label="Workspace"
              active={activeTab === 'workspace'}
            />
            <NotebookTabLink
              href={`/notebooks/${notebook.id}/schema`}
              label="Schema"
              active={activeTab === 'schema'}
            />
          </nav>
        </div>
      </div>

      <ConfirmDialog
        open={showDeleteDialog}
        onOpenChange={setShowDeleteDialog}
        title="Delete Notebook"
        description={`Are you sure you want to delete "${notebook.name}"? This action cannot be undone and will delete all sources, notes, and chat sessions.`}
        confirmText="Delete Forever"
        confirmVariant="destructive"
        onConfirm={handleDelete}
      />

      <ObsidianExportDialog
        open={obsidianDialogOpen}
        onOpenChange={setObsidianDialogOpen}
        notebookId={notebook.id}
      />

      <OkfExportDialog
        open={okfDialogOpen}
        onOpenChange={setOkfDialogOpen}
        notebookId={notebook.id}
      />
    </>
  )
}

function NotebookTabLink({
  href,
  label,
  active,
}: {
  href: string
  label: string
  active: boolean
}) {
  return (
    <Link
      href={href}
      aria-current={active ? 'page' : undefined}
      data-testid={`notebook-tab-${label.toLowerCase()}`}
      className={cn(
        'rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
        active
          ? 'bg-accent text-accent-foreground'
          : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground',
      )}
    >
      {label}
    </Link>
  )
}