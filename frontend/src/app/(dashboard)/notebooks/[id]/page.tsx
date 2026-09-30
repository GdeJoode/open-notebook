'use client'

import { useState, useEffect } from 'react'
import { useParams } from 'next/navigation'
import { NotebookHeader } from '../components/NotebookHeader'
import { SourcesColumn } from '../components/SourcesColumn'
import { NotesColumn } from '../components/NotesColumn'
import { ChatColumn } from '../components/ChatColumn'
import { SchemaSoftNudge } from '@/components/notebooks/schema/SchemaSoftNudge'
import { ExtractionPausedBanner } from '@/components/notebooks/schema/ExtractionPausedBanner'
import { ReextractPromptBanner } from '@/components/notebooks/schema/ReextractPromptBanner'
import { useNotebook } from '@/lib/hooks/use-notebooks'
import { useSources } from '@/lib/hooks/use-sources'
import { useNotes } from '@/lib/hooks/use-notes'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { PageBody } from '@/components/layout/PageBody'
import { PageHeader } from '@/components/layout/PageHeader'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useMinWidth } from '@/lib/hooks/use-media-query'

export type ContextMode = 'off' | 'insights' | 'full'

export interface ContextSelections {
  sources: Record<string, ContextMode>
  notes: Record<string, ContextMode>
}

export default function NotebookPage() {
  const params = useParams()

  // Ensure the notebook ID is properly decoded from URL
  const notebookId = decodeURIComponent(params.id as string)

  const { data: notebook, isLoading: notebookLoading } = useNotebook(notebookId)
  const { data: sources, isLoading: sourcesLoading, refetch: refetchSources } = useSources(notebookId)
  const { data: notes, isLoading: notesLoading } = useNotes(notebookId)
  // WB.5: Sources first. From lg up Sources sits beside Notes and Chat as
  // tabs; below lg all three are tabs, Sources first.
  const isLg = useMinWidth('lg')

  // Context selection state
  const [contextSelections, setContextSelections] = useState<ContextSelections>({
    sources: {},
    notes: {}
  })

  // Initialize default selections when sources/notes load
  useEffect(() => {
    if (sources && sources.length > 0) {
      setContextSelections(prev => {
        const newSourceSelections = { ...prev.sources }
        sources.forEach(source => {
          // Only set default if not already set
          if (!(source.id in newSourceSelections)) {
            // Default to 'insights' if has insights, otherwise 'full'
            newSourceSelections[source.id] = source.insights_count > 0 ? 'insights' : 'full'
          }
        })
        return { ...prev, sources: newSourceSelections }
      })
    }
  }, [sources])

  useEffect(() => {
    if (notes && notes.length > 0) {
      setContextSelections(prev => {
        const newNoteSelections = { ...prev.notes }
        notes.forEach(note => {
          // Only set default if not already set
          if (!(note.id in newNoteSelections)) {
            // Notes default to 'full'
            newNoteSelections[note.id] = 'full'
          }
        })
        return { ...prev, notes: newNoteSelections }
      })
    }
  }, [notes])

  // Handler to update context selection
  const handleContextModeChange = (itemId: string, mode: ContextMode, type: 'source' | 'note') => {
    setContextSelections(prev => ({
      ...prev,
      [type === 'source' ? 'sources' : 'notes']: {
        ...(type === 'source' ? prev.sources : prev.notes),
        [itemId]: mode
      }
    }))
  }

  const sourcesColumn = (
    <SourcesColumn
      sources={sources}
      isLoading={sourcesLoading}
      notebookId={notebookId}
      notebookName={notebook?.name}
      onRefresh={refetchSources}
      contextSelections={contextSelections.sources}
      onContextModeChange={(sourceId, mode) => handleContextModeChange(sourceId, mode, 'source')}
    />
  )
  // Mounted while hidden, so a switch of tab keeps a draft note or a chat.
  const notesTab = (
    <TabsContent value="notes" forceMount className="flex min-h-0 flex-col data-[state=inactive]:hidden">
      <NotesColumn
        notes={notes}
        isLoading={notesLoading}
        notebookId={notebookId}
        contextSelections={contextSelections.notes}
        onContextModeChange={(noteId, mode) => handleContextModeChange(noteId, mode, 'note')}
      />
    </TabsContent>
  )
  const chatTab = (
    <TabsContent value="chat" forceMount className="flex min-h-0 flex-col data-[state=inactive]:hidden">
      <ChatColumn notebookId={notebookId} contextSelections={contextSelections} />
    </TabsContent>
  )

  if (notebookLoading) {
    return (
      <PageBody fill>
        <div className="flex flex-1 items-center justify-center">
          <LoadingSpinner size="lg" />
        </div>
      </PageBody>
    )
  }

  if (!notebook) {
    return (
      <PageBody>
        <PageHeader
          title="Notebook Not Found"
          subtitle="The requested notebook could not be found."
        />
      </PageBody>
    )
  }

  return (
    <PageBody fill>
      <NotebookHeader notebook={notebook} />

      <div className="flex-1 overflow-hidden flex flex-col min-h-0">
        {/*
          Banner stack — all three self-hide when there's nothing to
          surface, so the column grid takes the full height in the
          common case. SchemaSoftNudge first by convention (it's the
          informational coverage banner); ExtractionPausedBanner is
          actionable (Resume) and points to the next user step;
          ReextractPromptBanner is the B.3d schema-changed prompt
          and lives below so it doesn't overshadow the paused-blocker
          when both fire at once.
        */}
        <div className="flex-shrink-0">
          <SchemaSoftNudge notebookId={notebookId} />
          <ExtractionPausedBanner notebookId={notebookId} />
          <ReextractPromptBanner notebookId={notebookId} />
        </div>
        {isLg ? (
          <div className="grid grid-cols-2 gap-6 flex-1 min-h-0" data-testid="notebook-columns">
            <div className="flex flex-col h-full min-h-0 overflow-hidden">{sourcesColumn}</div>
            <Tabs defaultValue="notes" className="min-h-0">
              <TabsList aria-label="Notes and chat">
                <TabsTrigger value="notes">Notes</TabsTrigger>
                <TabsTrigger value="chat">Chat</TabsTrigger>
              </TabsList>
              {notesTab}
              {chatTab}
            </Tabs>
          </div>
        ) : (
          <Tabs defaultValue="sources" className="flex-1 min-h-0" data-testid="notebook-tabs">
            <TabsList aria-label="Notebook sections">
              <TabsTrigger value="sources">Sources</TabsTrigger>
              <TabsTrigger value="notes">Notes</TabsTrigger>
              <TabsTrigger value="chat">Chat</TabsTrigger>
            </TabsList>
            <TabsContent
              value="sources"
              forceMount
              className="flex min-h-0 flex-col data-[state=inactive]:hidden"
            >
              {sourcesColumn}
            </TabsContent>
            {notesTab}
            {chatTab}
          </Tabs>
        )}
      </div>
    </PageBody>
  )
}
