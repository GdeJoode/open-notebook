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
import { cn } from '@/lib/utils'

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

  // One Tabs tree at every width, so Notes and Chat stay mounted across lg
  // (a draft note, the chat input, a streaming answer). Below lg `tab` is
  // shown; from lg Sources is always shown and the right column shows the
  // last of Notes and Chat that was picked.
  const [tab, setTab] = useState<'sources' | 'notes' | 'chat'>('sources')
  const [side, setSide] = useState<'notes' | 'chat'>('notes')
  const pickTab = (value: string) => {
    const next = value as 'sources' | 'notes' | 'chat'
    setTab(next)
    if (next !== 'sources') setSide(next)
  }
  // Hidden panels stay mounted: forceMount, hidden by CSS while inactive.
  const panel = 'flex min-h-0 flex-col data-[state=inactive]:hidden'

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

  // Below lg the page scrolls as a whole: the header and banners can take
  // most of a phone's height, and the tabs must stay reachable.
  return (
    <PageBody fill={isLg}>
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
        <Tabs
          value={isLg ? side : tab}
          onValueChange={pickTab}
          className={cn(
            'flex-1 min-h-0',
            isLg && 'grid grid-cols-2 grid-rows-[auto_minmax(0,1fr)] gap-x-6 gap-y-2'
          )}
          data-testid={isLg ? 'notebook-columns' : 'notebook-tabs'}
        >
          <TabsList
            aria-label={isLg ? 'Notes and chat' : 'Notebook sections'}
            className={cn(isLg && 'col-start-2 row-start-1')}
          >
            {!isLg && <TabsTrigger value="sources">Sources</TabsTrigger>}
            <TabsTrigger value="notes">Notes</TabsTrigger>
            <TabsTrigger value="chat">Chat</TabsTrigger>
          </TabsList>
          <TabsContent
            value="sources"
            forceMount
            className={isLg
              ? 'col-start-1 row-start-1 row-span-2 flex min-h-0 flex-col overflow-hidden'
              : panel}
          >
            <SourcesColumn
              sources={sources}
              isLoading={sourcesLoading}
              notebookId={notebookId}
              notebookName={notebook?.name}
              onRefresh={refetchSources}
              contextSelections={contextSelections.sources}
              onContextModeChange={(sourceId, mode) => handleContextModeChange(sourceId, mode, 'source')}
            />
          </TabsContent>
          <TabsContent value="notes" forceMount className={cn(panel, isLg && 'col-start-2 row-start-2')}>
            <NotesColumn
              notes={notes}
              isLoading={notesLoading}
              notebookId={notebookId}
              contextSelections={contextSelections.notes}
              onContextModeChange={(noteId, mode) => handleContextModeChange(noteId, mode, 'note')}
            />
          </TabsContent>
          <TabsContent value="chat" forceMount className={cn(panel, isLg && 'col-start-2 row-start-2')}>
            <ChatColumn notebookId={notebookId} contextSelections={contextSelections} />
          </TabsContent>
        </Tabs>
      </div>
    </PageBody>
  )
}
