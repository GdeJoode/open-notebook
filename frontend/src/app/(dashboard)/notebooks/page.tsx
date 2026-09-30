'use client'

import { useMemo, useState } from 'react'

import { NotebookList } from './components/NotebookList'
import { Button } from '@/components/ui/button'
import { Plus, RefreshCw } from 'lucide-react'
import { useNotebooks } from '@/lib/hooks/use-notebooks'
import { CreateNotebookDialog } from '@/components/notebooks/CreateNotebookDialog'
import { Input } from '@/components/ui/input'
import { PageBody } from '@/components/layout/PageBody'
import { PageHeader } from '@/components/layout/PageHeader'

export default function NotebooksPage() {
  const [createDialogOpen, setCreateDialogOpen] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const { data: notebooks, isLoading, refetch } = useNotebooks(false)
  const { data: archivedNotebooks } = useNotebooks(true)

  const normalizedQuery = searchTerm.trim().toLowerCase()

  const filteredActive = useMemo(() => {
    if (!notebooks) {
      return undefined
    }
    if (!normalizedQuery) {
      return notebooks
    }
    return notebooks.filter((notebook) =>
      notebook.name.toLowerCase().includes(normalizedQuery)
    )
  }, [notebooks, normalizedQuery])

  const filteredArchived = useMemo(() => {
    if (!archivedNotebooks) {
      return undefined
    }
    if (!normalizedQuery) {
      return archivedNotebooks
    }
    return archivedNotebooks.filter((notebook) =>
      notebook.name.toLowerCase().includes(normalizedQuery)
    )
  }, [archivedNotebooks, normalizedQuery])

  const hasArchived = (archivedNotebooks?.length ?? 0) > 0
  const isSearching = normalizedQuery.length > 0

  return (
    <>
      <PageBody>
        <PageHeader
          title="Notebooks"
          actions={
            <>
              <Button variant="outline" size="sm" onClick={() => refetch()} aria-label="Refresh notebooks">
                <RefreshCw className="h-4 w-4" />
              </Button>
              <Input
                value={searchTerm}
                onChange={(event) => setSearchTerm(event.target.value)}
                placeholder="Search notebooks..."
                aria-label="Search notebooks"
                className="w-64"
              />
              <Button onClick={() => setCreateDialogOpen(true)}>
                <Plus className="h-4 w-4 mr-2" />
                New Notebook
              </Button>
            </>
          }
        />

        <div className="space-y-8">
          <NotebookList 
            notebooks={filteredActive} 
            isLoading={isLoading}
            title="Active Notebooks"
            emptyTitle={isSearching ? 'No notebooks match your search' : undefined}
            emptyDescription={isSearching ? 'Try using a different notebook name.' : undefined}
          />
          
          {hasArchived && (
            <NotebookList 
              notebooks={filteredArchived} 
              isLoading={false}
              title="Archived Notebooks"
              collapsible
              emptyTitle={isSearching ? 'No archived notebooks match your search' : undefined}
              emptyDescription={isSearching ? 'Modify your search to find archived notebooks.' : undefined}
            />
          )}
        </div>
      </PageBody>

      <CreateNotebookDialog
        open={createDialogOpen}
        onOpenChange={setCreateDialogOpen}
      />
    </>
  )
}
