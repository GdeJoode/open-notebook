'use client'

import { useParams } from 'next/navigation'

import { AppShell } from '@/components/layout/AppShell'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { NotebookHeader } from '../../components/NotebookHeader'
import { SchemaBrowser } from '@/components/notebooks/schema/SchemaBrowser'
import { CoverageStatsTable } from '@/components/notebooks/schema/CoverageStatsTable'
import { OrphansDashboard } from '@/components/notebooks/orphans/OrphansDashboard'
import { PendingExtensionsPanel } from '@/components/notebooks/schema/PendingExtensionsPanel'
import { TtlDownloadButton } from '@/components/notebooks/schema/TtlDownloadButton'
import { NetworkxExportMenu } from '@/components/notebooks/exports/NetworkxExportMenu'
import { useNotebook } from '@/lib/hooks/use-notebooks'
import { useNotebookSchema } from '@/lib/hooks/use-notebook-schema'

/**
 * `/notebooks/[id]/schema` — Phase B.3a Schema tab.
 *
 * View-only browser for the notebook's effective schema (base ontology
 * + accepted extensions), per-source coverage statistics, pending
 * extensions, and a TTL download button. Edit mutations land in B.3b.
 *
 * Routing choice (Q-B-3): this is implemented as a separate sub-route
 * rather than an in-page tab on `/notebooks/[id]` so the existing
 * 3-column workspace page doesn't need to be rewritten — the
 * NotebookHeader carries a "Schema" link that toggles between the
 * two views.
 */
export default function NotebookSchemaPage() {
  const params = useParams()
  const notebookId = decodeURIComponent(params.id as string)

  const { data: notebook, isLoading: notebookLoading } = useNotebook(notebookId)
  const {
    data: schema,
    isLoading: schemaLoading,
    isError: schemaError,
  } = useNotebookSchema(notebookId)

  if (notebookLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  if (!notebook) {
    return (
      <AppShell>
        <div className="p-6">
          <h1 className="mb-4 text-2xl font-bold">Notebook Not Found</h1>
          <p className="text-muted-foreground">
            The requested notebook could not be found.
          </p>
        </div>
      </AppShell>
    )
  }

  return (
    <AppShell>
      <div className="flex min-h-0 flex-1 flex-col">
        <div className="flex-shrink-0 p-6 pb-0">
          <NotebookHeader notebook={notebook} activeTab="schema" />
        </div>

        <div className="flex-1 overflow-y-auto p-6 pt-6">
          <div
            className="mx-auto flex max-w-5xl flex-col gap-8"
            data-testid="schema-tab-content"
          >
            <section aria-labelledby="schema-heading">
              <div className="mb-3 flex items-center justify-between gap-3">
                <h2 id="schema-heading" className="text-lg font-semibold">
                  Effective schema
                </h2>
                <div className="flex items-center gap-2">
                  <TtlDownloadButton notebookId={notebookId} />
                  <NetworkxExportMenu notebookId={notebookId} />
                </div>
              </div>
              {schemaLoading ? (
                <div
                  role="status"
                  aria-live="polite"
                  className="flex items-center justify-center py-12"
                >
                  <LoadingSpinner />
                </div>
              ) : schemaError || !schema ? (
                <div
                  role="alert"
                  className="rounded-md border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive"
                >
                  Failed to load the notebook schema. Reload to retry.
                </div>
              ) : (
                <SchemaBrowser schema={schema} notebookId={notebookId} />
              )}
            </section>

            <section aria-labelledby="coverage-heading">
              <h2
                id="coverage-heading"
                className="mb-3 text-lg font-semibold"
              >
                Source coverage
              </h2>
              <CoverageStatsTable notebookId={notebookId} />
            </section>

            <section aria-labelledby="pending-heading">
              <h2
                id="pending-heading"
                className="mb-3 text-lg font-semibold"
              >
                Pending extensions
              </h2>
              {schema && (
                <PendingExtensionsPanel
                  extensions={schema.pending_extensions}
                  notebookId={notebookId}
                />
              )}
            </section>

            <section aria-labelledby="orphans-heading">
              <h2
                id="orphans-heading"
                className="mb-3 text-lg font-semibold"
              >
                Orphans
              </h2>
              <OrphansDashboard notebookId={notebookId} />
            </section>
          </div>
        </div>
      </div>
    </AppShell>
  )
}
