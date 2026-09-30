'use client'

import { useParams } from 'next/navigation'

import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { PageBody } from '@/components/layout/PageBody'
import { PageHeader } from '@/components/layout/PageHeader'
import { NotebookHeader } from '../../components/NotebookHeader'
import { SchemaBrowser } from '@/components/notebooks/schema/SchemaBrowser'
import { CoverageStatsTable } from '@/components/notebooks/schema/CoverageStatsTable'
import { OrphansDashboard } from '@/components/notebooks/orphans/OrphansDashboard'
import { AuditWidget } from '@/components/notebooks/audit/AuditWidget'
import { LibrarianSettings } from '@/components/notebooks/audit/LibrarianSettings'
import { PendingExtensionsPanel } from '@/components/notebooks/schema/PendingExtensionsPanel'
import { ExtractionPausedBanner } from '@/components/notebooks/schema/ExtractionPausedBanner'
import { NextStepLink } from '@/components/common/NextStepLink'
import { TtlDownloadButton } from '@/components/notebooks/schema/TtlDownloadButton'
import { NetworkxExportMenu } from '@/components/notebooks/exports/NetworkxExportMenu'
import { useNotebook } from '@/lib/hooks/use-notebooks'
import { useNotebookSchema, usePausedExtraction } from '@/lib/hooks/use-notebook-schema'

/**
 * `/notebooks/[id]/schema` — Phase B.3a Schema tab.
 *
 * The notebook's declared base ontology (read-only; the ontologies are
 * shared by every notebook and edited in the ontology editor, OE.5),
 * per-source coverage statistics, the Pass-1 proposal queue (Accept
 * publishes to a shared ontology), and a TTL download button.
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
  const paused = usePausedExtraction(notebookId)
  const nothingPaused = paused.isSuccess && paused.data.paused_count === 0

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
    <PageBody>
      <NotebookHeader notebook={notebook} activeTab="schema" />

      <div
        className="flex flex-col gap-8"
        data-testid="schema-tab-content"
      >
        <section aria-labelledby="schema-heading">
          <div className="mb-3 flex items-center justify-between gap-3">
            <h2 id="schema-heading" className="text-lg font-semibold">
              Schema
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
            <SchemaBrowser schema={schema} />
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
          {/* WB.2: a gated source waits here. Accepting a proposal does not
              resume its extraction; this banner does. WB.3: below the
              proposals, so they come first and Resume is secondary. */}
          <div className="mt-3">
            <ExtractionPausedBanner notebookId={notebookId} showSchemaLink={false} />
          </div>
          {/* WB.3: The Tray is next only once nothing here waits. */}
          {schema && schema.pending_extensions.length === 0 && nothingPaused && (
            <div className="mt-3">
              <NextStepLink href="/" data-testid="schema-next">
                Next: The Tray
              </NextStepLink>
            </div>
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

        <section aria-labelledby="audit-heading">
          <h2 id="audit-heading" className="mb-3 text-lg font-semibold">
            Quality audit
          </h2>
          <AuditWidget notebookId={notebookId} />
          <div className="mt-3">
            <LibrarianSettings notebookId={notebookId} />
          </div>
        </section>
      </div>
    </PageBody>
  )
}
