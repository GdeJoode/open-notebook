'use client'

/**
 * OE.4: the retype of existing entities. A publish changes the type of NEW
 * rows only; rows extracted before keep theirs, and the next mention of such a
 * name would become a second entity. This shows the dry run under the active
 * ontologies and applies exactly that plan on confirmation (OE-D3). Rows whose
 * new identity another row already holds are not retyped: they are listed for
 * the curator's merge on the resolution page.
 */

import Link from 'next/link'
import { useState } from 'react'

import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { ConfirmButton } from '@/components/ontologies/ConfirmButton'
import { Button } from '@/components/ui/button'
import type { RetypeApplyResult, RetypeRow } from '@/lib/api/ontologies'
import {
  errorDetail,
  useApplyRetype,
  useRetypeDryRun,
  useUndoRetype,
} from '@/lib/hooks/use-ontology-editor'

export function RetypePanel({ onClose }: { onClose: () => void }) {
  const dryRun = useRetypeDryRun()
  const apply = useApplyRetype()
  const undo = useUndoRetype()
  const [last, setLast] = useState<RetypeApplyResult | null>(null)
  const plan = dryRun.data
  const toWrite = plan ? plan.changes.length + plan.flag_only.length : 0
  const blockedByNotebooks = (plan?.notebook_edits.length ?? 0) > 0

  return (
    <div className="space-y-2 rounded-md border p-3 text-sm" data-testid="ontology-retype">
      <div className="flex items-center gap-2">
        <h3 className="font-semibold">Retype existing entities</h3>
        <span className="text-xs text-muted-foreground">
          every notebook, under the active ontologies
        </span>
        <div className="ml-auto flex items-center gap-2">
          <Button size="sm" variant="outline" disabled={dryRun.isFetching}
            onClick={() => dryRun.refetch()}>
            {dryRun.isFetching ? 'Checking…' : 'Dry run again'}
          </Button>
          <Button size="sm" variant="ghost" onClick={onClose}>
            Close
          </Button>
        </div>
      </div>

      {dryRun.isLoading ? (
        <LoadingSpinner />
      ) : dryRun.isError || !plan ? (
        <p className="text-destructive" role="alert">
          Could not run the retype dry run: {errorDetail(dryRun.error)}
        </p>
      ) : (
        <>
          <p data-testid="ontology-retype-counts">
            {plan.counts.changes} type change(s), {plan.counts.flag_only} literal(s) to flag,
            {' '}{plan.counts.collisions} collision(s) left for a merge.
          </p>
          {blockedByNotebooks && (
            <p className="text-destructive" role="alert">
              Notebooks with schema edits ({plan.notebook_edits.join(', ')}): the retype does not
              include them, so it cannot be applied.
            </p>
          )}
          {toWrite === 0 ? (
            <p className="text-muted-foreground" data-testid="ontology-retype-empty">
              Nothing to retype: every entity has the type the active ontologies give it.
            </p>
          ) : (
            <>
              <RetypeRows rows={plan.changes} testId="ontology-retype-changes" />
              {plan.flag_only.length > 0 && (
                <details className="text-xs text-muted-foreground">
                  <summary>
                    {plan.flag_only.length} literal(s) keep their type and are flagged as not a type
                  </summary>
                  <RetypeRows rows={plan.flag_only} />
                </details>
              )}
              <ConfirmButton size="sm" data-testid="ontology-retype-apply"
                disabled={blockedByNotebooks || apply.isPending || dryRun.isFetching}
                title={`Retype ${toWrite} entities?`}
                description={
                  'Exactly the rows listed are written, each only if it still has its old type. ' +
                  'If the plan changed since this dry run, nothing is written and the new one is ' +
                  'shown. The change can be undone afterwards.'}
                confirmLabel="Retype"
                onConfirm={() => apply.mutate(plan.fingerprint, { onSuccess: setLast })}>
                Apply this retype
              </ConfirmButton>
            </>
          )}
          {plan.collisions.length > 0 && (
            <div className="space-y-1" data-testid="ontology-retype-collisions">
              <p>
                Not retyped, because another entity already holds the new name and type. Those
                are one entity: merge them on the{' '}
                <Link href="/knowledge-graph/resolution" className="underline">
                  resolution page
                </Link>.
              </p>
              <RetypeRows rows={plan.collisions} />
            </div>
          )}
        </>
      )}

      {last && (
        <div className="flex flex-wrap items-center gap-2 border-t pt-2"
          data-testid="ontology-retype-result">
          <span>
            Written {last.written}; changed meanwhile {last.skipped.length}; failed{' '}
            {last.failed.length}.
          </span>
          {last.rollback_id && (
            <ConfirmButton size="sm" variant="ghost" disabled={undo.isPending}
              data-testid="ontology-retype-undo" title="Undo this retype?"
              description="The rows it wrote go back to their previous type, if nothing retyped them since."
              confirmLabel="Undo"
              onConfirm={() => undo.mutate(last.rollback_id as string,
                { onSuccess: () => setLast(null) })}>
              Undo
            </ConfirmButton>
          )}
        </div>
      )}
    </div>
  )
}

function RetypeRows({ rows, testId }: { rows: RetypeRow[]; testId?: string }) {
  return (
    <ul className="max-h-64 overflow-auto rounded border p-2 text-xs" data-testid={testId}>
      {rows.map((r) => (
        <li key={r.id}>
          {r.name ?? r.id} <code>{r.label}</code>: {r.old} → {r.new}
          {r.status !== 'active' && <span className="text-muted-foreground"> ({r.status})</span>}
          {r.held_by?.length ? (
            <span className="text-muted-foreground"> held by {r.held_by.join(', ')}</span>
          ) : null}
        </li>
      ))}
    </ul>
  )
}
