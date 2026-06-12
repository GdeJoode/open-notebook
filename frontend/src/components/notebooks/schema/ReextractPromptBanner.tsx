'use client'

/**
 * ReextractPromptBanner — Phase B.3d.
 *
 * After a schema edit op (rename / merge / split / accept / etc.)
 * SchemaEditService emits a `notebook_event{type:"schema_changed"}`.
 * The banner polls the unread stream and, when at least one such
 * event exists, surfaces:
 *
 *   "Schema gewijzigd. N affected sources re-extracten?"
 *
 * Three actions:
 *
 *   [Re-extract all]       → POST /schema/reextract with every
 *                            candidate source id.
 *   [Re-extract selected…] → Open a modal with a multi-select.
 *   [Later]                → Mark every unread schema_changed event
 *                            read so the banner disappears. Affected
 *                            sources retain a "Schema changed" hint
 *                            until their next re-extraction (rendered
 *                            elsewhere; out of scope for the banner).
 *
 * # Design decisions
 *
 *  1. **Render only when there are unread schema_changed events.**
 *     The banner is purely event-driven — no event, no banner. This
 *     keeps the workspace clean for users who never touch the schema.
 *
 *  2. **Share the B.3c `useNotebookEvents` hook.** After B.3c merged,
 *     the canonical events hook is `useNotebookEvents(id, { types,
 *     unread })`. The banner delegates to it with
 *     `types: ['schema_changed'], unread: true` so we keep a single
 *     polling loop per notebook regardless of how many banners are
 *     mounted (the React Query cache deduplicates concurrent
 *     subscribers automatically).
 *
 *  3. **Action labels stay in Dutch for the primary CTA** to match
 *     the plan's "Schema gewijzigd" wording; the secondary actions
 *     use English to match the existing UI vocabulary (Cancel,
 *     Later, etc.). A future i18n pass will unify these.
 *
 *  4. **Per-source processing indicators** are shown elsewhere (the
 *     workspace SourcesColumn renders job-status badges from the
 *     existing job-queue API). The banner only owns the prompt.
 *
 *  5. **Aria + keyboard.** The banner is a `role="region"` with an
 *     `aria-label`; the action buttons have explicit `aria-label`s.
 *     The select-modal is a Radix Dialog (focus trap + Esc close
 *     inherited).
 */

import { useState } from 'react'
import { AlertTriangle, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  useMarkEventRead,
  useNotebookEvents,
  useReextractCandidates,
  useReextractMutation,
} from '@/lib/hooks/use-notebook-schema'

interface ReextractPromptBannerProps {
  notebookId: string
}

export function ReextractPromptBanner({ notebookId }: ReextractPromptBannerProps) {
  // B.3c shipped `useNotebookEvents` as the canonical events hook with
  // a `types[]` filter; we delegate to it rather than maintain a
  // duplicate `useSchemaChangedEvents` (attempt 1 of B.3d carried a
  // duplicate that conflicted with B.3c on rebase).
  const eventsQuery = useNotebookEvents(notebookId, {
    types: ['schema_changed'],
    unread: true,
  })
  // The candidate list is fetched eagerly so the banner can show "N
  // sources" without waiting for a click. The user pays one extra GET
  // when the banner appears, which is fine — 4-30 source ids per
  // notebook in the typical range.
  const candidatesQuery = useReextractCandidates(notebookId, {
    enabled: !!notebookId && (eventsQuery.data?.length ?? 0) > 0,
  })
  const reextractMutation = useReextractMutation(notebookId)
  const markReadMutation = useMarkEventRead(notebookId)

  const [selectOpen, setSelectOpen] = useState(false)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())

  const events = eventsQuery.data ?? []
  const candidates = candidatesQuery.data?.source_ids ?? []
  const candidateCount = candidates.length

  // Hide the banner when there are no unread events. We intentionally
  // do NOT render a loading placeholder — the events query polls
  // silently in the background, and a flicker on every poll would be
  // visually noisy for users who never touch the schema.
  if (eventsQuery.isLoading || events.length === 0) {
    return null
  }

  const handleReextractAll = () => {
    if (candidateCount === 0) return
    reextractMutation.mutate(
      { source_ids: candidates },
      {
        onSuccess: () => {
          // Mark every active schema_changed event as read so the
          // banner disappears. We fire in parallel; failure of any one
          // is logged by the hook but not surfaced (the banner will
          // simply persist on the next poll).
          events.forEach((evt) => markReadMutation.mutate(evt.id))
        },
      },
    )
  }

  const handleOpenSelect = () => {
    // Default-select every candidate so the user can prune rather
    // than tick — matches the natural "I want to re-extract most"
    // flow after a schema rename.
    setSelectedIds(new Set(candidates))
    setSelectOpen(true)
  }

  const handleSubmitSelected = () => {
    const ids = Array.from(selectedIds)
    reextractMutation.mutate(
      { source_ids: ids },
      {
        onSuccess: () => {
          events.forEach((evt) => markReadMutation.mutate(evt.id))
          setSelectOpen(false)
        },
      },
    )
  }

  const handleLater = () => {
    events.forEach((evt) => markReadMutation.mutate(evt.id))
  }

  const isMutating = reextractMutation.isPending || markReadMutation.isPending
  const candidatesLoading = candidatesQuery.isLoading

  return (
    <>
      <div
        role="region"
        aria-label="Schema changed, re-extract prompt"
        data-testid="reextract-prompt-banner"
        className="mb-4 flex flex-wrap items-center gap-3 rounded-md border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-100"
      >
        <AlertTriangle className="h-4 w-4 flex-shrink-0" aria-hidden />
        <div className="flex-1 min-w-[200px]">
          <p className="font-medium">Schema gewijzigd.</p>
          <p className="text-xs opacity-90">
            {candidatesLoading
              ? 'Loading affected sources…'
              : `${candidateCount} affected source${candidateCount === 1 ? '' : 's'} can be re-extracted.`}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            type="button"
            size="sm"
            data-testid="reextract-all"
            aria-label={`Re-extract all ${candidateCount} affected sources`}
            disabled={
              isMutating || candidatesLoading || candidateCount === 0
            }
            onClick={handleReextractAll}
          >
            {reextractMutation.isPending && (
              <Loader2
                className="mr-1 h-3 w-3 animate-spin"
                aria-hidden
              />
            )}
            Re-extract all
          </Button>
          <Button
            type="button"
            size="sm"
            variant="outline"
            data-testid="reextract-selected"
            aria-label="Choose which sources to re-extract"
            disabled={isMutating || candidatesLoading || candidateCount === 0}
            onClick={handleOpenSelect}
          >
            Re-extract selected…
          </Button>
          <Button
            type="button"
            size="sm"
            variant="ghost"
            data-testid="reextract-later"
            aria-label="Dismiss prompt and re-extract later"
            disabled={isMutating}
            onClick={handleLater}
          >
            Later
          </Button>
        </div>
      </div>

      <Dialog open={selectOpen} onOpenChange={setSelectOpen}>
        <DialogContent
          className="max-w-md"
          data-testid="reextract-select-dialog"
        >
          <DialogHeader>
            <DialogTitle>Re-extract selected sources</DialogTitle>
            <DialogDescription>
              Pick the sources whose entity extraction should be re-run
              against the updated schema. Sources with an in-flight
              extraction job are skipped automatically.
            </DialogDescription>
          </DialogHeader>
          <ScrollArea className="max-h-72 rounded-md border">
            <ul className="divide-y" role="list">
              {candidates.length === 0 && (
                <li
                  className="p-3 text-sm text-muted-foreground"
                  data-testid="reextract-select-empty"
                >
                  No candidate sources to display.
                </li>
              )}
              {candidates.map((sourceId) => {
                const checked = selectedIds.has(sourceId)
                return (
                  <li
                    key={sourceId}
                    className="flex items-center gap-2 p-2"
                  >
                    <Checkbox
                      id={`reextract-pick-${sourceId}`}
                      data-testid={`reextract-pick-${sourceId}`}
                      checked={checked}
                      onCheckedChange={(v) => {
                        const next = new Set(selectedIds)
                        if (v) next.add(sourceId)
                        else next.delete(sourceId)
                        setSelectedIds(next)
                      }}
                    />
                    <label
                      htmlFor={`reextract-pick-${sourceId}`}
                      className="cursor-pointer truncate text-xs font-mono"
                    >
                      {sourceId}
                    </label>
                  </li>
                )
              })}
            </ul>
          </ScrollArea>
          <DialogFooter>
            <Button
              type="button"
              variant="ghost"
              onClick={() => setSelectOpen(false)}
              disabled={reextractMutation.isPending}
            >
              Cancel
            </Button>
            <Button
              type="button"
              data-testid="reextract-select-submit"
              disabled={reextractMutation.isPending || selectedIds.size === 0}
              onClick={handleSubmitSelected}
            >
              {reextractMutation.isPending && (
                <Loader2 className="mr-1 h-3 w-3 animate-spin" aria-hidden />
              )}
              Re-extract {selectedIds.size}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

/**
 * Default export so the consuming page can lazy-import if needed —
 * currently the page imports the named export.
 */
export default ReextractPromptBanner
