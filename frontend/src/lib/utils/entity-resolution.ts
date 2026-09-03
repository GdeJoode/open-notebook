/**
 * Pure helpers for the entity-resolution review hub (Track K.6).
 *
 * The project's node-environment vitest cannot render Radix components, so the
 * load-bearing logic the merge/overlay UI relies on lives here as framework-free
 * functions the `.test.ts` suite asserts directly. DOM/a11y/render behaviour is
 * covered by the Playwright E2E (`frontend/e2e/track-k/`).
 */

import type {
  ApplyClusterInput,
  MergeCandidate,
  MergeCluster,
  OverlayCreateInput,
} from '@/lib/api/entity-resolution'

/**
 * Build the explicit `POST /apply` payload for a reviewed cluster, honouring a
 * user-chosen winner. When the user re-picks the winner from a former loser, the
 * old winner becomes a loser so exactly one survivor remains — the apply path is
 * destructive, so the membership set must stay invariant under a winner swap.
 */
export function buildApplyCluster(
  cluster: MergeCluster,
  winnerId: string,
): ApplyClusterInput {
  const members = [cluster.winner_id, ...cluster.loser_ids]
  if (!members.includes(winnerId)) {
    // Defensive: an unknown winner id means "keep the planned winner".
    winnerId = cluster.winner_id
  }
  const loser_ids = members.filter((id) => id !== winnerId)
  return {
    new_canonical: cluster.new_canonical,
    entity_type: cluster.entity_type,
    winner_id: winnerId,
    loser_ids,
    member_surface_forms: cluster.member_surface_forms,
  }
}

/**
 * Human-readable summary of what an apply will do — the text shown in the
 * destructive-merge confirmation step (AC2: "show what merges before applying").
 */
export function describeMerge(cluster: ApplyClusterInput): {
  winnerLabel: string
  loserCount: number
  loserIds: string[]
} {
  return {
    winnerLabel: cluster.new_canonical,
    loserCount: cluster.loser_ids.length,
    loserIds: cluster.loser_ids,
  }
}

/**
 * Reject is reversible, never a hard delete: it records a `force-split` overlay
 * for the cluster's first two surface forms so the pair is never re-proposed
 * (AC3). Scope is global unless a notebook is supplied.
 */
export function buildRejectOverlay(
  cluster: Pick<MergeCluster, 'entity_type' | 'member_surface_forms'>,
  notebookId?: string,
): OverlayCreateInput | null {
  const [name_a, name_b] = cluster.member_surface_forms
  if (!name_a || !name_b) return null
  return {
    kind: 'split',
    name_a,
    name_b,
    entity_type: cluster.entity_type,
    scope: notebookId ? 'notebook' : 'global',
    notebook_id: notebookId ?? null,
  }
}

/** Same reject→split rule for a K.5 fuzzy candidate pair. */
export function buildCandidateRejectOverlay(
  candidate: Pick<MergeCandidate, 'entity_type' | 'name_a' | 'name_b'>,
  notebookId?: string,
): OverlayCreateInput {
  return {
    kind: 'split',
    name_a: candidate.name_a,
    name_b: candidate.name_b,
    entity_type: candidate.entity_type,
    scope: notebookId ? 'notebook' : 'global',
    notebook_id: notebookId ?? null,
  }
}

/** Promote a K.5 candidate to an apply-cluster (auto-merge band → confirm → apply). */
export function candidateToApplyCluster(
  candidate: MergeCandidate,
): ApplyClusterInput {
  // The apply repoints relations onto `winner_id`, so the surviving entity's
  // LABEL must be the winner's — mirroring `MergeCandidate.to_merge_cluster`
  // server-side, which spells this out. `name_a` unconditionally mislabels the
  // survivor whenever b wins on confidence.
  const bWins = candidate.winner_id === candidate.id_b
  return {
    new_canonical: bWins ? candidate.name_b : candidate.name_a,
    // Same argument for the type. A cross-type candidate carries two answers
    // (`entity_type_b`), and taking a's blindly labels the survivor with the
    // loser's type.
    entity_type:
      bWins && candidate.entity_type_b
        ? candidate.entity_type_b
        : candidate.entity_type,
    winner_id: candidate.winner_id,
    loser_ids: [candidate.loser_id],
    member_surface_forms: [candidate.name_a, candidate.name_b],
  }
}

/** True when a candidate's two entities carry different types.
 *
 * Only `fold_equal_cross_type` produces these, and there the two NAMES are
 * byte-identical — the type is the entire reason the graph holds two rows. A card
 * that does not surface it shows the same name twice with nothing to decide on.
 */
export function isCrossTypeCandidate(candidate: MergeCandidate): boolean {
  return Boolean(
    candidate.entity_type_b && candidate.entity_type_b !== candidate.entity_type,
  )
}

/** The type line for a candidate card: one type, or both when they differ.
 *
 * A pure function rather than JSX so it has a test that can fail — the card
 * itself mounts a Radix AlertDialog that the node-environment vitest cannot
 * render, which is the same reason `candidateToApplyCluster` lives here.
 */
export function candidateTypeLabel(candidate: MergeCandidate): string {
  return isCrossTypeCandidate(candidate)
    ? `${candidate.entity_type} ↔ ${candidate.entity_type_b}`
    : candidate.entity_type
}

/** Validate an overlay rule before POSTing (mirrors the server 422 guards). */
export function isValidOverlay(rule: OverlayCreateInput): boolean {
  if (rule.kind !== 'merge' && rule.kind !== 'split') return false
  if (!rule.name_a.trim() || !rule.name_b.trim()) return false
  if (rule.scope === 'notebook' && !rule.notebook_id) return false
  return true
}
