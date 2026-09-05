"""Retroactive entity canonicalization over the already-persisted KG (K.3).

K.1 + K.2 sharpened ``normalize_entity_name`` for NEW extractions, but entities
already in the graph keep their old fragmented ``canonical_name``s (the measured
897-distinct / 107-shared Convenant set). This service re-normalizes the
persisted entities, groups the collisions, and merges duplicates — reversibly.

Two phases, kept strictly separate so the destructive one is opt-in:

* ``plan_merges`` — a **dry-run** that re-normalizes every active entity,
  groups by ``(new_canonical, entity_type)``, and returns the clusters of size
  ≥2. **Zero writes.** Grouping is type-aware, so a ``person`` and an ``org``
  with the same name never land in the same cluster (the over-merge guard).
* ``apply_merge`` — one logical merge per cluster: ID-based relation
  repointing (with parallel-edge dedup), provenance merge into the winner
  reusing the ``upsert_entity`` merge math, ``entity_alias`` rows for the loser
  surface forms, and a soft ``status='merged'`` + ``merged_into`` on each loser
  (never a hard delete). Idempotent — re-applying a cluster whose losers are
  already merged is a no-op.
* ``apply_plan`` — batch wrapper with a HARD ``dry_run=True`` default.

Precondition (inherited from ``upsert_entity``): K.3 is an offline/maintenance
operation run without concurrent KG writers — the SELECT-then-UPDATE flow is not
atomic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from shared.utils.name_normalizer import normalize_entity_name
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.entity import EntityRepository


@dataclass
class MergeMember:
    """One entity participating in a merge cluster."""

    id: str
    surface_form: str
    entity_type: str
    confidence: float
    source_documents: List[str] = field(default_factory=list)
    created_at: Optional[str] = None


@dataclass
class MergeCluster:
    """A group of active entities that collide under the new normalizer.

    Members all share the same ``(new_canonical, entity_type)`` key. The winner
    is the surviving canonical entity; the losers are merged into it.
    """

    new_canonical: str
    entity_type: str
    winner_id: str
    loser_ids: List[str]
    member_surface_forms: List[str]
    total_source_docs: int
    members: List[MergeMember] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class MergePlan:
    """The dry-run output: every collision cluster of size ≥2."""

    clusters: List[MergeCluster]
    scope: str  # "global" or a notebook id
    total_active_entities: int

    @property
    def cluster_count(self) -> int:
        return len(self.clusters)


@dataclass
class MergeResult:
    """The outcome of applying a single cluster's merge."""

    new_canonical: str
    entity_type: str
    winner_id: str
    merged_loser_ids: List[str]
    relations_repointed: int
    aliases_created: int
    skipped: bool = False  # True when the cluster was already merged (no-op)


class RecanonicalizationService:
    """Plan + apply retroactive entity merges over the persisted KG."""

    def __init__(self, entity_repo: Optional[EntityRepository] = None) -> None:
        self.entity_repo = entity_repo or EntityRepository()

    # ------------------------------------------------------------------
    # Plan (dry-run — zero writes)
    # ------------------------------------------------------------------

    async def plan_merges(
        self, notebook_id: Optional[str] = None
    ) -> MergePlan:
        """Compute the dry-run merge plan. **No writes occur.**

        Loads active entities (optionally scoped to a notebook's sources),
        re-normalizes each ``canonical_name`` under the current
        ``normalize_entity_name``, groups by ``(new_canonical, entity_type)``,
        and returns the clusters with ≥2 members.

        Winner selection within a cluster (stable, deterministic):
            1. highest ``confidence``
            2. tie-break: most ``source_documents``
            3. tie-break: oldest ``created_at``
            4. final tie-break: lowest record id (total order)

        Args:
            notebook_id: When supplied, scope the plan to entities whose
                ``source_documents`` belong to that notebook (limits blast
                radius). ``None`` = global.

        Returns:
            A ``MergePlan``. Empty ``clusters`` when nothing collides.
        """
        source_ids: Optional[List[str]] = None
        if notebook_id is not None:
            source_ids = await self._sources_for_notebook(notebook_id)
            if not source_ids:
                logger.info(
                    "plan_merges: notebook={nb} has no sources — empty plan",
                    nb=notebook_id,
                )
                return MergePlan(
                    clusters=[],
                    scope=notebook_id,
                    total_active_entities=0,
                )

        rows = await self.entity_repo.list_active_entities(source_ids=source_ids)
        groups: Dict[Tuple[str, str], List[MergeMember]] = {}
        for row in rows:
            surface = row.get("canonical_name") or ""
            etype = row.get("entity_type") or ""
            new_canonical = normalize_entity_name(surface)
            if not new_canonical:
                # An entity that normalizes to empty (e.g. all-punctuation) is
                # not a safe merge key — leave it untouched.
                continue
            key = (new_canonical, etype)
            groups.setdefault(key, []).append(
                MergeMember(
                    id=str(row.get("id")),
                    surface_form=surface,
                    entity_type=etype,
                    confidence=float(row.get("confidence") or 0.0),
                    source_documents=list(row.get("source_documents") or []),
                    created_at=self._stringify(row.get("created_at")),
                )
            )

        clusters: List[MergeCluster] = []
        for (new_canonical, etype), members in groups.items():
            if len(members) < 2:
                continue
            ordered = self._rank_members(members)
            winner = ordered[0]
            losers = ordered[1:]
            total_docs = len(
                {d for m in members for d in m.source_documents}
            )
            clusters.append(
                MergeCluster(
                    new_canonical=new_canonical,
                    entity_type=etype,
                    winner_id=winner.id,
                    loser_ids=[m.id for m in losers],
                    member_surface_forms=[m.surface_form for m in members],
                    total_source_docs=total_docs,
                    members=ordered,
                )
            )

        # Stable output order: largest clusters first, then by canonical name.
        clusters.sort(key=lambda c: (-c.size, c.new_canonical))
        logger.info(
            "plan_merges: scope={scope} active={n} clusters={c}",
            scope=notebook_id or "global",
            n=len(rows),
            c=len(clusters),
        )
        return MergePlan(
            clusters=clusters,
            scope=notebook_id or "global",
            total_active_entities=len(rows),
        )

    @staticmethod
    def _rank_members(members: List[MergeMember]) -> List[MergeMember]:
        """Order members so the winner is first (see ``plan_merges`` rule)."""
        return sorted(
            members,
            key=lambda m: (
                -m.confidence,
                -len(m.source_documents),
                m.created_at or "9999",  # missing created_at sorts last (newest)
                m.id,
            ),
        )

    # ------------------------------------------------------------------
    # Apply (mutating — guarded behind apply_plan's dry_run default)
    # ------------------------------------------------------------------

    async def apply_merge(self, cluster: MergeCluster) -> MergeResult:
        """Apply one cluster's merge as a single logical operation.

        Steps (idempotent throughout):
            1. Re-point every ``relation`` edge from each loser onto the winner
               by ENTITY ID (parallel edges dedup'd by the repo helper). A
               pre-merge loser↔winner edge becomes a winner→winner self-loop
               under the re-point; such edges are **silently removed**, not
               recreated — a self-relation between the merged entity and the
               survivor is meaningless once they are the same node. This counts
               toward ``relations_repointed`` (the loser's edge IS consumed) but
               leaves no surviving edge. Likewise, when two losers each held an
               equivalent ``(in, out, relation_type)`` edge to the same third
               entity, the re-point collapses them to a single deduped edge on
               the winner.
            2. Merge the losers' provenance/scoring fields into the winner using
               the SAME semantics as ``upsert_entity`` (union source_documents /
               type_tags / provenance_chain, max confidence, overlay properties,
               union aliases) — reused via ``merge_into_winner``.
            3. CREATE an ``entity_alias`` row for each loser surface form →
               winner (the alias repo dedups on (entity, alias_text)).
            4. Soft-merge each loser: ``status='merged'`` + ``merged_into``.

        Idempotency: losers already at ``status='merged'`` are skipped entirely
        — no double-merge, no duplicate aliases, no relation churn.

        Returns:
            A ``MergeResult`` describing what changed (or ``skipped=True`` when
            the cluster was already fully merged).
        """
        # Idempotency gate: only operate on losers that are still active.
        active_losers = await self._active_loser_rows(cluster.loser_ids)
        if not active_losers:
            logger.info(
                "apply_merge: cluster '{c}' already merged — no-op",
                c=cluster.new_canonical,
            )
            return MergeResult(
                new_canonical=cluster.new_canonical,
                entity_type=cluster.entity_type,
                winner_id=cluster.winner_id,
                merged_loser_ids=[],
                relations_repointed=0,
                aliases_created=0,
                skipped=True,
            )

        # 1. Re-point relations (ID-based) for each still-active loser.
        relations_repointed = 0
        for loser in active_losers:
            relations_repointed += await self.entity_repo.repoint_relations(
                str(loser["id"]), cluster.winner_id
            )

        # 2. Provenance merge into the winner (reuses upsert merge math).
        await self.entity_repo.merge_into_winner(
            cluster.winner_id, active_losers
        )

        # 3. Alias rows for each loser surface form (alias repo dedups).
        aliases_created = 0
        for loser in active_losers:
            surface = loser.get("canonical_name") or ""
            if not surface:
                continue
            ok = await self.entity_repo.register_alias(
                cluster.winner_id,
                surface,
                # An identity-rule merge IS an exact match; the mechanism
                # travels in `method`. `match_type` is a closed vocabulary
                # (migration 80) and "retroactive_merge" is not in it.
                match_type="exact",
                similarity_score=1.0,
                method="recanonicalization",
            )
            if ok:
                aliases_created += 1

        # 4. Soft-merge each loser (status + merged_into).
        merged_ids: List[str] = []
        for loser in active_losers:
            lid = str(loser["id"])
            if await self.entity_repo.mark_merged(lid, cluster.winner_id):
                merged_ids.append(lid)

        logger.info(
            "apply_merge: winner={w} merged={n} relations={r} aliases={a}",
            w=cluster.winner_id,
            n=len(merged_ids),
            r=relations_repointed,
            a=aliases_created,
        )
        return MergeResult(
            new_canonical=cluster.new_canonical,
            entity_type=cluster.entity_type,
            winner_id=cluster.winner_id,
            merged_loser_ids=merged_ids,
            relations_repointed=relations_repointed,
            aliases_created=aliases_created,
            skipped=False,
        )

    async def apply_plan(
        self, plan: MergePlan, *, dry_run: bool = True
    ) -> List[MergeResult]:
        """Apply every cluster in a plan. **Hard ``dry_run=True`` default.**

        With ``dry_run=True`` (the default) NO writes occur — the method
        returns the would-be results with ``skipped=True`` so a caller can
        preview the batch without mutating the KG. A real apply requires the
        caller to pass ``dry_run=False`` explicitly.

        Returns:
            One ``MergeResult`` per cluster, in plan order.
        """
        if dry_run:
            logger.info(
                "apply_plan: DRY-RUN — {n} clusters previewed, 0 writes",
                n=plan.cluster_count,
            )
            return [
                MergeResult(
                    new_canonical=c.new_canonical,
                    entity_type=c.entity_type,
                    winner_id=c.winner_id,
                    merged_loser_ids=[],
                    relations_repointed=0,
                    aliases_created=0,
                    skipped=True,
                )
                for c in plan.clusters
            ]

        results: List[MergeResult] = []
        for cluster in plan.clusters:
            results.append(await self.apply_merge(cluster))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _active_loser_rows(
        self, loser_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Load full rows for losers still ``status='active'`` (idempotency)."""
        rows: List[Dict[str, Any]] = []
        for lid in loser_ids:
            entity = await self.entity_repo.get_entity(lid)
            if entity is None:
                continue
            if entity.status == "active":
                rows.append(
                    {
                        "id": lid,
                        "canonical_name": entity.canonical_name,
                        "entity_type": entity.entity_type,
                        "confidence": entity.confidence,
                        "source_documents": list(entity.source_documents),
                        "type_tags": list(entity.type_tags),
                        "provenance_chain": list(entity.provenance_chain),
                        "properties": dict(entity.properties),
                        "aliases": list(entity.aliases),
                    }
                )
        return rows

    async def _sources_for_notebook(self, notebook_id: str) -> List[str]:
        """Resolve source record ids linked to a notebook via ``reference``.

        Mirrors ``EntityRepository.list_orphans_with_status`` /
        ``NotebookMergeService._sources_for_notebook``: the source ↔ notebook
        edge lives on the ``reference`` RELATE table, not a column.
        """
        try:
            rows = await execute_query(
                "SELECT VALUE in FROM reference "
                "WHERE out = type::thing($notebook_id)",
                {"notebook_id": notebook_id},
            )
        except Exception as e:
            logger.error(
                f"_sources_for_notebook failed for '{notebook_id}': {e}"
            )
            return []
        return [str(r) for r in (rows or []) if r]

    @staticmethod
    def _stringify(value: Any) -> Optional[str]:
        """Coerce a DB datetime (or anything) to a sortable string, or None."""
        if value is None:
            return None
        return str(value)
