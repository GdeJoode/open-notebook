"""`mentions` (source → entity) edge regeneration service (Track U.2).

Wires the pure :func:`shared.retrieval.project_mentions_edges` into the DB:
loads the active entity → source mapping via the ``EntityRepository`` (the same
loader R.2 uses), runs the pure projection (R.2 weight × R.6 normalization), then
clears + RELATEs the ``mentions`` edge table idempotently. The ``mentions`` graph
is a DERIVED, regenerated view of ``entity.source_documents`` — re-running yields
an identical edge set with no duplicates, and the canonical ``entity`` / ``source``
rows are never mutated (only the ``mentions`` edge table is written).

This adds NOTHING to search (R.2 already computes the same shared-entity signal
on the fly); its value is making the document↔entity graph real —
traversable / drawable (U.4) / exportable (U.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger
from shared.retrieval.kg_source_scorer import EntityRecord
from shared.retrieval.mentions_projection import (
    MentionEdge,
    project_mentions_edges,
)

from surrealdb_service.repositories import EntityRepository, SourceRepository


@dataclass(frozen=True)
class RegenerateMentionsResult:
    """Outcome of a ``mentions`` regeneration (telemetry + acceptance).

    Attributes:
        cleared: Edges removed by the clear step (the prior edge set size).
        created: Edges RELATEd this run (the new edge set size).
        failed: Edges whose RELATE failed (expected 0; non-zero is a partial
            write worth surfacing).
        active_entities: Active entity rows the projection ran over.
        emitted_concepts: Concepts surviving R.6 (post case/type merge + drop).
        n_sources: Corpus source count used for the IDF rarity denominator.
        min_weight: The per-edge weight cutoff applied.
    """

    cleared: int
    created: int
    failed: int
    active_entities: int
    emitted_concepts: int
    n_sources: int
    min_weight: float


class MentionsProjectionService:
    """Regenerate + read the materialized ``mentions`` document↔entity graph."""

    def __init__(
        self,
        entity_repo: EntityRepository,
        source_repo: SourceRepository,
    ):
        self.entity_repo = entity_repo
        self.source_repo = source_repo

    @staticmethod
    def _to_entity_records(
        rows: List[Dict[str, Any]],
    ) -> List[EntityRecord]:
        """Project raw active-entity rows into the pure projection's input type.

        Identical projection to :meth:`KGRetrievalService._to_entity_records`
        so the materialized edges weight on the SAME inputs the search signal
        scores on.
        """
        records: List[EntityRecord] = []
        for row in rows or []:
            eid = row.get("id")
            if eid is None:
                continue
            records.append(
                EntityRecord(
                    entity_id=str(eid),
                    canonical_name=str(row.get("canonical_name") or ""),
                    entity_type=str(row.get("entity_type") or ""),
                    source_documents=tuple(
                        str(s) for s in (row.get("source_documents") or [])
                    ),
                )
            )
        return records

    async def _true_source_count(self) -> Optional[int]:
        """Fetch the real corpus source count for the IDF denominator.

        The total number of ``source`` rows (not the sources-with-active-entities
        subset), so the IDF rarity matches R.2 exactly. ``None`` on error / zero
        so the pure projection falls back to its observed-sources default.
        """
        try:
            total = await self.source_repo.count()
            return total if total and total > 0 else None
        except Exception as e:
            logger.warning("mentions IDF source-count failed, using fallback: {}", e)
            return None

    async def project_edges(
        self,
        *,
        drop_singletons: bool = True,
        min_weight: float = 0.0,
        named_only: bool = False,
        n_sources: Optional[int] = None,
    ) -> tuple[List[MentionEdge], int]:
        """Compute the projected ``mentions`` edges WITHOUT writing anything.

        The read-only half of regeneration — used both by :meth:`regenerate`
        (which then persists) and by a dry-run count (gated live-write workflow:
        compute the edge set, report its size, write nothing).

        Returns:
            ``(edges, n_sources)`` — the projected edge list and the source count
            used for the IDF (so a dry-run can report the exact figure a live
            regenerate would persist).
        """
        entity_rows = await self.entity_repo.load_active_entity_source_map()
        records = self._to_entity_records(entity_rows)
        if n_sources is None:
            n_sources = await self._true_source_count()

        edges, stats = project_mentions_edges(
            records,
            n_sources=n_sources,
            drop_singletons=drop_singletons,
            min_weight=min_weight,
            named_only=named_only,
        )
        effective_n = (
            int(n_sources)
            if n_sources is not None
            else len({s for r in records for s in r.source_documents})
        )
        logger.debug(
            "mentions projection: {rows} active rows -> {concepts} concepts "
            "-> {edges} edges (min_weight={mw}, n_sources={n})",
            rows=stats.input_entities,
            concepts=stats.emitted_concepts,
            edges=stats.edges,
            mw=stats.min_weight,
            n=effective_n,
        )
        return edges, effective_n

    async def regenerate(
        self,
        *,
        drop_singletons: bool = True,
        min_weight: float = 0.0,
        named_only: bool = False,
    ) -> RegenerateMentionsResult:
        """Clear + rebuild the ``mentions`` edge table from the projection.

        Idempotent: clears every existing ``mentions`` edge then RELATEs the
        freshly-projected set, so a re-run yields the SAME edge set with zero
        duplicates. Reversible — ``mentions`` is a derived view; the canonical
        ``entity.source_documents`` it projects from is untouched, so the table
        can be cleared/rebuilt at will.

        Args:
            drop_singletons: Drop df==1 concepts (the mandatory R.6 legibility
                step; the 466→67 filter). Leave True for the drawable graph.
            min_weight: Per-edge weight cutoff (default 0.0 — keep all df≥2
                edges, carry weight as a render attribute).
            named_only: The 0.3 named-only preset (overview skeleton).

        Returns:
            A :class:`RegenerateMentionsResult` with the cleared/created counts.
        """
        edges, n_sources = await self.project_edges(
            drop_singletons=drop_singletons,
            min_weight=min_weight,
            named_only=named_only,
        )

        cleared = await self.entity_repo.clear_mentions()

        created = 0
        failed = 0
        for edge in edges:
            try:
                ok = await self.entity_repo.relate_mention(
                    edge.source_id,
                    edge.entity_id,
                    weight=edge.weight,
                    concept_name=edge.concept_name,
                    concept_type=edge.concept_type,
                    document_frequency=edge.document_frequency,
                )
            except Exception as e:  # one bad edge must not abort the rebuild
                logger.warning(
                    "relate_mention raised for {s}->{t}: {e}",
                    s=edge.source_id,
                    t=edge.entity_id,
                    e=e,
                )
                ok = False
            if ok:
                created += 1
            else:
                failed += 1

        # Recompute concept/active counts for the result without a second DB hit:
        # they are stable across project_edges, so re-derive from a cheap stats
        # pass would duplicate work — instead approximate from the edge set.
        emitted_concepts = len({(e.entity_id) for e in edges})
        active_entities = len(
            await self.entity_repo.load_active_entity_source_map()
        )

        effective_min_weight = max(min_weight, 0.3) if named_only else min_weight
        logger.info(
            "mentions regenerated: cleared={cleared} created={created} "
            "failed={failed} (min_weight={mw})",
            cleared=cleared,
            created=created,
            failed=failed,
            mw=effective_min_weight,
        )
        return RegenerateMentionsResult(
            cleared=cleared,
            created=created,
            failed=failed,
            active_entities=active_entities,
            emitted_concepts=emitted_concepts,
            n_sources=n_sources,
            min_weight=effective_min_weight,
        )

    async def get_document_entity_edges(
        self,
        *,
        min_weight: float = 0.0,
        source_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch the materialized document↔entity edges (feeds U.4 viz / export).

        Returns the persisted ``mentions`` edges with their endpoints + weight +
        per-edge "why" (the shared concept name/type/df), ordered by weight
        descending. ``min_weight`` lets the UI slider collapse to the named-only
        skeleton; ``source_id`` scopes to one document's neighbourhood.
        """
        return await self.entity_repo.load_mentions_edges(
            min_weight=min_weight, source_id=source_id
        )
