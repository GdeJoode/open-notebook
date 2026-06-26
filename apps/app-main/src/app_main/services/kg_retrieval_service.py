"""KG-proximity source retrieval service (Track R.2).

Wires the pure :mod:`shared.retrieval.kg_source_scorer` into the DB: loads the
active entity → source mapping (and optionally active relations) via the
``EntityRepository``, runs the pure scorer, then hydrates the ranked source ids
to ``{id, title, score, explanation}`` for the API — the same row shape as the
R.1 dense ``/related`` endpoint, plus the per-pair explanation lineage.

This is the **KG signal** of the hybrid retriever (R.3 fuses it with the dense
R.1 signal). It is deliberately a separate scorer, NOT dense similarity: it
ranks by SHARED KNOWLEDGE (shared active canonical entities, weighted by type
salience × rarity), surfacing *which* entities drove the link as the "why
matched" provenance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger
from shared.retrieval.kg_source_scorer import (
    EntityRecord,
    RelationRecord,
    score_related_sources,
)

from surrealdb_service.repositories import EntityRepository, SourceRepository


class KGRetrievalService:
    """Rank sources related to a given source by KG proximity (R.2)."""

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
        """Project raw active-entity rows into the pure scorer's input type."""
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

    @staticmethod
    def _to_relation_records(
        rows: List[Dict[str, Any]],
    ) -> List[RelationRecord]:
        """Project raw active-relation rows into the pure scorer's edge type."""
        records: List[RelationRecord] = []
        for row in rows or []:
            in_id, out_id = row.get("in"), row.get("out")
            if in_id is None or out_id is None:
                continue
            records.append(
                RelationRecord(
                    in_id=str(in_id),
                    out_id=str(out_id),
                    relation_type=str(row.get("relation_type") or ""),
                )
            )
        return records

    async def find_related_by_kg(
        self,
        source_id: str,
        k: int,
        *,
        expand_relations: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return the top-``k`` sources related to ``source_id`` by KG proximity.

        Ranks other sources by shared ACTIVE canonical entities (weighted by
        type salience × inverse-source-frequency rarity), each result carrying
        the shared entities that drove its score. The query source and any
        zero-overlap source are excluded; an empty list means the source shares
        no active entities with any other source.

        Args:
            source_id: The source to find related sources for.
            k: Max number of related sources to return (the highest-scoring).
            expand_relations: Opt-in 1-hop relation expansion. OFF by default —
                the 1-hop path runs through the not-yet-trimmed duplicate
                predicate noise (R.6), so shared-entity scoring is the reliable
                default and relation proximity is gated behind this flag.

        Returns:
            A list of ``{"id", "title", "score", "explanation"}`` dicts, ranked
            by score descending (deterministic ``id`` tie-break). ``explanation``
            is a list of ``{"entity_id", "name", "type", "document_frequency",
            "weight", "via_relation"}`` — the per-pair "why matched" lineage.
        """
        entity_rows = await self.entity_repo.load_active_entity_source_map()
        if not entity_rows:
            return []
        entities = self._to_entity_records(entity_rows)

        relations: Optional[List[RelationRecord]] = None
        if expand_relations:
            relation_rows = await self.entity_repo.load_active_relations()
            relations = self._to_relation_records(relation_rows)

        scored = score_related_sources(
            source_id,
            entities,
            relations=relations,
            expand_relations=expand_relations,
            limit=k,
        )
        if not scored:
            return []

        titles = await self.source_repo.get_titles_by_ids(
            [s.source_id for s in scored]
        )

        results: List[Dict[str, Any]] = []
        for s in scored:
            results.append(
                {
                    "id": s.source_id,
                    "title": titles.get(s.source_id),
                    "score": s.score,
                    "explanation": [
                        {
                            "entity_id": c.entity_id,
                            "name": c.canonical_name,
                            "type": c.entity_type,
                            "document_frequency": c.document_frequency,
                            "weight": c.weight,
                            "via_relation": c.via_relation,
                        }
                        for c in s.contributions
                    ],
                }
            )
        logger.debug(
            "find_related_by_kg: source={src} k={k} expand={ex} -> {n} results",
            src=source_id,
            k=k,
            ex=expand_relations,
            n=len(results),
        )
        return results
