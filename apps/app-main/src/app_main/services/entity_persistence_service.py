"""
Service for persisting filtered extraction results to the knowledge graph.

Takes a FilteredResult from the entity-filtering pipeline and upserts
entities into the ``entity`` table and relations into the ``relation``
table in SurrealDB, making them visible on the Knowledge Graph page.

Phase B.1a (2026-06): entity upsert was historically a raw-SQL block that
wrote to ``name`` / ``weight`` / ``source_ids`` — drift from the SCHEMAFULL
schema declared in migration 39 (``canonical_name`` / no weight /
``source_documents``). The drift was caught by Phase B.0's roundtrip
canaries; this service now routes entity writes through
``EntityRepository.upsert_entity`` so the canonical schema and the
write-path stay aligned.
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from shared.models.entity import Entity
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.entity import EntityRepository


class EntityPersistenceService:
    """Persists filtered entities and relations to the KG tables."""

    def __init__(
        self, entity_repository: Optional[EntityRepository] = None
    ) -> None:
        """Initialize with an optional pre-built repository.

        Args:
            entity_repository: Inject for tests; defaults to a fresh
                ``EntityRepository`` using the global SurrealDB config.
        """
        self._entity_repo = entity_repository or EntityRepository()

    async def persist_match_candidates(
        self,
        source_id: str,
        candidates: List[Dict[str, Any]],
    ) -> int:
        """Store match decisions in the resolution_log table.

        Args:
            source_id: Source record ID.
            candidates: List of MatchCandidate dicts.

        Returns:
            Number of candidates stored.
        """
        stored = 0
        for c in candidates:
            try:
                await execute_query(
                    """
                    CREATE resolution_log SET
                        entity_a_text = $entity_a_text,
                        entity_b_text = $entity_b_text,
                        entity_a_label = $entity_a_label,
                        entity_b_label = $entity_b_label,
                        match = $match,
                        confidence = $confidence,
                        match_method = $match_method,
                        match_reasoning = $match_reasoning,
                        iterations = $iterations,
                        source_document_a = $source_document_a,
                        source_document_b = $source_document_b,
                        source_section_a = $source_section_a,
                        source_section_b = $source_section_b,
                        matched_by_model = $matched_by_model,
                        status = $status,
                        source_id = $source_id,
                        match_timestamp = time::now()
                    """,
                    {
                        "entity_a_text": c.get("entity_a_text", ""),
                        "entity_b_text": c.get("entity_b_text", ""),
                        "entity_a_label": c.get("entity_a_label", "UNKNOWN"),
                        "entity_b_label": c.get("entity_b_label", "UNKNOWN"),
                        "match": c.get("match", False),
                        "confidence": c.get("confidence", 0.0),
                        "match_method": c.get("match_method", "unknown"),
                        "match_reasoning": c.get("match_reasoning", ""),
                        "iterations": c.get("iterations", 1),
                        "source_document_a": c.get("source_document_a"),
                        "source_document_b": c.get("source_document_b"),
                        "source_section_a": c.get("source_section_a"),
                        "source_section_b": c.get("source_section_b"),
                        "matched_by_model": c.get("matched_by_model"),
                        "status": c.get("status", "pending"),
                        "source_id": source_id,
                    },
                )
                stored += 1
            except Exception as e:
                logger.warning(f"Failed to store match candidate: {e}")
        logger.info(f"Stored {stored} match candidates for source {source_id}")
        return stored

    async def persist_filtered_result(
        self,
        source_id: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        merge_groups: List[List[str]] | None = None,
        match_candidates: List[Dict[str, Any]] | None = None,
        extraction_method: str = "llm",
        extraction_model: str | None = None,
    ) -> Dict[str, Any]:
        """Upsert entities and create relations in the knowledge graph.

        Args:
            source_id: Source record ID (for provenance tracking).
            entities: Filtered entity dicts (text, label, confidence, properties).
            relations: Filtered relation dicts (source_entity, target_entity, relation_type, confidence).
            merge_groups: Optional dedup merge groups for provenance.

        Returns:
            Dict with ``entities_upserted`` and ``relations_created`` counts.
        """
        entities_upserted = 0
        entities_failed = 0
        relations_created = 0

        # Build merge group lookup: entity text → group members
        merge_lookup: Dict[str, List[str]] = {}
        if merge_groups:
            for group in merge_groups:
                for member in group:
                    merge_lookup[member] = group

        # 1. Upsert entities
        for entity in entities:
            text = entity.get("text", "")
            label = entity.get("label", "UNKNOWN")
            confidence = entity.get("confidence", 0.5)
            properties = entity.get("properties", {})

            if not text.strip():
                continue

            # Build entity properties for storage (exclude embedding to save space)
            stored_props = {
                k: v for k, v in properties.items()
                if k != "embedding" and v is not None
            }

            # Add merge history if available
            if text in merge_lookup:
                stored_props["merged_from"] = merge_lookup[text]

            # Record the model that produced this entity (provenance). The
            # `entity` table is SCHEMALESS in prod, so an extra prop is safe;
            # `extraction_method` is a first-class Entity field.
            if extraction_model:
                stored_props["extraction_model"] = extraction_model

            try:
                # Route through EntityRepository.upsert_entity so the write
                # uses the canonical schema field names (canonical_name,
                # source_documents, embedding=[]) declared in migration 39+44.
                # See Phase B.1a notes at top of this module for context.
                #
                # B.8a: thread the real extraction_method (was silently
                # defaulting to "llm" for every path — provenance bug).
                entity_model = Entity(
                    canonical_name=text,
                    entity_type=label,
                    confidence=confidence,
                    source_documents=[source_id],
                    properties=stored_props,
                    embedding=[],
                    extraction_method=extraction_method,
                )
                await self._entity_repo.upsert_entity(entity_model)
                entities_upserted += 1
            except Exception as e:
                # B.8a: do NOT silently swallow. Count the failure and log at
                # ERROR; a fully-failed batch raises below so the caller can't
                # report a successful extraction that wrote nothing.
                entities_failed += 1
                logger.error(f"Failed to upsert entity '{text}': {e}")

        # 2. Create relations
        for rel in relations:
            src_text = rel.get("source_entity", "")
            tgt_text = rel.get("target_entity", "")
            rel_type = rel.get("relation_type", "RELATED")
            confidence = rel.get("confidence", 0.5)
            properties = rel.get("properties", {})

            if not src_text or not tgt_text:
                continue

            try:
                # Field-name alignment to migration 39's SCHEMAFULL `relation`
                # table: lookups use `canonical_name`, and the edge carries
                # `source_documents` (array), not the legacy `source_id` scalar.
                await execute_query(
                    """
                    LET $src = (SELECT id FROM entity
                        WHERE canonical_name = $src_name LIMIT 1);
                    LET $tgt = (SELECT id FROM entity
                        WHERE canonical_name = $tgt_name LIMIT 1);
                    IF array::len($src) > 0 AND array::len($tgt) > 0 THEN {
                        RELATE $src[0].id->relation->$tgt[0].id SET
                            relation_type = $rel_type,
                            confidence = $confidence,
                            source_documents = [$source_id],
                            properties = $properties;
                    } END
                    """,
                    {
                        "src_name": src_text,
                        "tgt_name": tgt_text,
                        "rel_type": rel_type,
                        "confidence": confidence,
                        "source_id": source_id,
                        "properties": properties,
                    },
                )
                relations_created += 1
            except Exception as e:
                logger.warning(
                    f"Failed to create relation {src_text} -> {tgt_text}: {e}"
                )

        # 3. Store match candidates in resolution_log (stap 2C)
        candidates_stored = 0
        if match_candidates:
            candidates_stored = await self.persist_match_candidates(
                source_id, match_candidates
            )

        # B.8a: if entities were attempted and EVERY one errored, that is a hard
        # failure, not a silent success — surface it so callers/telemetry see the
        # extraction wrote nothing rather than reporting 0 quietly. (Entities
        # skipped for empty text are not failures, so this keys on
        # ``entities_failed``, not on a zero upsert count.)
        if entities_failed > 0 and entities_upserted == 0:
            raise RuntimeError(
                f"persist_filtered_result wrote 0 of {len(entities)} entities "
                f"for source {source_id} (all {entities_failed} upserts failed) — "
                f"see ERROR logs above"
            )

        if entities_failed:
            logger.error(
                f"persist_filtered_result: {entities_failed} of "
                f"{len(entities)} entities failed to upsert for source "
                f"{source_id} ({entities_upserted} succeeded)"
            )

        logger.info(
            f"Persisted to KG: {entities_upserted} entities, "
            f"{relations_created} relations, "
            f"{candidates_stored} match candidates for source {source_id}"
        )

        return {
            "entities_upserted": entities_upserted,
            "entities_failed": entities_failed,
            "relations_created": relations_created,
            "candidates_stored": candidates_stored,
        }
