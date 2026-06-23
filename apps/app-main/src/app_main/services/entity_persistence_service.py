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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger
from ontology_manager.canonical_bridge import resolve_ontology_type
from shared.models.entity import Entity
from shared.utils.entity_type_aliases import (
    ENTITY_TYPE_ALIASES,
    resolve_residual_type,
)
from shared.utils.entity_type_aliases import (
    _normalize as _normalize_alias_key,
)
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.entity import EntityRepository

if TYPE_CHECKING:
    from ontology_manager.schema import Ontology

# The ``entity.entity_type`` field carries an ``ASSERT $value INSIDE [...]``
# constraint (enforced even though the table is SCHEMALESS). LLMs — especially
# local models like qwen2.5 — return free-form / capitalised types
# ("Location", "Concept", "ABBREVIATION") that violate it, so every upsert
# fails. Normalise to this canonical set at the persist boundary; unknown
# types map to "other" (the raw value is preserved in properties). NOTE: this
# enum lives only on the live DB, not in any migration (schema drift) — keep it
# in sync if the DB constraint changes.
_ALLOWED_ENTITY_TYPES = frozenset(
    {
        "person", "organization", "topic", "location", "concept", "event",
        "product", "scholarly_article", "creative_work", "periodical",
        "dataset", "grant", "research_project", "policy_document",
        "legislation", "government_organization", "administrative_area",
        "public_consultation", "social_profile",
        # L.3: domain canonical types. ``programme`` is the coarse projection for
        # Deal / RegioDeal / Programma / Project (the deals stack); ``technology``
        # for Technology / Technologie. The live DB accepts any ``entity_type``
        # string since migration 50 (B.8) dropped the ASSERT — this set is the
        # CODE-side enum guard, so adding the type here lets the L.2 aliases and
        # the L.1 bridge land on the real canonical instead of re-pinning to
        # ``other``. No migration is needed (migration 50 already relaxed the
        # live constraint).
        "programme", "technology",
        "other",
    }
)


def _normalize_entity_type(raw: Any) -> str:
    """Map an LLM-provided entity type onto the canonical enum (coarse projection).

    Lower-cases + underscores, then resolves short forms / NER tags + the curated
    EN+NL residual map (``ENTITY_TYPE_ALIASES``, L.2); anything still outside the
    allowed set becomes ``"other"`` so the ``entity_type`` ASSERT never rejects a
    write. This is the enum-GUARDED coarse value — it is used by the relation
    endpoint resolution (which type-filters a RELATE and so must pass a valid
    enum member) and as the ``entity_type`` projection for an aliased label.

    As of L.3, ``technology`` and ``programme`` ARE members of
    ``_ALLOWED_ENTITY_TYPES``, so the EN+NL aliases (``technologie`` →
    ``technology``; ``programma`` / ``regiodeal`` / ``deal`` / ``project`` →
    ``programme``) pass through this guard unchanged and land on the real
    canonical (no longer re-pinned to ``other``). Any other not-yet-in-enum value
    is still re-pinned to ``other`` while the rich label is preserved separately.
    """
    norm = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    norm = ENTITY_TYPE_ALIASES.get(norm, norm)
    return norm if norm in _ALLOWED_ENTITY_TYPES else "other"


@dataclass
class ResolvedType:
    """The outcome of resolving an extracted label to a persistable type.

    ``entity_type`` is always a member of ``_ALLOWED_ENTITY_TYPES`` (the B.8
    enum contract — the enum-guard re-pins anything outside it to ``other``).
    ``primary_type`` / ``type_tags`` carry the rich type:

    - L.1 bridge hit: the ontology type + its parent trail.
    - L.2 residual hit / fallback: the raw label, ALWAYS preserved when it
      carried a non-trivial signal (never the historical silent ``other`` that
      discarded it).

    ``is_noise`` flags an extraction-noise label (``ABBREVIATION`` / ``Amount``)
    so the caller can mark it; ``False`` for a bridge/alias hit or a
    genuine-unknown label.
    """

    entity_type: str
    primary_type: Optional[str] = None
    type_tags: List[str] = field(default_factory=list)
    is_noise: bool = False


def _resolve_entity_type(
    raw_label: Any, schemas: Optional[List["Ontology"]] = None
) -> ResolvedType:
    """Resolve an extracted label to a canonical type, preserving rich typing.

    Order:

    1. **Ontology bridge (L.1)** — if ``schemas`` are supplied and the label is
       a type in one of them, walk its ``parent_type`` chain to a schema.org
       base and map to the canonical enum. The original ontology label is
       preserved in ``primary_type`` and the parent trail in ``type_tags``.
    2. **Canonical passthrough (L.2-fix)** — if the normalized raw label is
       ITSELF a member of ``_ALLOWED_ENTITY_TYPES`` (e.g. a model that emits the
       bare canonical name ``Organization`` / ``Person`` / ``Government
       Organization``), return it directly as the ``entity_type`` with the raw
       label preserved in ``primary_type``. This restores the old
       ``_normalize_entity_type`` passthrough that L.2 dropped: without it a bare
       canonical name misses the alias map and silently falls to the ``concept``
       default (the regression). Runs BEFORE the alias map so a canonical name is
       never re-routed by a colliding alias.
    3. **EN+NL residual alias map (L.2)** — a curated alias hit returns the
       canonical target (enum-guarded) and preserves the raw label in
       ``primary_type``.
    4. **Non-silent unknown-label fallback (L.2)** — an unmapped label coarses to
       a sensible default BUT ALWAYS preserves the raw label in ``primary_type``
       / ``type_tags`` (the anti-flattening guarantee). Extraction noise
       (``ABBREVIATION`` / ``Amount``) is flagged ``is_noise`` but still
       preserved — it is never the historical silent ``other``.

    The bridge degrades gracefully: absent schemas (cross-batch / re-ingest),
    an unknown label, or an orphaned chain all return ``None`` from the bridge
    and fall through to the residual path — persistence never crashes (AC7).

    The returned ``entity_type`` is always enum-guarded here. As of L.3,
    ``technology`` / ``programme`` are valid enum members, so the L.2 aliases that
    target them land on the real canonical; any value still outside the enum is
    re-pinned to ``other`` while the rich label survives in ``primary_type``.
    """
    if schemas:
        try:
            resolution = resolve_ontology_type(str(raw_label or ""), schemas)
        except Exception as e:
            # A bridge failure must never break the hot persist path; degrade
            # to the residual fallback.
            logger.warning(f"canonical bridge failed for {raw_label!r}: {e}")
            resolution = None
        if resolution is not None:
            # Defensive: the bridge map only emits valid enum values, but pin
            # the B.8 contract — a non-enum canonical falls back to "other".
            canonical = (
                resolution.canonical
                if resolution.canonical in _ALLOWED_ENTITY_TYPES
                else "other"
            )
            return ResolvedType(
                entity_type=canonical,
                primary_type=resolution.ontology_type,
                type_tags=list(resolution.type_tags),
            )

    # (2) Canonical passthrough (L.2-fix): the model emitted a label that IS
    # already a valid canonical enum member ("Organization", "Person",
    # "Government Organization"). The L.1 bridge missed (it only matches ontology
    # types), and the L.2 alias map below would miss too (canonicals aren't
    # aliases), dumping it into the ``concept`` default — the regression. Restore
    # the old ``_normalize_entity_type`` passthrough: normalize with the SAME
    # rule the alias map uses (lower + space/dash -> underscore) so multi-word
    # canonicals resolve ("Government Organization" -> government_organization),
    # then return it directly while preserving the raw label in primary_type
    # (anti-flattening). Runs before the alias map so a canonical name is never
    # accidentally re-routed by a colliding alias.
    raw_str = str(raw_label or "").strip()
    norm = _normalize_alias_key(raw_label)
    if norm in _ALLOWED_ENTITY_TYPES:
        return ResolvedType(
            entity_type=norm,
            primary_type=raw_str or None,
            type_tags=[raw_str] if raw_str else [],
        )

    # (3)+(4) L.2 residual: curated EN+NL alias hit, else non-silent fallback.
    # As of L.3 the residual's ``technology`` / ``programme`` targets are valid
    # enum members and pass the guard; any other not-yet-in-enum value re-pins to
    # "other" while the residual's preserved label (primary_type / type_tags) is
    # kept intact — the rich signal is never lost.
    residual = resolve_residual_type(raw_label)
    entity_type = (
        residual.entity_type
        if residual.entity_type in _ALLOWED_ENTITY_TYPES
        else "other"
    )
    return ResolvedType(
        entity_type=entity_type,
        primary_type=residual.primary_type,
        type_tags=list(residual.type_tags),
        is_noise=residual.is_noise,
    )


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
        applicable_schemas: Optional[List["Ontology"]] = None,
    ) -> Dict[str, Any]:
        """Upsert entities and create relations in the knowledge graph.

        Args:
            source_id: Source record ID (for provenance tracking).
            entities: Filtered entity dicts (text, label, confidence, properties).
            relations: Filtered relation dicts (source_entity, target_entity, relation_type, confidence).
            merge_groups: Optional dedup merge groups for provenance.
            applicable_schemas: The ontologies applied for this source's
                extraction (L.1). When supplied, an extracted ontology label
                is bridged to its canonical ``entity_type`` and the rich type
                is preserved in ``primary_type`` / ``type_tags``. Threaded down
                from ``entity_extraction_service`` (already computed by
                ``detect_applicable_schemas`` — never re-detected here). Absent
                (cross-batch / re-ingest) the bridge degrades to the alias/enum
                path.

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

        # K.7a: map each entity's persisted canonical_name → its normalized
        # entity_type so the relation step can resolve an endpoint by
        # (canonical_name, entity_type) when both endpoints are in this batch.
        # The key is the SAME value used as ``canonical_name`` at upsert (the
        # raw ``text``), and the value is the SAME normalized type the upsert
        # writes (``_normalize_entity_type``) — so the type-filtered RELATE
        # lookup matches the row this batch persisted. A relation whose
        # endpoint name is NOT in this batch (cross-batch) stays unmapped and
        # falls back to name-only resolution (zero regression).
        # L.1: resolve through the same bridge-aware path the upsert uses so the
        # relation endpoint type matches the canonical the row persisted with.
        type_by_name: Dict[str, str] = {}
        for entity in entities:
            etext = entity.get("text", "")
            if not etext.strip():
                continue
            type_by_name[etext] = _resolve_entity_type(
                entity.get("label", "concept"), applicable_schemas
            ).entity_type

        # 1. Upsert entities
        for entity in entities:
            text = entity.get("text", "")
            raw_label = entity.get("label", "concept")
            # L.1: bridge the rich ontology label to its canonical enum value
            # AND preserve the original type. ``entity_type`` (coarse) drives
            # dedup / the K (name,type) guard; ``primary_type`` / ``type_tags``
            # carry the rich type the LLM emitted.
            resolved = _resolve_entity_type(raw_label, applicable_schemas)
            label = resolved.entity_type
            confidence = entity.get("confidence", 0.5)
            properties = entity.get("properties", {})

            if not text.strip():
                continue

            # Build entity properties for storage (exclude embedding to save space)
            stored_props = {
                k: v for k, v in properties.items()
                if k != "embedding" and v is not None
            }
            # Preserve the model's original (un-normalized) type for provenance
            # when it didn't map cleanly onto the canonical enum. The check is
            # against the COARSE alias projection (not the bridge's primary_type)
            # so a bridged ``Gemeente`` (-> administrative_area) still records its
            # raw label even though L.1 also stamps it into primary_type.
            if raw_label and str(raw_label).strip().lower() != _normalize_entity_type(raw_label):
                stored_props["raw_entity_type"] = raw_label

            # L.2: flag an extraction-noise label (ABBREVIATION / Amount) so a
            # reviewer / L.6 metric can tell genuine-unknown types apart from
            # mis-tagged mention-forms. The label is still preserved in
            # primary_type — the flag is provenance, not a discard.
            if resolved.is_noise:
                stored_props["non_type_label"] = True

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
                    # L.1: the rich ontology type (bridge). L.2: the raw label
                    # for an aliased / unmapped residual — always preserved when
                    # a non-trivial label exists (never silently dropped).
                    primary_type=resolved.primary_type,
                    type_tags=resolved.type_tags,
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

            # K.7a: resolve each endpoint's type. Prefer a type the relation
            # already carries from extraction (``source_type``/``target_type``);
            # otherwise fall back to the same-batch entity map. A type that is
            # not in the canonical enum is normalized so the type-filtered
            # lookup matches the persisted row; an unknown / unmapped endpoint
            # stays None and resolves by name only (back-compat fallback).
            src_type = rel.get("source_type") or type_by_name.get(src_text)
            tgt_type = rel.get("target_type") or type_by_name.get(tgt_text)
            src_type = _normalize_entity_type(src_type) if src_type else None
            tgt_type = _normalize_entity_type(tgt_type) if tgt_type else None

            try:
                # Field-name alignment to migration 39's SCHEMAFULL `relation`
                # table: lookups use `canonical_name`, and the edge carries
                # `source_documents` (array), not the legacy `source_id` scalar.
                #
                # K.7a: when an endpoint's type is known, additionally filter the
                # SELECT by ``entity_type`` so a cross-type homograph (a person
                # AND an org both named "BZK") resolves to the type-correct
                # entity. When the type is None, the filter degrades to the
                # original name-only ``LIMIT 1`` — zero regression for today's
                # working (unambiguous) cases. The ``IF array::len(...) > 0``
                # guard and the RELATE SET fields are unchanged.
                src_clause = "canonical_name = $src_name"
                if src_type is not None:
                    src_clause += " AND entity_type = $src_type"
                tgt_clause = "canonical_name = $tgt_name"
                if tgt_type is not None:
                    tgt_clause += " AND entity_type = $tgt_type"
                await execute_query(
                    f"""
                    LET $src = (SELECT id FROM entity
                        WHERE {src_clause} LIMIT 1);
                    LET $tgt = (SELECT id FROM entity
                        WHERE {tgt_clause} LIMIT 1);
                    IF array::len($src) > 0 AND array::len($tgt) > 0 THEN {{
                        LET $sid = $src[0].id;
                        LET $tid = $tgt[0].id;
                        RELATE $sid->relation->$tid SET
                            relation_type = $rel_type,
                            confidence = $confidence,
                            source_documents = [$source_id],
                            properties = $properties;
                    }} END
                    """,
                    {
                        "src_name": src_text,
                        "tgt_name": tgt_text,
                        "src_type": src_type,
                        "tgt_type": tgt_type,
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
