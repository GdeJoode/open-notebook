"""Cross-notebook graph merge service (Phase B.6).

Merges entities and relations from N source notebooks into a target
notebook. The aggregation rules reuse B.1e's per-pass merge semantics
(``pipelines/ontology-extraction/multi_schema_orchestrator._merge_results``)
but apply them across notebooks rather than across extraction passes.

Algorithm
=========

1. For each source notebook: list entities (via ``source_documents``
   linkage) + their incident relations.
2. Bucket entities by ``normalize_entity_name(canonical_name)``. Each
   bucket aggregates:

   - ``type_tags``: union of all contributing labels (dedup preserving
     first-seen order).
   - ``primary_type``: label from the highest-confidence contributor;
     tiebreak by first-seen.
   - ``confidence``: max across contributors.
   - ``source_documents`` / ``provenance_chain``: union.

3. If ``type_match_required=True`` and a normalized name has DISJOINT
   ``type_tags`` across contributors (e.g. "Smith" appears as both
   ``Organization`` and ``Person``), record a conflict row and do NOT
   merge — the operator must reconcile manually.
4. Bucket relations by ``(normalize(source_text), normalize(target_text),
   relation_type)``. Max-confidence wins.
5. Write merged entities + relations into the target notebook via
   ``EntityRepository.upsert_entity`` (which is itself idempotent — same
   ``(canonical_name, entity_type)`` updates in place).

Idempotency
===========

Re-running the same merge produces ``entities_merged=0`` and
``relations_created=0``. The accountancy counters reflect *what changed*,
not what was inspected — the upsert call still runs, but produces
unchanged rows. Achieved by SEMANTIC CONTENT comparison: before each
upsert, we read the existing row (if any) and compute the would-be
merged ``Entity`` model. If the merged content matches the existing row
along ``(canonical_name, entity_type, type_tags, primary_type,
confidence, source_documents, properties)`` (with set semantics for
list-valued fields so reordering is ignored), the counter is NOT
incremented. Per-relation idempotency uses an existence probe that the
service also relies on as a "would-this-write-anything" signal.

The earlier approach (``updated_at`` snapshot) was abandoned because
``EntityRepository.upsert_entity`` unconditionally sets
``updated_at = time::now()`` on every UPDATE. A timestamp delta is
therefore present on EVERY second run, which would inflate the counter
to "all contributed entities" instead of zero.

Telemetry
=========

Emits one ``notebook_merge`` metric per call carrying entities, relations,
conflict counts. Wired through ``shared.services.metrics.record_metric``
(B.4 metrics layer).

Why a fresh service vs reusing ``EntityPersistenceService``
==========================================================

``EntityPersistenceService.persist_filtered_result`` writes a SINGLE
source's filtered output. The merge service operates across N notebooks
and needs the cross-notebook aggregation step (type-tag union,
conflict detection) BEFORE it hits the upsert path. Putting the
aggregation in a separate service keeps the persistence service
focused on the single-source path and avoids coupling B.6 to the B.1f
write-path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger
from shared.models.entity import Entity
from shared.services.metrics import record_metric
from shared.utils.name_normalizer import normalize_entity_name
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.entity import EntityRepository


@dataclass
class NotebookMergeConflict:
    """One unresolved type collision surfaced by a merge run.

    Emitted when ``type_match_required=True`` and the same normalized
    name appears with disjoint ``type_tags`` across source notebooks.
    The merge service drops the conflicting entity from the write set
    and records the row so the UI can ask the operator how to proceed.
    """

    normalized_name: str
    conflicting_type_tags: List[str]
    source_notebook_ids: List[str]


@dataclass
class NotebookMergeReport:
    """Outcome of one ``merge_notebooks`` call.

    Counters are *net changes* — re-running the same merge yields zeros.
    See module docstring on idempotency.
    """

    entities_merged: int
    relations_created: int
    conflicts: List[NotebookMergeConflict]
    source_notebook_ids: List[str]
    target_notebook_id: str
    dry_run: bool = False


@dataclass
class _MergedEntityAccumulator:
    """In-process aggregation of one normalized name across notebooks.

    Mirrors ``MergedEntity`` in the B.1e orchestrator but tracks the
    contributing source-notebook ids so conflict rows can name them.
    """

    best_text: str
    best_label: str
    best_confidence: float
    type_tags: List[str] = field(default_factory=list)
    source_documents: List[str] = field(default_factory=list)
    source_notebook_ids: List[str] = field(default_factory=list)
    provenance_chain: List[Any] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    def add(
        self,
        text: str,
        label: str,
        confidence: float,
        source_documents: Sequence[str],
        provenance_chain: Sequence[Any],
        properties: Dict[str, Any],
        source_notebook_id: str,
    ) -> None:
        """Fold another notebook's view of the same entity into the merge."""
        if label and label not in self.type_tags:
            self.type_tags.append(label)
        if source_notebook_id and source_notebook_id not in self.source_notebook_ids:
            self.source_notebook_ids.append(source_notebook_id)
        for doc in source_documents:
            if doc and doc not in self.source_documents:
                self.source_documents.append(doc)
        for entry in provenance_chain:
            if entry not in self.provenance_chain:
                self.provenance_chain.append(entry)
        # Property merge: new keys win, existing keys retained — matches
        # ``EntityRepository.upsert_entity`` semantics so the in-memory
        # aggregation and the DB merge agree.
        for k, v in properties.items():
            if k not in self.properties and v is not None:
                self.properties[k] = v

        if confidence > self.best_confidence:
            self.best_confidence = confidence
            self.best_label = label
            self.best_text = text


class NotebookMergeService:
    """Orchestrates the cross-notebook entity / relation merge."""

    def __init__(
        self,
        entity_repository: Optional[EntityRepository] = None,
    ) -> None:
        """Initialize with an optional pre-built repository.

        Args:
            entity_repository: Inject for tests; defaults to a fresh
                ``EntityRepository`` using the global SurrealDB config.
        """
        self._entity_repo = entity_repository or EntityRepository()

    async def merge_notebooks(
        self,
        source_notebook_ids: List[str],
        target_notebook_id: str,
        type_match_required: bool = True,
        dry_run: bool = False,
    ) -> NotebookMergeReport:
        """Merge entities/relations from N source notebooks into a target.

        See the module docstring for the algorithm + idempotency rules.

        Args:
            source_notebook_ids: Record IDs of the source notebooks
                (e.g. ``"notebook:abc"``). Empty list → empty report.
            target_notebook_id: Record ID of the target notebook. Receives
                the merged entities + relations.
            type_match_required: When True (default), a normalized name
                with disjoint ``type_tags`` across notebooks is recorded
                as a conflict and skipped. When False, the disjoint
                labels are unioned into ``type_tags`` and the entity is
                merged anyway (the highest-confidence label still wins
                ``primary_type``).
            dry_run: When True, perform the aggregation but skip all
                writes. The report carries the would-be counters.

        Returns:
            ``NotebookMergeReport`` with the accountancy counters,
            conflicts list, and the source/target ids echoed back for
            the UI.
        """
        if not source_notebook_ids:
            logger.info(
                "merge_notebooks: empty source list — returning empty report"
            )
            return NotebookMergeReport(
                entities_merged=0,
                relations_created=0,
                conflicts=[],
                source_notebook_ids=[],
                target_notebook_id=target_notebook_id,
                dry_run=dry_run,
            )

        # ----------------------------------------------------------------
        # Phase 1: load entities + relations per source notebook.
        # ----------------------------------------------------------------
        per_notebook_entities: Dict[str, List[Dict[str, Any]]] = {}
        per_notebook_relations: Dict[str, List[Dict[str, Any]]] = {}
        for nb_id in source_notebook_ids:
            per_notebook_entities[nb_id] = await self._list_entities_for_notebook(
                nb_id
            )
            per_notebook_relations[nb_id] = await self._list_relations_for_notebook(
                nb_id
            )

        # ----------------------------------------------------------------
        # Phase 2: aggregate entities by normalized name.
        # ----------------------------------------------------------------
        accumulators: Dict[str, _MergedEntityAccumulator] = {}
        insertion_order: List[str] = []
        # Map raw text → normalized key so relation-endpoint rewrites
        # can find the merged canonical surface form.
        text_to_key: Dict[str, str] = {}

        for nb_id in source_notebook_ids:
            for row in per_notebook_entities[nb_id]:
                text = row.get("canonical_name") or ""
                label = row.get("entity_type") or "UNKNOWN"
                key = normalize_entity_name(text)
                if not key:
                    # Defensive: an entity with no normalisable name is
                    # noise we can't address. Drop.
                    continue
                text_to_key[text] = key
                conf = float(row.get("confidence") or 0.0)
                source_docs = list(row.get("source_documents") or [])
                # Pre-existing ``type_tags`` on the row should also flow
                # into the aggregated tag list (an entity merged in a
                # previous B.6 run already has multi-type tags).
                existing_tags = list(row.get("type_tags") or [])
                props = dict(row.get("properties") or {})
                prov = list(row.get("provenance_chain") or [])
                if key not in accumulators:
                    accumulators[key] = _MergedEntityAccumulator(
                        best_text=text,
                        best_label=label,
                        best_confidence=conf,
                        type_tags=list(dict.fromkeys([label, *existing_tags])),
                        source_documents=list(source_docs),
                        source_notebook_ids=[nb_id],
                        provenance_chain=list(prov),
                        properties=dict(props),
                    )
                    insertion_order.append(key)
                else:
                    acc = accumulators[key]
                    for tag in existing_tags:
                        if tag and tag not in acc.type_tags:
                            acc.type_tags.append(tag)
                    acc.add(
                        text=text,
                        label=label,
                        confidence=conf,
                        source_documents=source_docs,
                        provenance_chain=prov,
                        properties=props,
                        source_notebook_id=nb_id,
                    )

        # ----------------------------------------------------------------
        # Phase 3: detect type-collision conflicts (per AC #3 + Q-B-5).
        # ----------------------------------------------------------------
        conflicts: List[NotebookMergeConflict] = []
        skipped_keys: set[str] = set()
        if type_match_required:
            for key in insertion_order:
                acc = accumulators[key]
                # A "collision" is when a single normalized name was
                # contributed by multiple notebooks with NO overlap in
                # type tags. A multi-tag entity from a single notebook
                # is not a conflict — that's just multi-type tagging.
                if len(acc.source_notebook_ids) < 2:
                    continue
                # Build per-notebook tag sets to test disjointness.
                # Reconstruct from the raw rows so we know which tags
                # came from which notebook.
                per_nb_tags = self._tags_by_notebook(
                    key, source_notebook_ids, per_notebook_entities
                )
                if self._tags_are_disjoint(per_nb_tags):
                    conflicts.append(
                        NotebookMergeConflict(
                            normalized_name=key,
                            conflicting_type_tags=list(acc.type_tags),
                            source_notebook_ids=list(acc.source_notebook_ids),
                        )
                    )
                    skipped_keys.add(key)

        # ----------------------------------------------------------------
        # Phase 4: write entities into the target notebook.
        # ----------------------------------------------------------------
        target_source_ids = await self._sources_for_notebook(target_notebook_id)
        entities_changed = 0
        for key in insertion_order:
            if key in skipped_keys:
                continue
            acc = accumulators[key]
            # Include target_notebook's sources in the entity's source_documents
            # bag so a follow-up ``list_orphans_with_status(target)`` query can
            # discover them. We add any target-bound sources that aren't
            # already in the bag.
            merged_sources = list(acc.source_documents)
            for s in target_source_ids:
                if s not in merged_sources:
                    merged_sources.append(s)

            entity_model = Entity(
                canonical_name=acc.best_text,
                entity_type=acc.best_label,
                confidence=acc.best_confidence,
                source_documents=merged_sources,
                properties=acc.properties,
                provenance_chain=acc.provenance_chain,
                type_tags=acc.type_tags,
                primary_type=acc.best_label,
                embedding=[],
            )

            existing_before = await self._find_canonical_entity(
                acc.best_text, acc.best_label
            )

            if dry_run:
                # Same "would this write change anything" question as the
                # commit path, but skip the upsert call. Compare the
                # would-be merged entity against the existing row.
                if existing_before is None or not self._entity_matches(
                    entity_model, existing_before
                ):
                    entities_changed += 1
                continue

            try:
                await self._entity_repo.upsert_entity(entity_model)
            except Exception as e:
                logger.warning(
                    f"merge_notebooks: upsert failed for '{acc.best_text}' "
                    f"({acc.best_label}): {e}"
                )
                continue

            # Semantic comparison: a fresh CREATE always counts; an
            # UPDATE only counts when the merged content actually
            # differs from the pre-write row. ``updated_at`` is
            # intentionally NOT in the comparison — the repo bumps it on
            # every UPDATE, so it would force a false-positive on
            # idempotent re-runs.
            if existing_before is None:
                entities_changed += 1
            elif not self._entity_matches(entity_model, existing_before):
                entities_changed += 1

        # ----------------------------------------------------------------
        # Phase 5: aggregate + write relations.
        # ----------------------------------------------------------------
        # Dedup key: (norm_src, norm_tgt, relation_type). Max-confidence
        # wins — same as B.1e's relation merge.
        relation_best: Dict[tuple, Dict[str, Any]] = {}
        relation_insertion_order: List[tuple] = []
        for nb_id in source_notebook_ids:
            for row in per_notebook_relations[nb_id]:
                src_text = row.get("source_text") or ""
                tgt_text = row.get("target_text") or ""
                rel_type = row.get("relation_type") or ""
                src_norm = normalize_entity_name(src_text)
                tgt_norm = normalize_entity_name(tgt_text)
                if not src_norm or not tgt_norm or not rel_type:
                    continue
                # Drop relations whose endpoints we dropped as conflicts.
                if src_norm in skipped_keys or tgt_norm in skipped_keys:
                    continue
                key = (src_norm, tgt_norm, rel_type)
                # Re-link endpoint text to the canonical merged form so
                # the downstream RELATE query can SELECT the entity row.
                canon_src = accumulators[src_norm].best_text if src_norm in accumulators else src_text
                canon_tgt = accumulators[tgt_norm].best_text if tgt_norm in accumulators else tgt_text
                candidate = {
                    "source_text": canon_src,
                    "target_text": canon_tgt,
                    "relation_type": rel_type,
                    "confidence": float(row.get("confidence") or 0.0),
                    "properties": dict(row.get("properties") or {}),
                }
                if key not in relation_best:
                    relation_best[key] = candidate
                    relation_insertion_order.append(key)
                elif candidate["confidence"] > relation_best[key]["confidence"]:
                    relation_best[key] = candidate

        relations_changed = 0
        # Tag relations with the target notebook's first source id (if
        # any) for provenance — matches how B.1f's persistence service
        # records `source_documents` on the RELATE edge.
        provenance_source = (
            target_source_ids[0] if target_source_ids else target_notebook_id
        )
        for key in relation_insertion_order:
            rel = relation_best[key]
            if dry_run:
                # Conservative estimate: every aggregated relation would
                # be RELATEd. The downstream IF-block dedups on RELATE
                # link existence, but the dry-run path can't run that
                # check cheaply — and from the UI's perspective, the
                # count is "how many relations were eligible".
                exists = await self._relation_exists(
                    rel["source_text"], rel["target_text"], rel["relation_type"]
                )
                if not exists:
                    relations_changed += 1
                continue

            try:
                created = await self._create_relation(
                    src_text=rel["source_text"],
                    tgt_text=rel["target_text"],
                    rel_type=rel["relation_type"],
                    confidence=rel["confidence"],
                    properties=rel["properties"],
                    provenance_source=provenance_source,
                )
            except Exception as e:
                logger.warning(
                    f"merge_notebooks: relation create failed for "
                    f"{rel['source_text']} -> {rel['target_text']} "
                    f"({rel['relation_type']}): {e}"
                )
                continue
            if created:
                relations_changed += 1

        # ----------------------------------------------------------------
        # Phase 6: telemetry (B.4 metrics).
        # ----------------------------------------------------------------
        await record_metric(
            "notebook_merge",
            {
                "entities": entities_changed,
                "relations": relations_changed,
                "conflicts": len(conflicts),
                "dry_run": dry_run,
                "source_notebook_count": len(source_notebook_ids),
            },
            notebook=target_notebook_id,
        )

        logger.info(
            "merge_notebooks: sources={sources} target={target} "
            "entities={ents} relations={rels} conflicts={confs} "
            "dry_run={dr}",
            sources=source_notebook_ids,
            target=target_notebook_id,
            ents=entities_changed,
            rels=relations_changed,
            confs=len(conflicts),
            dr=dry_run,
        )

        return NotebookMergeReport(
            entities_merged=entities_changed,
            relations_created=relations_changed,
            conflicts=conflicts,
            source_notebook_ids=list(source_notebook_ids),
            target_notebook_id=target_notebook_id,
            dry_run=dry_run,
        )

    # ------------------------------------------------------------------
    # Internal helpers — kept on the service so tests can swap the seam
    # by passing a mock repo; raw query helpers are protected by the
    # ``_`` prefix.
    # ------------------------------------------------------------------

    async def _sources_for_notebook(self, notebook_id: str) -> List[str]:
        """Resolve the source RecordIDs linked to a notebook via ``reference``.

        Mirrors the two-step join used by
        ``EntityRepository.list_orphans_with_status``: the
        ``source ↔ notebook`` linkage lives on the ``reference`` edge,
        not on ``source.notebook`` (no such column in migration 1).
        """
        if not notebook_id:
            return []
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

    async def _list_entities_for_notebook(
        self, notebook_id: str
    ) -> List[Dict[str, Any]]:
        """Return entity rows tied to a notebook via ``source_documents``.

        Entities don't carry a notebook FK directly; they carry the set
        of sources they were extracted from, and the source ↔ notebook
        edge is the ``reference`` table. So we resolve source ids first,
        then SELECT entities whose ``source_documents`` intersects.
        """
        source_ids = await self._sources_for_notebook(notebook_id)
        if not source_ids:
            return []
        try:
            rows = await execute_query(
                "SELECT id, canonical_name, entity_type, confidence, "
                "source_documents, properties, provenance_chain, "
                "type_tags, primary_type, updated_at "
                "FROM entity "
                "WHERE source_documents ANYINSIDE $source_ids",
                {"source_ids": source_ids},
            )
        except Exception as e:
            logger.error(
                f"_list_entities_for_notebook failed for '{notebook_id}': {e}"
            )
            return []
        return rows or []

    async def _list_relations_for_notebook(
        self, notebook_id: str
    ) -> List[Dict[str, Any]]:
        """Return relations whose endpoints are entities in this notebook.

        ``source_text`` / ``target_text`` are projected from the entity
        rows so callers downstream can run the canonical-surface-form
        rewrite without a second roundtrip.
        """
        source_ids = await self._sources_for_notebook(notebook_id)
        if not source_ids:
            return []
        try:
            rows = await execute_query(
                "SELECT id, in AS source_id, out AS target_id, "
                "relation_type, confidence, properties, source_documents "
                "FROM relation "
                "WHERE source_documents ANYINSIDE $source_ids",
                {"source_ids": source_ids},
            )
        except Exception as e:
            logger.error(
                f"_list_relations_for_notebook failed for '{notebook_id}': {e}"
            )
            return []
        if not rows:
            return []

        # Resolve endpoint canonical_name in batches. We need the
        # canonical text to feed the merge's normalization step.
        endpoint_ids = list({r["source_id"] for r in rows} | {r["target_id"] for r in rows})
        try:
            endpoint_rows = await execute_query(
                "SELECT id, canonical_name FROM entity WHERE id INSIDE $ids",
                {"ids": endpoint_ids},
            )
        except Exception as e:
            logger.error(
                f"_list_relations_for_notebook endpoint resolve failed: {e}"
            )
            endpoint_rows = []
        text_by_id: Dict[str, str] = {
            str(r["id"]): r.get("canonical_name", "") for r in (endpoint_rows or [])
        }
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "source_text": text_by_id.get(str(r["source_id"]), ""),
                    "target_text": text_by_id.get(str(r["target_id"]), ""),
                    "relation_type": r.get("relation_type", ""),
                    "confidence": r.get("confidence", 0.0),
                    "properties": r.get("properties", {}),
                }
            )
        return out

    async def _find_canonical_entity(
        self, text: str, label: str
    ) -> Optional[Dict[str, Any]]:
        """Look up an entity globally by ``(canonical_name, entity_type)``.

        Migration 39 keys entity rows globally by
        ``(canonical_name, entity_type)``, so this lookup returns the
        single canonical row regardless of which notebook contributed
        it. Used by the idempotency check to decide whether the upsert
        will CREATE a fresh row (counter += 1) or UPDATE an existing one
        (counter += 1 only when the merged content actually differs).
        """
        try:
            rows = await execute_query(
                "SELECT id, canonical_name, entity_type, type_tags, "
                "primary_type, confidence, source_documents, properties "
                "FROM entity "
                "WHERE canonical_name = $canonical_name "
                "AND entity_type = $entity_type LIMIT 1",
                {"canonical_name": text, "entity_type": label},
            )
        except Exception as e:
            logger.warning(
                f"_find_canonical_entity failed for '{text}' ({label}): {e}"
            )
            return None
        if not rows:
            return None
        return rows[0]

    @staticmethod
    def _entity_matches(merged: Entity, existing_row: Dict[str, Any]) -> bool:
        """Decide whether the merged Entity is semantically equal to the row.

        Compares the fields the upsert path actually writes:
        ``canonical_name``, ``entity_type``, ``type_tags`` (set
        semantics — order is meaningless), ``primary_type``,
        ``confidence``, ``source_documents`` (set semantics), and
        ``properties`` (key-value dict). ``updated_at`` and ``id`` are
        intentionally excluded so a re-run is recognised as a no-op even
        though the repo bumps the timestamp on every UPDATE.

        Returns True iff the upsert would produce a byte-equal row.
        """
        if merged.canonical_name != (existing_row.get("canonical_name") or ""):
            return False
        if merged.entity_type != (existing_row.get("entity_type") or ""):
            return False
        if (existing_row.get("primary_type") or "") != merged.primary_type:
            return False
        if float(existing_row.get("confidence") or 0.0) != float(merged.confidence):
            return False
        # Set semantics for list-valued fields — the upsert path unions
        # type_tags / source_documents, so order is incidental.
        if sorted(existing_row.get("type_tags") or []) != sorted(merged.type_tags):
            return False
        if sorted(existing_row.get("source_documents") or []) != sorted(
            merged.source_documents
        ):
            return False
        if dict(existing_row.get("properties") or {}) != dict(merged.properties):
            return False
        return True

    async def _relation_exists(
        self, src_text: str, tgt_text: str, rel_type: str
    ) -> bool:
        """Cheap existence probe used by the dry-run counter.

        The commit path folds this check into ``_create_relation`` so
        we save one round-trip per relation. The dry-run path keeps
        the standalone probe because it must NOT write.
        """
        try:
            rows = await execute_query(
                """
                LET $src = (SELECT VALUE id FROM entity
                    WHERE canonical_name = $src_name LIMIT 1);
                LET $tgt = (SELECT VALUE id FROM entity
                    WHERE canonical_name = $tgt_name LIMIT 1);
                IF array::len($src) > 0 AND array::len($tgt) > 0 THEN {
                    RETURN (SELECT id FROM relation
                        WHERE in = $src[0] AND out = $tgt[0]
                        AND relation_type = $rel_type LIMIT 1);
                } ELSE {
                    RETURN [];
                } END;
                """,
                {
                    "src_name": src_text,
                    "tgt_name": tgt_text,
                    "rel_type": rel_type,
                },
            )
        except Exception as e:
            logger.warning(
                f"_relation_exists probe failed for "
                f"{src_text} -> {tgt_text}: {e}"
            )
            return False
        if not rows:
            return False
        return bool(rows)

    async def _create_relation(
        self,
        src_text: str,
        tgt_text: str,
        rel_type: str,
        confidence: float,
        properties: Dict[str, Any],
        provenance_source: str,
    ) -> bool:
        """Idempotent RELATE that reports whether a new edge was written.

        SurrealDB's RELATE is not idempotent (each call creates a new
        edge row), so we probe-and-RELATE inside a single SurrealQL
        block to keep this to ONE round-trip per relation. The block
        returns a non-empty array iff a fresh edge was created; the
        caller uses the return value as the idempotency counter
        signal (Minor 1 fold).
        """
        try:
            rows = await execute_query(
                """
                LET $src = (SELECT VALUE id FROM entity
                    WHERE canonical_name = $src_name LIMIT 1);
                LET $tgt = (SELECT VALUE id FROM entity
                    WHERE canonical_name = $tgt_name LIMIT 1);
                IF array::len($src) > 0 AND array::len($tgt) > 0 THEN {
                    LET $existing = (SELECT id FROM relation
                        WHERE in = $src[0] AND out = $tgt[0]
                        AND relation_type = $rel_type LIMIT 1);
                    IF array::len($existing) == 0 THEN {
                        RETURN (RELATE $src[0]->relation->$tgt[0] SET
                            relation_type = $rel_type,
                            confidence = $confidence,
                            source_documents = [$source_id],
                            properties = $properties);
                    } ELSE {
                        RETURN [];
                    } END;
                } ELSE {
                    RETURN [];
                } END;
                """,
                {
                    "src_name": src_text,
                    "tgt_name": tgt_text,
                    "rel_type": rel_type,
                    "confidence": confidence,
                    "source_id": provenance_source,
                    "properties": properties,
                },
            )
        except Exception:
            # Caller logs the error; we just signal "no write happened".
            raise
        # The IF/RELATE branch returns the inserted edge row; the no-op
        # branches return an empty list. Either way, a non-empty result
        # means a fresh edge was written.
        return bool(rows)

    def _tags_by_notebook(
        self,
        key: str,
        notebook_ids: Sequence[str],
        per_notebook_entities: Dict[str, List[Dict[str, Any]]],
    ) -> List[set[str]]:
        """Project per-notebook tag sets for the given normalized key.

        Used by the conflict detector to test disjointness without
        re-scanning the raw rows from the public algorithm path.
        """
        out: List[set[str]] = []
        for nb_id in notebook_ids:
            tag_set: set[str] = set()
            for row in per_notebook_entities.get(nb_id, []):
                text = row.get("canonical_name") or ""
                if normalize_entity_name(text) != key:
                    continue
                tag_set.add(row.get("entity_type") or "UNKNOWN")
                for t in row.get("type_tags") or []:
                    if t:
                        tag_set.add(t)
            if tag_set:
                out.append(tag_set)
        return out

    @staticmethod
    def _tags_are_disjoint(tag_sets: List[set[str]]) -> bool:
        """Return True when no tag is shared across all the sets.

        Strict definition: any pair of sets has no overlap. If a single
        contributor exists, "disjoint" doesn't apply — return False.
        """
        if len(tag_sets) < 2:
            return False
        # Pair-wise intersection. If ANY pair shares a tag, the
        # name is not a conflict — they overlap somewhere and the
        # merge can proceed with the union.
        for i in range(len(tag_sets)):
            for j in range(i + 1, len(tag_sets)):
                if tag_sets[i] & tag_sets[j]:
                    return False
        return True
