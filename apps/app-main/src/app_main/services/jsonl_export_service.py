"""JSONL streaming export service (Phase D.2).

Emits two newline-delimited JSON streams (``entities.jsonl`` +
``relations.jsonl``) packaged inside a single ``application/zip`` archive
so a downstream Neo4j ``apoc.load.json`` / LangChain RAG-loader pipeline
can ingest the notebook graph with one HTTP round-trip.

Why a dedicated service?
========================
D.1 (Obsidian) renders *one markdown file per entity* — a flat-vault
target. D.3 (NetworkX) buffers a whole graph object and dispatches to a
networkx writer. D.2 (this) is the only format where the natural unit is
a **single line per row**, so streaming entity-by-entity into the zip
gives us bounded peak memory regardless of notebook size. The shared
concerns (filter + telemetry + ``ExportReport``) flow through
``shared.models.export`` so all three formats stay aligned on what gets
emitted into ``metrics{event_type: "export.*"}``.

Streaming strategy (Q-D-7)
==========================
V1 picks **build-then-stream**: write uncompressed JSONL one row at a
time into an in-memory ``zipfile.ZipFile`` (so only the *compressed*
archive is ever materialised, never the raw uncompressed payload), then
yield the finished compressed archive in 16KB chunks. The trade-off
versus a true chunk-as-you-go pipe is:

* Peak memory ≈ zip-archive size, not raw-jsonl size. Deflate compresses
  text-heavy entity rows ~3-5×, so a 5000-entity notebook stays well
  under the 200MB AC budget (~10MB compressed).
* No async-generator gymnastics for the writer side — ``ZipFile`` needs
  random access for the central directory anyway, which a streamed pipe
  doesn't give us cleanly.
* Per-entity ``writestr`` would peak in memory at one full ``entities.jsonl``
  string before zipping. We instead use ``ZipFile.open(..., "w")`` to get
  a file-like handle and write one line at a time so the in-memory
  payload during construction is the *compressed* archive only, never
  the raw uncompressed JSONL.

The 16KB chunk size for the yielded bytes matches Linux's default
TCP send-buffer (``tcp_wmem``) so we're not pre-fragmenting beyond
what the kernel would packetise anyway. It's also small enough that
the AC "yields >1 chunk for a 5000-entity fixture" test passes even
under aggressive deflate ratios on the repetitive test data.

Memory contract (AC #5)
=======================
Peak Python memory during a 5000-entity export must stay <200MB. The
key invariant is **never materialise the full entity list as Pydantic
model_dump strings before zipping** — instead we write each entity into
the open archive stream as soon as ``model_dump(mode="json", exclude=
{"embedding"})`` returns. The ``exclude={"embedding"}`` is critical:
each Entity carries a 768-float vector which would otherwise dominate
both the wire size *and* the in-memory cost (Q-D-1).

Filter pipeline (D.1c canonical)
================================
We mirror :class:`ObsidianExportService` exactly so all three exporters
share one pipeline. In order:

1. SurrealQL gate via ``ExportFilter`` (notebook scope, confidence,
   orphan_status, entity_types) — applied at the repo layer.
2. ``Entity.status`` post-filter (drop ``archived``/``merged``) — see
   ``EXCLUDED_ENTITY_STATUSES`` imported from the Obsidian service so
   there is exactly one source of truth.
3. ``min_connections`` post-filter — delegate to
   ``ObsidianExportService._apply_min_connections_filter`` (static
   method, no instance state) so any future tuning of the degree
   algorithm lands on both code paths simultaneously.
4. Q-D-4 silent-drop of relations whose endpoints didn't survive.

Per-line shape (Neo4j/LangChain compat)
=======================================
Each entity line carries::

    {
      "id": "entity:abc",
      "canonical_name": "Alice",
      "entity_type": "Person",
      "type_tags": ["Person", "Researcher"],
      "primary_type": "Researcher",
      "confidence": 0.95,
      "properties": {...},
      "source_documents": ["source:s1"],
      "extracted_at": "2026-06-14T10:00:00+00:00"
    }

Each relation line carries::

    {
      "id": "relation:xyz",
      "source_entity": "entity:abc",
      "target_entity": "entity:def",
      "relation_type": "WORKS_AT",
      "confidence": 0.9,
      "properties": {...},
      "source_documents": ["source:s1"]
    }

The ``source_entity``/``target_entity`` rename (vs DB-side ``in``/
``out``) is deliberate — these are the names downstream graph loaders
universally expect.

Telemetry contract (Q-D-8)
==========================
The ``record_metric("export.jsonl", ...)`` payload carries **counts
only** — never entity or relation IDs. The call is wrapped in a
``try/finally`` so failed exports still record their failure mode
(``{error: str, partial: bool}``). Tests assert single-call (no
duplicates on the error path) and counts-only invariants.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from typing import Any, AsyncIterator, Dict, List, Optional

from loguru import logger
from shared.models.entity import Entity, Relation
from shared.models.export import ExportFilter, ExportReport, JsonlExportRequest
from shared.services.metrics import record_metric

from app_main.services.obsidian_export_service import (
    EXCLUDED_ENTITY_STATUSES,
    ObsidianExportService,
)


# Yielded chunk size — picked so even modest 5000-entity fixtures split
# into multiple chunks (AC #5 streaming assertion). 16KB matches the
# typical TCP send-buffer size on Linux (``/proc/sys/net/ipv4/tcp_wmem``
# default 16384) so we're not artificially fragmenting beyond what the
# kernel would pack anyway.
_CHUNK_SIZE = 16 * 1024  # 16KB

# Fields excluded from the entity ``model_dump``. The 768-float embedding
# is the privacy + size concern; it is never useful in a JSONL export
# (downstream loaders recompute embeddings) and shipping it would balloon
# the wire size by ~6KB per entity.
_ENTITY_EXCLUDE = {"embedding"}


class JsonlExportService:
    """Builds a streamed JSONL zip from the notebook's filtered graph.

    Stateless aside from injected repos — one instance handles concurrent
    requests because every method takes ``notebook_id`` at call time.
    Threading through ``entity_repo`` and ``relation_repo`` independently
    keeps the constructor forward-compatible with a future split (today
    both methods live on the entity repo).
    """

    def __init__(
        self,
        entity_repository: Any,
        relation_repository: Optional[Any] = None,
    ) -> None:
        """Wire repository dependencies.

        Args:
            entity_repository: Object with
                ``list_entities_for_notebook(notebook_id, filter)`` and
                ``list_relations_for_notebook(notebook_id, filter)``.
            relation_repository: Reserved for a future split where
                relations migrate to their own repo. Defaults to the
                entity repo (both methods live there today).
        """
        self._entity_repo = entity_repository
        self._relation_repo = relation_repository or entity_repository

    async def stream_jsonl(
        self,
        notebook_id: str,
        request: JsonlExportRequest,
    ) -> AsyncIterator[bytes]:
        """Yield zip-archive bytes containing entities.jsonl + relations.jsonl.

        Pipeline (mirrors :meth:`ObsidianExportService.export` step-for-step
        from the filter side):

          1. Load entities via the D.0 projection.
          2. Drop ``status in {archived, merged}`` (shared
             ``EXCLUDED_ENTITY_STATUSES``).
          3. Apply ``min_connections`` post-filter (shared
             ``_apply_min_connections_filter`` staticmethod).
          4. Load relations via the D.0 projection.
          5. Stream entity rows into the zip's ``entities.jsonl`` member
             one line at a time — ``model_dump(mode="json",
             exclude={"embedding"})`` then ``json.dumps`` + newline.
          6. Stream relation rows whose BOTH endpoints survived the
             filter into ``relations.jsonl`` (Q-D-4 silent drop).
          7. Close the zip, yield the bytes in 16KB chunks.
          8. Emit ``record_metric("export.jsonl", ...)`` from a
             try/finally so failed exports still record their failure
             mode (Q-D-8).

        Args:
            notebook_id: SurrealDB record ID (e.g. ``"notebook:abc"``).
            request: ``JsonlExportRequest`` carrying the filter.

        Yields:
            Chunks of the zip archive bytes, each up to ``_CHUNK_SIZE``.
        """
        started = time.monotonic()
        error_payload: Optional[Dict[str, Any]] = None

        # Pre-bind defaults so the finally block always has well-formed
        # numbers even on early failures — matches the obsidian service
        # contract for the metric payload.
        entities_written = 0
        relations_written = 0
        bytes_written = 0

        try:
            entities, relations = await self._collect(notebook_id, request.filter)

            # Build the in-memory zip. ``ZipFile.open(...,"w")`` returns a
            # file-like sink so we can write one JSONL line at a time
            # without ever materialising the full uncompressed stream.
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
                # entities.jsonl — write entity-by-entity. The set of
                # surviving entity ids is the membership set for the
                # relation Q-D-4 drop below.
                surviving_ids = {str(e.id) for e in entities if e.id}
                with archive.open("entities.jsonl", "w") as fh:
                    for entity in entities:
                        line = self._entity_to_line(entity)
                        fh.write(line)
                        entities_written += 1

                # relations.jsonl — only emit relations whose BOTH
                # endpoints survived. Same Q-D-4 silent-drop semantic as
                # the Obsidian renderer.
                with archive.open("relations.jsonl", "w") as fh:
                    for relation in relations:
                        src = (
                            str(relation.in_entity)
                            if relation.in_entity
                            else None
                        )
                        dst = (
                            str(relation.out_entity)
                            if relation.out_entity
                            else None
                        )
                        if not src or not dst:
                            continue
                        if src not in surviving_ids or dst not in surviving_ids:
                            continue
                        line = self._relation_to_line(relation)
                        fh.write(line)
                        relations_written += 1

            payload = buf.getvalue()
            bytes_written = len(payload)

            duration_ms = int((time.monotonic() - started) * 1000)
            logger.info(
                "jsonl export: notebook={nb} entities={ents} relations={rels} "
                "bytes={bytes} duration_ms={ms}",
                nb=notebook_id,
                ents=entities_written,
                rels=relations_written,
                bytes=bytes_written,
                ms=duration_ms,
            )

            # Yield in chunks so the HTTP response is genuinely streamed.
            # The 16KB chunk matches Linux's tcp_wmem default; a 5000-
            # entity fixture compresses to ~55KB which splits into
            # multiple chunks (AC #5).
            for start in range(0, len(payload), _CHUNK_SIZE):
                yield payload[start : start + _CHUNK_SIZE]

        except BaseException as exc:
            # Widened from ``Exception`` to ``BaseException`` so that
            # ``GeneratorExit`` (Starlette cancelling on client disconnect /
            # tab close) and ``KeyboardInterrupt`` (Ctrl-C mid-export) are
            # also flagged as partial exports in the metric. Both are
            # ``BaseException`` subclasses; the narrower ``except
            # Exception`` would silently let the finally block emit a
            # success-looking metric for what is actually a cancelled or
            # interrupted stream.
            error_payload = {
                "error": f"{type(exc).__name__}: {exc}",
                "partial": True,
            }
            raise
        finally:
            duration_ms = int((time.monotonic() - started) * 1000)
            if error_payload is None:
                report = ExportReport(
                    entities_written=entities_written,
                    relations_written=relations_written,
                    files_written=2,  # entities.jsonl + relations.jsonl
                    bytes_written=bytes_written,
                    duration_ms=duration_ms,
                    filter_applied=request.filter,
                    metadata=None,
                )
                metric_payload: Dict[str, Any] = report.model_dump(mode="json")
            else:
                # On failure: still emit a well-formed metric so operators
                # can spot the failure mode. Counts-only per Q-D-8.
                metric_payload = {
                    "entities_written": entities_written,
                    "relations_written": relations_written,
                    "files_written": 0,
                    "bytes_written": bytes_written,
                    "duration_ms": duration_ms,
                    "filter_applied": request.filter.model_dump(mode="json"),
                    **error_payload,
                }
            await record_metric(
                "export.jsonl",
                metric_payload,
                source=None,
                notebook=notebook_id,
            )

    # ------------------------------------------------------------------
    # Filter pipeline (D.1c canonical)
    # ------------------------------------------------------------------

    async def _collect(
        self,
        notebook_id: str,
        filter_: ExportFilter,
    ) -> tuple[List[Entity], List[Relation]]:
        """Load + filter entities and relations.

        Mirrors :meth:`ObsidianExportService._collect` exactly so all
        three exporters share one filter pipeline. The order matters
        (status before min_connections — see D.1c BLOCKER fix in
        ``exports.py`` preview endpoint).
        """
        entities: List[Entity] = await self._entity_repo.list_entities_for_notebook(
            notebook_id, filter_
        )

        # 1. Status post-filter — drops archived + merged tombstones.
        #    Imported set so there's one source of truth across exporters.
        entities = [
            e
            for e in entities
            if (e.status or "active") not in EXCLUDED_ENTITY_STATUSES
        ]

        # 2. Load relations (D.0 SurrealQL gates on confidence + type).
        relations: List[Relation] = await self._relation_repo.list_relations_for_notebook(
            notebook_id, filter_
        )

        # 3. min_connections post-filter — delegate to the canonical
        #    staticmethod so any future tuning of the degree algorithm
        #    lands on the Obsidian path *and* the JSONL path
        #    simultaneously (Q-D-1c parity).
        entities = ObsidianExportService._apply_min_connections_filter(
            entities, relations, filter_.min_connections
        )

        return entities, relations

    # ------------------------------------------------------------------
    # Per-line serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _entity_to_line(entity: Entity) -> bytes:
        """Render one entity as a JSONL line (UTF-8 encoded with newline).

        Embedding exclusion has two independent layers, both load-bearing:

        1. ``model_dump(mode="json", exclude=_ENTITY_EXCLUDE)`` drops the
           768-float vector from the dumped dict — this is the *memory*
           guarantee (the vector is never even materialised into a Python
           list as part of the dump). Verified directly by
           ``test_embedding_excluded_by_model_dump_directly``.
        2. The named-key rebuild below (``row = {"id": full.get("id"),
           ...}``) is the *wire-format whitelist* — only the documented
           Neo4j/LangChain keys reach the JSONL line, regardless of what
           ``model_dump`` returns. Verified by ``test_entity_line_shape``.

        Both layers are defence-in-depth: a regression that flipped
        ``exclude=`` to a no-op would be caught by the direct dump test;
        a regression that broadened the rebuild whitelist would be caught
        by the shape test. ``json.dumps`` then re-encodes with a stable
        key order.
        """
        full = entity.model_dump(mode="json", exclude=_ENTITY_EXCLUDE)
        row = {
            "id": full.get("id"),
            "canonical_name": full.get("canonical_name"),
            "entity_type": full.get("entity_type"),
            "type_tags": full.get("type_tags", []),
            "primary_type": full.get("primary_type"),
            "confidence": full.get("confidence"),
            "properties": full.get("properties", {}),
            "source_documents": full.get("source_documents", []),
            "extracted_at": full.get("extracted_at"),
        }
        return (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")

    @staticmethod
    def _relation_to_line(relation: Relation) -> bytes:
        """Render one relation as a JSONL line.

        Renames the DB-side ``in``/``out`` to ``source_entity``/
        ``target_entity`` for downstream graph-loader compatibility
        (Neo4j ``apoc.load.json``, LangChain RAG loader). The Pydantic
        model already exposes these via the ``in_entity``/``out_entity``
        Python attribute names (with ``in``/``out`` aliases); we use the
        Python attrs so the rename is explicit in the JSONL payload.
        """
        full = relation.model_dump(mode="json")
        row = {
            "id": full.get("id"),
            "source_entity": (
                str(relation.in_entity) if relation.in_entity else None
            ),
            "target_entity": (
                str(relation.out_entity) if relation.out_entity else None
            ),
            "relation_type": full.get("relation_type"),
            "confidence": full.get("confidence"),
            "properties": full.get("properties", {}),
            "source_documents": full.get("source_documents", []),
        }
        return (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")


__all__ = ["JsonlExportService"]
