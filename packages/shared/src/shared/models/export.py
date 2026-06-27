"""
Export contracts shared across Track D output formats.

Track D adds three export surfaces on top of the Track-B knowledge graph:

* D.1 Obsidian vault (ZIP or direct-write)
* D.2 JSONL (entity + relation streams)
* D.3 NetworkX (graphml/gexf/gml/json-tree/edge-list/adjacency-list/pickle)

All three formats share the same upstream concerns -- which entities/
relations to include and what shape of telemetry to emit -- so the
filter and report contracts live here as a single source of truth.
This module is intentionally Pydantic-only so the model package stays
free of persistence concerns (mirrors how ``shared.models.entity`` is
kept clean of repository code).

V1 scope (Phase D.0):
* No user-visible behavior change. This module exists so D.1/D.2/D.3
  can ship without each one redefining filter knobs.
* JSONL/NetworkX stay sync-only in V1 (no JobType siblings for them --
  see ``shared.types.enums.JobType.EXPORT_OBSIDIAN`` for the sole async
  surface).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ExportFilter(BaseModel):
    """Shared filter knob set applied before entities/relations are emitted.

    Defaults pinned at the values approved in the Track D roadmap
    (§344): ``min_connections=5`` and ``min_confidence=0.9`` give the
    Track-B graph a sensible "publication-grade" subset by default.
    Callers can lower the thresholds to export the long tail.

    ``min_relation_confidence`` defaults to ``None``, which the
    repository layer treats as "use ``min_confidence``" -- relation
    confidence is on the same 0-1 scale as entity confidence, so the
    fallback keeps the common case to a single knob. Override only when
    relation-quality should diverge from entity-quality (e.g. emit
    high-confidence entities but only very-high-confidence edges).

    ``entity_types`` is an allow-list (``None`` = all types). Names are
    matched against ``Entity.primary_type`` / ``Entity.type_tags`` per
    the Track-B multi-type tagging (migration 44).

    ``include_orphans`` / ``include_archived`` map to the orphan-prune
    lifecycle landed in B.5b -- archived rows are filtered by default to
    keep exports clean.
    """

    min_connections: int = Field(
        default=5,
        ge=0,
        description=(
            "Minimum degree (in+out edges) required for an entity to be "
            "included. Default 5 per Track D roadmap §344."
        ),
    )
    min_confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum ``Entity.confidence`` to include. Default 0.9 per "
            "Track D roadmap §344."
        ),
    )
    min_relation_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum ``Relation.confidence`` to include. ``None`` means "
            "inherit from ``min_confidence`` -- relation and entity "
            "confidence share the same 0-1 scale so callers only need "
            "one knob in the common case."
        ),
    )
    entity_types: Optional[List[str]] = Field(
        default=None,
        description=(
            "Allow-list of entity types (e.g. ``['Person', 'Organization']``). "
            "``None`` includes all types. Matched against "
            "``Entity.primary_type`` and ``Entity.type_tags`` (migration 44)."
        ),
    )
    include_orphans: bool = Field(
        default=False,
        description=(
            "When False, entities flagged ``pending_reconnect`` (B.5b "
            "orphan-status lifecycle) are excluded. Defaults False to "
            "keep exports focused on connected entities."
        ),
    )
    include_archived: bool = Field(
        default=False,
        description=(
            "When False, entities with ``orphan_status == 'archived'`` "
            "(B.5b lifecycle) are excluded. Defaults False -- archived "
            "rows are tombstones, not export material."
        ),
    )
    include_document_layer: bool = Field(
        default=False,
        description=(
            "Track U.5 document graph. When True, the graph-native exporters "
            "(NetworkX, JSONL) ADD a document layer on top of the entity "
            "graph: ``source`` nodes plus ``mentions`` (document->entity) and "
            "``cites`` (document->document) edges from the materialized U.2/U.3 "
            "projections, scoped to the notebook's sources. Defaults False so "
            "the entity-only output is byte-for-byte unchanged -- existing "
            "entity-graph consumers are never affected. The Obsidian (markdown "
            "vault) exporter ignores this knob; the document layer does not map "
            "onto a flat-vault model (see the U.5 architecture note)."
        ),
    )


class ObsidianExportRequest(BaseModel):
    """Request body for the Obsidian vault export (Phase D.1).

    Two modes:

    * ``zip`` -- streams a downloadable archive (D.1a).
    * ``vault_path`` -- writes directly into a server-side vault
      directory (D.1b/D.1c). Overwrite semantics are documented in the
      D.1 plan (Q-D-6 -- overwrite mode).
    """

    mode: Literal["zip", "vault_path"] = Field(
        default="zip",
        description="Delivery mode: ``zip`` returns an archive, ``vault_path`` writes to disk.",
    )
    filter: ExportFilter = Field(
        default_factory=ExportFilter,
        description="Filter applied before vault generation.",
    )


class JsonlExportRequest(BaseModel):
    """Request body for the JSONL stream export (Phase D.2).

    Sync-only in V1 (no JobType per Q-D-2). The handler buffers via
    ``io.BytesIO`` per Q-D-7 then streams the response.
    """

    filter: ExportFilter = Field(
        default_factory=ExportFilter,
        description="Filter applied before JSONL emission.",
    )


# NetworkX format choices supported by D.3. Mirrors
# ``networkx.write_*`` / ``networkx.readwrite.json_graph`` surfaces.
# Pickle is included because it's the only format that round-trips
# arbitrary node/edge attributes losslessly -- a common request from
# analysts who chain NetworkX exports through pandas.
NetworkxFormat = Literal[
    "graphml",
    "gexf",
    "gml",
    "json-tree",
    "edge-list",
    "adjacency-list",
    "pickle",
]


class NetworkxExportRequest(BaseModel):
    """Request body for the NetworkX export (Phase D.3).

    Sync-only in V1 (no JobType per Q-D-2). NetworkX is single-process
    and CPU-bound; the foreground request is fine for the graph sizes
    the V1 filter defaults produce.
    """

    format: NetworkxFormat = Field(
        description="NetworkX output format. See ``NetworkxFormat`` for the closed set.",
    )
    filter: ExportFilter = Field(
        default_factory=ExportFilter,
        description="Filter applied before graph construction.",
    )


class ExportReport(BaseModel):
    """Result payload returned by every export handler.

    Fields cover what the front-end needs to render a "done" banner
    (counts), what the telemetry pipeline needs to size up runs
    (bytes + duration), and what the operator may want to audit
    (filter snapshot). Per Q-D-8 the payload carries counts only --
    never entity/relation IDs -- so it can be logged into the
    ``metrics`` table without PII risk.

    ``metadata`` is intentionally optional and free-form. D.1 will use
    it to report ``files_skipped`` for vault-path collisions; D.2/D.3
    may use it to report ``dropped_relations`` per Q-D-4 (silently
    dropped when the target entity is filtered out).
    """

    entities_written: int = Field(
        ge=0,
        description="Count of entities that made it through the filter and were emitted.",
    )
    relations_written: int = Field(
        ge=0,
        description="Count of relations that made it through the filter and were emitted.",
    )
    files_written: int = Field(
        ge=0,
        description=(
            "Count of physical files produced (1 for the JSONL streams, "
            "1 for the NetworkX writer, ``entities_written`` for the "
            "Obsidian vault)."
        ),
    )
    bytes_written: int = Field(
        ge=0,
        description="Total bytes emitted across all files.",
    )
    duration_ms: int = Field(
        ge=0,
        description="Wall-clock time spent in the export handler, in milliseconds.",
    )
    filter_applied: ExportFilter = Field(
        description="Snapshot of the filter that produced this report.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Free-form per-format extras (e.g. ``dropped_relations`` "
            "count for Q-D-4, ``files_skipped`` for D.1 vault-path "
            "collisions). Keys are not standardised across formats."
        ),
    )


__all__ = [
    "ExportFilter",
    "ObsidianExportRequest",
    "JsonlExportRequest",
    "NetworkxExportRequest",
    "NetworkxFormat",
    "ExportReport",
]
