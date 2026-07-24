"""Open Knowledge Format (OKF v0.1) export service (Phase OKF.1 + OKF.2).

Maps a notebook's curated knowledge-graph projection onto a **Knowledge
Bundle** conformant with the vendored OKF v0.1 spec
(``docs/tracks/OKF-open-knowledge-format/SPEC-v0.1.md``): a tree of UTF-8
markdown files, each a *concept* with a required ``type`` frontmatter key,
cross-linked with standard markdown links, plus a reserved root ``index.md``.

Relationship to the Obsidian exporter
=====================================
OKF is a sibling of :mod:`obsidian_export_service`. Both project the same
entity/relation subset — that shared selection lives in
:mod:`export_projection` (OKF-D2) so the two bundles never drift. OKF differs
only in:

* **Frontmatter keys** — OKF's ``type`` / ``title`` / ``description`` /
  ``resource`` / ``tags`` / ``timestamp`` (spec §4.1) instead of Obsidian's
  ``id`` / ``type_tags`` / ``confidence`` / ``sources`` bag.
* **Link syntax** — standard markdown links to a concept's bundle path
  (``[label](/entities/bob.md)``, spec §5) instead of ``[[wikilinks]]``.
* **Bundle conventions** — a directory tree with concepts grouped under
  ``entities/`` / ``notes/`` / ``sources/`` and a generated ``index.md``
  directory listing (spec §6), instead of a flat vault + README.

Lossy-by-design (OKF-D3)
========================
The bundle carries the *curated-knowledge* subset only. Embeddings,
chunk-level provenance, verdict/contradiction edges, similarity/hybrid-search
signal, and graph-centrality scores have **no OKF representation** and are
dropped. The loss is never silent: :data:`OKF_OMITTED_FIELDS` is copied into
every :class:`ExportReport` so a consumer can see exactly what the bundle does
*not* carry.

Purity / determinism (AC OKF.1.5)
=================================
:func:`build_okf_bundle` is pure: no repository, no clock. The caller injects
``timestamp`` so the same projection produces a byte-stable bundle. Frontmatter
is emitted via ``yaml.safe_dump(sort_keys=True)`` and every listing is sorted,
so re-running over identical input yields identical bytes.
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger
from pydantic import BaseModel, Field
from shared.models.entity import Entity, Relation
from shared.models.export import ExportFilter, ExportReport, ObsidianExportRequest
from shared.models.notebook import Note
from shared.models.source import Source
from shared.services.metrics import record_metric
from shared.utils.external_ids import resolve_external_ids
from shared.utils.name_normalizer import normalize_entity_name

from app_main.services.export_projection import project_graph

# ---------------------------------------------------------------------------
# Bundle conventions
# ---------------------------------------------------------------------------

#: Reserved filenames (spec §3.1). A concept document MUST NOT use these at
#: any level of the hierarchy, so a concept whose stem normalises to one of
#: them is suffixed (see :func:`_reserved_safe_stem`).
RESERVED_STEMS = frozenset({"index", "log"})

#: Sub-directory each concept kind lives under within the bundle.
ENTITY_DIR = "entities"
NOTE_DIR = "notes"
SOURCE_DIR = "source"  # spec's ``references/`` mirror is source-like; kept short

#: OKF spec version this exporter targets (declared in the root index.md
#: frontmatter, the only place frontmatter is permitted in an index — §11).
OKF_VERSION = "0.1"

# Same filesystem-unsafe character class as the Obsidian exporter's
# ``_ENTITY_FILENAME_UNSAFE`` so a concept path survives both POSIX and
# Windows filesystems and a zip's flat namelist. ``normalize_entity_name`` is
# a *content* normaliser only and does not strip path separators.
_UNSAFE_STEM = re.compile(r'[:/\\\r\n\t\x00"\'<>|?*]')
_MAX_STEM_LEN = 200


#: Omitted-field ledger (OKF-D3). Every substrate field that has no OKF
#: representation and is therefore dropped from the bundle. Copied verbatim
#: into each ExportReport so the loss is explicit, never silent.
OKF_OMITTED_FIELDS: Dict[str, str] = {
    "entity.embedding": "Vector embedding (semantic-search substrate) has no OKF representation.",
    "entity.confidence": "Extraction confidence score is not carried; OKF concepts are unscored.",
    "entity.pagerank": "Graph-centrality (PageRank) score has no OKF representation.",
    "entity.betweenness": "Graph-centrality (betweenness) score has no OKF representation.",
    "entity.community_id": "Community-detection cluster id has no OKF representation.",
    "entity.provenance_chain": "Chunk-level extraction provenance has no OKF representation.",
    "relation.confidence": "Edge confidence score is dropped; OKF links are untyped and unscored.",
    "relation.provenance_chain": "Chunk-level relation provenance has no OKF representation.",
    "note.embedding": "Note vector embedding has no OKF representation.",
    "source.embedding": "Source-level aggregate vector has no OKF representation.",
    "source.full_text": "Full document body / chunk-level text has no OKF representation.",
    "chunk_provenance": "Track X chunk-level source provenance is not projected into OKF.",
    "verdict_contradiction_edges": "Track Z verdict / contradiction edges have no OKF representation.",
    "similarity_hybrid_search": "Track W hybrid-search / similarity signal has no OKF representation.",
}


# ---------------------------------------------------------------------------
# Filename plumbing
# ---------------------------------------------------------------------------


def _safe_stem(name: str) -> str:
    """Normalise + strip filesystem-unsafe chars from a concept name.

    Layered on :func:`normalize_entity_name` (which lower-cases + collapses
    whitespace but leaves ``/``, ``:`` etc. intact) so the result is a single
    flat path segment. Empty input collapses to ``""`` and the caller
    substitutes ``"untitled"``.
    """
    base = normalize_entity_name(name or "")
    safe = _UNSAFE_STEM.sub("_", base)
    if len(safe) > _MAX_STEM_LEN:
        safe = safe[:_MAX_STEM_LEN]
    return safe


def _reserved_safe_stem(stem: str) -> str:
    """Ensure a concept stem never collides with a reserved filename (§3.1)."""
    if stem in RESERVED_STEMS:
        return f"{stem}-concept"
    return stem


def _build_concept_paths(
    items: List[Any],
    subdir: str,
    name_of: Any,
    id_of: Any,
) -> Dict[str, str]:
    """Build ``{record_id: "<subdir>/<stem>.md"}`` resolving collisions.

    Mirrors the Obsidian exporter's per-stem collision counter (``foo.md`` ->
    ``foo-2.md`` -> ``foo-3.md``) but scoped per sub-directory and with the
    reserved-name guard so ``index`` / ``log`` concepts don't shadow the
    reserved files. Iteration order follows the (already deterministic) input
    list, so the mapping is stable per projection.
    """
    seen: Counter[str] = Counter()
    mapping: Dict[str, str] = {}
    for item in items:
        rid = id_of(item)
        if not rid:
            continue
        stem = _reserved_safe_stem(_safe_stem(name_of(item)) or "untitled")
        count = seen[stem]
        filename = f"{stem}.md" if count == 0 else f"{stem}-{count + 1}.md"
        seen[stem] += 1
        mapping[str(rid)] = f"{subdir}/{filename}"
    return mapping


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _frontmatter_block(fields: Dict[str, Any]) -> str:
    """Render a spec §4 YAML frontmatter block with deterministic key order.

    Empty/``None`` values are dropped so the block only carries meaningful
    keys, except ``type`` which is REQUIRED (spec §4.1) and always present.
    ``sort_keys=True`` pins byte-stable output across runs.
    """
    cleaned = {
        k: v
        for k, v in fields.items()
        if v is not None and v != "" and v != []
    }
    cleaned.setdefault("type", fields.get("type") or "Concept")
    dumped = yaml.safe_dump(
        cleaned, sort_keys=True, allow_unicode=True, default_flow_style=False
    ).rstrip()
    return f"---\n{dumped}\n---"


def _link(label: str, path: str) -> str:
    """Bundle-relative absolute markdown link (spec §5.1)."""
    return f"[{label}](/{path})"


def _render_entity_concept(
    entity: Entity,
    relations: List[Relation],
    entity_paths: Dict[str, str],
    source_paths: Dict[str, str],
    notebook_stem: str,
    timestamp: Optional[str],
) -> str:
    """Render one entity concept document (frontmatter + structural body)."""
    entity_id = str(entity.id) if entity.id else ""

    external = resolve_external_ids(entity)
    resource = external[0] if external else _entity_urn(notebook_stem, entity_id)

    front = _frontmatter_block(
        {
            "type": entity.primary_type or entity.entity_type or "Concept",
            "title": entity.canonical_name or None,
            "description": entity.description or None,
            "resource": resource,
            "tags": sorted(entity.type_tags) if entity.type_tags else None,
            "timestamp": timestamp,
        }
    )

    lines: List[str] = [front, ""]
    lines.append(f"# {entity.canonical_name or 'Untitled'}")
    lines.append("")
    if entity.description:
        lines.append(entity.description)
        lines.append("")

    if entity.properties:
        lines.append("## Attributes")
        lines.append("")
        for key in sorted(entity.properties.keys()):
            lines.append(f"- **{key}**: {entity.properties[key]}")
        lines.append("")

    # Relations -> standard markdown links to surviving concepts (spec §5).
    # A relation whose target was filtered out is accounted in the report,
    # never emitted as a dangling link (Q-D-4 parity).
    related: List[str] = []
    for relation in relations:
        src = str(relation.in_entity) if relation.in_entity else None
        dst = str(relation.out_entity) if relation.out_entity else None
        if not src or not dst:
            continue
        if src == entity_id:
            other = dst
        elif dst == entity_id:
            other = src
        else:
            continue
        if other not in entity_paths:
            continue
        label = _path_title(entity_paths[other])
        rel_type = relation.relation_type or "related to"
        related.append(f"- {_link(label, entity_paths[other])} — {rel_type}")
    if related:
        lines.append("## Relations")
        lines.append("")
        lines.extend(sorted(related))
        lines.append("")

    # Citations -> links to the source concepts this entity was extracted
    # from, when those sources are in the bundle (spec §8).
    citations = sorted(
        {
            source_paths[str(sid)]
            for sid in entity.source_documents
            if str(sid) in source_paths
        }
    )
    if citations:
        lines.append("## Citations")
        lines.append("")
        for idx, path in enumerate(citations, start=1):
            lines.append(f"[{idx}] {_link(_path_title(path), path)}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_note_concept(note: Note, timestamp: Optional[str]) -> str:
    """Render one note concept document (``type: Note``)."""
    title = note.title or "Untitled note"
    front = _frontmatter_block(
        {
            "type": "Note",
            "title": title,
            "timestamp": timestamp,
        }
    )
    lines: List[str] = [front, "", f"# {title}", ""]
    if note.content:
        lines.append(note.content)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_source_concept(source: Source, timestamp: Optional[str]) -> str:
    """Render one source concept document (``type: Source``)."""
    title = source.title or "Untitled source"
    resource = None
    if source.asset is not None:
        resource = source.asset.url or source.asset.file_path

    front = _frontmatter_block(
        {
            "type": "Source",
            "title": title,
            "resource": resource,
            "tags": sorted(source.topics) if source.topics else None,
            "timestamp": timestamp,
        }
    )
    lines: List[str] = [front, "", f"# {title}", ""]
    if source.topics:
        lines.append("Topics: " + ", ".join(sorted(source.topics)))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_index(
    entity_paths: Dict[str, str],
    note_paths: Dict[str, str],
    source_paths: Dict[str, str],
    entity_titles: Dict[str, str],
    note_titles: Dict[str, str],
    source_titles: Dict[str, str],
    timestamp: Optional[str],
) -> str:
    """Render the reserved root ``index.md`` directory listing (spec §6).

    The bundle-root index is the only place frontmatter is permitted in an
    index file, and is where the OKF version is declared (§11).
    """
    front = _frontmatter_block(
        {
            "type": "Index",
            "okf_version": OKF_VERSION,
            "timestamp": timestamp,
        }
    )
    lines: List[str] = [front, "", "# Knowledge Bundle", ""]

    def _section(heading: str, paths: Dict[str, str], titles: Dict[str, str]) -> None:
        if not paths:
            return
        lines.append(f"# {heading}")
        lines.append("")
        for rid in sorted(paths, key=lambda r: paths[r]):
            path = paths[rid]
            title = titles.get(rid) or _path_title(path)
            lines.append(f"* {_link(title, path)}")
        lines.append("")

    _section("Entities", entity_paths, entity_titles)
    _section("Notes", note_paths, note_titles)
    _section("Sources", source_paths, source_titles)
    return "\n".join(lines).rstrip() + "\n"


def _entity_urn(notebook_stem: str, entity_id: str) -> str:
    """Notebook-scoped stable URN for an entity with no external identifier."""
    local = entity_id.split(":", 1)[-1] if entity_id else "unknown"
    return f"urn:open-notebook:{notebook_stem}:entity:{local}"


def _path_title(path: str) -> str:
    """Derive a display label from a concept path (filename stem)."""
    stem = path.rsplit("/", 1)[-1]
    if stem.endswith(".md"):
        stem = stem[:-3]
    return stem


# ---------------------------------------------------------------------------
# Pure bundle builder (OKF.1)
# ---------------------------------------------------------------------------


@dataclass
class OkfBuildResult:
    """Pure output of :func:`build_okf_bundle`.

    ``bundle`` maps ``relative_path -> file_text``. The counts + omitted
    ledger feed the :class:`ExportReport` the service returns.
    """

    bundle: Dict[str, str]
    entity_concepts: int
    note_concepts: int
    source_concepts: int
    relations_written: int
    dropped_relations: int
    omitted_fields: Dict[str, str] = field(default_factory=lambda: dict(OKF_OMITTED_FIELDS))


def build_okf_bundle(
    *,
    entities: List[Entity],
    relations: List[Relation],
    notes: List[Note],
    sources: List[Source],
    dropped_relations: int,
    notebook_id: str,
    timestamp: Optional[str],
) -> OkfBuildResult:
    """Map a filtered projection onto an in-memory OKF bundle (pure).

    Args:
        entities/relations: the shared-projection survivors (see
            :func:`export_projection.project_graph`). ``relations`` is the full
            list; endpoint survival is resolved here against ``entities``.
        notes/sources: the notebook's notes and sources to emit as concepts.
        dropped_relations: relations whose endpoint didn't survive, for the report.
        notebook_id: SurrealDB notebook record id (used for the urn scope only,
            never a clock).
        timestamp: ISO-8601 string injected into every concept's ``timestamp``
            frontmatter, or ``None`` to omit. Kept a caller argument so the
            bundle bytes stay deterministic.

    Returns:
        :class:`OkfBuildResult` with the ``{path: text}`` bundle + counts.
    """
    notebook_stem = _safe_stem(notebook_id) or "notebook"

    entity_paths = _build_concept_paths(
        entities, ENTITY_DIR, lambda e: e.canonical_name, lambda e: e.id
    )
    note_paths = _build_concept_paths(
        notes, NOTE_DIR, lambda n: n.title or "note", lambda n: n.id
    )
    source_paths = _build_concept_paths(
        sources, SOURCE_DIR, lambda s: s.title or "source", lambda s: s.id
    )

    bundle: Dict[str, str] = {}

    for entity in entities:
        if not entity.id:
            continue
        path = entity_paths[str(entity.id)]
        bundle[path] = _render_entity_concept(
            entity, relations, entity_paths, source_paths, notebook_stem, timestamp
        )

    for note in notes:
        if not note.id:
            continue
        bundle[note_paths[str(note.id)]] = _render_note_concept(note, timestamp)

    for source in sources:
        if not source.id:
            continue
        bundle[source_paths[str(source.id)]] = _render_source_concept(source, timestamp)

    entity_titles = {
        str(e.id): (e.canonical_name or "") for e in entities if e.id
    }
    note_titles = {str(n.id): (n.title or "") for n in notes if n.id}
    source_titles = {str(s.id): (s.title or "") for s in sources if s.id}

    bundle["index.md"] = _render_index(
        entity_paths,
        note_paths,
        source_paths,
        entity_titles,
        note_titles,
        source_titles,
        timestamp,
    )

    # Rendered-relation count: both endpoints survived (mirrors the Obsidian
    # exporter's _count_rendered_relations).
    relations_written = 0
    for relation in relations:
        src = str(relation.in_entity) if relation.in_entity else None
        dst = str(relation.out_entity) if relation.out_entity else None
        if src and dst and src in entity_paths and dst in entity_paths:
            relations_written += 1

    return OkfBuildResult(
        bundle=bundle,
        entity_concepts=len(entity_paths),
        note_concepts=len(note_paths),
        source_concepts=len(source_paths),
        relations_written=relations_written,
        dropped_relations=dropped_relations,
    )


# ---------------------------------------------------------------------------
# Service artifact
# ---------------------------------------------------------------------------


class OkfExportArtifact(BaseModel):
    """Return value of :meth:`OkfExportService.export`.

    ``zip_bytes`` carries the zipped bundle. ``report`` carries the counts +
    the omitted-field ledger (OKF-D3). ``job_id`` is populated only when the
    export was deferred to the job queue (large-notebook path, OKF.2).
    """

    report: ExportReport = Field(description="Counts, filter snapshot, omitted-field ledger.")
    zip_bytes: Optional[bytes] = Field(
        default=None, description="Zipped OKF bundle (inline path)."
    )
    job_id: Optional[str] = Field(
        default=None, description="Set when the export was deferred to the job queue."
    )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class OkfExportService:
    """Builds an OKF v0.1 Knowledge Bundle for a notebook.

    Stateless aside from injected repositories — one instance handles
    concurrent requests. ``entity_repo`` answers both
    ``list_entities_for_notebook`` and ``list_relations_for_notebook`` (D.0);
    ``notebook_repo`` answers ``get_notes`` / ``get_sources`` for the note +
    source concepts.
    """

    def __init__(
        self,
        entity_repository: Any,
        notebook_repository: Any,
        relation_repository: Optional[Any] = None,
    ) -> None:
        self._entity_repo = entity_repository
        self._relation_repo = relation_repository or entity_repository
        self._notebook_repo = notebook_repository

    async def export(
        self,
        notebook_id: str,
        request: ObsidianExportRequest,
        *,
        timestamp: Optional[str] = None,
    ) -> OkfExportArtifact:
        """Build + zip the OKF bundle, returning the archive bytes + report.

        Reuses :class:`ObsidianExportRequest` as the request contract — the
        two exporters share the ``filter`` knob set and the OKF exporter only
        ever runs in an in-memory-zip mode (the ``mode`` field is ignored).

        ``timestamp`` is injected into every concept's frontmatter; callers
        that need byte-stable output across runs pass a fixed value, otherwise
        the current UTC time is used.
        """
        started = time.monotonic()
        error_payload: Optional[Dict[str, Any]] = None
        result: Optional[OkfBuildResult] = None
        zip_bytes = b""

        if timestamp is None:
            from datetime import datetime, timezone

            timestamp = datetime.now(timezone.utc).isoformat()

        try:
            entities, relations, notes, sources, dropped = await self._collect(
                notebook_id, request.filter
            )
            result = build_okf_bundle(
                entities=entities,
                relations=relations,
                notes=notes,
                sources=sources,
                dropped_relations=dropped,
                notebook_id=notebook_id,
                timestamp=timestamp,
            )
            zip_bytes = self._zip_bundle(result.bundle)

            duration_ms = int((time.monotonic() - started) * 1000)
            report = ExportReport(
                entities_written=result.entity_concepts,
                relations_written=result.relations_written,
                files_written=len(result.bundle),
                bytes_written=len(zip_bytes),
                duration_ms=duration_ms,
                filter_applied=request.filter,
                metadata={
                    "notes_written": result.note_concepts,
                    "sources_written": result.source_concepts,
                    "dropped_relations": result.dropped_relations,
                    "omitted_fields": result.omitted_fields,
                    "okf_version": OKF_VERSION,
                },
            )
            logger.info(
                "okf export: notebook={nb} entities={ents} notes={notes} "
                "sources={srcs} relations={rels} files={files} bytes={bytes} "
                "duration_ms={ms}",
                nb=notebook_id,
                ents=result.entity_concepts,
                notes=result.note_concepts,
                srcs=result.source_concepts,
                rels=result.relations_written,
                files=len(result.bundle),
                bytes=len(zip_bytes),
                ms=duration_ms,
            )
            return OkfExportArtifact(report=report, zip_bytes=zip_bytes, job_id=None)

        except Exception as exc:
            error_payload = {
                "error": f"{type(exc).__name__}: {exc}",
                "partial": True,
            }
            raise
        finally:
            duration_ms = int((time.monotonic() - started) * 1000)
            if error_payload is None and result is not None:
                payload: Dict[str, Any] = {
                    "entities_written": result.entity_concepts,
                    "notes_written": result.note_concepts,
                    "sources_written": result.source_concepts,
                    "relations_written": result.relations_written,
                    "files_written": len(result.bundle),
                    "bytes_written": len(zip_bytes),
                    "duration_ms": duration_ms,
                    "dropped_relations": result.dropped_relations,
                }
            else:
                payload = {
                    "entities_written": 0,
                    "notes_written": 0,
                    "sources_written": 0,
                    "relations_written": 0,
                    "files_written": 0,
                    "bytes_written": 0,
                    "duration_ms": duration_ms,
                    **(error_payload or {}),
                }
            await record_metric(
                "export.okf", payload, source=None, notebook=notebook_id
            )

    async def count_entities(
        self, notebook_id: str, filter_: ExportFilter
    ) -> int:
        """Return the surviving-entity count for the async-threshold decision.

        Runs the same projection the export uses so the threshold reflects the
        real bundle size, not the raw pre-filter count.
        """
        entities = await self._entity_repo.list_entities_for_notebook(
            notebook_id, filter_
        )
        relations = await self._relation_repo.list_relations_for_notebook(
            notebook_id, filter_
        )
        return len(project_graph(entities, relations, filter_).entities)

    async def _collect(
        self,
        notebook_id: str,
        filter_: ExportFilter,
    ) -> tuple[List[Entity], List[Relation], List[Note], List[Source], int]:
        """Load + filter entities/relations, load notes + sources."""
        entities = await self._entity_repo.list_entities_for_notebook(
            notebook_id, filter_
        )
        relations = await self._relation_repo.list_relations_for_notebook(
            notebook_id, filter_
        )
        projection = project_graph(entities, relations, filter_)

        note_rows = await self._notebook_repo.get_notes(notebook_id)
        source_rows = await self._notebook_repo.get_sources(notebook_id)
        notes = [Note(**row) for row in note_rows] if note_rows else []
        sources = [Source(**row) for row in source_rows] if source_rows else []

        return (
            projection.entities,
            projection.relations,
            notes,
            sources,
            projection.dropped_relations,
        )

    @staticmethod
    def _zip_bundle(bundle: Dict[str, str]) -> bytes:
        """Zip the ``{path: text}`` bundle deterministically (sorted paths)."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(bundle.keys()):
                archive.writestr(path, bundle[path])
        return buf.getvalue()


__all__ = [
    "OkfExportService",
    "OkfExportArtifact",
    "OkfBuildResult",
    "build_okf_bundle",
    "OKF_OMITTED_FIELDS",
    "OKF_VERSION",
]
