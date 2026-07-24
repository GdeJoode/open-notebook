"""Tests for the OKF v0.1 bundle mapper + service (Phase OKF.1).

The bundle mapper (:func:`build_okf_bundle`) is pure — it takes a filtered
projection and produces a ``{path: text}`` bundle + counts. These tests are
the CI core: they run fully offline, mocking the repositories.

Coverage
========
* Every concept file carries a parseable ``---`` frontmatter block with a
  non-empty ``type`` (spec §9 conformance).
* ``index.md`` present at the bundle root and declares ``okf_version``.
* Relations render as resolvable bundle-relative markdown links whose targets
  exist in the bundle — no dangling links; a filtered-out target is accounted,
  not emitted.
* Notes -> ``type: Note`` concepts; sources -> ``type: Source`` concepts.
* The omitted-field ledger (OKF-D3) lists the dropped substrate fields.
* Byte-stability: same projection + fixed timestamp -> identical bundle on re-run.
* Reserved filenames (``index`` / ``log``) never collide with a concept file.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import yaml
from app_main.services.okf_export_service import (
    OKF_OMITTED_FIELDS,
    OkfExportService,
    build_okf_bundle,
)
from shared.models.entity import Entity, Relation
from shared.models.export import ExportFilter, ObsidianExportRequest
from shared.models.notebook import Note
from shared.models.source import Asset, Source

_FRONTMATTER = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
_LINK = re.compile(r"\[[^\]]*\]\(/([^)]+)\)")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _entity(
    eid: str,
    name: str,
    *,
    primary_type: str = "Person",
    type_tags: Optional[List[str]] = None,
    description: Optional[str] = None,
    properties: Optional[Dict[str, Any]] = None,
    source_documents: Optional[List[str]] = None,
    external_ids: Optional[List[str]] = None,
    status: str = "active",
) -> Entity:
    return Entity(
        id=eid,
        canonical_name=name,
        entity_type=primary_type,
        primary_type=primary_type,
        type_tags=type_tags or [primary_type],
        description=description,
        properties=properties or {},
        source_documents=source_documents or [],
        external_ids=external_ids or [],
        status=status,
        confidence=0.95,
    )


def _relation(src: str, dst: str, rel_type: str = "KNOWS") -> Relation:
    return Relation(**{"in": src, "out": dst, "relation_type": rel_type, "confidence": 0.9})


def _note(nid: str, title: str, content: str = "Body text") -> Note:
    return Note(id=nid, title=title, content=content, note_type="human")


def _source(sid: str, title: str, url: Optional[str] = None) -> Source:
    return Source(id=sid, title=title, asset=Asset(url=url) if url else None)


def _parse_frontmatter(text: str) -> Optional[Dict[str, Any]]:
    m = _FRONTMATTER.match(text)
    if not m:
        return None
    return yaml.safe_load(m.group(1))


def _concept_files(bundle: Dict[str, str]) -> Dict[str, str]:
    """Non-reserved concept documents (everything but index.md / log.md)."""
    return {
        p: t
        for p, t in bundle.items()
        if p.rsplit("/", 1)[-1] not in {"index.md", "log.md"}
    }


# ---------------------------------------------------------------------------
# Pure builder
# ---------------------------------------------------------------------------


def _small_projection():
    entities = [
        _entity("entity:alice", "Alice", description="A researcher."),
        _entity("entity:bob", "Bob"),
        _entity("entity:carol", "Carol"),
    ]
    relations = [
        _relation("entity:alice", "entity:bob", "KNOWS"),
        _relation("entity:bob", "entity:carol", "MENTORS"),
    ]
    notes = [_note("note:n1", "Meeting notes")]
    sources = [_source("source:s1", "Paper A", url="https://example.com/a")]
    return entities, relations, notes, sources


def test_every_concept_has_type_frontmatter():
    """Spec §9: every non-reserved .md has a parseable frontmatter with type."""
    entities, relations, notes, sources = _small_projection()
    result = build_okf_bundle(
        entities=entities,
        relations=relations,
        notes=notes,
        sources=sources,
        dropped_relations=0,
        notebook_id="notebook:x",
        timestamp="2026-07-24T00:00:00+00:00",
    )
    for path, text in _concept_files(result.bundle).items():
        front = _parse_frontmatter(text)
        assert front is not None, f"{path} has no frontmatter block"
        assert front.get("type"), f"{path} missing required non-empty type"


def test_index_present_and_declares_version():
    entities, relations, notes, sources = _small_projection()
    result = build_okf_bundle(
        entities=entities,
        relations=relations,
        notes=notes,
        sources=sources,
        dropped_relations=0,
        notebook_id="notebook:x",
        timestamp="2026-07-24T00:00:00+00:00",
    )
    assert "index.md" in result.bundle
    front = _parse_frontmatter(result.bundle["index.md"])
    assert front is not None and front.get("okf_version") == "0.1"
    # Index enumerates each concept group.
    body = result.bundle["index.md"]
    assert "# Entities" in body and "# Notes" in body and "# Sources" in body


def test_relations_are_resolvable_links_no_dangling():
    """Every bundle-relative link target must exist in the bundle."""
    entities, relations, notes, sources = _small_projection()
    result = build_okf_bundle(
        entities=entities,
        relations=relations,
        notes=notes,
        sources=sources,
        dropped_relations=0,
        notebook_id="notebook:x",
        timestamp="2026-07-24T00:00:00+00:00",
    )
    paths = set(result.bundle.keys())
    for path, text in result.bundle.items():
        for target in _LINK.findall(text):
            assert target in paths, f"dangling link {target!r} in {path}"
    # alice -> bob relation link is present and resolvable.
    alice = result.bundle["entities/alice.md"]
    assert "(/entities/bob.md)" in alice
    assert result.relations_written == 2


def test_filtered_out_target_not_emitted_as_dangling_link():
    """A relation to a non-projected entity is accounted, not linked."""
    entities = [_entity("entity:alice", "Alice")]
    # Relation points at bob, who is NOT in the entity projection.
    relations = [_relation("entity:alice", "entity:bob", "KNOWS")]
    result = build_okf_bundle(
        entities=entities,
        relations=relations,
        notes=[],
        sources=[],
        dropped_relations=1,
        notebook_id="notebook:x",
        timestamp="2026-07-24T00:00:00+00:00",
    )
    alice = result.bundle["entities/alice.md"]
    assert "bob" not in alice.lower()
    assert result.relations_written == 0
    assert result.dropped_relations == 1


def test_notes_and_sources_typed_concepts():
    _, _, notes, sources = _small_projection()
    result = build_okf_bundle(
        entities=[],
        relations=[],
        notes=notes,
        sources=sources,
        dropped_relations=0,
        notebook_id="notebook:x",
        timestamp="2026-07-24T00:00:00+00:00",
    )
    note_path = next(p for p in result.bundle if p.startswith("notes/"))
    src_path = next(p for p in result.bundle if p.startswith("source/"))
    assert _parse_frontmatter(result.bundle[note_path])["type"] == "Note"
    src_front = _parse_frontmatter(result.bundle[src_path])
    assert src_front["type"] == "Source"
    assert src_front["resource"] == "https://example.com/a"


def test_omitted_field_ledger_lists_dropped_substrate():
    """OKF-D3: the ledger must name the dropped substrate fields."""
    result = build_okf_bundle(
        entities=[_entity("entity:a", "A")],
        relations=[],
        notes=[],
        sources=[],
        dropped_relations=0,
        notebook_id="notebook:x",
        timestamp="2026-07-24T00:00:00+00:00",
    )
    ledger = result.omitted_fields
    assert ledger == OKF_OMITTED_FIELDS
    for key in ("entity.embedding", "relation.confidence", "note.embedding", "source.embedding"):
        assert key in ledger
    assert any("verdict" in k or "contradiction" in v.lower() for k, v in ledger.items())
    assert any("similarity" in k or "hybrid" in v.lower() for k, v in ledger.items())


def test_bundle_is_byte_stable_on_rerun():
    entities, relations, notes, sources = _small_projection()
    kwargs = dict(
        entities=entities,
        relations=relations,
        notes=notes,
        sources=sources,
        dropped_relations=0,
        notebook_id="notebook:x",
        timestamp="2026-07-24T00:00:00+00:00",
    )
    first = build_okf_bundle(**kwargs)
    second = build_okf_bundle(**kwargs)
    assert first.bundle == second.bundle


def test_external_id_becomes_resource_else_urn():
    entities = [
        _entity("entity:a", "A", external_ids=["https://doi.org/10.1/x"]),
        _entity("entity:b", "B"),
    ]
    result = build_okf_bundle(
        entities=entities,
        relations=[],
        notes=[],
        sources=[],
        dropped_relations=0,
        notebook_id="notebook:demo",
        timestamp="2026-07-24T00:00:00+00:00",
    )
    a_front = _parse_frontmatter(result.bundle["entities/a.md"])
    b_front = _parse_frontmatter(result.bundle["entities/b.md"])
    assert a_front["resource"] == "https://doi.org/10.1/x"
    assert b_front["resource"].startswith("urn:open-notebook:")
    assert b_front["resource"].endswith(":entity:b")


def test_reserved_name_never_collides_with_concept():
    """An entity named 'index' must not shadow the reserved index.md."""
    entities = [_entity("entity:i", "index"), _entity("entity:l", "log")]
    result = build_okf_bundle(
        entities=entities,
        relations=[],
        notes=[],
        sources=[],
        dropped_relations=0,
        notebook_id="notebook:x",
        timestamp="2026-07-24T00:00:00+00:00",
    )
    assert "entities/index.md" not in result.bundle
    assert "entities/index-concept.md" in result.bundle
    assert "entities/log-concept.md" in result.bundle
    # The reserved root index is still the real directory listing.
    assert _parse_frontmatter(result.bundle["index.md"])["type"] == "Index"


def test_filename_collision_suffixes_within_kind():
    entities = [_entity("entity:1", "Smith"), _entity("entity:2", "smith.")]
    result = build_okf_bundle(
        entities=entities,
        relations=[],
        notes=[],
        sources=[],
        dropped_relations=0,
        notebook_id="notebook:x",
        timestamp="2026-07-24T00:00:00+00:00",
    )
    assert "entities/smith.md" in result.bundle
    assert "entities/smith-2.md" in result.bundle


def test_concept_stems_are_link_safe_no_spaces():
    """Multi-word names hyphenate so standard markdown links don't break."""
    entities = [
        _entity("entity:a", "Alice de Jong"),
        _entity("entity:b", "Bob"),
    ]
    relations = [_relation("entity:a", "entity:b", "KNOWS")]
    result = build_okf_bundle(
        entities=entities,
        relations=relations,
        notes=[],
        sources=[],
        dropped_relations=0,
        notebook_id="notebook:x",
        timestamp="2026-07-24T00:00:00+00:00",
    )
    assert "entities/alice-de-jong.md" in result.bundle
    # No bundle path contains a space (would terminate a markdown link URL).
    for path in result.bundle:
        assert " " not in path, f"space in concept path {path!r}"
    # The relation link target resolves and is space-free.
    for target in _LINK.findall(result.bundle["entities/bob.md"]):
        assert " " not in target
        assert target in result.bundle


# ---------------------------------------------------------------------------
# Service (mocked repos)
# ---------------------------------------------------------------------------


def _make_service(entities, relations, notes_rows, source_rows) -> OkfExportService:
    entity_repo = AsyncMock()
    entity_repo.list_entities_for_notebook = AsyncMock(return_value=entities)
    entity_repo.list_relations_for_notebook = AsyncMock(return_value=relations)
    notebook_repo = AsyncMock()
    notebook_repo.get_notes = AsyncMock(return_value=notes_rows)
    notebook_repo.get_sources = AsyncMock(return_value=source_rows)
    return OkfExportService(
        entity_repository=entity_repo,
        notebook_repository=notebook_repo,
        relation_repository=entity_repo,
    )


async def test_service_export_zips_bundle_and_reports_ledger():
    entities = [
        _entity("entity:alice", "Alice"),
        _entity("entity:bob", "Bob"),
    ]
    relations = [_relation("entity:alice", "entity:bob")]
    svc = _make_service(
        entities,
        relations,
        notes_rows=[{"id": "note:n1", "title": "N1", "content": "c", "note_type": "human"}],
        source_rows=[{"id": "source:s1", "title": "S1"}],
    )
    request = ObsidianExportRequest(
        mode="zip", filter=ExportFilter(min_connections=0, min_confidence=0.0)
    )
    artifact = await svc.export(
        "notebook:demo", request, timestamp="2026-07-24T00:00:00+00:00"
    )

    import io
    import zipfile

    assert artifact.zip_bytes
    with zipfile.ZipFile(io.BytesIO(artifact.zip_bytes)) as z:
        names = set(z.namelist())
    assert "index.md" in names
    assert "entities/alice.md" in names
    assert "notes/n1.md" in names
    assert "source/s1.md" in names

    assert artifact.report.entities_written == 2
    assert artifact.report.relations_written == 1
    meta = artifact.report.metadata or {}
    assert meta["notes_written"] == 1
    assert meta["sources_written"] == 1
    assert meta["omitted_fields"] == OKF_OMITTED_FIELDS


async def test_service_export_emits_metric(monkeypatch):
    events: List[dict] = []

    async def _spy(event_type, payload, source=None, notebook=None):
        events.append({"event_type": event_type, "payload": payload, "notebook": notebook})

    monkeypatch.delenv("OPEN_NOTEBOOK_DISABLE_METRICS", raising=False)
    monkeypatch.setattr(
        "app_main.services.okf_export_service.record_metric", _spy
    )
    svc = _make_service([_entity("entity:a", "A")], [], [], [])
    request = ObsidianExportRequest(
        mode="zip", filter=ExportFilter(min_connections=0, min_confidence=0.0)
    )
    await svc.export("notebook:m", request, timestamp="2026-07-24T00:00:00+00:00")

    assert len(events) == 1
    assert events[0]["event_type"] == "export.okf"
    # Counts-only: no record ids leak.
    for value in events[0]["payload"].values():
        if isinstance(value, str):
            assert not value.startswith(("entity:", "source:", "note:", "notebook:"))


async def test_count_entities_uses_projection():
    entities = [
        _entity("entity:a", "A"),
        _entity("entity:b", "B"),
        _entity("entity:c", "C", status="archived"),  # dropped by projection
    ]
    svc = _make_service(entities, [], [], [])
    n = await svc.count_entities(
        "notebook:c", ExportFilter(min_connections=0, min_confidence=0.0)
    )
    assert n == 2
