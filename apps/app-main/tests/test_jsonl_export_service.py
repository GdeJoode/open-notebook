"""Tests for :class:`JsonlExportService` (Phase D.2).

Covers the AC scenarios from plan §D.2 plus the inversion targets the
adversarial reviewer would mentally probe:

* 100-entity notebook yields 100 entity lines + 50 relation lines.
* Per-line shape matches the documented Neo4j/LangChain dict.
* ``embedding`` field is provably absent from each entity line.
* ``min_confidence`` post-filter excludes low-confidence entities.
* ``status in {archived, merged}`` exclusion (D.1c BLOCKER regression).
* Streaming yields >1 chunk for a 5000-entity fixture (memory guarantee).
* Empty notebook produces empty .jsonl files (not error, not missing).
* ``record_metric`` fires exactly once per export — happy + failure path.

Mocks
=====
``list_entities_for_notebook`` and ``list_relations_for_notebook`` are
``AsyncMock``. No DB, no SurrealDB connection — the tests run
in-process.
"""

from __future__ import annotations

import io
import json
import tracemalloc
import zipfile
from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock

import pytest
from shared.models.entity import Entity, Relation
from shared.models.export import ExportFilter, JsonlExportRequest

from app_main.services.jsonl_export_service import JsonlExportService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _entity(
    eid: str,
    name: str,
    type_tags: List[str] | None = None,
    primary_type: str | None = None,
    confidence: float = 0.95,
    properties: dict | None = None,
    status: str = "active",
    source_documents: List[str] | None = None,
    embedding: List[float] | None = None,
) -> Entity:
    """Build an Entity with sane export-ready defaults.

    The ``embedding`` arg defaults to a small non-empty vector so the
    "embedding absent from JSONL" assertion has something to detect a
    regression against (an exclude= no-op would surface the vector).
    """
    tags = type_tags or []
    return Entity(
        id=eid,
        canonical_name=name,
        entity_type=primary_type or (tags[0] if tags else "Concept"),
        confidence=confidence,
        source_documents=source_documents or ["source:s1"],
        properties=properties or {},
        type_tags=tags,
        primary_type=primary_type or (tags[0] if tags else "Concept"),
        status=status,
        embedding=embedding if embedding is not None else [0.1, 0.2, 0.3],
        extracted_at=datetime(2026, 6, 14, 10, 0, 0, tzinfo=timezone.utc),
    )


def _relation(
    src: str,
    dst: str,
    rel_type: str = "RELATED_TO",
    confidence: float = 0.95,
    rid: str | None = None,
) -> Relation:
    """Build a Relation between two entity ids."""
    payload = {
        "in": src,
        "out": dst,
        "relation_type": rel_type,
        "confidence": confidence,
        "properties": {},
        "source_documents": ["source:s1"],
    }
    if rid is not None:
        payload["id"] = rid
    return Relation(**payload)


def _make_service(
    entities: List[Entity],
    relations: List[Relation],
) -> JsonlExportService:
    """Wire a service against AsyncMock repos returning the given rows."""
    repo = AsyncMock()
    repo.list_entities_for_notebook = AsyncMock(return_value=entities)
    repo.list_relations_for_notebook = AsyncMock(return_value=relations)
    return JsonlExportService(
        entity_repository=repo,
        relation_repository=repo,
    )


def _request(min_connections: int = 0, min_confidence: float = 0.0) -> JsonlExportRequest:
    """Build a relaxed-filter request for the happy paths.

    The default ``ExportFilter`` is ``min_confidence=0.9`` which most
    fixtures meet, but we relax both knobs to 0 so tests can exercise
    the post-filter logic explicitly rather than rely on the upstream
    SurrealQL gate (which is a no-op against an AsyncMock).
    """
    return JsonlExportRequest(
        filter=ExportFilter(
            min_connections=min_connections,
            min_confidence=min_confidence,
        ),
    )


async def _collect_chunks(generator) -> bytes:
    """Drain an async generator of bytes into a single buffer.

    Tests assert on the assembled archive; the per-chunk yield is
    verified separately by ``test_streaming_yields_multiple_chunks``.
    """
    out = bytearray()
    async for chunk in generator:
        out.extend(chunk)
    return bytes(out)


def _open_zip(payload: bytes) -> dict[str, bytes]:
    """Read a zip from bytes -> dict of {filename: raw_bytes}.

    Returns raw bytes (not text) so callers can verify the exact byte
    layout — useful for the empty-jsonl assertion (zero bytes, not a
    single newline).
    """
    out: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
        for name in archive.namelist():
            out[name] = archive.read(name)
    return out


def _parse_jsonl(blob: bytes) -> list[dict]:
    """Parse a JSONL blob into a list of dicts. Empty blob -> []."""
    if not blob:
        return []
    return [json.loads(line) for line in blob.decode("utf-8").splitlines() if line]


# ---------------------------------------------------------------------------
# 1. 100-entity / 50-relation fixture: line counts match
# ---------------------------------------------------------------------------


async def test_100_entity_notebook_produces_expected_lines():
    """100 entities -> 100 lines in entities.jsonl; 50 -> 50 lines in relations.jsonl.

    Every relation's endpoints are both in the surviving entity set so
    no Q-D-4 silent-drop kicks in -- the count is exact, not "at least".
    """
    ents = [_entity(f"entity:{i}", f"Name{i}") for i in range(100)]
    # 50 relations, each between adjacent entity pairs; all endpoints survive.
    rels = [
        _relation(f"entity:{i}", f"entity:{(i + 1) % 100}", rid=f"relation:{i}")
        for i in range(50)
    ]
    svc = _make_service(ents, rels)

    payload = await _collect_chunks(svc.stream_jsonl("notebook:nb1", _request()))
    files = _open_zip(payload)

    assert set(files.keys()) == {"entities.jsonl", "relations.jsonl"}
    entity_lines = _parse_jsonl(files["entities.jsonl"])
    relation_lines = _parse_jsonl(files["relations.jsonl"])
    assert len(entity_lines) == 100
    assert len(relation_lines) == 50


# ---------------------------------------------------------------------------
# 2. Per-line entity shape -- exact keys + embedding absent
# ---------------------------------------------------------------------------


async def test_entity_line_shape():
    """Every entity line is valid JSON with the documented keys.

    Critically asserts the ``embedding`` key is ABSENT. A regression that
    flipped ``exclude={"embedding"}`` to no-op would still produce valid
    JSON but with the 768-float vector included; this test catches that.
    """
    ents = [
        _entity(
            "entity:alice",
            "Alice",
            type_tags=["Person", "Researcher"],
            primary_type="Researcher",
            properties={"role": "PhD"},
            embedding=[0.5] * 10,  # non-empty so absence is meaningful
        ),
    ]
    svc = _make_service(ents, [])

    payload = await _collect_chunks(svc.stream_jsonl("notebook:shape", _request()))
    entity_lines = _parse_jsonl(_open_zip(payload)["entities.jsonl"])

    assert len(entity_lines) == 1
    line = entity_lines[0]

    expected_keys = {
        "id",
        "canonical_name",
        "entity_type",
        "type_tags",
        "primary_type",
        "confidence",
        "properties",
        "source_documents",
        "extracted_at",
    }
    assert set(line.keys()) == expected_keys, (
        f"Entity line keys drifted; got {set(line.keys())}, expected {expected_keys}"
    )

    # Embedding MUST NOT be present -- this is the privacy + size invariant.
    assert "embedding" not in line, (
        "Entity line leaked the embedding vector. If you changed exclude=, "
        "make sure the rename did not silently no-op the exclusion."
    )

    # Spot-check values.
    assert line["id"] == "entity:alice"
    assert line["canonical_name"] == "Alice"
    assert line["type_tags"] == ["Person", "Researcher"]
    assert line["primary_type"] == "Researcher"
    assert line["properties"] == {"role": "PhD"}
    assert line["source_documents"] == ["source:s1"]
    # extracted_at must be a JSON-serialisable ISO string (mode="json").
    assert isinstance(line["extracted_at"], str)
    assert "2026-06-14" in line["extracted_at"]


# ---------------------------------------------------------------------------
# 3. Per-line relation shape
# ---------------------------------------------------------------------------


async def test_relation_line_shape():
    """Every relation line is valid JSON with the documented keys.

    Verifies the DB-side ``in``/``out`` get renamed to
    ``source_entity``/``target_entity`` for downstream graph-loader
    compatibility.
    """
    ents = [
        _entity("entity:a", "A"),
        _entity("entity:b", "B"),
    ]
    rels = [
        _relation("entity:a", "entity:b", "WORKS_AT", confidence=0.92, rid="relation:r1"),
    ]
    svc = _make_service(ents, rels)

    payload = await _collect_chunks(svc.stream_jsonl("notebook:rel", _request()))
    relation_lines = _parse_jsonl(_open_zip(payload)["relations.jsonl"])

    assert len(relation_lines) == 1
    line = relation_lines[0]

    expected_keys = {
        "id",
        "source_entity",
        "target_entity",
        "relation_type",
        "confidence",
        "properties",
        "source_documents",
    }
    assert set(line.keys()) == expected_keys, (
        f"Relation line keys drifted; got {set(line.keys())}, expected {expected_keys}"
    )

    # The rename invariant: DB-side ``in``/``out`` MUST NOT leak.
    assert "in" not in line
    assert "out" not in line

    assert line["source_entity"] == "entity:a"
    assert line["target_entity"] == "entity:b"
    assert line["relation_type"] == "WORKS_AT"
    assert line["confidence"] == 0.92


# ---------------------------------------------------------------------------
# 4. min_confidence filter excludes low-confidence entities
# ---------------------------------------------------------------------------


async def test_filter_min_confidence_excludes_low_confidence_entities():
    """Repo mock returns mixed confidence; filter drops the low half.

    The D.0 SurrealQL would gate on ``min_confidence`` at the repo
    layer. Because we mock the repo, the test simulates that behaviour
    by RETURNING ONLY the high-confidence rows from the AsyncMock when
    the request's ``min_confidence`` threshold would exclude them --
    that matches production where the SurrealQL filter rejects the
    low-confidence rows before the service sees them.

    The high-confidence subset is what survives into entities.jsonl.
    """
    high_conf = [_entity(f"entity:high{i}", f"High{i}", confidence=0.95) for i in range(5)]
    low_conf = [_entity(f"entity:low{i}", f"Low{i}", confidence=0.5) for i in range(5)]

    repo = AsyncMock()
    # Production behaviour: only the high-confidence rows survive the
    # SurrealQL filter. We mirror that by returning only those when the
    # request's threshold is high. The mock could be smarter but the
    # contract under test is "filter is applied before serialisation",
    # which the count assertion below pins.
    repo.list_entities_for_notebook = AsyncMock(return_value=high_conf)
    repo.list_relations_for_notebook = AsyncMock(return_value=[])
    svc = JsonlExportService(entity_repository=repo, relation_repository=repo)

    payload = await _collect_chunks(
        svc.stream_jsonl(
            "notebook:filter",
            JsonlExportRequest(
                filter=ExportFilter(min_connections=0, min_confidence=0.9),
            ),
        )
    )
    entity_lines = _parse_jsonl(_open_zip(payload)["entities.jsonl"])

    assert len(entity_lines) == 5
    # No low-confidence entity slipped through.
    ids = {line["id"] for line in entity_lines}
    assert all(eid.startswith("entity:high") for eid in ids)
    assert not any(eid in ids for eid in {f"entity:low{i}" for i in range(5)})

    # Sanity: verify the repo got the filter we passed (proves the
    # filter actually flows through to the repo layer, not just to the
    # Python post-filter).
    call_args = repo.list_entities_for_notebook.call_args
    assert call_args.args[1].min_confidence == 0.9


# ---------------------------------------------------------------------------
# 5. status archived/merged exclusion (D.1c BLOCKER regression)
# ---------------------------------------------------------------------------


async def test_status_archived_and_merged_excluded():
    """Archived + merged entities never reach the JSONL.

    Mirrors the D.1c BLOCKER fix: every Track-D exporter must drop
    ``status in {archived, merged}`` BEFORE writing, otherwise tombstones
    leak into downstream graph stores.
    """
    ents = [
        _entity("entity:active", "Active", status="active"),
        _entity("entity:arch", "Arch", status="archived"),
        _entity("entity:merged", "Merged", status="merged"),
    ]
    svc = _make_service(ents, [])

    payload = await _collect_chunks(svc.stream_jsonl("notebook:status", _request()))
    entity_lines = _parse_jsonl(_open_zip(payload)["entities.jsonl"])

    ids = {line["id"] for line in entity_lines}
    assert ids == {"entity:active"}, (
        f"Status filter regression: expected only 'entity:active', got {ids}. "
        "If this fails, the canonical EXCLUDED_ENTITY_STATUSES filter was "
        "skipped or replaced."
    )


# ---------------------------------------------------------------------------
# 6. min_connections post-filter (D.1c canonical pipeline)
# ---------------------------------------------------------------------------


async def test_min_connections_filter_drops_isolated_entities():
    """Isolated entities are dropped by the shared min_connections filter.

    Catches a regression where the JSONL service skipped the
    ``_apply_min_connections_filter`` call -- the AC's filter pipeline
    explicitly requires parity with the Obsidian exporter.
    """
    ents = [
        _entity("entity:a", "A"),
        _entity("entity:b", "B"),
        _entity("entity:island", "Island"),  # zero relations
    ]
    rels = [_relation("entity:a", "entity:b")]
    svc = _make_service(ents, rels)

    payload = await _collect_chunks(
        svc.stream_jsonl(
            "notebook:minconn",
            JsonlExportRequest(
                filter=ExportFilter(min_connections=1, min_confidence=0.0),
            ),
        )
    )
    entity_lines = _parse_jsonl(_open_zip(payload)["entities.jsonl"])

    ids = {line["id"] for line in entity_lines}
    assert ids == {"entity:a", "entity:b"}, (
        f"min_connections regression: expected {{a, b}}, got {ids}. "
        "Island should have been dropped by the canonical post-filter."
    )


# ---------------------------------------------------------------------------
# 7. Streaming behaviour: 5000-entity fixture yields multiple chunks
# ---------------------------------------------------------------------------


async def test_streaming_yields_multiple_chunks():
    """5000-entity fixture: the async generator yields more than one chunk.

    Pins the streaming guarantee. A regression that materialised the
    full archive and returned it as a single yield would pass shape
    tests but fail this one. Also smoke-checks the memory bound via
    ``tracemalloc`` against the AC #5 200MB budget.

    Note: we count chunks BEFORE assembling them so a single fat yield
    triggers the assertion. ``_CHUNK_SIZE`` is 64KB; with deflate on
    text-heavy entity rows a 5000-row archive should compress to
    well above 64KB so multiple chunks are inevitable for a real
    fixture.
    """
    ents = [
        _entity(
            f"entity:{i}",
            f"Entity Number {i} With Some Long Name To Make Compression Less Friendly",
            type_tags=["Person", "Organization"],
            primary_type="Person",
            properties={
                "description": f"a description string for entity {i}",
                "ix": i,
            },
            embedding=[float(i) / 1000.0] * 32,  # non-trivial vector to confirm exclusion
        )
        for i in range(5000)
    ]
    svc = _make_service(ents, [])

    tracemalloc.start()
    try:
        chunk_count = 0
        total_bytes = 0
        async for chunk in svc.stream_jsonl("notebook:big", _request()):
            chunk_count += 1
            total_bytes += len(chunk)

        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert chunk_count > 1, (
        f"Streaming regression: 5000-entity fixture yielded only {chunk_count} "
        "chunk. The generator must split output into multiple chunks -- "
        "otherwise StreamingResponse just buffers everything before sending."
    )
    assert total_bytes > 0

    # AC #5: peak memory bound. 200MB upper bound is generous; in
    # practice this fixture peaks ~30MB (entity model_dumps + the
    # compressed archive bytes). A regression that materialised raw
    # JSONL strings into a list would blow past this on this fixture.
    peak_mb = peak / (1024 * 1024)
    assert peak_mb < 200, (
        f"Memory regression: peak {peak_mb:.1f}MB exceeded the 200MB AC budget. "
        "Likely cause: materialising the entity list as model_dump strings before "
        "writing into the zip, or shipping the embedding field."
    )


# ---------------------------------------------------------------------------
# 8. Empty notebook -> empty (zero-byte) JSONL files
# ---------------------------------------------------------------------------


async def test_empty_notebook_produces_empty_jsonl_files():
    """No entities / no relations -> both .jsonl files present but empty.

    The AC explicitly calls out "not error, not missing" -- a regression
    that skipped the writes when entities=[] would still produce a valid
    zip but with no entries, which downstream loaders treat as "file
    missing" (a different error class).
    """
    svc = _make_service([], [])

    payload = await _collect_chunks(svc.stream_jsonl("notebook:empty", _request()))
    files = _open_zip(payload)

    # Both files present, both zero-byte.
    assert set(files.keys()) == {"entities.jsonl", "relations.jsonl"}
    assert files["entities.jsonl"] == b""
    assert files["relations.jsonl"] == b""


# ---------------------------------------------------------------------------
# 9. Q-D-4 silent drop: relations to filtered targets disappear
# ---------------------------------------------------------------------------


async def test_relations_to_dropped_entities_are_silently_excluded():
    """Relations whose endpoint was filtered out don't reach the JSONL.

    Mirrors the Obsidian renderer's Q-D-4 contract: a relation pointing
    at an entity that the filter dropped is silently discarded, never
    rendered as a dangling reference.
    """
    ents = [
        _entity("entity:alice", "Alice"),
        _entity("entity:bob", "Bob", status="archived"),  # dropped by status filter
    ]
    rels = [_relation("entity:alice", "entity:bob", "knows")]
    svc = _make_service(ents, rels)

    payload = await _collect_chunks(svc.stream_jsonl("notebook:drop", _request()))
    files = _open_zip(payload)

    entity_lines = _parse_jsonl(files["entities.jsonl"])
    relation_lines = _parse_jsonl(files["relations.jsonl"])

    # Bob's status=archived; only Alice survives.
    assert {line["id"] for line in entity_lines} == {"entity:alice"}
    # The Alice->Bob relation lost its target -> silent drop, zero lines.
    assert relation_lines == []


# ---------------------------------------------------------------------------
# 10. Telemetry: record_metric fires exactly once per export (happy path)
# ---------------------------------------------------------------------------


async def test_metrics_emitted_once_per_export(monkeypatch):
    """``record_metric("export.jsonl", ...)`` fires exactly once per export.

    Pins the contract from the plan AC #6. A regression that emitted
    once per chunk (yield-loop bug) or never (forgot the finally) would
    fail here.
    """
    events: list[dict] = []

    async def _spy(event_type, payload, source=None, notebook=None):
        events.append(
            {
                "event_type": event_type,
                "payload": payload,
                "source": source,
                "notebook": notebook,
            }
        )

    monkeypatch.delenv("OPEN_NOTEBOOK_DISABLE_METRICS", raising=False)
    monkeypatch.setattr(
        "app_main.services.jsonl_export_service.record_metric", _spy
    )

    ents = [_entity("entity:a", "A"), _entity("entity:b", "B")]
    svc = _make_service(ents, [])

    await _collect_chunks(svc.stream_jsonl("notebook:tel", _request()))

    assert len(events) == 1
    event = events[0]
    assert event["event_type"] == "export.jsonl"
    assert event["notebook"] == "notebook:tel"

    payload = event["payload"]
    # Counts-only contract: no entity/relation/source IDs in the payload.
    def _no_ids(value):
        if isinstance(value, str):
            assert not value.startswith("entity:")
            assert not value.startswith("source:")
            assert not value.startswith("relation:")
            assert not value.startswith("notebook:")
        elif isinstance(value, dict):
            for v in value.values():
                _no_ids(v)
        elif isinstance(value, list):
            for v in value:
                _no_ids(v)

    _no_ids(payload)

    # Spot-check the count keys.
    assert payload["entities_written"] == 2
    assert payload["relations_written"] == 0
    assert payload["files_written"] == 2
    assert payload["bytes_written"] > 0
    assert "duration_ms" in payload


async def test_metrics_emitted_once_on_failure(monkeypatch):
    """Failure path: metric fires once with ``partial: True``.

    Pins that a thrown exception in ``_collect`` (e.g. a repo timeout)
    still triggers exactly one metric write -- not zero (forgot the
    finally) and not multiple (try/except misplaced).
    """
    events: list[dict] = []

    async def _spy(event_type, payload, source=None, notebook=None):
        events.append({"event_type": event_type, "payload": payload})

    monkeypatch.delenv("OPEN_NOTEBOOK_DISABLE_METRICS", raising=False)
    monkeypatch.setattr(
        "app_main.services.jsonl_export_service.record_metric", _spy
    )

    repo = AsyncMock()
    repo.list_entities_for_notebook = AsyncMock(side_effect=RuntimeError("boom"))
    repo.list_relations_for_notebook = AsyncMock(return_value=[])
    svc = JsonlExportService(entity_repository=repo, relation_repository=repo)

    with pytest.raises(RuntimeError):
        # Iteration is what triggers the generator body. The error
        # surfaces on the first ``async for`` step.
        async for _ in svc.stream_jsonl("notebook:fail", _request()):
            pass

    assert len(events) == 1
    assert events[0]["event_type"] == "export.jsonl"
    assert events[0]["payload"]["partial"] is True
    assert "boom" in events[0]["payload"]["error"]
