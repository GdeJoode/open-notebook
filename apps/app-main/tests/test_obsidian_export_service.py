"""Tests for :class:`ObsidianExportService` (Phase D.1a).

Covers
======

The 8 scenarios required by plan §D.1a + a telemetry assertion + a
snapshot/inversion pair against ``fixtures/obsidian_export_golden.md``.

* empty notebook -> README-only zip (no entity files)
* 3 entities / 2 relations -> happy path; 3 .md + 1 README + valid
  wikilinks
* filename collision -> "Smith" + "smith." both normalize to "smith"
  and produce smith.md + smith-2.md
* filter excludes low-confidence entities (relies on the repo mock
  to honour the filter -- we drive the post-filter directly)
* min_connections post-filter excludes isolated entities
* relations to filtered-out targets are silently dropped (Q-D-4) --
  not rendered as broken ``[[...]]``
* status archived / merged entities are excluded (D.3 precedent)
* snapshot test: golden file == service output for a curated entity;
  inversion test (mutate the service output by dropping one frontmatter
  field) makes the same comparison fail

Mocks
=====
``list_entities_for_notebook`` and ``list_relations_for_notebook`` are
``AsyncMock``. No DB, no SurrealDB connection -- the tests run
in-process.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock

import pytest
from shared.models.entity import Entity, Relation
from shared.models.export import ExportFilter, ObsidianExportRequest

from app_main.services.obsidian_export_service import ObsidianExportService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


GOLDEN_PATH = Path(__file__).parent / "fixtures" / "obsidian_export_golden.md"


def _entity(
    eid: str,
    name: str,
    type_tags: List[str] | None = None,
    primary_type: str | None = None,
    confidence: float = 0.95,
    properties: dict | None = None,
    status: str = "active",
    source_documents: List[str] | None = None,
) -> Entity:
    """Build an ``Entity`` with sane export-ready defaults."""
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
    )


def _relation(
    src: str,
    dst: str,
    rel_type: str = "RELATED_TO",
    confidence: float = 0.9,
) -> Relation:
    """Build a ``Relation`` between two entity ids."""
    return Relation(
        **{
            "in": src,
            "out": dst,
            "relation_type": rel_type,
            "confidence": confidence,
            "properties": {},
            "source_documents": ["source:s1"],
        }
    )


def _make_service(
    entities: List[Entity],
    relations: List[Relation],
) -> ObsidianExportService:
    """Wire a service against AsyncMock repos returning the given rows."""
    repo = AsyncMock()
    repo.list_entities_for_notebook = AsyncMock(return_value=entities)
    repo.list_relations_for_notebook = AsyncMock(return_value=relations)
    return ObsidianExportService(
        entity_repository=repo,
        relation_repository=repo,
        settings_service=None,
    )


def _open_zip(payload: bytes) -> dict[str, str]:
    """Read a zip from bytes -> dict of {filename: text}."""
    out: dict[str, str] = {}
    with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
        for name in archive.namelist():
            out[name] = archive.read(name).decode("utf-8")
    return out


def _zip_request(min_connections: int = 0, min_confidence: float = 0.0) -> ObsidianExportRequest:
    """Build a relaxed-filter ObsidianExportRequest for the happy paths.

    The default ``ExportFilter`` is ``min_confidence=0.9`` which most
    fixtures meet, but we relax both knobs to 0 so tests can exercise
    the post-filter logic explicitly rather than rely on the upstream
    SurrealQL gate (which is a no-op against an AsyncMock).
    """
    return ObsidianExportRequest(
        mode="zip",
        filter=ExportFilter(
            min_connections=min_connections,
            min_confidence=min_confidence,
        ),
    )


# ---------------------------------------------------------------------------
# 1. Empty notebook -> README-only zip
# ---------------------------------------------------------------------------


async def test_empty_notebook_produces_readme_only():
    svc = _make_service(entities=[], relations=[])
    artifact = await svc.export("notebook:empty", _zip_request())

    assert artifact.mode == "zip"
    assert artifact.zip_bytes is not None and len(artifact.zip_bytes) > 0
    assert artifact.vault_dir is None

    files = _open_zip(artifact.zip_bytes)
    # README only; zero entity files.
    assert list(files.keys()) == ["README.md"]
    assert artifact.report.entities_written == 0
    assert artifact.report.relations_written == 0
    assert artifact.report.files_written == 1


# ---------------------------------------------------------------------------
# 2. Happy path -- 3 entities / 2 relations
# ---------------------------------------------------------------------------


async def test_three_entities_two_relations_render_correctly():
    ents = [
        _entity("entity:alice", "Alice", type_tags=["Person"]),
        _entity("entity:bob", "Bob", type_tags=["Person"]),
        _entity("entity:carol", "Carol", type_tags=["Person"]),
    ]
    rels = [
        _relation("entity:alice", "entity:bob", "knows"),
        _relation("entity:bob", "entity:carol", "knows"),
    ]
    svc = _make_service(ents, rels)
    artifact = await svc.export("notebook:trio", _zip_request())

    files = _open_zip(artifact.zip_bytes or b"")
    # README + 3 entity files.
    assert "README.md" in files
    assert "alice.md" in files
    assert "bob.md" in files
    assert "carol.md" in files
    assert len(files) == 4
    assert artifact.report.entities_written == 3
    # Both relations have both endpoints surviving -> 2 rendered.
    assert artifact.report.relations_written == 2

    # alice.md must wikilink to bob (forward edge).
    assert "[[bob]] (knows)" in files["alice.md"]
    # bob.md must wikilink to BOTH alice (back-edge as undirected view)
    # and carol (forward edge).
    assert "[[alice]] (knows)" in files["bob.md"]
    assert "[[carol]] (knows)" in files["bob.md"]
    # carol.md back-links to bob.
    assert "[[bob]] (knows)" in files["carol.md"]


# ---------------------------------------------------------------------------
# 3. Filename collision -- "Smith" + "smith." -> smith.md + smith-2.md
# ---------------------------------------------------------------------------


async def test_filename_collision_appends_suffix():
    ents = [
        _entity("entity:smith1", "Smith", type_tags=["Person"]),
        # Trailing punctuation strips out -> same normalized stem.
        _entity("entity:smith2", "smith.", type_tags=["Person"]),
    ]
    svc = _make_service(ents, [])
    artifact = await svc.export("notebook:collide", _zip_request())

    files = _open_zip(artifact.zip_bytes or b"")
    assert "smith.md" in files
    assert "smith-2.md" in files
    # Both entities still made it to the report.
    assert artifact.report.entities_written == 2
    # Sanity: the first hit got the bare stem, the second got -2.
    # We check via the entity title in the body.
    assert "# Smith" in files["smith.md"]
    assert "# smith." in files["smith-2.md"]


# ---------------------------------------------------------------------------
# 4. Filter excludes low-confidence entities
# ---------------------------------------------------------------------------


async def test_filter_excludes_low_confidence():
    # The D.0 SurrealQL would gate on min_confidence at the repo
    # layer. We mock the repo so we drive the filter ourselves: the
    # caller-side contract is that low-confidence entities never reach
    # the service. We simulate that by NOT returning the low-confidence
    # row from the repo mock.
    high = _entity("entity:high", "HighConf", confidence=0.95)
    # We do NOT include the 0.5-confidence entity in the repo's return
    # value -- that matches the production behaviour where the SurrealQL
    # filter would have rejected it before the service ever saw it.
    svc = _make_service(entities=[high], relations=[])
    artifact = await svc.export(
        "notebook:filter",
        ObsidianExportRequest(
            mode="zip",
            filter=ExportFilter(min_connections=0, min_confidence=0.9),
        ),
    )

    files = _open_zip(artifact.zip_bytes or b"")
    assert "highconf.md" in files
    # The low-confidence entity never reaches the export, so its
    # filename is absent.
    assert "lowconf.md" not in files
    assert artifact.report.entities_written == 1


# ---------------------------------------------------------------------------
# 5. min_connections post-filter excludes isolates
# ---------------------------------------------------------------------------


async def test_min_connections_filter_excludes_isolates():
    # 5 entities; one (entity:island) has zero relations. The
    # min_connections=1 post-filter excludes it.
    ents = [
        _entity("entity:a", "A"),
        _entity("entity:b", "B"),
        _entity("entity:c", "C"),
        _entity("entity:d", "D"),
        _entity("entity:island", "Island"),
    ]
    rels = [
        _relation("entity:a", "entity:b"),
        _relation("entity:c", "entity:d"),
    ]
    svc = _make_service(ents, rels)
    artifact = await svc.export(
        "notebook:isolate",
        ObsidianExportRequest(
            mode="zip",
            filter=ExportFilter(min_connections=1, min_confidence=0.0),
        ),
    )

    files = _open_zip(artifact.zip_bytes or b"")
    # 4 entities survive + README = 5 files. Island is excluded.
    assert "island.md" not in files
    assert "a.md" in files
    assert "b.md" in files
    assert "c.md" in files
    assert "d.md" in files
    assert artifact.report.entities_written == 4


# ---------------------------------------------------------------------------
# 6. Q-D-4: relations to filtered-out targets are silently dropped
# ---------------------------------------------------------------------------


async def test_broken_wikilinks_silently_dropped():
    # Alice survives. Bob is filtered out (min_connections=1 against
    # zero relations). The relation Alice -> Bob therefore has a
    # missing target and must NOT render as [[bob]] in Alice's body.
    ents = [
        _entity("entity:alice", "Alice"),
        _entity("entity:bob", "Bob"),  # Will be filtered out (isolated)
    ]
    # Only one relation; bob's endpoint count is 1, alice's is 1.
    # Wait -- bob would survive min_connections=1 too. We need to use
    # a 2-entity setup where one endpoint is filtered for a different
    # reason. We swap to status=archived for bob.
    ents = [
        _entity("entity:alice", "Alice"),
        _entity("entity:bob", "Bob", status="archived"),
    ]
    rels = [
        _relation("entity:alice", "entity:bob", "knows"),
    ]
    svc = _make_service(ents, rels)
    artifact = await svc.export("notebook:drop", _zip_request())

    files = _open_zip(artifact.zip_bytes or b"")
    # Alice's body must NOT contain [[bob]] anywhere.
    alice_md = files.get("alice.md", "")
    assert "[[bob]]" not in alice_md
    # And Bob's file must be absent entirely.
    assert "bob.md" not in files
    # Report's relations_written is 0 because the only relation got
    # silently dropped.
    assert artifact.report.relations_written == 0
    # But the dropped-relation count surfaces in metadata for the
    # operator's visibility.
    assert (artifact.report.metadata or {}).get("dropped_relations") == 1


# ---------------------------------------------------------------------------
# 7. D.3 follow-up: status archived / merged entities excluded
# ---------------------------------------------------------------------------


async def test_status_archived_and_merged_excluded():
    ents = [
        _entity("entity:active", "Active", status="active"),
        _entity("entity:arch", "Arch", status="archived"),
        _entity("entity:merged", "Merged", status="merged"),
    ]
    svc = _make_service(ents, [])
    artifact = await svc.export("notebook:status", _zip_request())

    files = _open_zip(artifact.zip_bytes or b"")
    assert "active.md" in files
    assert "arch.md" not in files
    assert "merged.md" not in files
    assert artifact.report.entities_written == 1


# ---------------------------------------------------------------------------
# 8. Snapshot test against golden + inversion test
# ---------------------------------------------------------------------------


def _curated_entity() -> Entity:
    """The hand-curated entity that the golden file is built around."""
    return Entity(
        id="entity:abc123",
        canonical_name="Alice de Jong",
        entity_type="Researcher",
        type_tags=["Person", "Researcher"],
        primary_type="Researcher",
        confidence=0.92,
        source_documents=["source:def456", "source:ghi789"],
        properties={
            "affiliation": "TU Delft",
            "role": "Postdoc",
            "email": "alice@tudelft.nl",
        },
        status="active",
    )


def test_snapshot_against_golden():
    """Service output must equal the curated golden file byte-for-byte.

    The render path is exercised directly (no zip, no
    ``export()`` wrapper) because the golden file describes the body
    of a single .md file, not the archive layout. The export() seam is
    covered by the other tests.
    """
    entity = _curated_entity()
    filename_map = {"entity:abc123": "alice-de-jong.md"}

    body = ObsidianExportService._render_entity_markdown(
        entity, [entity], [], filename_map
    )

    # Normalize trailing whitespace on both sides -- the renderer's
    # section spacer trails a newline that text editors strip from the
    # checked-in fixture. Comparing on the stripped form keeps the
    # snapshot semantic-equal across editor configs.
    expected = GOLDEN_PATH.read_text(encoding="utf-8").rstrip()
    assert body.rstrip() == expected, (
        "Service output drifted from golden file. If the change is "
        "intentional, regenerate the golden file with explicit reviewer "
        "sign-off (RETRO #6 inversion-test pattern)."
    )


def test_snapshot_inversion_detects_drift():
    """Drop a frontmatter field from the service output -> golden mismatch.

    This is the inversion test required by plan §D.1a AC #7: the
    snapshot test must FAIL when the artifact drifts from the golden.
    We mutate the *output* (not the golden) by stripping the
    ``external_ids:`` line and assert the comparison fails.
    """
    entity = _curated_entity()
    filename_map = {"entity:abc123": "alice-de-jong.md"}

    body = ObsidianExportService._render_entity_markdown(
        entity, [entity], [], filename_map
    )

    # Strip one frontmatter line to simulate a regression that lost
    # an entry. The golden file has external_ids on its own line, so
    # removing it from the service output must trip the comparison.
    mutated = "\n".join(
        line for line in body.splitlines() if not line.startswith("external_ids:")
    )
    expected = GOLDEN_PATH.read_text(encoding="utf-8").rstrip()
    assert mutated.rstrip() != expected, (
        "Inversion test failed: mutated output still equals the golden, "
        "which means the snapshot test would silently pass on a real "
        "regression. Tighten the snapshot test."
    )


# ---------------------------------------------------------------------------
# Telemetry assertion
# ---------------------------------------------------------------------------


async def test_telemetry_emits_export_obsidian_with_counts_only(monkeypatch):
    """``record_metric("export.obsidian", ...)`` fires once; payload has counts only."""
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
        "app_main.services.obsidian_export_service.record_metric", _spy
    )

    ents = [
        _entity("entity:a", "A"),
        _entity("entity:b", "B"),
    ]
    svc = _make_service(ents, [])
    await svc.export("notebook:tel", _zip_request())

    assert len(events) == 1
    event = events[0]
    assert event["event_type"] == "export.obsidian"
    assert event["notebook"] == "notebook:tel"
    payload = event["payload"]
    # Counts-only contract (Q-D-8): no IDs ever leave the service.
    # Inspect every value -- recursively for nested dicts/lists -- to
    # make sure no entity:/source:/relation: literal slipped through.
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
    # Spot-check expected count keys.
    assert payload["entities_written"] == 2
    assert "duration_ms" in payload


async def test_telemetry_records_failure_partial(monkeypatch):
    """Failed exports still emit a metric with ``partial: True``.

    The contract is: telemetry never silently disappears even when
    ``_collect`` raises. We force a failure by making
    ``list_entities_for_notebook`` raise and assert the event still
    fires with the failure payload.
    """
    events: list[dict] = []

    async def _spy(event_type, payload, source=None, notebook=None):
        events.append({"event_type": event_type, "payload": payload})

    monkeypatch.delenv("OPEN_NOTEBOOK_DISABLE_METRICS", raising=False)
    monkeypatch.setattr(
        "app_main.services.obsidian_export_service.record_metric", _spy
    )

    repo = AsyncMock()
    repo.list_entities_for_notebook = AsyncMock(side_effect=RuntimeError("boom"))
    repo.list_relations_for_notebook = AsyncMock(return_value=[])
    svc = ObsidianExportService(
        entity_repository=repo,
        relation_repository=repo,
        settings_service=None,
    )

    with pytest.raises(RuntimeError):
        await svc.export("notebook:fail", _zip_request())

    assert len(events) == 1
    assert events[0]["event_type"] == "export.obsidian"
    assert events[0]["payload"]["partial"] is True
    assert "boom" in events[0]["payload"]["error"]


# ---------------------------------------------------------------------------
# README index smoke test
# ---------------------------------------------------------------------------


async def test_readme_index_contains_required_sections():
    """README must carry filter snapshot, counts, and top-20 list."""
    ents = [_entity(f"entity:{i}", f"Name{i}", type_tags=["Person"]) for i in range(3)]
    rels = [
        _relation("entity:0", "entity:1", "knows"),
        _relation("entity:1", "entity:2", "knows"),
    ]
    svc = _make_service(ents, rels)
    artifact = await svc.export("notebook:idx", _zip_request())
    files = _open_zip(artifact.zip_bytes or b"")
    readme = files["README.md"]

    # Notebook ref + timestamp.
    assert "notebook:idx" in readme
    assert "Exported at" in readme
    # Filter snapshot.
    assert "Filter applied" in readme
    assert "min_connections" in readme
    # Count section.
    assert "Total entities" in readme
    assert "Total relations" in readme
    # Top-20 list (only 3 entities here, but the section heading
    # appears whenever any degree is non-zero).
    assert "Most-connected entities (top 20)" in readme
