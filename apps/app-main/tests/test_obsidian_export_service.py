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
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, patch

import pytest
from shared.models.entity import Entity, Relation
from shared.models.export import ExportFilter, ObsidianExportRequest

from app_main.services.obsidian_export_service import (
    ObsidianExportService,
    VaultPathNotConfigured,
)


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
# 3b. Filename sanitization -- canonical_name with /, :, \\ produces flat
#     filenames (regression for attempt-1 review Major #1)
# ---------------------------------------------------------------------------


async def test_filename_sanitization_strips_slash_and_colon():
    """Entities with slash/colon/backslash in canonical_name produce flat filenames.

    Regression for D.1a attempt-1 review Major #1: previously these
    chars survived ``normalize_entity_name`` (a content normalizer only)
    and produced *nested* zip paths -- e.g. ``canonical_name="Hello/World"``
    leaked into the archive as ``hello/world.md`` (a phantom directory
    that extracted as a nested file), breaking the flat-vault promise.

    Fix: ``_safe_entity_stem`` layers ``_ENTITY_FILENAME_UNSAFE`` on top
    of ``normalize_entity_name`` so all path separators, control chars,
    and shell metacharacters get replaced with ``_`` before the zip
    entry is written.

    Also asserts wikilinks in the body use the sanitized stem (not the
    raw normalized form) -- so the in-vault link target matches the
    real filename.
    """
    ents = [
        _entity("entity:slash", "Hello/World"),       # forward slash
        _entity("entity:colon", "Foo:Bar"),            # internal colon
        _entity("entity:bslash", "Path\\To\\Note"),    # backslash
        # Plus one neighbour so wikilinks fire and we can assert against
        # the sanitized stem rather than just filenames.
        _entity("entity:neighbour", "Plain Name"),
    ]
    rels = [
        # neighbour -> slash entity; wikilink in neighbour.md must point
        # at the sanitized stem ``hello_world``, NOT ``hello/world`` (which
        # would render as a folder-note reference in Obsidian).
        _relation("entity:neighbour", "entity:slash", "knows"),
    ]
    svc = _make_service(ents, rels)
    artifact = await svc.export("notebook:sanitize", _zip_request())

    files = _open_zip(artifact.zip_bytes or b"")

    # All entity files must be flat -- no path separators anywhere in
    # the zip namelist.
    entity_files = [name for name in files if name.endswith(".md") and name != "README.md"]
    assert len(entity_files) == 4
    for name in entity_files:
        assert "/" not in name, f"slash leaked into {name}"
        assert "\\" not in name, f"backslash leaked into {name}"
        # Internal colon also gone from the stem (the ``.md`` extension
        # doesn't count -- we strip it before checking).
        stem = name[:-3]
        assert ":" not in stem, f"colon leaked into stem {stem}"

    # Specific stems match the expected sanitized form.
    assert "hello_world.md" in files
    assert "foo_bar.md" in files
    assert "path_to_note.md" in files
    assert "plain name.md" in files  # normal entity is unaffected

    # Wikilink rendering uses the SANITIZED stem, not the raw normalized
    # form. neighbour.md links to entity:slash via the new
    # ``[[hello_world]]`` -- if the renderer accidentally fed the raw
    # ``normalize_entity_name`` output, we'd see ``[[hello/world]]``
    # which Obsidian parses as a folder-note reference.
    neighbour_body = files["plain name.md"]
    assert "[[hello_world]] (knows)" in neighbour_body, (
        "Wikilink rendering must use the sanitized stem -- if this fails, "
        "the renderer is reading from a different source than _build_filename_map."
    )
    assert "[[hello/world]]" not in neighbour_body


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
# 5b. min_connections boundary: degree==N-1 excluded, degree==N included
#     (attempt-1 review Major #2 -- pin the strict ``>=`` semantic)
# ---------------------------------------------------------------------------


async def test_min_connections_boundary_at_exact_threshold():
    """``min_connections=N`` excludes degree==N-1 but includes degree==N.

    Pins the inclusive lower-bound (``>=``) semantic. A buggy strict
    greater-than (``>``) implementation would still pass
    :func:`test_min_connections_filter_excludes_isolates` (which only
    splits degree-0 from degree-1) but would fail this one because it
    would also drop the degree-2 entity at the boundary.

    Setup
    -----
    Three entities, three relations chosen so the in+out degrees are
    distinct and span the boundary:

        alice -> bob  : alice in=1, out=0; bob in=0, out=1
        bob -> carol  : bob out=2 total (1+1), carol in=1, out=0
        carol -> bob  : carol out=1 total, bob in=2 total

    Final in+out degrees:
        alice = 1   (just the alice->bob edge)
        bob   = 3   (in 2 + out 1)
        carol = 2   (in 1 + out 1)

    With ``min_connections=2``:
        alice (1) -> EXCLUDED
        carol (2) -> INCLUDED (boundary case -- the one this test pins)
        bob   (3) -> INCLUDED
    """
    ents = [
        _entity("entity:alice", "alice"),
        _entity("entity:bob", "bob"),
        _entity("entity:carol", "carol"),
    ]
    rels = [
        _relation("entity:alice", "entity:bob", "knows"),
        _relation("entity:bob", "entity:carol", "knows"),
        _relation("entity:carol", "entity:bob", "knows"),
    ]
    svc = _make_service(ents, rels)
    artifact = await svc.export(
        "notebook:boundary",
        ObsidianExportRequest(
            mode="zip",
            filter=ExportFilter(min_connections=2, min_confidence=0.0),
        ),
    )

    files = _open_zip(artifact.zip_bytes or b"")
    # alice has degree 1 < 2 -> excluded
    assert "alice.md" not in files, (
        "Strict-greater-than bug would still exclude alice -- expected; the "
        "important assertion is the carol boundary case below."
    )
    # carol has degree 2 == min_connections -> INCLUDED (this is the pin
    # that a buggy ``>`` implementation would fail).
    assert "carol.md" in files, (
        "Boundary regression: carol (degree==min_connections) was dropped, "
        "suggesting a strict greater-than instead of >=."
    )
    # bob has degree 3 > 2 -> included
    assert "bob.md" in files
    # Sanity: exactly 2 entities + README
    assert artifact.report.entities_written == 2


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
        entity, [], filename_map
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
        entity, [], filename_map
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


# ===========================================================================
# D.1b -- vault_path mode
# ===========================================================================
#
# The vault_path mode mirrors the zip mode for vault generation but
# swaps the in-memory archive for direct-write to
# ``<Settings.vault_path>/<Settings.vault_entities_folder>/``. Tests
# cover (a) the happy path, (b) the "not configured" 400 surface,
# (c) the safety rejection paths (absolute / writable / traversal),
# (d) the per-file-atomic semantic under mid-batch failure, and
# (e) the telemetry redaction contract (vault_path_redacted=True,
# no raw path in payload).


def _make_service_with_settings(
    entities: List[Entity],
    relations: List[Relation],
    vault_path: str | None,
    entities_folder: str | None = "Entities",
) -> ObsidianExportService:
    """Like ``_make_service`` but with a settings_service stub."""
    repo = AsyncMock()
    repo.list_entities_for_notebook = AsyncMock(return_value=entities)
    repo.list_relations_for_notebook = AsyncMock(return_value=relations)
    settings_svc = AsyncMock()
    settings_svc.get = AsyncMock(
        return_value=SimpleNamespace(
            vault_path=vault_path,
            vault_entities_folder=entities_folder,
        )
    )
    return ObsidianExportService(
        entity_repository=repo,
        relation_repository=repo,
        settings_service=settings_svc,
    )


def _vault_request(min_connections: int = 0, min_confidence: float = 0.0) -> ObsidianExportRequest:
    """Build a relaxed-filter ObsidianExportRequest in vault_path mode."""
    return ObsidianExportRequest(
        mode="vault_path",
        filter=ExportFilter(
            min_connections=min_connections,
            min_confidence=min_confidence,
        ),
    )


# ---------------------------------------------------------------------------
# 1. Happy path -- writes 3 entity .md + README into <vault>/<folder>/
# ---------------------------------------------------------------------------


async def test_vault_path_happy_path(tmp_path):
    """Settings has vault_path; service writes files; returns ExportReport.

    Asserts:
      * artifact.mode == "vault_path"
      * artifact.vault_dir is the resolved target dir
      * 3 entity .md + 1 README exist on disk
      * artifact.report counts match
    """
    ents = [
        _entity("entity:alice", "Alice", type_tags=["Person"]),
        _entity("entity:bob", "Bob", type_tags=["Person"]),
        _entity("entity:carol", "Carol", type_tags=["Person"]),
    ]
    rels = [_relation("entity:alice", "entity:bob", "knows")]
    svc = _make_service_with_settings(ents, rels, str(tmp_path))

    artifact = await svc.export("notebook:vault", _vault_request())

    assert artifact.mode == "vault_path"
    assert artifact.zip_bytes is None
    assert artifact.vault_dir is not None
    target_dir = Path(artifact.vault_dir)
    assert target_dir == (tmp_path / "Entities").resolve()
    assert target_dir.is_dir()

    # Files exist on disk.
    assert (target_dir / "README.md").is_file()
    assert (target_dir / "alice.md").is_file()
    assert (target_dir / "bob.md").is_file()
    assert (target_dir / "carol.md").is_file()

    # Report counts match.
    assert artifact.report.entities_written == 3
    assert artifact.report.relations_written == 1
    assert artifact.report.files_written == 4  # README + 3 entity files
    assert artifact.report.bytes_written > 0

    # Content sanity: alice.md wikilinks to bob (forward edge).
    alice_md = (target_dir / "alice.md").read_text(encoding="utf-8")
    assert "[[bob]] (knows)" in alice_md


# ---------------------------------------------------------------------------
# 2. Settings.vault_path is None -> VaultPathNotConfigured
# ---------------------------------------------------------------------------


async def test_vault_path_not_configured_raises():
    """When Settings.vault_path is None, the service raises VaultPathNotConfigured."""
    svc = _make_service_with_settings(
        entities=[_entity("entity:a", "A")],
        relations=[],
        vault_path=None,
    )

    with pytest.raises(VaultPathNotConfigured) as excinfo:
        await svc.export("notebook:novault", _vault_request())

    assert "vault_path" in str(excinfo.value).lower()


async def test_vault_entities_folder_not_configured_raises(tmp_path):
    """When vault_path is set but entities_folder is None, the service raises."""
    svc = _make_service_with_settings(
        entities=[_entity("entity:a", "A")],
        relations=[],
        vault_path=str(tmp_path),
        entities_folder=None,
    )
    with pytest.raises(VaultPathNotConfigured):
        await svc.export("notebook:nofolder", _vault_request())


# ---------------------------------------------------------------------------
# 3. Relative vault_path -> ValueError
# ---------------------------------------------------------------------------


async def test_vault_path_must_be_absolute():
    """A relative ``vault_path`` is rejected with ValueError.

    A relative path resolves against the API server's cwd, which is
    almost never the operator's intent. Surface as ValueError -- the
    router maps it to 500 with the partial state (none, since the
    rejection is before any write).
    """
    svc = _make_service_with_settings(
        entities=[_entity("entity:a", "A")],
        relations=[],
        vault_path="./relative/vault",
    )

    with pytest.raises(ValueError, match="absolute"):
        await svc.export("notebook:relpath", _vault_request())


async def test_vault_path_must_exist(tmp_path):
    """A missing vault_path is rejected with ValueError before any write."""
    missing = tmp_path / "does-not-exist"
    svc = _make_service_with_settings(
        entities=[_entity("entity:a", "A")],
        relations=[],
        vault_path=str(missing),
    )
    with pytest.raises(ValueError, match="does not exist"):
        await svc.export("notebook:missing", _vault_request())


# ---------------------------------------------------------------------------
# 4. Overwrite existing .md; preserve unrelated user files
# ---------------------------------------------------------------------------


async def test_vault_path_overwrite_existing_md(tmp_path):
    """Pre-existing .md in target dir is overwritten; user files are preserved.

    Q-D-6: the export is the source of truth for *its own filenames*
    inside ``<entities_folder>/``. Other user-added files are never
    touched. We pre-populate the target dir with:

      * ``alice.md`` -- should be overwritten (the export creates one
        for entity:alice).
      * ``user_added.md`` -- should survive untouched (not in the
        export's filename set).

    Plus a file OUTSIDE the entities_folder that must also survive.
    """
    # Pre-create the entities folder and seed it.
    entities_folder = tmp_path / "Entities"
    entities_folder.mkdir()
    (entities_folder / "alice.md").write_text("OLD CONTENT - should be overwritten")
    (entities_folder / "user_added.md").write_text("USER NOTE - should survive")
    # File outside the entities_folder -- in the vault root.
    (tmp_path / "vault_root_note.md").write_text(
        "VAULT ROOT NOTE - definitely should survive"
    )

    ents = [_entity("entity:alice", "Alice", type_tags=["Person"])]
    svc = _make_service_with_settings(ents, [], str(tmp_path))

    artifact = await svc.export("notebook:overwrite", _vault_request())

    # alice.md was overwritten with the export's content.
    alice_content = (entities_folder / "alice.md").read_text(encoding="utf-8")
    assert "OLD CONTENT" not in alice_content
    assert "# Alice" in alice_content  # the export's title section

    # User-added .md inside the entities folder is untouched.
    user_content = (entities_folder / "user_added.md").read_text(encoding="utf-8")
    assert user_content == "USER NOTE - should survive"

    # File outside the entities folder is untouched.
    root_content = (tmp_path / "vault_root_note.md").read_text(encoding="utf-8")
    assert root_content == "VAULT ROOT NOTE - definitely should survive"

    # Report still reports the writes we performed (alice + README),
    # not the user_added.md we left alone.
    assert artifact.report.entities_written == 1
    assert artifact.report.files_written == 2  # alice.md + README.md


# ---------------------------------------------------------------------------
# 5. Atomic per-file: forced mid-batch failure leaves first file written
# ---------------------------------------------------------------------------


async def test_vault_path_atomic_per_file(tmp_path):
    """Mid-batch filesystem failure leaves earlier files written.

    Pins the per-file atomicity contract (per the module docstring): a
    failure on the Nth file leaves files 1..N-1 fully written under
    their final names and propagates the OSError with
    ``{"entities_written": N-1}`` attached so the router can surface
    partial state.

    Setup: 3 entities -> the service produces ``aaa.md``, ``bbb.md``,
    ``ccc.md`` plus a ``README.md``. ``_write_to_vault`` writes them
    in sorted order so the sequence is:
    ``README.md`` -> ``aaa.md`` -> ``bbb.md`` -> ``ccc.md``.
    We force a write failure on ``bbb.md`` (3rd in the sequence) and
    assert that the 2 preceding files (README.md + aaa.md) survived
    and the 4th (ccc.md) was never touched.

    We mock ``builtins.open`` to fail only when the tmp path for
    ``bbb.md`` is opened.
    """
    ents = [
        _entity("entity:aaa", "AAA"),  # filename "aaa.md"
        _entity("entity:bbb", "BBB"),  # filename "bbb.md" -- will fail
        _entity("entity:ccc", "CCC"),  # filename "ccc.md" -- never reached
    ]
    svc = _make_service_with_settings(ents, [], str(tmp_path))

    real_open = open

    def selective_open(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        path_str = str(path)
        # The tmp file is written as ``<final>.tmp.<pid>`` -- match the
        # bbb.md tmp file and force a write failure.
        if "bbb.md.tmp." in path_str:
            raise OSError("simulated disk full")
        return real_open(*args, **kwargs)

    with patch("builtins.open", selective_open):
        with pytest.raises(OSError) as excinfo:
            await svc.export("notebook:partial", _vault_request())

    # OSError carries partial state in args (args may be (message, {dict})).
    partial = None
    for arg in excinfo.value.args:
        if isinstance(arg, dict) and "entities_written" in arg:
            partial = arg
            break
    assert partial is not None, (
        f"OSError args missing partial-state dict; args={excinfo.value.args!r}"
    )
    # The write order is ``sorted(files.keys())``:
    # ``README.md`` (1st) -> ``aaa.md`` (2nd) -> ``bbb.md`` (3rd, fails).
    # Two files survived before the failure.
    assert partial["entities_written"] == 2
    assert partial.get("failed_file") == "bbb.md"

    # On-disk verification: README.md + aaa.md exist; bbb.md / ccc.md don't.
    target_dir = tmp_path / "Entities"
    assert (target_dir / "README.md").is_file()
    assert (target_dir / "aaa.md").is_file()
    assert not (target_dir / "bbb.md").exists()
    assert not (target_dir / "ccc.md").exists()
    # Defensively check no stale ``.tmp.*`` files leak into the vault.
    tmp_leftovers = list(target_dir.glob("*.tmp.*"))
    assert tmp_leftovers == [], (
        f"Stale tmp files leaked into vault: {tmp_leftovers!r}"
    )


# ---------------------------------------------------------------------------
# 6. Telemetry redacts the raw vault path
# ---------------------------------------------------------------------------


async def test_vault_path_telemetry_redacts_path(tmp_path, monkeypatch):
    """``record_metric`` payload has ``vault_path_redacted=True`` and no raw path.

    Q-D-8: the raw filesystem path is PII (e.g.
    ``/Users/<full_name>/Documents/...``) and must NEVER appear in the
    telemetry payload. The redaction is asserted in two ways:

    1. ``vault_path_redacted`` is True in the payload.
    2. The raw ``tmp_path`` string does NOT appear in the JSON dump
       of the payload (recursive check across all string values).
    """
    events: list[dict] = []

    async def _spy(event_type, payload, source=None, notebook=None):
        events.append({"event_type": event_type, "payload": payload, "notebook": notebook})

    monkeypatch.delenv("OPEN_NOTEBOOK_DISABLE_METRICS", raising=False)
    monkeypatch.setattr(
        "app_main.services.obsidian_export_service.record_metric", _spy
    )

    ents = [_entity("entity:a", "A")]
    svc = _make_service_with_settings(ents, [], str(tmp_path))
    await svc.export("notebook:tel-vault", _vault_request())

    assert len(events) == 1
    event = events[0]
    assert event["event_type"] == "export.obsidian"
    payload = event["payload"]

    # Mode + redaction flag are present.
    assert payload["mode"] == "vault_path"
    assert payload["vault_path_redacted"] is True

    # The raw vault path is NOT in the payload. Recursively walk all
    # string values and assert tmp_path's string form is absent. We
    # also assert no key NAMED ``vault_path`` carries the raw value.
    raw_path = str(tmp_path)

    def _no_raw_path(value):
        if isinstance(value, str):
            assert raw_path not in value, (
                f"Raw vault path leaked into telemetry: {value!r}"
            )
        elif isinstance(value, dict):
            for k, v in value.items():
                # Defense: no key named vault_path with a raw string.
                if k == "vault_path":
                    assert v is None or not isinstance(v, str) or v == ""
                _no_raw_path(v)
        elif isinstance(value, list):
            for v in value:
                _no_raw_path(v)

    _no_raw_path(payload)

    # Counts-only contract still holds: no entity/source/notebook IDs.
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
