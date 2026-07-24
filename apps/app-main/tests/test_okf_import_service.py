"""Tests for the OKF v0.1 bundle import service (Phase OKF.3).

Three layers, mirroring the service:

* **Pure parsing / round-trip** — the headline OKF.1↔OKF.3 conformance proof.
  ``build_okf_bundle`` (OKF.1) produces a bundle; ``parse_okf_bundle`` (OKF.3)
  reconstructs the entities / relations / notes / sources. Fully offline.
* **Wiring (mocked)** — ``import_bundle`` drives the canonical persist surfaces
  (entity upsert, the injection-safe RELATE primitive, note/source repos) and the
  K.5 propose + K.3 apply-auto-merge dedup, with provenance tags. AsyncMock only.
* **DB (``@requires_docker``)** — export→import round-trip over a real SurrealDB
  container, idempotent re-import, and the full K.5/K.3 fuzzy-dedup collapse.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from app_main.services.okf_export_service import build_okf_bundle
from app_main.services.okf_import_service import (
    OkfImportService,
    parse_okf_bundle,
)
from shared.models.entity import Entity, Relation
from shared.models.notebook import Note
from shared.models.source import Asset, Source

# ---------------------------------------------------------------------------
# Fixture projection (mirrors the OKF.1 export-test fixtures)
# ---------------------------------------------------------------------------


def _entity(
    eid: str,
    name: str,
    *,
    primary_type: str = "Person",
    description: Optional[str] = None,
    external_ids: Optional[List[str]] = None,
) -> Entity:
    return Entity(
        id=eid,
        canonical_name=name,
        entity_type=primary_type,
        primary_type=primary_type,
        type_tags=[primary_type],
        description=description,
        external_ids=external_ids or [],
        confidence=0.95,
    )


def _relation(src: str, dst: str, rel_type: str = "KNOWS") -> Relation:
    return Relation(**{"in": src, "out": dst, "relation_type": rel_type, "confidence": 0.9})


def _projection():
    entities = [
        _entity("entity:alice", "Alice", description="A researcher."),
        _entity("entity:bob", "Bob"),
        _entity("entity:carol", "Carol", external_ids=["https://doi.org/10.1/x"]),
    ]
    relations = [
        _relation("entity:alice", "entity:bob", "KNOWS"),
        _relation("entity:bob", "entity:carol", "MENTORS"),
    ]
    notes = [Note(id="note:n1", title="Meeting notes", content="Body text here", note_type="human")]
    sources = [Source(id="source:s1", title="Paper A", asset=Asset(url="https://example.com/a"))]
    return entities, relations, notes, sources


def _bundle(**overrides) -> Dict[str, str]:
    entities, relations, notes, sources = _projection()
    kwargs = dict(
        entities=entities,
        relations=relations,
        notes=notes,
        sources=sources,
        dropped_relations=0,
        notebook_id="notebook:x",
        timestamp="2026-07-24T00:00:00+00:00",
    )
    kwargs.update(overrides)
    return build_okf_bundle(**kwargs).bundle


# ---------------------------------------------------------------------------
# Pure parsing / round-trip
# ---------------------------------------------------------------------------


def test_round_trip_reconstructs_entities_relations_notes_sources():
    parsed = parse_okf_bundle(_bundle())

    assert sorted(e.canonical_name for e in parsed.entities.values()) == [
        "Alice", "Bob", "Carol",
    ]
    alice = next(e for e in parsed.entities.values() if e.canonical_name == "Alice")
    assert alice.okf_type == "Person"
    assert alice.description == "A researcher."

    # Two undirected links per relation collapse back to two edges (direction is
    # lossy — endpoints are recorded in sorted order).
    rels = {(r.from_id, r.to_id, r.relation_type) for r in parsed.relations}
    assert rels == {
        ("entities/alice", "entities/bob", "KNOWS"),
        ("entities/bob", "entities/carol", "MENTORS"),
    }

    assert [(n.title, n.content) for n in parsed.notes] == [
        ("Meeting notes", "Body text here"),
    ]
    src = next(iter(parsed.sources.values()))
    assert src.title == "Paper A"
    assert src.resource == "https://example.com/a"

    assert parsed.skipped == []
    assert parsed.dangling_links == []


def test_real_external_uri_vs_internal_urn():
    parsed = parse_okf_bundle(_bundle())
    carol = next(e for e in parsed.entities.values() if e.canonical_name == "Carol")
    bob = next(e for e in parsed.entities.values() if e.canonical_name == "Bob")
    # Carol carried a real DOI; Bob got a synthesised internal urn.
    assert carol.resource == "https://doi.org/10.1/x"
    assert bob.resource.startswith("urn:open-notebook:")


def test_note_content_strips_leading_title_heading():
    parsed = parse_okf_bundle(_bundle())
    assert parsed.notes[0].content == "Body text here"
    assert not parsed.notes[0].content.startswith("#")


def test_malformed_frontmatter_concept_skipped_others_survive():
    bundle = _bundle()
    bundle["entities/broken.md"] = "no frontmatter here, just text\n"
    bundle["entities/notype.md"] = "---\ntitle: X\n---\n\nbody\n"
    parsed = parse_okf_bundle(bundle)

    reasons = {s.path: s.reason for s in parsed.skipped}
    assert "entities/broken.md" in reasons
    assert "entities/notype.md" in reasons
    # The well-formed concepts are unaffected.
    assert len(parsed.entities) == 3


def test_dangling_link_recorded_not_crash():
    bundle = _bundle()
    # Alice links a concept that isn't in the bundle.
    bundle["entities/alice.md"] += "\n## More\n\n- [Ghost](/entities/ghost.md) — HAUNTS\n"
    parsed = parse_okf_bundle(bundle)
    assert ("entities/alice", "entities/ghost") in parsed.dangling_links


def test_reserved_and_index_type_concepts_not_imported():
    bundle = _bundle()
    # index.md is present (reserved) and has type: Index — never a concept.
    assert "index.md" in bundle
    bundle["entities/log.md"] = "---\ntype: Person\ntitle: Log\n---\n\nbody\n"
    bundle["notes/log.md"] = "---\ntype: Log\ntitle: whatever\n---\n\nbody\n"
    parsed = parse_okf_bundle(bundle)
    # Reserved filename skipped regardless of frontmatter; Log-typed skipped.
    assert "entities/log" not in parsed.entities
    assert all("log" not in cid for cid in parsed.entities)
    assert all(s.path != "index.md" for s in parsed.skipped)


def test_relative_links_resolve_to_relations():
    bundle = {
        "team/a.md": "---\ntype: Person\ntitle: A\n---\n\n## Relations\n\n- [B](./b.md) — PAIRS\n",
        "team/b.md": "---\ntype: Person\ntitle: B\n---\n\n# B\n",
    }
    parsed = parse_okf_bundle(bundle)
    assert {(r.from_id, r.to_id, r.relation_type) for r in parsed.relations} == {
        ("team/a", "team/b", "PAIRS"),
    }


def test_parse_from_zip_bytes():
    import io
    import zipfile

    bundle = _bundle()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for path, text in bundle.items():
            z.writestr(path, text)
    parsed = parse_okf_bundle(buf.getvalue())
    assert len(parsed.entities) == 3
    assert len(parsed.notes) == 1


# ---------------------------------------------------------------------------
# Wiring (mocked)
# ---------------------------------------------------------------------------


def _mock_service(monkeypatch, *, existing_names=None, dedup_auto=None):
    """Build an OkfImportService with fully mocked collaborators.

    ``existing_names`` — canonical names ``_entity_exists`` should report as
    already present (matched, not created). ``dedup_auto`` — a list of fake
    auto-merge candidates the K.5 proposer returns.
    """
    existing = set(existing_names or [])

    async def _fake_exec(query, params=None, config=None):
        q = query.lower()
        if "from entity" in q:
            return ["entity:existing"] if params.get("name") in existing else []
        # note/source okf_key existence checks → nothing exists yet
        return []

    monkeypatch.setattr(
        "app_main.services.okf_import_service.execute_query", _fake_exec
    )

    entity_repo = AsyncMock()
    entity_repo.upsert_entity = AsyncMock(
        side_effect=lambda e: f"entity:{e.canonical_name.lower()}"
    )
    persistence = MagicMock()
    persistence._upsert_relation = AsyncMock(return_value=True)

    note_repo = AsyncMock()
    note_repo.create = AsyncMock(return_value=MagicMock(id="note:new"))
    note_repo.add_to_notebook = AsyncMock(return_value=True)

    source_repo = AsyncMock()
    source_repo.create = AsyncMock(side_effect=lambda d: MagicMock(id="source:new"))
    source_repo.add_to_notebook = AsyncMock(return_value=True)

    dedup = AsyncMock()
    report = MagicMock()
    report.auto_merge = dedup_auto or []
    dedup.propose_candidates = AsyncMock(return_value=report)
    recanon = AsyncMock()
    recanon.apply_merge = AsyncMock(return_value=MagicMock(skipped=False))

    svc = OkfImportService(
        entity_repository=entity_repo,
        persistence_service=persistence,
        note_repository=note_repo,
        source_repository=source_repo,
        candidate_dedup_service=dedup,
        recanonicalization_service=recanon,
    )
    return svc, entity_repo, persistence, note_repo, source_repo, dedup, recanon


async def test_import_upserts_entities_relations_notes_sources(monkeypatch):
    svc, entity_repo, persistence, note_repo, source_repo, _, _ = _mock_service(
        monkeypatch
    )
    report = await svc.import_bundle(_bundle())

    assert report.entities_created == 3
    assert report.entities_matched == 0
    assert report.relations_created == 2
    assert report.notes_created == 1
    assert report.sources_created == 1
    assert entity_repo.upsert_entity.await_count == 3
    assert persistence._upsert_relation.await_count == 2

    # Provenance: every entity is tagged okf-import + carries the anchor source.
    for call in entity_repo.upsert_entity.await_args_list:
        entity: Entity = call.args[0]
        assert entity.extraction_method == "okf-import"
        assert entity.properties.get("okf_import") is True
        assert entity.source_documents == [report.import_source_id]

    # The real DOI is promoted to external_ids; the internal urn is not.
    carol = next(
        c.args[0]
        for c in entity_repo.upsert_entity.await_args_list
        if c.args[0].canonical_name == "Carol"
    )
    assert carol.external_ids == ["https://doi.org/10.1/x"]
    bob = next(
        c.args[0]
        for c in entity_repo.upsert_entity.await_args_list
        if c.args[0].canonical_name == "Bob"
    )
    assert bob.external_ids == []
    assert bob.properties["okf_resource"].startswith("urn:open-notebook:")


async def test_import_matches_existing_entity_not_duplicated(monkeypatch):
    svc, entity_repo, *_ = _mock_service(monkeypatch, existing_names={"Alice"})
    report = await svc.import_bundle(_bundle())
    # Alice pre-exists → matched; Bob + Carol are new.
    assert report.entities_matched == 1
    assert report.entities_created == 2
    # Upsert is still called for all three (it is the dedup path itself).
    assert entity_repo.upsert_entity.await_count == 3


async def test_import_runs_k5_propose_and_applies_auto_merge(monkeypatch):
    candidate = MagicMock()
    candidate.to_merge_cluster = MagicMock(return_value="cluster")
    candidate.id_a, candidate.id_b = "entity:a", "entity:b"
    svc, _, _, _, _, dedup, recanon = _mock_service(
        monkeypatch, dedup_auto=[candidate]
    )
    report = await svc.import_bundle(_bundle(), notebook_id="notebook:z")

    dedup.propose_candidates.assert_awaited_once_with(notebook_id="notebook:z")
    recanon.apply_merge.assert_awaited_once_with("cluster")
    assert report.dedup_merges == 1


async def test_import_can_disable_dedup(monkeypatch):
    svc, _, _, _, _, dedup, recanon = _mock_service(monkeypatch)
    await svc.import_bundle(_bundle(), apply_dedup=False)
    dedup.propose_candidates.assert_not_called()
    recanon.apply_merge.assert_not_called()


async def test_malformed_concept_never_persisted(monkeypatch):
    svc, entity_repo, *_ = _mock_service(monkeypatch)
    bundle = _bundle()
    bundle["entities/broken.md"] = "not a concept\n"
    report = await svc.import_bundle(bundle)
    # Only the 3 well-formed entities upsert; the broken one is skip-recorded.
    assert entity_repo.upsert_entity.await_count == 3
    assert any(s.path == "entities/broken.md" for s in report.skipped)


async def test_import_report_serialises(monkeypatch):
    svc, *_ = _mock_service(monkeypatch)
    report = await svc.import_bundle(_bundle())
    d = report.to_dict()
    assert d["entities_created"] == 3
    assert d["relations_created"] == 2
    assert isinstance(d["skipped"], list)


# ---------------------------------------------------------------------------
# DB (requires docker)
# ---------------------------------------------------------------------------

from surrealdb_service.config import SurrealDBConfig  # noqa: E402
from surrealdb_service.connection import execute_query  # noqa: E402


def _bind_db_service(monkeypatch, config: SurrealDBConfig):
    """Wire a fully DB-backed OkfImportService at the live container.

    The service and the persistence service both reach SurrealDB through their
    module-level ``execute_query`` (no config arg), so patch those symbols with a
    wrapper injecting the container config; every repository is constructed
    ``config``-bound. The K.5/K.3 stack accesses the DB only via the config-bound
    ``EntityRepository`` + overlay repo (its ``execute_query`` paths fire only for
    notebook-scoped runs, which these tests never take), so global dedup works.
    """
    from app_main.services import (
        entity_persistence_service as eps_module,
    )
    from app_main.services import (
        okf_import_service as okf_module,
    )
    from app_main.services.entity_persistence_service import (
        EntityPersistenceService,
    )
    from app_main.services.entity_resolution import (
        CandidateDedupService,
        RecanonicalizationService,
    )
    from app_main.services.entity_resolution.overlay_service import OverlayService
    from surrealdb_service.repositories import NoteRepository, SourceRepository
    from surrealdb_service.repositories.alias_overlay import AliasOverlayRepository
    from surrealdb_service.repositories.entity import EntityRepository

    async def _exec(query, params=None, config_arg=None):
        return await execute_query(query, params, config=config or config_arg)

    monkeypatch.setattr(okf_module, "execute_query", _exec)
    monkeypatch.setattr(eps_module, "execute_query", _exec)

    entity_repo = EntityRepository(config=config)
    overlay = OverlayService(repo=AliasOverlayRepository(config=config))
    return OkfImportService(
        entity_repository=entity_repo,
        persistence_service=EntityPersistenceService(entity_repository=entity_repo),
        note_repository=NoteRepository(config=config),
        source_repository=SourceRepository(config=config),
        candidate_dedup_service=CandidateDedupService(
            entity_repo=entity_repo, overlay_service=overlay
        ),
        recanonicalization_service=RecanonicalizationService(entity_repo=entity_repo),
    )


@pytest.mark.requires_docker
async def test_db_round_trip_export_import(monkeypatch, live_surrealdb):
    config = live_surrealdb
    svc = _bind_db_service(monkeypatch, config)

    uid = uuid.uuid4().hex[:8]
    entities = [
        _entity(f"entity:a{uid}", f"Ada {uid}", primary_type="Person", description="Pioneer."),
        _entity(f"entity:b{uid}", f"Babbage {uid}", primary_type="Person"),
    ]
    relations = [_relation(f"entity:a{uid}", f"entity:b{uid}", "COLLABORATES")]
    notes = [Note(id="note:n", title=f"Note {uid}", content=f"content {uid}", note_type="human")]
    sources = [Source(id="source:s", title=f"Src {uid}", asset=Asset(url=f"https://x/{uid}"))]
    bundle = build_okf_bundle(
        entities=entities, relations=relations, notes=notes, sources=sources,
        dropped_relations=0, notebook_id="notebook:rt",
        timestamp="2026-07-24T00:00:00+00:00",
    ).bundle

    report = await svc.import_bundle(bundle, apply_dedup=False)
    assert report.entities_created == 2
    assert report.relations_created == 1
    assert report.notes_created == 1

    ent_rows = await execute_query(
        "SELECT canonical_name, extraction_method, properties FROM entity "
        "WHERE canonical_name IN [$a, $b];",
        {"a": f"Ada {uid}", "b": f"Babbage {uid}"},
        config=config,
    )
    assert len(ent_rows) == 2
    for row in ent_rows:
        assert row["extraction_method"] == "okf-import"
        assert row["properties"].get("okf_import") is True

    rel_rows = await execute_query(
        "SELECT relation_type FROM relation WHERE relation_type = 'COLLABORATES';",
        config=config,
    )
    assert any(r["relation_type"] == "COLLABORATES" for r in rel_rows)

    note_rows = await execute_query(
        "SELECT title FROM note WHERE title = $t;", {"t": f"Note {uid}"}, config=config
    )
    assert len(note_rows) == 1


@pytest.mark.requires_docker
async def test_db_reimport_is_idempotent(monkeypatch, live_surrealdb):
    config = live_surrealdb
    svc = _bind_db_service(monkeypatch, config)

    uid = uuid.uuid4().hex[:8]
    entities = [_entity(f"entity:x{uid}", f"Grace {uid}", primary_type="Person")]
    notes = [Note(id="note:n", title=f"N {uid}", content=f"c {uid}", note_type="human")]
    bundle = build_okf_bundle(
        entities=entities, relations=[], notes=notes, sources=[],
        dropped_relations=0, notebook_id="notebook:idem",
        timestamp="2026-07-24T00:00:00+00:00",
    ).bundle

    first = await svc.import_bundle(bundle, apply_dedup=False)
    second = await svc.import_bundle(bundle, apply_dedup=False)

    assert first.entities_created == 1
    assert second.entities_created == 0
    assert second.entities_matched == 1
    assert second.notes_created == 0
    assert second.notes_matched == 1

    # Exactly one entity + one note for these names — no duplication.
    ent = await execute_query(
        "SELECT count() AS c FROM entity WHERE canonical_name = $n GROUP ALL;",
        {"n": f"Grace {uid}"}, config=config,
    )
    assert ent[0]["c"] == 1
    note = await execute_query(
        "SELECT count() AS c FROM note WHERE title = $t GROUP ALL;",
        {"t": f"N {uid}"}, config=config,
    )
    assert note[0]["c"] == 1


@pytest.mark.requires_docker
async def test_db_fuzzy_dedup_collapses_near_duplicate(monkeypatch, live_surrealdb):
    config = live_surrealdb
    svc = _bind_db_service(monkeypatch, config)

    uid = uuid.uuid4().hex[:8]
    seeded_name = f"Koninkrijksrelaties{uid}"
    typo_name = f"Koninkrijksreiaties{uid}"  # one internal substitution (l->i)

    # Seed the "already present" entity natively (not via OKF).
    await execute_query(
        "CREATE entity SET canonical_name=$n, name=$n, entity_type='organization', "
        "embedding=[], confidence=0.9, source_documents=[], status='active';",
        {"n": seeded_name}, config=config,
    )

    bundle = build_okf_bundle(
        entities=[_entity(f"entity:t{uid}", typo_name, primary_type="Organization")],
        relations=[], notes=[], sources=[],
        dropped_relations=0, notebook_id="notebook:dedup",
        timestamp="2026-07-24T00:00:00+00:00",
    ).bundle

    report = await svc.import_bundle(bundle, apply_dedup=True)
    assert report.dedup_merges >= 1

    # After K.5 propose + K.3 apply, exactly ONE of the pair is still active.
    rows = await execute_query(
        "SELECT canonical_name, status FROM entity "
        "WHERE canonical_name IN [$a, $b];",
        {"a": seeded_name, "b": typo_name}, config=config,
    )
    active = [r for r in rows if r["status"] == "active"]
    merged = [r for r in rows if r["status"] == "merged"]
    assert len(active) == 1
    assert len(merged) == 1
